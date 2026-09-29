// Точка входа Mineflayer-стороны: поднимает РОЙ ботов (bot.count в
// config.json, имена по паттерну <username><separator><index>, например
// AI_1, AI_2) и гоняет общий тиковый цикл:
//   для каждого бота: собрать состояние -> отправить в Python ->
//   применить действие, пришедшее ДЛЯ ЭТОГО бота (роутинг по bot_id).
//
// Python-сторона (py/ai_loop.py) держит по боту отдельную сессию (задача,
// стек кадров, прошлый переход), но ОДНИ общие сети и replay buffer — все
// боты льют опыт в одно обучение. За тик от Python приходит по действию
// на каждый канал (ноги/голова/руки) — они выполняются одновременно.
//
// Для долгого обучения без присмотра: отвалившийся бот (кик, рестарт
// сервера) сам переподключается через reconnect_delay_ms.
//
// Первой должна быть запущена Python-сторона — она делает bind на
// ZMQ-порты; Node только connect-ится и переподключается сам.
//
// Именование: при count=1 бот зовётся ровно bot.username (без суффикса);
// при count>1 к имени добавляется разделитель и номер с 1.
// Разделитель — "_": ванильный Minecraft допускает в именах только
// [a-zA-Z0-9_]. С "#" (так было до 2026-09-26) бот заходил на сервер в
// offline-режиме, но его ник не принимали команды сервера (/op AI#1,
// /team join ...).

const mineflayer = require('mineflayer');
const { loadConfig } = require('./config');
const { ZmqBridge } = require('./zmq_bridge');
const { ActionExecutor } = require('./actions');
const { TargetManager } = require('./target');
const { Hearing } = require('./hearing');
const { StateBuilder } = require('./state');
const { WorldBorder } = require('./border');
const { RoutePlanner } = require('./route');
const { summarizeInventory } = require('./inventory');
const { classifyEntity, HOSTILE, PASSIVE, PLAYER, NON_LIVING } = require('./entities');

const config = loadConfig();

function botName(index) {
    if (config.bot.count <= 1) return config.bot.username;
    return `${config.bot.username}${config.bot.separator}${index}`;
}

// Пауза перед переподключением отвалившегося бота. Не ноль: сразу после
// кика сервер часто ещё держит старую сессию и отвергает вход с тем же ником.
const RECONNECT_DELAY_MS = 5000;

// Сколько ждать от Python действий на состояния ЭТОГО тика. Обычно ответ
// приходит за пару миллисекунд; если не пришёл — бот выполняет последнее
// известное действие (как раньше). Раньше ждали 1 мс, и почти всегда
// выполнялось действие, выбранное по прошлому кадру: сеть видела
// результат своего решения только через тик, и учиться было заметно
// труднее (особенно точному наведению в looking).
// 100 мс (было 60): с 20+ ботами Python отвечает всем за 40-60 мс, а тик
// 150 мс — время есть (сбор состояний ~1.3 мс на бота + ожидание + применение).
const ACTION_WAIT_MS = 100;

// Сообщения сервера о смерти (mineflayer отдаёт их по-английски): по ним в
// лог пишется, ОТЧЕГО умер бот — упал, утонул, убит мобом...
const DEATH_MESSAGE = /was slain|was shot|was killed|drowned|fell|hit the ground|burn|flames|blew up|blown up|suffocat|starved|lava|squashed|withered|froze|pricked|impaled|fireballed|died|kinetic|stung|poked|obliterated|doomed|left the confines/;

// Пересоздать бота через паузу. error и end часто приходят оба —
// флаг не даёт запустить два переподключения одного и того же бота.
// Тиковый цикл при этом не останавливается: он просто пропускает
// незаспавненных ботов, а остальные продолжают учиться.
function scheduleReconnect(ctx) {
    if (shuttingDown || ctx.reconnectScheduled) return;
    ctx.reconnectScheduled = true;
    console.log(`[bot#${ctx.id}] Переподключение через ${RECONNECT_DELAY_MS / 1000} с...`);
    setTimeout(() => {
        if (shuttingDown) return;
        // Старые обработчики не снимаем: без слушателя 'error' Node падает,
        // а повторный scheduleReconnect для старого ctx глушит флаг.
        try {
            ctx.bot.end();
        } catch {
            /* старое соединение и так мертво */
        }
        contexts[ctx.id - 1] = createBotContext(ctx.id);
    }, RECONNECT_DELAY_MS);
}

// Один контекст на бота: всё, что у одиночного бота было глобальным.
function createBotContext(index) {
    const bot = mineflayer.createBot({
        host: config.bot.host,
        port: config.bot.port,
        username: botName(index),
        version: config.bot.version || false, // false = автоопределение
        // Возрождает бота наш обработчик 'death' — через секунду. Сам
        // mineflayer делал это мгновенно, смерть проскакивала между тиками
        // (150 мс), и Python её почти не видел: ни штрафа за смерть, ни
        // нового свечения в салках после /kill (заметил автор, 2026-09-26).
        respawn: false,
    });

    const ctx = {
        id: index,       // bot_id в сообщениях Python <-> Node
        bot,
        executor: new ActionExecutor(bot, config),
        targetManager: new TargetManager(bot),
        stateBuilder: new StateBuilder(config, { packedVision: true }),
        hearing: null,
        border: new WorldBorder(bot), // граница мира — в состояние (walking не ставит цели за ней)
        spawned: false,
        reconnectScheduled: false,
    };

    bot.once('spawn', async () => {
        console.log(`[bot#${index}] ${bot.username} заспавнился @ ${bot.entity.position}`);
        ctx.hearing = new Hearing(bot, config);
        // Маршрут к цели (js/route.js). Пересчёты у ботов роя разнесены по
        // тикам — иначе все семеро считали бы A* в один и тот же тик.
        ctx.route = new RoutePlanner(bot.world, bot.registry, config.route);
        ctx.route.ticksSincePlan = index % (config.route.replan_ticks ?? 5);
        ctx.spawned = true;
        await bridge.start(); // идемпотентен: реально поднимается один раз
        registerChatCommands(ctx);
    });

    bot.on('death', () => {
        console.log(`[bot#${index}] Умер. Награда за смерть уходит в Python через флаг dead.`);
        // Без респавна бот остаётся на экране смерти: dead=1 шлётся бесконечно
        // и обучение не продолжается. Пауза — чтобы терминальный переход
        // точно дошёл до Python.
        setTimeout(() => {
            bot.respawn();
            console.log(`[bot#${index}] Респавн.`);
        }, 1000);
    });

    bot.on('messagestr', (message) => {
        // Все боты роя получают одно и то же сообщение — пишет только тот,
        // о ком оно (ник целым словом: AI_1 не должен ловить "AI_10").
        const aboutMe = message.split(/\s+/).includes(bot.username);
        if (aboutMe && DEATH_MESSAGE.test(message)) {
            console.log(`[bot#${index}] Причина смерти: ${message}`);
        }
        // Умер человек (не бот роя) — Python-у: в задачке hunt ("Останови
        // меня") это значит, что цель остановили. Сообщение о смерти
        // начинается с ника погибшего; слышат его все боты — шлёт один.
        const victim = message.split(/\s+/)[0];
        if (DEATH_MESSAGE.test(message) && victim && !victim.startsWith(ctx.executor.swarmPrefix)
            && ctx === contexts.find((c) => c.spawned)) {
            bridge.sendCommand({ cmd: 'player_died', name: victim, text: message, bot_id: ctx.id, via: 'chat' })
                .catch((err) => console.error('[bot] Не удалось отправить смерть игрока:', err.message));
        }
    });

    bot.on('kicked', (reason) => console.log(`[bot#${index}] Кикнут:`, reason));
    bot.on('error', (err) => {
        console.log(`[bot#${index}] Ошибка:`, err.message || err.code || err);
        // Сервер недоступен (ECONNREFUSED и т.п.): mineflayer шлёт только
        // error, без end — переподключаться надо и отсюда.
        if (!ctx.spawned) scheduleReconnect(ctx);
    });
    bot.on('end', () => {
        console.log(`[bot#${index}] Соединение закрыто.`);
        ctx.spawned = false;
        scheduleReconnect(ctx);
    });

    return ctx;
}

const contexts = [];
for (let i = 1; i <= Math.max(1, config.bot.count); i++) {
    contexts.push(createBotContext(i));
}

const bridge = new ZmqBridge(config);
let tickIndex = 0;
let loopTimer = null;
let shuttingDown = false;
// Пауза общая на весь рой: !stop останавливает тиковый цикл — Python
// перестаёт получать состояния (и не копит переходы "стоит = плохо"),
// действия из буфера ZMQ не применяются. !resume продолжает.
let paused = false;

function startLoop() {
    if (!loopTimer) loopTimer = setInterval(tick, config.train.tick_rate_ms);
}

function pauseLoop() {
    if (loopTimer) {
        clearInterval(loopTimer);
        loopTimer = null;
    }
}

// true, пока идёт тик. setInterval не ждёт async-функцию: если тик роя
// дольше tick_rate_ms, следующий начался бы поверх, и два receive() на
// одном ZMQ-сокете роняют zeromq.js ("socket is busy"). Лишний тик просто
// пропускаем.
let tickRunning = false;

// Статистика ответов Python: доля тиков, когда свежие действия не успели
// прийти за ACTION_WAIT_MS (если таких много — ИИ не успевает, это видно в логе).
const answerStats = { ticks: 0, late: 0, waitedMs: 0 };

async function tick() {
    if (tickRunning) return;
    tickRunning = true;
    try {
        const sentTick = tickIndex;
        const waiting = new Set(); // боты, для которых ждём свежие действия
        // Сначала собираем состояния всех ботов, потом шлём пачкой: Python
        // отвечает всем ботам разом (один проход сети на задачку), и пачка
        // должна прийти к нему целиком, а не вперемешку со сборкой картинок.
        const outgoing = [];
        for (const ctx of contexts) {
            if (!ctx.spawned) continue;
            ctx.executor.tick(); // отпустить протухшие клавиши движения

            const bot = ctx.bot;
            const inventoryCount = bot.inventory
                .items()
                .reduce((sum, item) => sum + item.count, 0);

            const state = ctx.stateBuilder.build(
                bot.world, bot.entity, ctx.hearing, tickIndex,
                ctx.targetManager.getTargetPosition(), ctx.targetManager.describe(),
                bot.health <= 0, ctx.targetManager.isHumanControlled(), inventoryCount, bot.food,
                bot.entities, ctx.border
            );
            state.bot_id = ctx.id; // ключ сессии на стороне Python
            // id своей сущности на сервере — у всех ботов роя он одинаковый
            // для одной и той же сущности: салки по нему говорят догоняющим,
            // за кем бежать.
            state.self.entity_id = bot.entity.id;
            // Что в инвентаре (блоки, еда, броня, что в руке) — сеть должна
            // знать, есть ли ей чем строить и что есть.
            state.inventory = summarizeInventory(bot);
            // Люди рядом — для салок: водящим можно гоняться и за человеком.
            state.humans = ctx.executor.visibleHumans();
            // По кому пришёлся удар attack_center с прошлого тика (id сущности
            // или null) — судья салок засчитывает по нему "осалил".
            state.attacked_id = ctx.executor.takeAttacked();
            // Кто ударил самого бота с прошлого тика (id сущности или null) —
            // так судья видит, что убегающего осалил человек-водящий.
            state.hurt_by = ctx.executor.takeHurtBy();
            // Достанет ли attack_center кого-нибудь прямо сейчас (1/0) — та же
            // проверка, что у самого удара. Без неё сеть водящего выводила
            // "можно бить" из угла и расстояния до цели (маленьких чисел), а
            // соседних убегающих, не цель, так и вовсе не видела "в зоне удара".
            state.strike = ctx.executor.findEntityInCrosshair() ? 1 : 0;
            // Поставил бы place_front блок прямо сейчас (1/0) — та же проверка,
            // что у самого макроса: у края моста окно узкое, и без этого сеть
            // не отличала "в окне" от "замерла чуть раньше".
            state.can_place = ctx.executor.canPlaceFront() ? 1 : 0;
            // Заряд удара (0..1) и по кому мои удары с прошлого тика нанесли
            // урон и с какой силой — учиться бить "в полную силу", а не
            // махать каждый тик вполсилы (идея автора).
            state.attack_charge = Math.round(ctx.executor.attackCharge() * 1000) / 1000;
            state.damage_dealt = ctx.executor.takeDamageDealt();
            // Высота цели-сущности (0 для точки): looking целится в "лицо",
            // а не в ноги. В сеть не идёт — только модулям.
            if (state.target) state.target.h = ctx.targetManager.getTargetHeight();
            // Маршрут к цели в обход препятствий: ближайшая точка маршрута и
            // сколько идти по нему. Сеть ходьбы идёт к этой точке, а не к
            // цели напрямую (py/training_modules/walking.py).
            state.route = (config.route.enabled && ctx.route) ? ctx.route.update(bot.entity.position, state.target) : null;
            // Инвентарь по предметам — для задачи добычи ("добыть 5 брёвен"):
            // одного общего inventory_count мало, надо знать, ЧЕГО прибавилось.
            state.inventory_items = {};
            for (const item of bot.inventory.items()) {
                state.inventory_items[item.name] = (state.inventory_items[item.name] ?? 0) + item.count;
            }
            outgoing.push(state);
            if (!state.dead) waiting.add(ctx.id); // на мёртвое состояние Python не отвечает действием
        }
        for (const state of outgoing) await bridge.sendState(state);
        tickIndex++;

        // Ждём действия на состояния этого тика (Python помечает ответ
        // номером тика), но не дольше ACTION_WAIT_MS. Запоздавший ответ на
        // прошлый тик берём, только если свежего так и не дождались.
        // Действие/цель адресуются конкретному боту через bot_id.
        const actions = new Map();
        const staleActions = new Map();
        const started = Date.now();
        for (;;) {
            const msg = await bridge.receiveAction(); // ждёт до 1 мс
            if (msg === null) {
                if (waiting.size === 0 || Date.now() - started >= ACTION_WAIT_MS) break;
                continue;
            }
            const id = msg.bot_id ?? 1;
            const ctx = contexts.find((c) => c.id === id);
            if (msg.type === 'action' && msg.actions) {
                if (msg.tick === sentTick) {
                    actions.set(id, msg);
                    waiting.delete(id);
                } else {
                    staleActions.set(id, msg);
                }
            } else if (msg.type === 'set_target') {
                if (ctx) ctx.targetManager.setModuleTarget(msg.position, msg.entity_id);
            } else if (msg.type === 'chat') {
                // Ответ Python на команду (например, !task) — туда же,
                // откуда пришла команда: в чат игры или в лог лаунчера.
                if (msg.via === 'console') console.log(`[cmd] [bot#${msg.bot_id}] ${msg.text}`);
                else if (ctx && ctx.spawned) ctx.bot.chat(msg.text);
            }
        }
        for (const [id, message] of staleActions) {
            if (!actions.has(id)) actions.set(id, message);
        }
        noteAnswerTime(Date.now() - started, waiting.size > 0);

        for (const [id, message] of actions) {
            const ctx = contexts.find((c) => c.id === id);
            if (!ctx || !ctx.spawned) continue;

            // Салки: кого из игроков можно бить (водящему — убегающих, удар =
            // осалил), остальным — никого.
            ctx.executor.setTaggable(message.tag_ids);
            ctx.executor.executeAll(message.actions);
        }
    } catch (err) {
        console.error('[bot] Ошибка тика:', err.message);
    } finally {
        tickRunning = false;
    }
}

function noteAnswerTime(waitedMs, late) {
    answerStats.ticks++;
    answerStats.waitedMs += waitedMs;
    if (late) answerStats.late++;
    if (answerStats.ticks < 1000) return;
    const lateShare = answerStats.late / answerStats.ticks;
    if (lateShare > 0.1) {
        console.log(`[bot] ИИ не успевает отвечать: в ${Math.round(lateShare * 100)}% тиков действия опоздали `
            + `(среднее ожидание ${Math.round(answerStats.waitedMs / answerStats.ticks)} мс). Python перегружен?`);
    }
    answerStats.ticks = 0;
    answerStats.late = 0;
    answerStats.waitedMs = 0;
}

// Команды приходят из двух мест: из чата игры и из строки команд лаунчера
// (py/launcher.py шлёт их в stdin, см. внизу файла). Логика одна —
// runCommand; различается только, куда отвечать: в чат или в консоль
// (консоль Node = лог лаунчера, чат игры не засоряется).
function registerChatCommands(ctx) {
    const { bot } = ctx;
    // Обычный чат слышат ВСЕ боты роя — команда оттуда действует на
    // каждого (так !setTarget player <ник> ставит цель сразу всем).
    // Чтобы обратиться к одному боту — личное сообщение: /msg AI_3 !task looking.
    const onMessage = (username, message, whisper) => {
        if (username === bot.username) return;
        if (!message.startsWith('!')) return;
        const args = message.slice(1).trim().split(/\s+/);
        runCommand(ctx, args, (text) => bot.chat(text), { direct: whisper, via: 'chat', speaker: username });
    };
    bot.on('chat', (username, message) => onMessage(username, message, false));
    bot.on('whisper', (username, message) => onMessage(username, message, true));
}

const HELP_TEXT = 'Команды: !setTarget player <имя> | position <x y z> | none, !task <задача> | all <задача>, '
    + '!greedy [номер] (показать выученное), !start [ник] ("Останови меня": охота, задача hunt), '
    + '!stop (пауза ИИ), !resume, !debug, !entities, !whatYouSee';

// args — команда без "!", разбитая по пробелам. reply(text) — куда отвечать.
// direct — команда адресована именно этому боту (личка или выбор бота в
// лаунчере), via — 'chat' или 'console': Python отвечает на task/greedy
// тем же путём.
function runCommand(ctx, args, reply, { direct = false, via = 'chat', speaker = null } = {}) {
    const { bot } = ctx;
    const command = args[0];
    const toPython = (payload) => bridge.sendCommand({ ...payload, bot_id: ctx.id, via })
        .catch((err) => console.error('[bot] Не удалось отправить команду:', err.message));

    switch (command) {
        case 'help':
            reply(HELP_TEXT);
            break;
        case 'task': {
            // Задачу знает только Python (обучающие модули) — просто
            // пересылаем, он сам проверит имя и ответит.
            // !task walking — этому боту, !task all walking — всему рою.
            const all = args[1] === 'all';
            const task = all ? args[2] : args[1];
            if (!task) return reply('Использование: !task <walking|looking|follow|gathering|crafting|mix|tag|hunt> или !task all <задача>');
            toPython({ cmd: 'set_task', task, all });
            break;
        }
        case 'setTarget':
            reply(`[${bot.username}] ` + ctx.targetManager.handleCommand(args.slice(1)));
            break;
        case 'greedy': {
            // Бот перестаёт действовать случайно (epsilon 0) и показывает,
            // чему сети уже научились. Повторная команда — вернуть обычное
            // исследование. Если команду слышат все боты (общий чат), то
            // реагирует один: номер из команды (!greedy 3), по умолчанию №1.
            // Адресованная одному боту — номер не нужен.
            const wanted = Number(args[1] ?? 1);
            if (!direct && wanted !== ctx.id) break;
            toPython({ cmd: 'toggle_greedy' });
            break;
        }
        case 'start': {
            // "Останови меня" (задачка hunt): охота на того, кто написал, или
            // на игрока из команды (!start <ник>). Общий чат слышат все боты —
            // Python-у шлёт один (первый подключённый).
            if (!direct && ctx !== contexts.find((c) => c.spawned)) break;
            toPython({ cmd: 'hunt_start', target: args[1] ?? speaker });
            break;
        }
        case 'stop':
            pauseLoop();
            for (const c of contexts) if (c.spawned) c.executor.stopMovement();
            paused = true;
            reply('Пауза: ИИ остановлен у всех ботов. !resume — продолжить.');
            break;
        case 'resume':
            if (!paused) return reply('ИИ и так работает.');
            paused = false;
            startLoop();
            reply('Продолжаю работу ИИ.');
            break;
        case 'debug': {
            const e = bot.entity;
            const deg = (rad) => Math.round((rad * 180) / Math.PI);
            reply(`[${bot.username}] pos (${e.position.x.toFixed(1)}, ${e.position.y.toFixed(1)}, ${e.position.z.toFixed(1)}), yaw ${deg(e.yaw)}° pitch ${deg(e.pitch)}°, hp ${bot.health}/20, еда ${bot.food}/20, ${e.onGround ? 'на земле' : 'в воздухе'}, тик ${tickIndex}${paused ? ', ПАУЗА' : ''}`);
            break;
        }
        case 'entities': {
            const list = nearestEntitySummary(bot);
            reply(`[${bot.username}] ` + (list.length ? list.join('; ') : 'Рядом никого нет.'));
            break;
        }
        case 'whatYouSee': {
            const res = config.vision.resolution;
            const entityCount = Object.values(bot.entities)
                .filter((e) => e.isValid && e !== bot.entity).length;
            reply(`[${bot.username}] Сетка ${res[0]}x${res[1]} (${config.vision.mode}), дальность ${config.vision.distance} чанка, сущностей рядом: ${entityCount}. Цель: ${ctx.targetManager.describe()}.`);
            break;
        }
        default:
            reply(`Не знаю команду "${command}". Напиши !help`);
    }
}

// Команда из строки лаунчера: to — номер бота или 'all'. Ответы — в консоль
// с меткой [cmd], лаунчер показывает их отдельным цветом.
//
// "Всем" ведёт себя как общий чат, но без семи одинаковых ответов:
// общеройные команды (stop/resume/help, task) выполняются один раз;
// setTarget — у каждого бота, а ответ печатается один; отчёты
// (debug/entities/whatYouSee) — от каждого бота.
const PER_BOT_REPORTS = new Set(['debug', 'entities', 'whatYouSee']);
const SWARM_WIDE = new Set(['stop', 'resume', 'help']);

function runConsoleCommand(to, text) {
    const args = text.replace(/^!/, '').trim().split(/\s+/);
    const say = (line) => console.log(`[cmd] ${line}`);
    const spawned = contexts.filter((c) => c.spawned);
    if (spawned.length === 0) return say('Нет подключённых ботов — команду выполнить некому.');

    if (to !== 'all') {
        const ctx = spawned.find((c) => c.id === Number(to));
        if (!ctx) return say(`Бот №${to} не подключён.`);
        return runCommand(ctx, args, say, { direct: true, via: 'console' });
    }

    const command = args[0];
    if (SWARM_WIDE.has(command)) return runCommand(spawned[0], args, say, { via: 'console' });
    if (command === 'task') {
        const task = args[1] === 'all' ? args[2] : args[1];
        return runCommand(spawned[0], ['task', 'all', ...(task ? [task] : [])], say, { via: 'console' });
    }
    spawned.forEach((ctx, index) => {
        const quiet = index > 0 && !PER_BOT_REPORTS.has(command);
        runCommand(ctx, args, quiet ? () => {} : say, { via: 'console' });
    });
}

// Краткий список ближайших сущностей для отладки в чате.
function nearestEntitySummary(bot) {
    return Object.values(bot.entities)
        .filter((e) => e && e.isValid && e !== bot.entity && e.position)
        .map((e) => ({ e, d: e.position.distanceTo(bot.entity.position) }))
        .filter(({ d }) => d <= config.entities.radius)
        .sort((a, b) => a.d - b.d)
        .slice(0, 5)
        .map(({ e, d }) => {
            const name = e.name ?? e.entityName ?? '?';
            const kind = {
                [PLAYER]: `игрок ${e.username}`,
                [HOSTILE]: `враждебный ${name}`,
                [PASSIVE]: `мирный ${name}`,
                [NON_LIVING]: name, // item, arrow, experience_orb...
            }[classifyEntity(e)];
            return `${kind} (${d.toFixed(1)}м)`;
        });
}

async function shutdown() {
    if (shuttingDown) return;
    shuttingDown = true;
    pauseLoop();
    // Незаспавненный бот (ещё подключается или ждёт переподключения) —
    // у него ещё нет физики и clearControlStates, трогать нечего.
    for (const ctx of contexts) if (ctx.spawned) ctx.executor.stopMovement();
    await bridge.stop();
    for (const ctx of contexts) {
        try {
            ctx.bot.quit();
        } catch {
            /* соединения и так нет */
        }
    }
    process.exit(0);
}
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);

// Лаунчер (py/launcher.py) общается с Node через stdin, построчно:
//   stop                                    — аккуратно остановиться (Ctrl+C
//                                             процессу без консоли не послать);
//   cmd {"to": "all" | <номер>, "text": "!task looking"} — команда из строки
//                                             команд лаунчера.
// В обычной консоли stdin — терминал, и это не мешает.
if (!process.stdin.isTTY) {
    process.stdin.setEncoding('utf8');
    let pending = '';
    process.stdin.on('data', (chunk) => {
        pending += chunk;
        const lines = pending.split(/\r?\n/);
        pending = lines.pop(); // незаконченная строка — дождаться остатка
        for (const line of lines) {
            const trimmed = line.trim();
            if (trimmed === 'stop') return shutdown();
            if (!trimmed.startsWith('cmd ')) continue;
            try {
                const { to, text } = JSON.parse(trimmed.slice(4));
                runConsoleCommand(to ?? 'all', String(text ?? ''));
            } catch (err) {
                console.log(`[cmd] Не понял команду от лаунчера: ${err.message}`);
            }
        }
    });
}

// Цикл стартует, когда заспавнился первый бот (bridge поднялся).
// Если ботов несколько и остальные подключаются позже — они просто
// добавляются в обработку своим флагом spawned.
const firstSpawnCheck = setInterval(() => {
    if (contexts.some((c) => c.spawned)) {
        clearInterval(firstSpawnCheck);
        startLoop();
    }
}, 500);
