// Режим записи геймплея (behavior cloning). Запуск: npm run record
//
// Бот-«призрак» подключается к серверу, следит за твоим игроком и каждый
// тик пишет в data/demonstrations.jsonl строку {state, actions}:
//   state   — то же наблюдение, что бот видел бы с твоей точки зрения
//             (raycast-сетка, слух, дельты движения);
//   actions — твои действия по каналам (ноги/голова/руки, как у бота —
//             см. CHANNELS в js/actions.js), выведенные из дельт (обратная
//             динамика): пошёл вперёд -> legs: walk_forward, чуть довернул
//             мышь -> head: look_left, махнул рукой -> hands: attack_center.
//             Каналы независимы: "иду и одновременно смотрю вверх" — это
//             walk_forward + look_up за один тик, как и у бота.
//
// Управление из чата: !rec start / !rec stop / !rec status.
// Одно твоё движение может маппиться не идеально (мы выводим его из
// позиций, а не читаем клавиатуру), но для BC важна статистика, а не
// попиксельная точность.

const fs = require('fs');
const mineflayer = require('mineflayer');
const { loadConfig } = require('./config');
const { StateBuilder } = require('./state');
const { Hearing } = require('./hearing');

const config = loadConfig();

const bot = mineflayer.createBot({
    host: config.bot.host,
    port: config.bot.port,
    username: 'AI_Recorder',
    version: config.bot.version || false,
});

const stateBuilder = new StateBuilder(config);
let hearing = null;

let recording = false;
let tickIndex = 0;
let written = 0;
let trackedPlayerName = null;
let outputStream = null;
// Руки: взмах рукой отслеживаемого игрока (удар/копание) между тиками.
// Сервер присылает анимацию взмаха всем игрокам рядом — обсёрверу тоже.
let swungSinceLastTick = false;

const OUTPUT_PATH = __dirname + '/../data/demonstrations.jsonl';

bot.once('spawn', () => {
    console.log(`[rec] Обсёрвер заспавнился @ ${bot.entity.position}`);
    hearing = new Hearing(bot, config);

    bot.on('chat', onChat);
    bot.on('entitySwingArm', (entity) => {
        if (trackedPlayerName && entity.username === trackedPlayerName) swungSinceLastTick = true;
    });
    setInterval(tick, config.train.tick_rate_ms);
});

// Игрок, за которым следим: ближайший, кто не сам обсёрвер.
function findPlayer() {
    let best = null;
    let bestDist = Infinity;
    for (const [name, player] of Object.entries(bot.players)) {
        if (name === bot.username || !player.entity) continue;
        const dist = player.entity.position.distanceTo(bot.entity.position);
        if (dist < bestDist) {
            bestDist = dist;
            best = player;
        }
    }
    return best;
}

function tick() {
    try {
        const player = findPlayer();
        if (!player) {
            if (recording) console.log('[rec] Игрок пропал — пауза записи.');
            recording = false;
            return;
        }
        trackedPlayerName = player.username;
        const entity = player.entity;

        const state = stateBuilder.build(
            bot.world,
            { position: entity.position, height: entity.height, eyeHeight: entity.eyeHeight, yaw: entity.yaw, pitch: entity.pitch },
            hearing,
            tickIndex++,
            null, 'none', false, false, null, null,
            bot.entities
        );

        if (recording) {
            const actions = {
                legs: inferLegs(state.self),
                head: inferHead(state.self),
                hands: swungSinceLastTick ? 'attack_center' : 'hands_idle',
            };
            swungSinceLastTick = false;
            outputStream.write(JSON.stringify({ state, actions }) + '\n');
            written++;
            if (written % 200 === 0) {
                console.log(`[rec] Записано тиков: ${written}`);
            }
        }
    } catch (err) {
        console.error('[rec] Ошибка тика:', err.message);
    }
}

// Обратная динамика: дельты -> макросы каналов. Пороги в радианах/блоках
// за один тик (tick_rate_ms = 150 мс).
const TURN_THRESHOLD = 0.30;     // ~17°: такой рывок мышью — это поворот корпуса (turn_*, 30°)
const LOOK_YAW_THRESHOLD = 0.08; // ~5°: мельче — доворот взгляда головой (look_left/right, 10°)
const PITCH_THRESHOLD = 0.10;
const MOVE_THRESHOLD = 0.15;     // блока за тик; ходьба даёт ~0.6

// Ноги: одно действие за тик — выбираем доминирующий сигнал.
function inferLegs(self) {
    const absYaw = Math.abs(self.dyaw);
    const absFwd = Math.abs(self.move_forward);
    const absRight = Math.abs(self.move_right);

    // Прыжок: поднялся и при этом двигался горизонтально.
    if (self.move_up > 0.2 && (absFwd > MOVE_THRESHOLD || absRight > MOVE_THRESHOLD)) {
        return 'jump_forward';
    }
    // Крупный поворот — работа корпуса, доминирует над шагом.
    if (absYaw >= TURN_THRESHOLD) {
        // yaw в mineflayer растёт влево: turn_left увеличивает yaw.
        return self.dyaw > 0 ? 'turn_left' : 'turn_right';
    }
    if (absFwd > MOVE_THRESHOLD) {
        return self.move_forward > 0 ? 'walk_forward' : 'walk_back';
    }
    if (absRight > MOVE_THRESHOLD) {
        return self.move_right > 0 ? 'strafe_right' : 'strafe_left';
    }
    return 'idle';
}

// Голова: наклон важнее доворота (доворот мелкий и часто просто дрожь
// мыши). Крупный поворот уже забрали ноги — голове остаётся только мелкий.
function inferHead(self) {
    const absYaw = Math.abs(self.dyaw);
    // pitch > 0 — вверх, yaw растёт влево (конвенция mineflayer).
    if (Math.abs(self.dpitch) > PITCH_THRESHOLD) {
        return self.dpitch > 0 ? 'look_up' : 'look_down';
    }
    if (absYaw >= LOOK_YAW_THRESHOLD && absYaw < TURN_THRESHOLD) {
        return self.dyaw > 0 ? 'look_left' : 'look_right';
    }
    return 'head_idle';
}

function onChat(username, message) {
    if (username === bot.username) return;
    if (!message.startsWith('!rec')) return;

    const args = message.slice(4).trim().split(/\s+/);
    const sub = args[0];

    if (sub === 'start') {
        if (recording) return bot.chat('Запись уже идёт.');
        outputStream = fs.createWriteStream(OUTPUT_PATH, { flags: 'a' });
        recording = true;
        bot.chat(`Запись пошла (слежу за ${trackedPlayerName}). Напиши !rec stop чтобы остановить.`);
        console.log('[rec] Запись начата.');
    } else if (sub === 'stop') {
        if (!recording) return bot.chat('Запись и не шла.');
        recording = false;
        outputStream.end();
        outputStream = null;
        bot.chat(`Запись остановлена, тиков записано: ${written}.`);
        console.log(`[rec] Запись остановлена. Всего тиков: ${written}`);
    } else if (sub === 'status') {
        bot.chat(`Запись: ${recording ? 'идёт' : 'стоит'}, тиков: ${written}, игрок: ${trackedPlayerName ?? 'не найден'}.`);
    } else {
        bot.chat('Использование: !rec start | stop | status');
    }
}

bot.on('kicked', (reason) => console.log('[rec] Кикнут:', reason));
bot.on('error', (err) => console.log('[rec] Ошибка:', err.message));
bot.on('end', () => {
    if (outputStream) outputStream.end();
    console.log('[rec] Соединение закрыто.');
});

async function shutdown() {
    if (outputStream) outputStream.end();
    bot.quit();
    process.exit(0);
}
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
