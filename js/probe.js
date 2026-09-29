// Живая проверка на настоящем сервере: npm run probe
//
// Бот "Probe" заходит на сервер из config.json, сверяет наш код с самим
// mineflayer и печатает отчёт. Никого не бьёт; крутит головой, делает пару
// шагов и выкапывает две клетки земли рядом с собой — и тут же засыпает их
// обратно (проверка рук). Зачем: синтетические тесты
// (test_vision.js) проверяют математику, а тут — что она совпадает с тем,
// как реально ведёт себя игра (в проекте уже были тихие ошибки: слепое
// зрение и зеркальный мир — и то и другое синтетика долго не ловила).
//
// Проверки:
//   1. луч "в прицел" (centerRaycast) против mineflayer'овского
//      bot.blockAtEntityCursor в десятках случайных направлений;
//   2. после bot.lookAt на ближайшего игрока он в списке сущностей
//      "прямо впереди" (right ≈ 0), а после turn_left — справа;
//   3. look_up поднимает взгляд, walk_forward ведёт туда, куда смотрим;
//   4. картинка сетки зрения — классы блоков буквами, чтобы глянуть глазами;
//   5. маршрут (js/route.js) в настоящем мире;
//   6. руки: копнуть и подобрать блок, поставить на грань в прицеле
//      (place_front) и под себя в прыжке (place_below) — в выживании.

const mineflayer = require('mineflayer');
const { loadConfig } = require('./config');
const { buildVisionGrid, centerRaycast, lookDirection, isPassThrough } = require('./vision');
const { buildEntitiesList } = require('./entities');
const { Vec3 } = require('vec3');
const { ActionExecutor } = require('./actions');
const { WorldBorder } = require('./border');
const { RoutePlanner } = require('./route');
const { isPlaceableBlock } = require('./inventory');

const config = loadConfig();
const bot = mineflayer.createBot({
    host: config.bot.host,
    port: config.bot.port,
    username: process.argv[2] || 'Probe',
    version: config.bot.version || false,
});
const border = new WorldBorder(bot); // до входа — чтобы не пропустить стартовый пакет границы

const results = [];
function check(name, ok, details = '') {
    results.push({ name, ok });
    console.log(`${ok ? '  OK ' : 'FAIL '} ${name}${details ? ' — ' + details : ''}`);
}
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const deg = (rad) => Math.round((rad * 180) / Math.PI);

const giveUp = setTimeout(() => {
    console.log('[probe] Не смог зайти на сервер за 30 с — он запущен? host/port в config.json верные?');
    process.exit(2);
}, 30000);

bot.on('kicked', (reason) => console.log('[probe] Кикнут:', reason));
bot.on('error', (err) => console.log('[probe] Ошибка:', err.message || err.code));

bot.once('spawn', async () => {
    clearTimeout(giveUp);
    try {
        await bot.waitForChunksToLoad();
        await sleep(1500);
        const p = bot.entity.position;
        const difficulty = bot.game.difficulty ? `, сложность ${bot.game.difficulty}` : '';
        console.log(`[probe] Версия ${bot.version}, режим ${bot.game.gameMode}${difficulty}, `
            + `позиция (${p.x.toFixed(1)}, ${p.y.toFixed(1)}, ${p.z.toFixed(1)})`);

        checkBorder();
        printVision();
        checkRaycast();
        await checkEntityFrame();
        checkLookUp();
        await checkWalking();
        checkRoute();
        await checkDigCancel();
        await checkHands();
    } catch (err) {
        console.log('[probe] Проверка упала:', err.stack || err.message);
        results.push({ name: 'без исключений', ok: false });
    }

    const failed = results.filter((r) => !r.ok).length;
    console.log(failed === 0 ? '[probe] Всё сходится.' : `[probe] Не сошлось проверок: ${failed} из ${results.length}.`);
    bot.quit();
    setTimeout(() => process.exit(failed === 0 ? 0 : 1), 500);
});

// Граница мира должна быть известна (охранник в bot.js без неё не работает),
// и сам пробник — внутри неё.
function checkBorder() {
    check('граница мира известна', border.known,
        border.known ? `центр (${border.centerX.toFixed(1)}, ${border.centerZ.toFixed(1)}), размер ${border.size.toFixed(0)}` : 'сервер её не прислал');
    if (border.known) {
        const p = bot.entity.position;
        const toEdge = border.size / 2 - Math.max(Math.abs(p.x - border.centerX), Math.abs(p.z - border.centerZ));
        check('пробник внутри границы мира', !border.isUnsafe(p, 0), `до края ${toEdge.toFixed(1)} блока`);
    }
}

// Сетка зрения буквами: . небо/пусто, W древесина, S камень, D земля,
// P растения, L жидкость, O руда, I снег/лёд, G стекло, ? прочее;
// строчная буква — ближе 4 блоков. Ряд 0 — верх, колонка 0 — прямо вперёд,
// дальше по кругу вправо.
function printVision() {
    const [resX, resY] = config.vision.resolution;
    const maxDistance = config.vision.distance * 16;
    const letters = '.WSDPLOIG?';
    const cells = buildVisionGrid(bot.world, bot.entity, config);
    console.log('[probe] Сетка зрения (ряд 0 — верх, колонка 0 — вперёд, дальше по кругу вправо):');
    for (let row = 0; row < resY; row++) {
        let line = '        ';
        for (let col = 0; col < resX; col++) {
            const cell = cells[row * resX + col];
            const letter = letters[cell.t] ?? '?';
            line += (cell.t !== 0 && cell.d * maxDistance < 4 ? letter.toLowerCase() : letter) + ' ';
        }
        console.log(line);
    }
    const sky = cells.filter((c) => c.t === 0).length;
    check('зрение что-то видит (не всё небо)', sky < cells.length, `небо/пусто в ${sky} из ${cells.length} лучей`);
}

// Наш луч в прицел против эталона mineflayer — bot.blockAtEntityCursor:
// направление взгляда он считает СВОЕЙ функцией (независимая проверка
// нашей конвенции углов). Одна поправка: сам mineflayer пускает этот луч с
// макушки (position + height = 1.8), а не из глаз (1.62, как в игре и у
// нас) — у земли под острым углом 18 см дают соседний блок. Поэтому
// эталону передаём ту же точку глаз, что у нас.
// Расхождение допустимо, только если mineflayer упёрся в проходимую
// растительность (траву, цветы), которую наш луч специально пропускает.
function checkRaycast() {
    const original = { yaw: bot.entity.yaw, pitch: bot.entity.pitch };
    const maxDistance = config.vision.distance * 16;
    let same = 0;
    let plants = 0;
    const mismatches = [];
    const tries = 60;
    for (let i = 0; i < tries; i++) {
        bot.entity.yaw = (Math.random() * 2 - 1) * Math.PI;
        bot.entity.pitch = -1.2 + Math.random() * 2.0; // от "под ноги" до "вверх"
        const ours = centerRaycast(bot.world, bot.entity, config);
        const eyes = { position: bot.entity.position, height: bot.entity.eyeHeight, yaw: bot.entity.yaw, pitch: bot.entity.pitch };
        const theirs = bot.blockAtEntityCursor(eyes, maxDistance);
        const oursPos = ours?.block?.position;
        if ((!oursPos && !theirs) || (oursPos && theirs && oursPos.equals(theirs.position))) {
            same++;
        } else if (theirs && isPassThrough(theirs.name)) {
            plants++;
        } else {
            mismatches.push(`yaw ${deg(bot.entity.yaw)}° pitch ${deg(bot.entity.pitch)}°: наш ${ours?.block?.name ?? 'ничего'}, mineflayer ${theirs?.name ?? 'ничего'}`);
        }
    }
    bot.entity.yaw = original.yaw;
    bot.entity.pitch = original.pitch;
    check('луч в прицел совпадает с эталоном mineflayer', mismatches.length <= 2,
        `совпало ${same}, трава/цветы ${plants}, расхождений ${mismatches.length} из ${tries}`);
    for (const m of mismatches.slice(0, 5)) console.log('        ' + m);
}

// Сущность-игрок: после lookAt — прямо впереди, после turn_left — справа.
async function checkEntityFrame() {
    const others = Object.values(bot.entities)
        .filter((e) => e.type === 'player' && e !== bot.entity && e.position.distanceTo(bot.entity.position) < config.entities.radius);
    if (others.length === 0) {
        console.log('  --  Рядом (ближе 16 блоков) нет игроков — проверку "впереди/справа" пропускаю. Подойди к боту Probe и запусти снова.');
        return;
    }
    const target = others.sort((a, b) => a.position.distanceTo(bot.entity.position) - b.position.distanceTo(bot.entity.position))[0];
    await bot.lookAt(target.position.offset(0, target.eyeHeight ?? 1.62, 0), true);
    await sleep(200);
    const find = () => buildEntitiesList(bot.entities, bot.entity, config).find((e) => e.id === target.id);
    const ahead = find();
    check(`${target.username} после lookAt — прямо впереди`, ahead && ahead.forward > 0 && Math.abs(ahead.right) < 0.03,
        ahead ? `forward ${ahead.forward}, right ${ahead.right}` : 'не найден в списке сущностей');

    const executor = new ActionExecutor(bot, config);
    const yawBefore = bot.entity.yaw;
    executor.executeAll({ legs: 'turn_left' });
    await sleep(200);
    const turned = find();
    check('turn_left увеличивает yaw (поворот влево в mineflayer)', Math.abs(deg(bot.entity.yaw - yawBefore) - 30) <= 1,
        `yaw ${deg(yawBefore)}° -> ${deg(bot.entity.yaw)}°`);
    check(`после поворота влево ${target.username} — справа`, turned && turned.right > 0.05,
        turned ? `forward ${turned.forward}, right ${turned.right}` : 'не найден');
    executor.stopMovement();
}

function checkLookUp() {
    const executor = new ActionExecutor(bot, config);
    bot.entity.pitch = 0;
    const before = lookDirection(bot.entity).y;
    executor.executeAll({ head: 'look_up' });
    const after = lookDirection(bot.entity).y;
    check('look_up поднимает взгляд', after > before, `y взгляда ${before.toFixed(2)} -> ${after.toFixed(2)}, pitch ${deg(bot.entity.pitch)}°`);
    bot.entity.pitch = 0;
}

// walk_forward ведёт туда, куда смотрит бот. Идём только туда, где впереди
// свободно (луч на уровне глаз не упирается ближе 5 блоков), и судим только
// по нормальному пройденному пути: упёрся боком в дерево/игрока — это не
// ошибка направления, а препятствие.
async function checkWalking() {
    const executor = new ActionExecutor(bot, config);
    const startYaw = bot.entity.yaw;
    for (let attempt = 0; attempt < 8; attempt++) {
        bot.entity.yaw = startYaw + attempt * (Math.PI / 4);
        bot.entity.pitch = 0;
        const ahead = centerRaycast(bot.world, bot.entity, config);
        if (ahead && ahead.distance < 5) continue; // впереди стена — другое направление
        const start = bot.entity.position.clone();
        const forward = lookDirection(bot.entity);
        for (let i = 0; i < 8; i++) {
            executor.executeAll({ legs: 'walk_forward' });
            await sleep(150);
        }
        executor.stopMovement();
        await sleep(300);
        const moved = bot.entity.position.minus(start);
        const flat = Math.hypot(moved.x, moved.z);
        if (flat < 2) continue; // упёрлись во что-то — пробуем другое направление
        const alignment = (moved.x * forward.x + moved.z * forward.z) / (flat * Math.hypot(forward.x, forward.z));
        check('walk_forward ведёт туда, куда смотрит бот', alignment > 0.8,
            `прошёл ${flat.toFixed(1)} блока, совпадение с направлением взгляда ${alignment.toFixed(2)}`);
        return;
    }
    console.log('  --  Нигде не удалось пройти 2 блока по прямой (зажат?) — проверку ходьбы пропускаю.');
}

// Маршрут в настоящем мире: к точке в 10 блоках путь найден, точка маршрута
// стоит на твёрдом, длина не короче прямой (с допуском на "дошёл до клетки
// рядом с целью").
function checkRoute() {
    const p = bot.entity.position;
    for (let attempt = 0; attempt < 8; attempt++) {
        const angle = attempt * (Math.PI / 4);
        const target = { x: p.x + Math.sin(angle) * 10, y: p.y, z: p.z + Math.cos(angle) * 10 };
        const planner = new RoutePlanner(bot.world, bot.registry, config.route);
        const started = Date.now();
        const route = planner.update(p, target);
        const spent = Date.now() - started;
        if (!route || !route.complete) continue;
        const w = route.waypoint;
        const onGround = planner.standable(Math.floor(w.x), w.y, Math.floor(w.z));
        const straight = Math.hypot(target.x - p.x, target.z - p.z);
        check('маршрут к точке в 10 блоках найден', onGround && route.length >= straight - 2.5,
            `длина ${route.length} (по прямой ${straight.toFixed(1)}), клеток ${planner.path.length}, `
            + `точка маршрута ${onGround ? 'на твёрдом' : 'В ВОЗДУХЕ'}, ${spent} мс`);
        return;
    }
    console.log('  --  Ни в одну сторону маршрут на 10 блоков не нашёлся (ямы, вода?) — проверку пропускаю.');
}

// Начали копать и отвернулись — копка прерывается, блок цел (раньше
// mineflayer докапывал начатое, куда бы бот ни смотрел и ни шёл — боты
// ломали блоки издалека). Подойдёт любой мягкий блок рядом: земля
// впереди под ногами или стенка ямки на уровне ног.
async function checkDigCancel() {
    if (bot.game.gameMode !== 'survival') return;
    const soft = /^(dirt|grass_block|coarse_dirt|rooted_dirt|podzol|mycelium|sand|red_sand)$/;
    const feet = bot.entity.position.floored();
    const executor = new ActionExecutor(bot, config);
    for (const [dx, dz] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        for (const target of [feet.offset(dx, 0, dz), feet.offset(dx, -1, dz)]) {
            if (!soft.test(bot.blockAt(target)?.name ?? '')) continue;
            await bot.lookAt(target.offset(0.5, 0.5, 0.5), true);
            const hit = centerRaycast(bot.world, bot.entity, config);
            if (!hit?.block?.position.equals(target)) continue; // загорожен — другой блок
            executor.executeAll({ hands: 'attack_center' });
            await sleep(150);
            bot.entity.yaw += Math.PI / 2;
            executor.tick(); // в bot.js — в начале каждого тика
            await sleep(1500);
            check('отвернулся посреди копки — копка прервалась, блок цел',
                bot.blockAt(target)?.boundingBox === 'block' && !executor.handsBusy,
                `${bot.blockAt(target)?.name} в ${target}, руки ${executor.handsBusy ? 'заняты' : 'свободны'}`);
            return;
        }
    }
    console.log('  --  Рядом нет мягкого блока в прицеле — проверку прерывания копки пропускаю.');
}

// Руки вживую. Впереди две клетки мягкой земли подряд под ногами (A, B):
//   1. копаем A (attack_center), спрыгиваем в ямку — блок подбирается;
//   2. из ямки копаем B (он теперь на уровне ног) — второй блок;
//   3. place_front, глядя на дно ямки B, — блок встаёт в ямку B;
//   4. jump + place_below в одном тике — блок встаёт в ямку A, бот на нём.
// Мир остаётся как был (разве что земля вместо травы).
async function checkHands() {
    if (bot.game.gameMode !== 'survival') {
        console.log(`  --  Режим ${bot.game.gameMode}, не выживание — проверку рук пропускаю.`);
        return;
    }
    const spot = findDigSpot();
    if (!spot.A) {
        console.log(`  --  Рядом нет ровной мягкой земли (две клетки подряд) — проверку рук пропускаю: ${spot.reasons}`);
        return;
    }
    const { A, B, dx, dz } = spot;
    const executor = new ActionExecutor(bot, config);
    const blocks = () => bot.inventory.items().filter((it) => isPlaceableBlock(bot, it)).reduce((n, it) => n + it.count, 0);
    const start = blocks();

    await bot.lookAt(A.offset(0.5, 0.95, 0.5), true);
    executor.executeAll({ hands: 'attack_center' });
    const dugA = await waitFor(() => bot.blockAt(A)?.boundingBox === 'empty', 6000);
    check('attack_center копает блок в прицеле', dugA, `${spot.name} в ${A}`);
    if (!dugA) return;
    bot.entity.yaw = Math.atan2(-dx, -dz);
    bot.entity.pitch = 0;
    for (let i = 0; i < 4; i++) {
        executor.executeAll({ legs: 'walk_forward' });
        await sleep(150);
    }
    executor.stopMovement();
    const gotA = await waitFor(() => blocks() > start, 3000);
    check('выкопанный блок подобран', gotA, `блоков в инвентаре ${start} -> ${blocks()}`);
    if (!gotA) return;

    await bot.lookAt(B.offset(0.5, 0.5, 0.5), true);
    executor.executeAll({ hands: 'attack_center' });
    const gotB = await waitFor(() => bot.blockAt(B)?.boundingBox === 'empty', 6000) && await waitFor(() => blocks() > start + 1, 3000);
    check('второй блок выкопан и подобран', gotB, `блоков в инвентаре ${blocks()}`);
    if (!gotB) return;

    // Дно ямки B — верхняя грань блока под ней: блок встанет на неё, в B.
    await bot.lookAt(B.offset(0.5, 0.05, 0.5), true);
    await waitFor(() => !executor.handsBusy, 2000);
    executor.executeAll({ hands: 'place_front' });
    const placedFront = await waitFor(() => bot.blockAt(B)?.boundingBox === 'block', 3000);
    check('place_front ставит блок на грань в прицеле', placedFront, `в ${B}: ${bot.blockAt(B)?.name}`);

    await waitFor(() => !executor.handsBusy, 3000);
    const feetBefore = bot.entity.position.y;
    executor.executeAll({ legs: 'jump', hands: 'place_below' }); // одним тиком
    const placedBelow = await waitFor(() => bot.blockAt(A)?.boundingBox === 'block', 3000);
    await sleep(600); // приземлиться
    executor.stopMovement();
    const rose = bot.entity.position.y - feetBefore;
    check('jump + place_below в одном тике — бот встал на свой блок', placedBelow && rose > 0.9,
        `в ${A}: ${bot.blockAt(A)?.name}, бот поднялся на ${rose.toFixed(2)}`);
}

// Где копать: { A, B, dx, dz, name } или { reasons } — почему не подошло
// ни одно направление (чтобы было видно, что мешает).
function findDigSpot() {
    const soft = /^(dirt|grass_block|coarse_dirt|rooted_dirt|podzol|mycelium|sand|red_sand)$/;
    const feet = bot.entity.position.floored();
    const solid = (p) => bot.blockAt(p)?.boundingBox === 'block';
    const empty = (p) => bot.blockAt(p)?.boundingBox === 'empty';
    const reasons = [];
    for (const [dx, dz] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const at = (k, dy) => feet.offset(k * dx, dy, k * dz);
        const A = at(1, -1);
        const B = at(2, -1);
        const names = [bot.blockAt(A)?.name, bot.blockAt(B)?.name];
        if (!names.every((name) => soft.test(name ?? ''))) {
            reasons.push(`(${dx},${dz}) под ногами ${names.join(' и ')}`);
            continue;
        }
        // Дно ямок — твёрдое (в ямке стоять, на дно ставить), над ямками свободно.
        if (!solid(at(1, -2)) || !solid(at(2, -2))) {
            reasons.push(`(${dx},${dz}) под землёй пусто`);
            continue;
        }
        if (![at(1, 0), at(1, 1), at(2, 0), at(2, 1)].every(empty)) {
            reasons.push(`(${dx},${dz}) над ямками занято`);
            continue;
        }
        return { A, B, dx, dz, name: names[0] };
    }
    return { reasons: reasons.join('; ') };
}

async function waitFor(condition, timeoutMs) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        if (condition()) return true;
        await sleep(50);
    }
    return condition();
}
