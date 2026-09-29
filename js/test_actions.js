// Тест макросов без сервера: npm run test-actions
//
// Автопрыжок (ActionExecutor.stepAhead): walk_forward/sprint_forward сами
// подпрыгивают, если по курсу уступ в один блок, над которым свободно, — и
// не прыгают у стены в два блока, под потолком и на ровном месте.
// Мир — словарь "x,y,z" -> твёрдый блок, остальное — воздух.

const assert = require('assert');
const { EventEmitter } = require('events');
const { Vec3 } = require('vec3');
const { ActionExecutor } = require('./actions');
const { loadConfig } = require('./config');
// Настоящий мир prismarine (как в js/test_vision.js): удару нужна прямая
// видимость до цели (lineOfSight), а она — raycast по блокам мира.
const registry = require('prismarine-registry')('1.20.1');
const Chunk = require('prismarine-chunk')(registry);
const World = require('prismarine-world')(registry);

const config = loadConfig();

function fakeBot(solidBlocks, yaw) {
    const controls = {};
    // События (entityHurt) — как у настоящего бота mineflayer.
    const bot = Object.assign(new EventEmitter(), {
        entity: { position: new Vec3(0.5, 64, 0.5), yaw, pitch: 0 },
        world: new World(() => new Chunk()).sync, // пустой мир: ничто не заслоняет
        blockAt: (p) => ({ boundingBox: solidBlocks.has(`${p.x},${p.y},${p.z}`) ? 'block' : 'empty' }),
        setControlState: (control, state) => { controls[control] = state; },
        clearControlStates: () => { for (const key of Object.keys(controls)) delete controls[key]; },
        controls,
    });
    return bot;
}

function jumps(solid, yaw = 0, action = 'walk_forward') {
    const bot = fakeBot(new Set(solid), yaw);
    const executor = new ActionExecutor(bot, config);
    executor.executeAll({ legs: action });
    return Boolean(bot.controls.jump) && Boolean(bot.controls.forward);
}

// yaw = 0 в mineflayer — взгляд на север, вперёд = -Z; бот стоит в (0.5, 64, 0.5).
assert.strictEqual(jumps([]), false, 'ровное место — не прыгать');
assert.strictEqual(jumps(['0,64,-1']), true, 'уступ в блок впереди — прыгнуть');
assert.strictEqual(jumps(['0,64,-1'], 0, 'sprint_forward'), true, 'и на бегу тоже');
assert.strictEqual(jumps(['0,64,-1', '0,65,-1']), false, 'стена в два блока — не перепрыгнуть');
assert.strictEqual(jumps(['0,64,-1', '0,66,0']), false, 'потолок над головой — не прыгать');
assert.strictEqual(jumps(['0,64,-1', '0,66,-1']), false, 'над уступом тесно — не пролезть');
// yaw = pi/2 — поворот влево от севера, взгляд на запад: вперёд = -X.
assert.strictEqual(jumps(['-1,64,0'], Math.PI / 2), true, 'уступ на западе, смотрит на запад — прыгнуть');
assert.strictEqual(jumps(['0,64,-1'], Math.PI / 2), false, 'уступ сбоку — не прыгать');
console.log('Автопрыжок в порядке.');

// Удар по убегающему проходит, даже когда руки заняты (копает): иначе
// водящий стоял вплотную и "не бил". Копка при этом бросается. Бить можно
// только тех, кого разрешил судья (tag_ids), — другого водящего нельзя.
{
    const bot = fakeBot(new Set(), 0);
    const runner = { id: 42, type: 'player', username: 'AI_7', name: 'player', height: 1.8, position: new Vec3(0.5, 64, -1.5) };
    bot.entity.height = 1.8;
    bot.entities = { 42: runner };
    const hits = [];
    let stopped = false;
    bot.attack = (entity) => hits.push(entity.id);
    bot.lookAt = () => {};
    bot.stopDigging = () => { stopped = true; };
    const executor = new ActionExecutor(bot, config);
    executor.handsBusy = true;          // занят копкой
    executor.digging = new Vec3(3, 63, 3);
    executor.setTaggable([7]);          // водящий, но AI_7 (id 42) сейчас тоже водит
    executor.executeAll({ hands: 'attack_center' });
    assert.deepStrictEqual(hits, [], 'водящего водящий не бьёт');
    executor.setTaggable([42]);         // AI_7 убегает
    executor.executeAll({ hands: 'attack_center' });
    assert.deepStrictEqual(hits, [42], 'удар по убегающему прошёл');
    assert.ok(stopped && executor.digging === null, 'копка брошена');
    assert.strictEqual(executor.takeAttacked(), 42, 'судье ушёл id осаленного');
    executor.setTaggable(undefined);    // убегающему бить игроков нельзя
    executor.executeAll({ hands: 'attack_center' });
    assert.deepStrictEqual(hits, [42], 'убегающий никого не бьёт');

    // Ударили самого бота — судье уходит, кто (так видно удар человека-водящего).
    assert.strictEqual(executor.takeHurtBy(), null, 'никто не бил');
    bot.emit('entityHurt', runner, { id: 99 });           // ударили не его
    bot.emit('entityHurt', bot.entity, { id: 77 });        // ударили его
    assert.strictEqual(executor.takeHurtBy(), 77, 'судье ушёл id ударившего');
    assert.strictEqual(executor.takeHurtBy(), null, 'и только один раз');
}
console.log('Удар водящего в порядке.');

// Прицел удара: водящий смотрит ровно (pitch 0) на север. Бьёт и вплотную,
// и чуть сбоку, и на уступе выше — раньше центр убегающего ближе 2 блоков
// уходил ниже конуса взгляда, и водящие стояли рядом и "не попадали".
{
    const reaches = (dx, dy, dz) => {
        const bot = fakeBot(new Set(), 0);
        bot.entity.height = 1.8;
        const runner = { id: 42, type: 'player', name: 'player', height: 1.8,
                         position: new Vec3(0.5 + dx, 64 + dy, 0.5 - dz) };
        bot.entities = { 42: runner };
        const executor = new ActionExecutor(bot, config);
        executor.setTaggable([42]);
        return executor.findEntityInCrosshair() === runner;
    };
    assert.ok(reaches(0, 0, 0.8), 'вплотную впереди');
    assert.ok(reaches(0, 0, 1.5), 'в полутора блоках');
    assert.ok(reaches(0.8, 0, 2), 'чуть сбоку (~22°)');
    assert.ok(reaches(0, 1, 1.5), 'на уступе в блок выше');
    assert.ok(reaches(0, 0, 3.2), 'на пределе удара');
    assert.ok(!reaches(0, 0, 4), 'дальше удара — нет');
    assert.ok(!reaches(2, 0, 1), 'сильно сбоку (~63°) — нет');
    assert.ok(!reaches(0, 0, -1.5), 'за спиной — нет');
}
console.log('Прицел удара в порядке.');

// Сквозь стену не бьёт (автор, 2026-09-27: "умеют сквозь блоки бить"):
// убегающий в двух блоках впереди, между ними — стена в два блока. Уступ в
// блок не мешает (голова и центр видны поверх), вода и трава — тоже.
(async () => {
    const reachesThrough = async (blocks) => {
        const world = new World(() => new Chunk());
        for (const [x, y, z, name] of blocks) {
            await world.setBlockStateId(new Vec3(x, y, z), registry.blocksByName[name].defaultState);
        }
        const bot = fakeBot(new Set(), 0);
        bot.world = world.sync;
        bot.entity.height = 1.8;
        const runner = { id: 42, type: 'player', name: 'player', height: 1.8, position: new Vec3(0.5, 64, -1.5) };
        bot.entities = { 42: runner };
        const executor = new ActionExecutor(bot, config);
        executor.setTaggable([42]);
        return executor.findEntityInCrosshair() === runner;
    };
    const grass = 'short_grass' in registry.blocksByName ? 'short_grass' : 'grass';
    assert.ok(await reachesThrough([]), 'ничего между — бьёт');
    assert.ok(!await reachesThrough([[0, 64, -1, 'stone'], [0, 65, -1, 'stone']]), 'стена в два блока — не бьёт');
    assert.ok(!await reachesThrough([[0, 64, -1, 'glass'], [0, 65, -1, 'glass']]), 'стекло — тоже стена');
    assert.ok(await reachesThrough([[0, 64, -1, 'stone']]), 'уступ в блок — бьёт поверх');
    assert.ok(await reachesThrough([[0, 64, -1, 'water'], [0, 65, -1, 'water']]), 'вода не мешает');
    assert.ok(await reachesThrough([[0, 64, -1, grass]]), 'трава не мешает');
    console.log('Удар сквозь стены не проходит.');
})().catch((err) => {
    console.error(err);
    process.exit(1);
});

// Заряд удара — как в игре: сразу после удара почти ноль, через 0.25 с
// (кулак, скорость атаки 4) — полный; урон от моего удара уходит с зарядом.
{
    const bot = fakeBot(new Set(), 0);
    const executor = new ActionExecutor(bot, config);
    assert.strictEqual(executor.attackCharge(), 1, 'давно не бил — полный заряд');
    executor.lastAttackAt = Date.now();
    assert.ok(executor.attackCharge() < 0.15, 'только что ударил — заряда нет');
    executor.lastAttackAt = Date.now() - 100;
    assert.ok(Math.abs(executor.attackCharge() - 0.5) < 0.05, 'через 2 тика — половина');
    executor.lastAttackAt = Date.now() - 300;
    assert.strictEqual(executor.attackCharge(), 1, 'через 6 тиков — полный');
    bot.entity.attributes = { 'minecraft:attack_speed': { value: 1.6, modifiers: [] } }; // меч
    assert.ok(executor.attackCharge() < 0.6, 'у меча перезарядка дольше');
    executor.lastAttackCharge = 0.8;
    bot.emit('entityHurt', { id: 42 }, bot.entity);
    assert.deepStrictEqual(executor.takeDamageDealt(), [{ id: 42, charge: 0.8, crit: false }], 'мой удар нанёс урон — с зарядом');
    assert.deepStrictEqual(executor.takeDamageDealt(), []);
}
// Крит: полная сила, в падении после прыжка, не на бегу.
{
    const bot = fakeBot(new Set(), 0);
    bot.entity.height = 1.8;
    const runner = { id: 42, type: 'player', name: 'player', height: 1.8, position: new Vec3(0.5, 64, -1) };
    bot.entities = { 42: runner };
    bot.attack = () => {};
    bot.lookAt = () => {};
    bot.getControlState = (control) => Boolean(bot.controls[control]);
    const executor = new ActionExecutor(bot, config);
    executor.setTaggable([42]);
    executor.handsBusy = true; // копать не полезет
    const hit = (onGround, vy, sprint) => {
        bot.entity.onGround = onGround;
        bot.entity.velocity = new Vec3(0, vy, 0);
        executor.lastAttackAt = 0; // заряд полный
        // Бег — действие ног того же тика (executeAll сначала отпускает клавиши).
        executor.executeAll({ legs: sprint ? 'sprint_forward' : 'idle', hands: 'attack_center' });
        return executor.lastAttackCrit;
    };
    assert.strictEqual(hit(false, -0.2, false), true, 'в падении — крит');
    assert.strictEqual(hit(true, 0, false), false, 'на земле — нет');
    assert.strictEqual(hit(false, 0.3, false), false, 'на взлёте — нет');
    assert.strictEqual(hit(false, -0.2, true), false, 'на бегу — нет (как в игре)');
}
console.log('Заряд удара и крит в порядке.');
