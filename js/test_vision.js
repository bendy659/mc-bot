// Проверка зрения и геометрии на синтетическом мире, без сервера:
// npm run test-vision
//
// Зачем: тут уже дважды были тихие ошибки, из-за которых бот учился на
// мусоре. (1) raycast prismarine-world возвращает сам блок, а не
// { block, distance } — все лучи были "небом". (2) Система углов была не
// та, что у mineflayer: мир для бота был зеркальным по Z, "вверх" и "вниз"
// перепутаны. Тест проверяет всё против эталона — вектора взгляда из
// самой физики prismarine (см. комментарий в prismarine-physics/index.js:
// x = -sin(yaw)cos(pitch), y = sin(pitch), z = -cos(yaw)cos(pitch)).

const assert = require('assert');
const { Vec3 } = require('vec3');
const registry = require('prismarine-registry')('1.20.1');
const Chunk = require('prismarine-chunk')(registry);
const World = require('prismarine-world')(registry);
const { buildVisionGrid, centerBlockInfo, lookDirection, rawRaycast, fastRaycast } = require('./vision');
const { buildEntitiesList } = require('./entities');
const { StateBuilder } = require('./state');
const { loadConfig } = require('./config');

async function main() {
    const world = new World(() => new Chunk());
    const put = (x, y, z, name) => world.setBlockStateId(new Vec3(x, y, z), registry.blocksByName[name].defaultState);

    // Бот стоит в (0.5, 64, 0.5), yaw=0 — в mineflayer это взгляд на СЕВЕР (-Z).
    // Впереди (z=-3, грань на z=-2) — стена камня; справа (для взгляда на
    // север правая рука — восток, +X; грань на x=3) — стена брёвен; прямо
    // перед носом — высокая трава, её луч пропускает.
    for (let y = 60; y < 72; y++) {
        for (let d = -3; d <= 3; d++) {
            await put(d, y, -3, 'stone');
            await put(3, y, d, 'oak_log');
        }
    }
    // Земля под ногами — по ней проверяем, что нижний ряд сетки смотрит вниз.
    for (let x = -8; x <= 2; x++) for (let z = -2; z <= 8; z++) await put(x, 63, z, 'dirt');
    await put(0, 64, -1, 'short_grass' in registry.blocksByName ? 'short_grass' : 'grass');
    await put(0, 65, -1, 'tall_grass');

    const sync = world.sync;
    const config = loadConfig();
    const [resX, resY] = config.vision.resolution;
    const maxDistance = config.vision.distance * 16;
    const entity = { position: new Vec3(0.5, 64, 0.5), height: 1.8, yaw: 0, pitch: 0 };

    // Вектор взгляда — как в физике prismarine.
    for (const [yaw, pitch] of [[0, 0], [1, 0.3], [-2.5, -0.7]]) {
        const d = lookDirection({ yaw, pitch });
        const ref = new Vec3(-Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch), -Math.cos(yaw) * Math.cos(pitch));
        assert.ok(d.minus(ref).norm() < 1e-9, `lookDirection не совпадает с физикой при yaw=${yaw}`);
    }

    const cells = buildVisionGrid(sync, entity, config);
    const midRow = Math.floor(resY / 2);
    const at = (row, col) => cells[row * resX + col];

    // Колонка 0 в circular-режиме — прямо вперёд: камень (класс 2) в ~2.5
    // блока, трава перед носом не мешает.
    const ahead = at(midRow, 0);
    console.log('впереди:', ahead);
    assert.strictEqual(ahead.t, 2, 'впереди должен быть камень');
    assert.ok(Math.abs(ahead.d * maxDistance - 2.5) < 0.3, `дистанция до камня ~2.5, а не ${ahead.d * maxDistance}`);

    // Четверть круга по часовой — правая сторона: брёвна (класс 1).
    const right = at(midRow, resX / 4);
    console.log('справа:', right);
    assert.strictEqual(right.t, 1, 'справа должны быть брёвна');
    assert.strictEqual(at(midRow, (3 * resX) / 4).t, 0, 'слева пусто');

    // Сзади пусто — небо на максимальной дальности.
    const behind = at(midRow, resX / 2);
    assert.strictEqual(behind.t, 0, 'сзади пусто');
    assert.strictEqual(behind.d, 1.0);

    // Ряд 0 — верх, последний — низ: сзади вверху небо, сзади внизу земля.
    assert.strictEqual(at(0, resX / 2).t, 0, 'верхний ряд сзади — небо');
    assert.strictEqual(at(resY - 1, resX / 2).t, 3, 'нижний ряд сзади — земля');

    // circular-зрение не зависит от наклона головы.
    const lookingUp = buildVisionGrid(sync, { ...entity, pitch: 1.2 }, config);
    assert.deepStrictEqual(lookingUp, cells, 'circular-сетка не должна зависеть от pitch');

    // Прицел: прямо — камень; вверх (pitch > 0) — небо; вниз — земля.
    const center = centerBlockInfo(sync, entity, config);
    console.log('в прицеле:', center);
    assert.strictEqual(center.name, 'stone');
    assert.ok(center.distance > 2 && center.distance < 3);
    assert.strictEqual(centerBlockInfo(sync, { ...entity, pitch: 1.3 }, config), null, 'вверх — небо');
    assert.strictEqual(centerBlockInfo(sync, { ...entity, yaw: Math.PI, pitch: -1.2 }, config).name, 'dirt', 'вниз — земля');

    // Сущности в локальной системе: впереди — forward > 0; справа (восток) — right > 0.
    const other = (x, z) => ({ isValid: true, position: new Vec3(x, 64, z), type: 'mob', name: 'cow', height: 1.4 });
    const [aheadEntity] = buildEntitiesList({ 1: other(0.5, -4.5) }, entity, config);
    const [rightEntity] = buildEntitiesList({ 1: other(5.5, 0.5) }, entity, config);
    console.log('сущность впереди:', aheadEntity.forward, aheadEntity.right, '| справа:', rightEntity.forward, rightEntity.right);
    assert.ok(aheadEntity.forward > 0.2 && Math.abs(aheadEntity.right) < 1e-6);
    assert.ok(rightEntity.right > 0.2 && Math.abs(rightEntity.forward) < 1e-6);

    // Классы сущностей — по категориям minecraft-data новых версий (26.1) и
    // старым 'mob'/'object'. Живые — первыми в списке: 9 выпавших предметов
    // вплотную не вытесняют игрока в 10 блоках.
    const placed = (x, extra) => ({ isValid: true, position: new Vec3(x, 64, 0.5), height: 1.8, ...extra });
    const kinds = {
        1: placed(3, { type: 'hostile', name: 'zombie' }),
        2: placed(4, { type: 'animal', name: 'cow' }),
        3: placed(5, { type: 'player', username: 'Someone' }),
        4: placed(6, { type: 'mob', name: 'slime' }),
        5: placed(7, { type: 'mob', name: 'iron_golem' }),
        6: placed(8, { type: 'projectile', name: 'arrow' }),
        7: placed(9, { type: 'other', name: 'experience_orb' }),
        8: placed(10, { type: 'object', name: 'item' }), // старые версии
    };
    const typeOf = Object.fromEntries(buildEntitiesList(kinds, entity, config).map((e) => [e.name, e.type]));
    console.log('классы сущностей:', typeOf);
    assert.deepStrictEqual(typeOf, { zombie: 0, cow: 1, Someone: 2, slime: 0, iron_golem: 1, arrow: 3, experience_orb: 3, item: 3 });
    const crowd = { 100: placed(10, { type: 'player', username: 'Author' }) };
    for (let i = 0; i < 9; i++) crowd[i] = placed(1 + i * 0.1, { type: 'other', name: 'item' });
    const listed = buildEntitiesList(crowd, entity, config);
    assert.strictEqual(listed[0].name, 'Author', 'игрок — раньше россыпи предметов');

    // Дельты движения: шаг по взгляду — move_forward > 0.
    const builder = new StateBuilder(config);
    const self = (pos) => ({ position: pos, height: 1.8, yaw: 1.0, pitch: 0, onGround: true });
    const hearing = { tick: () => [] };
    builder.build(sync, self(new Vec3(0.5, 64, 0.5)), hearing, 0);
    const step = lookDirection({ yaw: 1.0, pitch: 0 }).scaled(0.6);
    const moved = builder.build(sync, self(new Vec3(0.5, 64, 0.5).plus(step)), hearing, 1).self;
    console.log('шаг вперёд: move_forward', moved.move_forward, 'move_right', moved.move_right);
    assert.ok(Math.abs(moved.move_forward - 0.6) < 0.01 && Math.abs(moved.move_right) < 0.01);

    // Быстрый луч сетки зрения (fastRaycast) обязан совпадать с эталонным
    // world.raycast (rawRaycast) бит в бит: разные формы блоков (полублок,
    // забор, стекло), проходимые (трава), жидкость, пустота.
    const mixed = [['stone_slab', 66], ['oak_fence', 66], ['glass', 65], ['water', 64], ['tall_grass', 64],
        ['stone', 64], ['oak_log', 65], ['barrier', 66], ['cobblestone_wall', 64], ['torch', 64]];
    for (let i = 0; i < 400; i++) {
        const [name, y] = mixed[i % mixed.length];
        if (!(name in registry.blocksByName)) continue;
        await put(-6 + (i * 7) % 13, y + ((i * 3) % 3), 2 + (i * 5) % 11, name);
    }
    let rays = 0;
    let rng = 12345;
    const random = () => ((rng = (rng * 1103515245 + 12345) % 2147483648) / 2147483648);
    for (let i = 0; i < 3000; i++) {
        const origin = new Vec3(-5 + random() * 10, 64 + random() * 4, -1 + random() * 12);
        const args = [origin, random() * 6.3 - 3.15, random() * 3 - 1.5, 0, 0, maxDistance];
        const slow = rawRaycast(sync, ...args);
        const fast = fastRaycast(sync, ...args);
        const same = (!slow && !fast) || (slow && fast && slow.block.name === fast.block.name && slow.distance === fast.distance);
        assert.ok(same, `быстрый луч разошёлся с эталоном: ${JSON.stringify({ slow: slow && [slow.block.name, slow.distance], fast: fast && [fast.block.name, fast.distance], args })}`);
        if (slow) rays++;
    }
    console.log(`быстрый луч = эталон на 3000 лучах (попали в блок: ${rays})`);

    console.log('Зрение и геометрия в порядке.');
}

main().catch((err) => {
    console.error('ТЕСТ ЗРЕНИЯ НЕ ПРОШЁЛ:', err.message);
    process.exit(1);
});
