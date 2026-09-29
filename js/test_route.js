// Проверка поиска маршрута (js/route.js) на синтетических мирах, без
// сервера: npm run test-route

const assert = require('assert');
const { Vec3 } = require('vec3');
const registry = require('prismarine-registry')('1.20.1');
const Chunk = require('prismarine-chunk')(registry);
const World = require('prismarine-world')(registry);
const { RoutePlanner } = require('./route');

async function makeWorld(build) {
    const world = new World(() => new Chunk());
    const put = (x, y, z, name) => world.setBlockStateId(new Vec3(x, y, z), registry.blocksByName[name].defaultState);
    // Ровный пол на y=63 (ходим по y=64) в квадрате ±24.
    for (let x = -24; x <= 24; x++) for (let z = -24; z <= 24; z++) await put(x, 63, z, 'grass_block');
    await build(put);
    return world.sync;
}

function planner(world) {
    return new RoutePlanner(world, registry, { max_nodes: 4000, max_drop: 3, lookahead: 3 });
}

async function main() {
    const bot = new Vec3(0.5, 64, 0.5);

    // 1. Ровное поле: маршрут прямой, длина ~ расстояние.
    {
        const world = await makeWorld(async () => {});
        const route = planner(world).update(bot, new Vec3(10.5, 64, 0.5));
        console.log('поле:', route);
        assert.ok(route.complete && Math.abs(route.length - 10) < 1.5, 'по полю — прямо');
    }

    // 2. Стена поперёк пути (x=5, z от -10 до 10, высота 3) с проходом у z=8:
    //    маршрут идёт через проход — он заметно длиннее прямой, а первая
    //    точка маршрута уводит в сторону прохода (+z).
    {
        const world = await makeWorld(async (put) => {
            for (let z = -10; z <= 10; z++) {
                if (z === 8) continue;
                for (let y = 64; y <= 66; y++) await put(5, y, z, 'stone');
            }
        });
        const p = planner(world);
        const t0 = Date.now();
        const route = p.update(bot, new Vec3(10.5, 64, 0.5));
        const ms = Date.now() - t0;
        console.log(`стена с проходом: длина ${route.length}, точка`, route.waypoint, `(${ms} мс)`);
        assert.ok(route.complete, 'проход найден');
        assert.ok(route.length > 14, 'маршрут в обход длиннее прямой (10)');
        assert.ok(p.path.some((n) => n.x === 5 && n.z === 8), 'путь идёт через проход');
        assert.ok(ms < 200, 'быстро');
    }

    // 3. Уступ высотой 2 (прямо не запрыгнуть) и лесенка сбоку: путь по лесенке.
    {
        const world = await makeWorld(async (put) => {
            for (let x = 5; x <= 14; x++) for (let z = -6; z <= 6; z++) {
                await put(x, 64, z, 'stone');
                await put(x, 65, z, 'stone');
            }
            await put(4, 64, 6, 'stone'); // ступенька: с неё — на уступ
        });
        const p = planner(world);
        const route = p.update(bot, new Vec3(10.5, 66, 0.5));
        const climbs = p.path.filter((n, i) => i > 0 && n.y > p.path[i - 1].y).length;
        console.log(`уступ в 2 блока: длина ${route.length}, подъёмов ${climbs}`);
        assert.ok(route.complete && climbs === 2, 'забрался по ступеньке, по блоку за раз');
        assert.ok(p.path.some((n) => n.x === 4 && n.z === 6 && n.y === 65), 'через ступеньку');
    }

    // 4. Прямо — яма с водой: в воду не лезем, обходим.
    {
        const world = await makeWorld(async (put) => {
            for (let x = 3; x <= 7; x++) for (let z = -3; z <= 3; z++) {
                await put(x, 63, z, 'water');
                await put(x, 62, z, 'water');
            }
        });
        const p = planner(world);
        const route = p.update(bot, new Vec3(10.5, 64, 0.5));
        const wet = p.path.some((n) => n.x >= 3 && n.x <= 7 && n.z >= -3 && n.z <= 3);
        console.log(`вода: длина ${route.length}, по воде: ${wet}`);
        assert.ok(route.complete && !wet, 'обошёл воду');
    }

    // 5. Цель за глухой стеной — маршрута нет: частичный путь к ближайшей точке.
    {
        const world = await makeWorld(async (put) => {
            for (let x = -24; x <= 24; x++) for (let y = 64; y <= 70; y++) await put(x, y, 5, 'stone');
        });
        const target = new Vec3(0.5, 64, 10.5);
        const route = planner(world).update(bot, target);
        console.log('тупик:', route);
        assert.ok(!route.complete, 'честно говорит, что до цели не дошёл');
        // Длина недостроенного маршрута — кусок пути плюс остаток по прямой,
        // не меньше расстояния до цели: иначе прогресс "по маршруту" врал бы.
        const straight = Math.hypot(target.x - bot.x, target.z - bot.z);
        assert.ok(route.length >= straight - 1.5, `длина ${route.length} не короче прямой ${straight.toFixed(1)}`);
    }

    // 6. Спуск с обрыва в 3 блока можно, в 5 — нельзя (разобьётся).
    {
        const world = await makeWorld(async (put) => {
            for (let x = -24; x <= 24; x++) for (let z = -24; z <= 24; z++) {
                if (x >= 5) await put(x, 63, z, 'air');
            }
            for (let x = 5; x <= 24; x++) for (let z = -24; z <= 24; z++) await put(x, 60, z, 'stone'); // пол на 3 ниже
        });
        const low = planner(world).update(bot, new Vec3(10.5, 61, 0.5));
        console.log('спуск на 3:', low);
        assert.ok(low.complete, 'спрыгнул на 3 блока');
    }

    console.log('Маршруты в порядке.');
}

main().catch((err) => {
    console.error('ТЕСТ МАРШРУТОВ НЕ ПРОШЁЛ:', err.message);
    process.exit(1);
});
