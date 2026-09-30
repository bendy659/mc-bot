// Эталон для сверки симуляции (py/sim/check_parity.py): строит тот же мир
// в prismarine-world и считает зрение (buildVisionGrid, centerBlockInfo) и
// маршруты (RoutePlanner) настоящим кодом Node. Вход — JSON в stdin:
//   { fills: [[x1, y1, z1, x2, y2, z2, name]...], poses: [{x, y, z, yaw, pitch}...],
//     routes: [{ start: [x, y, z], target: [x, y, z] }...] }
// Выход — JSON в stdout: { vision: [[байты r,g,b,d,t...]...], centers: [...], grounds: [...], places: [...],
//   routes: [...] } (places — canPlaceFront бота с землёй в инвентаре).

const { EventEmitter } = require('events');

const { Vec3 } = require('vec3');
const { buildWorld } = require('./sim_world');
const { buildVisionGrid, centerBlockInfo, groundProbe } = require('./vision');
const { RoutePlanner } = require('./route');
const { ActionExecutor } = require('./actions');
const { loadConfig } = require('./config');

function toByte(value) {
    return Math.round(Math.min(Math.max(value, 0), 1) * 255); // как packVision в js/state.js
}

async function main() {
    let input = '';
    for await (const chunk of process.stdin) input += chunk;
    const request = JSON.parse(input);
    const { poses, routes } = request;
    const config = loadConfig();
    const { registry, world } = await buildWorld(request);
    const sync = world.sync;

    const vision = [];
    const centers = [];
    const grounds = [];
    const places = [];
    for (const pose of poses) {
        const entity = {
            position: new Vec3(pose.x, pose.y, pose.z), height: 1.8, width: 0.6, eyeHeight: 1.62, yaw: pose.yaw, pitch: pose.pitch,
        };
        const cells = buildVisionGrid(sync, entity, config);
        const bytes = [];
        for (const c of cells) bytes.push(toByte(c.r), toByte(c.g), toByte(c.b), toByte(c.d), c.t);
        vision.push(bytes);
        centers.push(centerBlockInfo(sync, entity, config));
        grounds.push(groundProbe(sync, entity));
        // Бот с землёй в инвентаре, один в мире — как в симуляции.
        const bot = Object.assign(new EventEmitter(), {
            entity, registry, world: sync, entities: {}, health: 20,
            inventory: { items: () => [{ name: 'dirt', count: 64 }] },
            blockAt: (p) => sync.getBlock(p),
            setControlState: () => {}, clearControlStates: () => {},
        });
        places.push(new ActionExecutor(bot, config).canPlaceFront());
    }

    const routeResults = [];
    for (const { start, target } of routes) {
        const planner = new RoutePlanner(sync, registry, config.route);
        routeResults.push(planner.update(new Vec3(...start), { x: target[0], y: target[1], z: target[2] }));
    }
    process.stdout.write(JSON.stringify({ vision, centers, grounds, places, routes: routeResults }));
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});
