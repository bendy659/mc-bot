// Эталон физики для сверки симуляции (py/sim/check_parity.py): настоящий
// prismarine-physics (им двигается бот mineflayer) прогоняет заданные
// "клавиши" по тикам на том же мире. Вход — JSON в stdin:
//   { fills: [...как в sim_reference.js], runs: [{ start: [x, y, z], yaw, ticks: [{forward, back, left,
//     right, jump, sprint}...] }...] }
// Выход — позиции ног после каждого тика: { runs: [[[x, y, z]...]...] }.

const { Vec3 } = require('vec3');
const registry = require('prismarine-registry')('1.20.1');
const Chunk = require('prismarine-chunk')(registry);
const World = require('prismarine-world')(registry);
const { Physics, PlayerState } = require('prismarine-physics');

async function main() {
    let input = '';
    for await (const chunk of process.stdin) input += chunk;
    const { fills, runs } = JSON.parse(input);

    const world = new World(() => new Chunk());
    for (const [x1, y1, z1, x2, y2, z2, name] of fills) {
        if (name === 'air') continue;
        const stateId = registry.blocksByName[name].defaultState;
        for (let x = Math.min(x1, x2); x <= Math.max(x1, x2); x++) {
            for (let y = Math.min(y1, y2); y <= Math.max(y1, y2); y++) {
                for (let z = Math.min(z1, z2); z <= Math.max(z1, z2); z++) {
                    await world.setBlockStateId(new Vec3(x, y, z), stateId);
                }
            }
        }
    }
    const physics = Physics(registry, world.sync);

    const out = [];
    for (const run of runs) {
        const bot = {
            version: '1.20.1',
            entity: {
                position: new Vec3(...run.start), velocity: new Vec3(0, 0, 0), onGround: true,
                isInWater: false, isInLava: false, isInWeb: false, isCollidedHorizontally: false,
                isCollidedVertically: false, elytraFlying: false, attributes: {}, yaw: run.yaw, pitch: 0, effects: {},
            },
            jumpTicks: 0, jumpQueued: false, fireworkRocketDuration: 0, inventory: { slots: [] },
        };
        const track = [];
        for (const keys of run.ticks) {
            const control = { forward: false, back: false, left: false, right: false, jump: false, sprint: false, sneak: false, ...keys };
            const state = new PlayerState(bot, control);
            physics.simulatePlayer(state, world.sync);
            state.apply(bot);
            track.push([bot.entity.position.x, bot.entity.position.y, bot.entity.position.z]);
        }
        out.push(track);
    }
    process.stdout.write(JSON.stringify({ runs: out }));
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});
