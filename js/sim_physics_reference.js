// Эталон физики для сверки симуляции (py/sim/check_parity.py): настоящий
// prismarine-physics (им двигается бот mineflayer) прогоняет заданные
// "клавиши" по тикам на том же мире. Вход — JSON в stdin:
//   { fills: [...как в sim_reference.js], runs: [{ start: [x, y, z], yaw, ticks: [{forward, back, left,
//     right, jump, sprint}...] }...] }
// Выход — позиции ног после каждого тика: { runs: [[[x, y, z]...]...], velocities: [[[vx, vy, vz, onGround]...]...] }.

const { Vec3 } = require('vec3');
const { buildWorld } = require('./sim_world');
const { Physics, PlayerState } = require('prismarine-physics');

async function main() {
    let input = '';
    for await (const chunk of process.stdin) input += chunk;
    const request = JSON.parse(input);
    const { runs } = request;
    const { registry, world } = await buildWorld(request);
    const physics = Physics(registry, world.sync);

    const out = [];
    const velocities = [];
    for (const run of runs) {
        const bot = {
            version: registry.version.minecraftVersion,
            entity: {
                position: new Vec3(...run.start), velocity: new Vec3(0, 0, 0), onGround: true,
                isInWater: false, isInLava: false, isInWeb: false, isCollidedHorizontally: false,
                isCollidedVertically: false, elytraFlying: false, attributes: {}, yaw: run.yaw, pitch: 0, effects: {},
            },
            jumpTicks: 0, jumpQueued: false, fireworkRocketDuration: 0, inventory: { slots: [] },
        };
        const track = [];
        const speeds = []; // скорость и onGround после тика — для отладки расхождений
        for (const keys of run.ticks) {
            const control = { forward: false, back: false, left: false, right: false, jump: false, sprint: false, sneak: false, ...keys };
            const state = new PlayerState(bot, control);
            physics.simulatePlayer(state, world.sync);
            state.apply(bot);
            track.push([bot.entity.position.x, bot.entity.position.y, bot.entity.position.z]);
            speeds.push([bot.entity.velocity.x, bot.entity.velocity.y, bot.entity.velocity.z, bot.entity.onGround]);
        }
        out.push(track);
        velocities.push(speeds);
    }
    process.stdout.write(JSON.stringify({ runs: out, velocities }));
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});
