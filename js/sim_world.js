// Мир для эталонов сверки симуляции (js/sim_reference.js, sim_physics_reference.js):
// из заливок ({fills: [[x1, y1, z1, x2, y2, z2, name]...]}, арена и мосты —
// версия 1.20.1, как было) или из состояний блоков карты бедварса
// ({version: '26.1', states: [[x, y, z, "minecraft:oak_stairs[facing=east,...]"]...]}):
// там формы ступеней и плит той же версии, что у сервера.

const { Vec3 } = require('vec3');

function parseState(state) {
    const match = /^(?:minecraft:)?([a-z0-9_]+)(?:\[(.*)\])?$/.exec(state);
    const properties = {};
    for (const pair of (match[2] || '').split(',').filter(Boolean)) {
        const [key, value] = pair.split('=');
        properties[key] = value;
    }
    return { name: match[1], properties };
}

async function buildWorld({ fills = [], states = [], version = '1.20.1' }) {
    const registry = require('prismarine-registry')(version);
    const Chunk = require('prismarine-chunk')(registry);
    const World = require('prismarine-world')(registry);
    const Block = require('prismarine-block')(registry);
    const world = new World(() => new Chunk());
    for (const [x1, y1, z1, x2, y2, z2, name] of fills) {
        if (name === 'air') continue; // новый мир и так пуст
        const stateId = registry.blocksByName[name].defaultState;
        for (let x = Math.min(x1, x2); x <= Math.max(x1, x2); x++) {
            for (let y = Math.min(y1, y2); y <= Math.max(y1, y2); y++) {
                for (let z = Math.min(z1, z2); z <= Math.max(z1, z2); z++) {
                    await world.setBlockStateId(new Vec3(x, y, z), stateId);
                }
            }
        }
    }
    // Все чанки в пределах карты — есть (как на сервере, где карта загружена
    // целиком): в пустом чанке getBlock иначе вернёт null, и prismarine сочтёт
    // стоящего на краю бота летящим (blockUnder === null).
    if (states.length) {
        let [x0, x1, z0, z1] = [Infinity, -Infinity, Infinity, -Infinity];
        for (const [x, , z] of states) {
            x0 = Math.min(x0, x);
            x1 = Math.max(x1, x);
            z0 = Math.min(z0, z);
            z1 = Math.max(z1, z);
        }
        for (let cx = Math.floor(x0 / 16) - 1; cx <= Math.floor(x1 / 16) + 1; cx++) {
            for (let cz = Math.floor(z0 / 16) - 1; cz <= Math.floor(z1 / 16) + 1; cz++) {
                await world.setBlockStateId(new Vec3(cx * 16, 0, cz * 16), 0);
            }
        }
    }
    const ids = new Map();
    for (const [x, y, z, state] of states) {
        if (!ids.has(state)) {
            const { name, properties } = parseState(state);
            ids.set(state, Block.fromProperties(name, properties, 0).stateId);
        }
        await world.setBlockStateId(new Vec3(x, y, z), ids.get(state));
    }
    return { registry, world };
}

module.exports = { buildWorld };
