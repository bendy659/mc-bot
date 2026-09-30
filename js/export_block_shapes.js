// Формы и свойства блоков для симуляции (py/sim/world.py) — ровно те, что
// видит бот mineflayer: коробки коллизии (block.shapes, по ним бьёт луч
// зрения и считается физика), проходной ли блок для зрения (isPassThrough),
// boundingBox (маршруты: 'block' | 'empty'), прочность и чем копается.
//
// Запуск: node js/export_block_shapes.js <состояния.json> <выход.json>
// состояния.json — список строк "minecraft:oak_stairs[facing=east,half=bottom,...]"
// (так их пишет py/bedwars_maps.py). Версия — та, что у сервера (config.json
// bot.version или по умолчанию 26.1).

const fs = require('fs');
const path = require('path');
const { isPassThrough } = require('./vision');

const [input, output] = process.argv.slice(2);
if (!input || !output) {
    console.error('node js/export_block_shapes.js <состояния.json> <выход.json>');
    process.exit(1);
}

const config = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'config.json'), 'utf8'));
const version = config.bot.version || '26.1';
const mcData = require('minecraft-data')(version);
const Block = require('prismarine-block')(mcData.version.minecraftVersion);

function parseState(state) {
    const match = /^(?:minecraft:)?([a-z0-9_]+)(?:\[(.*)\])?$/.exec(state);
    if (!match) throw new Error('не понял состояние ' + state);
    const properties = {};
    for (const pair of (match[2] || '').split(',').filter(Boolean)) {
        const [key, value] = pair.split('=');
        properties[key] = value;
    }
    return { name: match[1], properties };
}

const out = {};
for (const state of JSON.parse(fs.readFileSync(input, 'utf8'))) {
    const { name, properties } = parseState(state);
    const block = Block.fromProperties(name, properties, 0);
    const data = mcData.blocksByName[name];
    out[state] = {
        name: block.name,
        shapes: block.shapes || [],
        boundingBox: block.boundingBox,
        passThrough: isPassThrough(block.name),
        hardness: data ? data.hardness : null,
        // Добывается рукой, если у блока нет списка нужных инструментов.
        handHarvest: !data || !data.harvestTools,
    };
}
fs.writeFileSync(output, JSON.stringify(out, null, 1));
console.log(`Состояний: ${Object.keys(out).length} -> ${output}`);
