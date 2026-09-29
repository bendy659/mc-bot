// Разбор инвентаря — общий для действий рук (js/actions.js: что ставить,
// что есть, что надевать) и для наблюдения сети (сводка в state.inventory:
// без неё сеть не знает, есть ли ей чем строить и что есть).

const ARMOR = /helmet|chestplate|leggings|boots/;
const TOOL = /pickaxe|_axe$|shovel|hoe|shears/;
const WEAPON = /sword|bow|crossbow|trident|mace/;
const ARMOR_SLOTS = ['head', 'torso', 'legs', 'feet'];

// Блок, которым можно строить: предмет-блок с полной формой (земля, камень,
// доски...). Факелы, цветы, кнопки — не для столба под себя.
function isPlaceableBlock(bot, item) {
    const block = bot.registry.blocksByName[item.name];
    return !!block && block.boundingBox === 'block';
}

function isFood(bot, item) {
    return !!bot.registry.foodsByName?.[item.name];
}

function isArmor(item) {
    return ARMOR.test(item.name);
}

function heldKind(bot, item) {
    if (!item) return 'none';
    if (isPlaceableBlock(bot, item)) return 'block';
    if (isFood(bot, item)) return 'food';
    if (TOOL.test(item.name) || WEAPON.test(item.name)) return 'tool';
    return 'other';
}

// Сводка для сети: сколько блоков, еды, брони в инвентаре, сколько брони
// надето и что в руке.
function summarizeInventory(bot) {
    const items = bot.inventory.items();
    const count = (predicate) => items.filter(predicate).reduce((sum, item) => sum + item.count, 0);
    const worn = ARMOR_SLOTS.filter((slot) => bot.inventory.slots[bot.getEquipmentDestSlot(slot)]).length;
    return {
        blocks: count((item) => isPlaceableBlock(bot, item)),
        food: count((item) => isFood(bot, item)),
        armor_items: count(isArmor),
        armor_worn: worn,
        held: heldKind(bot, bot.heldItem),
    };
}

module.exports = { isPlaceableBlock, isFood, isArmor, summarizeInventory, ARMOR_SLOTS };
