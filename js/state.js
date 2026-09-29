// Сборка JSON-состояния одного тика — единственное, что видит Python
// о мире. Держать компактным: это уходит по сети каждые tick_rate_ms.
//
// StateBuilder — класс, а не голая функция, потому что наблюдение содержит
// дельты относительно прошлого тика (реальный сдвиг, поворот), и их надо
// где-то хранить между тиками. У бота и у бота-обсёрвера свой экземпляр.

const { buildVisionGrid, centerBlockInfo, groundProbe } = require('./vision');
const { buildEntitiesList } = require('./entities');

class StateBuilder {
    // packedVision — зрение байтами (живые боты), иначе списком (запись игры).
    constructor(config, { packedVision = false } = {}) {
        this.packedVision = packedVision;
        this.config = config;
        this.prev = null; // { x, y, z, yaw, pitch } прошлого тика
    }

    // world — bot.world (raycast из любой точки), entity — сущность, чьими
    // глазами строится состояние (у бота — bot.entity, у обсёрвера — игрок).
    //
    // sourceEntities — словарь сущностей мира (bot.entities) для "зрения
    // на сущностей"; важно: сервер шлёт обсёрверу только сущностей вокруг
    // САМОГО обсёрвера, поэтому при записи геймплея список может быть
    // неполным относительно игрока — задокументированное ограничение.
    //
    // targetIsHuman / inventoryCount осмысленны только для живого бота
    // (bot.js их передаёт); обсёрвер (record.js) их не знает — цель и
    // инвентарь наблюдаемого игрока боту недоступны по сети, поэтому там
    // остаются дефолты (false / null), и записи геймплея это не портит:
    // behavior cloning имитирует только движение, не gathering/looking.
    //
    // worldBorder — WorldBorder из js/border.js (или null): walking-модуль
    // выбирает цели-точки внутри настоящей границы мира, а не по ручному
    // прямоугольнику из конфига. В сеть граница не идёт — только модулю.
    visionPayload(cells) {
        return this.packedVision ? { packed: packVision(cells) } : { cells: flattenVision(cells) };
    }

    build(
        world, entity, hearing, tickIndex,
        target = null, targetDescription = 'none', dead = false,
        targetIsHuman = false, inventoryCount = null, foodLevel = null,
        sourceEntities = {}, worldBorder = null
    ) {
        const pos = entity.position;

        const self = {
            x: round(pos.x), y: round(pos.y), z: round(pos.z),
            yaw: round(entity.yaw), pitch: round(entity.pitch),
            health: round(entity.health ?? 20),
            // Сытость — источник bot.food, а не entity (mineflayer не
            // зеркалит её на entity, в отличие от health). Актуальна не
            // только сама по себе: без неё нельзя спринтовать (см.
            // sprint_forward в actions.js) — ванильный порог ~6/20.
            food: round(foodLevel ?? 20),
            on_ground: !!entity.onGround,
        };

        // Дельты — "память о движении": сеть по одному кадру не отличает
        // "я иду вперёд" от "стожу, а мир крутится", а по дельтам отличает.
        // move_forward/move_right — сдвиг в локальной системе взгляда:
        // ходил ли вперёд, упёрся ли в стену (0 при walk_forward).
        if (this.prev) {
            const dx = pos.x - this.prev.x;
            const dy = pos.y - this.prev.y;
            const dz = pos.z - this.prev.z;
            // Взгляд проецируем на горизонталь: f = (-sin yaw, -cos yaw)
            // (конвенция mineflayer, см. js/vision.js), правый вектор —
            // поворот f на 90° по часовой: r = (-f.z, f.x).
            const fx = -Math.sin(entity.yaw);
            const fz = -Math.cos(entity.yaw);
            self.move_forward = round(dx * fx + dz * fz);
            self.move_right = round(dx * -fz + dz * fx);
            self.move_up = round(dy);
            // Кратчайшая угловая разница с учётом перехода через ±pi.
            self.dyaw = round(angleDiff(entity.yaw, this.prev.yaw));
            self.dpitch = round(angleDiff(entity.pitch, this.prev.pitch));
        } else {
            self.move_forward = 0; self.move_right = 0; self.move_up = 0;
            self.dyaw = 0; self.dpitch = 0;
        }
        this.prev = { x: pos.x, y: pos.y, z: pos.z, yaw: entity.yaw, pitch: entity.pitch };

        return {
            type: 'state',
            tick: tickIndex,
            vision: {
                // Плоский массив ячеек [r,g,b,d,t] * (resX*resY), порядок как
                // пиксели экрана: слева-направо, сверху-вниз. t — класс блока
                // (целое 0..block_classes-1). Размер фиксирован конфигом,
                // обе стороны читают один config.json.
                resolution: this.config.vision.resolution,
                ...this.visionPayload(buildVisionGrid(world, entity, this.config)),
            },
            // Тип блока строго по центру прицела — отдельным полем, не через
            // цвет: gathering-модулю нужно точно знать, что можно долбить
            // (attack_center), а не приближённо угадывать по палитре цветов.
            center_block: centerBlockInfo(world, entity, this.config),
            // Сколько пола до края впереди/справа/сзади/слева (js/vision.js:
            // groundProbe) — край под ногами сетка зрения не видит.
            ground: groundProbe(world, entity),
            // Ближайшие сущности в локальной системе взгляда (js/entities.js):
            // мобы/игроки/предметы как сигналы "впереди враг", "слева овца".
            entities: buildEntitiesList(sourceEntities, entity, this.config),
            hearing: this.config.hearing.enabled ? hearing.tick(pos, entity.yaw) : [],
            self,
            target: target ? { x: round(target.x), y: round(target.y), z: round(target.z) } : null,
            target_description: targetDescription,
            // true, если текущую цель держит человек через !setTarget —
            // обучающий модуль на это ориентируется, чтобы не пытаться
            // подсунуть свою цель поверх человеческой.
            target_is_human: targetIsHuman,
            // Суммарное количество предметов в инвентаре — gathering-модуль
            // использует рост этого числа как сигнал "что-то добыл", не
            // разбираясь, что именно (сломанный блок или добыча с моба).
            inventory_count: inventoryCount,
            world_border: worldBorder && worldBorder.known
                ? { center_x: round(worldBorder.centerX), center_z: round(worldBorder.centerZ), size: round(worldBorder.size) }
                : null,
            dead,
        };
    }
}

function angleDiff(a, b) {
    let d = a - b;
    while (d > Math.PI) d -= Math.PI * 2;
    while (d < -Math.PI) d += Math.PI * 2;
    return d;
}

// Живым ботам — байтами (packed: base64 от [r, g, b, d, t] * ячеек, r/g/b/d
// в 0..255): ~1300 чисел текстом Python разбирал ~0.4 мс на бота за тик, а
// с десятками ботов это и было пределом роя. Запись игры (js/record.js) —
// по-прежнему списком cells, читается человеком и старыми загрузчиками.
function packVision(cells) {
    const bytes = Buffer.alloc(cells.length * 5);
    for (let i = 0; i < cells.length; i++) {
        const c = cells[i];
        bytes[i * 5] = toByte(c.r);
        bytes[i * 5 + 1] = toByte(c.g);
        bytes[i * 5 + 2] = toByte(c.b);
        bytes[i * 5 + 3] = toByte(c.d);
        bytes[i * 5 + 4] = c.t;
    }
    return bytes.toString('base64');
}

function toByte(value) {
    return Math.round(Math.min(Math.max(value, 0), 1) * 255);
}

function flattenVision(cells) {
    const out = new Array(cells.length * 5);
    for (let i = 0; i < cells.length; i++) {
        const c = cells[i];
        out[i * 5] = round(c.r);
        out[i * 5 + 1] = round(c.g);
        out[i * 5 + 2] = round(c.b);
        out[i * 5 + 3] = round(c.d);
        out[i * 5 + 4] = c.t; // класс блока — целое, округлять нечего
    }
    return out;
}

function round(value) {
    return Math.round(value * 1000) / 1000;
}

module.exports = { StateBuilder };
