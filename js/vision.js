// "Псевдо-реальное" зрение бота: сетка raycast-лучей вокруг направления
// взгляда, аналогично тому, как рендерится экран, только очень низкого
// разрешения. Бот видит лишь то, что физически лежит на линии взгляда —
// без "рентгена" сквозь блоки (raycast останавливается на первом попадании).
//
// Выход — плоский массив ячеек, по одной на луч, в порядке слева-направо
// и сверху-вниз (как пиксели экрана). Каждая ячейка:
//   { r, g, b, d, t } — средний цвет блока (0..1), дистанция (0..1,
//                       1 = луч ни во что не попал в пределах дальности)
//                       и класс блока t (целое, см. BLOCK_CLASS_PATTERNS).
//
// Математика направлений — КОНВЕНЦИЯ MINEFLAYER (не протокола Minecraft!):
//   yaw — поворот вокруг вертикали: 0 = север (-Z), растёт ВЛЕВО (против
//   часовой, если смотреть сверху): pi/2 = запад (-X);
//   pitch — наклон: положительный = ВВЕРХ.
//   Направляющий вектор взгляда:
//     x = -sin(yaw) * cos(pitch)
//     y =  sin(pitch)
//     z = -cos(yaw) * cos(pitch)
//   Проверено по исходникам mineflayer: bot.lookAt считает
//   yaw = atan2(-dx, -dz), pitch = atan2(dy, горизонталь). (До 2026-09-24
//   тут была конвенция протокола — z = +cos(yaw), pitch вниз, — и весь мир
//   для бота был зеркальным по Z, а "вверх" и "вниз" перепутаны.)
//   Эта же конвенция — в js/entities.js, js/state.js, js/hearing.js,
//   py/state_encoder.py, py/training_modules/targets.py и walking.py.

const { Vec3 } = require('vec3');

// Направление взгляда сущности "как есть", без отклонений — нужно и сетке
// зрения (как частный случай hAngle=vAngle=0), и action-макросам вроде
// attack_center, которым нужно понять, что находится прямо по центру.
function lookDirection(entity) {
    const { yaw, pitch } = entity;
    const x = -Math.sin(yaw) * Math.cos(pitch);
    const y = Math.sin(pitch);
    const z = -Math.cos(yaw) * Math.cos(pitch);
    return new Vec3(x, y, z);
}

function rayDirection(yaw, pitch, hAngle, vAngle) {
    const dirYaw = yaw + hAngle;
    const dirPitch = pitch + vAngle;

    const x = -Math.sin(dirYaw) * Math.cos(dirPitch);
    const y = Math.sin(dirPitch);
    const z = -Math.cos(dirYaw) * Math.cos(dirPitch);
    return new Vec3(x, y, z);
}

// Сырой raycast с отклонением (hAngle, vAngle) от текущего взгляда entity.
// Направление обязано быть Vec3 — world.raycast вызывает его методы
// (offset/scaled) и падает на объекте-литерале.
//
// matcher пропускает "проходимые" блоки (см. PASS_THROUGH): иначе луч
// застревает в траве/цветке, и сеть видит стену вплотную там, где можно
// просто пройти — walking зря штрафовал бы за движение вперёд.
// Возвращает { block, distance } или null.
//
// ВАЖНО: world.raycast (prismarine-world) возвращает САМ блок (с полями
// name, intersect — точка попадания, face), а не { block, distance }.
// Раньше результат читался как hit.block/hit.distance — оба всегда были
// undefined, и боты были полностью слепы: все 64 луча = "небо на
// максимальной дальности", center_block всегда null, attack_center ни разу
// не копал блок. Приводим результат к ожидаемому виду здесь, в одном месте.
//
// Вторая ловушка: если raycast'у передан matcher, он возвращает ПЕРВЫЙ
// блок, для которого matcher вернул true, — без проверки формы блока.
// Поэтому воздух надо отсеивать самим (иначе луч "упирается" в воздух у
// глаз), а у твёрдых блоков — самим проверить пересечение с формой
// (полублок, забор: луч может пройти мимо) и запомнить точку попадания.
// Жидкости формы не имеют, но видны как есть (класс "жидкость").
function rawRaycast(world, origin, yaw, pitch, hAngle, vAngle, maxDistance) {
    const direction = rayDirection(yaw, pitch, hAngle, vAngle);
    const matcher = (block, iter) => {
        if (isAir(block.name) || isPassThrough(block.name)) return false;
        if (!block.shapes || block.shapes.length === 0) return true; // жидкость и т.п.
        const intersect = iter.intersect(block.shapes, block.position);
        if (!intersect) return false;
        block.intersect = intersect.pos;
        block.face = intersect.face; // грань попадания — на неё ставит блок place_front
        return true;
    };
    const block = world.raycast(origin, direction, maxDistance, matcher);
    if (!block) return null;
    const hitPoint = block.intersect ?? block.position.offset(0.5, 0.5, 0.5);
    return { block, distance: hitPoint.distanceTo(origin) };
}

// Быстрый луч для сетки зрения: тот же обход клеток, что у RaycastIterator
// prismarine-world, и та же проверка формы блока, но без создания объекта
// Block на каждой клетке каждого луча — свойства блока кэшируются по
// stateId. С сеткой 16x16 (256 лучей на бота) world.raycast стоил ~11 мс на
// бота, и с дюжиной ботов тик роя растягивался вдвое-втрое: боты решали раз
// в полсекунды и реагировали с задержкой (заметил автор, 2026-09-26).
// Результат — ровно как у rawRaycast (сверка — npm run test-vision).
const blockInfoByState = new Map(); // stateId -> { name, type, skip, shapes }

function blockInfo(world, stateId, position) {
    let info = blockInfoByState.get(stateId);
    if (!info) {
        const block = world.getBlock(position);
        info = {
            name: block.name,
            type: block.type,
            skip: isAir(block.name) || isPassThrough(block.name),
            shapes: block.shapes || [],
        };
        blockInfoByState.set(stateId, info);
    }
    return info;
}

function fastRaycast(world, origin, yaw, pitch, hAngle, vAngle, maxDistance) {
    const direction = rayDirection(yaw, pitch, hAngle, vAngle);
    const MAX = Number.MAX_VALUE;
    const dx = direction.x;
    const dy = direction.y;
    const dz = direction.z;
    const invX = dx === 0 ? MAX : 1 / dx;
    const invY = dy === 0 ? MAX : 1 / dy;
    const invZ = dz === 0 ? MAX : 1 / dz;
    const stepX = Math.sign(dx);
    const stepY = Math.sign(dy);
    const stepZ = Math.sign(dz);
    const tDeltaX = dx === 0 ? MAX : Math.abs(1 / dx);
    const tDeltaY = dy === 0 ? MAX : Math.abs(1 / dy);
    const tDeltaZ = dz === 0 ? MAX : Math.abs(1 / dz);
    let bx = Math.floor(origin.x);
    let by = Math.floor(origin.y);
    let bz = Math.floor(origin.z);
    let tMaxX = dx === 0 ? MAX : Math.abs((bx + (dx > 0 ? 1 : 0) - origin.x) / dx);
    let tMaxY = dy === 0 ? MAX : Math.abs((by + (dy > 0 ? 1 : 0) - origin.y) / dy);
    let tMaxZ = dz === 0 ? MAX : Math.abs((bz + (dz > 0 ? 1 : 0) - origin.z) / dz);
    const cell = new Vec3(0, 0, 0);

    for (;;) {
        cell.x = bx;
        cell.y = by;
        cell.z = bz;
        const stateId = world.getBlockStateId(cell); // 0 — воздух или незагруженный чанк
        if (stateId) {
            const info = blockInfo(world, stateId, cell);
            if (!info.skip) {
                if (info.shapes.length === 0) {
                    // Жидкость: как rawRaycast без точки попадания — центр блока.
                    const cx = bx + 0.5 - origin.x;
                    const cy = by + 0.5 - origin.y;
                    const cz = bz + 0.5 - origin.z;
                    return { block: { name: info.name, type: info.type }, distance: Math.sqrt(cx * cx + cy * cy + cz * cz) };
                }
                // Форма блока — как RaycastIterator.intersect.
                const px = origin.x - bx;
                const py = origin.y - by;
                const pz = origin.z - bz;
                let t = MAX;
                for (const shape of info.shapes) {
                    let tmin = (shape[invX > 0 ? 0 : 3] - px) * invX;
                    let tmax = (shape[invX > 0 ? 3 : 0] - px) * invX;
                    const tymin = (shape[invY > 0 ? 1 : 4] - py) * invY;
                    const tymax = (shape[invY > 0 ? 4 : 1] - py) * invY;
                    if ((tmin > tymax) || (tymin > tmax)) continue;
                    if (tymin > tmin) tmin = tymin;
                    if (tymax < tmax) tmax = tymax;
                    const tzmin = (shape[invZ > 0 ? 2 : 5] - pz) * invZ;
                    const tzmax = (shape[invZ > 0 ? 5 : 2] - pz) * invZ;
                    if ((tmin > tzmax) || (tzmin > tmax)) continue;
                    if (tzmin > tmin) tmin = tzmin;
                    if (tzmax < tmax) tmax = tzmax;
                    if (tmin < t) t = tmin;
                }
                if (t !== MAX) {
                    // Точка попадания и расстояние до неё — как Vec3.plus/scaled/distanceTo.
                    const hx = origin.x - (origin.x + dx * t);
                    const hy = origin.y - (origin.y + dy * t);
                    const hz = origin.z - (origin.z + dz * t);
                    return { block: { name: info.name, type: info.type }, distance: Math.sqrt(hx * hx + hy * hy + hz * hz) };
                }
            }
        }
        // RaycastIterator.next(): дальше предела — луч пуст.
        if (Math.min(Math.min(tMaxX, tMaxY), tMaxZ) > maxDistance) return null;
        if (tMaxX < tMaxY) {
            if (tMaxX < tMaxZ) {
                bx += stepX;
                tMaxX += tDeltaX;
            } else {
                bz += stepZ;
                tMaxZ += tDeltaZ;
            }
        } else if (tMaxY < tMaxZ) {
            by += stepY;
            tMaxY += tDeltaY;
        } else {
            bz += stepZ;
            tMaxZ += tDeltaZ;
        }
    }
}

function isAir(name) {
    return name === 'air' || name === 'cave_air' || name === 'void_air';
}

// Растительность и мелочь, сквозь которые можно пройти, но у которых есть
// форма коллизии — они не препятствие и не достойны занимать луч зрения.
// ^grass$ нужен, чтобы не зацепить grass_block (это полноценный блок).
const PASS_THROUGH = new RegExp([
    '^grass$', 'tall_grass', 'fern', 'flower', 'poppy', 'dandelion', 'orchid',
    'allium', 'bluet', 'daisy', 'cornflower', 'lily', 'tulip', 'rose_bush',
    'peony', 'sunflower', 'sapling', 'dead_bush', 'sprouts', 'crop', 'wheat',
    'carrots', 'potatoes', 'beetroot', 'sugar_cane', 'vine', 'kelp',
    'seagrass', 'mushroom', 'torch', 'rail', 'azalea',
].join('|'));

function isPassThrough(name) {
    return PASS_THROUGH.test(name);
}

// Прямая видимость между двумя точками: нет ли на отрезке блока с формой
// (стена, столб, стекло...). Для удара (js/actions.js: findEntityInCrosshair):
// в игре прицел упирается в блок, а бот бил сквозь стены (заметил автор,
// 2026-09-27) — сервер проверяет только расстояние. Жидкость и "проходное"
// (трава, цветы) не мешают — прицел игры их тоже пропускает. Симуляция
// считает так же (py/sim/game.py: SimArena._clear).
function lineOfSight(world, from, to) {
    const delta = to.minus(from);
    const distance = delta.norm();
    if (distance < 1e-6) return true;
    const matcher = (block, iter) => {
        if (isAir(block.name) || isPassThrough(block.name)) return false;
        if (!block.shapes || block.shapes.length === 0) return false; // жидкость — не стена
        const hit = iter.intersect(block.shapes, block.position);
        return !!hit && hit.pos.distanceTo(from) < distance;
    };
    return !world.raycast(from, delta.scaled(1 / distance), distance, matcher);
}

// Глаза — на eyeHeight сущности (у игроков в mineflayer 1.62, как в игре;
// так же считает bot.lookAt). Если сущность его не знает — 90% роста.
function eyePosition(entity) {
    return entity.position.offset(0, entity.eyeHeight ?? entity.height * 0.9, 0);
}

// Строит сетку из точки зрения `entity` (у него нужны position, height,
// yaw, pitch). world — это bot.world: raycast работает из любой позиции,
// поэтому тот же код используется и ботом, и ботом-обсёрвером, смотрящим
// на игрока при записи геймплея.
//
// Два режима (config.vision.mode):
//   "camera"   — классическая сетка внутри FOV вокруг направления взгляда,
//                как экран: col слева-направо, row сверху-вниз.
//   "circular" — круговое зрение: горизонтальный охват всегда 360°,
//                col 0 = прямо перед ботом, дальше по кругу ВПРАВО (по
//                часовой; кольцо поворачивается вместе с ботом),
//                вертикальный охват — vertical_fov вокруг горизонта
//                (НЕ привязан к pitch — прицел отдельно, зрение отдельно).
//                Идея: поворот головы нужен только чтобы взаимодействовать
//                (attack_center), а не чтобы "увидеть" — сущности у нас и
//                так видны на 360°, и камерное зрение создавало слепые зоны.
function buildVisionGrid(world, entity, config) {
    const [resX, resY] = config.vision.resolution;
    const maxDistance = config.vision.distance * 16; // чанки -> блоки
    const circular = config.vision.mode === 'circular';
    const horizontalSpan = circular ? Math.PI * 2 : (config.fov * Math.PI) / 180;
    const verticalSpan = ((circular ? config.vision.vertical_fov : config.fov) * Math.PI) / 180;

    const eyePos = eyePosition(entity);
    const { yaw, pitch } = entity;

    const cells = new Array(resX * resY);
    let idx = 0;

    // Опорный наклон сетки: в camera-режиме — текущий pitch (как взгляд), в
    // circular — горизонт (радар, от pitch не зависит). rayDirection сам
    // прибавляет vAngle к опорному наклону, поэтому vAngle ниже — чистое
    // отклонение. (Раньше pitch входил и в vAngle, и в rayDirection: в camera
    // наклон учитывался дважды, а circular-зрение "ездило" за головой.)
    const basePitch = circular ? 0 : pitch;

    for (let row = 0; row < resY; row++) {
        // Вертикальное отклонение: ряд 0 — верх, последний — низ, как пиксели
        // экрана. pitch > 0 — вверх: от +span/2 (верх) до -span/2 (низ).
        const vAngle = verticalSpan / 2 - (row * verticalSpan) / (resY - 1);
        for (let col = 0; col < resX; col++) {
            // Горизонтальный угол относительно взгляда (yaw растёт ВЛЕВО):
            //   camera — от +fov/2 (лево) до -fov/2 (право);
            //   circular — col 0 = прямо вперёд, дальше по кругу вправо.
            const hAngle = circular
                ? -(col * horizontalSpan) / resX
                : horizontalSpan / 2 - (col * horizontalSpan) / (resX - 1);
            const hit = fastRaycast(world, eyePos, yaw, basePitch, hAngle, vAngle, maxDistance);
            cells[idx++] = cellFromHit(hit, maxDistance);
        }
    }
    return cells;
}

function cellFromHit(hit, maxDistance) {
    // hit может прийти и без блока (пересечение с жидкостью, границей
    // незагруженного чанка и т.п.) — считаем такой луч "пустым".
    if (!hit || !hit.block) {
        // t=0 — класс "воздух/небо" (см. BLOCK_CLASS_PATTERNS).
        return { r: 0.5, g: 0.7, b: 1.0, d: 1.0, t: 0 };
    }

    const color = averageBlockColor(hit.block);
    const distance = Math.min(hit.distance / maxDistance, 1.0);
    return { r: color.r, g: color.g, b: color.b, d: distance, t: blockClassId(hit.block.name) };
}

// Классы блоков для зрения. Идея та же, что у палитры цветов: сети не
// нужно точное имя блока — нужен устойчивый сигнал "что за материал перед
// лучом". Классов ровно vision.block_classes (из config.json); Python
// получает голые id и one-hot'ит их, семантика живёт только здесь.
// ПОРЯДОК НЕ МЕНЯТЬ: id — часть контракта наблюдения.
//   0 воздух/небо, 1 древесина, 2 камень, 3 земля/песок, 4 растения,
//   5 жидкость, 6 руда, 7 снег/лёд, 8 стекло, 9 прочее.
const BLOCK_CLASS_PATTERNS = [
    [/^log$|_log$|^wood$|planks?|fence|stem|bamboo/, 1],
    [/stone|cobble|andesite|diorite|granite|deepslate|tuff|basalt|blackstone|brick|obsidian|bedrock|concrete|terracotta|netherrack|end_stone/, 2],
    [/dirt|mud|clay|farmland|path|grass_block|sand|gravel|powder_snow/, 3],
    [/grass|leaves|moss|fern|crop|wheat|carrot|potato|beet|flower|vine|fungus|seagrass|kelp/, 4],
    [/water|lava/, 5],
    [/ore|ancient_debris|amethyst/, 6],
    [/snow|ice/, 7],
    [/glass/, 8],
];

function blockClassId(name) {
    if (name === 'air' || name === 'cave_air' || name === 'void_air') return 0;
    for (const [pattern, classId] of BLOCK_CLASS_PATTERNS) {
        if (pattern.test(name)) return classId;
    }
    return 9; // прочее/неизвестное
}

// Что именно находится строго по центру прицела — отдельным полем состояния
// (не размазанным по цвету): модулю gathering — какой блок долбить, а сети —
// во что она целится (круговая сетка зрения за наклоном головы не следует).
// Возвращает { id, name, distance, t } или null (воздух / ничего в пределах
// дальности); t — класс блока, как в сетке зрения.
function centerBlockInfo(world, entity, config) {
    const maxDistance = config.vision.distance * 16;
    const eyePos = eyePosition(entity);
    const { yaw, pitch } = entity;

    const hit = rawRaycast(world, eyePos, yaw, pitch, 0, 0, maxDistance);
    if (!hit || !hit.block || hit.block.name === 'air') return null;

    // distance — от глаз до точки попадания, в блоках: модулю добычи надо
    // знать, дотягивается ли бот (в ванили ~4.5 блока).
    return {
        id: hit.block.type,
        name: hit.block.name,
        distance: Math.round(hit.distance * 100) / 100,
        t: blockClassId(hit.block.name),
    };
}

// Сырой центральный raycast (тот же луч, что и centerBlockInfo, но с полным
// объектом hit.block) — нужен action-макросу attack_center, чтобы получить
// настоящий Block-объект для bot.dig(), а не только id/name.
function centerRaycast(world, entity, config) {
    const maxDistance = config.vision.distance * 16;
    const eyePos = eyePosition(entity);
    const { yaw, pitch } = entity;
    return rawRaycast(world, eyePos, yaw, pitch, 0, 0, maxDistance);
}

// Средний цвет блока. У mineflayer нет готовой текстурной палитры, поэтому
// используем приблизительные цвета по имени/типу блока с разумным фолбэком.
// Кэшируем результат по имени блока — цветов много, raycast'ов ещё больше.
const colorCache = new Map();

function averageBlockColor(block) {
    const name = block.name;
    if (colorCache.has(name)) return colorCache.get(name);

    const color = guessBlockColor(name);
    colorCache.set(name, color);
    return color;
}

// Грубая палитра распространённых блоков. Значения 0..1.
// Это не пытается быть точным рендером — цель лишь дать сети
// устойчивый, консистентный сигнал "что за блок перед лучом".
const PALETTE = [
    [/grass| moss|fern|leaves|crop|wheat|carrot|potato|beet/, { r: 0.30, g: 0.55, b: 0.20 }],
    [/dirt|mud|clay|farmland|path/, { r: 0.55, g: 0.38, b: 0.24 }],
    [/stone|cobble|andesite|diorite|granite|deepslate|tuff|basalt|blackstone/, { r: 0.50, g: 0.50, b: 0.50 }],
    [/sand|sandstone|gravel/, { r: 0.86, g: 0.80, b: 0.60 }],
    [/wood|log|plank|oak|birch|spruce|jungle|acacia|dark_oak|mangrove|cherry|fence|stem/, { r: 0.65, g: 0.50, b: 0.30 }],
    [/water/, { r: 0.20, g: 0.35, b: 0.80 }],
    [/lava|magma/, { r: 0.90, g: 0.40, b: 0.10 }],
    [/iron|copper|gold/, { r: 0.80, g: 0.70, b: 0.50 }],
    [/diamond|emerald/, { r: 0.30, g: 0.90, b: 0.80 }],
    [/redstone|netherrack|nether_brick/, { r: 0.60, g: 0.15, b: 0.15 }],
    [/snow|ice|packed_ice|frosted/, { r: 0.90, g: 0.95, b: 1.00 }],
    [/glass|glass_pane/, { r: 0.70, g: 0.85, b: 0.90 }],
    [/wool|carpet|bed|concrete|terracotta/, { r: 0.75, g: 0.60, b: 0.55 }],
    [/brick/, { r: 0.65, g: 0.35, b: 0.30 }],
    [/obsidian|coal|bedrock/, { r: 0.15, g: 0.10, b: 0.25 }],
];

function guessBlockColor(name) {
    for (const [pattern, color] of PALETTE) {
        if (pattern.test(name)) return color;
    }
    return { r: 0.45, g: 0.45, b: 0.45 }; // неизвестный блок — нейтральный серый
}

// Чувство пола под ногами: сколько пола до края впереди, справа, сзади и
// слева (относительно взгляда), в блоках, со ЗНАКОМ и не дальше GROUND_RANGE:
// над полом — сколько ещё пола в эту сторону; над пустотой (свесился с края)
// — минус сколько до пола в эту сторону (к краю -0.29, от края -3). Сетка
// зрения смотрит вниз самое большее на 35° и пол ближе ~2.3 блока не видит,
// прицел — одна точка: без этого чувства мост строили крадучись от самого
// старта — где край, сеть не знала (2026-09-28). Пол — блок с формой
// столкновения на уровне под ногами: шаг вниз — уже край (мост на высоте:
// блок ниже — не пол, а падение с уровня). То же — plugin Vision.groundProbe
// и py/sim/vision.py: ground_probe.
const GROUND_RANGE = 3;

function groundProbe(world, entity) {
    const pos = entity.position;
    const cell = new Vec3(0, Math.floor(pos.y - 0.01), 0);
    const floorAt = (x, z) => {
        cell.x = x;
        cell.z = z;
        const stateId = world.getBlockStateId(cell); // 0 — воздух или незагруженный чанк
        return !!stateId && blockInfo(world, stateId, cell).shapes.length > 0;
    };
    // Вперёд f = (-sin yaw, -cos yaw), вправо r = (-f.z, f.x) — как в js/state.js.
    const fx = -Math.sin(entity.yaw);
    const fz = -Math.cos(entity.yaw);
    const directions = [[fx, fz], [-fz, fx], [-fx, -fz], [fz, -fx]];
    return directions.map(([dx, dz]) => Math.round(floorDistance(floorAt, pos.x, pos.z, dx, dz) * 1000) / 1000);
}

// Идём от (x, z) по направлению (dx, dz) клетка за клеткой (как луч зрения,
// только по горизонтали) до первой клетки, где с полом не так, как под
// серединой бота: был пол — там край, не было — там пол начинается.
function floorDistance(floorAt, x, z, dx, dz) {
    let cx = Math.floor(x);
    let cz = Math.floor(z);
    const onFloor = floorAt(cx, cz);
    const stepX = dx > 0 ? 1 : dx < 0 ? -1 : 0;
    const stepZ = dz > 0 ? 1 : dz < 0 ? -1 : 0;
    // Сколько пройти до следующей границы клеток по x и по z.
    let nextX = stepX > 0 ? (cx + 1 - x) / dx : stepX < 0 ? (cx - x) / dx : Infinity;
    let nextZ = stepZ > 0 ? (cz + 1 - z) / dz : stepZ < 0 ? (cz - z) / dz : Infinity;
    const deltaX = stepX !== 0 ? Math.abs(1 / dx) : Infinity;
    const deltaZ = stepZ !== 0 ? Math.abs(1 / dz) : Infinity;
    for (;;) {
        let t;
        if (nextX < nextZ) {
            t = nextX;
            cx += stepX;
            nextX += deltaX;
        } else {
            t = nextZ;
            cz += stepZ;
            nextZ += deltaZ;
        }
        if (t >= GROUND_RANGE) return onFloor ? GROUND_RANGE : -GROUND_RANGE;
        if (floorAt(cx, cz) !== onFloor) return onFloor ? t : -t;
    }
}

module.exports = {
    buildVisionGrid, centerBlockInfo, centerRaycast, lookDirection, eyePosition, blockClassId, isPassThrough, lineOfSight,
    groundProbe, GROUND_RANGE,
    rawRaycast, fastRaycast, // для сверки быстрого луча с эталоном (js/test_vision.js)
};
