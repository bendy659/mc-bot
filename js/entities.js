// "Зрение на сущностей": ближайшие мобы/игроки/предметы как часть состояния.
//
// Mineflayer видит всех сущностей вокруг бота (bot.entities — словарь
// id -> entity с position и type). Для сети важно не мировое положение,
// а сущность В СИСТЕМЕ КООРДИНАТ НАБЛЮДАТЕЛЯ: вперёд/вправо/вверх
// относительно взгляда — тогда сигнал не зависит от того, куда бот
// повёрнут в мире, и сеть может напрямую учиться "разворачиваться на
// сущность впереди".
//
// Выход — список фиксированной длины max_tracked (паддинг нулями),
// каждая запись: { id, forward, right, up, dist, type }.
//   forward/right/up — компоненты сдвига в локальной системе взгляда,
//                      нормированные на радиус слежения (-1..1);
//   dist             — дистанция, нормированная на радиус (0..1);
//   type             — 0 враждебная, 1 мирная, 2 игрок, 3 неживое (выпавший
//                      предмет, стрела, шар опыта...).
// id нужен Python-модулям (gathering ловит исчезновение моба = убийство).

// Классы сущностей — контракт с py/training_modules/targets.py.
const HOSTILE = 0;
const PASSIVE = 1;
const PLAYER = 2;
const NON_LIVING = 3; // выпавшие предметы, шары опыта, стрелы, лодки...

// Враждебные мобы — те, что в ванилле сами атакуют игрока. Нужен для
// категории 'mob' (слизни, гасты, големы вперемешку) и старых версий
// mineflayer, где все мобы — 'mob'.
const HOSTILE_MOBS = new Set([
    'zombie', 'husk', 'drowned', 'skeleton', 'stray', 'bogged', 'creeper',
    'spider', 'cave_spider', 'enderman', 'witch', 'pillager', 'vindicator',
    'ravager', 'evoker', 'vex', 'slime', 'magma_cube', 'phantom', 'silverfish',
    'guardian', 'elder_guardian', 'blaze', 'ghast', 'hoglin', 'piglin',
    'piglin_brute', 'zoglin', 'warden', 'breeze', 'wither_skeleton',
    'shulker', 'ender_dragon',
]);

// Порядок: живые (игроки, мобы) первыми, потом неживое; внутри — самые
// близкие первыми. Если сущностей больше лимита, отбрасываются дальние и
// неживые: россыпь выпавших предметов (копка, drop_item в салках) иначе
// вытесняла из поля зрения сети игроков и мобов.
function buildEntitiesList(sourceEntities, observer, config) {
    const radius = config.entities.radius;
    const maxTracked = config.entities.max_tracked;

    const eye = observer.position; // позиция наблюдателя (бот или записываемый игрок)
    // Вперёд по взгляду в конвенции mineflayer (см. js/vision.js).
    const fx = -Math.sin(observer.yaw);
    const fz = -Math.cos(observer.yaw);

    const seen = [];
    for (const entity of Object.values(sourceEntities)) {
        // isValid=false — сервер удалил сущность, но mineflayer ещё не вычистил.
        if (!entity || !entity.isValid || !entity.position) continue;
        if (entity === observer) continue;

        const dx = entity.position.x - eye.x;
        const dy = entity.position.y - eye.y;
        const dz = entity.position.z - eye.z;
        const dist = Math.hypot(dx, dy, dz);
        if (dist > radius || dist < 1e-6) continue;

        seen.push({ entity, dx, dy, dz, dist, type: classifyEntity(entity) });
    }
    seen.sort((a, b) => (a.type === NON_LIVING) - (b.type === NON_LIVING) || a.dist - b.dist);

    const out = [];
    for (let i = 0; i < maxTracked; i++) {
        if (i >= seen.length) {
            // Паддинг: present=0, всё остальное нули — сеть учится отличать
            // "слот пустой" по самому первому полю.
            out.push({ id: 0, present: 0, forward: 0, right: 0, up: 0, dist: 0, type: 0, name: '', height: 0 });
            continue;
        }
        const { entity, dx, dy, dz, dist, type } = seen[i];
        out.push({
            id: entity.id,
            present: 1,
            // Локальная система: f = (-sin yaw, -cos yaw) — вперёд по взгляду,
            // правый вектор — поворот f на 90°: r = (-f.z, f.x). Та же
            // математика, что у move_forward/move_right в state.js.
            forward: round(dx * fx + dz * fz, radius),
            right: round(dx * -fz + dz * fx, radius),
            up: round(dy, radius),
            dist: round(dist, radius, true),
            type,
            // Имя (ник игрока или вид моба) — в сеть не идёт, нужно
            // модулям: follow отличает человека от ботов своего роя.
            name: entity.username ?? entity.name ?? '',
            // Высота сущности — куда смотреть (looking целится в "лицо").
            height: Math.round((entity.height ?? 0) * 100) / 100,
        });
    }
    return out;
}

// entity.type у mineflayer — категория из minecraft-data, и в разных версиях
// она разная. В новых (26.1): 'hostile' (зомби, скелеты, криперы...),
// 'animal' / 'passive' / 'water_creature' / 'ambient' (мирные), 'mob'
// (слизни, гасты, големы вперемешку), 'player', а 'other' — выпавшие
// предметы, шары опыта, лодки; 'projectile' — стрелы. В старых — просто
// 'mob' / 'object' / 'orb'. До 2026-09-26 тут знали только старые имена,
// и на 26.1 зомби и выпавшие предметы считались "мирными мобами": follow
// мог пойти за зомби, looking — следить за предметом, gathering не видел
// выпавших брёвен и не засчитывал убитых мобов.
function classifyEntity(entity) {
    switch (entity.type) {
        case 'player':
            return PLAYER;
        case 'hostile':
            return HOSTILE;
        case 'animal':
        case 'passive':
        case 'water_creature':
        case 'ambient':
            return PASSIVE;
        case 'mob': {
            // У старых версий имя вида — entityName.
            const name = entity.name ?? entity.entityName ?? '';
            return HOSTILE_MOBS.has(name) ? HOSTILE : PASSIVE;
        }
        default:
            return NON_LIVING; // 'other', 'object', 'orb', 'projectile', 'living' (стойка для брони)...
    }
}

function round(value, radius, zeroToOne = false) {
    const normalized = zeroToOne ? value / radius : Math.max(-1, Math.min(1, value / radius));
    return Math.round(normalized * 1000) / 1000;
}

module.exports = { buildEntitiesList, classifyEntity, HOSTILE_MOBS, HOSTILE, PASSIVE, PLAYER, NON_LIVING };
