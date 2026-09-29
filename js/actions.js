// Дискретное пространство действий-макросов. Каждый макрос — готовая
// комбинация управлений, выполняемая за один вызов: сеть выбирает
// "jump_forward", а не подбирает прыжок и движение вперёд по отдельности.
// Это сильно сужает пространство поиска на этапе "научиться ходить".
//
// Действия разбиты на КАНАЛЫ (ноги / голова / руки): у каждого канала
// своя сеть на стороне Python, и за тик приходит по одному макросу на
// каждый канал — они выполняются одновременно (идти + смотреть + бить).

const { Vec3 } = require('vec3');
const { centerRaycast, lookDirection, eyePosition, lineOfSight } = require('./vision');
const { isPlaceableBlock, isFood, isArmor, ARMOR_SLOTS } = require('./inventory');
const { getAttributeValue } = require('prismarine-physics/lib/attribute');

const TURN_STEP = 30 * (Math.PI / 180);      // поворот корпуса (ноги), радианы
// Наклон взгляда (голова). 10°, а не 20°: "в прицеле" в looking — это
// отклонение < 8°, и с шагом 20° нужный наклон часто лежал между шагами —
// навестись по вертикали было физически невозможно. С шагом 10° худший
// промах — 5°.
const LOOK_STEP = 10 * (Math.PI / 180);
const LOOK_YAW_STEP = 10 * (Math.PI / 180);  // доворот взгляда (голова) — мельче, чем turn_*
const ACTION_DURATION_MS = 250;           // сколько держать движение

// attack_center (удар в салках, gathering): насколько далеко и насколько
// точно "впереди" должна быть сущность, чтобы бить её, а не копать блок.
// Конус — только по горизонтали: вверх-вниз удар наводит голову сам.
const ATTACK_RANGE = 3.5;
const ATTACK_CONE_COS = Math.cos(30 * (Math.PI / 180)); // ~30° влево-вправо от взгляда
// Дальность копания: в ванили ~4.5 блока от глаз (выживание).
const DIG_REACH = 4.5;
// Бревно руками копается ~3 с, камень без кирки — дольше; 10 с с запасом.
const DIG_TIMEOUT_MS = 10000;
// Остальные дела рук (поставить блок, съесть, надеть, выбросить) — быстрые;
// если сервер не ответил за это время, руки освобождаются.
const HANDS_TIMEOUT_MS = 2500;
const PLACE_REACH = 4.5;
// Сколько place_below ждёт, пока прыжок поднимет ноги на блок выше опоры:
// выше блока бот держится примерно со 150-й по 400-ю мс прыжка.
const PLACE_BELOW_WAIT_MS = 500;
const PHYSICS_TICK_MS = 50;
const WEAPON_BACK_DECISIONS = 3; // решений (по 150 мс) со свободными руками — и оружие обратно в руку
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
// Грань блока (номер из raycast prismarine-world) -> куда от неё ставить.
const FACE_VECTORS = [
    new Vec3(0, -1, 0), new Vec3(0, 1, 0), new Vec3(0, 0, -1),
    new Vec3(0, 0, 1), new Vec3(-1, 0, 0), new Vec3(1, 0, 0),
];
// Куда блок ставится (как в симуляции: только в воздух).
const AIR_NAMES = new Set(['air', 'cave_air', 'void_air']);
// Сущности, которые ставить блок не мешают (у сервера так же: предметы, опыт, стрелы).
const NOT_BLOCKING_BUILD = new Set(['item', 'experience_orb', 'arrow', 'spectral_arrow', 'trident']);

// Какую живность считаем допустимой целью для attack_center. Список
// сознательно небольшой и без крипера (взрыв в упор — плохая идея для
// зачаточной политики, которая ещё не умеет отступать) — расширяется
// одной строкой при необходимости.
const ATTACKABLE_ENTITY_NAMES = new Set(['sheep', 'chicken', 'cow', 'pig', 'zombie']);

const MOVES = {
    forward: { forward: true },
    back: { back: true },
    left: { left: true },
    right: { right: true },
};

class ActionExecutor {
    constructor(bot, config) {
        this.bot = bot;
        this.config = config;
        this.activeUntil = 0; // до какого момента (Date.now) держим текущее движение
        this.activeMove = null;
        // true, пока руки заняты долгим делом (копают, ставят блок, едят...):
        // всё это асинхронно, и новое дело поверх незаконченного не начинаем.
        this.handsBusy = false;
        // Позиция блока, который сейчас копаем (null — не копаем): каждый тик
        // проверяем, что он всё ещё в прицеле и рядом (checkDigging).
        this.digging = null;
        // Салки: кого из игроков можно ударить (удар и есть "осалил") — id
        // сущностей убегающих, ботов и людей. Список приходит от Python с
        // каждым действием (tag_ids): кто сейчас убегает, знает судья. Не
        // водящему и в остальных задачках он пуст — игроков не бьём.
        this.taggableIds = new Set();
        this.swarmPrefix = `${config.bot.username}${config.bot.separator ?? '_'}`;
        // По кому пришёлся последний удар (id сущности) — уходит в состояние
        // (attacked_id), по нему судья салок и засчитывает "осалил".
        this.attackedId = null;
        // Кто последним ударил самого бота (id сущности) — уходит в состояние
        // (hurt_by): так судья узнаёт, что убегающего осалил человек-водящий,
        // его удары Python больше ниоткуда не видно.
        this.hurtBy = null;
        // Заряд удара: когда бил последний раз и с какой силой. С 1.9 удар
        // "перезаряжается" (кулак — 0.25 с), урон — 0.2 + 0.8 * заряд²: бить
        // каждый тик — вполсилы. Сеть видит заряд (state.attack_charge), а
        // охота награждает за настоящий урон с учётом силы (damage_dealt).
        this.lastAttackAt = 0;
        this.lastAttackCharge = 1;
        // Крит (идея автора: "удар в падении — это очень хорошо"): полная
        // сила, бот падает после прыжка, не бежит — урон x1.5, как в игре.
        this.lastAttackCrit = false;
        this.damageDealt = [];
        bot.on('entityHurt', (entity, source) => {
            if (entity === bot.entity && source) {
                this.hurtBy = source.id;
            } else if (source === bot.entity) {
                // Мой удар нанёс урон (событие урона сервер шлёт всем рядом —
                // в неуязвимость после прошлого удара его нет).
                this.damageDealt.push({
                    id: entity.id,
                    charge: Math.round(this.lastAttackCharge * 1000) / 1000,
                    crit: this.lastAttackCrit,
                });
            }
        });
    }

    // Скорость атаки (полных ударов в секунду): кулак — 4, меч — 1.6. Сервер
    // присылает атрибут игрока; нет его — как у кулака.
    attackSpeed() {
        const attributes = this.bot.entity?.attributes ?? {};
        const key = Object.keys(attributes).find((name) => name.endsWith('attack_speed'));
        return key ? getAttributeValue(attributes[key]) : 4.0;
    }

    // Заряд удара 0..1 — как getAttackStrengthScale в игре: тики с прошлого
    // удара против "перезарядки" 20 / скорость атаки.
    attackCharge() {
        const delayTicks = 20 / Math.max(this.attackSpeed(), 0.1);
        const ticks = (Date.now() - this.lastAttackAt) / 50;
        return Math.min(1, Math.max(0, (ticks + 0.5) / delayTicks));
    }

    takeDamageDealt() {
        const dealt = this.damageDealt;
        this.damageDealt = [];
        return dealt;
    }

    takeAttacked() {
        const id = this.attackedId;
        this.attackedId = null;
        return id;
    }

    takeHurtBy() {
        const id = this.hurtBy;
        this.hurtBy = null;
        return id;
    }

    setTaggable(ids) {
        this.taggableIds = new Set(ids ?? []);
    }

    // Игрок, которого можно ударить: только водящему в салках и только
    // убегающего (список от судьи).
    isTaggable(entity) {
        return entity.type === 'player' && this.taggableIds.has(entity.id);
    }

    // Люди рядом (не боты роя): ник, id сущности, где стоит, режим игры и
    // команда сервера — судья салок (py/tag_game.py) по команде знает, играет
    // ли человек и кем (it — водит, runners — убегает), а по режиму — что
    // наблюдателя трогать не надо.
    visibleHumans() {
        const bot = this.bot;
        const humans = [];
        for (const [name, player] of Object.entries(bot.players)) {
            const entity = player.entity;
            if (!entity || entity === bot.entity || name.startsWith(this.swarmPrefix)) continue;
            const p = entity.position;
            humans.push({
                name, id: entity.id, x: p.x, y: p.y, z: p.z,
                gamemode: player.gamemode ?? 0,
                team: bot.teamMap?.[name]?.team ?? null,
            });
        }
        return humans;
    }

    // Выполнить макросы всех каналов за тик: { legs, head, hands }.
    // Отсутствующий канал = ничего не делать. Имена — контракт с Python
    // (CHANNELS в py/protocol.py), менять только синхронно.
    executeAll(actions) {
        this.stopMovement();
        this.executeLegs(actions.legs ?? 'idle');
        this.executeHead(actions.head ?? 'head_idle');
        this.executeHands(actions.hands ?? 'hands_idle');
    }

    executeLegs(name) {
        const bot = this.bot;
        switch (name) {
            case 'idle':
                break;
            case 'walk_forward':
                this.holdMove({ forward: true, jump: this.stepAhead() });
                break;
            case 'sprint_forward':
                // Бег — тот же макрос, что и walk_forward, плюс control
                // state 'sprint'. Если сытости не хватает (<6/20 в ванили),
                // сервер сам тихо не даст бежать — бот просто пойдёт шагом,
                // никакой ошибки не будет.
                this.holdMove({ forward: true, sprint: true, jump: this.stepAhead() });
                break;
            case 'walk_back':
                this.holdMove(MOVES.back);
                break;
            case 'strafe_left':
                this.holdMove(MOVES.left);
                break;
            case 'strafe_right':
                this.holdMove(MOVES.right);
                break;
            // yaw в mineflayer растёт ВЛЕВО (см. конвенцию в js/vision.js).
            case 'turn_left':
                this.turnView(TURN_STEP, 0);
                break;
            case 'turn_right':
                this.turnView(-TURN_STEP, 0);
                break;
            case 'jump_forward':
                // Комбинированный макрос: прыжок + движение вперёд,
                // чтобы перепрыгивать препятствия высотой в один блок.
                this.holdMove({ forward: true, jump: true });
                break;
            case 'jump':
                // Прыжок на месте: например, чтобы в прыжке поставить блок под
                // себя (руки: place_below) — так строят столб вверх.
                this.holdMove({ jump: true });
                break;
            case 'sneak_back':
                // Задом крадучись: крадущийся не сходит с края блока (игра сама
                // не пускает) — так строят мост над пустотой. Решение раз в
                // 150 мс, без шифта бот у края срывался бы.
                this.holdMove({ back: true, sneak: true });
                break;
            case 'sneak':
                this.holdMove({ sneak: true });
                break;
            default:
                console.warn(`[actions] Неизвестное действие ног: ${name}`);
        }
    }

    // Голова. В Minecraft yaw взгляда и направления ходьбы — одно число,
    // поэтому look_left/look_right тоже чуть поворачивают курс; они просто
    // мельче turn_* ног и нужны для точного наведения (looking/gathering).
    executeHead(name) {
        switch (name) {
            case 'head_idle':
                break;
            // pitch в mineflayer > 0 — ВВЕРХ, yaw растёт влево.
            case 'look_up':
                this.turnView(0, LOOK_STEP);
                break;
            case 'look_down':
                this.turnView(0, -LOOK_STEP);
                break;
            case 'look_left':
                this.turnView(LOOK_YAW_STEP, 0);
                break;
            case 'look_right':
                this.turnView(-LOOK_YAW_STEP, 0);
                break;
            default:
                console.warn(`[actions] Неизвестное действие головы: ${name}`);
        }
    }

    // Автопрыжок, как одноимённая настройка в самой игре: идёт вперёд, а по
    // курсу уступ высотой в блок, над которым свободно, — подпрыгнуть. Без
    // этого боты упирались в уступ и "шли" на месте (заметил автор): сеть
    // его почти не видит — лучи зрения идут от глаз и проходят над ним, —
    // а прыгнуть в нужный миг надо было угадать. Стену в 2 блока так не
    // перепрыгнуть — её обходить, как и раньше.
    stepAhead() {
        const bot = this.bot;
        const position = bot.entity.position;
        const feetY = Math.floor(position.y);
        const here = { x: Math.floor(position.x), z: Math.floor(position.z) };
        const solid = (x, y, z) => bot.blockAt(new Vec3(x, y, z))?.boundingBox === 'block';
        const free = (x, y, z) => bot.blockAt(new Vec3(x, y, z))?.boundingBox === 'empty';
        if (!free(here.x, feetY + 2, here.z)) return false; // над головой потолок — не подпрыгнуть
        const forward = lookDirection({ yaw: bot.entity.yaw, pitch: 0 });
        for (const distance of [0.5, 1.0, 1.4]) {
            const x = Math.floor(position.x + forward.x * distance);
            const z = Math.floor(position.z + forward.z * distance);
            if (x === here.x && z === here.z) continue; // ещё своя клетка
            if (solid(x, feetY, z)) return free(x, feetY + 1, z) && free(x, feetY + 2, z);
            if (!free(x, feetY, z)) return false; // забор, стекло, вода — не то
        }
        return false;
    }

    // Повернуть взгляд и СРАЗУ показать это серверу (force). Сам mineflayer
    // шлёт серверу поворот головы плавно — не быстрее 172°/с, — а прицел
    // (raycast, копка, удар) у нас меняется мгновенно: видимая голова
    // отставала, и казалось, что бот ломает не там, куда смотрит (заметил
    // автор вживую).
    turnView(dyaw, dpitch) {
        const bot = this.bot;
        const pitch = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, bot.entity.pitch + dpitch));
        bot.look(bot.entity.yaw + dyaw, pitch, true);
    }

    executeHands(name) {
        switch (name) {
            case 'hands_idle':
                break;
            case 'attack_center':
                this.attackOrDigCenter();
                break;
            case 'place_below':
                this.placeBelow();
                break;
            case 'place_front':
                this.placeFront();
                break;
            case 'use_item':
                this.useItem();
                break;
            case 'equip_armor':
                this.equipArmor();
                break;
            case 'drop_item':
                this.dropItem();
                break;
            default:
                console.warn(`[actions] Неизвестное действие рук: ${name}`);
        }
    }

    // gathering: удар/копание того, что прямо по центру прицела. Сначала
    // проверяем ближайшую подходящую сущность впереди (findEntityInCrosshair, бой
    // мгновенный), и только если её нет — копаем блок под прицелом
    // (bot.dig — "длинная" операция, ~3 сек руками, поэтому не await'им
    // её тут, а просто помечаем себя "занят", чтобы не запускать вторую
    // копку поверх текущей).
    // Долгое дело рук: пока оно идёт, новые не начинаются; если сервер не
    // ответил за timeoutMs — руки освобождаются сами (иначе отключились бы
    // навсегда). Ошибки (не дали поставить, нечего есть) — обычное дело для
    // случайных действий, не шумим ими в логе.
    runHands(task, timeoutMs, onTimeout = null) {
        if (this.handsBusy) return;
        this.handsBusy = true;
        const timeout = setTimeout(() => {
            if (!this.handsBusy) return;
            if (onTimeout) {
                try {
                    onTimeout();
                } catch {
                    /* уже не занят */
                }
            }
            this.handsBusy = false;
        }, timeoutMs);
        Promise.resolve()
            .then(task)
            .catch(() => {})
            .finally(() => {
                clearTimeout(timeout);
                this.handsBusy = false;
            });
    }

    findItem(predicate) {
        return this.bot.inventory.items().find(predicate) ?? null;
    }

    // Поставить блок под себя — столб. Получится только в прыжке: клетка под
    // ногами должна освободиться (стоя на месте, сервер откажет — бот в ней
    // стоит). Опора — ближайший твёрдый блок внизу (тот, с которого прыгнули).
    // Ноги на блок выше опоры окажутся не сразу (физика тикает раз в 50 мс,
    // через 150 мс после прыжка бот бывает поднят лишь на 0.75 блока), поэтому
    // макрос сам ждёт этой высоты до PLACE_BELOW_WAIT_MS. Так столб выходит и
    // при прыжке в этом же тике (ноги: jump, руки: place_below), и при прыжке
    // тиком раньше; не прыгнул — ничего не ставит. Стоит на земле —
    // подпрыгивает сам: ноги решает другая сеть, и "прыжок + блок под себя"
    // в один тик двум сетям не давался — боты стояли у столба цели и не
    // строились (2026-09-28). Как игрок: прыжок и правая кнопка разом.
    placeBelow() {
        const bot = this.bot;
        const item = this.findItem((it) => isPlaceableBlock(bot, it));
        if (!item || this.handsBusy) return;
        const support = this.supportBelow();
        if (!support) return;
        if (bot.entity.onGround) this.holdMove({ jump: true });
        this.runHands(async () => {
            await bot.equip(item, 'hand');
            const deadline = Date.now() + PLACE_BELOW_WAIT_MS;
            while (bot.entity.position.y < support.position.y + 2) {
                if (Date.now() > deadline) return; // так и не подпрыгнул
                // Приземлился, а блок ещё не поставлен (команда пришла в
                // воздухе) — прыгнуть снова, а не ждать впустую.
                if (bot.entity.onGround) this.holdMove({ jump: true });
                await sleep(PHYSICS_TICK_MS / 2);
            }
            // Взгляд не трогаем (как place_front): сервер направление взгляда не
            // проверяет, а поворот головы "на опору" сбивал угол на некратный 10° —
            // симуляция этого не знает, и мост после столба шёл вкось (2026-09-29).
            await bot._placeBlockWithOptions(support, new Vec3(0, 1, 0), { swingArm: 'right', forceLook: 'ignore' });
        }, HANDS_TIMEOUT_MS);
    }

    // Ближайший твёрдый блок под ногами (до 3 вниз) — на него и ставим.
    supportBelow() {
        const feet = this.bot.entity.position;
        const x = Math.floor(feet.x);
        const z = Math.floor(feet.z);
        for (let y = Math.floor(feet.y) - 1; y >= Math.floor(feet.y) - 3; y--) {
            const block = this.bot.blockAt(new Vec3(x, y, z));
            if (block && block.boundingBox === 'block') return block;
        }
        return null;
    }

    // Поставить блок на грань того, во что смотрит прицел (стена, мост...).
    placeFront() {
        const bot = this.bot;
        const item = this.findItem((it) => isPlaceableBlock(bot, it));
        if (!item || this.handsBusy) return;
        const target = this.placeFrontTarget();
        if (!target) return;
        this.runHands(async () => {
            await bot.equip(item, 'hand');
            await bot._placeBlockWithOptions(target.block, target.face, { swingArm: 'right', forceLook: 'ignore' });
        }, HANDS_TIMEOUT_MS);
    }

    // Куда встал бы блок place_front: клетка у грани в прицеле, или null —
    // прицел пуст или дальше руки, грань неизвестна, это клетка самого бота.
    placeFrontTarget() {
        const bot = this.bot;
        const hit = centerRaycast(bot.world, bot.entity, this.config);
        if (!hit || !hit.block || hit.distance > PLACE_REACH || hit.block.face == null) return null;
        const face = FACE_VECTORS[hit.block.face];
        const destination = hit.block.position.plus(face);
        // В клетку, где стоит сам бот, сервер блок не поставит.
        const feet = bot.entity.position.floored();
        if (destination.equals(feet) || destination.equals(feet.offset(0, 1, 0))) return null;
        return { block: hit.block, face, destination };
    }

    // Поставил бы place_front блок прямо сейчас (state.can_place): есть чем,
    // грань в прицеле, клетка за ней — воздух, и никто в ней не стоит. На краю
    // моста блок встаёт, только когда бот свесился на 0.2857..0.3 (луч из-за
    // края — в бок блока под ногами), а крадущегося игра двигает шагами по
    // 0.05: бот может замереть чуть раньше окна. По чувству пола сеть этого
    // не различала (разница в третьем знаке), по can_place — сразу (2026-09-29).
    canPlaceFront() {
        const bot = this.bot;
        if (bot.health <= 0 || !this.findItem((it) => isPlaceableBlock(bot, it))) return false;
        const target = this.placeFrontTarget();
        if (!target) return false;
        const cell = target.destination;
        const block = bot.blockAt(cell);
        if (!block || !AIR_NAMES.has(block.name)) return false;
        const entities = new Set([bot.entity, ...Object.values(bot.entities)]);
        for (const entity of entities) {
            if (!entity || !entity.position || NOT_BLOCKING_BUILD.has(entity.name)) continue;
            // Размеры — как у сервера, float: половина ширины 0.6f / 2 = 0.30000001,
            // и стоящий ровно впритык к клетке для сервера её уже задевает.
            const half = Math.fround(Math.fround(entity.width ?? 0.6) / 2);
            const height = Math.fround(entity.height ?? 1.8);
            const { x, y, z } = entity.position;
            if (cell.x < x + half && cell.x + 1 > x - half && cell.z < z + half && cell.z + 1 > z - half
                && cell.y < y + height && cell.y + 1 > y) return false;
        }
        return true;
    }

    // Голоден и есть еда — поесть; иначе — "использовать" то, что в руке.
    useItem() {
        const bot = this.bot;
        if (this.handsBusy) return;
        const food = bot.food < 20 ? this.findItem((it) => isFood(bot, it)) : null;
        if (food) {
            this.runHands(async () => {
                await bot.equip(food, 'hand');
                await bot.consume();
            }, 5000);
            return;
        }
        if (!bot.heldItem) return;
        this.runHands(async () => {
            bot.activateItem();
            await new Promise((resolve) => setTimeout(resolve, ACTION_DURATION_MS));
            bot.deactivateItem();
        }, HANDS_TIMEOUT_MS);
    }

    // Надеть первую подходящую броню в пустой слот.
    equipArmor() {
        const bot = this.bot;
        if (this.handsBusy) return;
        const patterns = { head: /helmet/, torso: /chestplate/, legs: /leggings/, feet: /boots/ };
        for (const slot of ARMOR_SLOTS) {
            if (bot.inventory.slots[bot.getEquipmentDestSlot(slot)]) continue; // уже надето
            const piece = this.findItem((it) => isArmor(it) && patterns[slot].test(it.name));
            if (piece) {
                this.runHands(() => bot.equip(piece, slot), HANDS_TIMEOUT_MS);
                return;
            }
        }
    }

    // Выбросить то, что в руке.
    dropItem() {
        const bot = this.bot;
        if (this.handsBusy || !bot.heldItem) return;
        const item = bot.heldItem;
        this.runHands(() => bot.tossStack(item), HANDS_TIMEOUT_MS);
    }

    attackOrDigCenter() {
        const bot = this.bot;

        // Удар по сущности — мгновенный, и долгое дело рук его не блокирует:
        // иначе водящий в салках, начав копать или есть, стоял вплотную к
        // убегающему и "не бил" (заметил автор). Начатую копку удар бросает —
        // как в самой игре.
        const entity = this.findEntityInCrosshair();
        if (entity) {
            if (this.digging) {
                this.digging = null;
                bot.stopDigging();
            }
            // Бьёт по тому, что впереди (до ~30° влево-вправо, по высоте —
            // любое), — пусть и смотрит на того, кого бьёт.
            bot.lookAt(entity.position.offset(0, (entity.height || 1) * 0.5, 0), true);
            this.lastAttackCharge = this.attackCharge(); // с какой силой бьёт — до сброса заряда
            this.lastAttackCrit = this.lastAttackCharge > 0.9 && !bot.entity.onGround
                && (bot.entity.velocity?.y ?? 0) < 0 && !bot.getControlState?.('sprint');
            this.lastAttackAt = Date.now();
            bot.attack(entity);
            this.attackedId = entity.id;
            return;
        }
        if (this.handsBusy) return;

        const hit = centerRaycast(bot.world, bot.entity, this.config);
        if (!hit || !hit.block || hit.block.name === 'air') return;
        // Луч прицела бьёт на всю дальность зрения (32 блока), а копать
        // сервер даст только вблизи — дальний блок даже не пробуем.
        if (hit.distance > DIG_REACH) return;
        if (!bot.canDigBlock || !bot.canDigBlock(hit.block)) return;

        // Если копка почему-то не завершилась (блок сломал кто-то другой,
        // сервер не ответил) — через DIG_TIMEOUT_MS бросаем её. А если бот
        // отвернулся или отошёл — её прервёт checkDigging на следующем тике.
        const block = hit.block;
        this.digging = block.position.clone();
        this.runHands(async () => {
            try {
                await bot.dig(block, true); // true — голова на блок сразу, а не плавно
            } finally {
                this.digging = null;
            }
        }, DIG_TIMEOUT_MS, () => bot.stopDigging());
    }

    // Копать можно, только пока блок в прицеле и рядом — как в игре:
    // отвернулся или отошёл — копка прерывается. Сам mineflayer, начав,
    // докапывает блок, куда бы бот ни ушёл, — и боты ломали блоки издалека
    // (заметил автор вживую).
    checkDigging() {
        if (!this.digging) return;
        const hit = centerRaycast(this.bot.world, this.bot.entity, this.config);
        const stillAiming = hit && hit.block && hit.block.position.equals(this.digging) && hit.distance <= DIG_REACH;
        if (stillAiming) return;
        this.digging = null;
        this.bot.stopDigging(); // промис копки отклонится, руки освободит runHands
    }

    // Ближайшая допустимая сущность перед ботом в пределах ATTACK_RANGE: по
    // горизонтали — не дальше ~30° от направления взгляда, по вертикали —
    // любая (удар сам наводит голову: bot.lookAt в executeHands). Раньше
    // конус считался от взгляда с наклоном, а взгляд у бегущих обычно ровный
    // (walking штрафует голову за наклон): центр убегающего ближе 2 блоков
    // уходил ниже конуса, и водящие стояли вплотную и "не попадали" (автор,
    // 2026-09-26). Возвращает null — тогда attack_center копает блок.
    findEntityInCrosshair() {
        const bot = this.bot;
        const eyePos = eyePosition(bot.entity);
        const forward = lookDirection({ yaw: bot.entity.yaw, pitch: 0 }); // только горизонталь

        let best = null;
        let bestCos = ATTACK_CONE_COS;

        for (const entity of Object.values(bot.entities)) {
            if (!entity || entity === bot.entity) continue;
            if (!ATTACKABLE_ENTITY_NAMES.has(entity.name) && !this.isTaggable(entity)) continue;

            const entityCenter = entity.position.offset(0, (entity.height || 1) * 0.5, 0);
            const toEntity = entityCenter.minus(eyePos);
            const distance = toEntity.norm();
            if (distance > ATTACK_RANGE || distance < 1e-6) continue;

            // Угол влево-вправо; стоит почти вплотную по горизонтали (прямо
            // над или под ботом) — считаем, что впереди.
            const flat = Math.hypot(toEntity.x, toEntity.z);
            const cos = flat < 0.3 ? 1 : (toEntity.x * forward.x + toEntity.z * forward.z) / flat;
            if (cos <= bestCos) continue;
            // Сквозь стену не бьём: нужна прямая видимость до центра или до
            // головы (из-за невысокого уступа бывает виден только верх).
            const head = entity.position.offset(0, (entity.height || 1) * 0.9, 0);
            if (!lineOfSight(bot.world, eyePos, entityCenter) && !lineOfSight(bot.world, eyePos, head)) continue;
            bestCos = cos;
            best = entity;
        }

        return best;
    }

    holdMove(move) {
        for (const [control, state] of Object.entries(move)) {
            this.bot.setControlState(control, state);
        }
        this.activeMove = move;
        this.activeUntil = Date.now() + ACTION_DURATION_MS;
    }

    // Вызывается каждый тик: если время действия истекло — отпустить клавиши;
    // копаем то, что уже не в прицеле, — бросить.
    tick() {
        if (this.activeMove && Date.now() >= this.activeUntil) {
            // Клавиши движения отпускаем, а шифт держим до следующего решения
            // (как плагин): ответ Python запоздал — бот у края стоит крадучись.
            const sneaking = this.bot.getControlState('sneak');
            this.stopMovement();
            if (sneaking) this.bot.setControlState('sneak', true);
        }
        this.checkDigging();
        // Руки свободны полсекунды — оружие обратно в руку (как плагин,
        // Actions.weaponToHand): для постройки макросы берут блоки, бить надо
        // мечом; заранее, а не в миг удара — смена предмета обнуляет заряд.
        this.idleHandsTicks = this.handsBusy ? 0 : (this.idleHandsTicks ?? 0) + 1;
        if (this.idleHandsTicks === WEAPON_BACK_DECISIONS) this.weaponToHand();
    }

    weaponToHand() {
        const bot = this.bot;
        const rank = (item) => (!item ? -1 : /_sword$/.test(item.name) ? 4 : /_axe$/.test(item.name) ? 3
            : item.name === 'trident' ? 2 : item.name === 'mace' ? 1 : -1);
        let best = null;
        for (const item of bot.inventory.items()) {
            if (rank(item) > rank(best ?? bot.heldItem)) best = item;
        }
        if (best && rank(best) > rank(bot.heldItem)) bot.equip(best, 'hand').catch(() => {});
    }

    stopMovement() {
        const bot = this.bot;
        bot.clearControlStates();
        this.activeMove = null;
        this.activeUntil = 0;
    }
}

// Каналы и их действия — контракт с CHANNELS в py/protocol.py. Порядок
// внутри канала важен Python'у (индексы выходов сети), здесь — только имена.
const CHANNELS = {
    legs: [
        'idle',
        'walk_forward',
        'sprint_forward',
        'walk_back',
        'strafe_left',
        'strafe_right',
        'jump_forward',
        'turn_left',
        'turn_right',
        'jump',
        'sneak_back',
        'sneak',
    ],
    head: [
        'head_idle',
        'look_up',
        'look_down',
        'look_left',
        'look_right',
    ],
    hands: [
        'hands_idle',
        'attack_center',
        'place_below',
        'place_front',
        'use_item',
        'equip_armor',
        'drop_item',
    ],
};

const ACTION_NAMES = Object.values(CHANNELS).flat();

// Шаги поворота — контракт с ACTION_ROTATION в py/protocol.py.
module.exports = { ActionExecutor, ACTION_NAMES, CHANNELS, TURN_STEP, LOOK_STEP, LOOK_YAW_STEP };
