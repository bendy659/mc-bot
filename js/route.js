// Поиск маршрута по блокам (A*) — "куда идти в обход", а не "по прямой".
//
// Зачем: сеть ходьбы умеет идти туда, куда указывает цель (и реагировать
// на препятствие прямо перед собой), но карты у неё нет и памяти — три
// кадра: обойти обрыв, овраг или стену она не может в принципе — правильный
// путь сначала УВОДИТ от цели, и награда "за каждый блок ближе" за это
// наказывала. Вживую боты шли к игроку напрямую и падали в овраг.
//
// Разделение труда: маршрут считает этот планировщик, а ИДЁТ по нему
// по-прежнему сеть — модули walking/follow подменяют ей цель на ближайшую
// точку маршрута (state.route.waypoint) и награждают за продвижение по
// маршруту (state.route.length). Шаги, прыжки, повороты, реакция на
// препятствия — всё ещё выученное поведение сети.
//
// Сетка: клетка (x, y, z) — место, где могут стоять ноги: под ней твёрдый
// блок, в ней и над ней — проходимо (воздух, трава...). Шаги: 8 соседей на
// том же уровне, запрыгнуть на 1 блок вверх, спрыгнуть до max_drop блоков
// (дальше — урон от падения). Вода, лава, огонь, паутина и т.п. — непроходимы
// (боты тонули).

const DANGEROUS = /lava|fire|magma|cactus|campfire|sweet_berry|cobweb|powder_snow|wither_rose|pointed_dripstone/;
const LIQUID = /water|lava|bubble_column|seagrass|kelp/;

// Куча с минимумом по f — открытое множество A*.
class MinHeap {
    constructor() { this.items = []; }
    get size() { return this.items.length; }
    push(node) {
        const items = this.items;
        items.push(node);
        let i = items.length - 1;
        while (i > 0) {
            const parent = (i - 1) >> 1;
            if (items[parent].f <= items[i].f) break;
            [items[parent], items[i]] = [items[i], items[parent]];
            i = parent;
        }
    }
    pop() {
        const items = this.items;
        const top = items[0];
        const last = items.pop();
        if (items.length > 0) {
            items[0] = last;
            let i = 0;
            for (;;) {
                const left = 2 * i + 1;
                const right = left + 1;
                let smallest = i;
                if (left < items.length && items[left].f < items[smallest].f) smallest = left;
                if (right < items.length && items[right].f < items[smallest].f) smallest = right;
                if (smallest === i) break;
                [items[smallest], items[i]] = [items[i], items[smallest]];
                i = smallest;
            }
        }
        return top;
    }
}

const key = (x, y, z) => `${x},${y},${z}`;

class RoutePlanner {
    // world — bot.world (синхронный), registry — bot.registry.
    constructor(world, registry, config = {}) {
        this.world = world;
        this.registry = registry;
        this.maxNodes = config.max_nodes ?? 3000;
        this.maxDrop = config.max_drop ?? 3;
        this.lookahead = config.lookahead ?? 3;
        this.replanTicks = config.replan_ticks ?? 5;
        this.maxDistance = config.max_distance ?? 48;
        this.kindCache = new Map(); // stateId -> 'solid' | 'passable' | 'blocked'

        this.path = null;       // [{x, y, z}] — клетки от старта до цели
        this.remaining = null;  // длина маршрута от клетки i до конца
        this.complete = false;  // дошли ли в поиске до самой цели
        this.index = 0;
        this.ticksSincePlan = 0;
        this.plannedTarget = null;
    }

    // Что это за блок для ходьбы: твёрдый пол / проходимо / нельзя.
    kind(x, y, z) {
        const stateId = this.world.getBlockStateId({ x, y, z });
        if (stateId == null) return 'blocked'; // чанк не загружен — не знаем, не идём
        let result = this.kindCache.get(stateId);
        if (result) return result;
        const block = this.registry.blocksByStateId[stateId];
        if (!block) {
            result = 'blocked';
        } else if (DANGEROUS.test(block.name) || LIQUID.test(block.name)) {
            result = 'blocked';
        } else if (block.boundingBox === 'empty') {
            result = 'passable';
        } else {
            result = 'solid';
        }
        this.kindCache.set(stateId, result);
        return result;
    }

    standable(x, y, z) {
        return this.kind(x, y - 1, z) === 'solid' && this.kind(x, y, z) === 'passable' && this.kind(x, y + 1, z) === 'passable';
    }

    // Ближайшая клетка, где можно стоять, в колонке под точкой (бот в
    // прыжке, цель на ступеньке): до 3 блоков вниз.
    groundCell(position) {
        const x = Math.floor(position.x);
        const z = Math.floor(position.z);
        const top = Math.floor(position.y + 0.3);
        for (let y = top; y >= top - 3; y--) {
            if (this.standable(x, y, z)) return { x, y, z };
        }
        return null;
    }

    neighbours(node) {
        const out = [];
        const { x, y, z } = node;
        for (let dx = -1; dx <= 1; dx++) {
            for (let dz = -1; dz <= 1; dz++) {
                if (dx === 0 && dz === 0) continue;
                const diagonal = dx !== 0 && dz !== 0;
                const nx = x + dx;
                const nz = z + dz;
                // По диагонали — только если оба боковых прохода свободны
                // (иначе бот цепляет угол и застревает).
                if (diagonal && !(this.clear(x + dx, y, z) && this.clear(x, y, z + dz))) continue;

                if (this.standable(nx, y, nz)) {
                    out.push({ x: nx, y, z: nz, step: diagonal ? Math.SQRT2 : 1, cost: diagonal ? Math.SQRT2 : 1 });
                    continue;
                }
                if (diagonal) continue; // прыжки и спуски — только по прямой

                // Запрыгнуть на блок: над головой должно быть место для прыжка.
                if (this.standable(nx, y + 1, nz) && this.kind(x, y + 2, z) === 'passable') {
                    out.push({ x: nx, y: y + 1, z: nz, step: Math.SQRT2, cost: 2 });
                    continue;
                }

                // Спрыгнуть: колонка над местом приземления свободна.
                if (!this.clear(nx, y, nz)) continue;
                for (let drop = 1; drop <= this.maxDrop; drop++) {
                    if (this.standable(nx, y - drop, nz)) {
                        out.push({ x: nx, y: y - drop, z: nz, step: Math.hypot(1, drop), cost: 1 + 0.5 * drop });
                        break;
                    }
                    if (this.kind(nx, y - drop, nz) !== 'passable') break;
                }
            }
        }
        return out;
    }

    clear(x, y, z) {
        return this.kind(x, y, z) === 'passable' && this.kind(x, y + 1, z) === 'passable';
    }

    // A* от start до клетки рядом с goal. Возвращает { path, complete } или
    // null. Если до цели не дошли (далеко, нет пути) — путь до исследованной
    // клетки, ближайшей к цели: лучше, чем ничего.
    plan(start, goal) {
        // Оценка остатка пути — "октильная" (8 соседей: прямо — 1, по
        // диагонали — √2) плюс по полблока за уровень высоты: на ровном месте
        // она точная, и A* идёт почти прямо к цели. Прямая (евклидова)
        // занижала путь, и поиск разбирал сотни лишних клеток — до 50 мс на
        // маршрут; с дюжиной ботов тик роя растягивался вдвое (автор,
        // 2026-09-26: "начинают идти с задержками"). Не завышает (прыжок стоит
        // 2, спуск — 1 + 0.5 на уровень), так что маршрут по-прежнему кратчайший.
        const h = (n) => {
            const dx = Math.abs(n.x - goal.x);
            const dz = Math.abs(n.z - goal.z);
            return Math.max(dx, dz) + (Math.SQRT2 - 1) * Math.min(dx, dz) + 0.5 * Math.abs(n.y - goal.y);
        };
        const open = new MinHeap();
        const best = new Map();
        const parents = new Map();
        const startKey = key(start.x, start.y, start.z);
        best.set(startKey, 0);
        open.push({ ...start, g: 0, f: h(start), key: startKey });
        let closest = { node: start, h: h(start) };
        let expanded = 0;

        while (open.size > 0 && expanded < this.maxNodes) {
            const node = open.pop();
            if (node.g > best.get(node.key)) continue; // устаревшая запись
            expanded++;
            const distance = h(node);
            if (distance < closest.h) closest = { node, h: distance };
            if (Math.hypot(node.x - goal.x, node.z - goal.z) <= 1.5 && Math.abs(node.y - goal.y) <= 1) {
                return { path: this.unwind(parents, node), complete: true };
            }
            for (const next of this.neighbours(node)) {
                if (Math.hypot(next.x - start.x, next.z - start.z) > this.maxDistance) continue;
                const nextKey = key(next.x, next.y, next.z);
                const g = node.g + next.cost;
                if (g >= (best.get(nextKey) ?? Infinity)) continue;
                best.set(nextKey, g);
                parents.set(nextKey, { x: node.x, y: node.y, z: node.z, step: next.step, key: node.key });
                open.push({ x: next.x, y: next.y, z: next.z, g, f: g + h(next), key: nextKey });
            }
        }
        return { path: this.unwind(parents, closest.node), complete: false };
    }

    unwind(parents, node) {
        const path = [{ x: node.x, y: node.y, z: node.z, step: 0 }];
        let current = key(node.x, node.y, node.z);
        while (parents.has(current)) {
            const parent = parents.get(current);
            path[0].step = parent.step; // длина шага ИЗ родителя в эту клетку
            path.unshift({ x: parent.x, y: parent.y, z: parent.z, step: 0 });
            current = parent.key;
        }
        return path;
    }

    // Вызывается каждый тик: пересчитывает маршрут, когда пора, и отдаёт
    // { waypoint: {x,y,z}, length, complete } или null (цели нет / стоять
    // негде / маршрута нет). length — сколько идти ПО МАРШРУТУ до цели.
    update(position, target) {
        if (!target) {
            this.path = null;
            return null;
        }
        const targetMoved = !this.plannedTarget ||
            Math.hypot(target.x - this.plannedTarget.x, target.z - this.plannedTarget.z) > 2 ||
            Math.abs(target.y - this.plannedTarget.y) > 1.5;
        const deviated = this.path && this.distanceTo(this.path[this.index], position) > 2.5;
        this.ticksSincePlan++;
        if (!this.path || targetMoved || deviated || this.ticksSincePlan >= this.replanTicks) {
            this.replan(position, target);
        }
        if (!this.path || this.path.length === 0) return null;

        // Где мы на маршруте: ближайшая клетка чуть впереди прошлой.
        let bestIndex = this.index;
        let bestDistance = Infinity;
        const last = Math.min(this.path.length - 1, this.index + 6);
        for (let i = this.index; i <= last; i++) {
            const d = this.distanceTo(this.path[i], position);
            if (d < bestDistance) {
                bestDistance = d;
                bestIndex = i;
            }
        }
        this.index = bestIndex;
        const waypointNode = this.path[Math.min(this.index + this.lookahead, this.path.length - 1)];
        let length = bestDistance + this.remaining[this.index];
        if (!this.complete) {
            // Маршрут до цели не дотянулся (далеко или пути нет): остаток — по
            // прямой от конца маршрута. Иначе "длина" мерила бы только найденный
            // кусок пути, и награда за прогресс скакала бы при каждом пересчёте.
            const end = this.path[this.path.length - 1];
            length += Math.hypot(end.x + 0.5 - target.x, end.y - target.y, end.z + 0.5 - target.z);
        }
        return {
            waypoint: { x: waypointNode.x + 0.5, y: waypointNode.y, z: waypointNode.z + 0.5 },
            length: Math.round(length * 100) / 100,
            complete: this.complete,
        };
    }

    replan(position, target) {
        this.ticksSincePlan = 0;
        this.plannedTarget = { x: target.x, y: target.y, z: target.z };
        const start = this.groundCell(position);
        const goal = this.groundCell(target) ?? { x: Math.floor(target.x), y: Math.floor(target.y), z: Math.floor(target.z) };
        if (!start) {
            // В воздухе/в воде — стоять негде: оставляем старый маршрут, если он есть.
            return;
        }
        const result = this.plan(start, goal);
        this.path = result.path;
        this.complete = result.complete;
        this.index = 0;
        // Длина маршрута от каждой клетки до конца — для награды за продвижение.
        this.remaining = new Array(this.path.length).fill(0);
        for (let i = this.path.length - 2; i >= 0; i--) {
            this.remaining[i] = this.remaining[i + 1] + this.path[i + 1].step;
        }
    }

    distanceTo(node, position) {
        return Math.hypot(node.x + 0.5 - position.x, node.y - position.y, node.z + 0.5 - position.z);
    }
}

module.exports = { RoutePlanner };
