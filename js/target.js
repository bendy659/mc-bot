// Управление целью бота. Есть два источника цели:
//   - человек через чат (!setTarget ...) — как только человек её задаёт,
//     она главнее обучающего модуля и остаётся такой, пока человек не
//     скажет !setTarget none;
//   - обучающий модуль на стороне Python (сообщение {"type":"set_target"}
//     через ZMQ-канал действий) — работает, только пока человек не
//     перехватил управление.
// Python использует итоговую (см. getTargetPosition) цель для награды.

// Сколько помнить последнюю позицию игрока-цели, когда он пропал из
// видимости. Сервер присылает боту сущности игроков только в радиусе
// нескольких десятков блоков (вживую — ~35), и бот, отошедший дальше, терял
// цель совсем. С памятью он смотрит/идёт туда, где видел игрока последний
// раз, — и, подойдя, снова его видит.
const PLAYER_MEMORY_MS = 30000;

class TargetManager {
    constructor(bot) {
        this.bot = bot;
        this.humanTarget = null;    // { kind: 'player'|'position', ... }
        this.lastSeen = null;       // { position, height, at } — игрок-цель, когда видели в последний раз
        this.humanOverride = false; // true, пока человек держит цель на себе
        this.modulePosition = null; // {x,y,z} | null — последняя цель от модуля
        this.moduleEntityId = null; // id сущности, если модуль "прицепил" цель к ней (follow/looking)
    }

    // true, если сейчас цель под управлением человека — модуль должен
    // воздержаться от собственных попыток её сменить.
    isHumanControlled() {
        return this.humanOverride;
    }

    // Вызывается из bot.js при {"type":"set_target", position} от Python.
    // Если человек сейчас держит цель на себе, значение всё равно
    // запоминается (пригодится, когда человек отпустит цель), но не
    // используется в getTargetPosition до этого момента.
    //
    // entityId — цель-сущность (follow/looking выбирают моба или игрока):
    // её позиция берётся заново каждый тик из bot.entities, а не
    // пересылается из Python, поэтому цель не отстаёт от движущейся сущности.
    // position вместе с entityId — запасная позиция: если сущности бот не
    // видит (сервер показывает сущности только поблизости), цель — там
    // (салки: судья знает, где убегающий, из его собственного состояния).
    setModuleTarget(position, entityId = null) {
        this.modulePosition = position || null;
        this.moduleEntityId = entityId ?? null;
    }

    // Сущность текущей цели (или null, если цель — просто точка).
    getTargetEntity() {
        if (this.humanOverride) {
            if (this.humanTarget && this.humanTarget.kind === 'player') {
                const player = this.bot.players[this.humanTarget.username];
                return (player && player.entity) || null;
            }
            return null;
        }
        if (this.moduleEntityId == null) return null;
        const entity = this.bot.entities[this.moduleEntityId];
        return (entity && entity.isValid) ? entity : null;
    }

    // Возвращает позицию цели {x,y,z} или null (цели нет / игрок оффлайн /
    // модуль пока ничего не выбрал).
    getTargetPosition() {
        if (this.humanOverride) {
            if (!this.humanTarget) return null;
            if (this.humanTarget.kind === 'player') {
                const player = this.bot.players[this.humanTarget.username];
                if (player && player.entity) {
                    this.lastSeen = { position: player.entity.position.clone(), height: player.entity.height, at: Date.now() };
                    return player.entity.position;
                }
                return this.recentlySeen() ? this.lastSeen.position : null;
            }
            if (this.humanTarget.kind === 'position') {
                return this.humanTarget.position;
            }
            return null;
        }
        if (this.moduleEntityId != null) {
            const entity = this.getTargetEntity();
            // Сущность пропала: есть запасная позиция — туда, иначе цели нет
            // (модуль выберет новую).
            return entity ? entity.position : this.modulePosition;
        }
        return this.modulePosition;
    }

    recentlySeen() {
        return this.lastSeen != null && Date.now() - this.lastSeen.at < PLAYER_MEMORY_MS;
    }

    // Рост цели-сущности (0 для точки) — модули целятся в "лицо", а не в
    // ноги. Для игрока, которого сейчас не видно, — рост из памяти.
    getTargetHeight() {
        const entity = this.getTargetEntity();
        if (entity) return entity.height ?? 1.8;
        if (this.humanOverride && this.humanTarget?.kind === 'player' && this.recentlySeen()) return this.lastSeen.height ?? 1.8;
        return 0;
    }

    // Человекочитаемое описание для чата/логов.
    describe() {
        if (this.humanOverride) {
            if (!this.humanTarget) return 'none';
            const pos = this.getTargetPosition();
            const remembered = this.humanTarget.kind === 'player' && !this.getTargetEntity() && pos;
            const where = pos
                ? ` @ (${pos.x.toFixed(1)}, ${pos.y.toFixed(1)}, ${pos.z.toFixed(1)})${remembered ? ' — по памяти, сейчас не видна' : ''}`
                : ' (не видна)';
            if (this.humanTarget.kind === 'player') return `player ${this.humanTarget.username}${where}`;
            if (this.humanTarget.kind === 'position') return `position${where}`;
            return this.humanTarget.kind;
        }
        if (this.moduleEntityId != null) {
            const entity = this.getTargetEntity();
            const who = entity ? (entity.username ?? entity.name ?? 'сущность') : 'пропала';
            return `module: сущность ${who}`;
        }
        if (!this.modulePosition) return 'module: none';
        const p = this.modulePosition;
        return `module @ (${p.x.toFixed(1)}, ${p.y.toFixed(1)}, ${p.z.toFixed(1)})`;
    }

    // Обработка строки чата с командой цели. Возвращает ответ для чата.
    handleCommand(args) {
        const kind = args[0];
        if (kind === 'none') {
            this.humanTarget = null;
            this.humanOverride = false; // отдаём управление обратно модулю
            return 'Цель сброшена, управление у обучающего модуля.';
        }
        if (kind === 'player') {
            const username = args[1];
            if (!username) return 'Использование: !setTarget player <username>';
            this.humanTarget = { kind: 'player', username };
            this.lastSeen = null;
            this.humanOverride = true;
            return `Цель: игрок ${username}.`;
        }
        if (kind === 'position') {
            const [x, y, z] = args.slice(1).map(Number);
            if (![x, y, z].every((v) => Number.isFinite(v))) {
                return 'Использование: !setTarget position <x> <y> <z>';
            }
            this.humanTarget = { kind: 'position', position: { x, y, z } };
            this.humanOverride = true;
            return `Цель: позиция (${x}, ${y}, ${z}).`;
        }
        return 'Использование: !setTarget player <имя> | position <x y z> | none';
    }
}

module.exports = { TargetManager };
