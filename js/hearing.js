// "Слух" бота — низкоприоритетный круговой сенсор. Minecraft генерирует
// звуковые события (шаги, разрушение блоков, мобы...), и у каждого есть
// мировая позиция источника. Мы делим круг вокруг бота на N секторов и
// кладём в сектор источника "заряд", который затухает со временем.
//
// Выход — вектор длиной `sectors`, значения 0..1. Отсутствие звуков = нули.
// Сеть получает его как отдельный маленький вход наряду с raycast-зрением.

const DECAY_PER_TICK = 0.85;   // множитель затухания заряда за тик
const MAX_CHARGE = 1.0;

class Hearing {
    constructor(bot, config) {
        this.bot = bot;
        this.sectors = config.hearing.sectors;
        this.radius = config.hearing.radius;
        this.charges = new Float32Array(this.sectors);
        this.lastYaw = 0;

        // soundEffectHeard: (soundName, position, volume, pitch)
        bot.on('soundEffectHeard', (soundName, position, volume) => {
            this.addSound(soundName, position, volume);
        });
        // entitySoundHeard: звук от конкретной сущности (шаги игрока и т.п.)
        bot.on('entitySoundHeard', (soundName, entity, volume) => {
            if (entity && entity.position) {
                this.addSound(soundName, entity.position, volume);
            }
        });
    }

    addSound(soundName, position, volume) {
        const botPos = this.bot.entity.position;
        const dx = position.x - botPos.x;
        const dz = position.z - botPos.z;
        const distance = Math.hypot(dx, dz);
        if (distance > this.radius) return; // слишком далеко — не слышим

        // Угол от бота к источнику в мировых координатах (atan2 в системе,
        // где 0 = +Z, растёт к +X), переводим в 0..2pi.
        const worldAngle = Math.atan2(dx, dz);
        const twoPi = Math.PI * 2;
        const normalized = ((worldAngle % twoPi) + twoPi) % twoPi;

        const sector = Math.min(
            Math.floor((normalized / twoPi) * this.sectors),
            this.sectors - 1
        );

        // Сила звука: громкость события * близость (линейно от края радиуса к боту).
        const proximity = 1.0 - distance / this.radius;
        const strength = Math.max(0, Math.min(volume, 1.0)) * proximity;
        this.charges[sector] = Math.min(this.charges[sector] + strength, MAX_CHARGE);
    }

    // Вызывается раз в тик контроллера. referencePos/yaw — точка и взгляд
    // той сущности, "чьи уши" сейчас читаются (у бота — свои, у обсёрвера —
    // игрока, которого он записывает). Возвращает секторный вектор,
    // повернутый в систему координат смотрящего (сектор 0 = прямо перед ним),
    // чтобы сигнал был инвариантен к абсолютному yaw.
    tick(referencePos, yaw) {
        // Затухание.
        for (let i = 0; i < this.charges.length; i++) {
            this.charges[i] *= DECAY_PER_TICK;
            if (this.charges[i] < 0.01) this.charges[i] = 0;
        }

        // Угол сектора отсчитывается от +Z к +X, а взгляд в mineflayer —
        // (-sin yaw, -cos yaw): его угол в той же системе = yaw + pi (оба
        // растут в одну сторону). Сдвиг на него переводит мировые сектора в
        // локальные. (Раньше тут был просто yaw — "перед" был сзади.)
        const facing = yaw + Math.PI;
        const yawNormalized = ((facing % (Math.PI * 2)) + Math.PI * 2) % (Math.PI * 2);
        const shift = Math.round((yawNormalized / (Math.PI * 2)) * this.sectors);

        const result = new Array(this.sectors);
        for (let i = 0; i < this.sectors; i++) {
            // Сектор 0 результата — прямо по взгляду: из мирового сектора
            // (i + shift) mod sectors.
            result[i] = Math.round(this.charges[(i + shift) % this.sectors] * 1000) / 1000;
        }
        return result;
    }
}

module.exports = { Hearing };
