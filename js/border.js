// Слежение за границей мира (world border). Mineflayer сам её не знает
// (в API есть только косвенные признаки вроде worldBorderHit у предметов),
// поэтому читаем сырые пакеты сервера и запоминаем центр + размер.
//
// Зачем: за границей мира бот получает урон ("left the confines of this
// world") и умирает. Модель имеет полное право выбрать walk_forward в
// сторону края — охранник в bot.js перехватывает такое движение до того,
// как оно станет смертельным, а walking выбирает цели внутри границы.
//
// Пакеты и их поля за годы менялись — разбираем все известные варианты
// (проверено по minecraft-data, в т.ч. для 26.1):
//   1.17+: initialize_world_border {x, z, oldDiameter, newDiameter, ...},
//          world_border_center {x, z}, world_border_size {diameter},
//          world_border_lerp_size {oldDiameter, newDiameter, speed};
//   до 1.17: один пакет world_border с полем action и полями x, z,
//          radius / old_radius / new_radius (по смыслу это тоже диаметр).
// Раньше здесь слушался несуществующий 'initialize_border' и читались
// поля centerX/newSize — граница НИКОГДА не становилась известной, и
// охранник не срабатывал (боты уходили за край и умирали).

class WorldBorder {
    constructor(bot) {
        this.bot = bot;
        this.size = null; // диаметр границы в блоках
        this.centerX = 0;
        this.centerZ = 0;
        this.announced = false;

        const on = (name, handler) => bot._client.on(name, (packet) => {
            handler(packet);
            this.announce();
        });

        on('initialize_world_border', (packet) => {
            this.setCenter(packet);
            this.setSize(packet);
        });
        on('world_border_center', (packet) => this.setCenter(packet));
        on('world_border_size', (packet) => this.setSize(packet));
        on('world_border_lerp_size', (packet) => this.setSize(packet));
        on('world_border', (packet) => {
            this.setCenter(packet);
            this.setSize(packet);
        });
    }

    setCenter(packet) {
        const x = packet.x ?? packet.centerX;
        const z = packet.z ?? packet.centerZ;
        if (Number.isFinite(x)) this.centerX = x;
        if (Number.isFinite(z)) this.centerZ = z;
    }

    // Пока граница плавно меняет размер (lerp: old -> new за speed мс),
    // берём МЕНЬШИЙ из двух: при сжатии бот должен уйти от края заранее.
    setSize(packet) {
        const candidates = [
            packet.diameter, packet.newDiameter, packet.oldDiameter,
            packet.radius, packet.new_radius, packet.old_radius,
        ].filter((value) => Number.isFinite(value) && value > 0);
        if (candidates.length > 0) this.size = Math.min(...candidates);
    }

    announce() {
        if (this.announced || !this.known) return;
        this.announced = true;
        console.log(`[border] ${this.bot.username}: граница мира — центр (${this.centerX.toFixed(1)}, `
            + `${this.centerZ.toFixed(1)}), размер ${this.size.toFixed(0)} блоков.`);
    }

    get known() {
        return this.size != null && this.size > 0;
    }

    // Запретная зона: всё, что ближе `margin` блоков к границе изнутри
    // (и всё снаружи). margin оставляет запас на инерцию/лаг позиции.
    isUnsafe(position, margin) {
        if (!this.known) return false;
        const half = this.size / 2;
        const dx = Math.abs(position.x - this.centerX);
        const dz = Math.abs(position.z - this.centerZ);
        return dx > half - margin || dz > half - margin;
    }
}

module.exports = { WorldBorder };
