// Загрузка и валидация единого config.json из корня проекта.
// Оба конца системы (Node и Python) читают один и тот же файл,
// поэтому рассинхрон настроек (например, разрешения сетки зрения)
// между сторонами невозможен по построению.

const fs = require('fs');
const path = require('path');

// MCBOT_CONFIG — другой config.json (тестовый рой рядом с рабочим: свои
// порты ZMQ и сервер). Обычно не задан — config.json в корне проекта.
const CONFIG_PATH = process.env.MCBOT_CONFIG || path.join(__dirname, '..', 'config.json');

// Дефолты на случай отсутствия ключа в config.json. Значения взяты
// из "эталонной" конфигурации MVP: 8x8 лучей, дальность 2 чанка.
const DEFAULTS = {
    bot: { host: 'localhost', port: 25565, username: 'AI_Bot', version: false, count: 1, separator: '_', body: 'mineflayer', names: [], task_worlds: {} },
    fov: 70,
    vision: { mode: 'circular', vertical_fov: 70, resolution: [8, 8], distance: 2, tick_delay: 2, block_classes: 10 },
    entities: { max_tracked: 8, radius: 16 },
    hearing: { enabled: true, sectors: 8, radius: 16 },
    zmq: { node_to_py: 5555, py_to_node: 5556 },
    train: { tick_rate_ms: 150 },
    route: { enabled: true, replan_ticks: 5, max_nodes: 3000, max_drop: 3, lookahead: 3, max_distance: 48 },
    reward: {
        reach_goal: 100, closer: 1.0, idle: -1.0, death: -100, goal_radius: 2.0
    }
};

function deepMerge(base, override) {
    const result = { ...base };
    for (const key of Object.keys(override || {})) {
        if (
            base[key] && typeof base[key] === 'object' && !Array.isArray(base[key]) &&
            override[key] && typeof override[key] === 'object' && !Array.isArray(override[key])
        ) {
            result[key] = deepMerge(base[key], override[key]);
        } else {
            result[key] = override[key];
        }
    }
    return result;
}

function loadConfig() {
    let raw;
    try {
        raw = fs.readFileSync(CONFIG_PATH, 'utf8');
    } catch (err) {
        console.error(`[config] Не могу прочитать ${CONFIG_PATH}: ${err.message}`);
        console.error('[config] Использую значения по умолчанию.');
        return structuredClone(DEFAULTS);
    }

    let parsed;
    try {
        parsed = JSON.parse(raw);
    } catch (err) {
        console.error(`[config] Ошибка разбора JSON: ${err.message}`);
        process.exit(1);
    }

    const config = deepMerge(DEFAULTS, parsed);
    validate(config);
    return config;
}

function validate(config) {
    const problems = [];
    const [resX, resY] = config.vision.resolution;
    if (!Number.isInteger(resX) || !Number.isInteger(resY) || resX < 1 || resY < 1) {
        problems.push(`vision.resolution должен быть двумя целыми >= 1, получено [${resX}, ${resY}]`);
    }
    if (config.fov <= 0 || config.fov >= 180) {
        problems.push(`fov должен быть в (0, 180), получено ${config.fov}`);
    }
    if (config.vision.distance < 1) {
        problems.push(`vision.distance должен быть >= 1 чанка, получено ${config.vision.distance}`);
    }
    if (config.hearing.sectors < 1) {
        problems.push(`hearing.sectors должен быть >= 1, получено ${config.hearing.sectors}`);
    }
    if (problems.length > 0) {
        console.error('[config] Некорректный config.json:');
        for (const p of problems) console.error(`  - ${p}`);
        process.exit(1);
    }
}

module.exports = { loadConfig, CONFIG_PATH };
