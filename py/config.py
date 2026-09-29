"""Загрузка единого config.json из корня проекта.

Node-сторона (js/config.js) читает тот же файл, поэтому размеры наблюдений
и пространства действий всегда согласованы между сторонами.
"""

import json
import os
from pathlib import Path

# MCBOT_CONFIG — другой config.json (тестовый рой рядом с рабочим: свои
# порты ZMQ и сервер). Обычно не задан — config.json в корне проекта.
CONFIG_PATH = Path(os.environ.get("MCBOT_CONFIG") or Path(__file__).resolve().parent.parent / "config.json")

# Дефолты — те же, что в js/config.js.
DEFAULTS = {
    "bot": {
        "host": "localhost",
        "port": 25565,
        "username": "AI_Bot",
        "version": False,
        "count": 1,
        "separator": "_",
        # Чем боты ходят по миру: "mineflayer" — клиенты js/bot.js (Node),
        # "plugin" — игроки сервера из плагина plugin/ (Paper; лаунчер не
        # запускает Node, а шлёт серверу /mcbot start по RCON).
        "body": "mineflayer",
        # Имена ботов для людей: "Егор [AI_1]" в чате, списке игроков и над
        # головой — случайные, без повторов (показывает плагин; ник — AI_N).
        "names": [],
        # Миры задачек (плагин создаёт их сам, пустыми): задачка -> имя мира.
        # Обычный мир под задачки не перестраивается (автор, 2026-09-28).
        "task_worlds": {},
    },
    "fov": 70,
    "vision": {
        "mode": "circular",
        "vertical_fov": 70,
        "resolution": [8, 8],
        "distance": 2,
        "tick_delay": 2,
        "block_classes": 10,
    },
    "hearing": {"enabled": True, "sectors": 8, "radius": 16},
    "entities": {"max_tracked": 8, "radius": 16},
    "memory": {"frame_stack": 3},
    "zmq": {"node_to_py": 5555, "py_to_node": 5556},
    "train": {
        "gamma": 0.99,
        "batch_size": 256,
        "buffer_size": 100000,
        "lr": 1e-4,
        "learn_steps_per_tick": 0.5,
        "max_learn_per_cycle": 8,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        # Шагов (своих, у каждой задачки) до минимума случайных действий;
        # задачка может переопределить: modules.<задачка>.epsilon_decay_steps.
        "epsilon_decay_steps": 150000,
        "target_update_every": 2000,  # в шагах обучения, не среды
        "double_dqn": True,
        "n_step": 3,
        "tick_rate_ms": 150,
        "autosave_ticks": 5000,
        "metrics_every_seconds": 30,
        # Новый мозг задачки стартует с весов мозга родственной задачки
        # (если тот уже обучен): follow и gathering — та же ходьба.
        "warm_start_from": {"follow": "walking", "gathering": "walking"},
        # Режим --task mix: какие задачи раздавать ботам и через сколько
        # тиков (своих, на бота) переходить к следующей.
        "mix_tasks": ["walking", "looking", "follow", "gathering"],
        "mix_rotate_ticks": 3000,
        "demo_preload": True,
        "demo_fraction": 0.25,
        "demo_margin": 1.0,
        # Сколько первых ботов роя в салках и охоте ведёт учитель
        # (py/sim/teacher.py) — демонстрации держат выученное (ai_loop.main).
        "live_teachers": 0,
    },
    "bc": {"epochs": 60, "lr": 3e-4, "batch_size": 256},
    "reward": {
        "reach_goal": 100,
        "closer": 1.0,
        "idle": -1.0,
        "death": -100,
        "goal_radius": 2.0,
    },
    # Параметры конкретных обучающих модулей (py/training_modules/).
    # "reward" выше — общие для всех модулей величины (death/idle и т.п.),
    # тут — специфика каждой специализации. См. py/training_modules/base.py
    # про то, как добавить свой модуль.
    "modules": {
        "walking": {
            "min_radius": 5.0,
            "max_radius": 15.0,
            "stuck_ticks_limit": 100,
            "obstacle_close_distance": 2.0,
            "obstacle_avoid_bonus": 0.3,
            "obstacle_bump_penalty": -0.3,
            "pitch_penalty": 0.5,
            "health_loss_penalty": 2.0,
            "head_move_penalty": 0.05,
            # "Давление времени": минус за каждый тик, пока цель не достигнута —
            # быстрее дойти выгоднее, и бег (sprint_forward) наконец окупается.
            "time_penalty": 0.05,
            "heading_reward": 0.5,
            "near_goal_reward": 1.0,
            "hands_penalty": 0.1,
            "border_target_margin": 4.0,
            # Запасной вариант, если сервер не прислал world border.
            # Предположение: центр барьера в (0, 0) — это дефолт, если его
            # не двигали командой /worldborder center. Если у тебя не так —
            # поправь эти 4 числа под реальные координаты твоего барьера.
            "bounds": {"min_x": -50.0, "max_x": 50.0, "min_z": -50.0, "max_z": 50.0},
        },
        "looking": {
            "epsilon_decay_steps": 60000,  # простая задачка — меньше блуждать наугад
            "death_penalty": 0.0,  # убежать в looking нельзя — нечему и учиться
            "gamma": 0.9,  # короткий горизонт: "навестись" решается за секунду
            # Навестись можно по одним данным о цели: без зрения, сущностей и
            # слуха мозгу не по чему заучивать место вместо правила.
            "inputs": {"vision": False, "entities": False, "hearing": False},
            "center_angle_deg": 8.0,
            "track_reward": 2.0,
            "in_view_reward": 0.5,
            "center_reward": 2.0,
            "idle_penalty": -0.2,
            "legs_move_penalty": 0.1,
            "occlusion_margin": 1.0,
        },
        "follow": {
            # Награда за каждый шаг к цели плотная — далеко вперёд заглядывать
            # незачем (догнать цель в 16 блоках — ~25 тиков), а короткий
            # горизонт делает разницу между "к цели" и "от цели" контрастнее.
            "gamma": 0.95,
            "epsilon_decay_steps": 80000,  # ~30 минут роем из 7 ботов
            "min_distance": 2.0,
            "max_distance": 5.0,
            "in_band_reward": 0.5,
            "too_close_penalty": -0.2,
        },
        "gathering": {
            "resource_items": "_log$",  # регулярка по именам предметов
            "resource_class": 1,         # класс блока в зрении (1 — древесина)
            "quota": 5,
            "item_reward": 10.0,
            "other_item_reward": 1.0,
            "quota_reward": 100.0,
            "approach_reward": 0.5,
            "pickup_reward": 0.5,
            "aim_reward": 0.1,
            "hit_reward": 0.2,
            "kill_reward": 15.0,
            "kill_radius": 4.0,
            "miss_penalty": -0.5,
        },
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in (override or {}).items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(base[key], value)
        else:
            result[key] = value
    return result


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        parsed = json.load(f)
    return _deep_merge(DEFAULTS, parsed)


CONFIG = load_config()
