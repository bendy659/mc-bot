"""Сверка зрения ботов плагина (plugin/) с симуляцией — а она сверена с
js/vision.js бит в бит (check_parity.py). Бот плагина встаёт в случайные позы
на арене, его состояние (/mcbot state) сравнивается с py/sim на той же позе:
сетка зрения байт в байт, блок по центру прицела, чувство пола и "блок
встанет" (can_place; и у краёв: островок с мостиком в воздухе над ареной,
бот — в том числе свесившись, и точно у границы окна для блока).

Нужен сервер с плагином и ПЛОСКИМ миром (как в симуляции), арена построена
(её строит ai_loop при первом подключении). Лучше — тестовый, не рабочий:
его RCON — в config.json, на который указывает MCBOT_CONFIG:

    set MCBOT_CONFIG=путь\\к\\тестовому\\config.json
    python py/sim/check_plugin_parity.py

2026-09-27: 40 поз — центр прицела 40/40, сетка совпала вся, кроме цвета
кучек земли, на которые на сервере расползлась трава (случайные тики; теперь
training_server выключает их: random_tick_speed 0).
"""

from __future__ import annotations

import base64
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CONFIG  # noqa: E402
from rcon import Rcon  # noqa: E402
from bridge_course import fill_command  # noqa: E402
from sim.game import free_for_block, place_front_cell  # noqa: E402
from sim.vision import EYE_HEIGHT, Vision, ground_probe  # noqa: E402
from sim.world import ArenaWorld  # noqa: E402

POSES = 40


def mineflayer_yaw(server_yaw: float) -> float:
    """Серверный yaw (градусы) -> mineflayer (как Geometry.yaw в плагине)."""
    value = math.fmod(math.pi - math.radians(server_yaw), 2 * math.pi)
    return value + 2 * math.pi if value < 0 else value


def mineflayer_pitch(server_pitch: float) -> float:
    value = math.fmod(math.radians(-server_pitch) + math.pi, 2 * math.pi)
    return (value + 2 * math.pi if value < 0 else value) - math.pi


def main() -> int:
    arena = CONFIG["server"]["arena"]
    rcon_cfg = CONFIG["server"].get("rcon")
    if not rcon_cfg:
        print("В конфиге нет server.rcon — не к чему подключаться.")
        return 1
    world = ArenaWorld(arena)
    vision = Vision(CONFIG)
    rcon = Rcon(rcon_cfg.get("host", "127.0.0.1"), rcon_cfg["port"], rcon_cfg["password"])
    rcon.connect()
    rcon.command("mcbot stop")
    rcon.command("mcbot start 2")  # два бота: ники AI_1, AI_2 (один — просто AI)
    rcon.command("clear AI_1")
    rcon.command("give AI_1 minecraft:dirt 64")  # can_place — есть чем строить
    rng = random.Random(5)
    half = arena["size"] // 2
    x0, z0 = arena["center"][0] - half, arena["center"][1] - half
    floor = arena["floor_y"]
    bytes_diff = center_diff = ground_diff = checked = total = 0
    for pose in range(POSES):
        while True:
            bx, bz = rng.randint(x0 + 1, x0 + arena["size"] - 2), rng.randint(z0 + 1, z0 + arena["size"] - 2)
            if not world.solid(bx, floor, bz) and not world.solid(bx, floor + 1, bz):
                break
        x, z = bx + rng.choice([0.3, 0.5, 0.71]), bz + rng.choice([0.3, 0.5, 0.64])
        yaw_deg, pitch_deg = round(rng.uniform(-180, 180), 1), round(rng.uniform(-60, 60), 1)
        rcon.command(f"tp AI_1 {x} {floor} {z} {yaw_deg} {pitch_deg}")
        # Ответ — путь к файлу (последней строкой: раньше могут прийти строки чата).
        path = rcon.command("mcbot state AI_1").strip().splitlines()[-1]
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        eye = np.array([[x, floor + EYE_HEIGHT, z]])
        yaw, pitch = mineflayer_yaw(yaw_deg), mineflayer_pitch(pitch_deg)
        expected = vision.grids(world, eye, np.array([yaw]))[0].tobytes()
        got = base64.b64decode(state["vision"]["packed"])
        diff = sum(a != b for a, b in zip(expected, got))
        sim_center = vision.center_blocks(world, eye, np.array([yaw]), np.array([pitch]))[0]
        center = state["center_block"]
        same_center = (sim_center is None) == (center is None) and (
            center is None or (center["name"] == sim_center["name"] and center["t"] == sim_center["t"]
                               and abs(center["distance"] - sim_center["distance"]) < 1e-9))
        same_ground = _same_ground(state.get("ground"), ground_probe(world, (x, floor, z), yaw))
        same_ground = same_ground and bool(state.get("can_place")) == _can_place(world, (x, floor, z), yaw, pitch)
        bytes_diff += diff
        center_diff += not same_center
        ground_diff += not same_ground
        total += len(got)
        checked += 1
        if diff or not same_center or not same_ground:
            print(f"поза {pose}: ({x}, {z}) yaw {yaw_deg} pitch {pitch_deg}: байтов зрения не совпало {diff}; "
                  f"центр: плагин {center}, симуляция {sim_center}; пол: плагин {state.get('ground')}, "
                  f"симуляция {ground_probe(world, (x, floor, z), yaw)}")
    edge_diff = check_edges(rcon, world, arena, rng)
    rcon.command("mcbot stop")
    rcon.close()
    print(f"поз: {checked}; байтов зрения не совпало {bytes_diff} из {total}; центр прицела не совпал: {center_diff}; "
          f"чувство пола или 'блок встанет' не совпали: {ground_diff} (у краёв — {edge_diff})")
    return 0 if bytes_diff == 0 and center_diff == 0 and ground_diff == 0 and edge_diff == 0 else 1


def _can_place(world: ArenaWorld, pos: tuple, yaw: float, pitch: float) -> bool:
    """can_place симуляции для бота с блоками, одного в мире."""
    cell = place_front_cell(world, (pos[0], pos[1] + EYE_HEIGHT, pos[2]), yaw, pitch, pos)
    return cell is not None and free_for_block(world, cell, [pos])


def _same_ground(plugin: list | None, sim: list) -> bool:
    # Поворот сервер хранит float-градусами: на диагоналях третий знак может разойтись.
    return plugin is not None and len(plugin) == len(sim) and all(abs(a - b) <= 0.002 for a, b in zip(plugin, sim))


def check_edges(rcon: Rcon, world: ArenaWorld, arena: dict, rng: random.Random) -> int:
    """Чувство пола у краёв: островок 5x5 и мостик 1x4 в воздухе над ареной
    (на сервере — /fill, в симуляции — те же блоки); бот — на них, в том числе
    свесившись с края до 0.29 (держится, не падает)."""
    cx, cz = arena["center"]
    y = arena["floor_y"] + 8
    fills = [(cx - 2, y, cz - 2, cx + 2, y, cz + 2, "stone_bricks"), (cx, y, cz + 3, cx, y, cz + 6, "dirt")]
    for fill in fills:
        rcon.command(fill_command(fill))
        world._fill(*fill)
    poses = []
    for _ in range(POSES):
        if rng.random() < 0.5:  # на мостике — и свесившись с конца и боков
            x = round(cx + 0.5 + rng.uniform(-0.79, 0.79), 3)
            z = round(cz + rng.uniform(3.2, 7.29), 3)
        else:                   # на островке, у краёв
            x = round(cx + rng.uniform(-2.29, 3.29), 3)
            z = round(cz + rng.uniform(-2.29, 3.0), 3)
        yaw_deg = 10 * rng.randint(-18, 17) if rng.random() < 0.5 else round(rng.uniform(-180, 180), 1)
        pitch_deg = 80 if rng.random() < 0.5 else round(rng.uniform(-30, 90), 1)  # сервер: вниз — плюс
        poses.append((x, z, yaw_deg, pitch_deg))
    # Точно у границы окна: спиной к пропасти за концом мостика (z = cz + 7),
    # взгляд -80° (на сервере 80), свес 0.27..0.2999 — блок встаёт с 0.2857.
    for overhang in (0.27, 0.284, 0.2855, 0.2858, 0.29, 0.2999):
        poses.append((cx + 0.5, cz + 7 + overhang, 180, 80))
    bad = 0
    for pose, (x, z, yaw_deg, pitch_deg) in enumerate(poses):
        rcon.command(f"tp AI_1 {x} {y + 1} {z} {yaw_deg} {pitch_deg}")
        path = rcon.command("mcbot state AI_1").strip().splitlines()[-1]
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        yaw, pitch = mineflayer_yaw(yaw_deg), mineflayer_pitch(pitch_deg)
        sim = ground_probe(world, (x, y + 1, z), yaw)
        sim_place = _can_place(world, (x, y + 1, z), yaw, pitch)
        if not _same_ground(state.get("ground"), sim) or bool(state.get("can_place")) != sim_place:
            bad += 1
            print(f"край, поза {pose}: ({x}, {z}) yaw {yaw_deg} pitch {pitch_deg}: пол плагин {state.get('ground')}, "
                  f"симуляция {sim}; блок встанет: плагин {state.get('can_place')}, симуляция {sim_place}")
    print(f"у краёв: поз {len(poses)}, блок встанет (симуляция) — в "
          f"{sum(_can_place(world, (x, y + 1, z), mineflayer_yaw(a), mineflayer_pitch(p)) for x, z, a, p in poses)}")
    for fill in fills:
        rcon.command(fill_command(fill[:6] + ("air",)))
    return bad


if __name__ == "__main__":
    sys.exit(main())
