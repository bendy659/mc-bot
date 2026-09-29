"""Зрение в симуляции — перенос js/vision.js (режим circular): сетка лучей
вокруг бота, каждый луч — до первого блока (цвет, дальность, класс), и луч
прицела (center_block) — туда, куда смотрит голова, с наклоном.

Лучи шагают по клеткам ровно как RaycastIterator из prismarine-world
(Amanatides-Woo, тот же выбор оси при равенстве), дальность попадания —
вход луча в блок (у нас все блоки — целые кубы). Все лучи всех ботов
считаются разом, массивами numpy.
"""

from __future__ import annotations

import math

import numpy as np

from .world import AIR, BLOCK_NAMES, CLASS_BY_ID, COLOR_BYTES, SKY_BYTES, ArenaWorld

EYE_HEIGHT = 1.62  # entity.eyeHeight у игроков mineflayer (js/vision.js: eyePosition)
BIG = 1.7976931348623157e308  # Number.MAX_VALUE — так RaycastIterator помечает ось без движения


def raycast(world: ArenaWorld, origins: np.ndarray, directions: np.ndarray, max_distance: float):
    """Лучи из origins (M, 3) по единичным directions (M, 3) — до первого
    непустого блока не дальше max_distance. Возвращает (дальность (M,),
    id блока (M,)); id 0 — ни во что не попал."""
    count = len(origins)
    cell = np.floor(origins).astype(np.int64)
    d = directions
    moving = d != 0
    with np.errstate(divide="ignore", invalid="ignore"):
        step = np.sign(d).astype(np.int64)
        t_delta = np.where(moving, np.abs(1.0 / np.where(moving, d, 1.0)), BIG)
        boundary = cell + (d > 0)
        t_max = np.where(moving, np.abs((boundary - origins) / np.where(moving, d, 1.0)), BIG)

    distance = np.full(count, np.inf)
    hit = np.zeros(count, dtype=np.uint8)
    t_entry = np.zeros(count)
    active = np.arange(count)
    while active.size:
        ids = world.ids(cell[active])
        solid = ids != AIR
        if solid.any():
            done = active[solid]
            distance[done] = t_entry[done]
            hit[done] = ids[solid]
            active = active[~solid]
            if not active.size:
                break
        tm = t_max[active]
        # Дальше предела — луч пуст (как next() итератора: проверка ДО шага).
        within = tm.min(axis=1) <= max_distance
        active, tm = active[within], tm[within]
        if not active.size:
            break
        tx, ty, tz = tm[:, 0], tm[:, 1], tm[:, 2]
        axis = np.where(tx < ty, np.where(tx < tz, 0, 2), np.where(ty < tz, 1, 2))
        rows = np.arange(len(active))
        t_entry[active] = tm[rows, axis]
        cell[active, axis] += step[active, axis]
        t_max[active, axis] += t_delta[active, axis]
    return distance, hit


class Vision:
    def __init__(self, config: dict):
        vision = config["vision"]
        if vision.get("mode") != "circular":
            raise ValueError("Симуляция умеет только круговое зрение (vision.mode = circular), как у роя")
        self.res_x, self.res_y = vision["resolution"]
        self.max_distance = vision["distance"] * 16  # чанки -> блоки
        vertical_span = math.radians(vision["vertical_fov"])
        rows = np.arange(self.res_y)
        cols = np.arange(self.res_x)
        # Ряд 0 — верх (+span/2), колонка 0 — прямо вперёд, дальше по кругу
        # вправо: ровно как buildVisionGrid.
        v_angle = vertical_span / 2 - rows * vertical_span / (self.res_y - 1)
        h_angle = -(cols * 2 * math.pi) / self.res_x
        self.v_angle = np.repeat(v_angle, self.res_x)          # (cells,)
        self.h_angle = np.tile(h_angle, self.res_y)            # (cells,)

    def grids(self, world: ArenaWorld, eyes: np.ndarray, yaws: np.ndarray) -> np.ndarray:
        """Сетки зрения всех ботов: (N, cells*5) uint8 — байты [r, g, b, d, t]
        на ячейку, как vision.packed от Node (js/state.js: packVision)."""
        count = len(eyes)
        cells = self.res_x * self.res_y
        ray_yaw = yaws[:, None] + self.h_angle[None, :]
        cos_v = np.cos(self.v_angle)[None, :]
        directions = np.stack([
            -np.sin(ray_yaw) * cos_v,
            np.broadcast_to(np.sin(self.v_angle)[None, :], ray_yaw.shape),
            -np.cos(ray_yaw) * cos_v,
        ], axis=-1).reshape(-1, 3)
        origins = np.repeat(eyes, cells, axis=0)
        distance, ids = raycast(world, origins, directions, self.max_distance)

        out = np.empty((count * cells, 5), dtype=np.uint8)
        seen = ids != AIR
        out[:, :3] = np.where(seen[:, None], COLOR_BYTES[ids], SKY_BYTES[None, :])
        d = np.minimum(np.where(seen, distance, 0.0) / self.max_distance, 1.0)
        out[:, 3] = np.where(seen, np.floor(d * 255.0 + 0.5), 255).astype(np.uint8)
        out[:, 4] = CLASS_BY_ID[ids]
        return out.reshape(count, cells * 5)

    def center_blocks(self, world: ArenaWorld, eyes: np.ndarray, yaws: np.ndarray, pitches: np.ndarray) -> list:
        """Что строго по центру прицела (с наклоном головы) — как
        centerBlockInfo: {"id", "name", "distance", "t"} или None."""
        directions = np.stack([
            -np.sin(yaws) * np.cos(pitches),
            np.sin(pitches),
            -np.cos(yaws) * np.cos(pitches),
        ], axis=-1)
        distance, ids = raycast(world, eyes, directions, self.max_distance)
        out = []
        for dist, block in zip(distance, ids):
            if block == AIR:
                out.append(None)
                continue
            out.append({"id": int(block), "name": BLOCK_NAMES[block],
                        "distance": math.floor(dist * 100 + 0.5) / 100, "t": int(CLASS_BY_ID[block])})
        return out


# Чувство пола под ногами — как groundProbe в js/vision.js: сколько пола до
# края впереди, справа, сзади и слева (относительно взгляда), в блоках, со
# знаком (над пустотой — минус сколько до пола) и не дальше GROUND_RANGE. Пол
# — блок на уровне под ногами (в симуляции все блоки — полные кубы).
GROUND_RANGE = 3.0


def ground_probe(world: ArenaWorld, pos, yaw: float) -> list[float]:
    level = math.floor(pos[1] - 0.01)

    def floor_at(x: int, z: int) -> bool:
        return world.solid(x, level, z)

    # Вперёд f = (-sin yaw, -cos yaw), вправо r = (-f.z, f.x) — как в js/state.js.
    fx, fz = -math.sin(yaw), -math.cos(yaw)
    directions = ((fx, fz), (-fz, fx), (-fx, -fz), (fz, -fx))
    return [math.floor(_floor_distance(floor_at, pos[0], pos[2], dx, dz) * 1000 + 0.5) / 1000
            for dx, dz in directions]


def _floor_distance(floor_at, x: float, z: float, dx: float, dz: float) -> float:
    """Клетка за клеткой по горизонтали (как floorDistance в js/vision.js) до
    первой клетки, где с полом не так, как под серединой бота."""
    cx, cz = math.floor(x), math.floor(z)
    on_floor = floor_at(cx, cz)
    step_x = 1 if dx > 0 else -1 if dx < 0 else 0
    step_z = 1 if dz > 0 else -1 if dz < 0 else 0
    next_x = (cx + 1 - x) / dx if step_x > 0 else (cx - x) / dx if step_x < 0 else math.inf
    next_z = (cz + 1 - z) / dz if step_z > 0 else (cz - z) / dz if step_z < 0 else math.inf
    delta_x = abs(1 / dx) if step_x else math.inf
    delta_z = abs(1 / dz) if step_z else math.inf
    while True:
        if next_x < next_z:
            t = next_x
            cx += step_x
            next_x += delta_x
        else:
            t = next_z
            cz += step_z
            next_z += delta_z
        if t >= GROUND_RANGE:
            return GROUND_RANGE if on_floor else -GROUND_RANGE
        if floor_at(cx, cz) != on_floor:
            return t if on_floor else -t
