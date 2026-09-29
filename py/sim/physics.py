"""Физика игрока в симуляции — перенос prismarine-physics (им двигается
бот mineflayer), тик 50 мс: прыжок, ускорение и трение на земле и в
воздухе, гравитация, столкновения с блоками по осям (y, потом x, потом z),
крадучись (медленнее и не сходит с края — мост над пустотой).
Порядок и числа — из node_modules/prismarine-physics/index.js. Упрощено:
нет воды, лестниц, эффектов и "шага" на полблока (на арене только целые
блоки, на них шагом не зайти — только прыжком).
"""

from __future__ import annotations

import math

import numpy as np

from .world import ArenaWorld

GRAVITY = 0.08
AIRDRAG = float(np.float32(0.98))          # Math.fround(1 - 0.02)
JUMP_VELOCITY = float(np.float32(0.42))    # Math.fround(0.42)
SPRINT_JUMP_BOOST = 0.2
PLAYER_SPEED = 0.1
SPRINT_MULTIPLIER = 1.3                    # модификатор скорости бега (+30%)
GROUND_INERTIA = 0.6 * 0.91                # обычные блоки: скольжение 0.6
AIR_ACCELERATION = 0.02
AIR_SPRINT_EXTRA = 0.02 * 0.3
AIR_INERTIA = 0.91
AUTOJUMP_COOLDOWN = 10                     # тиков между прыжками, пока зажат прыжок
NEGLIGIBLE = 0.003
SNEAK_SPEED = 0.3                          # крадучись — втрое медленнее (prismarine: sneakSpeed)
SNEAK_STEP = 0.05                          # шаг, которым крадущийся "отступает" от края
HALF_WIDTH = 0.3
HEIGHT = 1.8


class Body:
    """Тело игрока: позиция ног, скорость (блоков за тик), взгляд, на земле
    ли, нажатые "клавиши" (control states mineflayer)."""

    def __init__(self, x: float, y: float, z: float, yaw: float = 0.0):
        self.pos = [x, y, z]
        self.vel = [0.0, 0.0, 0.0]
        self.yaw = yaw
        self.pitch = 0.0
        self.on_ground = True
        self.jump_ticks = 0
        self.controls = {}

    def teleport(self, x: float, y: float, z: float) -> None:
        self.pos = [x, y, z]
        self.vel = [0.0, 0.0, 0.0]
        self.on_ground = True
        self.jump_ticks = 0


def physics_tick(body: Body, world: ArenaWorld) -> None:
    vel = body.vel
    for axis in range(3):
        if abs(vel[axis]) < NEGLIGIBLE:
            vel[axis] = 0.0

    controls = body.controls
    sprint = bool(controls.get("sprint"))
    if controls.get("jump"):
        if body.jump_ticks > 0:
            body.jump_ticks -= 1
        if body.on_ground and body.jump_ticks == 0:
            vel[1] = JUMP_VELOCITY
            if sprint:
                yaw = math.pi - body.yaw
                vel[0] -= math.sin(yaw) * SPRINT_JUMP_BOOST
                vel[2] += math.cos(yaw) * SPRINT_JUMP_BOOST
            body.jump_ticks = AUTOJUMP_COOLDOWN
    else:
        body.jump_ticks = 0

    strafe = (bool(controls.get("right")) - bool(controls.get("left"))) * 0.98
    forward = (bool(controls.get("forward")) - bool(controls.get("back"))) * 0.98
    if controls.get("sneak"):
        strafe *= SNEAK_SPEED
        forward *= SNEAK_SPEED

    if body.on_ground:
        inertia = GROUND_INERTIA
        speed = PLAYER_SPEED * (SPRINT_MULTIPLIER if sprint else 1.0)
        acceleration = speed * (0.1627714 / (inertia * inertia * inertia))
    else:
        inertia = AIR_INERTIA
        acceleration = AIR_ACCELERATION + (AIR_SPRINT_EXTRA if sprint else 0.0)
    _apply_heading(body, strafe, forward, acceleration)
    _move(body, world, vel[0], vel[1], vel[2])

    vel[1] -= GRAVITY
    vel[1] *= AIRDRAG
    vel[0] *= inertia
    vel[2] *= inertia


def _apply_heading(body: Body, strafe: float, forward: float, multiplier: float) -> None:
    speed = math.sqrt(strafe * strafe + forward * forward)
    if speed < 0.01:
        return
    speed = multiplier / max(speed, 1.0)
    strafe *= speed
    forward *= speed
    yaw = math.pi - body.yaw
    sin, cos = math.sin(yaw), math.cos(yaw)
    body.vel[0] -= strafe * cos + forward * sin
    body.vel[2] += forward * cos - strafe * sin


def _move(body: Body, world: ArenaWorld, dx: float, dy: float, dz: float) -> None:
    x, y, z = body.pos
    if body.controls.get("sneak") and body.on_ground:
        dx, dz = _back_off_from_edge(world, body.pos, dx, dz)
    box = [x - HALF_WIDTH, y, z - HALF_WIDTH, x + HALF_WIDTH, y + HEIGHT, z + HALF_WIDTH]
    blocks = _blocks_near(world, box, dx, dy, dz)
    # Как prismarine: откат от края — не столкновение, скорость он не гасит
    # (old_* берутся уже после него).
    old_dx, old_dy, old_dz = dx, dy, dz

    for bx, by, bz in blocks:  # сначала по вертикали
        if box[3] > bx and box[0] < bx + 1 and box[5] > bz and box[2] < bz + 1:
            if dy > 0 and box[4] <= by:
                dy = min(by - box[4], dy)
            elif dy < 0 and box[1] >= by + 1:
                dy = max(by + 1 - box[1], dy)
    box[1] += dy
    box[4] += dy
    for bx, by, bz in blocks:
        if box[4] > by and box[1] < by + 1 and box[5] > bz and box[2] < bz + 1:
            if dx > 0 and box[3] <= bx:
                dx = min(bx - box[3], dx)
            elif dx < 0 and box[0] >= bx + 1:
                dx = max(bx + 1 - box[0], dx)
    box[0] += dx
    box[3] += dx
    for bx, by, bz in blocks:
        if box[4] > by and box[1] < by + 1 and box[3] > bx and box[0] < bx + 1:
            if dz > 0 and box[5] <= bz:
                dz = min(bz - box[5], dz)
            elif dz < 0 and box[2] >= bz + 1:
                dz = max(bz + 1 - box[2], dz)
    box[2] += dz
    box[5] += dz

    body.pos = [box[0] + HALF_WIDTH, box[1], box[2] + HALF_WIDTH]
    body.on_ground = dy != old_dy and old_dy < 0
    if dx != old_dx:
        body.vel[0] = 0.0
    if dz != old_dz:
        body.vel[2] = 0.0
    if dy != old_dy:
        body.vel[1] = 0.0


def _back_off_from_edge(world: ArenaWorld, pos: list, dx: float, dz: float) -> tuple[float, float]:
    """Крадущийся на земле не сходит с края (prismarine-physics, moveEntity):
    сдвиг по x, потом по z, потом по обоим уменьшается шагами SNEAK_STEP, пока
    рядом с коробкой игрока, сдвинутой на него, нет ни одного блока — в слое
    под ногами и по всей высоте тела (prismarine смотрит так, а не только под
    ноги: "иначе не выходит как в ванили")."""
    def unsupported(ox: float, oz: float) -> bool:
        x, y, z = pos
        return not _any_solid(world, x - HALF_WIDTH + ox, y, z - HALF_WIDTH + oz,
                              x + HALF_WIDTH + ox, y + HEIGHT, z + HALF_WIDTH + oz)

    def shrink(value: float) -> float:
        if -SNEAK_STEP <= value < SNEAK_STEP:
            return 0.0
        return value - SNEAK_STEP if value > 0 else value + SNEAK_STEP

    while dx != 0 and unsupported(dx, 0.0):
        dx = shrink(dx)
    while dz != 0 and unsupported(0.0, dz):
        dz = shrink(dz)
    while dx != 0 and dz != 0 and unsupported(dx, dz):
        dx = shrink(dx)
        dz = shrink(dz)
    return dx, dz


def _any_solid(world: ArenaWorld, x0: float, y0: float, z0: float, x1: float, y1: float, z1: float) -> bool:
    """Есть ли твёрдый блок в клетках коробки — и в слое под её низом
    (prismarine: getSurroundingBBs берёт от floor(minY) - 1)."""
    for bx in range(math.floor(x0), math.floor(x1) + 1):
        for by in range(math.floor(y0) - 1, math.floor(y1) + 1):
            for bz in range(math.floor(z0), math.floor(z1) + 1):
                if world.solid(bx, by, bz):
                    return True
    return False


def _blocks_near(world: ArenaWorld, box: list, dx: float, dy: float, dz: float) -> list:
    """Твёрдые клетки, которые может задеть коробка игрока, сдвинутая на (dx, dy, dz)."""
    lo = [math.floor(min(box[0], box[0] + dx)), math.floor(min(box[1], box[1] + dy)) - 1,
          math.floor(min(box[2], box[2] + dz))]
    hi = [math.floor(max(box[3], box[3] + dx)), math.floor(max(box[4], box[4] + dy)),
          math.floor(max(box[5], box[5] + dz))]
    out = []
    for bx in range(lo[0], hi[0] + 1):
        for by in range(lo[1], hi[1] + 1):
            for bz in range(lo[2], hi[2] + 1):
                if world.solid(bx, by, bz):
                    out.append((bx, by, bz))
    return out


def step_ahead(body: Body, world: ArenaWorld) -> bool:
    """Автопрыжок (js/actions.js: stepAhead): по курсу уступ в один блок, над
    которым свободно, — прыгнуть."""
    x, y, z = body.pos
    feet_y = math.floor(y)
    here_x, here_z = math.floor(x), math.floor(z)
    if world.solid(here_x, feet_y + 2, here_z):
        return False  # над головой потолок — не подпрыгнуть
    fx, fz = -math.sin(body.yaw), -math.cos(body.yaw)
    for distance in (0.5, 1.0, 1.4):
        cx = math.floor(x + fx * distance)
        cz = math.floor(z + fz * distance)
        if cx == here_x and cz == here_z:
            continue
        if world.solid(cx, feet_y, cz):
            return not world.solid(cx, feet_y + 1, cz) and not world.solid(cx, feet_y + 2, cz)
    return False
