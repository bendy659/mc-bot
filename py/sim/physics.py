"""Физика игрока в симуляции — перенос prismarine-physics (им двигается
бот mineflayer), тик 50 мс: прыжок, ускорение и трение на земле и в
воздухе, гравитация, столкновения с блоками по осям (y, потом x, потом z),
крадучись (медленнее и не сходит с края — мост над пустотой).
Порядок и числа — из node_modules/prismarine-physics/index.js: столкновения
по коробкам формы блоков (карты бедварса: ступени, плиты, заборы), подъём на
уступ до 0.6 блока без прыжка, скользкий лёд, лестницы и лианы, песок душ
(как ванильный сервер — см. _move). Упрощено: нет воды и эффектов.
"""

from __future__ import annotations

import math

import numpy as np

from .world import BLOCK_NAMES, ArenaWorld

GRAVITY = 0.08
AIRDRAG = float(np.float32(0.98))          # Math.fround(1 - 0.02)
JUMP_VELOCITY = float(np.float32(0.42))    # Math.fround(0.42)
SPRINT_JUMP_BOOST = 0.2
PLAYER_SPEED = 0.1
SPRINT_MULTIPLIER = 1.3                    # модификатор скорости бега (+30%)
AIR_ACCELERATION = 0.02
AIR_SPRINT_EXTRA = 0.02 * 0.3
AIR_INERTIA = 0.91
AUTOJUMP_COOLDOWN = 10                     # тиков между прыжками, пока зажат прыжок
NEGLIGIBLE = 0.003
SNEAK_SPEED = 0.3                          # крадучись — втрое медленнее (prismarine: sneakSpeed)
SNEAK_STEP = 0.05                          # шаг, которым крадущийся "отступает" от края
HALF_WIDTH = 0.3
HEIGHT = 1.8
STEP_HEIGHT = 0.6                          # уступ, на который заходят без прыжка (плита, ступень)
# Скольжение блока под ногами (по умолчанию 0.6): лёд — едешь по инерции.
SLIPPERINESS = {"ice": 0.98, "packed_ice": 0.98, "frosted_ice": 0.98, "blue_ice": 0.989, "slime_block": 0.8}
DEFAULT_SLIPPERINESS = 0.6
# Замедление на блоке (ваниль: Block.speedFactor) — песок душ и мёд.
SPEED_FACTOR = {"soul_sand": 0.4, "honey_block": 0.4}
CLIMBABLE = {"ladder", "vine", "scaffolding"}
LADDER_MAX_SPEED = 0.15
LADDER_CLIMB_SPEED = 0.2


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
        self.collided_horizontally = False

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

    x, y, z = body.pos
    if body.on_ground:
        under = BLOCK_NAMES[world.block(math.floor(x), math.floor(y - 1), math.floor(z))]
        inertia = SLIPPERINESS.get(under, DEFAULT_SLIPPERINESS) * 0.91
        speed = PLAYER_SPEED * (SPRINT_MULTIPLIER if sprint else 1.0)
        acceleration = speed * (0.1627714 / (inertia * inertia * inertia))
    else:
        inertia = AIR_INERTIA
        acceleration = AIR_ACCELERATION + (AIR_SPRINT_EXTRA if sprint else 0.0)
    _apply_heading(body, strafe, forward, acceleration)
    # Лестница, лиана: падать не быстрее LADDER_MAX_SPEED (крадучись — висеть),
    # в стену или с прыжком — лезть вверх.
    if _on_ladder(world, body.pos):
        vel[0] = min(max(vel[0], -LADDER_MAX_SPEED), LADDER_MAX_SPEED)
        vel[2] = min(max(vel[2], -LADDER_MAX_SPEED), LADDER_MAX_SPEED)
        vel[1] = max(vel[1], 0.0 if controls.get("sneak") else -LADDER_MAX_SPEED)
    _move(body, world, vel[0], vel[1], vel[2])
    if _on_ladder(world, body.pos) and (body.collided_horizontally or controls.get("jump")):
        vel[1] = LADDER_CLIMB_SPEED

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


def _extend(box: list, dx: float, dy: float, dz: float) -> list:
    """AABB.extend prismarine: коробка, растянутая в сторону сдвига."""
    x0, y0, z0, x1, y1, z1 = box
    return [x0 + min(dx, 0), y0 + min(dy, 0), z0 + min(dz, 0), x1 + max(dx, 0), y1 + max(dy, 0), z1 + max(dz, 0)]


def _offset_y(block: tuple, box: list, dy: float) -> float:
    """AABB.computeOffsetY: насколько можно сдвинуть box по y, не войдя в block."""
    if block[3] > box[0] and block[0] < box[3] and block[5] > box[2] and block[2] < box[5]:
        if dy > 0 and box[4] <= block[1]:
            dy = min(block[1] - box[4], dy)
        elif dy < 0 and box[1] >= block[4]:
            dy = max(block[4] - box[1], dy)
    return dy


def _offset_x(block: tuple, box: list, dx: float) -> float:
    if block[4] > box[1] and block[1] < box[4] and block[5] > box[2] and block[2] < box[5]:
        if dx > 0 and box[3] <= block[0]:
            dx = min(block[0] - box[3], dx)
        elif dx < 0 and box[0] >= block[3]:
            dx = max(block[3] - box[0], dx)
    return dx


def _offset_z(block: tuple, box: list, dz: float) -> float:
    if block[4] > box[1] and block[1] < box[4] and block[3] > box[0] and block[0] < box[3]:
        if dz > 0 and box[5] <= block[2]:
            dz = min(block[2] - box[5], dz)
        elif dz < 0 and box[2] >= block[5]:
            dz = max(block[5] - box[2], dz)
    return dz


def _shift(box: list, dx: float, dy: float, dz: float) -> list:
    return [box[0] + dx, box[1] + dy, box[2] + dz, box[3] + dx, box[4] + dy, box[5] + dz]


def _collide(blocks: list, box: list, dx: float, dy: float, dz: float) -> tuple[list, float, float, float]:
    """Сдвиг коробки с упором в блоки: по y, потом x, потом z (как prismarine)."""
    for block in blocks:
        dy = _offset_y(block, box, dy)
    box = _shift(box, 0, dy, 0)
    for block in blocks:
        dx = _offset_x(block, box, dx)
    box = _shift(box, dx, 0, 0)
    for block in blocks:
        dz = _offset_z(block, box, dz)
    return _shift(box, 0, 0, dz), dx, dy, dz


def _move(body: Body, world: ArenaWorld, dx: float, dy: float, dz: float) -> None:
    """moveEntity из prismarine-physics: откат крадущегося от края, упор в
    коробки блоков (ступени, плиты, кровати — по их форме), подъём на уступ
    до STEP_HEIGHT без прыжка, песок душ замедляет."""
    x, y, z = body.pos
    if body.controls.get("sneak") and body.on_ground:
        dx, dz = _back_off_from_edge(world, body.pos, dx, dz)
    # Как prismarine: откат от края — не столкновение, скорость он не гасит
    # (old_* берутся уже после него).
    old_dx, old_dy, old_dz = dx, dy, dz
    start = [x - HALF_WIDTH, y, z - HALF_WIDTH, x + HALF_WIDTH, y + HEIGHT, z + HALF_WIDTH]
    blocks = _boxes_near(world, _extend(start, dx, dy, dz))
    box, dx, dy, dz = _collide(blocks, start, dx, dy, dz)

    # Уступ ниже STEP_HEIGHT (плита, ступень) — зайти без прыжка: два
    # варианта подъёма (с учётом сдвига и без), берётся тот, что дальше ушёл.
    if (body.on_ground or (dy != old_dy and old_dy < 0)) and (dx != old_dx or dz != old_dz):
        col_dx, col_dy, col_dz, col_box = dx, dy, dz, box
        up = STEP_HEIGHT
        near = _boxes_near(world, _extend(start, old_dx, up, old_dz))
        box1, box2 = list(start), list(start)
        box_xz = _extend(box1, dx, 0, dz)
        dy1 = dy2 = up
        for block in near:
            dy1 = _offset_y(block, box_xz, dy1)
            dy2 = _offset_y(block, box2, dy2)
        box1, box2 = _shift(box1, 0, dy1, 0), _shift(box2, 0, dy2, 0)
        dx1 = dx2 = old_dx
        for block in near:
            dx1 = _offset_x(block, box1, dx1)
            dx2 = _offset_x(block, box2, dx2)
        box1, box2 = _shift(box1, dx1, 0, 0), _shift(box2, dx2, 0, 0)
        dz1 = dz2 = old_dz
        for block in near:
            dz1 = _offset_z(block, box1, dz1)
            dz2 = _offset_z(block, box2, dz2)
        box1, box2 = _shift(box1, 0, 0, dz1), _shift(box2, 0, 0, dz2)
        if dx1 * dx1 + dz1 * dz1 > dx2 * dx2 + dz2 * dz2:
            dx, dy, dz, box = dx1, -dy1, dz1, box1
        else:
            dx, dy, dz, box = dx2, -dy2, dz2, box2
        for block in near:
            dy = _offset_y(block, box, dy)
        box = _shift(box, 0, dy, 0)
        if col_dx * col_dx + col_dz * col_dz >= dx * dx + dz * dz:
            dx, dy, dz, box = col_dx, col_dy, col_dz, col_box

    body.pos = [box[0] + HALF_WIDTH, box[1], box[2] + HALF_WIDTH]
    body.on_ground = dy != old_dy and old_dy < 0
    body.collided_horizontally = dx != old_dx or dz != old_dz
    if dx != old_dx:
        body.vel[0] = 0.0
    if dz != old_dz:
        body.vel[2] = 0.0
    if dy != old_dy:
        body.vel[1] = 0.0
    # Песок душ и мёд замедляют — как ванильный сервер (Entity.getBlockSpeedFactor:
    # блок у ног, а если он не замедляет — блок на полблока ниже). prismarine
    # для 26.1 этого не делает (lib/features.json кончается на 1.20), но тело
    # ботов — плагин, у него физика сервера.
    px, py_, pz = body.pos
    factor = SPEED_FACTOR.get(BLOCK_NAMES[world.block(math.floor(px), math.floor(py_), math.floor(pz))], 1.0)
    if factor == 1.0:
        factor = SPEED_FACTOR.get(BLOCK_NAMES[world.block(math.floor(px), math.floor(py_ - 0.5000001), math.floor(pz))], 1.0)
    body.vel[0] *= factor
    body.vel[2] *= factor


def _on_ladder(world: ArenaWorld, pos: list) -> bool:
    return BLOCK_NAMES[world.block(math.floor(pos[0]), math.floor(pos[1]), math.floor(pos[2]))] in CLIMBABLE


def _back_off_from_edge(world: ArenaWorld, pos: list, dx: float, dz: float) -> tuple[float, float]:
    """Крадущийся на земле не сходит с края (prismarine-physics, moveEntity):
    сдвиг по x, потом по z, потом по обоим уменьшается шагами SNEAK_STEP, пока
    рядом с коробкой игрока, сдвинутой на него, нет ни одного блока — в слое
    под ногами и по всей высоте тела (prismarine смотрит так, а не только под
    ноги: "иначе не выходит как в ванили")."""
    def unsupported(ox: float, oz: float) -> bool:
        x, y, z = pos
        return not _boxes_near(world, [x - HALF_WIDTH + ox, y, z - HALF_WIDTH + oz,
                                       x + HALF_WIDTH + ox, y + HEIGHT, z + HALF_WIDTH + oz])

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


def _boxes_near(world: ArenaWorld, query: list) -> list:
    """getSurroundingBBs: коробки всех блоков в клетках запроса — и в слое
    под его низом (от floor(minY) - 1)."""
    out = []
    for bx in range(math.floor(query[0]), math.floor(query[3]) + 1):
        for by in range(math.floor(query[1]) - 1, math.floor(query[4]) + 1):
            for bz in range(math.floor(query[2]), math.floor(query[5]) + 1):
                if world.solid(bx, by, bz):
                    out.extend(world.boxes(bx, by, bz))
    return out


def step_ahead(body: Body, world: ArenaWorld) -> bool:
    """Автопрыжок (js/actions.js: stepAhead): по курсу уступ в один блок, над
    которым свободно, — прыгнуть."""
    x, y, z = body.pos
    feet_y = math.floor(y)
    here_x, here_z = math.floor(x), math.floor(z)
    # Как в JS: "твёрдый" и "свободный" — по boundingBox блока ('block' | 'empty').
    if world.box_block(here_x, feet_y + 2, here_z):
        return False  # над головой потолок — не подпрыгнуть
    fx, fz = -math.sin(body.yaw), -math.cos(body.yaw)
    for distance in (0.5, 1.0, 1.4):
        cx = math.floor(x + fx * distance)
        cz = math.floor(z + fz * distance)
        if cx == here_x and cz == here_z:
            continue
        if world.box_block(cx, feet_y, cz):
            return not world.box_block(cx, feet_y + 1, cz) and not world.box_block(cx, feet_y + 2, cz)
    return False
