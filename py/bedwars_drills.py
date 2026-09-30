"""Упражнения бедварса (автор, 2026-09-30: "рано пошли учить их играть в
бедварс сразу — сначала нужно было оттачивать механики: удары очень плохо,
ходьба по краю очень плохо, падают"): короткие раунды в пустом мире задачки
bedwars, у каждой пары (дуэль) или каждого бота (край) — своя дорожка.

  - дуэль (DuelLayout): двое на площадке над пустотой — мост в 1–3 блока между
    двумя пятачками или островок; у обоих деревянный меч. Упал или погиб —
    проиграл. Учит бить полным зарядом, быстро разворачиваться, отбрасывать и
    не падать самому.
  - ходьба по краю (EdgeLayout): узкая тропа над пустотой — повороты, уступы
    вверх (автопрыжок) и вниз — от пятачка старта до пятачка цели. Упал —
    умер (пустота).

Раскладка — одна на симуляцию и сервер (как py/bridge_course.py): судья
(py/bedwars_game.py) строит дорожки командами fill в мире упражнений
(config.json bot.task_worlds.drills — свой мир, НЕ мир карт: пустой мир
бедварса с каждой перезагрузкой заново генерировал вечно загруженные чанки
дорожек, и сервер висел до минуты). Участок дорожек загружает плагин —
drill_command() (/mcbot drill), симуляция по ней же переходит в свой пустой
мир (ArenaWorld.empty, границы — bounds()) и исполняет те же fill.
Дорожки — вдоль +z, на расстоянии LANE_SPACING по x: соседи не мешают.
"""

from __future__ import annotations

import math
import random

DRILL_Y = 64            # верх площадок; стоять — на DRILL_Y + 1
LANE_SPACING = 24       # между дорожками по x
LANE_HALF = 10          # что чистить вокруг середины дорожки по x
LANE_BACK = -4          # и по z: от LANE_BACK до LANE_FAR
LANE_FAR = 44
CLEAR_BELOW = 8         # чистить от DRILL_Y - CLEAR_BELOW (21 x 21 x 49 блоков — один /fill)
CLEAR_ABOVE = 12        # ...до DRILL_Y + CLEAR_ABOVE
VOID_BELOW = 6          # ноги ниже DRILL_Y - VOID_BELOW — упал в пустоту (судья убивает)
PAD = 1                 # полуразмер пятачков старта и цели: 3 x 3
# Блоки площадок — как на картах бедварса (все есть в data/bedwars/block_shapes.json:
# их знает и симуляция). Ломать их нельзя: fill — не "поставлено игроком".
MATERIALS = ("white_wool", "red_wool", "blue_wool", "lime_wool", "yellow_wool", "oak_planks", "sandstone",
             "stone_bricks", "terracotta", "cobblestone")
FACING_CHANCE = 0.6     # дуэль: лицом друг к другу; иначе — куда попало (развернуться — тоже навык)

# Тропа ходьбы по краю.
EDGE_LENGTH = (18, 30)  # клеток по оси тропы
EDGE_SEGMENT = (3, 7)   # длина отрезка между поворотами и уступами
EDGE_WIDTHS = (1, 1, 1, 2, 2, 3)
EDGE_TURN_CHANCE = 0.45
EDGE_STEP_CHANCE = 0.4  # отрезок начинается уступом (вверх или вниз на блок)
EDGE_SIDE_LIMIT = 7     # тропа не дальше стольких блоков от середины дорожки по x
EDGE_RISE = (-2, 3)     # и по высоте — в этих пределах от DRILL_Y
EDGE_FAR = 36           # тропа кончается не дальше этого z (пятачок цели — за ней)


def lane_x(lane: int) -> int:
    return lane * LANE_SPACING


def server_yaw(dx: float, dz: float) -> int:
    """Угол сервера (yaw 0 — на юг, +z; кратно 10°) для взгляда вдоль (dx, dz)."""
    return 10 * round(math.degrees(math.atan2(-dx, dz)) / 10)


class DuelLayout:
    """Площадка дуэли на дорожке lane: мост между двумя пятачками или островок."""

    def __init__(self, lane: int, rng: random.Random):
        self.lane = lane
        self.x = lane_x(lane)
        self.block = rng.choice(MATERIALS)
        self.kind = rng.choice(("bridge", "island"))
        self.cells: list[tuple] = []  # (x1, y1, z1, x2, y2, z2) — прямоугольники площадки
        if self.kind == "bridge":
            self.width = rng.choice((1, 1, 2, 3))
            self.length = rng.randint(6, 14)
            far = 2 * PAD + 1 + self.length  # середина дальнего пятачка по z
            left = self.x - (self.width - 1) // 2
            self.cells = [(self.x - PAD, DRILL_Y, -PAD, self.x + PAD, DRILL_Y, PAD),
                          (left, DRILL_Y, PAD + 1, left + self.width - 1, DRILL_Y, far - PAD - 1),
                          (self.x - PAD, DRILL_Y, far - PAD, self.x + PAD, DRILL_Y, far + PAD)]
            ends = [(self.x, 0), (self.x, far)]
        else:
            self.width = rng.randint(4, 8)
            self.length = rng.randint(6, 12)
            left = self.x - self.width // 2
            self.cells = [(left, DRILL_Y, 0, left + self.width - 1, DRILL_Y, self.length - 1)]
            # Не с самого края островка: упасть в первый же тик — не урок.
            ends = [(left + rng.randrange(1, self.width - 1), 1), (left + rng.randrange(1, self.width - 1), self.length - 2)]
        # Лицом друг к другу — или куда попало: развернуться к врагу тоже надо уметь.
        self.spawns = []
        for index, (sx, sz) in enumerate(ends):
            ox, oz = ends[1 - index]
            yaw = server_yaw(ox - sx, oz - sz) if rng.random() < FACING_CHANCE else 10 * rng.randint(-18, 17)
            self.spawns.append((sx + 0.5, DRILL_Y + 1, sz + 0.5, yaw))

    def fills(self) -> list[tuple]:
        return [(*cell, self.block) for cell in self.cells]

    def describe(self) -> str:
        if self.kind == "bridge":
            return f"мост {self.width}x{self.length}"
        return f"островок {self.width}x{self.length}"


class EdgeLayout:
    """Тропа над пустотой на дорожке lane: пятачок старта у z = 0, дальше —
    отрезки по +z и вбок (вбок — всегда потом снова по +z: тропа сама себя не
    касается), уступы на блок вверх и вниз, в конце — пятачок цели."""

    def __init__(self, lane: int, rng: random.Random):
        self.lane = lane
        self.x = lane_x(lane)
        self.block = rng.choice(MATERIALS)
        self.width = rng.choice(EDGE_WIDTHS)
        self.cells: list[tuple] = [(self.x - PAD, DRILL_Y, -PAD, self.x + PAD, DRILL_Y, PAD)]
        self.turns = 0
        self.steps = 0
        x, y, z = self.x, DRILL_Y, PAD  # последняя клетка оси тропы (край пятачка старта)
        heading = (0, 1)
        length, target = 0, rng.randint(*EDGE_LENGTH)
        while length < target and z < EDGE_FAR:
            segment = rng.randint(*EDGE_SEGMENT)
            # Поворот: вбок — только с +z, и не дальше EDGE_SIDE_LIMIT от середины
            # дорожки (с шириной); с боку — обратно на +z.
            turned = False
            if heading != (0, 1):
                heading, turned = (0, 1), True
            elif length > 0 and rng.random() < EDGE_TURN_CHANCE:
                side = rng.choice((-1, 1))
                room = EDGE_SIDE_LIMIT - (self.width - 1) - side * (x - self.x)  # сколько можно вбок
                if room < 2:
                    side = -side
                    room = EDGE_SIDE_LIMIT - (self.width - 1) - side * (x - self.x)
                if room >= 2:
                    heading, turned = (side, 0), True
                    segment = min(segment, room)
            if turned:
                self.turns += 1
            # Уступ в начале отрезка — только на прямой (не на повороте): широкие
            # отрезки на углу перекрываются, и на разной высоте вышла бы стенка.
            elif length > 0 and rng.random() < EDGE_STEP_CHANCE:
                dy = rng.choice((-1, 1))
                if EDGE_RISE[0] <= y + dy - DRILL_Y <= EDGE_RISE[1]:
                    y += dy
                    self.steps += 1
            if heading == (0, 1):
                segment = min(segment, EDGE_FAR - z)
            if segment <= 0:
                break
            x1, z1 = x + heading[0], z + heading[1]
            x2, z2 = x + heading[0] * segment, z + heading[1] * segment
            # Ширина — вбок от оси (вправо по ходу: для +z это -x, для ±x — ±z).
            if heading == (0, 1):
                self.cells.append((min(x1, x2) - (self.width - 1), y, z1, max(x1, x2), y, z2))
            else:
                self.cells.append((min(x1, x2), y, z1, max(x1, x2), y, z2 + (self.width - 1)))
            x, z = x2, z2
            length += segment
        # Пятачок цели — сразу за последней клеткой, по ходу тропы.
        self.goal_cell = (x + heading[0] * (PAD + 1), y, z + heading[1] * (PAD + 1))
        gx, gy, gz = self.goal_cell
        self.cells.append((gx - PAD, gy, gz - PAD, gx + PAD, gy, gz + PAD))
        self.length = length
        self.start_yaw = 10 * rng.randint(-18, 17)  # развернуться к тропе — тоже часть навыка

    def start(self) -> tuple[float, float, float, int]:
        return (self.x + 0.5, DRILL_Y + 1, 0.5, self.start_yaw)

    def goal(self) -> tuple[float, float, float]:
        gx, gy, gz = self.goal_cell
        return (gx + 0.5, gy + 1, gz + 0.5)

    def on_goal(self, x: float, y: float, z: float) -> bool:
        """Дошёл: стоит на пятачке цели (на его уровне)."""
        gx, gy, gz = self.goal_cell
        return (abs(x - (gx + 0.5)) <= PAD + 0.8 and abs(z - (gz + 0.5)) <= PAD + 0.8
                and gy + 0.8 <= y <= gy + 1.3)

    def fills(self) -> list[tuple]:
        return [(*cell, self.block) for cell in self.cells]

    def describe(self) -> str:
        return f"тропа {self.length} шириной {self.width}, поворотов {self.turns}, уступов {self.steps}"


def clear_fill(lane: int) -> tuple:
    """Дорожка целиком — воздухом (21 x 21 x 49 < 32768 — один /fill)."""
    x = lane_x(lane)
    return (x - LANE_HALF, DRILL_Y - CLEAR_BELOW, LANE_BACK, x + LANE_HALF, DRILL_Y + CLEAR_ABOVE, LANE_FAR, "air")


def bounds(lanes: int) -> tuple[int, int, int, int, int, int]:
    """Коробка всех дорожек (мир симуляции, forceload на сервере), с запасом."""
    return (-LANE_HALF - 2, DRILL_Y - CLEAR_BELOW - 2, LANE_BACK - 2,
            lane_x(max(1, lanes) - 1) + LANE_HALF + 2, DRILL_Y + CLEAR_ABOVE + 2, LANE_FAR + 2)


def void_y() -> float:
    return DRILL_Y - VOID_BELOW


def drill_command(lanes: int) -> str:
    """/mcbot drill x0 z0 x1 z1 — плагин загружает участок дорожек в мире
    упражнений и держит его загруженным (fill в незагруженном чанке не
    работает); симуляция по ней переходит в мир упражнений."""
    x0, _, z0, x1, _, z1 = bounds(lanes)
    return f"mcbot drill {x0} {z0} {x1} {z1}"
