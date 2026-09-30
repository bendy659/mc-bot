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
  - прокопаться к кровати (DigLayout): на островке кровать под куполом из 1–2
    слоёв (шерсть, доски — как её закрывают в бедварсе); сломать кровать,
    ломая всё, что на пути (автор: "ломание блоков для добирания до целей —
    плохо").
  - мини-бедварс 1 на 1 (MiniLayout): два островка с кроватями через пропасть
    в 5–10 блоков, у обоих меч и шерсть — мост навстречу врагу, драка,
    кровать (автор: "когда враг рядом — ссыкуют и стопорятся").
  - защита кровати (GuardLayout): островок защитника с кроватью и пятачок
    нападающего, между ними готовый мост; защитник закрывает кровать шерстью
    и сбивает нападающего, нападающий — сломать кровать (автор: "даже
    кровать не защищают").

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

# Прокопаться к кровати.
DIG_ISLAND = 5          # полуразмер островка: 11 x 11
DIG_LAYERS = (1, 2, 2)  # слоёв укрытия кровати: первый — шерсть, второй — доски
DIG_WOOL = ("white_wool", "red_wool", "blue_wool", "lime_wool", "yellow_wool")
BED_COLOR = "red"       # кровати на картах Hypixel все красные — других симуляция и не знает
BED_FACINGS = {"south": (0, 1), "north": (0, -1), "east": (1, 0), "west": (-1, 0)}  # куда от ног голова

# Мини-бедварс 1 на 1.
MINI_ISLAND = 2         # полуразмер островка: 5 x 5
MINI_GAP = (5, 10)      # пропасть между островками
MINI_BLOCKS = ("stone_bricks", "sandstone", "terracotta", "cobblestone", "oak_planks")

# Защита кровати.
GUARD_ISLAND = 3        # полуразмер островка защитника: 7 x 7
GUARD_PAD = 1           # полуразмер пятачка нападающего: 3 x 3
GUARD_GAP = (7, 12)     # длина готового моста между ними (короче — защитник не успевал закрыть кровать)
GUARD_BRIDGE_WIDTHS = (1, 2, 3)

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


def bed_commands(foot: tuple, facing: str, color: str) -> list[str]:
    """Кровать — две команды setblock (ноги и голова, у каждой своё состояние):
    fill одним блоком ставил бы две "ножные" половины, и игра их убрала бы."""
    dx, dz = BED_FACINGS[facing]
    x, y, z = foot
    return [f"setblock {x} {y} {z} minecraft:{color}_bed[facing={facing},part=foot]",
            f"setblock {x + dx} {y} {z + dz} minecraft:{color}_bed[facing={facing},part=head]"]


class DigLayout:
    """Прокопаться к кровати на дорожке lane: островок, в середине кровать под
    куполом (клетки на "расстоянии" 1..layers от кровати — по горизонтали плюс
    вверх: первый слой — шерсть, второй — доски). Укрытие помечено как
    "поставленное игроком" (placed_box, /mcbot placed) — его можно ломать, как
    в бедварсе; сам островок — нет. Старт — на краю островка, лицом куда попало."""

    def __init__(self, lane: int, rng: random.Random):
        self.lane = lane
        self.x = lane_x(lane)
        cz = DIG_ISLAND + 1
        self.facing = rng.choice(tuple(BED_FACINGS))
        self.color = BED_COLOR
        self.layers = rng.choice(DIG_LAYERS)
        dx, dz = BED_FACINGS[self.facing]
        y = DRILL_Y + 1
        self.foot = (self.x, y, cz)
        self.head = (self.x + dx, y, cz + dz)
        self.cells = [(self.x - DIG_ISLAND, DRILL_Y, cz - DIG_ISLAND, self.x + DIG_ISLAND, DRILL_Y, cz + DIG_ISLAND)]
        self.block = "stone_bricks"  # островок: рукой не сломать, да и нельзя (не "поставлено")
        wool = rng.choice(DIG_WOOL)
        self.cover: list[tuple] = []
        reach = self.layers + 1
        for cx in range(self.x - reach, self.x + reach + 1):
            for cz2 in range(cz - reach, cz + reach + 1):
                for cy in range(y, y + self.layers + 1):
                    if (cx, cy, cz2) in (self.foot, self.head):
                        continue
                    distance = min(abs(cx - hx) + abs(cz2 - hz) for hx, _, hz in (self.foot, self.head)) + (cy - y)
                    if 1 <= distance <= self.layers:
                        self.cover.append((cx, cy, cz2, wool if distance == 1 else "oak_planks"))
        # Старт — на островке (не с края), но не вплотную к укрытию: клетки
        # не ближе layers + 2 от ног кровати (по большей из осей).
        starts = [(sx, sz) for sx in range(self.x - DIG_ISLAND + 1, self.x + DIG_ISLAND)
                  for sz in range(cz - DIG_ISLAND + 1, cz + DIG_ISLAND)
                  if max(abs(sx - self.x), abs(sz - cz)) >= self.layers + 2]
        self.start_cell = rng.choice(starts)
        self.start_yaw = 10 * rng.randint(-18, 17)

    def start(self) -> tuple[float, float, float, int]:
        sx, sz = self.start_cell
        return (sx + 0.5, DRILL_Y + 1, sz + 0.5, self.start_yaw)

    def bed(self) -> dict:
        return {"head": list(self.head), "foot": list(self.foot)}

    def fills(self) -> list[tuple]:
        return [(*cell, self.block) for cell in self.cells] + [(x, y, z, x, y, z, block) for x, y, z, block in self.cover]

    def commands(self) -> list[str]:
        """Кроме fill: кровать и пометка укрытия "поставлено" (ломать можно)."""
        x0, y0, z0 = (min(c[i] for c in self.cover) for i in range(3))
        x1, y1, z1 = (max(c[i] for c in self.cover) for i in range(3))
        return bed_commands(self.foot, self.facing, self.color) + [f"mcbot placed {{world}} {x0} {y0} {z0} {x1} {y1} {z1}"]

    def describe(self) -> str:
        return f"кровать под {self.layers} слоями"


class MiniLayout:
    """Мини-бедварс на дорожке lane: два островка 5 x 5 через пропасть, у
    каждого в заднем ряду кровать (поперёк дорожки), точка появления — в
    середине островка, лицом к сопернику."""

    def __init__(self, lane: int, rng: random.Random):
        self.lane = lane
        self.x = lane_x(lane)
        self.gap = rng.randint(*MINI_GAP)
        self.block = rng.choice(MINI_BLOCKS)
        near = MINI_ISLAND                                    # середина ближнего островка по z
        far = 2 * MINI_ISLAND + 1 + self.gap + MINI_ISLAND    # ...дальнего
        self.cells = [(self.x - MINI_ISLAND, DRILL_Y, near - MINI_ISLAND, self.x + MINI_ISLAND, DRILL_Y, near + MINI_ISLAND),
                      (self.x - MINI_ISLAND, DRILL_Y, far - MINI_ISLAND, self.x + MINI_ISLAND, DRILL_Y, far + MINI_ISLAND)]
        y = DRILL_Y + 1
        # Кровати — в заднем ряду, поперёк дорожки (ноги слева, голова справа).
        self.beds = [((self.x - 1, y, near - MINI_ISLAND), "east"), ((self.x + 1, y, far + MINI_ISLAND), "west")]
        self.spawns = [(self.x + 0.5, y, near + 0.5, 0), (self.x + 0.5, y, far + 0.5, 180)]

    def bed(self, side: int) -> dict:
        foot, facing = self.beds[side]
        dx, dz = BED_FACINGS[facing]
        return {"head": [foot[0] + dx, foot[1], foot[2] + dz], "foot": list(foot)}

    def fills(self) -> list[tuple]:
        return [(*cell, self.block) for cell in self.cells]

    def commands(self) -> list[str]:
        return [command for foot, facing in self.beds for command in bed_commands(foot, facing, BED_COLOR)]

    def describe(self) -> str:
        return f"мини {self.gap}"


class GuardLayout:
    """Защита кровати на дорожке lane: островок защитника (сторона
    defend_side: 0 — ближе к z = 0, 1 — дальше) с кроватью в заднем ряду,
    напротив — пятачок нападающего, между ними уже готовый мост в 1–3 блока.
    Точки появления: защитник — посреди островка, нападающий — посреди
    пятачка, оба лицом к мосту."""

    def __init__(self, lane: int, rng: random.Random, defend_side: int):
        self.lane = lane
        self.x = lane_x(lane)
        self.defend_side = defend_side
        self.gap = rng.randint(*GUARD_GAP)
        self.width = rng.choice(GUARD_BRIDGE_WIDTHS)
        self.block = rng.choice(MINI_BLOCKS)
        halves = [GUARD_ISLAND if side == defend_side else GUARD_PAD for side in (0, 1)]
        near = halves[0]                                      # середина ближней площадки по z
        far = 2 * halves[0] + 1 + self.gap + halves[1]        # ...дальней
        centers = (near, far)
        left = self.x - (self.width - 1) // 2
        self.cells = [(self.x - halves[0], DRILL_Y, near - halves[0], self.x + halves[0], DRILL_Y, near + halves[0]),
                      (left, DRILL_Y, near + halves[0] + 1, left + self.width - 1, DRILL_Y, far - halves[1] - 1),
                      (self.x - halves[1], DRILL_Y, far - halves[1], self.x + halves[1], DRILL_Y, far + halves[1])]
        y = DRILL_Y + 1
        # Кровать — в заднем ряду островка (дальше всего от моста), поперёк дорожки.
        back = near - GUARD_ISLAND if defend_side == 0 else far + GUARD_ISLAND
        self.bed_foot, self.bed_facing = ((self.x - 1, y, back), "east") if defend_side == 0 else ((self.x + 1, y, back), "west")
        self.spawns = [(self.x + 0.5, y, centers[0] + 0.5, 0), (self.x + 0.5, y, centers[1] + 0.5, 180)]

    def bed(self, side: int) -> dict | None:
        if side != self.defend_side:
            return None
        dx, dz = BED_FACINGS[self.bed_facing]
        foot = self.bed_foot
        return {"head": [foot[0] + dx, foot[1], foot[2] + dz], "foot": list(foot)}

    def fills(self) -> list[tuple]:
        return [(*cell, self.block) for cell in self.cells]

    def commands(self) -> list[str]:
        return bed_commands(self.bed_foot, self.bed_facing, BED_COLOR)

    def describe(self) -> str:
        return f"защита, мост {self.width}x{self.gap}"


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
