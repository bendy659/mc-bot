"""Трасса задачки bridge (мост над пустотой): у каждого бота своя дорожка —
остров старта и остров цели, между ними пропасть. Одна раскладка на всё:
симуляция строит по ней мир (py/sim/world.py: ArenaWorld.for_course), сервер
— в мире задачки (config.json bot.task_worlds, команды fill по RCON), судья
(ai_loop._bridge_turn) берёт отсюда, куда ставить бота, где цель и что
убрать после попытки.

Каждая попытка — своя раскладка (BridgeLayout), и вид моста тоже случайный
(автор: "расстояние каждый раз одинаковое"; "у всех — разные вариации
мостов, чтоб боты реально самостоятельно решали проблемы, а не копировали
друг у друга"):
  - STRAIGHT — прямо: пропасть MIN_GAP..MAX_GAP, остров цели сдвинут вбок до
    GOAL_SHIFT, но накрывает колонну старта;
  - UP — остров цели выше на 1..MAX_RISE: сначала столб, потом мост;
  - DOWN — ниже на 1..MAX_DROP: мост до острова и спрыгнуть (засчитано,
    только когда стоит на острове);
  - TURN — остров цели в стороне (TURN_SHIFT блоков вбок): мост буквой "Г" —
    вперёд до ряда острова, поворот, вбок до него.
Старт — в разных точках острова (START_SHIFT). Цель — точка, куда ведёт
правильный мост: у прямых — на острове цели в колонне старта (наискосок мост
не построить: луч прицела шагом 10°), у "Г" — середина острова.

Острова — из каменного кирпича: рукой он копается 7.5 с и ничего не даёт,
так что свой остров бот не разберёт; строит он землёй из инвентаря.
Дорожки — вдоль +z, на расстоянии LANE_SPACING по x: чужой мост не мешает.
"""

from __future__ import annotations

import random

WORLD_Y = 64            # верхний блок острова старта; стоять — на WORLD_Y + 1
ISLAND = 2              # полуразмер острова: 5x5
LANE_SPACING = 20       # между дорожками по x (остров "Г" — до 9 блоков вбок)
MIN_GAP = 3             # пропасть прямого моста — от MIN_GAP до MAX_GAP блоков
MAX_GAP = 12
GAPS = (4, 5, 6, 7, 8, 9, 10, 11, 12)  # раскладка "по умолчанию" (первая постройка трассы)
GOAL_SHIFT = 2          # прямой мост: остров цели сдвинут вбок не больше чем на столько
START_SHIFT = 1         # старт — не дальше стольких блоков от середины острова
MAX_RISE = 2            # UP: остров цели выше на 1..MAX_RISE
MAX_DROP = 3            # DOWN: ниже на 1..MAX_DROP (падать с 3 блоков — без урона)
TURN_SHIFT = (5, 7)     # TURN: остров цели вбок на столько (в любую сторону)
TURN_GAP = (1, 6)       # TURN: пропасть до ряда острова цели
TURN_COST = 2.0         # TURN: поворот на углу стоит как ~2 блока пути (норма времени)
VOID_DEPTH = 12         # ниже верха острова старта на столько — упал в пустоту
ISLAND_BLOCK = "stone_bricks"
LANE_HALF_WIDTH = 9     # что чистить вокруг дорожки (TURN_SHIFT + ISLAND < LANE_HALF_WIDTH + 1)
CLEAR_ABOVE = 48        # и на сколько вверх: сеть, бывает, строит столбы — вживую
                        # чужие столбы до 92-й высоты (выше прежней чистки до +12)
                        # сбивали сеть: она видела их и сама лезла строить (2026-09-29);
                        # дорожка целиком — 19 x 28 x 61 блок, один /fill (до 32768)
FILL_LIMIT = 32768      # /fill берёт не больше стольких блоков за раз

STRAIGHT, UP, DOWN, TURN = "straight", "up", "down", "turn"
KINDS = (STRAIGHT, UP, DOWN, TURN)
KIND_WEIGHTS = (0.4, 0.2, 0.2, 0.2)  # прямых — больше: это основа, остальные — на ней


class BridgeLayout:
    """Одна попытка на дорожке: где стоит бот, где остров цели и какой он."""

    def __init__(self, lane: int, gap: int, start_dx: int = 0, start_dz: int = 0, goal_dx: int = 0,
                 goal_dy: int = 0, kind: str = STRAIGHT):
        self.lane = lane
        self.kind = kind
        self.gap = gap
        self.start_dx = start_dx
        self.start_dz = start_dz
        self.goal_dx = goal_dx
        self.goal_dy = goal_dy
        self.x = lane * LANE_SPACING                 # середина дорожки по x
        self.goal_z = 2 * ISLAND + 1 + gap           # середина острова цели по z
        self.goal_y = WORLD_Y + goal_dy              # верхний блок острова цели

    def start(self) -> tuple[float, float, float]:
        return (self.x + self.start_dx + 0.5, WORLD_Y + 1, self.start_dz + 0.5)

    def goal(self) -> tuple[float, float, float]:
        """Куда ведёт правильный мост: прямой — в колонне старта, "Г" — середина острова."""
        x = self.x + (self.goal_dx if self.kind == TURN else self.start_dx) + 0.5
        return (x, self.goal_y + 1, self.goal_z + 0.5)

    def on_goal_island(self, x: float, y: float, z: float) -> bool:
        """Дошёл: стоит на острове цели (на его уровне, а не над ним — остров
        ниже засчитан, только когда спрыгнул)."""
        cx = self.x + self.goal_dx + 0.5
        return (abs(x - cx) <= ISLAND + 0.8 and abs(z - (self.goal_z + 0.5)) <= ISLAND + 0.8
                and self.goal_y + 0.8 <= y <= self.goal_y + 1.3)

    def blocks_to_cross(self) -> float:
        """Длина пути до острова цели в блоках (для нормы времени): по z до
        ближнего края острова, по x — если остров в стороне, плюс подъём или
        спуск и поворот у "Г"."""
        sx, _, sz = self.start()
        along = (self.goal_z - ISLAND) - sz
        aside = max(0.0, abs(sx - (self.x + self.goal_dx + 0.5)) - (ISLAND + 0.5))
        return along + aside + abs(self.goal_dy) + (TURN_COST if self.kind == TURN else 0.0)

    def islands(self) -> list[tuple]:
        return [
            (self.x - ISLAND, WORLD_Y, -ISLAND, self.x + ISLAND, WORLD_Y, ISLAND, ISLAND_BLOCK),
            (self.x + self.goal_dx - ISLAND, self.goal_y, self.goal_z - ISLAND,
             self.x + self.goal_dx + ISLAND, self.goal_y, self.goal_z + ISLAND, ISLAND_BLOCK),
        ]

    def describe(self) -> str:
        return {STRAIGHT: f"прямо {self.gap}", UP: f"вверх +{self.goal_dy}, {self.gap}",
                DOWN: f"вниз {self.goal_dy}, {self.gap}", TURN: f"Г {self.gap} и вбок {self.goal_dx:+d}"}[self.kind]


class BridgeCourse:
    def __init__(self, lanes: int):
        self.lanes = max(1, lanes)

    def layout(self, lane: int, randomize: bool = False, kind: str | None = None) -> BridgeLayout:
        """Раскладка попытки: randomize — случайная (так судья начинает каждую
        попытку; kind — вид моста, иначе тоже случайный), без него — "по
        умолчанию" (прямо, пропасть по номеру дорожки, всё по середине: первая
        постройка трассы, возрождение в симуляции)."""
        if not randomize:
            return BridgeLayout(lane, GAPS[lane % len(GAPS)])
        kind = kind or random.choices(KINDS, KIND_WEIGHTS)[0]
        start_dx = random.randint(-START_SHIFT, START_SHIFT)
        start_dz = random.randint(-START_SHIFT, START_SHIFT)
        if kind == TURN:
            goal_dx = random.choice((-1, 1)) * random.randint(*TURN_SHIFT)
            return BridgeLayout(lane, random.randint(*TURN_GAP), start_dx, start_dz, goal_dx, 0, TURN)
        # Остров цели сдвинут, но колонну старта накрывает (|сдвиг - старт| <= ISLAND).
        goal_dx = random.randint(max(-GOAL_SHIFT, start_dx - ISLAND), min(GOAL_SHIFT, start_dx + ISLAND))
        if kind == UP:
            return BridgeLayout(lane, random.randint(MIN_GAP, MAX_GAP - 2), start_dx, start_dz, goal_dx,
                                random.randint(1, MAX_RISE), UP)
        if kind == DOWN:
            return BridgeLayout(lane, random.randint(MIN_GAP, MAX_GAP), start_dx, start_dz, goal_dx,
                                -random.randint(1, MAX_DROP), DOWN)
        return BridgeLayout(lane, random.randint(MIN_GAP, MAX_GAP), start_dx, start_dz, goal_dx)

    # Раскладка по умолчанию — для первой постройки трассы и возрождения в симуляции.
    def gap(self, lane: int) -> int:
        return self.layout(lane).gap

    def lane_x(self, lane: int) -> int:
        return lane * LANE_SPACING

    def start(self, lane: int) -> tuple[float, float, float]:
        return self.layout(lane).start()

    def goal(self, lane: int) -> tuple[float, float, float]:
        return self.layout(lane).goal()

    def on_goal_island(self, lane: int, x: float, y: float, z: float) -> bool:
        return self.layout(lane).on_goal_island(x, y, z)

    def fell(self, y: float) -> bool:
        return y < WORLD_Y - VOID_DEPTH

    def islands(self, lane: int) -> list[tuple]:
        return self.layout(lane).islands()

    def fills(self) -> list[tuple]:
        """Вся трасса (раскладка по умолчанию) — (x1, y1, z1, x2, y2, z2, блок), как /fill."""
        return [fill for lane in range(self.lanes) for fill in self.islands(lane)]

    def clear_fills(self) -> list[tuple]:
        """Вся коробка трассы — воздухом, ломтями по x (/fill берёт до
        FILL_LIMIT блоков): в начале работы, чтобы в мире задачки не осталось
        мусора прошлых запусков (старые раскладки, мосты вбок, высокие столбы)."""
        x0, y0, z0, x1, y1, z1 = self.bounds()
        width = max(1, FILL_LIMIT // ((y1 - y0 + 1) * (z1 - z0 + 1)))
        return [(x, y0, z0, min(x + width - 1, x1), y1, z1, "air") for x in range(x0, x1 + 1, width)]

    def reset_fills(self, lane: int, layout: BridgeLayout | None = None) -> list[tuple]:
        """Перед попыткой: всё на дорожке — прочь (от дна пустоты до
        CLEAR_ABOVE над островами: /fill берёт до 32768 блоков), острова
        раскладки — заново."""
        layout = layout or self.layout(lane)
        x = self.lane_x(lane)
        far = 2 * ISLAND + 1 + MAX_GAP + ISLAND + 3
        clear = (x - LANE_HALF_WIDTH, WORLD_Y - VOID_DEPTH, -ISLAND - 3,
                 x + LANE_HALF_WIDTH, WORLD_Y + CLEAR_ABOVE, far, "air")
        return [clear] + layout.islands()

    def bounds(self) -> tuple[int, int, int, int, int, int]:
        """Коробка мира трассы для симуляции: все дорожки, с запасом, снизу —
        на глубину пустоты (за ней — воздух, как пустота мира в игре)."""
        far = 2 * ISLAND + 1 + MAX_GAP + ISLAND + 4
        return (-LANE_HALF_WIDTH - 1, WORLD_Y - VOID_DEPTH - 2, -ISLAND - 4,
                self.lane_x(self.lanes - 1) + LANE_HALF_WIDTH + 1, WORLD_Y + CLEAR_ABOVE + 2, far)


def fill_command(fill: tuple, world: str | None = None) -> str:
    """Команда /fill (в мире задачки — через execute in)."""
    x1, y1, z1, x2, y2, z2, name = fill
    command = f"fill {x1} {y1} {z1} {x2} {y2} {z2} minecraft:{name}"
    return f"execute in minecraft:{world} run {command}" if world else command
