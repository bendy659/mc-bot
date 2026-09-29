"""Мир симуляции: блоки арены — из тех же команд /fill, которыми её строит
сервер (py/arena.py), плюс слои плоского мира под полом (бедрок, две
земли, дёрн — как у "плоского" мира сервера). Блоки можно ломать и ставить
(set_block; время копки — SimArena по HARDNESS), reset() возвращает арену
как построена (после каждой охоты и раунда салок — как /mcbot restore в игре).

Координаты — мировые, как в игре (арена вокруг center, пол на floor_y), —
чтобы состояния ботов выглядели ровно как от Node.
"""

from __future__ import annotations

import re

import numpy as np

from arena import arena_commands, bounds

AIR = 0
BLOCK_NAMES = ["air", "grass_block", "dirt", "bedrock", "barrier", "oak_planks", "stone_bricks", "oak_log"]
BLOCK_ID = {name: index for index, name in enumerate(BLOCK_NAMES)}

# Цвет и класс блока в сетке зрения — ровно как у js/vision.js (PALETTE и
# BLOCK_CLASS_PATTERNS). Снято с самого JS; bedrock там попадает в палитру
# "bed" (кровать) — так и оставлено: важно совпасть с игрой, а не быть
# правильным. Сверка с JS — py/sim/check_parity.py.
VISION_COLOR = {
    "grass_block": (0.30, 0.55, 0.20),
    "dirt": (0.55, 0.38, 0.24),
    "bedrock": (0.75, 0.60, 0.55),
    "barrier": (0.45, 0.45, 0.45),
    "oak_planks": (0.65, 0.50, 0.30),
    "stone_bricks": (0.50, 0.50, 0.50),
    "oak_log": (0.65, 0.50, 0.30),
}
VISION_CLASS = {"grass_block": 3, "dirt": 3, "bedrock": 2, "barrier": 9, "oak_planks": 1, "stone_bricks": 2, "oak_log": 1}
# Копка — как в игре (Block.getDestroyProgress): за тик прогресс 1 / прочность
# / (30, если блок добывается рукой, иначе 100), в воздухе — впятеро медленнее.
# Прочность -1 — не ломается. Каменный кирпич рукой копается 7.5 с и ничего не
# даёт (нужна кирка); дёрн даёт землю.
HARDNESS = {"grass_block": 0.6, "dirt": 0.5, "bedrock": -1.0, "barrier": -1.0, "oak_planks": 2.0,
            "stone_bricks": 1.5, "oak_log": 2.0}
HAND_HARVEST = {"grass_block", "dirt", "oak_planks", "oak_log"}
DROPS = {"grass_block": "dirt"}
SKY_COLOR = (0.5, 0.7, 1.0)  # луч ни во что не попал: класс 0, дальность 1

# Плоский мир — и за стенами арены, на столько блоков: на сервере он там
# продолжается, и чувство пола у стены (vision.ground_probe, до 3 блоков)
# должно видеть за ней пол, как в игре. Зрению и маршрутам это всё равно —
# барьер их не пропускает.
GROUND_MARGIN = 4

FILL = re.compile(r"fill (-?\d+) (-?\d+) (-?\d+) (-?\d+) (-?\d+) (-?\d+) minecraft:(\w+)")


def js_byte(value: float) -> int:
    """Как toByte в js/state.js: Math.round(clamp(v, 0, 1) * 255). Python-овский
    round округляет 0.5 к чётному, а Math.round — вверх: для 0.5*255 это
    разные байты."""
    return int(np.floor(min(max(value, 0.0), 1.0) * 255.0 + 0.5))


# Байты цвета и класс по id блока — для быстрой сборки сетки зрения.
COLOR_BYTES = np.zeros((len(BLOCK_NAMES), 3), dtype=np.uint8)
CLASS_BY_ID = np.zeros(len(BLOCK_NAMES), dtype=np.uint8)
for _name, _color in VISION_COLOR.items():
    COLOR_BYTES[BLOCK_ID[_name]] = [js_byte(c) for c in _color]
    CLASS_BY_ID[BLOCK_ID[_name]] = VISION_CLASS[_name]
SKY_BYTES = np.array([js_byte(c) for c in SKY_COLOR], dtype=np.uint8)


class ArenaWorld:
    def __init__(self, arena: dict):
        self.arena = arena
        x0, x1, z0, z1 = bounds(arena)
        self.floor_y = arena["floor_y"]
        top = self.floor_y + arena["height"]
        # Сетка: от бедрока до верха арены с запасом, по x/z — вместе со
        # стенами и полом за ними (GROUND_MARGIN). Всё за её пределами —
        # воздух (за стены лучам и так не пройти: они из барьера и выше головы).
        margin = GROUND_MARGIN
        self.origin = np.array([x0 - 1 - margin, self.floor_y - 4, z0 - 1 - margin], dtype=np.int64)
        self.size = np.array([x1 - x0 + 3 + 2 * margin, top - (self.floor_y - 4) + 2, z1 - z0 + 3 + 2 * margin],
                             dtype=np.int64)
        self.blocks = np.zeros(tuple(self.size), dtype=np.uint8)
        # Соседи клеток для маршрутов (sim/route.py) — общие для всех ботов,
        # пока блоки не менялись; любая смена блока кэш очищает.
        self.route_cache: dict = {}
        self.inner = (x0, x1, z0, z1)
        # Все заливки по порядку — чтобы построить тот же мир в JS для сверки
        # (py/sim/check_parity.py).
        self.fills: list[tuple] = []

        # Плоский мир под полом (на всю сетку: под стенами и за ними тоже).
        wx0, wz0 = x0 - 1 - margin, z0 - 1 - margin
        wx1, wz1 = x1 + 1 + margin, z1 + 1 + margin
        self._fill(wx0, self.floor_y - 4, wz0, wx1, self.floor_y - 4, wz1, "bedrock")
        self._fill(wx0, self.floor_y - 3, wz0, wx1, self.floor_y - 2, wz1, "dirt")
        self._fill(wx0, self.floor_y - 1, wz0, wx1, self.floor_y - 1, wz1, "grass_block")
        for command in arena_commands(arena):
            match = FILL.match(command)
            if match is None:
                raise ValueError(f"Не понял команду арены: {command}")
            *coords, name = match.groups()
            self._fill(*map(int, coords), name)
        self.snapshot()

    @classmethod
    def for_course(cls, course) -> "ArenaWorld":
        """Мир задачки с трассой (bridge_course.BridgeCourse): только её
        блоки, вокруг и снизу — пустота (за сеткой — воздух, как пустота мира
        задачки на сервере). Пол для остального кода — верх островов."""
        world = cls.__new__(cls)
        world.arena = None
        x0, y0, z0, x1, y1, z1 = course.bounds()
        world.floor_y = course.start(0)[1]
        world.origin = np.array([x0, y0, z0], dtype=np.int64)
        world.size = np.array([x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1], dtype=np.int64)
        world.blocks = np.zeros(tuple(world.size), dtype=np.uint8)
        world.route_cache = {}
        world.inner = (x0, x1, z0, z1)
        world.fills = []
        for fill in course.fills():
            world._fill(*fill)
        world.snapshot()
        return world

    def snapshot(self) -> None:
        """Запомнить арену как есть (после постройки) — её вернёт reset()."""
        self.original = self.blocks.copy()

    def reset(self) -> None:
        self.blocks[...] = self.original
        self.route_cache.clear()

    def inside(self, x: int, y: int, z: int) -> bool:
        ix, iy, iz = x - self.origin[0], y - self.origin[1], z - self.origin[2]
        return 0 <= ix < self.size[0] and 0 <= iy < self.size[1] and 0 <= iz < self.size[2]

    def set_block(self, x: int, y: int, z: int, name: str) -> bool:
        """Поставить/убрать ("air") блок; вне сетки — нельзя (False)."""
        if not self.inside(x, y, z):
            return False
        self.blocks[x - self.origin[0], y - self.origin[1], z - self.origin[2]] = BLOCK_ID[name]
        self.route_cache.clear()
        return True

    def _fill(self, ax: int, ay: int, az: int, bx: int, by: int, bz: int, name: str) -> None:
        self.fills.append((ax, ay, az, bx, by, bz, name))
        lo = np.maximum(np.minimum([ax, ay, az], [bx, by, bz]) - self.origin, 0)
        hi = np.minimum(np.maximum([ax, ay, az], [bx, by, bz]) - self.origin + 1, self.size)
        if np.any(lo >= hi):
            return
        self.blocks[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = BLOCK_ID[name]
        self.route_cache.clear()

    def block(self, x: int, y: int, z: int) -> int:
        """id блока в мировой клетке (x, y, z); вне сетки — воздух."""
        ix, iy, iz = x - self.origin[0], y - self.origin[1], z - self.origin[2]
        if 0 <= ix < self.size[0] and 0 <= iy < self.size[1] and 0 <= iz < self.size[2]:
            return int(self.blocks[ix, iy, iz])
        return AIR

    def solid(self, x: int, y: int, z: int) -> bool:
        return self.block(x, y, z) != AIR

    def ids(self, cells: np.ndarray) -> np.ndarray:
        """id блоков для массива мировых клеток (M, 3); вне сетки — воздух."""
        local = cells - self.origin
        inside = np.all((local >= 0) & (local < self.size), axis=1)
        out = np.zeros(len(cells), dtype=np.uint8)
        idx = local[inside]
        out[inside] = self.blocks[idx[:, 0], idx[:, 1], idx[:, 2]]
        return out

    def top_y(self, x: int, z: int) -> int:
        """Высота, на которой стоят ноги поверх самого высокого блока колонки
        (как /spreadplayers ставит игроков)."""
        ix, iz = x - self.origin[0], z - self.origin[2]
        column = np.nonzero(self.blocks[ix, :, iz])[0]
        return int(self.origin[1] + column.max() + 1) if len(column) else self.floor_y
