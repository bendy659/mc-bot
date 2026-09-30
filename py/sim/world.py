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
# Реестр блоков: id -> имя (BLOCK_NAMES), состояние (STATES, для карт
# бедварса — "minecraft:oak_stairs[facing=east,...]"), форма и свойства.
# Первые восемь — блоки арены и мостов (полные кубы); состояния карт
# бедварса дописываются при загрузке карты (register_state). Таблицы по id —
# массивы на CAPACITY мест: их можно дописывать на месте, не меняя ссылок.
CAPACITY = 4096
BLOCK_NAMES: list[str] = []
STATES: list[str] = []
BLOCK_ID: dict[str, int] = {}   # имя или состояние -> id (имя — первое его состояние)
SHAPES: list[list[tuple]] = []  # коробки столкновения в клетке (как block.shapes у mineflayer)
COLOR_BYTES = np.zeros((CAPACITY, 3), dtype=np.uint8)
CLASS_BY_ID = np.zeros(CAPACITY, dtype=np.uint8)
COLLIDES = np.zeros(CAPACITY, dtype=bool)       # есть коробки столкновения
FULL_CUBE = np.zeros(CAPACITY, dtype=bool)      # ровно одна коробка на всю клетку
PASS_THROUGH = np.zeros(CAPACITY, dtype=bool)   # луч зрения идёт сквозь (цветы, факелы, рельсы...)
BOX_BLOCK = np.zeros(CAPACITY, dtype=bool)      # boundingBox === 'block' (маршруты, js/route.js)

# Цвет и класс блока в сетке зрения — ровно как у js/vision.js: PALETTE и
# BLOCK_CLASS_PATTERNS по имени блока, по порядку, первое совпадение (bedrock
# попадает в палитру "bed" — важно совпасть с игрой, а не быть правильным).
# Сверка с JS — py/sim/check_parity.py.
PALETTE = [
    (r"grass| moss|fern|leaves|crop|wheat|carrot|potato|beet", (0.30, 0.55, 0.20)),
    (r"dirt|mud|clay|farmland|path", (0.55, 0.38, 0.24)),
    (r"stone|cobble|andesite|diorite|granite|deepslate|tuff|basalt|blackstone", (0.50, 0.50, 0.50)),
    (r"sand|sandstone|gravel", (0.86, 0.80, 0.60)),
    (r"wood|log|plank|oak|birch|spruce|jungle|acacia|dark_oak|mangrove|cherry|fence|stem", (0.65, 0.50, 0.30)),
    (r"water", (0.20, 0.35, 0.80)),
    (r"lava|magma", (0.90, 0.40, 0.10)),
    (r"iron|copper|gold", (0.80, 0.70, 0.50)),
    (r"diamond|emerald", (0.30, 0.90, 0.80)),
    (r"redstone|netherrack|nether_brick", (0.60, 0.15, 0.15)),
    (r"snow|ice|packed_ice|frosted", (0.90, 0.95, 1.00)),
    (r"glass|glass_pane", (0.70, 0.85, 0.90)),
    (r"wool|carpet|bed|concrete|terracotta", (0.75, 0.60, 0.55)),
    (r"brick", (0.65, 0.35, 0.30)),
    (r"obsidian|coal|bedrock", (0.15, 0.10, 0.25)),
]
UNKNOWN_COLOR = (0.45, 0.45, 0.45)
CLASS_PATTERNS = [
    (r"^log$|_log$|^wood$|planks?|fence|stem|bamboo", 1),
    (r"stone|cobble|andesite|diorite|granite|deepslate|tuff|basalt|blackstone|brick|obsidian|bedrock|concrete"
     r"|terracotta|netherrack|end_stone", 2),
    (r"dirt|mud|clay|farmland|path|grass_block|sand|gravel|powder_snow", 3),
    (r"grass|leaves|moss|fern|crop|wheat|carrot|potato|beet|flower|vine|fungus|seagrass|kelp", 4),
    (r"water|lava", 5),
    (r"ore|ancient_debris|amethyst", 6),
    (r"snow|ice", 7),
    (r"glass", 8),
]


def vision_color(name: str) -> tuple:
    return next((color for pattern, color in PALETTE if re.search(pattern, name)), UNKNOWN_COLOR)


def vision_class(name: str) -> int:
    if name in ("air", "cave_air", "void_air"):
        return 0
    return next((class_id for pattern, class_id in CLASS_PATTERNS if re.search(pattern, name)), 9)


# Копка — как в игре (Block.getDestroyProgress): за тик прогресс 1 / прочность
# / (30, если блок добывается рукой, иначе 100), в воздухе — впятеро медленнее.
# Прочность -1 — не ломается. Каменный кирпич рукой копается 7.5 с и ничего не
# даёт (нужна кирка); дёрн даёт землю. По имени блока (у всех состояний одна).
HARDNESS = {"air": -1.0}
HAND_HARVEST: set[str] = set()
DROPS = {"grass_block": "dirt"}
SKY_COLOR = (0.5, 0.7, 1.0)  # луч ни во что не попал: класс 0, дальность 1

# Плоский мир — и за стенами арены, на столько блоков: на сервере он там
# продолжается, и чувство пола у стены (vision.ground_probe, до 3 блоков)
# должно видеть за ней пол, как в игре. Зрению и маршрутам это всё равно —
# барьер их не пропускает.
GROUND_MARGIN = 4
# Над картой бедварса — столько блоков запаса на постройки (выше сетки —
# воздух, ставить туда нельзя: Hypixel тоже ограничивает высоту).
BEDWARS_BUILD_ABOVE = 16

FILL = re.compile(r"fill (-?\d+) (-?\d+) (-?\d+) (-?\d+) (-?\d+) (-?\d+) minecraft:(\w+)")


def js_byte(value: float) -> int:
    """Как toByte в js/state.js: Math.round(clamp(v, 0, 1) * 255). Python-овский
    round округляет 0.5 к чётному, а Math.round — вверх: для 0.5*255 это
    разные байты."""
    return int(np.floor(min(max(value, 0.0), 1.0) * 255.0 + 0.5))


FULL = [(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)]


def register_state(state: str, name: str, shapes: list, bounding_box: str = "block", pass_through: bool = False,
                   hardness: float | None = None, hand_harvest: bool = True) -> int:
    """id состояния блока (новое — дописать в реестр). shapes — коробки
    [x0, y0, z0, x1, y1, z1] в клетке (0..1), как block.shapes mineflayer."""
    if state in BLOCK_ID and STATES[BLOCK_ID[state]] == state:
        return BLOCK_ID[state]
    block_id = len(STATES)
    if block_id >= CAPACITY:
        raise ValueError("реестр блоков симуляции полон (CAPACITY)")
    STATES.append(state)
    BLOCK_NAMES.append(name)
    BLOCK_ID[state] = block_id
    BLOCK_ID.setdefault(name, block_id)
    boxes = [tuple(float(v) for v in box) for box in shapes]
    SHAPES.append(boxes)
    if block_id != AIR:
        COLOR_BYTES[block_id] = [js_byte(c) for c in vision_color(name)]
        CLASS_BY_ID[block_id] = vision_class(name)
        COLLIDES[block_id] = bool(boxes)
        FULL_CUBE[block_id] = boxes == FULL
        PASS_THROUGH[block_id] = pass_through
        BOX_BLOCK[block_id] = bounding_box == "block"
        if name not in HARDNESS:
            HARDNESS[name] = -1.0 if hardness is None else float(hardness)
            if hand_harvest:
                HAND_HARVEST.add(name)
    return block_id


# Блоки арены и мостов (все — полные кубы). Прочность и "рукой" — как у игры.
register_state("air", "air", [], "empty")
for _name, _hardness, _hand in (("grass_block", 0.6, True), ("dirt", 0.5, True), ("bedrock", -1.0, False),
                                ("barrier", -1.0, False), ("oak_planks", 2.0, True), ("stone_bricks", 1.5, False),
                                ("oak_log", 2.0, True)):
    register_state(_name, _name, FULL, "block", False, _hardness, _hand)
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
        self.blocks = np.zeros(tuple(self.size), dtype=np.uint16)
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
        world.blocks = np.zeros(tuple(world.size), dtype=np.uint16)
        world.route_cache = {}
        world.inner = (x0, x1, z0, z1)
        world.fills = []
        for fill in course.fills():
            world._fill(*fill)
        world.snapshot()
        return world

    @classmethod
    def for_bedwars_map(cls, name: str, maps_dir=None) -> "ArenaWorld":
        """Мир карты бедварса: сетка data/bedwars/maps/<карта>.npz (блоки —
        состояния, как их перевёл сервер) и формы блоков из Node
        (data/bedwars/block_shapes.json). Вокруг и снизу — пустота, как в
        мире задачки на сервере. Сверху — запас на постройки."""
        import json
        from pathlib import Path

        maps_dir = Path(maps_dir) if maps_dir else Path(__file__).resolve().parents[2] / "data" / "bedwars" / "maps"
        grid = np.load(maps_dir / f"{name}.npz")
        shapes = json.loads((maps_dir.parent / "block_shapes.json").read_text(encoding="utf-8"))
        ids = np.zeros(len(grid["palette"]), dtype=np.uint16)
        for index, state in enumerate(str(s) for s in grid["palette"]):
            if state == "minecraft:air":
                continue
            info = shapes[state]
            ids[index] = register_state(state, info["name"], info["shapes"], info["boundingBox"], info["passThrough"],
                                        info["hardness"], info["handHarvest"])
        for state, info in shapes.items():  # шерсть команд и прочее, что ставят боты
            register_state(state, info["name"], info["shapes"], info["boundingBox"], info["passThrough"],
                           info["hardness"], info["handHarvest"])
        world = cls.__new__(cls)
        world.arena = None
        blocks = ids[grid["blocks"]]
        above = BEDWARS_BUILD_ABOVE
        world.origin = np.array(grid["origin"], dtype=np.int64)
        world.size = np.array([blocks.shape[0], blocks.shape[1] + above, blocks.shape[2]], dtype=np.int64)
        world.blocks = np.zeros(tuple(world.size), dtype=np.uint16)
        world.blocks[:, :blocks.shape[1], :] = blocks
        world.floor_y = int(world.origin[1])
        world.route_cache = {}
        world.inner = (int(world.origin[0]), int(world.origin[0] + world.size[0] - 1),
                       int(world.origin[2]), int(world.origin[2] + world.size[2] - 1))
        world.fills = []
        world.map_name = name
        world.snapshot()
        return world

    @classmethod
    def empty(cls, box: tuple, name: str) -> "ArenaWorld":
        """Пустой мир в коробке box = (x0, y0, z0, x1, y1, z1): упражнения
        бедварса (py/bedwars_drills.py) — площадки строит судья командами fill,
        как в мире задачки на сервере без карты (/mcbot bedwars void)."""
        world = cls.__new__(cls)
        world.arena = None
        x0, y0, z0, x1, y1, z1 = box
        world.origin = np.array([x0, y0, z0], dtype=np.int64)
        world.size = np.array([x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1], dtype=np.int64)
        world.blocks = np.zeros(tuple(world.size), dtype=np.uint16)
        world.floor_y = y0
        world.route_cache = {}
        world.inner = (x0, x1, z0, z1)
        world.fills = []
        world.map_name = name
        world.snapshot()
        return world

    def fill(self, ax: int, ay: int, az: int, bx: int, by: int, bz: int, name: str) -> None:
        """/fill судьи: как _fill, но не в список построек мира (fills — для
        сверки с Node: упражнения перестраивают дорожки тысячи раз)."""
        lo = np.maximum(np.minimum([ax, ay, az], [bx, by, bz]) - self.origin, 0)
        hi = np.minimum(np.maximum([ax, ay, az], [bx, by, bz]) - self.origin + 1, self.size)
        if np.any(lo >= hi):
            return
        self.blocks[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = BLOCK_ID[name]
        self.route_cache.clear()

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
        """Есть ли у блока коробки столкновения (у арены и мостов — все полные кубы)."""
        return bool(COLLIDES[self.block(x, y, z)])

    def box_block(self, x: int, y: int, z: int) -> bool:
        """boundingBox === 'block' у mineflayer (автопрыжок, маршруты)."""
        return bool(BOX_BLOCK[self.block(x, y, z)])

    def boxes(self, x: int, y: int, z: int) -> list[tuple]:
        """Коробки столкновения блока в мировых координатах."""
        return [(x + a, y + b, z + c, x + d, y + e, z + f) for a, b, c, d, e, f in SHAPES[self.block(x, y, z)]]

    def ids(self, cells: np.ndarray) -> np.ndarray:
        """id блоков для массива мировых клеток (M, 3); вне сетки — воздух."""
        local = cells - self.origin
        inside = np.all((local >= 0) & (local < self.size), axis=1)
        out = np.zeros(len(cells), dtype=np.uint16)
        idx = local[inside]
        out[inside] = self.blocks[idx[:, 0], idx[:, 1], idx[:, 2]]
        return out

    def top_y(self, x: int, z: int) -> int:
        """Высота, на которой стоят ноги поверх самого высокого блока колонки
        (как /spreadplayers ставит игроков)."""
        ix, iz = x - self.origin[0], z - self.origin[2]
        column = np.nonzero(self.blocks[ix, :, iz])[0]
        return int(self.origin[1] + column.max() + 1) if len(column) else self.floor_y
