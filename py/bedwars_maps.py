"""Карты бедварса (server/bw_maps/<карта>/ — миры Hypixel 1.8, скачанные
WorldDownloader'ом) -> описание карты для судьи, сервера и симуляции:
data/bedwars/maps/<карта>.json.

Запуск (из корня проекта):
    python py/bedwars_maps.py              # все карты
    python py/bedwars_maps.py Airshow Hollow
    python py/bedwars_maps.py --convert    # сетки для симуляции (нужен запущенный сервер, см. convert)
    python py/bedwars_maps.py --convert --server <папка сервера> Airshow
    python py/bedwars_maps.py --shapes     # формы блоков из Node -> data/bedwars/block_shapes.json

Что в описании: границы карты, острова (сверху — связные области блоков),
восемь команд (кровать, цвет, где появляться, где генератор), точки
генераторов алмазов и изумрудов. Генераторов и магазинов в самих мирах нет (на Hypixel
это сущности, в скачанном мире их не осталось) — их места выводятся из
раскладки островов (см. describe); JSON можно поправить руками, тогда
перегенерировать его нельзя (ключ "manual": true — не перезаписывать).

Мир на сервер ставится копией регионов (region/*.mca): Paper 26.1 сам
переводит чанки 1.8 в новый формат при загрузке (проверено 2026-09-30:
кровати, цвет шерсти на месте). Числовые id блоков 1.8 нужны только здесь,
для разбора.
"""

from __future__ import annotations

import gzip
import io
import json
import math
import struct
import sys
import time
import zlib
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MAPS_DIR = ROOT / "server" / "bw_maps"
OUT_DIR = ROOT / "data" / "bedwars" / "maps"

# Карта — квадрат вокруг (0, 0) не больше этого (RoofTop скачан вместе с
# лобби Hypixel далеко на востоке — его не берём).
MAP_LIMIT = 130
# Острова меньше стольких клеток сверху — украшения (облачка, лампы).
MIN_ISLAND_CELLS = 30

# id блоков 1.8, которые нужны разбору.
BED = 26
# Не опора под ноги: воздух, жидкости, растения, факелы, таблички, лестницы,
# рельсы, кнопки, нажимные плиты, паутина, провода, лианы, ковёр.
NOT_FLOOR = {0, 6, 8, 9, 10, 11, 30, 31, 32, 37, 38, 39, 40, 50, 51, 55, 59, 63, 65, 66, 68, 69, 70, 72,
             75, 76, 77, 78, 83, 90, 104, 105, 106, 111, 115, 131, 132, 141, 142, 143, 147, 148, 171, 175}


# --- чтение мира 1.8: region/*.mca -> NBT чанков -> блоки ------------------

def read_nbt(data: bytes):
    """Минимальный разбор NBT (все теги 1.8): -> (имя, значение). Составные —
    dict, списки — list, массивы байт — bytes, массивы int — list."""
    stream = io.BytesIO(data)

    def read(fmt):
        size = struct.calcsize(fmt)
        return struct.unpack(">" + fmt, stream.read(size))[0]

    def payload(tag):
        if tag == 1:
            return read("b")
        if tag == 2:
            return read("h")
        if tag == 3:
            return read("i")
        if tag == 4:
            return read("q")
        if tag == 5:
            return read("f")
        if tag == 6:
            return read("d")
        if tag == 7:
            return stream.read(read("i"))
        if tag == 8:
            return stream.read(read("H")).decode("utf-8", "replace")
        if tag == 9:
            item_tag, count = read("b"), read("i")
            return [payload(item_tag) for _ in range(count)]
        if tag == 10:
            result = {}
            while True:
                child = read("b")
                if child == 0:
                    return result
                name = stream.read(read("H")).decode("utf-8", "replace")
                result[name] = payload(child)
        if tag == 11:
            return [read("i") for _ in range(read("i"))]
        if tag == 12:
            return [read("q") for _ in range(read("i"))]
        raise ValueError(f"неизвестный тег NBT {tag}")

    tag = read("b")
    name = stream.read(read("H")).decode("utf-8", "replace")
    return name, payload(tag)


def read_chunks(world: Path):
    """Все чанки мира: Level-составные из region/r.*.mca."""
    for path in sorted((world / "region").glob("r.*.mca")):
        data = path.read_bytes()
        for index in range(1024):
            offset = int.from_bytes(data[index * 4:index * 4 + 3], "big") * 4096
            if offset == 0 or offset >= len(data):
                continue
            length = int.from_bytes(data[offset:offset + 4], "big")
            compression = data[offset + 4]
            raw = data[offset + 5:offset + 4 + length]
            raw = zlib.decompress(raw) if compression == 2 else gzip.decompress(raw)
            yield read_nbt(raw)[1]["Level"]


def read_blocks(world: Path) -> dict[tuple[int, int, int], tuple[int, int]]:
    """Непустые блоки карты: (x, y, z) -> (id, метаданные). Только в пределах
    MAP_LIMIT; старший байт id (Add) картам Hypixel не нужен (id < 256)."""
    blocks = {}
    for level in read_chunks(world):
        chunk_x, chunk_z = level["xPos"] * 16, level["zPos"] * 16
        if abs(chunk_x) > MAP_LIMIT + 16 or abs(chunk_z) > MAP_LIMIT + 16:
            continue
        for section in level.get("Sections", []):
            base_y = section["Y"] * 16
            ids, meta = section["Blocks"], section["Data"]
            for index, block_id in enumerate(ids):
                if block_id == 0:
                    continue
                x = chunk_x + (index & 15)
                z = chunk_z + ((index >> 4) & 15)
                if abs(x) > MAP_LIMIT or abs(z) > MAP_LIMIT:
                    continue
                y = base_y + (index >> 8)
                nibble = meta[index >> 1]
                blocks[(x, y, z)] = (block_id, (nibble >> 4) & 15 if index & 1 else nibble & 15)
    return blocks


def read_modern_blocks(regions: Path, area: tuple[int, int, int, int] | None = None) -> dict[tuple[int, int, int], str]:
    """Непустые блоки карты из регионов НОВОГО формата (после того как сервер
    перевёл карту: /mcbot bedwars <карта>, прогрузить чанки, save-all):
    (x, y, z) -> состояние блока "minecraft:oak_stairs[facing=east,...]".
    Имена — ровно те, что видит игра (зрение по имени блока), поэтому
    симуляция берёт карту отсюда, а не переводит id 1.8 сама. area — (x0, z0,
    x1, z1) карты: чанки вне её не читаются (и могут быть не переведены)."""
    x0, z0, x1, z1 = area or (-MAP_LIMIT, -MAP_LIMIT, MAP_LIMIT, MAP_LIMIT)

    def outside(chunk_x: int, chunk_z: int) -> bool:
        return chunk_x + 15 < x0 or chunk_x > x1 or chunk_z + 15 < z0 or chunk_z > z1

    blocks = {}
    for path in sorted(regions.glob("r.*.mca")):
        data = path.read_bytes()
        for index in range(1024):
            offset = int.from_bytes(data[index * 4:index * 4 + 3], "big") * 4096
            if offset == 0 or offset >= len(data):
                continue
            length = int.from_bytes(data[offset:offset + 4], "big")
            raw = data[offset + 5:offset + 4 + length]
            chunk = read_nbt(zlib.decompress(raw) if data[offset + 4] == 2 else gzip.decompress(raw))[1]
            if "Level" in chunk:
                # Чанк 1.8, который сервер не загружал, — не переведён. За
                # пределами карты это нормально (WorldDownloader сохранял
                # регионы целиком), внутри — карту прогрузили не всю.
                level_x, level_z = chunk["Level"]["xPos"] * 16, chunk["Level"]["zPos"] * 16
                if outside(level_x, level_z):
                    continue
                raise ValueError(f"{path.name}: чанк ({level_x}, {level_z}) не переведён — прогрузи всю карту и save-all")
            chunk_x, chunk_z = chunk["xPos"] * 16, chunk["zPos"] * 16
            if outside(chunk_x, chunk_z):
                continue
            for section in chunk.get("sections", []):
                states = section.get("block_states")
                if not states:
                    continue
                palette = [state_string(entry) for entry in states["palette"]]
                if len(palette) == 1:
                    if palette[0] != "minecraft:air":
                        for index_in in range(4096):
                            blocks[section_cell(chunk_x, section["Y"], chunk_z, index_in)] = palette[0]
                    continue
                # Индексы палитры упакованы в long-и, не переходя границу long-а (с 1.16).
                bits = max(4, (len(palette) - 1).bit_length())
                per_long = 64 // bits
                mask = (1 << bits) - 1
                for index_in in range(4096):
                    word = states["data"][index_in // per_long] & 0xFFFFFFFFFFFFFFFF
                    state = palette[(word >> (bits * (index_in % per_long))) & mask]
                    if state != "minecraft:air":
                        blocks[section_cell(chunk_x, section["Y"], chunk_z, index_in)] = state
    return blocks


def section_cell(chunk_x: int, section_y: int, chunk_z: int, index: int) -> tuple[int, int, int]:
    return chunk_x + (index & 15), section_y * 16 + (index >> 8), chunk_z + ((index >> 4) & 15)


def state_string(entry: dict) -> str:
    properties = entry.get("Properties")
    if not properties:
        return entry["Name"]
    return entry["Name"] + "[" + ",".join(f"{key}={value}" for key, value in sorted(properties.items())) + "]"


# --- разбор карты ----------------------------------------------------------

def find_islands(blocks: dict) -> list[dict]:
    """Острова — связные (с диагоналями) области клеток x/z, над которыми
    есть хоть один блок. Мелочь (< MIN_ISLAND_CELLS) — украшения, не острова."""
    cells = {(x, z) for x, _, z in blocks}
    seen, islands = set(), []
    for start in cells:
        if start in seen:
            continue
        seen.add(start)
        queue, members = deque([start]), []
        while queue:
            x, z = queue.popleft()
            members.append((x, z))
            for dx in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    other = (x + dx, z + dz)
                    if other in cells and other not in seen:
                        seen.add(other)
                        queue.append(other)
        if len(members) >= MIN_ISLAND_CELLS:
            cx = sum(x for x, _ in members) / len(members)
            cz = sum(z for _, z in members) / len(members)
            islands.append({"cells": set(members), "center": (cx, cz), "radius": math.hypot(cx, cz)})
    islands.sort(key=lambda island: island["radius"])
    return islands


def stand_points(blocks: dict, cells: set) -> list[tuple[int, int, int]]:
    """Где на острове можно стоять: верхний блок столбца — опора, над ним
    два блока воздуха. -> (x, y ног, z)."""
    tops: dict[tuple[int, int], int] = {}
    for (x, y, z), (block_id, _) in blocks.items():
        if (x, z) in cells and block_id not in NOT_FLOOR and y > tops.get((x, z), -999):
            tops[(x, z)] = y
    points = []
    for (x, z), y in tops.items():
        if all(blocks.get((x, y + dy, z), (0, 0))[0] in NOT_FLOOR for dy in (1, 2)):
            points.append((x, y + 1, z))
    return points


def nearest_point(points: list, target: tuple[float, float]) -> tuple[int, int, int] | None:
    return min(points, key=lambda p: (p[0] - target[0]) ** 2 + (p[2] - target[1]) ** 2, default=None)


# Цвета команд — по кругу (против часовой, от востока), в порядке Hypixel.
# По блокам острова цвет не угадать: на картах везде свой цвет темы.
TEAM_COLORS = ["red", "blue", "lime", "yellow", "cyan", "white", "pink", "gray"]
# Сколько генераторов алмазов на карте (у Hypixel на 8 команд — 4).
DIAMOND_COUNT = 4


def describe(name: str, blocks: dict) -> dict:
    """Описание карты.

    Команды — острова с одной кроватью (не в центре: на Lighthouse в центре
    кровати-украшения). Появляются в середине острова, генератор железа и
    золота — там же (на Hypixel кузница рядом с точкой появления).
    Алмазы — DIAMOND_COUNT самых больших островов, кроме командных и
    центрального; если их меньше (Waterfall: алмазные острова срослись с
    центром мостами) — точки на полпути к командам, между соседними парами.
    Изумруды — точка, где можно стоять, ближе всего к центру карты."""
    islands = find_islands(blocks)
    heads = [(x, y, z) for (x, y, z), (block_id, meta) in blocks.items() if block_id == BED and meta >= 8]
    all_points = {id(island): stand_points(blocks, island["cells"]) for island in islands}

    teams = []
    for island in islands:
        on_island = [head for head in heads if (head[0], head[2]) in island["cells"]]
        if len(on_island) != 1 or island["radius"] < 20:
            continue
        head = on_island[0]
        foot = next(((head[0] + dx, head[1], head[2] + dz) for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1))
                     if blocks.get((head[0] + dx, head[1], head[2] + dz), (0, 0))[0] == BED), head)
        spawn = nearest_point(all_points[id(island)], island["center"])
        teams.append({"island": island, "bed": {"head": list(head), "foot": list(foot)},
                      "spawn": list(spawn) if spawn else None,
                      "angle": math.atan2(island["center"][1], island["center"][0]) % (2 * math.pi)})
    teams.sort(key=lambda team: team["angle"])
    team_ids = {id(team["island"]) for team in teams}

    others = [island for island in islands if id(island) not in team_ids]
    center = min(others, key=lambda island: island["radius"], default=None)
    points_off_teams = [p for island in others for p in all_points[id(island)]]
    emerald = nearest_point(points_off_teams, (0.0, 0.0))

    side_islands = sorted((island for island in others if island is not center),
                          key=lambda island: len(island["cells"]), reverse=True)[:DIAMOND_COUNT]
    diamonds = [nearest_point(all_points[id(island)], island["center"]) for island in side_islands]
    if len(diamonds) < DIAMOND_COUNT and teams:
        team_radius = sum(team["island"]["radius"] for team in teams) / len(teams)
        diamonds = []
        for index in range(0, len(teams), 2):  # между командами 0-1, 2-3, ... — половина пар
            a = teams[index]["angle"]
            b = teams[(index + 1) % len(teams)]["angle"]
            angle = math.atan2(math.sin(a) + math.sin(b), math.cos(a) + math.cos(b))
            target = (math.cos(angle) * team_radius / 2, math.sin(angle) * team_radius / 2)
            diamonds.append(nearest_point(points_off_teams, target))
    diamonds = [list(point) for point in diamonds if point]

    xs = [x for x, _, _ in blocks]
    ys = [y for _, y, _ in blocks]
    zs = [z for _, _, z in blocks]
    kinds = {id(island): "diamond" for island in side_islands if len(side_islands) == DIAMOND_COUNT}
    kinds.update({island_id: "team" for island_id in team_ids})
    if center is not None:
        kinds[id(center)] = "center"
    return {
        "name": name,
        "manual": False,
        "bounds": {"min": [min(xs), min(ys), min(zs)], "max": [max(xs), max(ys), max(zs)]},
        # Ниже самого нижнего блока карты — пустота (упал — умер).
        "void_y": min(ys) - 10,
        "teams": [{"color": TEAM_COLORS[index % len(TEAM_COLORS)], "bed": team["bed"], "spawn": team["spawn"],
                   "generator": team["spawn"]} for index, team in enumerate(teams)],
        "diamonds": diamonds,
        "emeralds": [list(emerald)] if emerald else [],
        "islands": [{"center": [round(i["center"][0], 1), round(i["center"][1], 1)], "cells": len(i["cells"]),
                     "kind": kinds.get(id(i), "other")} for i in islands],
    }


# --- карта для симуляции: сетка блоков в новом формате ---------------------

def save_grid(name: str, blocks: dict[tuple[int, int, int], str]) -> Path:
    """Сетка карты для симуляции: data/bedwars/maps/<карта>.npz —
    origin (x, y, z самого угла), blocks[x, y, z] — номер в palette
    (0 — воздух), palette — состояния блоков как у сервера."""
    palette = ["minecraft:air"] + sorted(set(blocks.values()))
    number = {state: index for index, state in enumerate(palette)}
    xs, ys, zs = zip(*blocks)
    origin = np.array([min(xs), min(ys), min(zs)], dtype=np.int32)
    grid = np.zeros((max(xs) - origin[0] + 1, max(ys) - origin[1] + 1, max(zs) - origin[2] + 1), dtype=np.uint16)
    for (x, y, z), state in blocks.items():
        grid[x - origin[0], y - origin[1], z - origin[2]] = number[state]
    out = OUT_DIR / f"{name}.npz"
    np.savez_compressed(out, origin=origin, blocks=grid, palette=np.array(palette))
    return out


def convert(names: list[str], server_dir: Path) -> int:
    """Карты -> сетки для симуляции ЧЕРЕЗ СЕРВЕР: он сам переводит чанки 1.8
    (так имена блоков в симуляции — ровно те же, что в игре). Нужен запущенный
    сервер с плагином (RCON из config.json; MCBOT_CONFIG — тестовый конфиг)
    и мир задачки bedwars. Мир задачки при этом перезагружается."""
    from config import CONFIG
    from rcon import Rcon

    rcon_cfg = CONFIG["server"]["rcon"]
    rcon = Rcon(rcon_cfg.get("host", "127.0.0.1"), rcon_cfg["port"], rcon_cfg["password"])
    rcon.connect()
    world = CONFIG["bot"]["task_worlds"]["bedwars"]
    regions = server_dir / "world" / "dimensions" / "minecraft" / world / "region"
    in_world = f"execute in minecraft:{world} run "
    for name in names or sorted(p.name for p in MAPS_DIR.iterdir() if p.is_dir()):
        print(f"{name}: {rcon.command('mcbot bedwars ' + name).strip()}")
        description = json.loads((OUT_DIR / f"{name}.json").read_text(encoding="utf-8"))
        (x0, _, z0), (x1, _, z1) = description["bounds"]["min"], description["bounds"]["max"]
        middle = (z0 + z1) // 2
        # Прогрузить всю карту: forceload — не больше 256 чанков за раз.
        rcon.command(in_world + "forceload remove all")
        rcon.command(in_world + f"forceload add {x0} {z0} {x1} {middle}")
        rcon.command(in_world + f"forceload add {x0} {middle + 1} {x1} {z1}")
        for attempt in range(30):
            time.sleep(3)
            rcon.command("save-all flush")
            try:
                blocks = read_modern_blocks(regions, (x0, z0, x1, z1))
            except ValueError as err:
                if attempt == 29:
                    raise
                print(f"  ещё грузится ({err})")
                continue
            break
        rcon.command(in_world + "forceload remove all")
        blocks = {p: state for p, state in blocks.items() if x0 <= p[0] <= x1 and z0 <= p[2] <= z1}
        out = save_grid(name, blocks)
        print(f"  блоков {len(blocks)}, разных {len(set(blocks.values()))} -> {out.name} ({out.stat().st_size // 1024} КБ)")
    rcon.close()
    return 0


# Блоки, которые появляются в игре кроме блоков карт: их ставят игроки и
# боты (шерсть цвета команды — как на Hypixel), в симуляции — тоже.
EXTRA_STATES = [f"minecraft:{color}_wool" for color in TEAM_COLORS] + ["minecraft:dirt"]
SHAPES_FILE = OUT_DIR.parent / "block_shapes.json"


def export_shapes() -> int:
    """Формы и свойства всех состояний блоков карт (и EXTRA_STATES) — из Node,
    как их видит mineflayer (js/export_block_shapes.js) -> data/bedwars/block_shapes.json."""
    import subprocess
    import tempfile

    states = set(EXTRA_STATES)
    for grid in sorted(OUT_DIR.glob("*.npz")):
        states.update(str(state) for state in np.load(grid)["palette"])
    states.discard("minecraft:air")
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(sorted(states), f)
    subprocess.run(["node", str(ROOT / "js" / "export_block_shapes.js"), f.name, str(SHAPES_FILE)], check=True)
    Path(f.name).unlink()
    return 0


def main(names: list[str]) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if names[:1] == ["--shapes"]:
        return export_shapes()
    if names[:1] == ["--convert"]:
        server_dir = Path(names[2]) if names[1:2] == ["--server"] else ROOT / "server"
        return convert(names[3:] if names[1:2] == ["--server"] else names[1:], server_dir)
    worlds = [MAPS_DIR / name for name in names] if names else sorted(p for p in MAPS_DIR.iterdir() if p.is_dir())
    for world in worlds:
        out = OUT_DIR / f"{world.name}.json"
        if out.exists() and json.loads(out.read_text(encoding="utf-8")).get("manual"):
            print(f"{world.name}: правлен руками (manual) — не трогаю.")
            continue
        description = describe(world.name, read_blocks(world))
        out.write_text(json.dumps(description, ensure_ascii=False, indent=1), encoding="utf-8")
        teams = description["teams"]
        print(f"{world.name}: команд {len(teams)} ({', '.join(t['color'] for t in teams)}), "
              f"алмазов {len(description['diamonds'])}, изумрудов {len(description['emeralds'])}, "
              f"пустота ниже y={description['void_y']}"
              + ("" if all(t["spawn"] for t in teams) else " — НЕ У ВСЕХ КОМАНД НАШЛАСЬ ТОЧКА ПОЯВЛЕНИЯ"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
