"""Сверка симуляции с Node: видит ли бот в симуляции ровно то же, что в игре.

Запуск: python py/sim/check_parity.py

Та же арена строится в prismarine-world (js/sim_reference.js — настоящий
код js/vision.js и js/route.js) и в симуляции (py/sim/world.py), и для
случайных поз бота сравниваются сетка зрения (байты r, g, b, d, класс),
центр прицела, чувство пола (и у краёв мостов трассы bridge), "блок встанет"
(can_place) и маршруты. Меняешь js/vision.js или js/route.js — гоняй:
иначе мозг, обученный в симуляции, в игре будет видеть другое.

Арена здесь поднята на floor_y = 68: prismarine-chunk по умолчанию не держит
отрицательных высот, а зрению и маршрутам высота мира безразлична.
"""

from __future__ import annotations

import json
import math
import random
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = HERE.parent
ROOT = PY.parent
sys.path.insert(0, str(PY))

import numpy as np  # noqa: E402

from config import CONFIG  # noqa: E402
from sim.game import free_for_block, place_front_cell  # noqa: E402
from sim.physics import Body, physics_tick  # noqa: E402
from sim.route import RoutePlanner  # noqa: E402
from sim.vision import EYE_HEIGHT, Vision, ground_probe  # noqa: E402
from sim.world import ArenaWorld  # noqa: E402
from bridge_course import ISLAND, WORLD_Y, BridgeCourse  # noqa: E402

POSES = 60
ROUTES = 40


def standing_spots(world: ArenaWorld, rng: random.Random, count: int) -> list[tuple]:
    """Случайные места, где можно стоять: поверх самого верхнего блока колонки."""
    x0, x1, z0, z1 = world.inner
    spots = []
    while len(spots) < count:
        x, z = rng.randint(x0, x1), rng.randint(z0, z1)
        spots.append((x + rng.uniform(0.3, 0.7), world.top_y(x, z), z + rng.uniform(0.3, 0.7)))
    return spots


def main() -> int:
    rng = random.Random(7)
    arena = dict(CONFIG["server"]["arena"], floor_y=68)
    world = ArenaWorld(arena)
    vision = Vision(CONFIG)

    poses = [{"x": x, "y": y, "z": z, "yaw": rng.uniform(-math.pi, math.pi), "pitch": rng.uniform(-1.3, 1.3)}
             for x, y, z in standing_spots(world, rng, POSES)]
    route_pairs = [{"start": list(a), "target": list(b)}
                   for a, b in zip(standing_spots(world, rng, ROUTES), standing_spots(world, rng, ROUTES))]

    request = {"fills": [list(f) for f in world.fills], "poses": poses, "routes": route_pairs}
    result = subprocess.run(["node", str(ROOT / "js" / "sim_reference.js")], input=json.dumps(request),
                            capture_output=True, text=True, cwd=ROOT, encoding="utf-8")
    if result.returncode != 0:
        print(result.stderr)
        return 1
    reference = json.loads(result.stdout)

    # --- зрение ---
    eyes = np.array([(p["x"], p["y"] + EYE_HEIGHT, p["z"]) for p in poses])
    yaws = np.array([p["yaw"] for p in poses])
    pitches = np.array([p["pitch"] for p in poses])
    grids = vision.grids(world, eyes, yaws)
    centers = vision.center_blocks(world, eyes, yaws, pitches)
    js_grids = np.array(reference["vision"], dtype=np.int64)
    ours = grids.astype(np.int64)
    color_class = np.concatenate([np.arange(0, ours.shape[1], 5)[:, None] + [0, 1, 2, 4]]).ravel()
    distance = np.arange(3, ours.shape[1], 5)
    exact = np.array_equal(ours[:, color_class], js_grids[:, color_class])
    diff = np.abs(ours[:, distance] - js_grids[:, distance])
    print(f"зрение: {len(poses)} поз x {ours.shape[1] // 5} лучей — цвет и класс "
          f"{'совпали' if exact else 'РАЗОШЛИСЬ'}, дальность: расхождений {int((diff > 0).sum())}, "
          f"больше 1 байта {int((diff > 1).sum())}")
    ok = exact and not (diff > 1).any()

    bad_centers = 0
    for ours_c, js_c in zip(centers, reference["centers"]):
        if (ours_c is None) != (js_c is None):
            bad_centers += 1
        elif ours_c is not None and (ours_c["t"] != js_c["t"] or ours_c["name"] != js_c["name"]
                                     or abs(ours_c["distance"] - js_c["distance"]) > 0.011):
            bad_centers += 1
    print(f"центр прицела: {len(centers) - bad_centers}/{len(centers)} совпали")
    ok = ok and bad_centers == 0

    grounds = [ground_probe(world, (p["x"], p["y"], p["z"]), p["yaw"]) for p in poses]
    ok = compare_grounds("чувство пола (арена)", grounds, reference["grounds"]) and ok
    ok = compare_places("блок встанет (арена)", world, poses, reference["places"]) and ok
    ok = check_bridge_grounds(rng) and ok

    # --- маршруты ---
    lengths, waypoints, completes = [], 0, 0
    for pair, js_route in zip(route_pairs, reference["routes"]):
        planner = RoutePlanner(world, CONFIG["route"])
        ours_r = planner.update(tuple(pair["start"]), tuple(pair["target"]))
        if (ours_r is None) != (js_route is None):
            print("маршрут есть только у одного:", pair, ours_r, js_route)
            ok = False
            continue
        if ours_r is None:
            continue
        lengths.append(abs(ours_r["length"] - js_route["length"]))
        completes += ours_r["complete"] == js_route["complete"]
        w, jw = ours_r["waypoint"], js_route["waypoint"]
        waypoints += math.dist((w["x"], w["y"], w["z"]), (jw["x"], jw["y"], jw["z"])) < 1e-9
    print(f"маршруты: {len(lengths)} — длина расходится в среднем на {np.mean(lengths):.3f} блока "
          f"(макс {np.max(lengths):.3f}), 'дошёл до цели' совпало {completes}/{len(lengths)}, "
          f"точка маршрута та же {waypoints}/{len(lengths)} (при равных путях A* может выбрать другой)")
    ok = ok and np.max(lengths) < 0.05 and completes == len(lengths)

    ok = check_physics(world) and ok
    print("СИМУЛЯЦИЯ = ИГРА" if ok else "ЕСТЬ РАСХОЖДЕНИЯ")
    return 0 if ok else 1


def compare_grounds(title: str, ours: list, theirs: list) -> bool:
    bad = sum(1 for a, b in zip(ours, theirs) if any(abs(x - y) > 1e-9 for x, y in zip(a, b)))
    edges = sum(1 for a in ours for x in a if x < 3.0)
    print(f"{title}: {len(ours) - bad}/{len(ours)} совпали (край ближе 3 блоков — в {edges} направлениях)")
    for a, b in list((a, b) for a, b in zip(ours, theirs) if a != b)[:3]:
        print("   симуляция", a, "Node", b)
    return bad == 0


def can_place(world: ArenaWorld, pose: dict) -> bool:
    """can_place симуляции для бота с блоками, одного в мире."""
    pos = (pose["x"], pose["y"], pose["z"])
    cell = place_front_cell(world, (pose["x"], pose["y"] + EYE_HEIGHT, pose["z"]), pose["yaw"], pose["pitch"], pos)
    return cell is not None and free_for_block(world, cell, [pos])


def compare_places(title: str, world: ArenaWorld, poses: list, theirs: list) -> bool:
    ours = [can_place(world, p) for p in poses]
    bad = sum(a != b for a, b in zip(ours, theirs))
    print(f"{title}: {len(ours) - bad}/{len(ours)} совпали (встанет — в {sum(ours)} позах)")
    return bad == 0


def check_bridge_grounds(rng: random.Random) -> bool:
    """Чувство пола у краёв: трасса моста (острова над пустотой) и недостроенные
    мосты разной длины; позы — у краёв, лицом и строго по осям (как на старте
    попытки: угол кратен 10°), и как попало."""
    course = BridgeCourse(3)
    world = ArenaWorld.for_course(course)
    for lane in range(course.lanes):
        x = course.lane_x(lane)
        world._fill(x, WORLD_Y, ISLAND + 1, x, WORLD_Y, ISLAND + lane + 1, "dirt")  # мост в 1..3 блока
    poses = []
    for _ in range(POSES):
        lane = rng.randrange(course.lanes)
        x = course.lane_x(lane) + rng.uniform(-ISLAND - 0.3, ISLAND + 1.3)
        z = rng.uniform(-ISLAND - 0.3, ISLAND + lane + 2.3)
        yaw = math.radians(10 * rng.randint(-18, 17)) if rng.random() < 0.5 else rng.uniform(-math.pi, math.pi)
        # Взгляд -80° (так строят мост: блок встаёт, только когда свесился на
        # 0.2857..0.3) — у половины, у остальных — как попало.
        pitch = math.radians(-80) if rng.random() < 0.5 else rng.uniform(-1.5, 0.5)
        poses.append({"x": x, "y": WORLD_Y + 1 + rng.choice([0.0, 0.0, 0.42, 1.1]), "z": z, "yaw": yaw, "pitch": pitch})
    # И точно у границы окна: спиной к пропасти, взгляд -80°, свес за конец моста
    # 0.27..0.2999 (блок встаёт с 0.2857).
    for lane in range(course.lanes):
        end = ISLAND + lane + 2  # конец мостика (последний блок — ISLAND + lane + 1)
        for overhang in (0.27, 0.28, 0.284, 0.2855, 0.2858, 0.287, 0.29, 0.295, 0.2999):
            poses.append({"x": course.lane_x(lane) + 0.5, "y": WORLD_Y + 1, "z": end + overhang,
                          "yaw": 0.0, "pitch": math.radians(-80)})
    request = {"fills": [list(f) for f in world.fills], "poses": poses, "routes": []}
    result = subprocess.run(["node", str(ROOT / "js" / "sim_reference.js")], input=json.dumps(request),
                            capture_output=True, text=True, cwd=ROOT, encoding="utf-8")
    if result.returncode != 0:
        print(result.stderr)
        return False
    reference = json.loads(result.stdout)
    grounds = [ground_probe(world, (p["x"], p["y"], p["z"]), p["yaw"]) for p in poses]
    same = compare_grounds("чувство пола (мосты)", grounds, reference["grounds"])
    return compare_places("блок встанет (мосты)", world, poses, reference["places"]) and same


def physics_runs(world: ArenaWorld) -> list[dict]:
    """Прогоны "клавиш" по тикам: по ровному, бегом, в прыжке, на уступ, в
    стенку в два блока, назад и вбок."""
    fills = world.fills
    step = next(f for f in fills if f[6] == "oak_planks")    # уступ 3x3 в блок
    wall = next(f for f in fills if f[6] == "stone_bricks")  # стенка в два блока
    floor = world.floor_y

    def facing(start, point):
        return math.atan2(-(point[0] - start[0]), -(point[1] - start[1]))

    def approach(fill, distance):
        cx, cz = (fill[0] + fill[3]) / 2 + 0.5, (fill[2] + fill[5]) / 2 + 0.5
        start = (cx + distance, floor, cz + 0.3)
        return start, facing((start[0], start[2]), (cx, cz))

    runs = []
    open_spot = (world.inner[0] + 3.5, floor, world.inner[2] + 3.5)
    for keys in ({"forward": True}, {"forward": True, "sprint": True}, {"forward": True, "sprint": True, "jump": True},
                 {"back": True, "left": True}):
        runs.append({"start": list(open_spot), "yaw": -2.3, "ticks": [keys] * 60})
    start, yaw = approach(step, 4.0)
    runs.append({"start": list(start), "yaw": yaw, "ticks": [{"forward": True, "jump": True}] * 60})
    start, yaw = approach(wall, 4.0)
    runs.append({"start": list(start), "yaw": yaw, "ticks": [{"forward": True}] * 30 + [{"left": True}] * 30})
    # Крадучись (мост над пустотой): по ровному медленнее, а с уступа задом —
    # не сходит с края (prismarine: откат от края при sneak на земле).
    runs.append({"start": list(open_spot), "yaw": -2.3, "ticks": [{"forward": True, "sneak": True}] * 60})
    top = (
        (step[0] + step[3]) / 2 + 0.5,
        max(step[1], step[4]) + 1,
        (step[2] + step[5]) / 2 + 0.5,
    )
    runs.append({"start": list(top), "yaw": 0.4, "ticks": [{"back": True, "sneak": True}] * 60})
    runs.append({"start": list(top), "yaw": 1.9, "ticks": [{"back": True, "left": True, "sneak": True}] * 60})
    return runs


def check_physics(world: ArenaWorld) -> bool:
    runs = physics_runs(world)
    request = {"fills": [list(f) for f in world.fills], "runs": runs}
    result = subprocess.run(["node", str(ROOT / "js" / "sim_physics_reference.js")], input=json.dumps(request),
                            capture_output=True, text=True, cwd=ROOT, encoding="utf-8")
    if result.returncode != 0:
        print(result.stderr)
        return False
    reference = json.loads(result.stdout)["runs"]
    worst = 0.0
    names = ["шаг", "бег", "бег с прыжками", "назад-влево", "на уступ", "в стенку и вдоль",
             "крадучись", "задом крадучись с уступа", "назад-влево крадучись с уступа"]
    for name, run, track in zip(names, runs, reference):
        body = Body(*run["start"], yaw=run["yaw"])
        errors = []
        for keys, expected in zip(run["ticks"], track):
            body.controls = dict(keys)
            physics_tick(body, world)
            errors.append(math.dist(body.pos, expected))
        worst = max(worst, max(errors))
        moved = math.dist(track[0], track[-1])
        print(f"физика, {name}: за {len(track)} тиков прошёл {moved:.2f} блока, расхождение до {max(errors):.2e}")
    print(f"физика: худшее расхождение {worst:.2e} блока")
    return worst < 1e-6


if __name__ == "__main__":
    sys.exit(main())
