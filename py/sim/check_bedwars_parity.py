"""Сверка симуляции с Node на картах бедварса (как check_parity.py на
арене): тот же мир карты (все блоки, состояния — как их перевёл сервер) в
prismarine-world версии сервера; зрение, центр прицела, чувство пола, "блок
встанет", маршруты и физика (ступени, плиты, заборы, песок душ, пустота)
должны совпасть.

Запуск: python py/sim/check_bedwars_parity.py [карта ... | --all]  (по
умолчанию три разные: Airshow, Lighthouse, Ashfire)
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
from sim.check_parity import compare_grounds, compare_places  # noqa: E402
from sim.physics import SPEED_FACTOR, Body, physics_tick  # noqa: E402
from sim.route import RoutePlanner  # noqa: E402
from sim.vision import EYE_HEIGHT, Vision, ground_probe  # noqa: E402
from sim.world import BLOCK_NAMES, COLLIDES, SHAPES, STATES, ArenaWorld  # noqa: E402

VERSION = "26.1"  # как у сервера (Paper 26.1.2) и js/export_block_shapes.js
POSES = 60
ROUTES = 30
RUNS = 60
TICKS = 60
DEFAULT_MAPS = ["Airshow", "Lighthouse", "Ashfire"]


def spots(world: ArenaWorld, rng: random.Random, count: int) -> list[tuple]:
    """Случайные места, где можно стоять: верх блока с коробками, над ним
    два блока без коробок. Высота ног — верх его формы (плита — на полблока)."""
    solid = np.argwhere(COLLIDES[world.blocks])
    out = []
    while len(out) < count:
        ix, iy, iz = solid[rng.randrange(len(solid))]
        x, y, z = (int(v) for v in world.origin + (ix, iy, iz))
        if world.solid(x, y + 1, z) or world.solid(x, y + 2, z):
            continue
        top = max(box[4] for box in SHAPES[world.block(x, y, z)])
        out.append((x + rng.uniform(0.3, 0.7), y + top, z + rng.uniform(0.3, 0.7)))
    return out


def not_comparable(world: ArenaWorld, pos) -> bool:
    """Песок душ (prismarine для 26.1 не замедляет, сервер замедляет —
    симуляция как сервер) и вода (в симуляции её нет: на всех картах 90
    блоков воды, пруды-украшения) — такие прогоны не сверяются."""
    x, y, z = (math.floor(v) for v in pos)
    for dx in (-1, 0, 1):
        for dz in (-1, 0, 1):
            for level in (y - 1, y, y + 1):
                name = BLOCK_NAMES[world.block(x + dx, level, z + dz)]
                if name in SPEED_FACTOR or name == "water":
                    return True
    return False


def run_node(script: str, request: dict) -> dict | None:
    result = subprocess.run(["node", str(ROOT / "js" / script)], input=json.dumps(request),
                            capture_output=True, text=True, cwd=ROOT, encoding="utf-8")
    if result.returncode != 0:
        print(result.stderr)
        return None
    return json.loads(result.stdout)


def check_map(name: str, rng: random.Random) -> bool:
    world = ArenaWorld.for_bedwars_map(name)
    cells = np.argwhere(world.blocks > 0)
    states = [[int(x), int(y), int(z), STATES[world.blocks[ix, iy, iz]]]
              for (ix, iy, iz), (x, y, z) in zip(cells, cells + world.origin)]
    print(f"--- {name}: блоков {len(states)}")

    poses = [{"x": x, "y": y, "z": z, "yaw": rng.uniform(-math.pi, math.pi), "pitch": rng.uniform(-1.3, 1.3)}
             for x, y, z in spots(world, rng, POSES)]
    starts = spots(world, rng, ROUTES)
    # Маршрут — к точке неподалёку (на том же острове или через мост): через
    # пустоту маршрута нет ни у кого, это сверять неинтересно.
    routes = []
    for start in starts:
        near = [p for p in spots(world, rng, 40) if 4 < math.dist(p, start) < 24]
        if near:
            routes.append({"start": list(start), "target": list(near[0])})
    request = {"version": VERSION, "states": states, "poses": poses, "routes": routes}
    reference = run_node("sim_reference.js", request)
    if reference is None:
        return False

    vision = Vision(CONFIG)
    eyes = np.array([(p["x"], p["y"] + EYE_HEIGHT, p["z"]) for p in poses])
    yaws = np.array([p["yaw"] for p in poses])
    pitches = np.array([p["pitch"] for p in poses])
    ours = vision.grids(world, eyes, yaws).astype(np.int64)
    theirs = np.array(reference["vision"], dtype=np.int64)
    color_class = (np.arange(0, ours.shape[1], 5)[:, None] + [0, 1, 2, 4]).ravel()
    distance = np.arange(3, ours.shape[1], 5)
    exact = np.array_equal(ours[:, color_class], theirs[:, color_class])
    diff = np.abs(ours[:, distance] - theirs[:, distance])
    print(f"зрение: {len(poses)} поз — цвет и класс {'совпали' if exact else 'РАЗОШЛИСЬ'} "
          f"(разных лучей {int((ours[:, color_class] != theirs[:, color_class]).any(axis=0).sum())}), "
          f"дальность: расхождений {int((diff > 0).sum())}, больше 1 байта {int((diff > 1).sum())}")
    ok = exact and not (diff > 1).any()

    bad = 0
    for ours_c, js_c in zip(vision.center_blocks(world, eyes, yaws, pitches), reference["centers"]):
        if (ours_c is None) != (js_c is None) or (ours_c is not None and (
                ours_c["t"] != js_c["t"] or ours_c["name"] != js_c["name"]
                or abs(ours_c["distance"] - js_c["distance"]) > 0.011)):
            bad += 1
            if bad <= 3:
                print("  прицел разошёлся:", ours_c, js_c)
    print(f"центр прицела: {len(poses) - bad}/{len(poses)} совпали")
    ok = ok and bad == 0

    grounds = [ground_probe(world, (p["x"], p["y"], p["z"]), p["yaw"]) for p in poses]
    ok = compare_grounds("чувство пола", grounds, reference["grounds"]) and ok
    ok = compare_places("блок встанет", world, poses, reference["places"]) and ok

    lengths, completes = [], 0
    for pair, js_route in zip(routes, reference["routes"]):
        ours_r = RoutePlanner(world, CONFIG["route"]).update(tuple(pair["start"]), tuple(pair["target"]))
        if (ours_r is None) != (js_route is None):
            print("  маршрут есть только у одного:", pair, ours_r, js_route)
            ok = False
            continue
        if ours_r is None:
            continue
        lengths.append(abs(ours_r["length"] - js_route["length"]))
        completes += ours_r["complete"] == js_route["complete"]
    if lengths:
        print(f"маршруты: {len(lengths)} — длина расходится до {max(lengths):.3f}, "
              f"'дошёл до цели' совпало {completes}/{len(lengths)}")
        ok = ok and max(lengths) < 0.05 and completes == len(lengths)

    # Физика: случайные "клавиши" с мест на карте — ступени, плиты, края, пустота.
    keysets = [{"forward": True}, {"forward": True, "sprint": True}, {"forward": True, "jump": True},
               {"forward": True, "sprint": True, "jump": True}, {"back": True, "sneak": True},
               {"left": True}, {"forward": True, "right": True}]
    runs = []
    for x, y, z in spots(world, rng, RUNS):
        ticks = []
        while len(ticks) < TICKS:
            ticks += [rng.choice(keysets)] * rng.randint(5, 20)
        runs.append({"start": [x, y, z], "yaw": rng.uniform(-math.pi, math.pi), "ticks": ticks[:TICKS]})
    physics = run_node("sim_physics_reference.js", {"version": VERSION, "states": states, "runs": runs})
    if physics is None:
        return False
    worst, moved, skipped = 0.0, [], 0
    for run, track in zip(runs, physics["runs"]):
        if any(not_comparable(world, point) for point in track):
            skipped += 1
            continue
        body = Body(*run["start"], yaw=run["yaw"])
        for keys, expected in zip(run["ticks"], track):
            body.controls = dict(keys)
            physics_tick(body, world)
            error = math.dist(body.pos, expected)
            if error > worst:
                worst = error
                worst_at = (run["start"], expected, body.pos)
        moved.append(math.dist(track[0], track[-1]))
    print(f"физика: {len(runs)} прогонов по {TICKS} тиков (в среднем прошёл {np.mean(moved):.1f} блока), "
          f"худшее расхождение {worst:.2e} блока" + (f" — {worst_at}" if worst > 1e-6 else "")
          + (f"; через воду и песок душ — {skipped} (не сверялись)" if skipped else ""))
    return ok and worst < 1e-6


def main() -> int:
    rng = random.Random(11)
    ok = True
    names = sys.argv[1:] or DEFAULT_MAPS
    if names == ["--all"]:
        names = sorted(p.stem for p in (ROOT / "data" / "bedwars" / "maps").glob("*.npz"))
    for name in names:
        ok = check_map(name, rng) and ok
    print("СИМУЛЯЦИЯ = ИГРА (карты бедварса)" if ok else "ЕСТЬ РАСХОЖДЕНИЯ")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
