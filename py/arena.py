"""Арена для обучения на нашем сервере (server/): ровное поле size x size
(плоский мир), по краю — стена из барьера (невидима и нерушима: боты не
уйдут), внутри — препятствия (выбор автора: "плоский + препятствия"):

  - уступы в блок (доски 3x3) — запрыгивать (автопрыжок в js/actions.js);
  - стенки в два блока (каменный кирпич) — обходить, прятаться в салках;
  - столбы в три блока (брёвна) — укрытия и ориентиры;
  - кучки земли 2x2 — её быстро копать руками: вот и блоки для столба.

Раскладка случайная, но всегда одна и та же (seed) и не трогает круг у
спавна. arena_commands() — команды /fill для сервера (TrainingServer
строит арену один раз на мир).
"""

from __future__ import annotations

import random


def bounds(arena: dict) -> tuple[int, int, int, int]:
    """(x0, x1, z0, z1) — включительно, клетки внутри стен."""
    cx, cz = arena["center"]
    half = arena["size"] // 2
    return cx - half, cx + half - 1, cz - half, cz + half - 1


def arena_commands(arena: dict) -> list[str]:
    x0, x1, z0, z1 = bounds(arena)
    y = arena["floor_y"]  # уровень ног на полу арены
    top = y + arena["height"]
    wall_top = y + arena["wall_height"]
    commands = []

    # Расчистить. fill — не больше 32768 блоков за раз, поэтому ломтями по x:
    # с height 12 хватало двух половин, а с 48 половина — уже 100 тысяч, и
    # сервер такой fill не выполнил бы.
    slice_width = max(1, 32768 // ((top - y + 1) * (z1 - z0 + 1)))
    for x in range(x0, x1 + 1, slice_width):
        commands.append(f"fill {x} {y} {z0} {min(x + slice_width - 1, x1)} {top} {z1} minecraft:air")
    # Стены из барьера по краю.
    commands.append(f"fill {x0 - 1} {y} {z0 - 1} {x1 + 1} {wall_top} {z0 - 1} minecraft:barrier")
    commands.append(f"fill {x0 - 1} {y} {z1 + 1} {x1 + 1} {wall_top} {z1 + 1} minecraft:barrier")
    commands.append(f"fill {x0 - 1} {y} {z0} {x0 - 1} {wall_top} {z1} minecraft:barrier")
    commands.append(f"fill {x1 + 1} {y} {z0} {x1 + 1} {wall_top} {z1} minecraft:barrier")

    rng = random.Random(arena.get("seed", 7))
    cx, cz = arena["center"]
    clear = arena.get("spawn_clear_radius", 6)

    def spot(width: int, depth: int) -> tuple[int, int]:
        """Угол препятствия width x depth: внутри арены с отступом и не у спавна."""
        while True:
            x = rng.randint(x0 + 2, x1 - 1 - width)
            z = rng.randint(z0 + 2, z1 - 1 - depth)
            if max(abs(x + width / 2 - cx), abs(z + depth / 2 - cz)) > clear + max(width, depth) / 2:
                return x, z

    for _ in range(arena.get("steps", 8)):  # уступы в блок
        x, z = spot(3, 3)
        commands.append(f"fill {x} {y} {z} {x + 2} {y} {z + 2} minecraft:oak_planks")
    for _ in range(arena.get("walls", 8)):  # стенки в два блока
        length = rng.randint(4, 8)
        if rng.random() < 0.5:
            x, z = spot(length, 1)
            commands.append(f"fill {x} {y} {z} {x + length - 1} {y + 1} {z} minecraft:stone_bricks")
        else:
            x, z = spot(1, length)
            commands.append(f"fill {x} {y} {z} {x} {y + 1} {z + length - 1} minecraft:stone_bricks")
    for _ in range(arena.get("pillars", 6)):  # столбы в три блока
        x, z = spot(1, 1)
        commands.append(f"fill {x} {y} {z} {x} {y + 2} {z} minecraft:oak_log")
    for _ in range(arena.get("dirt_piles", 6)):  # кучки земли — копать на блоки
        x, z = spot(2, 2)
        commands.append(f"fill {x} {y} {z} {x + 1} {y} {z + 1} minecraft:dirt")
    return commands
