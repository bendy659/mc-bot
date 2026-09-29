"""Маршруты в симуляции — перенос js/route.js один в один: A* по клеткам,
где могут стоять ноги (8 соседей, запрыгнуть на блок, спрыгнуть до
max_drop), точка маршрута на lookahead клеток впереди, длина — по
маршруту. Мир симуляции не меняется, поэтому "что за блок" — просто
твёрдый/воздух (опасных блоков и жидкостей на арене нет).
"""

from __future__ import annotations

import math

from .world import ArenaWorld

SQRT2 = math.sqrt(2)


class MinHeap:
    """Куча по f — ровно как MinHeap в js/route.js (не heapq): при равных f
    порядок извлечения зависит от устройства кучи, а от него — какой из
    равноценных маршрутов выберет A*. Так маршрут выходит тем же, что в игре."""

    def __init__(self):
        self.items = []  # [f, node, g]

    def push(self, item) -> None:
        items = self.items
        items.append(item)
        i = len(items) - 1
        while i > 0:
            parent = (i - 1) >> 1
            if items[parent][0] <= items[i][0]:
                break
            items[parent], items[i] = items[i], items[parent]
            i = parent

    def pop(self):
        items = self.items
        top = items[0]
        last = items.pop()
        if items:
            items[0] = last
            i = 0
            while True:
                left = 2 * i + 1
                right = left + 1
                smallest = i
                if left < len(items) and items[left][0] < items[smallest][0]:
                    smallest = left
                if right < len(items) and items[right][0] < items[smallest][0]:
                    smallest = right
                if smallest == i:
                    break
                items[smallest], items[i] = items[i], items[smallest]
                i = smallest
        return top


class RoutePlanner:
    def __init__(self, world: ArenaWorld, config: dict):
        self.world = world
        self.max_nodes = config.get("max_nodes", 3000)
        self.max_drop = config.get("max_drop", 3)
        self.lookahead = config.get("lookahead", 3)
        self.replan_ticks = config.get("replan_ticks", 5)
        self.max_distance = config.get("max_distance", 48)
        self.path = None        # [(x, y, z, step)] — клетки от старта до цели
        self.remaining = None   # длина маршрута от клетки i до конца
        self.complete = False
        self.index = 0
        self.ticks_since_plan = 0
        self.planned_target = None

    # --- что за клетка ------------------------------------------------------

    def kind(self, x: int, y: int, z: int) -> str:
        # Вне сетки мира — как незагруженный чанк у JS: не знаем, не идём.
        w = self.world
        ix, iy, iz = x - w.origin[0], y - w.origin[1], z - w.origin[2]
        if not (0 <= ix < w.size[0] and 0 <= iy < w.size[1] and 0 <= iz < w.size[2]):
            return "blocked"
        return "solid" if w.blocks[ix, iy, iz] else "passable"

    def standable(self, x: int, y: int, z: int) -> bool:
        return self.kind(x, y - 1, z) == "solid" and self.kind(x, y, z) == "passable" and self.kind(x, y + 1, z) == "passable"

    def clear(self, x: int, y: int, z: int) -> bool:
        return self.kind(x, y, z) == "passable" and self.kind(x, y + 1, z) == "passable"

    def ground_cell(self, position) -> tuple | None:
        x = math.floor(position[0])
        z = math.floor(position[2])
        top = math.floor(position[1] + 0.3)
        for y in range(top, top - 4, -1):
            if self.standable(x, y, z):
                return (x, y, z)
        return None

    def neighbours(self, node):
        # Соседи клетки зависят только от блоков вокруг — одни и те же для
        # всех ботов, пока мир не менялся (кэш мира чистится при каждой смене
        # блока). Без кэша A* каждого бота раз в replan_ticks считал их заново,
        # и недостижимая цель (в коробке, на столбе) стоила max_nodes клеток
        # на бота — симуляция шла вчетверо медленнее реального времени.
        cached = self.world.route_cache.get(node)
        if cached is not None:
            return cached
        out = self._neighbours(node)
        self.world.route_cache[node] = out
        return out

    def _neighbours(self, node):
        x, y, z = node
        out = []
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dz == 0:
                    continue
                diagonal = dx != 0 and dz != 0
                nx, nz = x + dx, z + dz
                if diagonal and not (self.clear(x + dx, y, z) and self.clear(x, y, z + dz)):
                    continue
                if self.standable(nx, y, nz):
                    out.append(((nx, y, nz), SQRT2 if diagonal else 1.0, SQRT2 if diagonal else 1.0))
                    continue
                if diagonal:
                    continue
                if self.standable(nx, y + 1, nz) and self.kind(x, y + 2, z) == "passable":
                    out.append(((nx, y + 1, nz), SQRT2, 2.0))
                    continue
                if not self.clear(nx, y, nz):
                    continue
                for drop in range(1, self.max_drop + 1):
                    if self.standable(nx, y - drop, nz):
                        out.append(((nx, y - drop, nz), math.hypot(1, drop), 1 + 0.5 * drop))
                        break
                    if self.kind(nx, y - drop, nz) != "passable":
                        break
        return out  # [(клетка, длина шага, цена шага)]

    # --- A* -----------------------------------------------------------------

    def plan(self, start, goal):
        gx, gy, gz = goal

        def h(n):  # как в js/route.js: октильная + полблока за уровень высоты
            dx, dz = abs(n[0] - gx), abs(n[2] - gz)
            return max(dx, dz) + (SQRT2 - 1) * min(dx, dz) + 0.5 * abs(n[1] - gy)

        open_heap = MinHeap()
        open_heap.push([h(start), start, 0.0])
        best = {start: 0.0}
        parents = {}
        closest = (start, h(start))
        expanded = 0
        while open_heap.items and expanded < self.max_nodes:
            _, node, g = open_heap.pop()
            if g > best[node]:
                continue  # устаревшая запись
            expanded += 1
            distance = h(node)
            if distance < closest[1]:
                closest = (node, distance)
            if math.hypot(node[0] - gx, node[2] - gz) <= 1.5 and abs(node[1] - gy) <= 1:
                return self.unwind(parents, node), True
            for nxt, step, cost in self.neighbours(node):
                if math.hypot(nxt[0] - start[0], nxt[2] - start[2]) > self.max_distance:
                    continue
                g2 = g + cost
                if g2 >= best.get(nxt, math.inf):
                    continue
                best[nxt] = g2
                parents[nxt] = (node, step)
                open_heap.push([g2 + h(nxt), nxt, g2])
        return self.unwind(parents, closest[0]), False

    @staticmethod
    def unwind(parents, node):
        path = [[node, 0.0]]
        while node in parents:
            parent, step = parents[node]
            path[0][1] = step  # длина шага ИЗ родителя в эту клетку
            path.insert(0, [parent, 0.0])
            node = parent
        return path

    # --- как RoutePlanner.update в JS ----------------------------------------

    def update(self, position, target) -> dict | None:
        """position, target — (x, y, z). Возвращает {"waypoint", "length",
        "complete"} или None, как state.route от Node."""
        if target is None:
            self.path = None
            return None
        target_moved = (self.planned_target is None
                        or math.hypot(target[0] - self.planned_target[0], target[2] - self.planned_target[2]) > 2
                        or abs(target[1] - self.planned_target[1]) > 1.5)
        deviated = self.path is not None and self._distance_to(self.path[self.index][0], position) > 2.5
        self.ticks_since_plan += 1
        if self.path is None or target_moved or deviated or self.ticks_since_plan >= self.replan_ticks:
            self.replan(position, target)
        if not self.path:
            return None

        best_index, best_distance = self.index, math.inf
        last = min(len(self.path) - 1, self.index + 6)
        for i in range(self.index, last + 1):
            d = self._distance_to(self.path[i][0], position)
            if d < best_distance:
                best_distance, best_index = d, i
        self.index = best_index
        wx, wy, wz = self.path[min(self.index + self.lookahead, len(self.path) - 1)][0]
        length = best_distance + self.remaining[self.index]
        if not self.complete:
            ex, ey, ez = self.path[-1][0]
            length += math.hypot(ex + 0.5 - target[0], ey - target[1], ez + 0.5 - target[2])
        return {"waypoint": {"x": wx + 0.5, "y": wy, "z": wz + 0.5},
                "length": math.floor(length * 100 + 0.5) / 100, "complete": self.complete}

    def replan(self, position, target) -> None:
        self.ticks_since_plan = 0
        self.planned_target = tuple(target)
        start = self.ground_cell(position)
        goal = self.ground_cell(target) or (math.floor(target[0]), math.floor(target[1]), math.floor(target[2]))
        if start is None:
            return  # в воздухе — оставить старый маршрут, если он есть
        path, complete = self.plan(start, goal)
        self.path, self.complete, self.index = path, complete, 0
        self.remaining = [0.0] * len(path)
        for i in range(len(path) - 2, -1, -1):
            self.remaining[i] = self.remaining[i + 1] + path[i + 1][1]

    @staticmethod
    def _distance_to(node, position) -> float:
        return math.hypot(node[0] + 0.5 - position[0], node[1] - position[1], node[2] + 0.5 - position[2])
