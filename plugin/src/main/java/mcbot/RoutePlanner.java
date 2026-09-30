package mcbot;

import com.google.gson.JsonObject;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import net.minecraft.world.phys.Vec3;

/**
 * Маршрут к цели в обход препятствий — перенос js/route.js один в один
 * (та же куча, тот же порядок соседей, та же эвристика): сеть ходьбы идёт к
 * ближайшей точке маршрута (state.route), и симуляция (py/sim/route.py)
 * сверена с JS — так же обязан считать и плагин.
 */
final class RoutePlanner {
    private record Node(int x, int y, int z, double g, double f, long key) {
    }

    private record Step(int x, int y, int z, double step, double cost) {
    }

    private record Parent(int x, int y, int z, double step, long key) {
    }

    private record Cell(int x, int y, int z, double step) {
    }

    /** js/route.js: MinHeap — своя куча, чтобы при равных f порядок был как в JS. */
    private static final class MinHeap {
        private final List<Node> items = new ArrayList<>();

        int size() {
            return items.size();
        }

        void push(Node node) {
            items.add(node);
            int i = items.size() - 1;
            while (i > 0) {
                int parent = (i - 1) >> 1;
                if (items.get(parent).f <= items.get(i).f) break;
                swap(parent, i);
                i = parent;
            }
        }

        Node pop() {
            Node top = items.get(0);
            Node last = items.remove(items.size() - 1);
            if (!items.isEmpty()) {
                items.set(0, last);
                int i = 0;
                for (;;) {
                    int left = 2 * i + 1;
                    int right = left + 1;
                    int smallest = i;
                    if (left < items.size() && items.get(left).f < items.get(smallest).f) smallest = left;
                    if (right < items.size() && items.get(right).f < items.get(smallest).f) smallest = right;
                    if (smallest == i) break;
                    swap(smallest, i);
                    i = smallest;
                }
            }
            return top;
        }

        private void swap(int a, int b) {
            Node t = items.get(a);
            items.set(a, items.get(b));
            items.set(b, t);
        }
    }

    private static final double SQRT2 = Math.sqrt(2);

    private final int maxNodes;
    private final int maxDrop;
    private final int lookahead;
    private final int replanTicks;
    private final double maxDistance;
    private WorldView world;
    private List<Cell> path;
    private double[] remaining;
    private boolean complete;
    private int index;
    int ticksSincePlan;
    private Vec3 plannedTarget;

    RoutePlanner(ProjectConfig config) {
        maxNodes = config.integer("route.max_nodes", 3000);
        maxDrop = config.integer("route.max_drop", 3);
        lookahead = config.integer("route.lookahead", 3);
        replanTicks = config.integer("route.replan_ticks", 5);
        maxDistance = config.number("route.max_distance", 48);
    }

    int replanTicks() {
        return replanTicks;
    }

    private static long key(int x, int y, int z) {
        return ((long) (x & 0x3FFFFFF) << 38) | ((long) (z & 0x3FFFFFF) << 12) | (y & 0xFFF);
    }

    private int kind(int x, int y, int z) {
        BlockInfo info = world.info(x, y, z);
        return info == null ? BlockInfo.ROUTE_BLOCKED : info.routeKind; // чанк не загружен — не идём
    }

    private boolean standable(int x, int y, int z) {
        return kind(x, y - 1, z) == BlockInfo.ROUTE_SOLID && kind(x, y, z) == BlockInfo.ROUTE_PASSABLE
            && kind(x, y + 1, z) == BlockInfo.ROUTE_PASSABLE;
    }

    private boolean clear(int x, int y, int z) {
        return kind(x, y, z) == BlockInfo.ROUTE_PASSABLE && kind(x, y + 1, z) == BlockInfo.ROUTE_PASSABLE;
    }

    private int[] groundCell(Vec3 position) {
        int x = (int) Math.floor(position.x);
        int z = (int) Math.floor(position.z);
        int top = (int) Math.floor(position.y + 0.3);
        for (int y = top; y >= top - 3; y--) {
            if (standable(x, y, z)) return new int[] {x, y, z};
        }
        return null;
    }

    private List<Step> neighbours(Node node) {
        List<Step> out = new ArrayList<>();
        int x = node.x, y = node.y, z = node.z;
        for (int dx = -1; dx <= 1; dx++) {
            for (int dz = -1; dz <= 1; dz++) {
                if (dx == 0 && dz == 0) continue;
                boolean diagonal = dx != 0 && dz != 0;
                int nx = x + dx;
                int nz = z + dz;
                // По диагонали — только если по обеим боковым клеткам можно пройти
                // (стена — цепляет угол; пустота — срезал угол над пропастью).
                if (diagonal && !(standable(x + dx, y, z) && standable(x, y, z + dz))) continue;
                if (standable(nx, y, nz)) {
                    out.add(new Step(nx, y, nz, diagonal ? SQRT2 : 1, diagonal ? SQRT2 : 1));
                    continue;
                }
                if (diagonal) continue; // прыжки и спуски — только по прямой
                if (standable(nx, y + 1, nz) && kind(x, y + 2, z) == BlockInfo.ROUTE_PASSABLE) {
                    out.add(new Step(nx, y + 1, nz, SQRT2, 2));
                    continue;
                }
                if (!clear(nx, y, nz)) continue;
                for (int drop = 1; drop <= maxDrop; drop++) {
                    if (standable(nx, y - drop, nz)) {
                        out.add(new Step(nx, y - drop, nz, Math.hypot(1, drop), 1 + 0.5 * drop));
                        break;
                    }
                    if (kind(nx, y - drop, nz) != BlockInfo.ROUTE_PASSABLE) break;
                }
            }
        }
        return out;
    }

    private record Plan(List<Cell> path, boolean complete) {
    }

    private Plan plan(int[] start, int[] goal) {
        MinHeap open = new MinHeap();
        Map<Long, Double> best = new HashMap<>();
        Map<Long, Parent> parents = new HashMap<>();
        long startKey = key(start[0], start[1], start[2]);
        best.put(startKey, 0.0);
        double startH = h(start[0], start[1], start[2], goal);
        open.push(new Node(start[0], start[1], start[2], 0, startH, startKey));
        Node closest = new Node(start[0], start[1], start[2], 0, startH, startKey);
        double closestH = startH;
        int expanded = 0;
        while (open.size() > 0 && expanded < maxNodes) {
            Node node = open.pop();
            if (node.g > best.get(node.key)) continue; // устаревшая запись
            expanded++;
            double distance = h(node.x, node.y, node.z, goal);
            if (distance < closestH) {
                closest = node;
                closestH = distance;
            }
            if (Math.hypot(node.x - goal[0], node.z - goal[2]) <= 1.5 && Math.abs(node.y - goal[1]) <= 1) {
                return new Plan(unwind(parents, node), true);
            }
            for (Step next : neighbours(node)) {
                if (Math.hypot(next.x - start[0], next.z - start[2]) > maxDistance) continue;
                long nextKey = key(next.x, next.y, next.z);
                double g = node.g + next.cost;
                if (g >= best.getOrDefault(nextKey, Double.POSITIVE_INFINITY)) continue;
                best.put(nextKey, g);
                parents.put(nextKey, new Parent(node.x, node.y, node.z, next.step, node.key));
                open.push(new Node(next.x, next.y, next.z, g, g + h(next.x, next.y, next.z, goal), nextKey));
            }
        }
        return new Plan(unwind(parents, closest), false);
    }

    private static double h(int x, int y, int z, int[] goal) {
        int dx = Math.abs(x - goal[0]);
        int dz = Math.abs(z - goal[2]);
        return Math.max(dx, dz) + (SQRT2 - 1) * Math.min(dx, dz) + 0.5 * Math.abs(y - goal[1]);
    }

    private static List<Cell> unwind(Map<Long, Parent> parents, Node node) {
        List<Cell> path = new ArrayList<>();
        path.add(new Cell(node.x, node.y, node.z, 0));
        long current = key(node.x, node.y, node.z);
        while (parents.containsKey(current)) {
            Parent parent = parents.get(current);
            Cell first = path.get(0);
            path.set(0, new Cell(first.x, first.y, first.z, parent.step)); // длина шага ИЗ родителя сюда
            path.add(0, new Cell(parent.x, parent.y, parent.z, 0));
            current = parent.key;
        }
        return path;
    }

    /** Ближайшая точка маршрута и сколько идти (state.route) или null. */
    JsonObject update(WorldView view, Vec3 position, Vec3 target) {
        this.world = view;
        if (target == null) {
            path = null;
            return null;
        }
        boolean targetMoved = plannedTarget == null
            || Math.hypot(target.x - plannedTarget.x, target.z - plannedTarget.z) > 2
            || Math.abs(target.y - plannedTarget.y) > 1.5;
        boolean deviated = path != null && distanceTo(path.get(index), position) > 2.5;
        ticksSincePlan++;
        if (path == null || targetMoved || deviated || ticksSincePlan >= replanTicks) {
            replan(position, target);
        }
        if (path == null || path.isEmpty()) return null;
        int bestIndex = index;
        double bestDistance = Double.POSITIVE_INFINITY;
        int last = Math.min(path.size() - 1, index + 6);
        for (int i = index; i <= last; i++) {
            double d = distanceTo(path.get(i), position);
            if (d < bestDistance) {
                bestDistance = d;
                bestIndex = i;
            }
        }
        index = bestIndex;
        Cell waypoint = path.get(Math.min(index + lookahead, path.size() - 1));
        Cell next = path.get(Math.min(index + 1, path.size() - 1)); // следующая клетка (учителю)
        double length = bestDistance + remaining[index];
        if (!complete) {
            Cell end = path.get(path.size() - 1);
            double ex = end.x + 0.5 - target.x, ey = end.y - target.y, ez = end.z + 0.5 - target.z;
            length += Math.sqrt(ex * ex + ey * ey + ez * ez);
        }
        JsonObject out = new JsonObject();
        JsonObject point = new JsonObject();
        point.addProperty("x", waypoint.x + 0.5);
        point.addProperty("y", waypoint.y);
        point.addProperty("z", waypoint.z + 0.5);
        out.add("waypoint", point);
        JsonObject step = new JsonObject();
        step.addProperty("x", next.x + 0.5);
        step.addProperty("y", next.y);
        step.addProperty("z", next.z + 0.5);
        out.add("next", step);
        out.addProperty("length", Math.round(length * 100) / 100.0);
        out.addProperty("complete", complete);
        return out;
    }

    private void replan(Vec3 position, Vec3 target) {
        ticksSincePlan = 0;
        plannedTarget = target;
        int[] start = groundCell(position);
        int[] goal = groundCell(target);
        if (goal == null) goal = new int[] {(int) Math.floor(target.x), (int) Math.floor(target.y), (int) Math.floor(target.z)};
        if (start == null) return; // в воздухе (прыжок, падение) — оставить старый маршрут
        Plan result = plan(start, goal);
        path = result.path;
        complete = result.complete;
        index = 0;
        remaining = new double[path.size()];
        for (int i = path.size() - 2; i >= 0; i--) {
            remaining[i] = remaining[i + 1] + path.get(i + 1).step;
        }
    }

    private static double distanceTo(Cell node, Vec3 position) {
        double dx = node.x + 0.5 - position.x, dy = node.y - position.y, dz = node.z + 0.5 - position.z;
        return Math.sqrt(dx * dx + dy * dy + dz * dz);
    }
}
