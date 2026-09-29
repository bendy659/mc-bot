package mcbot;

import net.minecraft.world.entity.Entity;
import net.minecraft.world.phys.Vec3;

/**
 * Зрение бота — перенос js/vision.js ЧИСЛО В ЧИСЛО: мозги учились на этих
 * лучах (и в симуляции py/sim/vision.py — тоже его копия). Обход клеток —
 * как RaycastIterator prismarine-world, пересечение с формой блока — как
 * его intersect, порядок арифметики — тот же, sin/cos — StrictMath.
 */
final class Vision {
    // prismarine-world BlockFace: грань, в которую попал луч прицела (place_front).
    static final int FACE_UNKNOWN = -999;
    static final int FACE_BOTTOM = 0;
    static final int FACE_TOP = 1;
    static final int FACE_NORTH = 2;
    static final int FACE_SOUTH = 3;
    static final int FACE_WEST = 4;
    static final int FACE_EAST = 5;

    private static final double MAX = Double.MAX_VALUE;

    /** Попадание луча: блок, его клетка, дальность до точки попадания, грань. */
    record Hit(BlockInfo info, int x, int y, int z, double distance, int face) {
    }

    private final int resX;
    private final int resY;
    private final double maxDistance;
    private final double verticalSpan;
    private final double horizontalSpan;

    Vision(ProjectConfig config) {
        resX = config.integer("vision.resolution.0", 16);
        resY = config.integer("vision.resolution.1", 16);
        maxDistance = config.number("vision.distance", 2) * 16; // чанки -> блоки
        // Только круговое зрение ("circular" в config.json) — им учились мозги.
        horizontalSpan = Math.PI * 2;
        verticalSpan = (config.number("vision.vertical_fov", 70) * Math.PI) / 180;
    }

    int resX() {
        return resX;
    }

    int resY() {
        return resY;
    }

    /**
     * Сетка зрения (js/vision.js: buildVisionGrid, режим circular) сразу
     * байтами для Python: [r, g, b, d, t] на луч (js/state.js: packVision).
     * Колонка 0 — прямо вперёд, дальше по кругу вправо; ряд 0 — верх.
     */
    byte[] grid(WorldView world, Vec3 eye, double yaw) {
        byte[] out = new byte[resX * resY * 5];
        int index = 0;
        for (int row = 0; row < resY; row++) {
            double vAngle = verticalSpan / 2 - (row * verticalSpan) / (resY - 1);
            for (int col = 0; col < resX; col++) {
                double hAngle = -(col * horizontalSpan) / resX;
                double dirYaw = yaw + hAngle;
                double dirPitch = 0 + vAngle;
                double cosPitch = StrictMath.cos(dirPitch);
                double dx = -StrictMath.sin(dirYaw) * cosPitch;
                double dy = StrictMath.sin(dirPitch);
                double dz = -StrictMath.cos(dirYaw) * cosPitch;
                Hit hit = raycast(world, eye.x, eye.y, eye.z, dx, dy, dz, maxDistance, true);
                if (hit == null) {
                    // Небо / ничего в пределах дальности (js: cellFromHit без блока).
                    out[index++] = toByte(0.5);
                    out[index++] = toByte(0.7);
                    out[index++] = toByte(1.0);
                    out[index++] = toByte(1.0);
                    out[index++] = 0;
                } else {
                    double[] color = hit.info.color;
                    out[index++] = toByte(color[0]);
                    out[index++] = toByte(color[1]);
                    out[index++] = toByte(color[2]);
                    out[index++] = toByte(Math.min(hit.distance / maxDistance, 1.0));
                    out[index++] = (byte) hit.info.visionClass;
                }
            }
        }
        return out;
    }

    /** Луч прицела — строго по взгляду (js/vision.js: centerRaycast / rawRaycast). */
    Hit center(WorldView world, Entity entity, double yaw, double pitch) {
        Vec3 eye = Geometry.eye(entity);
        double cosPitch = StrictMath.cos(pitch);
        return raycast(world, eye.x, eye.y, eye.z,
            -StrictMath.sin(yaw) * cosPitch, StrictMath.sin(pitch), -StrictMath.cos(yaw) * cosPitch, maxDistance, true);
    }

    /**
     * Прямая видимость между точками (js/vision.js: lineOfSight): нет ли на
     * отрезке блока с формой. Жидкость и "проходное" не мешают.
     */
    static boolean lineOfSight(WorldView world, Vec3 from, Vec3 to) {
        double dx = to.x - from.x;
        double dy = to.y - from.y;
        double dz = to.z - from.z;
        double distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (distance < 1e-6) return true;
        Hit hit = raycast(world, from.x, from.y, from.z, dx / distance, dy / distance, dz / distance, distance, false);
        return hit == null || !(hit.distance < distance);
    }

    /**
     * Луч до первого непустого блока не дальше maxDistance (js: fastRaycast /
     * world.raycast с matcher'ом rawRaycast). liquidsStop — жидкость (блок без
     * формы, но не "проходной") останавливает луч в центре клетки, как у
     * зрения; для прямой видимости — нет.
     */
    static Hit raycast(WorldView world, double ox, double oy, double oz, double dx, double dy, double dz,
                       double maxDistance, boolean liquidsStop) {
        double invX = dx == 0 ? MAX : 1 / dx;
        double invY = dy == 0 ? MAX : 1 / dy;
        double invZ = dz == 0 ? MAX : 1 / dz;
        int stepX = (int) Math.signum(dx);
        int stepY = (int) Math.signum(dy);
        int stepZ = (int) Math.signum(dz);
        double tDeltaX = dx == 0 ? MAX : Math.abs(1 / dx);
        double tDeltaY = dy == 0 ? MAX : Math.abs(1 / dy);
        double tDeltaZ = dz == 0 ? MAX : Math.abs(1 / dz);
        int bx = (int) Math.floor(ox);
        int by = (int) Math.floor(oy);
        int bz = (int) Math.floor(oz);
        double tMaxX = dx == 0 ? MAX : Math.abs((bx + (dx > 0 ? 1 : 0) - ox) / dx);
        double tMaxY = dy == 0 ? MAX : Math.abs((by + (dy > 0 ? 1 : 0) - oy) / dy);
        double tMaxZ = dz == 0 ? MAX : Math.abs((bz + (dz > 0 ? 1 : 0) - oz) / dz);

        for (;;) {
            BlockInfo info = world.info(bx, by, bz);
            if (info != null && !info.skip) {
                if (info.shapes.length == 0) {
                    if (liquidsStop) {
                        // Жидкость: попадание в центр клетки (как rawRaycast без точки).
                        double cx = bx + 0.5 - ox;
                        double cy = by + 0.5 - oy;
                        double cz = bz + 0.5 - oz;
                        return new Hit(info, bx, by, bz, Math.sqrt(cx * cx + cy * cy + cz * cz), FACE_UNKNOWN);
                    }
                } else {
                    // Форма блока — как RaycastIterator.intersect (с гранью).
                    double px = ox - bx;
                    double py = oy - by;
                    double pz = oz - bz;
                    double t = MAX;
                    int face = FACE_UNKNOWN;
                    for (double[] shape : info.shapes) {
                        double tmin = (shape[invX > 0 ? 0 : 3] - px) * invX;
                        double tmax = (shape[invX > 0 ? 3 : 0] - px) * invX;
                        double tymin = (shape[invY > 0 ? 1 : 4] - py) * invY;
                        double tymax = (shape[invY > 0 ? 4 : 1] - py) * invY;
                        int shapeFace = stepX > 0 ? FACE_WEST : FACE_EAST;
                        if ((tmin > tymax) || (tymin > tmax)) continue;
                        if (tymin > tmin) {
                            tmin = tymin;
                            shapeFace = stepY > 0 ? FACE_BOTTOM : FACE_TOP;
                        }
                        if (tymax < tmax) tmax = tymax;
                        double tzmin = (shape[invZ > 0 ? 2 : 5] - pz) * invZ;
                        double tzmax = (shape[invZ > 0 ? 5 : 2] - pz) * invZ;
                        if ((tmin > tzmax) || (tzmin > tmax)) continue;
                        if (tzmin > tmin) {
                            tmin = tzmin;
                            shapeFace = stepZ > 0 ? FACE_NORTH : FACE_SOUTH;
                        }
                        if (tzmax < tmax) tmax = tzmax;
                        if (tmin < t) {
                            t = tmin;
                            face = shapeFace;
                        }
                    }
                    if (t != MAX) {
                        double hx = ox - (ox + dx * t);
                        double hy = oy - (oy + dy * t);
                        double hz = oz - (oz + dz * t);
                        return new Hit(info, bx, by, bz, Math.sqrt(hx * hx + hy * hy + hz * hz), face);
                    }
                }
            }
            // RaycastIterator.next(): дальше предела — луч пуст.
            if (Math.min(Math.min(tMaxX, tMaxY), tMaxZ) > maxDistance) return null;
            if (tMaxX < tMaxY) {
                if (tMaxX < tMaxZ) {
                    bx += stepX;
                    tMaxX += tDeltaX;
                } else {
                    bz += stepZ;
                    tMaxZ += tDeltaZ;
                }
            } else if (tMaxY < tMaxZ) {
                by += stepY;
                tMaxY += tDeltaY;
            } else {
                bz += stepZ;
                tMaxZ += tDeltaZ;
            }
        }
    }

    /** js/state.js: toByte — 0..1 -> 0..255 (Math.round). */
    static byte toByte(double value) {
        return (byte) Math.round(Math.min(Math.max(value, 0), 1) * 255);
    }

    /** js/vision.js: GROUND_RANGE — дальше чувство пола не меряет. */
    static final double GROUND_RANGE = 3;

    /**
     * Чувство пола под ногами — js/vision.js: groundProbe. Сколько пола до края
     * впереди, справа, сзади и слева (относительно взгляда), в блоках, со знаком
     * и не дальше GROUND_RANGE: над полом — сколько ещё пола в эту сторону, над
     * пустотой — минус сколько до пола. Пол — блок с формой столкновения на
     * уровне под ногами.
     */
    static double[] groundProbe(WorldView world, Vec3 pos, double yaw) {
        int level = (int) Math.floor(pos.y - 0.01);
        // Вперёд f = (-sin yaw, -cos yaw), вправо r = (-f.z, f.x) — как в js/state.js.
        double fx = -StrictMath.sin(yaw);
        double fz = -StrictMath.cos(yaw);
        double[][] directions = {{fx, fz}, {-fz, fx}, {-fx, -fz}, {fz, -fx}};
        double[] out = new double[directions.length];
        for (int i = 0; i < directions.length; i++) {
            double distance = floorDistance(world, level, pos.x, pos.z, directions[i][0], directions[i][1]);
            out[i] = Math.round(distance * 1000) / 1000.0;
        }
        return out;
    }

    private static boolean floorAt(WorldView world, int level, int x, int z) {
        BlockInfo info = world.info(x, level, z); // null — чанк не загружен: как воздух у mineflayer
        return info != null && info.solid;
    }

    /** js/vision.js: floorDistance — клетка за клеткой до первой, где с полом не так, как под серединой. */
    private static double floorDistance(WorldView world, int level, double x, double z, double dx, double dz) {
        int cx = (int) Math.floor(x);
        int cz = (int) Math.floor(z);
        boolean onFloor = floorAt(world, level, cx, cz);
        int stepX = dx > 0 ? 1 : dx < 0 ? -1 : 0;
        int stepZ = dz > 0 ? 1 : dz < 0 ? -1 : 0;
        double nextX = stepX > 0 ? (cx + 1 - x) / dx : stepX < 0 ? (cx - x) / dx : Double.POSITIVE_INFINITY;
        double nextZ = stepZ > 0 ? (cz + 1 - z) / dz : stepZ < 0 ? (cz - z) / dz : Double.POSITIVE_INFINITY;
        double deltaX = stepX != 0 ? Math.abs(1 / dx) : Double.POSITIVE_INFINITY;
        double deltaZ = stepZ != 0 ? Math.abs(1 / dz) : Double.POSITIVE_INFINITY;
        for (;;) {
            double t;
            if (nextX < nextZ) {
                t = nextX;
                cx += stepX;
                nextX += deltaX;
            } else {
                t = nextZ;
                cz += stepZ;
                nextZ += deltaZ;
            }
            if (t >= GROUND_RANGE) return onFloor ? GROUND_RANGE : -GROUND_RANGE;
            if (floorAt(world, level, cx, cz) != onFloor) return onFloor ? t : -t;
        }
    }
}
