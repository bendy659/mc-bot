package mcbot;

import com.google.gson.JsonArray;

/**
 * Слух — перенос js/hearing.js: круг вокруг бота делится на секторы, звук
 * кладёт "заряд" в сектор источника, заряд затухает каждое решение. Звуки —
 * те же пакеты, что получил бы клиент mineflayer (BotConnection -> Bot).
 */
final class Hearing {
    private static final double DECAY_PER_TICK = 0.85;
    private static final double MAX_CHARGE = 1.0;

    private final int sectors;
    private final double radius;
    private final double[] charges;

    Hearing(ProjectConfig config) {
        sectors = config.integer("hearing.sectors", 8);
        radius = config.number("hearing.radius", 16);
        charges = new double[sectors];
    }

    /** Звук в точке (sx, sz) слышит бот, стоящий в (bx, bz). */
    void addSound(double bx, double bz, double sx, double sz, double volume) {
        double dx = sx - bx;
        double dz = sz - bz;
        double distance = Math.hypot(dx, dz);
        if (distance > radius) return;
        // Угол в мире: 0 = +Z, растёт к +X (atan2(dx, dz)), 0..2pi.
        double twoPi = Math.PI * 2;
        double normalized = ((StrictMath.atan2(dx, dz) % twoPi) + twoPi) % twoPi;
        int sector = (int) Math.min(Math.floor((normalized / twoPi) * sectors), sectors - 1);
        double strength = Math.max(0, Math.min(volume, 1.0)) * (1.0 - distance / radius);
        charges[sector] = Math.min(charges[sector] + strength, MAX_CHARGE);
    }

    /** Раз в решение: затухание и вектор секторов в системе взгляда (сектор 0 — впереди). */
    JsonArray tick(double yaw) {
        for (int i = 0; i < sectors; i++) {
            charges[i] *= DECAY_PER_TICK;
            if (charges[i] < 0.01) charges[i] = 0;
        }
        double facing = yaw + Math.PI;
        double yawNormalized = ((facing % (Math.PI * 2)) + Math.PI * 2) % (Math.PI * 2);
        int shift = (int) Math.floor((yawNormalized / (Math.PI * 2)) * sectors + 0.5); // Math.round
        JsonArray out = new JsonArray();
        for (int i = 0; i < sectors; i++) {
            out.add(Math.round(charges[(i + shift) % sectors] * 1000) / 1000.0);
        }
        return out;
    }
}
