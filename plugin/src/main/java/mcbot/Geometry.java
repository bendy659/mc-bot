package mcbot;

import net.minecraft.world.entity.Entity;
import net.minecraft.world.phys.Vec3;

/**
 * Углы и направления — В КОНВЕНЦИИ MINEFLAYER (AGENTS.md): Python и мозги
 * знают только её. yaw = 0 — север (-Z), растёт влево (pi/2 — запад);
 * pitch > 0 — вверх; взгляд (-sin yaw·cos pitch, sin pitch, -cos yaw·cos pitch).
 *
 * Сервер хранит углы по-своему (градусы, yaw 0 — юг, pitch > 0 — вниз);
 * перевод — как в mineflayer (conv.fromNotchianYaw/Pitch).
 *
 * sin/cos — StrictMath (fdlibm, как Math.sin в V8): лучи зрения обязаны
 * совпадать с js/vision.js число в число.
 */
final class Geometry {
    static final double PI_2 = Math.PI * 2;
    /** Глаза игрока (mineflayer: entity.eyeHeight = 1.62). */
    static final double EYE_HEIGHT = 1.62;

    private Geometry() {
    }

    /** Серверный yaw (градусы) -> yaw mineflayer (радианы, 0..2pi). */
    static double yaw(float serverYawDegrees) {
        return euclideanMod(Math.PI - Math.toRadians(serverYawDegrees), PI_2);
    }

    /** Серверный pitch (градусы, вниз > 0) -> pitch mineflayer (вверх > 0). */
    static double pitch(float serverPitchDegrees) {
        return euclideanMod(Math.toRadians(-serverPitchDegrees) + Math.PI, PI_2) - Math.PI;
    }

    /** yaw mineflayer -> серверный (градусы). */
    static float serverYaw(double yaw) {
        return (float) Math.toDegrees(Math.PI - yaw);
    }

    /** pitch mineflayer -> серверный (градусы). */
    static float serverPitch(double pitch) {
        return (float) Math.toDegrees(-pitch);
    }

    static double yawOf(Entity entity) {
        return yaw(entity.getYRot());
    }

    static double pitchOf(Entity entity) {
        return pitch(entity.getXRot());
    }

    static Vec3 eye(Entity entity) {
        return entity.position().add(0, EYE_HEIGHT, 0);
    }

    static double euclideanMod(double numerator, double denominator) {
        double result = numerator % denominator;
        return result < 0 ? result + denominator : result;
    }

    /** Кратчайшая разница углов (js/state.js: angleDiff). */
    static double angleDiff(double a, double b) {
        double d = a - b;
        while (d > Math.PI) d -= PI_2;
        while (d < -Math.PI) d += PI_2;
        return d;
    }

    /** Math.round(v * 1000) / 1000 — как round() в js/state.js. */
    static double round3(double value) {
        return Math.round(value * 1000) / 1000.0;
    }

    static double round2(double value) {
        return Math.round(value * 100) / 100.0;
    }
}
