package mcbot;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Set;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.Mob;
import net.minecraft.world.entity.MobCategory;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.phys.AABB;

/**
 * "Зрение на сущностей" — перенос js/entities.js: ближайшие в системе
 * взгляда наблюдателя, фиксированный список max_tracked. Классы — контракт с
 * py/training_modules/targets.py: 0 враждебная, 1 мирная, 2 игрок, 3 неживое.
 */
final class Entities {
    static final int HOSTILE = 0;
    static final int PASSIVE = 1;
    static final int PLAYER = 2;
    static final int NON_LIVING = 3;

    // js/entities.js: HOSTILE_MOBS — для мобов вне категории "монстры".
    private static final Set<String> HOSTILE_MOBS = Set.of(
        "zombie", "husk", "drowned", "skeleton", "stray", "bogged", "creeper",
        "spider", "cave_spider", "enderman", "witch", "pillager", "vindicator",
        "ravager", "evoker", "vex", "slime", "magma_cube", "phantom", "silverfish",
        "guardian", "elder_guardian", "blaze", "ghast", "hoglin", "piglin",
        "piglin_brute", "zoglin", "warden", "breeze", "wither_skeleton",
        "shulker", "ender_dragon");

    private final int radius;
    private final int maxTracked;

    Entities(ProjectConfig config) {
        radius = config.integer("entities.radius", 16);
        maxTracked = config.integer("entities.max_tracked", 8);
    }

    /** Имя сущности: ник игрока или вид ("zombie", "item") — как у mineflayer. */
    static String nameOf(Entity entity) {
        if (entity instanceof Player player) return player.getGameProfile().name();
        return BuiltInRegistries.ENTITY_TYPE.getKey(entity.getType()).getPath();
    }

    /** Класс сущности (js/entities.js: classifyEntity по категориям mineflayer). */
    static int classify(Entity entity) {
        if (entity instanceof Player) return PLAYER;
        if (!(entity instanceof Mob)) return NON_LIVING; // предметы, стрелы, стойки для брони...
        MobCategory category = entity.getType().getCategory();
        if (category == MobCategory.MONSTER) return HOSTILE;
        if (category == MobCategory.MISC) return HOSTILE_MOBS.contains(nameOf(entity)) ? HOSTILE : PASSIVE;
        return PASSIVE; // животные, водные, летучие мыши...
    }

    /** Рост сущности по её виду (mineflayer: из minecraft-data), не текущей позы. */
    static double heightOf(Entity entity) {
        return entity.getType().getDimensions().height();
    }

    /** Видит ли клиент бота эту сущность (наблюдателей сервер другим не показывает). */
    static boolean visible(Entity entity, Entity observer) {
        if (entity == observer || entity.isRemoved()) return false;
        return !(entity instanceof ServerPlayer player && player.isSpectator());
    }

    JsonArray list(Entity observer, double yaw) {
        double ox = observer.getX();
        double oy = observer.getY();
        double oz = observer.getZ();
        double fx = -StrictMath.sin(yaw);
        double fz = -StrictMath.cos(yaw);

        record Seen(Entity entity, double dx, double dy, double dz, double dist, int type) {
        }
        List<Seen> seen = new ArrayList<>();
        AABB box = new AABB(ox - radius, oy - radius, oz - radius, ox + radius, oy + radius, oz + radius);
        for (Entity entity : observer.level().getEntities(observer, box, e -> visible(e, observer))) {
            double dx = entity.getX() - ox;
            double dy = entity.getY() - oy;
            double dz = entity.getZ() - oz;
            double dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
            if (dist > radius || dist < 1e-6) continue;
            seen.add(new Seen(entity, dx, dy, dz, dist, classify(entity)));
        }
        // Живые первыми, внутри — ближние первыми (россыпь предметов не вытесняет игроков).
        seen.sort(Comparator.comparingInt((Seen s) -> s.type == NON_LIVING ? 1 : 0).thenComparingDouble(s -> s.dist));

        JsonArray out = new JsonArray();
        for (int i = 0; i < maxTracked; i++) {
            JsonObject slot = new JsonObject();
            if (i >= seen.size()) {
                slot.addProperty("id", 0);
                slot.addProperty("present", 0);
                slot.addProperty("forward", 0);
                slot.addProperty("right", 0);
                slot.addProperty("up", 0);
                slot.addProperty("dist", 0);
                slot.addProperty("type", 0);
                slot.addProperty("name", "");
                slot.addProperty("height", 0);
            } else {
                Seen s = seen.get(i);
                slot.addProperty("id", s.entity.getId());
                slot.addProperty("present", 1);
                slot.addProperty("forward", clip(s.dx * fx + s.dz * fz));
                slot.addProperty("right", clip(s.dx * -fz + s.dz * fx));
                slot.addProperty("up", clip(s.dy));
                slot.addProperty("dist", Geometry.round3(s.dist / radius));
                slot.addProperty("type", s.type);
                slot.addProperty("name", nameOf(s.entity));
                slot.addProperty("height", Geometry.round2(heightOf(s.entity)));
            }
            out.add(slot);
        }
        return out;
    }

    /** js/entities.js: round(value, radius) — доля радиуса, обрезанная в -1..1. */
    private double clip(double value) {
        return Geometry.round3(Math.max(-1, Math.min(1, value / radius)));
    }
}
