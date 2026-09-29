package mcbot;

import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.phys.Vec3;

/**
 * Цель бота — перенос js/target.js: точка или сущность от обучающего модуля
 * (Python: set_target) либо цель человека (!setTarget), которая главнее.
 * Сущность-цель: позиция берётся каждый тик у самой сущности; не видна —
 * запасная позиция от Python.
 */
final class TargetManager {
    private static final int PLAYER_MEMORY_TICKS = 600; // 30 с: помним, где видели игрока-цель

    private final BotPlayer bot;
    private String humanKind;          // null | "player" | "position"
    private String humanPlayer;
    private Vec3 humanPosition;
    private boolean humanOverride;
    private Vec3 lastSeenPosition;
    private double lastSeenHeight;
    private int lastSeenTick;
    private Vec3 modulePosition;
    private Integer moduleEntityId;

    TargetManager(BotPlayer bot) {
        this.bot = bot;
    }

    boolean isHumanControlled() {
        return humanOverride;
    }

    void setModuleTarget(Vec3 position, Integer entityId) {
        modulePosition = position;
        moduleEntityId = entityId;
    }

    private ServerPlayer humanTargetPlayer() {
        ServerPlayer player = bot.level().getServer().getPlayerList().getPlayerByName(humanPlayer);
        return player != null && Entities.visible(player, bot) ? player : null;
    }

    Entity targetEntity() {
        if (humanOverride) {
            return "player".equals(humanKind) ? humanTargetPlayer() : null;
        }
        if (moduleEntityId == null) return null;
        Entity entity = bot.level().getEntity(moduleEntityId);
        return entity != null && Entities.visible(entity, bot) ? entity : null;
    }

    Vec3 position() {
        if (humanOverride) {
            if (humanKind == null) return null;
            if (humanKind.equals("player")) {
                ServerPlayer player = humanTargetPlayer();
                if (player != null) {
                    lastSeenPosition = player.position();
                    lastSeenHeight = Entities.heightOf(player);
                    lastSeenTick = bot.tickCount;
                    return player.position();
                }
                return recentlySeen() ? lastSeenPosition : null;
            }
            return humanPosition;
        }
        if (moduleEntityId != null) {
            Entity entity = targetEntity();
            return entity != null ? entity.position() : modulePosition;
        }
        return modulePosition;
    }

    private boolean recentlySeen() {
        return lastSeenPosition != null && bot.tickCount - lastSeenTick < PLAYER_MEMORY_TICKS;
    }

    /** Рост сущности-цели (looking целится в "лицо"); точка — 0. */
    double height() {
        Entity entity = targetEntity();
        if (entity != null) return Entities.heightOf(entity);
        if (humanOverride && "player".equals(humanKind) && recentlySeen()) return lastSeenHeight;
        return 0;
    }

    String describe() {
        if (humanOverride) {
            if (humanKind == null) return "none";
            Vec3 pos = position();
            boolean remembered = humanKind.equals("player") && targetEntity() == null && pos != null;
            String where = pos != null
                ? String.format(java.util.Locale.ROOT, " @ (%.1f, %.1f, %.1f)%s", pos.x, pos.y, pos.z,
                    remembered ? " — по памяти, сейчас не видна" : "")
                : " (не видна)";
            return humanKind.equals("player") ? "player " + humanPlayer + where : "position" + where;
        }
        if (moduleEntityId != null) {
            Entity entity = targetEntity();
            return "module: сущность " + (entity != null ? Entities.nameOf(entity) : "пропала");
        }
        if (modulePosition == null) return "module: none";
        return String.format(java.util.Locale.ROOT, "module @ (%.1f, %.1f, %.1f)", modulePosition.x, modulePosition.y, modulePosition.z);
    }

    /** !setTarget player <ник> | position <x y z> | none */
    String handleCommand(String[] args) {
        String kind = args.length > 0 ? args[0] : "";
        switch (kind) {
            case "none" -> {
                humanKind = null;
                humanOverride = false;
                return "Цель сброшена, управление у обучающего модуля.";
            }
            case "player" -> {
                if (args.length < 2) return "Использование: !setTarget player <username>";
                humanKind = "player";
                humanPlayer = args[1];
                lastSeenPosition = null;
                humanOverride = true;
                return "Цель: игрок " + args[1] + ".";
            }
            case "position" -> {
                try {
                    humanPosition = new Vec3(Double.parseDouble(args[1]), Double.parseDouble(args[2]), Double.parseDouble(args[3]));
                } catch (RuntimeException err) {
                    return "Использование: !setTarget position <x> <y> <z>";
                }
                humanKind = "position";
                humanOverride = true;
                return String.format(java.util.Locale.ROOT, "Цель: позиция (%s, %s, %s).", args[1], args[2], args[3]);
            }
            default -> {
                return "Использование: !setTarget player <имя> | position <x y z> | none";
            }
        }
    }
}
