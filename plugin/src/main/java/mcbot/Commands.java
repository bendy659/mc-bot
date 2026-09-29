package mcbot;

import com.google.gson.JsonObject;
import java.util.Collection;
import java.util.List;
import java.util.Locale;
import java.util.function.Consumer;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.phys.AABB;
import net.minecraft.world.phys.Vec3;

/**
 * Команды ботам — перенос runCommand из js/bot.js. Откуда пришла команда:
 * чат игры (via "chat") или строка команд лаунчера (via "console", через
 * /mcbot cmd). Python отвечает туда же (сообщение chat с тем же via).
 *
 * direct — команда одному боту (личка /msg AI_3 !..., или номер бота из
 * лаунчера); иначе — "всем, кто слышит", как общий чат слышали все боты.
 */
final class Commands {
    static final String HELP_TEXT = "Команды: !setTarget player <имя> | position <x y z> | none, !task <задача> | all <задача>, "
        + "!greedy [номер] (показать выученное), !start [ник] (\"Останови меня\": охота, задача hunt), "
        + "!stop (пауза ИИ), !resume, !debug, !entities, !whatYouSee";

    private final Swarm swarm;

    Commands(Swarm swarm) {
        this.swarm = swarm;
    }

    private void toPython(Bot bot, String via, JsonObject payload) {
        payload.addProperty("bot_id", bot.id);
        payload.addProperty("via", via);
        swarm.sendCommand(payload);
    }

    /**
     * args — команда без "!", по пробелам; target — бот для direct (или null:
     * всем); speaker — кто написал (для !start без ника).
     */
    void run(String[] args, Consumer<String> reply, Bot target, String via, String speaker) {
        if (swarm.bots().isEmpty()) {
            reply.accept("Нет подключённых ботов — команду выполнить некому.");
            return;
        }
        Collection<Bot> audience = target != null ? List.of(target) : swarm.bots().values();
        Bot first = target != null ? target : swarm.first();
        String command = args.length > 0 ? args[0] : "";
        switch (command) {
            case "help" -> reply.accept(HELP_TEXT);
            case "task" -> {
                // Задачу знает только Python — пересылаем, он проверит имя и ответит.
                boolean all = args.length > 1 && args[1].equals("all");
                String task = all ? (args.length > 2 ? args[2] : null) : (args.length > 1 ? args[1] : null);
                if (task == null) {
                    reply.accept("Использование: !task <walking|looking|follow|gathering|crafting|mix|tag|hunt> или !task all <задача>");
                    return;
                }
                // "!task all X" — Python сам раздаст всем; "!task X" в общем чате
                // слышал каждый бот и менял задачу себе.
                for (Bot bot : all ? List.of(first) : audience) {
                    JsonObject payload = new JsonObject();
                    payload.addProperty("cmd", "set_task");
                    payload.addProperty("task", task);
                    payload.addProperty("all", all);
                    toPython(bot, via, payload);
                }
            }
            case "setTarget" -> {
                String answer = null;
                String[] rest = java.util.Arrays.copyOfRange(args, 1, args.length);
                for (Bot bot : audience) answer = "[" + bot.name + "] " + bot.target.handleCommand(rest);
                reply.accept(answer);
            }
            case "greedy" -> {
                // Показать выученное: без случайных действий. Всем — реагирует
                // бот с номером из команды (по умолчанию №1).
                Bot bot = target;
                if (bot == null) {
                    int wanted = args.length > 1 ? parseInt(args[1], 1) : 1;
                    bot = swarm.bots().get(wanted);
                }
                if (bot == null) {
                    reply.accept("Нет такого бота.");
                    return;
                }
                JsonObject payload = new JsonObject();
                payload.addProperty("cmd", "toggle_greedy");
                toPython(bot, via, payload);
            }
            case "start" -> {
                JsonObject payload = new JsonObject();
                payload.addProperty("cmd", "hunt_start");
                String who = args.length > 1 ? args[1] : speaker;
                if (who != null) payload.addProperty("target", who); else payload.add("target", null);
                toPython(first, via, payload);
            }
            case "stop" -> {
                swarm.pause();
                reply.accept("Пауза: ИИ остановлен у всех ботов. !resume — продолжить.");
            }
            case "resume" -> {
                if (!swarm.paused) {
                    reply.accept("ИИ и так работает.");
                    return;
                }
                swarm.paused = false;
                reply.accept("Продолжаю работу ИИ.");
            }
            case "debug" -> {
                for (Bot bot : audience) {
                    var p = bot.player;
                    reply.accept(String.format(Locale.ROOT, "[%s] pos (%.1f, %.1f, %.1f), yaw %d° pitch %d°, hp %.0f/20, еда %d/20, %s%s",
                        bot.name, p.getX(), p.getY(), p.getZ(), Math.round(Math.toDegrees(bot.yaw)), Math.round(Math.toDegrees(bot.pitch)),
                        p.getHealth(), p.getFoodData().getFoodLevel(), p.onGround() ? "на земле" : "в воздухе",
                        swarm.paused ? ", ПАУЗА" : ""));
                }
            }
            case "entities" -> {
                for (Bot bot : audience) reply.accept("[" + bot.name + "] " + nearest(bot));
            }
            case "whatYouSee" -> {
                for (Bot bot : audience) {
                    reply.accept(String.format(Locale.ROOT, "[%s] Сетка %dx%d (circular), дальность %s чанка. Цель: %s.",
                        bot.name, swarm.config.integer("vision.resolution.0", 16), swarm.config.integer("vision.resolution.1", 16),
                        swarm.config.string("vision.distance", "2"), bot.target.describe()));
                }
            }
            default -> reply.accept("Не знаю команду \"" + command + "\". Напиши !help");
        }
    }

    private static int parseInt(String text, int fallback) {
        try {
            return Integer.parseInt(text);
        } catch (NumberFormatException err) {
            return fallback;
        }
    }

    /** Краткий список ближайших сущностей (js/bot.js: nearestEntitySummary). */
    private String nearest(Bot bot) {
        double radius = swarm.config.integer("entities.radius", 16);
        Vec3 at = bot.player.position();
        List<Entity> near = bot.player.level().getEntities(bot.player, new AABB(at, at).inflate(radius),
            e -> Entities.visible(e, bot.player) && e.position().distanceTo(at) <= radius);
        near.sort(java.util.Comparator.comparingDouble(e -> e.position().distanceTo(at)));
        if (near.isEmpty()) return "Рядом никого нет.";
        StringBuilder out = new StringBuilder();
        for (Entity e : near.subList(0, Math.min(5, near.size()))) {
            if (!out.isEmpty()) out.append("; ");
            String kind = switch (Entities.classify(e)) {
                case Entities.PLAYER -> "игрок " + Entities.nameOf(e);
                case Entities.HOSTILE -> "враждебный " + Entities.nameOf(e);
                case Entities.PASSIVE -> "мирный " + Entities.nameOf(e);
                default -> Entities.nameOf(e);
            };
            out.append(String.format(Locale.ROOT, "%s (%.1fм)", kind, e.position().distanceTo(at)));
        }
        return out.toString();
    }
}
