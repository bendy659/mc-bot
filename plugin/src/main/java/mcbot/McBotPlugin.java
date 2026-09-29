package mcbot;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import org.bukkit.Bukkit;
import org.bukkit.Location;
import org.bukkit.World;
import org.bukkit.command.Command;
import org.bukkit.command.CommandSender;
import org.bukkit.plugin.java.JavaPlugin;

/**
 * Плагин mc-bot: боты — игроки сервера (Swarm/BotPlayer) вместо клиентов
 * mineflayer из js/. Python (py/ai_loop.py) тот же: плагин говорит с ним
 * протоколом js/bot.js (состояния на zmq.node_to_py, действия с py_to_node).
 *
 * Команды (оператору, из консоли сервера и по RCON — так их шлёт лаунчер):
 *   /mcbot start [сколько]  — боты AI_1..N заходят и слушаются Python
 *                             (по умолчанию bot.count из config.json)
 *   /mcbot stop             — все боты уходят
 *   /mcbot cmd <номер|all> <команда> — как строка команд лаунчера (!task hunt...)
 *   /mcbot list             — кто где, здоровье
 *   /mcbot walk <бот> [тиков] [sprint] [jump] — пробная ходьба вперёд
 *   /mcbot goto <задачка|main> [ник] — перейти в мир задачки (bot.task_worlds)
 *                             или обратно в обычный мир: посмотреть на ботов
 */
public final class McBotPlugin extends JavaPlugin {
    private ProjectConfig config;
    private Swarm swarm;
    private Commands commands;
    private TaskWorlds taskWorlds;

    @Override
    public void onEnable() {
        try {
            config = ProjectConfig.load(Bukkit.getWorldContainer().toPath());
        } catch (IOException | RuntimeException err) {
            getLogger().severe("Не прочитал config.json проекта: " + err.getMessage() + " — плагин выключен.");
            Bukkit.getPluginManager().disablePlugin(this);
            return;
        }
        getLogger().info("Конфиг проекта: " + config.path);
        // Миры задачек — пустые, свои: обычный мир под задачки не перестраивается.
        taskWorlds = new TaskWorlds(config.taskWorlds(), getLogger());
        taskWorlds.loadAll();
        swarm = new Swarm(getLogger(), config);
        commands = new Commands(swarm);
        // К Python плагин подключается только на /mcbot start: иначе при рое
        // mineflayer (js/bot.js) Python раздавал бы действия по кругу и Node,
        // и плагину — половина действий ботов терялась бы.
        Bukkit.getPluginManager().registerEvents(new McBotListener(this, swarm, commands), this);
        Bukkit.getScheduler().runTaskTimer(this, () -> swarm.tick(), 1L, 1L);
    }

    @Override
    public void onDisable() {
        if (swarm != null) {
            swarm.removeAll();
            swarm.disconnect();
        }
    }

    @Override
    public boolean onCommand(CommandSender sender, Command command, String label, String[] args) {
        if (args.length == 0) return false;
        switch (args[0].toLowerCase(Locale.ROOT)) {
            case "start", "spawn" -> {
                int count = args.length > 1 ? Integer.parseInt(args[1]) : config.botCount();
                World world = Bukkit.getWorlds().getFirst();
                swarm.connect();
                List<Bot> added = swarm.spawn(count, world);
                swarm.paused = false;
                sender.sendMessage("Ботов зашло: " + added.size() + " (всего " + swarm.bots().size() + ").");
            }
            case "stop", "remove" -> {
                int count = swarm.bots().size();
                swarm.removeAll();
                swarm.disconnect();
                sender.sendMessage("Ботов вышло: " + count + ".");
            }
            case "cmd" -> {
                if (args.length < 3) return false;
                Bot target = null;
                if (!args[1].equalsIgnoreCase("all")) {
                    target = swarm.bots().get(Integer.parseInt(args[1]));
                    if (target == null) {
                        sender.sendMessage("Бот №" + args[1] + " не подключён.");
                        return true;
                    }
                }
                String[] words = Arrays.copyOfRange(args, 2, args.length);
                commands.run(McBotListener.split(String.join(" ", words)), line -> sender.sendMessage("[cmd] " + line), target, "console", null);
            }
            case "state" -> {
                // Отладка: состояние бота — в файл (сверка с симуляцией, py/sim).
                // Меняет память бота (дельты, слух) — делать на паузе (!stop).
                if (args.length < 2) return false;
                Bot bot = swarm.byName(args[1]);
                if (bot == null) {
                    sender.sendMessage("Нет бота " + args[1] + ".");
                    return true;
                }
                try {
                    java.nio.file.Path file = getDataFolder().toPath().resolve("state_" + bot.name + ".json");
                    java.nio.file.Files.createDirectories(file.getParent());
                    java.nio.file.Files.writeString(file, swarm.debugState(bot), java.nio.charset.StandardCharsets.UTF_8);
                    sender.sendMessage(file.toAbsolutePath().toString());
                } catch (IOException err) {
                    sender.sendMessage("Не записал: " + err.getMessage());
                }
            }
            case "restore" -> sender.sendMessage("Арена восстановлена: блоков " + swarm.restore() + ".");
            case "goto" -> {
                // Человеку — посмотреть на ботов в мире задачки: /mcbot goto bridge (и /mcbot goto main).
                if (args.length < 2) return false;
                org.bukkit.entity.Player player = args.length > 2 ? Bukkit.getPlayerExact(args[2])
                    : sender instanceof org.bukkit.entity.Player self ? self : null;
                if (player == null) {
                    sender.sendMessage("Кого переносить? /mcbot goto <задачка|main> <ник>");
                    return true;
                }
                World world = taskWorlds.forTask(args[1].toLowerCase(Locale.ROOT));
                player.teleport(world.getSpawnLocation());
                sender.sendMessage(player.getName() + " -> мир " + world.getName() + ".");
            }
            case "act" -> {
                // Отладка: один набор действий боту без Python — /mcbot act AI_1 jump head_idle place_below
                if (args.length < 5) return false;
                Bot bot = swarm.byName(args[1]);
                if (bot == null) {
                    sender.sendMessage("Нет бота " + args[1] + ".");
                    return true;
                }
                com.google.gson.JsonObject actions = new com.google.gson.JsonObject();
                actions.addProperty("legs", args[2]);
                actions.addProperty("head", args[3]);
                actions.addProperty("hands", args[4]);
                bot.actions.executeAll(actions);
                sender.sendMessage(bot.name + ": " + actions);
            }
            case "list" -> {
                if (swarm.bots().isEmpty()) sender.sendMessage("Ботов нет.");
                for (Bot bot : swarm.bots().values()) {
                    Location at = bot.player.getBukkitEntity().getLocation();
                    sender.sendMessage(String.format(Locale.ROOT, "%s: (%.2f, %.2f, %.2f) yaw %.0f, здоровье %.1f%s",
                        bot.name, at.getX(), at.getY(), at.getZ(), at.getYaw(), bot.player.getHealth(),
                        bot.player.isDeadOrDying() ? ", мёртв" : ""));
                }
            }
            case "walk" -> {
                if (args.length < 2) return false;
                Bot bot = swarm.byName(args[1]);
                if (bot == null) {
                    sender.sendMessage("Нет бота " + args[1] + ".");
                    return true;
                }
                int ticks = args.length > 2 ? Integer.parseInt(args[2]) : 40;
                List<String> flags = List.of(args).subList(Math.min(3, args.length), args.length);
                Location start = bot.player.getBukkitEntity().getLocation();
                BotPlayer.Controls c = bot.player.controls;
                c.clear();
                c.forward = true;
                c.sprint = flags.contains("sprint");
                c.jump = flags.contains("jump");
                if (flags.contains("debug")) bot.player.debugTicks = ticks + 2;
                // Через ticks тиков — отпустить и записать в лог, сколько прошёл.
                Bukkit.getScheduler().runTaskLater(this, () -> {
                    c.clear();
                    Location end = bot.player.getBukkitEntity().getLocation();
                    getLogger().info(String.format(Locale.ROOT, "%s прошёл %.2f блока за %d тиков: (%.2f, %.2f, %.2f) -> (%.2f, %.2f, %.2f)",
                        bot.name, start.distance(end), ticks, start.getX(), start.getY(), start.getZ(), end.getX(), end.getY(), end.getZ()));
                }, ticks);
                sender.sendMessage(bot.name + " идёт " + ticks + " тиков.");
            }
            default -> {
                return false;
            }
        }
        return true;
    }
}
