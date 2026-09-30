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
    private BedwarsRules bedwarsRules;

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
        bedwarsRules = new BedwarsRules(config.taskWorlds().get("bedwars"), config.taskWorlds().get("drills"));
        Bukkit.getPluginManager().registerEvents(bedwarsRules, this);
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
            case "bedwars" -> {
                // Карта бедварса в мир задачки bedwars — заново (и сброс карты
                // между играми): /mcbot bedwars <карта из server/bw_maps>. Судья —
                // Python (по RCON); описание карты — data/bedwars/maps/<карта>.json.
                if (args.length < 2) return false;
                String map = String.join(" ", Arrays.copyOfRange(args, 1, args.length));
                java.nio.file.Path regions = Bukkit.getWorldContainer().toPath().resolve("bw_maps").resolve(map).resolve("region");
                long started = System.currentTimeMillis();
                try {
                    World world = taskWorlds.loadMap("bedwars", regions);
                    if (world != null) bedwarsRules.clear(world.getName());
                    sender.sendMessage(world == null
                        ? "Карта не загружена: нет " + regions + " или мира задачки bedwars (bot.task_worlds)."
                        : "Карта " + map + " — в мире " + world.getName() + " (" + (System.currentTimeMillis() - started) + " мс).");
                } catch (IOException err) {
                    sender.sendMessage("Карта не загружена: " + err.getMessage());
                }
            }
            case "drill" -> {
                // Упражнения бедварса (py/bedwars_drills.py) — в своём мире задачки
                // (bot.task_worlds.drills), который НЕ перезагружается: пустой мир
                // бедварса с каждой загрузкой заново генерировал вечно загруженные
                // чанки дорожек, и сервер висел до минуты (2026-09-30). Здесь —
                // загрузить участок дорожек (сразу, не "когда-нибудь": fill в
                // незагруженном чанке не работает) и держать загруженным:
                // /mcbot drill <x0> <z0> <x1> <z1>. Второй раз — мгновенно.
                if (args.length < 5) return false;
                if (config.taskWorlds().get("drills") == null) {
                    sender.sendMessage("Нет мира упражнений: config.json bot.task_worlds.drills.");
                    return true;
                }
                World world = taskWorlds.forTask("drills");
                bedwarsRules.clear(world.getName());
                int x0 = Integer.parseInt(args[1]) >> 4, z0 = Integer.parseInt(args[2]) >> 4;
                int x1 = Integer.parseInt(args[3]) >> 4, z1 = Integer.parseInt(args[4]) >> 4;
                long started = System.currentTimeMillis();
                int loaded = 0;
                for (int cx = Math.min(x0, x1); cx <= Math.max(x0, x1); cx++) {
                    for (int cz = Math.min(z0, z1); cz <= Math.max(z0, z1); cz++) {
                        if (!world.isChunkForceLoaded(cx, cz)) {
                            world.getChunkAt(cx, cz);  // загрузить (сгенерировать пустоту) сейчас
                            world.setChunkForceLoaded(cx, cz, true);
                            loaded++;
                        }
                    }
                }
                sender.sendMessage("Упражнения: мир " + world.getName() + ", новых чанков " + loaded + " ("
                    + (System.currentTimeMillis() - started) + " мс).");
            }
            case "placed" -> {
                // Упражнение "прокопаться к кровати": укрытие кровати судья строит
                // командами fill, а ломать в мирах бедварса можно только поставленное
                // игроками — пометить блоки коробки как поставленные:
                // /mcbot placed <мир> x0 y0 z0 x1 y1 z1.
                if (args.length < 8) return false;
                World world = Bukkit.getWorld(args[1]);
                if (world == null) {
                    sender.sendMessage("Нет мира " + args[1] + ".");
                    return true;
                }
                int[] c = new int[6];
                for (int i = 0; i < 6; i++) c[i] = Integer.parseInt(args[2 + i]);
                int marked = bedwarsRules.markPlaced(world, c[0], c[1], c[2], c[3], c[4], c[5]);
                sender.sendMessage("Поставленным помечено блоков: " + marked + ".");
            }
            case "unplace" -> {
                // Дорожку упражнения перестраивают (fill): пометки "поставлено" на ней
                // — прочь, иначе новая площадка на тех же клетках ломалась бы:
                // /mcbot unplace <мир> x0 y0 z0 x1 y1 z1.
                if (args.length < 8) return false;
                int[] c = new int[6];
                for (int i = 0; i < 6; i++) c[i] = Integer.parseInt(args[2 + i]);
                bedwarsRules.unmark(args[1], c[0], c[1], c[2], c[3], c[4], c[5]);
            }
            case "act" -> {
                // Отладка: один набор действий боту без Python — /mcbot act AI_1 jump head_idle place_below
                // [кого можно бить: ник] — /mcbot act AI_1 idle head_idle attack_center AI_2
                if (args.length < 5) return false;
                Bot bot = swarm.byName(args[1]);
                if (bot == null) {
                    sender.sendMessage("Нет бота " + args[1] + ".");
                    return true;
                }
                if (args.length > 5) {
                    org.bukkit.entity.Player victim = Bukkit.getPlayerExact(args[5]);
                    com.google.gson.JsonArray ids = new com.google.gson.JsonArray();
                    if (victim != null) ids.add(victim.getEntityId());
                    bot.actions.setTaggable(ids);
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
