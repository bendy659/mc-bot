package mcbot;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.mojang.authlib.GameProfile;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;
import net.kyori.adventure.text.Component;
import net.minecraft.core.UUIDUtil;
import net.minecraft.network.DisconnectionDetails;
import net.minecraft.network.protocol.game.ServerboundClientCommandPacket;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.server.network.CommonListenerCookie;
import net.minecraft.world.phys.Vec3;
import org.bukkit.Bukkit;
import org.bukkit.Location;
import org.bukkit.World;
import org.bukkit.craftbukkit.CraftServer;
import org.bukkit.craftbukkit.CraftWorld;

/**
 * Рой ботов-игроков и цикл решений — то, что делал js/bot.js: раз в
 * tick_rate_ms (3 тика сервера) собрать состояния всех ботов и отправить в
 * Python, а пришедшие действия (по bot_id) выполнить. Умерший бот
 * возрождается через секунду (Python должен увидеть смерть: state.dead).
 */
final class Swarm {
    private static final int RESPAWN_TICKS = 20;

    final Logger log;
    final ProjectConfig config;
    private final Vision vision;
    private final Entities entities;
    private final boolean hearingEnabled;
    private final boolean routeEnabled;
    private final int decisionTicks;
    private final Map<Integer, Bot> bots = new LinkedHashMap<>();
    private final Map<String, String> joining = new java.util.HashMap<>(); // ник -> имя, пока бот входит
    /**
     * Блоки арены, изменённые с прошлого restore: клетка -> каким был блок
     * ДО первого изменения (копка, постройка — ботов и людей). /mcbot restore
     * возвращает их — арена после охоты и раунда салок снова как построена
     * (в симуляции — SimArena.reset_world).
     */
    private final Map<org.bukkit.block.Block, org.bukkit.block.data.BlockData> changed = new LinkedHashMap<>();
    private static final String NAMES_OBJECTIVE = "mcbot_names";
    private Bridge bridge;
    private int serverTicks;
    private int tickIndex;
    boolean paused;  // !stop: состояния не шлются, боты стоят

    Swarm(Logger log, ProjectConfig config) {
        this.log = log;
        this.config = config;
        this.vision = new Vision(config);
        this.entities = new Entities(config);
        this.hearingEnabled = config.flag("hearing.enabled", true);
        this.routeEnabled = config.flag("route.enabled", true);
        this.decisionTicks = Math.max(1, (int) Math.round(config.number("train.tick_rate_ms", 150) / 50.0));
    }

    Map<Integer, Bot> bots() {
        return bots;
    }

    String prefix() {
        return config.botUsername() + config.botSeparator();
    }

    boolean isBotName(String name) {
        return name.startsWith(prefix()) || (bots.size() == 1 && name.equals(config.botUsername()));
    }

    Bot byName(String name) {
        for (Bot bot : bots.values()) {
            if (bot.name.equalsIgnoreCase(name)) return bot;
        }
        return null;
    }

    Bot byEntityId(int entityId) {
        for (Bot bot : bots.values()) {
            if (bot.player.getId() == entityId) return bot;
        }
        return null;
    }

    Bot first() {
        return bots.isEmpty() ? null : bots.values().iterator().next();
    }

    /** Ник бота — как botName в js/bot.js: один бот — просто username. */
    String nameOf(int id, int count) {
        if (count <= 1) return config.botUsername();
        return config.botUsername() + config.botSeparator() + id;
    }

    // --- связь с Python -------------------------------------------------------------

    void connect() {
        if (bridge != null) return;
        int nodeToPy = config.integer("zmq.node_to_py", 5555);
        int pyToNode = config.integer("zmq.py_to_node", 5556);
        bridge = new Bridge(nodeToPy, pyToNode);
        log.info("Связь с Python: состояния -> :" + nodeToPy + ", действия <- :" + pyToNode);
    }

    void disconnect() {
        if (bridge != null) bridge.close();
        bridge = null;
    }

    /** Команда для Python (как bridge.sendCommand в js/bot.js). */
    void sendCommand(JsonObject command) {
        command.addProperty("type", "command");
        if (bridge != null) bridge.send(command.toString());
    }

    // --- боты ---------------------------------------------------------------------------

    List<Bot> spawn(int count, World world) {
        // Имена людям (config.json bot.names): случайные и без повторов — из
        // тех, что не заняты ботами, которые уже в игре.
        List<String> free = new ArrayList<>(config.botNames());
        for (Bot bot : bots.values()) free.remove(bot.humanName);
        java.util.Collections.shuffle(free);
        List<Bot> added = new ArrayList<>();
        for (int id = 1; id <= count; id++) {
            if (bots.containsKey(id)) continue;
            String humanName = free.isEmpty() ? null : free.removeFirst();
            Bot bot = spawnOne(id, nameOf(id, count), humanName, world);
            bots.put(id, bot);
            added.add(bot);
        }
        return added;
    }

    /** Как показать людям только что зашедшего бота (событие входа — McBotListener). */
    String joiningDisplayName(String profileName) {
        return joining.get(profileName);
    }

    private Bot spawnOne(int id, String name, String humanName, World world) {
        MinecraftServer server = ((CraftServer) Bukkit.getServer()).getServer();
        ServerLevel level = ((CraftWorld) world).getHandle();
        GameProfile profile = new GameProfile(UUIDUtil.createOfflinePlayerUUID(name), name);
        BotPlayer player = new BotPlayer(server, level, profile);
        // Бот входит "с чистого листа" на точку спавна мира (её ставит
        // py/training_server.py — центр арены): сохранённые позиция и
        // инвентарь прошлого раза не грузятся — как в симуляции, где у ботов
        // ничего нет. (Живой вход грузит их в PrepareSpawnTask, до
        // placeNewPlayer; без этого бот стоял бы в (0, 0, 0).)
        Location spawn = world.getSpawnLocation();
        player.snapTo(spawn.getBlockX() + 0.5, spawn.getY(), spawn.getBlockZ() + 0.5, spawn.getYaw(), 0f);
        Bot bot = new Bot(id, name, player, log, config, vision);
        bot.humanName = humanName;
        BotConnection connection = new BotConnection(bot::onPacket);
        // Дальше — как вход живого игрока: событие входа, "зашёл в игру"
        // (уже с именем — McBotListener.onJoin).
        joining.put(name, bot.displayName());
        try {
            server.getPlayerList().placeNewPlayer(connection, player, CommonListenerCookie.createInitial(profile, false));
        } finally {
            joining.remove(name);
        }
        showName(bot);
        bot.syncLook();
        Location at = player.getBukkitEntity().getLocation();
        log.info(String.format("Бот %s зашёл: (%.1f, %.1f, %.1f)", name, at.getX(), at.getY(), at.getZ()));
        return bot;
    }

    /**
     * Имя бота людям: в чате и в списке игроков (Tab) — "Егор [AI_1]", над
     * головой — строкой под ником (счёт в слоте "под именем" с текстом вместо
     * числа). Сам ник над головой — AI_1: его и роли салок (команды сервера
     * it/runners) показывает игра.
     */
    private void showName(Bot bot) {
        var bukkit = bot.player.getBukkitEntity();
        bukkit.displayName(Component.text(bot.displayName()));
        bukkit.playerListName(Component.text(bot.displayName()));
        if (bot.humanName == null) return;
        names().getScore(bot.name).setScore(0);
        names().getScore(bot.name).numberFormat(io.papermc.paper.scoreboard.numbers.NumberFormat.fixed(
            Component.text(bot.humanName)));
    }

    /** Табло "под именем": у ботов — имя, у людей — пусто (формат по умолчанию). */
    private org.bukkit.scoreboard.Objective names() {
        var board = Bukkit.getScoreboardManager().getMainScoreboard();
        var objective = board.getObjective(NAMES_OBJECTIVE);
        if (objective == null) {
            objective = board.registerNewObjective(NAMES_OBJECTIVE, org.bukkit.scoreboard.Criteria.DUMMY, Component.empty());
            objective.numberFormat(io.papermc.paper.scoreboard.numbers.NumberFormat.blank());
        }
        if (objective.getDisplaySlot() != org.bukkit.scoreboard.DisplaySlot.BELOW_NAME) {
            objective.setDisplaySlot(org.bukkit.scoreboard.DisplaySlot.BELOW_NAME);
        }
        return objective;
    }

    void removeAll() {
        for (Bot bot : bots.values()) {
            if (bot.humanName != null) names().getScore(bot.name).resetScore();
            bot.player.connection.onDisconnect(new DisconnectionDetails(net.minecraft.network.chat.Component.literal("mc-bot: бот выключен")));
        }
        bots.clear();
    }

    // --- тик сервера ----------------------------------------------------------------------

    void tick() {
        serverTicks++;
        for (Bot bot : bots.values()) {
            BotPlayer player = bot.player;
            if (player.isDeadOrDying() || player.isRemoved()) {
                bot.deadTicks++;
                if (bot.deadTicks == RESPAWN_TICKS) {
                    player.connection.handleClientCommand(
                        new ServerboundClientCommandPacket(ServerboundClientCommandPacket.Action.PERFORM_RESPAWN));
                }
            } else {
                bot.deadTicks = 0;
            }
            bot.actions.tick();
        }
        receive();
        if (!paused && serverTicks % decisionTicks == 0) sendStates();
    }

    /** Состояние бота для отладки (/mcbot state) — как ушло бы в Python. */
    String debugState(Bot bot) {
        return bot.buildState(tickIndex, vision, entities, hearingEnabled, routeEnabled, humans()).toString();
    }

    private void sendStates() {
        if (bridge == null || bots.isEmpty()) return;
        JsonArray humans = humans();
        for (Bot bot : bots.values()) {
            JsonObject state = bot.buildState(tickIndex, vision, entities, hearingEnabled, routeEnabled, humans);
            bridge.send(state.toString());
        }
        tickIndex++;
    }

    /**
     * Люди рядом (js/actions.js: visibleHumans): ник, id, где стоит, режим
     * игры и команда сервера — судья салок по ним знает, кто играет и кем.
     * Наблюдателей клиент не видит — их нет и здесь.
     */
    private JsonArray humans() {
        JsonArray out = new JsonArray();
        MinecraftServer server = ((CraftServer) Bukkit.getServer()).getServer();
        for (ServerPlayer player : server.getPlayerList().getPlayers()) {
            if (player instanceof BotPlayer || isBotName(player.getGameProfile().name()) || player.isSpectator()) continue;
            JsonObject human = new JsonObject();
            human.addProperty("name", player.getGameProfile().name());
            human.addProperty("id", player.getId());
            human.addProperty("x", player.getX());
            human.addProperty("y", player.getY());
            human.addProperty("z", player.getZ());
            human.addProperty("gamemode", player.gameMode.getGameModeForPlayer().getId());
            if (player.getTeam() != null) human.addProperty("team", player.getTeam().getName());
            else human.add("team", null);
            out.add(human);
        }
        return out;
    }

    /**
     * Сообщения от Python: действия, цели, ответы на команды. Действия — только
     * самый свежий ответ на бота и не старше уже выполненного: если Python
     * запоздал и ответы накопились, выполнить все подряд значило бы повернуть
     * голову или поставить столб дважды (так и было: вживую взгляд "сам" уходил,
     * боты лишний раз прыгали столбом и падали с моста, 2026-09-29). js/bot.js
     * так же берёт ответ на свой тик, запоздавший — только если свежего нет.
     */
    private void receive() {
        if (bridge == null) return;
        Map<Integer, JsonObject> freshest = new LinkedHashMap<>();
        for (String text = bridge.receive(); text != null; text = bridge.receive()) {
            JsonObject message;
            try {
                message = JsonParser.parseString(text).getAsJsonObject();
            } catch (RuntimeException err) {
                log.warning("Не понял сообщение от Python: " + err.getMessage());
                continue;
            }
            String type = message.has("type") ? message.get("type").getAsString() : "";
            Bot bot = bots.get(message.has("bot_id") ? message.get("bot_id").getAsInt() : 1);
            switch (type) {
                case "action" -> {
                    if (bot == null) continue;
                    JsonObject kept = freshest.get(bot.id);
                    if (kept == null || tickOf(message) >= tickOf(kept)) freshest.put(bot.id, message);
                }
                case "set_target" -> {
                    if (bot == null) continue;
                    JsonElement position = message.get("position");
                    Vec3 point = position != null && position.isJsonObject()
                        ? new Vec3(position.getAsJsonObject().get("x").getAsDouble(),
                            position.getAsJsonObject().get("y").getAsDouble(),
                            position.getAsJsonObject().get("z").getAsDouble())
                        : null;
                    JsonElement entity = message.get("entity_id");
                    bot.target.setModuleTarget(point, entity != null && !entity.isJsonNull() ? entity.getAsInt() : null);
                }
                case "chat" -> {
                    String reply = message.has("text") ? message.get("text").getAsString() : "";
                    String via = message.has("via") ? message.get("via").getAsString() : "chat";
                    String who = bot != null ? bot.displayName() : "AI";
                    if (via.equals("console")) {
                        log.info("[cmd] [bot#" + (bot != null ? bot.id : 0) + "] " + reply);
                    } else {
                        Bukkit.broadcast(Component.text("<" + who + "> " + reply));
                    }
                }
                default -> {
                }
            }
        }
        for (Map.Entry<Integer, JsonObject> entry : freshest.entrySet()) {
            Bot bot = bots.get(entry.getKey());
            JsonObject message = entry.getValue();
            if (bot == null || paused || bot.player.isDeadOrDying()) continue;
            int tick = tickOf(message);
            if (tick <= bot.lastActionTick) continue; // ответ на этот тик (или новее) уже выполнен
            bot.lastActionTick = tick;
            bot.actions.setTaggable(message.has("tag_ids") && message.get("tag_ids").isJsonArray()
                ? message.getAsJsonArray("tag_ids") : null);
            if (message.has("actions") && message.get("actions").isJsonObject()) {
                bot.actions.executeAll(message.getAsJsonObject("actions"));
            }
        }
    }

    /** Номер тика состояния, на которое ответ Python (нет номера — считаем свежим). */
    private static int tickOf(JsonObject message) {
        return message.has("tick") && !message.get("tick").isJsonNull() ? message.get("tick").getAsInt() : Integer.MAX_VALUE;
    }

    /** Внутри ли арены (config.json server.arena) клетка — вместе со стенами и над ними. */
    boolean inArena(org.bukkit.block.Block block) {
        int cx = config.integer("server.arena.center.0", 0);
        int cz = config.integer("server.arena.center.1", 0);
        int half = config.integer("server.arena.size", 64) / 2;
        int floor = config.integer("server.arena.floor_y", -60);
        int top = floor + config.integer("server.arena.height", 48);
        return block.getX() >= cx - half - 1 && block.getX() <= cx + half && block.getZ() >= cz - half - 1
            && block.getZ() <= cz + half && block.getY() >= floor - 4 && block.getY() <= top;
    }

    /** Запомнить блок до изменения (только первое изменение клетки — исходный блок). */
    void remember(org.bukkit.block.Block block, org.bukkit.block.data.BlockData before) {
        if (inArena(block)) changed.putIfAbsent(block, before);
    }

    /** Вернуть арену как была: все запомненные клетки — прежним блоком. */
    int restore() {
        int count = changed.size();
        for (var entry : changed.entrySet()) entry.getKey().setBlockData(entry.getValue(), false);
        changed.clear();
        return count;
    }

    void pause() {
        paused = true;
        for (Bot bot : bots.values()) bot.actions.stopMovement();
    }
}
