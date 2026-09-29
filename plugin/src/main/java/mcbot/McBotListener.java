package mcbot;

import com.google.gson.JsonObject;
import io.papermc.paper.event.player.AsyncChatEvent;
import java.util.Arrays;
import java.util.Locale;
import java.util.regex.Pattern;
import net.kyori.adventure.text.Component;
import net.kyori.adventure.text.serializer.plain.PlainTextComponentSerializer;
import net.kyori.adventure.translation.GlobalTranslator;
import org.bukkit.Bukkit;
import org.bukkit.entity.Entity;
import org.bukkit.entity.Player;
import org.bukkit.entity.Projectile;
import org.bukkit.event.EventHandler;
import org.bukkit.event.EventPriority;
import org.bukkit.event.Listener;
import org.bukkit.event.entity.EntityDamageByEntityEvent;
import org.bukkit.event.entity.PlayerDeathEvent;
import org.bukkit.event.player.PlayerCommandPreprocessEvent;
import org.bukkit.event.player.PlayerJoinEvent;
import org.bukkit.event.player.PlayerQuitEvent;
import net.kyori.adventure.text.format.NamedTextColor;
import org.bukkit.plugin.Plugin;

/**
 * События сервера, которые у mineflayer приходили пакетами: урон (кто кого
 * ударил — hurt_by и damage_dealt в состоянии), смерти людей (охота hunt:
 * цель остановили), команды в чате (!task, !start...) и личкой (/msg AI_3 !...).
 */
final class McBotListener implements Listener {
    private static final Pattern WHISPER = Pattern.compile("^/(msg|tell|w|whisper)\\s+(\\S+)\\s+(!.*)$", Pattern.CASE_INSENSITIVE);

    private final Plugin plugin;
    private final Swarm swarm;
    private final Commands commands;

    McBotListener(Plugin plugin, Swarm swarm, Commands commands) {
        this.plugin = plugin;
        this.swarm = swarm;
        this.commands = commands;
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onDamage(EntityDamageByEntityEvent event) {
        Entity damager = event.getDamager();
        if (damager instanceof Projectile projectile && projectile.getShooter() instanceof Entity shooter) damager = shooter;
        Bot victim = swarm.byEntityId(event.getEntity().getEntityId());
        if (victim != null) victim.actions.hurtBy = damager.getEntityId();
        Bot attacker = swarm.byEntityId(damager.getEntityId());
        if (attacker != null) {
            // Мой удар нанёс урон — с какой силой и критом (охота награждает по урону).
            JsonObject hit = new JsonObject();
            hit.addProperty("id", event.getEntity().getEntityId());
            hit.addProperty("charge", Geometry.round3(attacker.actions.lastAttackCharge));
            hit.addProperty("crit", attacker.actions.lastAttackCrit);
            attacker.actions.damageDealt.add(hit);
        }
    }

    /** Сломали блок на арене — запомнить, каким был (вернёт /mcbot restore). */
    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onBreak(org.bukkit.event.block.BlockBreakEvent event) {
        swarm.remember(event.getBlock(), event.getBlock().getBlockData());
    }

    /**
     * Бот выкопал блок — добыча сразу ему в инвентарь, а не на землю: в
     * симуляции (py/sim/game.py) выкопанное сразу у бота, и так же должно быть
     * в игре (иначе предмет лежал бы в полутора блоках, и бот его не подбирал).
     */
    @EventHandler(priority = EventPriority.HIGH, ignoreCancelled = true)
    public void onDrop(org.bukkit.event.block.BlockDropItemEvent event) {
        if (swarm.byEntityId(event.getPlayer().getEntityId()) == null) return;
        var inventory = event.getPlayer().getInventory();
        event.getItems().removeIf(item -> inventory.addItem(item.getItemStack()).isEmpty());
    }

    /** Поставили блок на арене — запомнить, что было в клетке до него. */
    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onPlace(org.bukkit.event.block.BlockPlaceEvent event) {
        swarm.remember(event.getBlock(), event.getBlockReplacedState().getBlockData());
    }

    /** Бот зашёл — "Егор [AI_1] зашёл в игру" (имя из config.json bot.names). */
    @EventHandler(priority = EventPriority.HIGH)
    public void onJoin(PlayerJoinEvent event) {
        String display = swarm.joiningDisplayName(event.getPlayer().getName());
        if (display != null) {
            event.joinMessage(Component.translatable("multiplayer.player.joined", Component.text(display)).color(NamedTextColor.YELLOW));
        }
    }

    @EventHandler(priority = EventPriority.HIGH)
    public void onQuit(PlayerQuitEvent event) {
        Bot bot = swarm.byName(event.getPlayer().getName());
        if (bot != null && bot.humanName != null) {
            event.quitMessage(Component.translatable("multiplayer.player.left", Component.text(bot.displayName())).color(NamedTextColor.YELLOW));
        }
    }

    @EventHandler(priority = EventPriority.MONITOR)
    public void onDeath(PlayerDeathEvent event) {
        Component message = event.deathMessage();
        String text = message == null ? event.getPlayer().getName() + " died"
            : PlainTextComponentSerializer.plainText().serialize(GlobalTranslator.render(message, Locale.US));
        String name = event.getPlayer().getName();
        if (swarm.isBotName(name)) {
            swarm.log.info("Причина смерти: " + text);
            return;
        }
        // Умер человек — Python-у: в охоте ("Останови меня") это конец охоты.
        Bot first = swarm.first();
        if (first == null) return;
        JsonObject payload = new JsonObject();
        payload.addProperty("cmd", "player_died");
        payload.addProperty("name", name);
        payload.addProperty("text", text);
        payload.addProperty("bot_id", first.id);
        payload.addProperty("via", "chat");
        swarm.sendCommand(payload);
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onChat(AsyncChatEvent event) {
        String text = PlainTextComponentSerializer.plainText().serialize(event.message()).trim();
        if (!text.startsWith("!")) return;
        String speaker = event.getPlayer().getName();
        String[] args = text.substring(1).trim().split("\\s+");
        // Чат — в своём потоке; боты и Python — только из главного.
        Bukkit.getScheduler().runTask(plugin, () -> commands.run(args, this::say, null, "chat", speaker));
    }

    /** Личка боту (/msg AI_3 !task looking) — команда только ему. */
    @EventHandler(priority = EventPriority.LOWEST, ignoreCancelled = true)
    public void onWhisper(PlayerCommandPreprocessEvent event) {
        var match = WHISPER.matcher(event.getMessage());
        if (!match.matches()) return;
        Bot bot = swarm.byName(match.group(2));
        if (bot == null) return;
        event.setCancelled(true);
        Player speaker = event.getPlayer();
        String[] args = match.group(3).substring(1).trim().split("\\s+");
        commands.run(args, line -> speaker.sendMessage(Component.text("<" + bot.name + "> " + line)), bot, "chat", speaker.getName());
    }

    private void say(String line) {
        Bot first = swarm.first();
        Bukkit.broadcast(Component.text("<" + (first != null ? first.name : "AI") + "> " + line));
    }

    static String[] split(String text) {
        return Arrays.stream(text.replaceFirst("^!", "").trim().split("\\s+")).toArray(String[]::new);
    }
}
