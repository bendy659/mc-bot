package mcbot;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import org.bukkit.Tag;
import org.bukkit.block.Block;
import org.bukkit.event.EventHandler;
import org.bukkit.event.EventPriority;
import org.bukkit.event.Listener;
import org.bukkit.event.block.BlockBreakEvent;
import org.bukkit.event.block.BlockPlaceEvent;
import org.bukkit.event.player.PlayerBedEnterEvent;

/**
 * Правила бедварса в мире задачки bedwars и в мире его упражнений (drills;
 * судья — py/bedwars_game.py): ломать
 * можно только то, что поставили игроки (и боты), и кровати; сама карта не
 * ломается — как на Hypixel. Сломанная кровать предмета не даёт. Так же в
 * симуляции (py/sim/game.py: SimArena._breakable) — сеть учится там.
 * Поставленное помнится до новой карты (/mcbot bedwars — clear()), в каждом мире своё.
 */
final class BedwarsRules implements Listener {
    private final Set<String> worldNames = new HashSet<>();  // мир бедварса и мир его упражнений
    private final Map<String, Set<Long>> placed = new HashMap<>();  // мир -> поставленные блоки

    BedwarsRules(String... worldNames) {
        for (String name : worldNames) {
            if (name != null) this.worldNames.add(name);
        }
    }

    void clear(String worldName) {
        placed.remove(worldName);
    }

    private Set<Long> placedIn(Block block) {
        return placed.computeIfAbsent(block.getWorld().getName(), name -> new HashSet<>());
    }

    private boolean inBedwars(Block block) {
        return worldNames.contains(block.getWorld().getName());
    }

    private static long key(Block block) {
        return ((long) block.getX() & 0x3FFFFFF) << 38 | ((long) block.getZ() & 0x3FFFFFF) << 12 | (block.getY() & 0xFFF);
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onPlace(BlockPlaceEvent event) {
        if (inBedwars(event.getBlock())) placedIn(event.getBlock()).add(key(event.getBlock()));
    }

    /** Спать в бедварсе нельзя (как на Hypixel): кровать — цель, а не постель. */
    @EventHandler(priority = EventPriority.LOW, ignoreCancelled = true)
    public void onBedEnter(PlayerBedEnterEvent event) {
        if (inBedwars(event.getBed())) event.setCancelled(true);
    }

    @EventHandler(priority = EventPriority.LOW, ignoreCancelled = true)
    public void onBreak(BlockBreakEvent event) {
        Block block = event.getBlock();
        if (!inBedwars(block)) return;
        if (Tag.BEDS.isTagged(block.getType())) {
            event.setDropItems(false);
        } else if (!placedIn(block).remove(key(block))) {
            event.setCancelled(true);
        }
    }
}
