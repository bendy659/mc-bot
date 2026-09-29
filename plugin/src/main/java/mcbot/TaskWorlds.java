package mcbot;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;
import java.util.logging.Logger;
import org.bukkit.Bukkit;
import org.bukkit.Location;
import org.bukkit.World;
import org.bukkit.WorldCreator;
import org.bukkit.generator.ChunkGenerator;

/**
 * Миры задачек (config.json bot.task_worlds: задачка -> имя мира). Автор:
 * "для специфических задач ботов — свои миры, обычный мир не дёргать и не
 * перестраивать по таскам". Мир создаётся пустым (без земли — пустота), при
 * первом запуске плагина; дальше Paper грузит его сам из папки сервера.
 * Что в нём строить и кого куда ставить, решает Python (судья задачки) —
 * командами "execute in minecraft:<мир> run ..." по RCON (для моста —
 * py/bridge_course.py). Людям посмотреть — /mcbot goto <задачка|main>.
 */
final class TaskWorlds {
    /** Спавн мира задачки — над первой дорожкой трассы (py/bridge_course.py: WORLD_Y 64). */
    static final double SPAWN_X = 0.5;
    static final double SPAWN_Y = 65;
    static final double SPAWN_Z = 0.5;

    private final Map<String, String> names;
    private final Logger log;

    TaskWorlds(Map<String, String> names, Logger log) {
        this.names = new LinkedHashMap<>(names);
        this.log = log;
    }

    /** Создать (или загрузить) все миры задачек. */
    void loadAll() {
        for (Map.Entry<String, String> entry : names.entrySet()) {
            World world = load(entry.getValue());
            if (world != null) {
                log.info("Мир задачки " + entry.getKey() + ": " + world.getName() + " (" + world.getKey() + ")");
            }
        }
    }

    /** Мир задачки по имени задачки; "main" (или неизвестная задачка) — обычный мир. */
    World forTask(String task) {
        String name = names.get(task);
        return name != null ? load(name) : Bukkit.getWorlds().getFirst();
    }

    private World load(String name) {
        World world = Bukkit.getWorld(name);
        if (world != null) return world;
        world = new WorldCreator(name).generator(new VoidGenerator()).createWorld();
        if (world == null) {
            log.warning("Мир задачки " + name + " не создался.");
            return null;
        }
        world.setSpawnLocation(new Location(world, SPAWN_X, SPAWN_Y, SPAWN_Z));
        return world;
    }

    /** Пустота: ни земли, ни пещер, ни построек, ни мобов — только то, что построят командами. */
    static final class VoidGenerator extends ChunkGenerator {
        @Override
        public Location getFixedSpawnLocation(World world, Random random) {
            return new Location(world, SPAWN_X, SPAWN_Y, SPAWN_Z);
        }
    }
}
