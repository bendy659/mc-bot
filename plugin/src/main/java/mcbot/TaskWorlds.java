package mcbot;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
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

    /**
     * Мир задачки заново — из регионов готовой карты (бедварс: server/bw_maps/<карта>/region,
     * миры Hypixel 1.8 — Paper переводит их чанки в новый формат сам, при загрузке).
     * Это и установка карты, и её сброс между играми: всё, что построили и сломали,
     * пропадает вместе со старыми регионами. Кто был в мире — переносятся в обычный мир.
     * Возвращает мир или null (карты нет, мир не выгрузился).
     */
    World loadMap(String task, Path regions) throws IOException {
        String name = names.get(task);
        if (name == null || !Files.isDirectory(regions)) return null;
        World world = load(name);
        if (world == null) return null;
        Path folder = world.getWorldFolder().toPath();
        Location away = Bukkit.getWorlds().getFirst().getSpawnLocation();
        for (org.bukkit.entity.Player player : world.getPlayers()) player.teleport(away);
        if (!Bukkit.unloadWorld(world, false)) {
            log.warning("Мир " + name + " не выгрузился — карта не сменена.");
            return null;
        }
        // Старые чанки, сущности и точки интереса — прочь; иначе поверх новой
        // карты остались бы чужие постройки и мобы.
        for (String part : new String[] {"region", "entities", "poi"}) deleteTree(folder.resolve(part));
        Files.createDirectories(folder.resolve("region"));
        try (var files = Files.list(regions)) {
            for (Path file : (Iterable<Path>) files::iterator) {
                if (file.getFileName().toString().endsWith(".mca")) {
                    Files.copy(file, folder.resolve("region").resolve(file.getFileName()));
                }
            }
        }
        return load(name);
    }

    private static void deleteTree(Path root) throws IOException {
        if (!Files.exists(root)) return;
        try (var paths = Files.walk(root)) {
            for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) Files.delete(path);
        }
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
