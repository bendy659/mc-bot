package mcbot;

import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;
import net.minecraft.core.BlockPos;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.world.level.EmptyBlockGetter;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.phys.AABB;

/**
 * Что бот знает о блоке — по состоянию блока, с кэшем (состояний тысячи,
 * лучей — сотни тысяч). Правила — ровно js/vision.js (классы, цвета,
 * "проходное"), js/route.js (опасное, жидкость) и prismarine (boundingBox:
 * 'block' — есть форма столкновения, 'empty' — нет).
 */
final class BlockInfo {
    // js/vision.js: PASS_THROUGH — растительность и мелочь, сквозь которые
    // проходят; ^grass$ — чтобы не зацепить grass_block.
    private static final Pattern PASS_THROUGH = Pattern.compile(String.join("|",
        "^grass$", "tall_grass", "fern", "flower", "poppy", "dandelion", "orchid",
        "allium", "bluet", "daisy", "cornflower", "lily", "tulip", "rose_bush",
        "peony", "sunflower", "sapling", "dead_bush", "sprouts", "crop", "wheat",
        "carrots", "potatoes", "beetroot", "sugar_cane", "vine", "kelp",
        "seagrass", "mushroom", "torch", "rail", "azalea"));

    // js/vision.js: BLOCK_CLASS_PATTERNS. ПОРЯДОК И НОМЕРА — контракт с сетью:
    // 0 воздух/небо, 1 древесина, 2 камень, 3 земля/песок, 4 растения,
    // 5 жидкость, 6 руда, 7 снег/лёд, 8 стекло, 9 прочее.
    private static final List<Map.Entry<Pattern, Integer>> CLASS_PATTERNS = List.of(
        Map.entry(Pattern.compile("^log$|_log$|^wood$|planks?|fence|stem|bamboo"), 1),
        Map.entry(Pattern.compile("stone|cobble|andesite|diorite|granite|deepslate|tuff|basalt|blackstone|brick|obsidian|bedrock|concrete|terracotta|netherrack|end_stone"), 2),
        Map.entry(Pattern.compile("dirt|mud|clay|farmland|path|grass_block|sand|gravel|powder_snow"), 3),
        Map.entry(Pattern.compile("grass|leaves|moss|fern|crop|wheat|carrot|potato|beet|flower|vine|fungus|seagrass|kelp"), 4),
        Map.entry(Pattern.compile("water|lava"), 5),
        Map.entry(Pattern.compile("ore|ancient_debris|amethyst"), 6),
        Map.entry(Pattern.compile("snow|ice"), 7),
        Map.entry(Pattern.compile("glass"), 8));

    // js/vision.js: PALETTE — грубый средний цвет по имени; неизвестное — серое.
    private static final List<Map.Entry<Pattern, double[]>> PALETTE = List.of(
        Map.entry(Pattern.compile("grass| moss|fern|leaves|crop|wheat|carrot|potato|beet"), new double[] {0.30, 0.55, 0.20}),
        Map.entry(Pattern.compile("dirt|mud|clay|farmland|path"), new double[] {0.55, 0.38, 0.24}),
        Map.entry(Pattern.compile("stone|cobble|andesite|diorite|granite|deepslate|tuff|basalt|blackstone"), new double[] {0.50, 0.50, 0.50}),
        Map.entry(Pattern.compile("sand|sandstone|gravel"), new double[] {0.86, 0.80, 0.60}),
        Map.entry(Pattern.compile("wood|log|plank|oak|birch|spruce|jungle|acacia|dark_oak|mangrove|cherry|fence|stem"), new double[] {0.65, 0.50, 0.30}),
        Map.entry(Pattern.compile("water"), new double[] {0.20, 0.35, 0.80}),
        Map.entry(Pattern.compile("lava|magma"), new double[] {0.90, 0.40, 0.10}),
        Map.entry(Pattern.compile("iron|copper|gold"), new double[] {0.80, 0.70, 0.50}),
        Map.entry(Pattern.compile("diamond|emerald"), new double[] {0.30, 0.90, 0.80}),
        Map.entry(Pattern.compile("redstone|netherrack|nether_brick"), new double[] {0.60, 0.15, 0.15}),
        Map.entry(Pattern.compile("snow|ice|packed_ice|frosted"), new double[] {0.90, 0.95, 1.00}),
        Map.entry(Pattern.compile("glass|glass_pane"), new double[] {0.70, 0.85, 0.90}),
        Map.entry(Pattern.compile("wool|carpet|bed|concrete|terracotta"), new double[] {0.75, 0.60, 0.55}),
        Map.entry(Pattern.compile("brick"), new double[] {0.65, 0.35, 0.30}),
        Map.entry(Pattern.compile("obsidian|coal|bedrock"), new double[] {0.15, 0.10, 0.25}));
    private static final double[] UNKNOWN_COLOR = {0.45, 0.45, 0.45};

    // js/route.js: сюда маршрут не ведёт.
    private static final Pattern DANGEROUS = Pattern.compile("lava|fire|magma|cactus|campfire|sweet_berry|cobweb|powder_snow|wither_rose|pointed_dripstone");
    private static final Pattern LIQUID = Pattern.compile("water|lava|bubble_column|seagrass|kelp");

    private static final Map<BlockState, BlockInfo> CACHE = new IdentityHashMap<>();

    final String name;
    final int id;              // номер блока в реестре (prismarine: block.type)
    final boolean air;
    /** Луч зрения/прицела проходит сквозь (воздух и "проходное"). */
    final boolean skip;
    /** Формы столкновения [x0, y0, z0, x1, y1, z1] внутри клетки (prismarine: block.shapes). */
    final double[][] shapes;
    /** prismarine boundingBox === 'block': есть форма столкновения. */
    final boolean solid;
    final int visionClass;
    final double[] color;
    /** Для маршрута: 0 — твёрдое, 1 — проходимое, 2 — нельзя. */
    final int routeKind;

    static final int ROUTE_SOLID = 0;
    static final int ROUTE_PASSABLE = 1;
    static final int ROUTE_BLOCKED = 2;

    private BlockInfo(BlockState state) {
        name = BuiltInRegistries.BLOCK.getKey(state.getBlock()).getPath();
        id = BuiltInRegistries.BLOCK.getId(state.getBlock());
        air = state.isAir();
        skip = air || PASS_THROUGH.matcher(name).find();
        List<AABB> boxes = state.getCollisionShape(EmptyBlockGetter.INSTANCE, BlockPos.ZERO).toAabbs();
        shapes = new double[boxes.size()][];
        for (int i = 0; i < boxes.size(); i++) {
            AABB box = boxes.get(i);
            shapes[i] = new double[] {box.minX, box.minY, box.minZ, box.maxX, box.maxY, box.maxZ};
        }
        solid = shapes.length > 0;
        visionClass = air ? 0 : classOf(name);
        color = colorOf(name);
        if (DANGEROUS.matcher(name).find() || LIQUID.matcher(name).find()) {
            routeKind = ROUTE_BLOCKED;
        } else {
            routeKind = solid ? ROUTE_SOLID : ROUTE_PASSABLE;
        }
    }

    static BlockInfo of(BlockState state) {
        return CACHE.computeIfAbsent(state, BlockInfo::new);
    }

    private static int classOf(String name) {
        for (Map.Entry<Pattern, Integer> entry : CLASS_PATTERNS) {
            if (entry.getKey().matcher(name).find()) return entry.getValue();
        }
        return 9;
    }

    private static double[] colorOf(String name) {
        for (Map.Entry<Pattern, double[]> entry : PALETTE) {
            if (entry.getKey().matcher(name).find()) return entry.getValue();
        }
        return UNKNOWN_COLOR;
    }
}
