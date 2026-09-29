package mcbot;

import com.google.gson.JsonObject;
import java.util.regex.Pattern;
import net.minecraft.core.BlockPos;
import net.minecraft.core.component.DataComponents;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.world.entity.EquipmentSlot;
import net.minecraft.world.entity.player.Inventory;
import net.minecraft.world.item.BlockItem;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.level.EmptyBlockGetter;

/**
 * Инвентарь для сети и для рук — перенос js/inventory.js.
 */
final class InventoryInfo {
    private static final Pattern ARMOR = Pattern.compile("helmet|chestplate|leggings|boots");
    private static final Pattern TOOL = Pattern.compile("pickaxe|_axe$|shovel|hoe|shears");
    private static final Pattern WEAPON = Pattern.compile("sword|bow|crossbow|trident|mace");
    static final EquipmentSlot[] ARMOR_SLOTS = {EquipmentSlot.HEAD, EquipmentSlot.CHEST, EquipmentSlot.LEGS, EquipmentSlot.FEET};

    private InventoryInfo() {
    }

    static String name(ItemStack stack) {
        return BuiltInRegistries.ITEM.getKey(stack.getItem()).getPath();
    }

    /**
     * Блок, которым можно строить: предмет с тем же именем, что блок, и у блока
     * есть форма столкновения (prismarine: blocksByName[name].boundingBox === 'block').
     */
    static boolean isPlaceableBlock(ItemStack stack) {
        if (stack.isEmpty() || !(stack.getItem() instanceof BlockItem blockItem)) return false;
        String blockName = BuiltInRegistries.BLOCK.getKey(blockItem.getBlock()).getPath();
        if (!blockName.equals(name(stack))) return false;
        return !blockItem.getBlock().defaultBlockState().getCollisionShape(EmptyBlockGetter.INSTANCE, BlockPos.ZERO).isEmpty();
    }

    static boolean isFood(ItemStack stack) {
        return !stack.isEmpty() && stack.has(DataComponents.FOOD);
    }

    static boolean isArmor(ItemStack stack) {
        return !stack.isEmpty() && ARMOR.matcher(name(stack)).find();
    }

    /** Оружие и насколько оно лучше (меч > топор > трезубец > булава), -1 — не оружие. */
    static int weaponRank(ItemStack stack) {
        if (stack.isEmpty()) return -1;
        String name = name(stack);
        if (name.endsWith("_sword")) return 4;
        if (name.endsWith("_axe")) return 3;
        if (name.equals("trident")) return 2;
        if (name.equals("mace")) return 1;
        return -1;
    }

    static String heldKind(ItemStack stack) {
        if (stack.isEmpty()) return "none";
        if (isPlaceableBlock(stack)) return "block";
        if (isFood(stack)) return "food";
        String name = name(stack);
        if (TOOL.matcher(name).find() || WEAPON.matcher(name).find()) return "tool";
        return "other";
    }

    /** Сводка для сети (state.inventory): блоки, еда, броня, что в руке. */
    static JsonObject summary(BotPlayer bot) {
        int blocks = 0, food = 0, armor = 0;
        for (ItemStack stack : bot.getInventory().getNonEquipmentItems()) {
            if (isPlaceableBlock(stack)) blocks += stack.getCount();
            if (isFood(stack)) food += stack.getCount();
            if (isArmor(stack)) armor += stack.getCount();
        }
        int worn = 0;
        for (EquipmentSlot slot : ARMOR_SLOTS) {
            if (!bot.getItemBySlot(slot).isEmpty()) worn++;
        }
        JsonObject out = new JsonObject();
        out.addProperty("blocks", blocks);
        out.addProperty("food", food);
        out.addProperty("armor_items", armor);
        out.addProperty("armor_worn", worn);
        out.addProperty("held", heldKind(bot.getMainHandItem()));
        return out;
    }

    static int count(BotPlayer bot) {
        int total = 0;
        for (ItemStack stack : bot.getInventory().getNonEquipmentItems()) total += stack.getCount();
        return total;
    }

    static JsonObject items(BotPlayer bot) {
        JsonObject out = new JsonObject();
        for (ItemStack stack : bot.getInventory().getNonEquipmentItems()) {
            if (stack.isEmpty()) continue;
            String name = name(stack);
            int have = out.has(name) ? out.get(name).getAsInt() : 0;
            out.addProperty(name, have + stack.getCount());
        }
        return out;
    }

    /** Номер слота с первым подходящим предметом (0..35) или -1. */
    static int find(Inventory inventory, java.util.function.Predicate<ItemStack> predicate) {
        var items = inventory.getNonEquipmentItems();
        for (int slot = 0; slot < items.size(); slot++) {
            if (predicate.test(items.get(slot))) return slot;
        }
        return -1;
    }
}
