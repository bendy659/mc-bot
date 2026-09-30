package mcbot;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import java.util.HashSet;
import java.util.Set;
import java.util.regex.Pattern;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Direction;
import net.minecraft.world.InteractionHand;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.EquipmentSlot;
import net.minecraft.world.entity.ExperienceOrb;
import net.minecraft.world.entity.item.ItemEntity;
import net.minecraft.world.entity.projectile.arrow.AbstractArrow;
import net.minecraft.world.entity.player.Inventory;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.phys.AABB;
import net.minecraft.world.phys.BlockHitResult;
import net.minecraft.world.phys.Vec3;

/**
 * Макросы действий — перенос js/actions.js: за решение приходит по макросу
 * на канал (ноги, голова, руки), они выполняются одновременно. Имена и
 * смысл — контракт с py/protocol.py (CHANNELS) и симуляцией (py/sim/game.py).
 *
 * Время — в тиках сервера (50 мс), а не в миллисекундах: так макросы честно
 * ускоряются вместе с сервером (/tick rate).
 */
final class Actions {
    static final double TURN_STEP = 30 * (Math.PI / 180);     // поворот корпуса (ноги)
    static final double LOOK_STEP = 10 * (Math.PI / 180);     // наклон взгляда (голова)
    static final double LOOK_YAW_STEP = 10 * (Math.PI / 180); // доворот взгляда (голова)
    static final int HOLD_TICKS = 5;                          // ACTION_DURATION_MS 250: сколько держать движение
    static final double ATTACK_RANGE = 3.5;
    static final double ATTACK_CONE_COS = Math.cos(30 * (Math.PI / 180));
    static final double DIG_REACH = 4.5;
    static final double PLACE_REACH = 4.5;
    static final int DIG_TIMEOUT_TICKS = 200;       // 10 с
    static final int HANDS_TIMEOUT_TICKS = 50;      // 2.5 с
    static final int PLACE_BELOW_WAIT_TICKS = 10;   // 500 мс: ждать, пока прыжок поднимет ноги
    static final int EAT_TIMEOUT_TICKS = 100;       // 5 с

    // Кого можно бить, кроме игроков из tag_ids (js/actions.js: ATTACKABLE_ENTITY_NAMES).
    private static final Set<String> ATTACKABLE = Set.of("sheep", "chicken", "cow", "pig", "zombie");
    private static final Pattern HELMET = Pattern.compile("helmet");
    private static final Pattern CHESTPLATE = Pattern.compile("chestplate");
    private static final Pattern LEGGINGS = Pattern.compile("leggings");
    private static final Pattern BOOTS = Pattern.compile("boots");
    // prismarine-world BlockFace -> куда ставить (js/actions.js: FACE_VECTORS).
    private static final int[][] FACE_VECTORS = {{0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}, {-1, 0, 0}, {1, 0, 0}};
    private static final Direction[] FACE_DIRECTIONS = {Direction.DOWN, Direction.UP, Direction.NORTH, Direction.SOUTH, Direction.WEST, Direction.EAST};

    private final Bot bot;
    private final BotPlayer player;
    private final Vision vision;

    private int holdTicks;           // сколько ещё держать клавиши движения
    private Set<Integer> taggable = new HashSet<>();
    private Integer attackedId;      // по кому пришёлся удар (state.attacked_id)
    Integer hurtBy;                  // кто ударил бота (state.hurt_by) — пишет McBotListener
    double lastAttackCharge = 1;     // с какой силой бил последний раз (для damage_dealt)
    boolean lastAttackCrit;
    final JsonArray damageDealt = new JsonArray();

    // Долгое дело рук (копка, постройка, еда): пока идёт — новых не начинаем.
    private enum Task { NONE, DIG, PLACE_BELOW, USE, EAT }
    private Task task = Task.NONE;
    private int taskTicks;           // сколько тиков дело уже идёт
    private int taskTimeout;
    private BlockPos digging;        // что копаем
    private float digProgress;
    private BlockPos support;        // place_below: на что ставить
    private int idleHandsTicks;      // сколько тиков руки свободны (вернуть оружие)
    static final int WEAPON_BACK_TICKS = 10; // руки свободны полсекунды — оружие обратно в руку

    Actions(Bot bot, Vision vision) {
        this.bot = bot;
        this.player = bot.player;
        this.vision = vision;
    }

    // --- то, что уходит в состояние --------------------------------------------

    Integer takeAttacked() {
        Integer id = attackedId;
        attackedId = null;
        return id;
    }

    Integer takeHurtBy() {
        Integer id = hurtBy;
        hurtBy = null;
        return id;
    }

    JsonArray takeDamageDealt() {
        JsonArray out = damageDealt.deepCopy();
        while (!damageDealt.isEmpty()) damageDealt.remove(0);
        return out;
    }

    void setTaggable(JsonArray ids) {
        taggable = new HashSet<>();
        if (ids != null) ids.forEach(id -> taggable.add(id.getAsInt()));
    }

    /** Заряд удара 0..1 — как getAttackStrengthScale в игре (он и есть). */
    double attackCharge() {
        return player.getAttackStrengthScale(0.5f);
    }

    // --- решение: макросы всех каналов ----------------------------------------------

    void executeAll(JsonObject actions) {
        stopMovement();
        executeLegs(text(actions, "legs", "idle"));
        executeHead(text(actions, "head", "head_idle"));
        executeHands(text(actions, "hands", "hands_idle"));
    }

    private static String text(JsonObject actions, String channel, String fallback) {
        return actions.has(channel) && !actions.get(channel).isJsonNull() ? actions.get(channel).getAsString() : fallback;
    }

    private void executeLegs(String name) {
        BotPlayer.Controls c = player.controls;
        switch (name) {
            case "idle" -> {
            }
            case "walk_forward" -> hold(() -> {
                c.forward = true;
                c.jump = stepAhead();
            });
            case "sprint_forward" -> hold(() -> {
                c.forward = true;
                c.sprint = true;
                c.jump = stepAhead();
            });
            case "walk_back" -> hold(() -> c.back = true);
            case "strafe_left" -> hold(() -> c.left = true);
            case "strafe_right" -> hold(() -> c.right = true);
            // yaw в конвенции mineflayer растёт ВЛЕВО.
            case "turn_left" -> turnView(TURN_STEP, 0);
            case "turn_right" -> turnView(-TURN_STEP, 0);
            case "jump_forward" -> hold(() -> {
                c.forward = true;
                c.jump = true;
            });
            case "jump" -> hold(() -> c.jump = true);
            // Задом крадучись — мост над пустотой: крадущегося игра с края не
            // пускает (решение раз в 150 мс — без шифта бот у края срывался бы).
            case "sneak_back" -> hold(() -> {
                c.back = true;
                c.sneak = true;
            });
            case "sneak" -> hold(() -> c.sneak = true);
            default -> bot.log.warning("Неизвестное действие ног: " + name);
        }
    }

    private void executeHead(String name) {
        switch (name) {
            case "head_idle" -> {
            }
            case "look_up" -> turnView(0, LOOK_STEP);
            case "look_down" -> turnView(0, -LOOK_STEP);
            case "look_left" -> turnView(LOOK_YAW_STEP, 0);
            case "look_right" -> turnView(-LOOK_YAW_STEP, 0);
            default -> bot.log.warning("Неизвестное действие головы: " + name);
        }
    }

    private void executeHands(String name) {
        switch (name) {
            case "hands_idle" -> {
            }
            case "attack_center" -> attackOrDigCenter();
            case "place_below" -> placeBelow();
            case "place_front" -> placeFront();
            case "use_item" -> useItem();
            case "equip_armor" -> equipArmor();
            case "drop_item" -> dropItem();
            default -> bot.log.warning("Неизвестное действие рук: " + name);
        }
    }

    private void hold(Runnable press) {
        press.run();
        holdTicks = HOLD_TICKS;
    }

    void stopMovement() {
        player.controls.clear();
        holdTicks = 0;
    }

    /** Каждый тик сервера: отпустить протухшие клавиши, вести долгое дело рук. */
    void tick() {
        if (holdTicks > 0 && --holdTicks == 0) {
            // Клавиши движения отпускаем, а шифт держим до следующего решения
            // (как игрок, который думает, не отпуская Shift): ответ Python
            // запоздал — бот у края стоит крадучись, а не съезжает в пустоту.
            boolean sneaking = player.controls.sneak;
            player.controls.clear();
            player.controls.sneak = sneaking;
        }
        tickHands();
        idleHandsTicks = task == Task.NONE ? idleHandsTicks + 1 : 0;
        if (idleHandsTicks == WEAPON_BACK_TICKS) weaponToHand();
    }

    /**
     * Оружие — обратно в руку, когда руки свободны (автор: "не умеют свапать
     * предметы в руках"): для постройки макросы берут блоки, а бить надо
     * мечом. Заранее, а не в миг удара: смена предмета в руке обнуляет заряд
     * удара (Player.tick), и удар сразу после неё был бы слабым.
     */
    private boolean weaponToHand() {
        Inventory inventory = player.getInventory();
        int best = -1;
        int bestRank = InventoryInfo.weaponRank(player.getMainHandItem());
        var items = inventory.getNonEquipmentItems();
        for (int slot = 0; slot < items.size(); slot++) {
            int rank = InventoryInfo.weaponRank(items.get(slot));
            if (rank > bestRank) {
                best = slot;
                bestRank = rank;
            }
        }
        if (best < 0) return false;
        equipToHand(best);
        return true;
    }

    // --- ноги и взгляд ---------------------------------------------------------------

    /**
     * Автопрыжок (js/actions.js: stepAhead): идёт вперёд, а по курсу уступ в
     * блок, над которым свободно, — подпрыгнуть. Стену в 2 блока не перепрыгнуть.
     */
    private boolean stepAhead() {
        WorldView world = new WorldView(player.level());
        Vec3 p = player.position();
        int feetY = (int) Math.floor(p.y);
        int hereX = (int) Math.floor(p.x);
        int hereZ = (int) Math.floor(p.z);
        if (!free(world, hereX, feetY + 2, hereZ)) return false; // над головой потолок
        double fx = -StrictMath.sin(bot.yaw);
        double fz = -StrictMath.cos(bot.yaw);
        for (double distance : new double[] {0.5, 1.0, 1.4}) {
            int x = (int) Math.floor(p.x + fx * distance);
            int z = (int) Math.floor(p.z + fz * distance);
            if (x == hereX && z == hereZ) continue; // ещё своя клетка
            if (solid(world, x, feetY, z)) return free(world, x, feetY + 1, z) && free(world, x, feetY + 2, z);
            if (!free(world, x, feetY, z)) return false;
        }
        return false;
    }

    private static boolean solid(WorldView world, int x, int y, int z) {
        BlockInfo info = world.info(x, y, z);
        return info != null && info.solid;
    }

    private static boolean free(WorldView world, int x, int y, int z) {
        BlockInfo info = world.info(x, y, z);
        return info != null && !info.solid;
    }

    /** Повернуть взгляд сразу (как bot.look(..., true)): прицел и голова совпадают. */
    void turnView(double dyaw, double dpitch) {
        double pitch = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, bot.pitch + dpitch));
        bot.setLook(bot.yaw + dyaw, pitch);
    }

    /** Посмотреть на точку (bot.lookAt): yaw = atan2(-dx, -dz), pitch = atan2(dy, по горизонтали). */
    private void lookAt(Vec3 point) {
        Vec3 eye = Geometry.eye(player);
        double dx = point.x - eye.x;
        double dy = point.y - eye.y;
        double dz = point.z - eye.z;
        bot.setLook(StrictMath.atan2(-dx, -dz), StrictMath.atan2(dy, Math.sqrt(dx * dx + dz * dz)));
    }

    // --- удар и копка -----------------------------------------------------------------

    /**
     * Ближайшая допустимая сущность впереди в пределах удара (js/actions.js:
     * findEntityInCrosshair): по горизонтали до ~30° от взгляда, по высоте —
     * любая, и не сквозь стену (видны центр или голова). Удар сам наводит голову.
     */
    Entity findEntityInCrosshair() {
        Vec3 eye = Geometry.eye(player);
        double fx = -StrictMath.sin(bot.yaw);
        double fz = -StrictMath.cos(bot.yaw);
        WorldView world = new WorldView(player.level());
        Entity best = null;
        double bestCos = ATTACK_CONE_COS;
        AABB box = player.getBoundingBox().inflate(ATTACK_RANGE + 2);
        for (Entity entity : player.level().getEntities(player, box, e -> Entities.visible(e, player))) {
            boolean allowed = ATTACKABLE.contains(Entities.nameOf(entity))
                || (entity instanceof Player && taggable.contains(entity.getId()));
            if (!allowed) continue;
            double height = Entities.heightOf(entity);
            Vec3 center = entity.position().add(0, (height > 0 ? height : 1) * 0.5, 0);
            double tx = center.x - eye.x, ty = center.y - eye.y, tz = center.z - eye.z;
            double distance = Math.sqrt(tx * tx + ty * ty + tz * tz);
            if (distance > ATTACK_RANGE || distance < 1e-6) continue;
            double flat = Math.hypot(tx, tz);
            double cos = flat < 0.3 ? 1 : (tx * fx + tz * fz) / flat;
            if (cos <= bestCos) continue;
            Vec3 head = entity.position().add(0, (height > 0 ? height : 1) * 0.9, 0);
            if (!Vision.lineOfSight(world, eye, center) && !Vision.lineOfSight(world, eye, head)) continue;
            bestCos = cos;
            best = entity;
        }
        return best;
    }

    private void attackOrDigCenter() {
        // Удар по сущности мгновенный и долгое дело рук не ждёт (иначе водящий,
        // начав копать, стоял вплотную и "не бил"); начатую копку бросает.
        Entity entity = findEntityInCrosshair();
        if (entity != null) {
            if (task == Task.DIG) finishTask();
            // В руке не оружие (строил), а оружие есть — сперва взять его, бить —
            // следующим решением: смена предмета сбрасывает заряд (js/actions.js так же).
            if (weaponToHand()) return;
            double height = Entities.heightOf(entity);
            lookAt(entity.position().add(0, (height > 0 ? height : 1) * 0.5, 0));
            lastAttackCharge = attackCharge(); // с какой силой бьёт — до сброса заряда
            lastAttackCrit = lastAttackCharge > 0.9 && !player.onGround()
                && player.getDeltaMovement().y < 0 && !player.isSprinting();
            player.attack(entity);
            player.swing(InteractionHand.MAIN_HAND);
            player.resetAttackStrengthTicker(); // замах сбрасывает заряд — как у клиента игры
            attackedId = entity.getId();
            return;
        }
        if (task != Task.NONE) return;
        WorldView world = new WorldView(player.level());
        Vision.Hit hit = vision.center(world, player, bot.yaw, bot.pitch);
        if (hit == null || hit.info().air) return;
        // Луч прицела бьёт на всю дальность зрения, а копать сервер даст только вблизи.
        if (hit.distance() > DIG_REACH) return;
        BlockPos pos = new BlockPos(hit.x(), hit.y(), hit.z());
        BlockState state = player.level().getBlockState(pos);
        if (state.getDestroySpeed(player.level(), pos) < 0 || !player.gameMode.getGameModeForPlayer().isSurvival()) return;
        digging = pos;
        digProgress = 0;
        start(Task.DIG, DIG_TIMEOUT_TICKS);
    }

    /** Копать можно, пока блок в прицеле и рядом — отвернулся или отошёл, копка прерывается. */
    private boolean stillAiming() {
        Vision.Hit hit = vision.center(new WorldView(player.level()), player, bot.yaw, bot.pitch);
        return hit != null && hit.x() == digging.getX() && hit.y() == digging.getY() && hit.z() == digging.getZ()
            && hit.distance() <= DIG_REACH;
    }

    // --- постройка, еда, броня ---------------------------------------------------------

    /**
     * Поставить блок под себя — столб: только в прыжке, когда ноги поднялись
     * над опорой на блок (ждём до PLACE_BELOW_WAIT_TICKS). Опора — ближайший
     * твёрдый блок снизу (до 3). Стоит на земле — подпрыгивает сам (как
     * js/actions.js): "прыжок ногами + блок руками" в один тик двум сетям не
     * давался, боты стояли у столба цели и не строились.
     */
    private void placeBelow() {
        if (task != Task.NONE) return;
        int slot = InventoryInfo.find(player.getInventory(), InventoryInfo::isPlaceableBlock);
        if (slot < 0) return;
        support = supportBelow();
        if (support == null) return;
        equipToHand(slot);
        if (player.onGround()) hold(() -> player.controls.jump = true);
        start(Task.PLACE_BELOW, HANDS_TIMEOUT_TICKS);
    }

    private BlockPos supportBelow() {
        WorldView world = new WorldView(player.level());
        Vec3 feet = player.position();
        int x = (int) Math.floor(feet.x);
        int z = (int) Math.floor(feet.z);
        for (int y = (int) Math.floor(feet.y) - 1; y >= (int) Math.floor(feet.y) - 3; y--) {
            if (solid(world, x, y, z)) return new BlockPos(x, y, z);
        }
        return null;
    }

    /** Поставить блок на грань того, во что смотрит прицел (стена, мост...). */
    private void placeFront() {
        if (task != Task.NONE) return;
        int slot = InventoryInfo.find(player.getInventory(), InventoryInfo::isPlaceableBlock);
        if (slot < 0) return;
        PlaceTarget target = placeFrontTarget();
        if (target == null) return;
        equipToHand(slot);
        place(target.against(), FACE_DIRECTIONS[target.face()]);
        start(Task.USE, 1); // руки заняты до следующего тика — как короткое дело
    }

    /** Куда встал бы блок place_front: опора, клетка у её грани и сама грань. */
    private record PlaceTarget(BlockPos against, BlockPos destination, int face) {
    }

    /** js/actions.js: placeFrontTarget — клетка у грани в прицеле, или null (прицел пуст
     *  или дальше руки, грань неизвестна, это клетка самого бота). */
    private PlaceTarget placeFrontTarget() {
        Vision.Hit hit = vision.center(new WorldView(player.level()), player, bot.yaw, bot.pitch);
        if (hit == null || hit.distance() > PLACE_REACH || hit.face() < 0) return null;
        int[] v = FACE_VECTORS[hit.face()];
        BlockPos against = new BlockPos(hit.x(), hit.y(), hit.z());
        BlockPos destination = against.offset(v[0], v[1], v[2]);
        // В клетку, где стоит сам бот, сервер блок не поставит.
        BlockPos feet = player.blockPosition();
        if (destination.equals(feet) || destination.equals(feet.above())) return null;
        return new PlaceTarget(against, destination, hit.face());
    }

    /**
     * js/actions.js: canPlaceFront — поставил бы place_front блок прямо сейчас
     * (state.can_place): есть чем, грань в прицеле, клетка за ней — воздух, и
     * никто в ней не стоит (предметы, опыт и стрелы не мешают — как у сервера).
     */
    boolean canPlaceFront() {
        if (player.isDeadOrDying()) return false;
        if (InventoryInfo.find(player.getInventory(), InventoryInfo::isPlaceableBlock) < 0) return false;
        PlaceTarget target = placeFrontTarget();
        if (target == null) return false;
        BlockState state = new WorldView(player.level()).state(target.destination().getX(),
            target.destination().getY(), target.destination().getZ());
        if (state == null || !state.isAir()) return false;
        return player.level().getEntities((Entity) null, new AABB(target.destination()),
            entity -> entity.isAlive() && !(entity instanceof ItemEntity) && !(entity instanceof ExperienceOrb)
                && !(entity instanceof AbstractArrow)).isEmpty();
    }

    /** Поставить блок из руки на грань face блока against (как bot._placeBlockWithOptions). */
    private boolean place(BlockPos against, Direction face) {
        Vec3 point = Vec3.atCenterOf(against).add(face.getStepX() * 0.5, face.getStepY() * 0.5, face.getStepZ() * 0.5);
        ItemStack stack = player.getMainHandItem();
        var result = player.gameMode.useItemOn(player, player.level(), stack, InteractionHand.MAIN_HAND,
            new BlockHitResult(point, face, against, false));
        player.swing(InteractionHand.MAIN_HAND);
        return result.consumesAction();
    }

    /** Голоден и есть еда — поесть; иначе "использовать" то, что в руке. */
    private void useItem() {
        if (task != Task.NONE) return;
        if (player.getFoodData().getFoodLevel() < 20) {
            int slot = InventoryInfo.find(player.getInventory(), InventoryInfo::isFood);
            if (slot >= 0) {
                equipToHand(slot);
                player.gameMode.useItem(player, player.level(), player.getMainHandItem(), InteractionHand.MAIN_HAND);
                start(Task.EAT, EAT_TIMEOUT_TICKS);
                return;
            }
        }
        if (player.getMainHandItem().isEmpty()) return;
        player.gameMode.useItem(player, player.level(), player.getMainHandItem(), InteractionHand.MAIN_HAND);
        start(Task.USE, HOLD_TICKS); // отпустить через 250 мс
    }

    /** Надеть первую подходящую броню в пустой слот. */
    private void equipArmor() {
        if (task != Task.NONE) return;
        Pattern[] patterns = {HELMET, CHESTPLATE, LEGGINGS, BOOTS};
        Inventory inventory = player.getInventory();
        for (int i = 0; i < InventoryInfo.ARMOR_SLOTS.length; i++) {
            EquipmentSlot slot = InventoryInfo.ARMOR_SLOTS[i];
            if (!player.getItemBySlot(slot).isEmpty()) continue; // уже надето
            Pattern pattern = patterns[i];
            int index = InventoryInfo.find(inventory, s -> InventoryInfo.isArmor(s) && pattern.matcher(InventoryInfo.name(s)).find());
            if (index >= 0) {
                ItemStack piece = inventory.getItem(index);
                inventory.setItem(index, ItemStack.EMPTY);
                player.setItemSlot(slot, piece);
                start(Task.USE, 1);
                return;
            }
        }
    }

    /** Выбросить то, что в руке (стопкой — как bot.tossStack). */
    private void dropItem() {
        if (task != Task.NONE || player.getMainHandItem().isEmpty()) return;
        player.drop(true);
    }

    /** Взять предмет из слота в руку (как bot.equip(item, 'hand')). */
    private void equipToHand(int slot) {
        Inventory inventory = player.getInventory();
        if (Inventory.isHotbarSlot(slot)) {
            inventory.setSelectedSlot(slot);
            return;
        }
        int selected = inventory.getSelectedSlot();
        ItemStack inHand = inventory.getItem(selected);
        inventory.setItem(selected, inventory.getItem(slot));
        inventory.setItem(slot, inHand);
    }

    // --- долгое дело рук ----------------------------------------------------------------

    private void start(Task next, int timeoutTicks) {
        task = next;
        taskTicks = 0;
        taskTimeout = timeoutTicks;
    }

    private void finishTask() {
        if (task == Task.DIG && digging != null) {
            player.level().destroyBlockProgress(player.getId(), digging, -1); // убрать трещины
            digging = null;
        }
        if (task == Task.USE || task == Task.EAT) player.releaseUsingItem();
        task = Task.NONE;
    }

    private void tickHands() {
        if (task == Task.NONE) return;
        taskTicks++;
        switch (task) {
            case DIG -> {
                if (!stillAiming()) {
                    finishTask();
                    return;
                }
                BlockState state = player.level().getBlockState(digging);
                if (state.isAir()) {
                    finishTask();
                    return;
                }
                digProgress += state.getDestroyProgress(player, player.level(), digging);
                if (digProgress >= 1.0f) {
                    player.gameMode.destroyBlock(digging);
                    finishTask();
                    return;
                }
                player.level().destroyBlockProgress(player.getId(), digging, (int) (digProgress * 10));
                if (taskTicks % 4 == 0) player.swing(InteractionHand.MAIN_HAND);
            }
            case PLACE_BELOW -> {
                // Ноги выше опоры на блок — ставим; не подпрыгнул вовремя — не ставим.
                if (player.getY() >= support.getY() + 2) {
                    // Взгляд не трогаем: сервер направление взгляда не проверяет, а
                    // поворот головы "на опору" сбивал угол на некратный 10° —
                    // симуляция этого не знает, и мост после столба шёл вкось (2026-09-29).
                    place(support, Direction.UP);
                    finishTask();
                    return;
                }
                if (taskTicks > PLACE_BELOW_WAIT_TICKS) {
                    finishTask();
                    return;
                }
                // Приземлился, а блок ещё не поставлен (команда пришла в
                // воздухе) — прыгнуть снова, а не ждать впустую.
                if (player.onGround()) hold(() -> player.controls.jump = true);
            }
            case EAT -> {
                if (!player.isUsingItem()) {
                    finishTask();
                    return;
                }
            }
            case USE, NONE -> {
            }
        }
        if (taskTicks >= taskTimeout) finishTask();
    }
}
