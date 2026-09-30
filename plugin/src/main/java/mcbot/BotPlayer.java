package mcbot;

import com.mojang.authlib.GameProfile;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ClientInformation;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;

/**
 * Бот — настоящий игрок сервера. Разница с живым игроком одна: движение
 * живого считает его клиент и присылает серверу готовые координаты, а у бота
 * клиента нет — его шаг физики сервер делает сам, ровно тем кодом, что и
 * клиент игры (LivingEntity.aiStep -> travel): скольжение вдоль стен, вода,
 * лестницы — всё ванильное (у mineflayer это была копия, prismarine-physics).
 *
 * "Клавиши" (controls) выставляют макросы действий (Actions) — как
 * setControlState у mineflayer.
 */
final class BotPlayer extends ServerPlayer {
    /** Нажатые "клавиши" — как control states mineflayer. */
    static final class Controls {
        boolean forward;
        boolean back;
        boolean left;
        boolean right;
        boolean jump;
        boolean sprint;
        boolean sneak;  // крадучись: медленнее, и с края блока игра не пускает (мост)

        void clear() {
            forward = back = left = right = jump = sprint = sneak = false;
        }
    }

    final Controls controls = new Controls();

    BotPlayer(MinecraftServer server, ServerLevel level, GameProfile profile) {
        super(server, level, profile, ClientInformation.createDefault());
    }

    @Override
    public void tick() {
        // Живому игроку сетевой обработчик сверяет позицию с присланной
        // клиентом; у бота её никто не присылает — "последняя хорошая"
        // позиция и есть текущая (так делает и Carpet с его ботами).
        if (this.tickCount % 10 == 0) {
            this.connection.resetPosition();
            this.level().getChunkSource().move(this);
        }
        if (pendingMotion != null) {
            setDeltaMovement(pendingMotion);  // отдача от удара (Bot.onPacket)
            pendingMotion = null;
        }
        applyControls();
        super.tick();   // серверные дела игрока: режим игры, неуязвимость, инвентарь
        this.doTick();  // сам игрок: физика шага, урон, еда — как у клиента игры
        if (debugTicks > 0) {
            debugTicks--;
            org.bukkit.Bukkit.getLogger().info(String.format(java.util.Locale.ROOT,
                "[debug] %s t%d pos (%.4f, %.4f, %.4f) v (%.4f, %.4f, %.4f) ground %s zza %.3f sprint %s speed %.4f",
                getGameProfile().name(), tickCount, getX(), getY(), getZ(), getDeltaMovement().x, getDeltaMovement().y,
                getDeltaMovement().z, onGround(), zza, isSprinting(), getSpeed()));
        }
    }

    int debugTicks; // отладка физики: сколько тиков писать в лог

    /** Скорость из пакета ClientboundSetEntityMotionPacket (отдача): как клиент. */
    net.minecraft.world.phys.Vec3 pendingMotion;

    /** Клавиши -> ввод движения: как у клиента (LocalPlayer), влево = +1. */
    private void applyControls() {
        this.xxa = (controls.left == controls.right) ? 0f : (controls.left ? 1f : -1f);
        this.zza = (controls.forward == controls.back) ? 0f : (controls.forward ? 1f : -1f);
        if (controls.sneak) {
            // Крадучись медленнее — это делает клиент игры (LocalPlayer: ввод x
            // скорость крадучись, 0.3), серверу приходит уже медленный шаг. Наш
            // бот живёт на сервере — замедляем сами, как клиент (и как
            // prismarine в симуляции). Без этого бот крался в полную скорость, и
            // стоило отпустить шифт между решениями — разгон сносил его с края.
            float slow = (float) this.getAttributeValue(net.minecraft.world.entity.ai.attributes.Attributes.SNEAKING_SPEED);
            this.xxa *= slow;
            this.zza *= slow;
        }
        this.setJumping(controls.jump);
        // Шифт: ванильная физика сама замедлит шаг и не даст сойти с края
        // (maybeBackOffFromEdge) — как у клиента, зажавшего Shift.
        this.setShiftKeyDown(controls.sneak);
        // Бег — только вперёд (как в игре) и не крадучись; отпустил — перестал бежать.
        this.setSprinting(controls.sprint && controls.forward && !controls.back && !controls.sneak);
    }
}
