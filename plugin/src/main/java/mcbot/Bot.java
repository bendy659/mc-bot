package mcbot;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import java.util.Base64;
import java.util.logging.Logger;
import net.minecraft.network.protocol.Packet;
import net.minecraft.network.protocol.game.ClientboundBundlePacket;
import net.minecraft.network.protocol.game.ClientboundSetEntityMotionPacket;
import net.minecraft.network.protocol.game.ClientboundSoundEntityPacket;
import net.minecraft.network.protocol.game.ClientboundSoundPacket;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.level.border.WorldBorder;
import net.minecraft.world.phys.Vec3;

/**
 * Один бот роя: игрок сервера и всё, что у бота в js/bot.js было в ctx —
 * действия, цель, слух, маршрут и память для состояния (дельты с прошлого
 * решения).
 */
final class Bot {
    final int id;          // bot_id в сообщениях с Python
    final String name;     // ник: AI_1, AI_2...
    final BotPlayer player;
    final Logger log;
    final Actions actions;
    final TargetManager target;
    final Hearing hearing;
    final RoutePlanner route;
    int deadTicks;         // сколько тиков бот мёртв (возрождение — Swarm.tick)
    int lastActionTick = -1;  // тик состояния, ответ на которое выполнен последним (Swarm.receive)
    /** Имя для людей (config.json bot.names) или null — тогда просто ник. */
    String humanName;

    /**
     * Взгляд — в конвенции mineflayer и double (Geometry): его видит Python и
     * им считаются лучи; серверному игроку уходит копия во float-градусах.
     */
    double yaw;
    double pitch;
    private double[] prev;  // x, y, z, yaw, pitch прошлого состояния

    Bot(int id, String name, BotPlayer player, Logger log, ProjectConfig config, Vision vision) {
        this.id = id;
        this.name = name;
        this.player = player;
        this.log = log;
        this.yaw = Geometry.yawOf(player);
        this.pitch = Geometry.pitchOf(player);
        this.actions = new Actions(this, vision);
        this.target = new TargetManager(player);
        this.hearing = new Hearing(config);
        this.route = new RoutePlanner(config);
        // Пересчёты маршрутов у ботов роя — в разные тики (как в js/bot.js).
        this.route.ticksSincePlan = id % route.replanTicks();
    }

    /** Как бота показывать людям: "Егор [AI_1]" — ник остаётся для команд и Python. */
    String displayName() {
        return humanName == null ? name : humanName + " [" + name + "]";
    }

    void setLook(double newYaw, double newPitch) {
        yaw = Geometry.euclideanMod(newYaw, Geometry.PI_2);
        pitch = newPitch;
        float serverYaw = Geometry.serverYaw(yaw);
        player.setYRot(serverYaw);
        player.setYHeadRot(serverYaw);
        player.setXRot(Geometry.serverPitch(pitch));
    }

    /** Сервер сам повернул бота (телепорт, /spreadplayers, возрождение) — взять его взгляд. */
    void syncLook() {
        float serverYaw = Geometry.serverYaw(yaw);
        float serverPitch = Geometry.serverPitch(pitch);
        double yawGap = Math.abs(net.minecraft.util.Mth.wrapDegrees(player.getYRot() - serverYaw));
        if (yawGap > 0.01 || Math.abs(player.getXRot() - serverPitch) > 0.01) {
            yaw = Geometry.yawOf(player);
            pitch = Geometry.pitchOf(player);
        }
    }

    /** Всё, что сервер шлёт клиенту бота: звуки — в слух (js/hearing.js). */
    void onPacket(Packet<?> packet) {
        if (packet instanceof ClientboundBundlePacket bundle) {
            for (Packet<?> inner : bundle.subPackets()) onPacket(inner);
        } else if (packet instanceof ClientboundSetEntityMotionPacket motion && motion.id() == player.getId()) {
            // Отдача от удара: у игрока её применяет КЛИЕНТ по этому пакету, а
            // сервер (Player.attack) свою скорость цели после отправки
            // возвращает назад. Клиента у бота нет — применяем сами, как он.
            // Без этого ботов плагина удары не отталкивали (автор, 2026-09-30),
            // а в симуляции — отталкивали. Не сразу, а в начале своего тика:
            // Player.attack сразу после отправки пакета возвращает цели старую
            // скорость — клиент получил бы пакет уже после этого.
            player.pendingMotion = motion.movement();
        } else if (packet instanceof ClientboundSoundPacket sound) {
            hearing.addSound(player.getX(), player.getZ(), sound.getX(), sound.getZ(), sound.getVolume());
        } else if (packet instanceof ClientboundSoundEntityPacket sound) {
            Entity source = player.level().getEntity(sound.getId());
            if (source != null) hearing.addSound(player.getX(), player.getZ(), source.getX(), source.getZ(), sound.getVolume());
        }
    }

    /** Состояние этого решения — ровно то, что собирал js/state.js + js/bot.js. */
    JsonObject buildState(int tickIndex, Vision vision, Entities entities, boolean hearingEnabled, boolean routeEnabled,
                          JsonArray humans) {
        syncLook();
        WorldView world = new WorldView(player.level());
        Vec3 pos = player.position();
        boolean dead = player.isDeadOrDying();

        JsonObject self = new JsonObject();
        self.addProperty("x", Geometry.round3(pos.x));
        self.addProperty("y", Geometry.round3(pos.y));
        self.addProperty("z", Geometry.round3(pos.z));
        self.addProperty("yaw", Geometry.round3(yaw));
        self.addProperty("pitch", Geometry.round3(pitch));
        self.addProperty("health", Geometry.round3(player.getHealth()));
        self.addProperty("food", Geometry.round3(player.getFoodData().getFoodLevel()));
        self.addProperty("on_ground", player.onGround());
        // Дельты — "память о движении": сдвиг в системе взгляда и поворот.
        if (prev != null) {
            double dx = pos.x - prev[0], dy = pos.y - prev[1], dz = pos.z - prev[2];
            double fx = -StrictMath.sin(yaw), fz = -StrictMath.cos(yaw);
            self.addProperty("move_forward", Geometry.round3(dx * fx + dz * fz));
            self.addProperty("move_right", Geometry.round3(dx * -fz + dz * fx));
            self.addProperty("move_up", Geometry.round3(dy));
            self.addProperty("dyaw", Geometry.round3(Geometry.angleDiff(yaw, prev[3])));
            self.addProperty("dpitch", Geometry.round3(Geometry.angleDiff(pitch, prev[4])));
        } else {
            self.addProperty("move_forward", 0);
            self.addProperty("move_right", 0);
            self.addProperty("move_up", 0);
            self.addProperty("dyaw", 0);
            self.addProperty("dpitch", 0);
        }
        prev = new double[] {pos.x, pos.y, pos.z, yaw, pitch};
        self.addProperty("entity_id", player.getId());

        JsonObject state = new JsonObject();
        state.addProperty("type", "state");
        state.addProperty("tick", tickIndex);

        JsonObject visionJson = new JsonObject();
        JsonArray resolution = new JsonArray();
        resolution.add(vision.resX());
        resolution.add(vision.resY());
        visionJson.add("resolution", resolution);
        visionJson.addProperty("packed", Base64.getEncoder().encodeToString(vision.grid(world, Geometry.eye(player), yaw)));
        state.add("vision", visionJson);

        Vision.Hit center = vision.center(world, player, yaw, pitch);
        if (center == null || center.info().air) {
            state.add("center_block", null);
        } else {
            JsonObject block = new JsonObject();
            block.addProperty("id", center.info().id);
            block.addProperty("name", center.info().name);
            block.addProperty("distance", Geometry.round2(center.distance()));
            block.addProperty("t", center.info().visionClass);
            state.add("center_block", block);
        }
        // Сколько пола до края впереди/справа/сзади/слева — js/state.js: ground.
        JsonArray ground = new JsonArray();
        for (double distance : Vision.groundProbe(world, pos, yaw)) ground.add(distance);
        state.add("ground", ground);
        state.add("entities", entities.list(player, yaw));
        state.add("hearing", hearingEnabled ? hearing.tick(yaw) : new JsonArray());
        state.add("self", self);

        Vec3 targetPos = target.position();
        if (targetPos != null) {
            JsonObject t = new JsonObject();
            t.addProperty("x", Geometry.round3(targetPos.x));
            t.addProperty("y", Geometry.round3(targetPos.y));
            t.addProperty("z", Geometry.round3(targetPos.z));
            t.addProperty("h", target.height());
            state.add("target", t);
        } else {
            state.add("target", null);
        }
        state.addProperty("target_description", target.describe());
        state.addProperty("target_is_human", target.isHumanControlled());
        state.addProperty("inventory_count", InventoryInfo.count(player));
        WorldBorder border = player.level().getWorldBorder();
        JsonObject borderJson = new JsonObject();
        borderJson.addProperty("center_x", Geometry.round3(border.getCenterX()));
        borderJson.addProperty("center_z", Geometry.round3(border.getCenterZ()));
        borderJson.addProperty("size", Geometry.round3(border.getSize()));
        state.add("world_border", borderJson);
        state.addProperty("dead", dead);
        state.addProperty("bot_id", id);
        state.add("inventory", InventoryInfo.summary(player));
        state.add("humans", humans);
        Integer attacked = actions.takeAttacked();
        if (attacked != null) state.addProperty("attacked_id", attacked); else state.add("attacked_id", null);
        Integer hurtBy = actions.takeHurtBy();
        if (hurtBy != null) state.addProperty("hurt_by", hurtBy); else state.add("hurt_by", null);
        state.addProperty("strike", !dead && actions.findEntityInCrosshair() != null ? 1 : 0);
        state.addProperty("can_place", !dead && actions.canPlaceFront() ? 1 : 0);
        state.addProperty("attack_charge", Geometry.round3(actions.attackCharge()));
        state.add("damage_dealt", actions.takeDamageDealt());
        state.add("route", routeEnabled ? route.update(world, pos, targetPos) : null);
        state.add("inventory_items", InventoryInfo.items(player));
        return state;
    }
}
