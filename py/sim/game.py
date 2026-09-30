"""Арена симуляции с ботами: собирает каждому боту состояние ровно в том
виде, в каком его шлёт Node (js/bot.js, js/state.js), выполняет действия
каналов (js/actions.js) и команды судьи салок (/kill, /spreadplayers).

Учит всё тот же ai_loop (py/train_tag_sim.py): ему уходят состояния, от
него приходят действия, цели и tag_ids — те же сообщения, что ходят между
Python и Node.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from pathlib import Path

import numpy as np

from protocol import ACTION_ROTATION  # noqa: F401  (шаги поворотов — те же, что у Node)
from state_encoder import PackedCells

from .physics import HALF_WIDTH, HEIGHT, Body, physics_tick, step_ahead
from .route import RoutePlanner
from .teacher import flee_actions
from .vision import EYE_HEIGHT, Vision, raycast, ground_probe
from .world import AIR, BLOCK_ID, BLOCK_NAMES, COLLIDES, DROPS, FULL_CUBE, HAND_HARVEST, HARDNESS, PASS_THROUGH, SHAPES, ArenaWorld

BEDWARS_MAPS = Path(__file__).resolve().parents[2] / "data" / "bedwars" / "maps"  # сетки карт (py/bedwars_maps.py)
BEDWARS_VOID_DEPTH = 40                   # бедварс: ниже карты на столько — страховка (судья убивает раньше, на void_y)
TICK_SECONDS = 0.15
PHYSICS_TICKS_PER_DECISION = 3            # 150 мс решения = 3 тика физики по 50 мс
HOLD_TICKS = 5                            # движение держится 250 мс (ACTION_DURATION_MS)
TURN_STEP = math.radians(30)              # как js/actions.js
LOOK_STEP = math.radians(10)
LOOK_YAW_STEP = math.radians(10)
ATTACK_RANGE = 3.5
ATTACK_CONE_COS = math.cos(math.radians(30))
PLAYER_ENTITY_HEIGHT = 1.8
RESPAWN_SECONDS = 1.0                     # js/bot.js: возрождение через секунду после смерти
INVULNERABLE_SECONDS = 0.5                # после урона 10 тиков урона нет
FIST_DAMAGE = 1.0                         # урон кулака при полном заряде
ATTACK_DELAY_TICKS = 5                    # "перезарядка" удара кулаком: 20 / скорость атаки 4
SWORD_DAMAGE = 4.0                        # деревянный меч: 1 (рука) + 3
SWORD_DELAY_TICKS = 12.5                  # перезарядка удара мечом: 20 / скорость атаки 1.6
WEAPON_BACK_TICKS = 10                    # руки свободны полсекунды — меч снова в руке (плагин: Actions)
KIT_EMPTY_CHANCE = 0.2                    # случайный набор: столько охотников без блоков вовсе
KIT_NO_SWORD_CHANCE = 0.15                # ...и столько — без меча (кулаки)
KNOCKBACK = 0.4
REGEN_SECONDS = 1.0                       # мирная сложность: +1 здоровья в секунду
DIG_REACH = 4.5                           # js/actions.js: копать — только вблизи
PLACE_REACH = 4.5
DIG_TIMEOUT_TICKS = 200                   # 10 с — копка брошена
PLACE_BELOW_WAIT_TICKS = 10               # 500 мс: ждать, пока прыжок поднимет ноги над опорой
VIEW_DISTANCE = 32.0                      # луч прицела — на всю дальность зрения (vision.distance * 16)
# Грань блока, в которую попал луч (prismarine BlockFace), -> куда ставить (js/actions.js: FACE_VECTORS).
FACE_UNKNOWN, FACE_BOTTOM, FACE_TOP, FACE_NORTH, FACE_SOUTH, FACE_WEST, FACE_EAST = -999, 0, 1, 2, 3, 4, 5
FACE_VECTORS = ((0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1), (-1, 0, 0), (1, 0, 0))
HEARING_DECAY = 0.85                      # js/hearing.js

MOVES = {
    "walk_back": {"back": True},
    "strafe_left": {"left": True},
    "strafe_right": {"right": True},
    "jump_forward": {"forward": True, "jump": True},
    "jump": {"jump": True},
    "sneak_back": {"back": True, "sneak": True},  # мост: задом крадучись, с края не сойдёт
    "sneak": {"sneak": True},
}


def js_round(value: float, digits: int = 3) -> float:
    """Math.round(v * 1000) / 1000, как round() в js/state.js."""
    scale = 10 ** digits
    return math.floor(value * scale + 0.5) / scale


def angle_diff(a: float, b: float) -> float:
    d = a - b
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d


def normalize_yaw(yaw: float) -> float:
    return (yaw + math.pi) % (2 * math.pi) - math.pi


class Hearing:
    """Слух — перенос js/hearing.js: 8 секторов вокруг, заряд от звука
    затухает. Звуки в симуляции — удары (в игре их слышно: звук удара и
    урона игрока)."""

    def __init__(self, config: dict):
        self.sectors = config["hearing"]["sectors"]
        self.radius = config["hearing"]["radius"]
        self.charges = [0.0] * self.sectors

    def add_sound(self, listener_pos, source_pos, volume: float) -> None:
        dx = source_pos[0] - listener_pos[0]
        dz = source_pos[2] - listener_pos[2]
        distance = math.hypot(dx, dz)
        if distance > self.radius:
            return
        two_pi = 2 * math.pi
        normalized = math.atan2(dx, dz) % two_pi
        sector = min(math.floor(normalized / two_pi * self.sectors), self.sectors - 1)
        strength = max(0.0, min(volume, 1.0)) * (1.0 - distance / self.radius)
        self.charges[sector] = min(self.charges[sector] + strength, 1.0)

    def tick(self, yaw: float) -> list[float]:
        for i in range(self.sectors):
            self.charges[i] *= HEARING_DECAY
            if self.charges[i] < 0.01:
                self.charges[i] = 0.0
        facing = (yaw + math.pi) % (2 * math.pi)
        shift = math.floor(facing / (2 * math.pi) * self.sectors + 0.5)  # Math.round
        return [js_round(self.charges[(i + shift) % self.sectors]) for i in range(self.sectors)]


class Agent:
    def __init__(self, index: int, config: dict, world: ArenaWorld):
        self.id = index                      # id сессии бота (bot_id) = номер в нике
        self.entity_id = 1000 + index        # id сущности "на сервере"
        self.name = f"{config['bot']['username']}{config['bot'].get('separator', '_')}{index}"
        self.body = Body(0.5, world.floor_y, 0.5)
        self.health = 20.0
        self.dead_until: float | None = None
        self.invulnerable_until = 0.0
        self.regen_at = 0.0
        self.hold_ticks = 0
        self.hearing = Hearing(config)
        self.route = RoutePlanner(world, config["route"])
        self.prev_pose = None                # (x, y, z, yaw, pitch) прошлого состояния
        self.target_entity: int | None = None
        self.target_fallback = None
        self.taggable: set[int] = set()
        self.attacked_id: int | None = None  # кого ударил с прошлого состояния
        self.hurt_by: int | None = None      # кто ударил его
        self.last_attack_time = -1e9         # заряд удара — как js/actions.js: attackCharge
        self.damage_dealt: list[dict] = []   # урон от его ударов с прошлого состояния
        self.pending = None                  # действие, которое применится на тике физики delay
        self.scripted = False                # "человек" симуляции: ведёт учитель, в ai_loop не идёт
        self.memory: dict = {}               # память учителя для него
        self.inventory: dict[str, int] = {}  # блоки: добыты копкой или выданы (имя -> сколько)
        # Меч — в охоте у автора его выдают всем (server/functions/hunt.mcfunction).
        # Он в руке, пока макрос постройки не взял блок; руки свободны
        # WEAPON_BACK_TICKS — меч обратно (как плагин: Actions.weaponToHand).
        # Без него сеть в симуляции видела в руке блок, а в игре — меч.
        self.sword = False
        self.holding_block = False
        self.idle_hands_ticks = 0
        self.dig: tuple | None = None        # клетка, которую копает
        self.dig_progress = 0.0
        self.dig_ticks = 0
        self.place_support: tuple | None = None  # place_below: опора, ждём прыжка
        self.place_wait = 0
        self.hands_busy_ticks = 0            # короткое дело рук (place_front)
        self.out = False                     # бедварс: выбыл (зритель) — вне игры до новой игры

    @property
    def dead(self) -> bool:
        return self.dead_until is not None

    @property
    def absent(self) -> bool:
        """Нет в мире для других: мёртв или выбыл (зритель)."""
        return self.dead_until is not None or self.out

    def eye(self) -> tuple:
        x, y, z = self.body.pos
        return (x, y + EYE_HEIGHT, z)

    def center(self) -> tuple:
        x, y, z = self.body.pos
        return (x, y + PLAYER_ENTITY_HEIGHT * 0.5, z)


class SimArena:
    def __init__(self, config: dict, bots: int, rng: random.Random, latency: bool = True,
                 target_name: str | None = None, target_sprint: bool = False, pillar_chance: float = 0.0,
                 regen_seconds: float = REGEN_SECONDS, box_chance: float = 0.0, sword_sharpness: int = 0,
                 hide_distance: float = 5.0, random_kit: bool = False, course=None, bedwars: bool = False):
        """target_name — добавить "человека" (для охоты hunt): его ведёт
        учитель убегающего, боты видят его в state.humans и бьют. Он ходит
        шагом (target_sprint — бегает): поддаётся, чтобы было кого догнать."""
        self.target_sprint = target_sprint
        # "Человек" в охоте: с этой вероятностью за охоту, когда охотник
        # подошёл, строит столб и сидит наверху — достать можно, только
        # поднявшись следом или выкопав блоки под ним.
        self.pillar_chance = pillar_chance
        # ...а с этой — закрывается коробкой: стены в два блока вокруг себя
        # (автор строил такую вживую — боты не ломали и не лезли).
        self.box_chance = box_chance
        # Прятаться "человек" решает, когда ближайший охотник ближе стольки
        # блоков. Вплотную (5) с мечом, убивающим с одного удара, он почти
        # никогда не успевал достроить — столб и коробка оставались недостроены.
        self.hide_distance = hide_distance
        # Случайный набор у каждого охотника в каждой охоте (блоков от нуля до
        # start_blocks, меч — не всегда): сеть должна смотреть, что у неё
        # есть, а не заучить "у меня всегда 64 и меч" (автор: "надо, чтоб боты
        # поняли, есть у них чем строиться или нет").
        self.random_kit = random_kit
        self.regen_seconds = regen_seconds  # +1 здоровья раз в столько секунд (мирная — 1, лёгкая — ~4)
        # Меч с остротой этого уровня у всех в охоте (0 — меча нет, кулаки):
        # в игре у автора острота 255 — почти любой удар убивает сразу.
        self.sword_sharpness = sword_sharpness
        self.stats: Counter = Counter()  # копка и постройка — для сводки тренера
        # Охота по укрытиям "человека" (столб, коробка, просто бег): сколько
        # минут охотились и сколько раз остановили — для сводки и проверок.
        self.hide_minutes: Counter = Counter()
        self.hide_kills: Counter = Counter()
        self.hide_timeouts: Counter = Counter()  # охоты, где "человека" так и не остановили
        self.hunt_clock: float | None = None  # с какого времени ещё не учтена текущая охота
        self.config = config
        self.arena_cfg = config["server"]["arena"]
        # Трасса задачки (bridge_course.BridgeCourse) — свой мир вместо арены:
        # острова над пустотой, у каждого бота своя дорожка.
        self.course = course
        # Бедварс: мир — карта Hypixel (py/bedwars_maps.py), её грузит судья
        # командой "mcbot bedwars <карта>" (как плагин). Ломать можно только
        # поставленные блоки (placed) и кровати; сломанные кровати — судье.
        self.bedwars = bedwars
        self.map_worlds: dict[str, ArenaWorld] = {}
        self.placed: set[tuple] = set()
        if bedwars:
            self.world = self._map_world(sorted(p.stem for p in BEDWARS_MAPS.glob("*.npz"))[0])
        elif course is not None:
            self.world = ArenaWorld.for_course(course)
        else:
            self.world = ArenaWorld(self.arena_cfg)
        self.vision = Vision(config)
        self.rng = rng
        self.latency = latency  # ответ Python приходит не сразу: иногда тик физики — ещё по старому
        self.time = 0.0
        self.tick_index = 0
        self.agents = {i: Agent(i, config, self.world) for i in range(1, bots + 1)}
        if target_name:
            target = Agent(bots + 1, config, self.world)
            target.name, target.scripted = target_name, True
            self.agents[target.id] = target
        self.by_entity = {agent.entity_id: agent for agent in self.agents.values()}
        self.scripted_states: dict[int, dict] = {}
        self.scripted_deaths: list[tuple[str, str | None]] = []  # (ник погибшего "человека", кто добил)
        cx, cz = self.arena_cfg["center"]
        self.spawn = (cx + 0.5, self.world.floor_y, cz + 0.5) if course is None else course.start(0)
        self.hits = 0
        if bedwars:
            # В бедварсе всех расставляет судья; до его команд (gamemode survival,
            # телепорт на остров) боты — зрители: иначе падали бы с края карты.
            for agent in self.agents.values():
                agent.out = True
        else:
            self.spread()

    # --- состояния ----------------------------------------------------------

    def build_states(self, wants_route) -> list[dict]:
        """Состояния всех ботов за тик — как пачка от Node. wants_route(id) —
        нужен ли боту маршрут к цели (водящему — да; убегающему его модуль
        не использует, а A* дорогой). "Человек" (scripted) в пачку не
        попадает: его состояние — только его учителю (scripted_states)."""
        agents = list(self.agents.values())
        eyes = np.array([a.eye() for a in agents])
        yaws = np.array([a.body.yaw for a in agents])
        pitches = np.array([a.body.pitch for a in agents])
        grids = self.vision.grids(self.world, eyes, yaws)
        centers = self.vision.center_blocks(self.world, eyes, yaws, pitches)
        border = {"center_x": float(self.arena_cfg["center"][0]), "center_z": float(self.arena_cfg["center"][1]),
                  "size": float(self.arena_cfg["size"] + 2)}
        states = []
        for agent, grid, center in zip(agents, grids, centers):
            x, y, z = agent.body.pos
            yaw, pitch = agent.body.yaw, agent.body.pitch
            me = {"x": js_round(x), "y": js_round(y), "z": js_round(z), "yaw": js_round(yaw), "pitch": js_round(pitch),
                  "health": js_round(agent.health), "food": 20, "on_ground": agent.body.on_ground,
                  "entity_id": agent.entity_id}
            if agent.prev_pose is not None:
                px, py, pz, pyaw, ppitch = agent.prev_pose
                fx, fz = -math.sin(yaw), -math.cos(yaw)
                dx, dy, dz = x - px, y - py, z - pz
                me.update(move_forward=js_round(dx * fx + dz * fz), move_right=js_round(dx * -fz + dz * fx),
                          move_up=js_round(dy), dyaw=js_round(angle_diff(yaw, pyaw)),
                          dpitch=js_round(angle_diff(pitch, ppitch)))
            else:
                me.update(move_forward=0, move_right=0, move_up=0, dyaw=0, dpitch=0)
            agent.prev_pose = (x, y, z, yaw, pitch)

            target = self._scripted_threat(agent) if agent.scripted else self._target_of(agent)
            state = {
                "type": "state", "tick": self.tick_index, "bot_id": agent.id,
                "vision": {"resolution": [self.vision.res_x, self.vision.res_y], "cells": PackedCells(grid.tobytes())},
                "center_block": center,
                "ground": ground_probe(self.world, agent.body.pos, yaw),
                "entities": self._entities_for(agent),
                "hearing": agent.hearing.tick(yaw),
                "self": me,
                "target": target,
                "target_description": "sim",
                "target_is_human": False,
                "inventory_count": sum(agent.inventory.values()),
                "world_border": border,
                "dead": agent.dead,
                # Как js/inventory.js: строительные — блоки (слитки бедварса — нет).
                "inventory": {"blocks": _block_count(agent.inventory), "food": 0, "armor_items": 0, "armor_worn": 0,
                              "held": self._held(agent)},
                "inventory_items": dict(agent.inventory),
                "humans": self._humans(),
                "attacked_id": agent.attacked_id,
                "hurt_by": agent.hurt_by,
                "strike": 1 if not agent.absent and self._strike_target(agent) is not None else 0,
                "can_place": 1 if self._can_place_front(agent) else 0,
                "attack_charge": js_round(self._charge(agent)),
                "damage_dealt": agent.damage_dealt,
            }
            agent.attacked_id = None
            agent.hurt_by = None
            agent.damage_dealt = []
            if target is not None and not agent.scripted and wants_route(agent.id):
                state["route"] = agent.route.update((x, y, z), (target["x"], target["y"], target["z"]))
            else:
                state["route"] = None
            if agent.scripted:
                self.scripted_states[agent.id] = state
            else:
                states.append(state)
        self.tick_index += 1
        return states

    def _humans(self) -> list[dict]:
        """Как js/actions.js: visibleHumans — люди рядом (для судьи охоты)."""
        return [{"name": a.name, "id": a.entity_id, "x": a.body.pos[0], "y": a.body.pos[1], "z": a.body.pos[2],
                 "gamemode": 0, "team": None} for a in self.agents.values() if a.scripted and not a.dead]

    def _scripted_threat(self, agent: Agent) -> dict | None:
        """Человеку угроза — ближайший живой бот (его учитель от неё бежит)."""
        bots = [a for a in self.agents.values() if not a.scripted and not a.dead]
        if not bots:
            return None
        nearest = min(bots, key=lambda a: math.dist(a.body.pos, agent.body.pos))
        x, y, z = nearest.body.pos
        return {"x": js_round(x), "y": js_round(y), "z": js_round(z), "h": PLAYER_ENTITY_HEIGHT}

    def drive_scripted(self) -> None:
        """Ход человека: учитель убегающего по его состоянию этого тика."""
        for agent in self.agents.values():
            state = self.scripted_states.get(agent.id)
            if agent.scripted and not agent.dead and state is not None:
                actions = self._scripted_hide(agent) or flee_actions(state, self.config, agent.memory)
                if not self.target_sprint and actions["legs"] == "sprint_forward":
                    actions["legs"] = "walk_forward"
                agent.pending = (0, actions, [])

    def _scripted_hide(self, agent: Agent) -> dict | None:
        """Как человек прячется: раз за охоту решает, когда охотник ближе 5
        блоков, — столб (pillar_chance: прыжками с блоком под себя до высоты
        3–6; выкопали блок под ним — достраивает, пока есть блоки), коробка
        (box_chance: стены в два блока вокруг себя) или просто убегать."""
        memory = agent.memory
        if "hide" not in memory:
            if self._held_block(agent) is None:
                return None
            hunters = [a for a in self.agents.values() if not a.scripted and not a.dead]
            if not hunters or min(math.dist(a.body.pos, agent.body.pos) for a in hunters) > self.hide_distance:
                return None
            roll = self.rng.random()
            if roll < self.pillar_chance:
                memory["hide"] = "pillar"
                memory["pillar_top"] = math.floor(agent.body.pos[1]) + self.rng.randint(3, 6)
                self.stats["столбов цели"] += 1
            elif roll < self.pillar_chance + self.box_chance:
                memory["hide"] = "box"
                x, y, z = (math.floor(v) for v in agent.body.pos)
                memory["box"] = [(x + dx, y + dy, z + dz) for dy in (0, 1) for dx in (-1, 0, 1) for dz in (-1, 0, 1)
                                 if (dx, dz) != (0, 0)]
                self.stats["коробок цели"] += 1
            else:
                memory["hide"] = "run"
        if memory["hide"] == "pillar":
            if agent.body.pos[1] < memory["pillar_top"] - 0.01 and self._held_block(agent) is not None:
                return {"legs": "jump", "head": "head_idle", "hands": "place_below"}
            return {"legs": "idle", "head": "head_idle", "hands": "hands_idle"}
        if memory["hide"] == "box":
            # По блоку в стену за решение (человек ставит быстро); клетку, где
            # стоит охотник, — позже. Готово или кончились блоки — сидит внутри.
            for cell in memory["box"]:
                if self.world.block(*cell) == AIR and self._place(agent, *cell):
                    break
            return {"legs": "idle", "head": "head_idle", "hands": "hands_idle"}
        return None

    def new_hunt(self, start_blocks: int) -> None:
        """Новая охота: целая арена, всем по start_blocks блоков, человек
        заново решает, прятаться ли (столб, коробка). Прошлую охоту не
        закончили остановкой (кончилось время) — это в сводку."""
        target = next((a for a in self.agents.values() if a.scripted), None)
        if target is not None and self.hunt_clock is not None:
            self.count_hunt_time(target)
            self.hide_timeouts[target.memory.get("hide", "run")] += 1
        self.reset_world()
        self.give_blocks(start_blocks)
        self.hunt_clock = self.time
        for agent in self.agents.values():
            if agent.scripted:
                agent.memory.clear()
            if self.sword_sharpness > 0:  # give @a wooden_sword (hunt.mcfunction)
                agent.sword = True
                agent.holding_block = False
            if self.random_kit and not agent.scripted:  # "человеку" — полный: ему прятаться
                count = 0 if self.rng.random() < KIT_EMPTY_CHANCE else self.rng.randint(1, max(1, start_blocks))
                agent.inventory = {"dirt": count} if count > 0 else {}
                agent.sword = self.sword_sharpness > 0 and self.rng.random() >= KIT_NO_SWORD_CHANCE

    def count_hunt_time(self, target: Agent | None = None) -> None:
        """Время текущей охоты с прошлого учёта — укрытию, которое "человек"
        выбрал (пока не выбрал — бегу)."""
        if self.hunt_clock is None:
            return
        if target is None:
            target = next((a for a in self.agents.values() if a.scripted), None)
            if target is None:
                return  # охота на ботов — укрытий нет
        hide = target.memory.get("hide", "run")
        self.hide_minutes[hide] += (self.time - self.hunt_clock) / 60
        self.hunt_clock = self.time

    def take_scripted_deaths(self) -> list[tuple[str, str | None]]:
        deaths, self.scripted_deaths = self.scripted_deaths, []
        return deaths

    def _target_of(self, agent: Agent) -> dict | None:
        """Как js/target.js: сущность-цель — её позиция сейчас и рост; не
        видна (мертва) — запасная позиция от Python, рост 0."""
        if agent.target_entity is not None:
            other = self.by_entity.get(agent.target_entity)
            if other is not None and not other.absent:
                x, y, z = other.body.pos
                return {"x": js_round(x), "y": js_round(y), "z": js_round(z), "h": PLAYER_ENTITY_HEIGHT}
        if agent.target_fallback is not None:
            p = agent.target_fallback
            return {"x": js_round(p["x"]), "y": js_round(p["y"]), "z": js_round(p["z"]), "h": 0}
        return None

    def _entities_for(self, agent: Agent) -> list[dict]:
        """Как js/entities.js: ближайшие (до radius) — в системе взгляда."""
        radius = self.config["entities"]["radius"]
        max_tracked = self.config["entities"]["max_tracked"]
        ox, oy, oz = agent.body.pos
        fx, fz = -math.sin(agent.body.yaw), -math.cos(agent.body.yaw)
        seen = []
        for other in self.agents.values():
            if other is agent or other.absent:
                continue
            dx, dy, dz = other.body.pos[0] - ox, other.body.pos[1] - oy, other.body.pos[2] - oz
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dist > radius or dist < 1e-6:
                continue
            seen.append((dist, other, dx, dy, dz))
        seen.sort(key=lambda item: item[0])
        out = []
        for i in range(max_tracked):
            if i >= len(seen):
                out.append({"id": 0, "present": 0, "forward": 0, "right": 0, "up": 0, "dist": 0, "type": 0,
                            "name": "", "height": 0})
                continue
            dist, other, dx, dy, dz = seen[i]
            clip = lambda v: max(-1.0, min(1.0, v / radius))  # noqa: E731
            out.append({"id": other.entity_id, "present": 1,
                        "forward": js_round(clip(dx * fx + dz * fz)), "right": js_round(clip(dx * -fz + dz * fx)),
                        "up": js_round(clip(dy)), "dist": js_round(dist / radius), "type": 2,
                        "name": other.name, "height": PLAYER_ENTITY_HEIGHT})
        return out

    def _charge(self, agent: Agent) -> float:
        delay = SWORD_DELAY_TICKS if self._held(agent) == "tool" else ATTACK_DELAY_TICKS
        ticks = (self.time - agent.last_attack_time) / 0.05
        return min(1.0, max(0.0, (ticks + 0.5) / delay))

    def _strike_target(self, agent: Agent) -> Agent | None:
        """Как findEntityInCrosshair (js/actions.js): кого из разрешённых
        достанет удар — до ATTACK_RANGE, до 30° влево-вправо, по высоте любой,
        и не сквозь стену (прямая видимость до центра или до головы)."""
        if not agent.taggable:
            return None
        ex, ey, ez = agent.eye()
        fx, fz = -math.sin(agent.body.yaw), -math.cos(agent.body.yaw)
        best, best_cos = None, ATTACK_CONE_COS
        for entity_id in agent.taggable:
            other = self.by_entity.get(entity_id)
            if other is None or other is agent or other.absent:
                continue
            cx, cy, cz = other.center()
            dx, dy, dz = cx - ex, cy - ey, cz - ez
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)
            if distance > ATTACK_RANGE or distance < 1e-6:
                continue
            flat = math.hypot(dx, dz)
            cos = 1.0 if flat < 0.3 else (dx * fx + dz * fz) / flat
            if cos <= best_cos:
                continue
            if not self._clear(agent.eye(), other.center()) and not self._clear(agent.eye(), other.eye()):
                continue
            best, best_cos = other, cos
        return best

    def _clear(self, start: tuple, end: tuple) -> bool:
        """Прямая видимость (js/vision.js: lineOfSight): на отрезке нет блока.
        В симуляции все блоки — полные кубы, жидкостей и травы нет, поэтому
        хватает луча зрения: блок заслоняет, если луч вошёл в него раньше конца."""
        delta = np.subtract(end, start)
        distance = float(np.linalg.norm(delta))
        if distance < 1e-6:
            return True
        hit_distance, _ = raycast(self.world, np.array([start], dtype=np.float64),
                                  np.array([delta / distance]), distance)
        return not hit_distance[0] < distance

    # --- действия -----------------------------------------------------------

    def receive(self, outbox) -> None:
        """Сообщения от ai_loop за тик: цели и действия."""
        for bot_id, message in outbox.targets.items():
            agent = self.agents.get(bot_id)
            if agent is not None:
                agent.target_entity = message.get("entity_id")
                agent.target_fallback = message.get("position")
        outbox.targets.clear()
        for bot_id, message in outbox.actions.items():
            agent = self.agents.get(bot_id)
            if agent is None or agent.absent:
                continue
            # Ответ Python приходит не мгновенно: в игре (плагин) состояние
            # уходит в начале тика, ответ выполняется в следующем — тик физики
            # бот всегда делает прежнее, а если Python ответил дольше 50 мс
            # (видеокарта занята, "тихий режим" процессора) — два. Было 0–1:
            # мост в симуляции учился тормозить у края впритык, а вживую бот
            # успевал проехать лишние 0.2 блока и падал (2026-09-29).
            delay = (1 + (self.rng.random() < 0.5)) if self.latency else 0
            agent.pending = (delay, message.get("actions") or {}, message.get("tag_ids") or [])
        outbox.actions.clear()

    def _execute(self, agent: Agent, actions: dict, tag_ids: list) -> None:
        """executeAll из js/actions.js: отпустить клавиши, ноги, голова, руки."""
        agent.taggable = set(tag_ids)
        body = agent.body
        body.controls = {}
        agent.hold_ticks = HOLD_TICKS
        legs = actions.get("legs", "idle")
        if legs == "walk_forward":
            body.controls = {"forward": True, "jump": step_ahead(body, self.world)}
        elif legs == "sprint_forward":
            body.controls = {"forward": True, "sprint": True, "jump": step_ahead(body, self.world)}
        elif legs in MOVES:
            body.controls = dict(MOVES[legs])
        elif legs == "turn_left":
            body.yaw = normalize_yaw(body.yaw + TURN_STEP)
        elif legs == "turn_right":
            body.yaw = normalize_yaw(body.yaw - TURN_STEP)

        head = actions.get("head", "head_idle")
        if head == "look_up":
            body.pitch = min(math.pi / 2, body.pitch + LOOK_STEP)
        elif head == "look_down":
            body.pitch = max(-math.pi / 2, body.pitch - LOOK_STEP)
        elif head == "look_left":
            body.yaw = normalize_yaw(body.yaw + LOOK_YAW_STEP)
        elif head == "look_right":
            body.yaw = normalize_yaw(body.yaw - LOOK_YAW_STEP)

        # Руки — как js/actions.js: удар по сущности (мгновенный, бросает
        # копку), иначе копать блок в прицеле; столб под себя; блок перед
        # собой; выбросить то, что в руке. Еды и брони в симуляции нет.
        hands = actions.get("hands", "hands_idle")
        if hands == "attack_center":
            victim = self._strike_target(agent)
            if victim is not None and agent.sword and agent.holding_block:
                # В руке блок (строил), меч есть — сперва взять меч, бить —
                # следующим решением: смена предмета сбрасывает заряд (как плагин и JS).
                agent.dig = None
                agent.holding_block = False
                agent.last_attack_time = self.time
            elif victim is not None:
                agent.dig = None
                self._hit(agent, victim)
            elif not self._busy(agent):
                self._start_dig(agent)
        elif hands == "place_below":
            self._start_place_below(agent)
        elif hands == "place_front":
            self._place_front(agent)
        elif hands == "drop_item":
            held = self._held(agent)
            if held == "tool" and not self._busy(agent):
                agent.sword = False  # в руке меч — его и выбросил, как плагин (player.drop)
            elif held == "block" and not self._busy(agent):
                del agent.inventory[self._held_block(agent)]  # вся стопка — как bot.tossStack

    # --- руки: копка и постройка ------------------------------------------

    def _busy(self, agent: Agent) -> bool:
        return agent.dig is not None or agent.place_support is not None or agent.hands_busy_ticks > 0

    def _held(self, agent: Agent) -> str:
        """Что в руке для сети (state.inventory.held, как InventoryInfo.heldKind
        плагина): меч — "tool"; блок — если макрос взял его для постройки или
        меча нет вовсе (тогда блоки в первом слоте, он и выбран); иначе "none"."""
        if agent.sword and not agent.holding_block:
            return "tool"
        return "block" if self._held_block(agent) is not None else "none"

    def _take_block(self, agent: Agent) -> None:
        """Макрос постройки берёт блок в руку вместо меча."""
        if agent.sword and not agent.holding_block:
            agent.holding_block = True
            agent.last_attack_time = self.time  # смена предмета в руке сбрасывает заряд (Player.tick)

    def _tick_weapon(self, agent: Agent) -> None:
        """Тик физики: руки свободны WEAPON_BACK_TICKS — меч обратно в руку."""
        agent.idle_hands_ticks = 0 if self._busy(agent) else agent.idle_hands_ticks + 1
        if agent.idle_hands_ticks == WEAPON_BACK_TICKS and agent.sword and agent.holding_block:
            agent.holding_block = False
            agent.last_attack_time = self.time  # и снова заряд с нуля

    @staticmethod
    def _held_block(agent: Agent) -> str | None:
        """Что в руке: первый блок инвентаря (как слот 0 хотбара у бота в игре:
        выкопанное ложится туда, он и выбран) или None — рука пуста. Слитки
        (бедварс: железо, золото) — не блоки: их не ставят."""
        return next((name for name, count in agent.inventory.items() if count > 0 and name in BLOCK_ID), None)

    def _start_dig(self, agent: Agent) -> None:
        hit = center_hit(self.world, agent.eye(), agent.body.yaw, agent.body.pitch)
        if hit is None or hit[4] > DIG_REACH or HARDNESS[BLOCK_NAMES[hit[5]]] < 0:
            return  # далеко, воздух до горизонта или бедрок/барьер
        if not self._breakable((hit[0], hit[1], hit[2]), hit[5]):
            return  # бедварс: карту не ломают
        agent.dig = (hit[0], hit[1], hit[2])
        agent.dig_progress = 0.0
        agent.dig_ticks = 0

    def _start_place_below(self, agent: Agent) -> None:
        """Столб: опора — ближайший твёрдый блок снизу (до 3); ставить, когда
        прыжок поднимет ноги на блок над ней (js/actions.js: placeBelow)."""
        if self._busy(agent) or self._held_block(agent) is None:
            return
        self._take_block(agent)
        x, y, z = agent.body.pos
        bx, bz = math.floor(x), math.floor(z)
        for by in range(math.floor(y) - 1, math.floor(y) - 4, -1):
            if self.world.solid(bx, by, bz):
                agent.place_support = (bx, by, bz)
                agent.place_wait = 0
                if agent.body.on_ground:  # подпрыгнуть самому — как плагин и js/actions.js (placeBelow)
                    agent.body.controls["jump"] = True
                    agent.hold_ticks = HOLD_TICKS
                return

    def _place_front(self, agent: Agent) -> None:
        """Блок на грань того, во что смотрит прицел (js/actions.js: placeFront)."""
        if self._busy(agent) or self._held_block(agent) is None:
            return
        self._take_block(agent)
        dest = self._place_front_cell(agent)
        if dest is not None and self._place(agent, *dest):
            agent.hands_busy_ticks = 1

    def _place_front_cell(self, agent: Agent) -> tuple | None:
        return place_front_cell(self.world, agent.eye(), agent.body.yaw, agent.body.pitch, agent.body.pos)

    def _can_place_front(self, agent: Agent) -> bool:
        """state.can_place: place_front сейчас поставил бы блок (js/actions.js:
        canPlaceFront) — есть чем, грань в прицеле, клетка за ней свободна."""
        if agent.absent or self._held_block(agent) is None:
            return False
        dest = self._place_front_cell(agent)
        return dest is not None and self._free_for_block(*dest)

    def _free_for_block(self, x: int, y: int, z: int) -> bool:
        return free_for_block(self.world, (x, y, z), [a.body.pos for a in self.agents.values() if not a.absent])

    def _place(self, agent: Agent, x: int, y: int, z: int) -> bool:
        block = self._held_block(agent)
        if block is None or not self._free_for_block(x, y, z):
            return False
        if not self.world.set_block(x, y, z, block):
            return False
        self.placed.add((x, y, z))
        self.stats["поставила цель" if agent.scripted else "поставили боты"] += 1
        agent.inventory[block] -= 1
        if agent.inventory[block] <= 0:
            del agent.inventory[block]
        return True

    def _tick_hands(self, agent: Agent) -> None:
        """Тик физики: долгие дела рук — копка (прогресс, как на сервере) и
        ожидание прыжка для столба."""
        if agent.hands_busy_ticks > 0:
            agent.hands_busy_ticks -= 1
        if agent.place_support is not None:
            sx, sy, sz = agent.place_support
            agent.place_wait += 1
            if agent.body.pos[1] >= sy + 2:
                agent.place_support = None
                self._place(agent, sx, sy + 1, sz)
            elif agent.place_wait > PLACE_BELOW_WAIT_TICKS:
                agent.place_support = None  # так и не подпрыгнул
            elif agent.body.on_ground:
                # Приземлился, а блок ещё не поставлен (команда пришла в
                # воздухе) — прыгнуть снова, как плагин и js/actions.js.
                agent.body.controls["jump"] = True
                agent.hold_ticks = HOLD_TICKS
        if agent.dig is None:
            return
        hit = center_hit(self.world, agent.eye(), agent.body.yaw, agent.body.pitch)
        if hit is None or (hit[0], hit[1], hit[2]) != agent.dig or hit[4] > DIG_REACH:
            agent.dig = None  # отвернулся или отошёл — копка прерывается
            return
        name = BLOCK_NAMES[hit[5]]
        speed = 1.0 if agent.body.on_ground else 0.2  # в воздухе — впятеро медленнее
        agent.dig_progress += speed / HARDNESS[name] / (30 if name in HAND_HARVEST else 100)
        agent.dig_ticks += 1
        if agent.dig_progress >= 1.0:
            self.world.set_block(*agent.dig, "air")
            self.placed.discard(agent.dig)
            self.stats["выкопали боты" if not agent.scripted else "выкопала цель"] += 1
            if name.endswith("_bed"):
                # Кровать — две клетки: сломал одну, пропадает и вторая (как в игре).
                # Предмета нет (в бедварсе сломанная кровать ничего не даёт).
                x, y, z = agent.dig
                for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    if BLOCK_NAMES[self.world.block(x + dx, y, z + dz)] == name:
                        self.world.set_block(x + dx, y, z + dz, "air")
                self.stats["сломали кроватей"] += 1
                agent.dig = None
                return
            if name in HAND_HARVEST:  # каменный кирпич рукой — ничего не выпадает
                drop = DROPS.get(name, name)
                agent.inventory[drop] = agent.inventory.get(drop, 0) + 1
            agent.dig = None
        elif agent.dig_ticks >= DIG_TIMEOUT_TICKS:
            agent.dig = None

    def _breakable(self, cell: tuple, block: int) -> bool:
        """Бедварс: ломать можно только поставленное игроками и кровати (как
        плагин в мире бедварса); в остальных задачках — всё, что не бедрок."""
        return not self.bedwars or cell in self.placed or BLOCK_NAMES[block].endswith("_bed")

    def _map_world(self, name: str) -> ArenaWorld:
        if name not in self.map_worlds:
            self.map_worlds[name] = ArenaWorld.for_bedwars_map(name)
        return self.map_worlds[name]

    def load_map(self, name: str) -> None:
        """/mcbot bedwars <карта>: карта заново (как плагин: мир перезагружается
        из её регионов) — всё построенное и сломанное пропадает."""
        self.world = self._map_world(name)
        self.world.reset()
        self.placed.clear()
        x, z = 0, 0
        self.spawn = (x + 0.5, self.world.top_y(x, z), z + 0.5)
        self.reset_world()

    def reset_world(self) -> None:
        """Арена как построена (после охоты, в начале раунда салок): копка и
        столбы прошлого раза убраны, маршруты — заново."""
        self.world.reset()
        self.placed.clear()
        for agent in self.agents.values():
            agent.dig = None
            agent.place_support = None
            agent.route = RoutePlanner(self.world, self.config["route"])

    def give_blocks(self, count: int, name: str = "dirt") -> None:
        """Каждому — count блоков (как give @a dirt в server/functions)."""
        for agent in self.agents.values():
            agent.inventory = {name: count} if count > 0 else {}

    def _hit(self, attacker: Agent, victim: Agent) -> None:
        # Удар сам наводит голову на того, кого бьёт (bot.lookAt), и уходит
        # судье (attacked_id) — урон и отбрасывание только вне неуязвимости.
        # Урон — как в игре: 0.2 + 0.8 * заряд², взмах сбрасывает заряд.
        charge = self._charge(attacker)
        attacker.last_attack_time = self.time
        body = attacker.body
        crit = charge > 0.9 and not body.on_ground and body.vel[1] < 0 and not body.controls.get("sprint")
        ex, ey, ez = attacker.eye()
        cx, cy, cz = victim.center()
        dx, dy, dz = cx - ex, cy - ey, cz - ez
        attacker.body.yaw = math.atan2(-dx, -dz)
        attacker.body.pitch = math.atan2(dy, math.hypot(dx, dz))
        attacker.attacked_id = victim.entity_id
        self.hits += 1
        if self.time < victim.invulnerable_until:
            return
        victim.invulnerable_until = self.time + INVULNERABLE_SECONDS
        if self._held(attacker) == "tool":
            # Меч, как Player.attack в игре: оружие * (0.2 + 0.8 * заряд²) (крит
            # x1.5) плюс острота (0.5 * уровень + 0.5) * заряд.
            damage = (SWORD_DAMAGE * (0.2 + 0.8 * charge * charge) * (1.5 if crit else 1.0)
                      + (0.5 * self.sword_sharpness + 0.5) * charge)
        else:
            damage = FIST_DAMAGE * (0.2 + 0.8 * charge * charge) * (1.5 if crit else 1.0)
        victim.health -= damage
        victim.hurt_by = attacker.entity_id
        attacker.damage_dealt.append({"id": victim.entity_id, "charge": js_round(charge), "crit": crit})
        vel = victim.body.vel
        flat = math.hypot(dx, dz) or 1.0
        vel[0] = vel[0] / 2 + KNOCKBACK * dx / flat
        vel[2] = vel[2] / 2 + KNOCKBACK * dz / flat
        if victim.body.on_ground:
            vel[1] = min(0.4, vel[1] / 2 + KNOCKBACK)
        victim.body.on_ground = False
        for listener in self.agents.values():  # звук удара и урона
            if not listener.absent:
                listener.hearing.add_sound(listener.body.pos, victim.body.pos, 1.0)
        if victim.health <= 0:
            self._kill(victim, killer=attacker)

    # --- ход времени --------------------------------------------------------

    def step(self) -> None:
        """150 мс: три тика физики; действия применяются на своём тике."""
        for physics_index in range(PHYSICS_TICKS_PER_DECISION):
            for agent in self.agents.values():
                if agent.absent:
                    continue
                if agent.pending is not None and agent.pending[0] == physics_index:
                    _, actions, tag_ids = agent.pending
                    agent.pending = None
                    self._execute(agent, actions, tag_ids)
                if agent.hold_ticks > 0:
                    agent.hold_ticks -= 1
                    if agent.hold_ticks == 0:
                        # Движение отпускаем, шифт держим до следующего решения (как плагин и js).
                        agent.body.controls = {"sneak": True} if agent.body.controls.get("sneak") else {}
                physics_tick(agent.body, self.world)
                self._tick_hands(agent)
                self._tick_weapon(agent)
                if self.bedwars and agent.body.pos[1] < self.world.origin[1] - BEDWARS_VOID_DEPTH:
                    self._kill(agent)
                if self.course is not None and agent.body.pos[1] < self.course.start(0)[1] - 80:
                    # Страховка: падение засчитывает и возвращает судья (как в игре,
                    # ai_loop._bridge_turn) задолго до этой глубины.
                    self._kill(agent)
            self.time += TICK_SECONDS / PHYSICS_TICKS_PER_DECISION
            self._regenerate()
        self._respawn()

    def _regenerate(self) -> None:
        for agent in self.agents.values():
            if agent.absent or agent.health >= 20 or self.time < agent.regen_at:
                continue
            agent.health = min(20.0, agent.health + 1.0)
            agent.regen_at = self.time + self.regen_seconds

    def _kill(self, agent: Agent, killer: Agent | None = None) -> None:
        if agent.dead:
            return
        if agent.scripted and self.hunt_clock is not None:
            self.count_hunt_time(agent)
            self.hide_kills[agent.memory.get("hide", "run")] += 1
            self.hunt_clock = None  # до следующей охоты время не идёт
        if agent.scripted:  # как сообщение сервера о смерти игрока
            self.scripted_deaths.append((agent.name, killer.name if killer else None))
        agent.health = 0.0
        agent.dead_until = self.time + RESPAWN_SECONDS
        agent.body.controls = {}
        agent.pending = None

    def _respawn(self) -> None:
        for agent in self.agents.values():
            if agent.dead and self.time >= agent.dead_until:
                agent.dead_until = None
                agent.health = 20.0
                agent.body.teleport(*(self.spawn if self.course is None else self._lane_start(agent)))
                if agent.scripted:
                    self._place_randomly(agent)  # человек — не у спавна, где стоят охотники

    # --- команды сервера (от судьи салок) -------------------------------------

    def command(self, text: str) -> None:
        words = text.lstrip("/").split()
        # "execute in <мир> run <команда>" — мир задачки: в симуляции он один.
        if len(words) > 4 and words[0] == "execute" and words[1] == "in" and words[3] == "run":
            words = words[4:]
        if not words:
            return
        if words[0] == "tp" and len(words) >= 5:
            self._teleport(words[1], words[2:])
            return
        if words[0] == "fill" and len(words) == 8:
            x1, y1, z1, x2, y2, z2 = map(int, words[1:7])
            name = words[7].split(":")[-1]
            for x in range(min(x1, x2), max(x1, x2) + 1):
                for y in range(min(y1, y2), max(y1, y2) + 1):
                    for z in range(min(z1, z2), max(z1, z2) + 1):
                        self.world.set_block(x, y, z, name)
            return
        if words[0] == "give" and len(words) >= 3:
            item = words[2].split(":")[-1].split("[")[0]
            count = int(words[3]) if len(words) > 3 else 1
            for agent in self._named(words[1]):
                if item.endswith("_sword"):
                    agent.sword, agent.holding_block = True, False  # меч сразу в руке, как у плагина
                else:
                    agent.inventory[item] = agent.inventory.get(item, 0) + count
            return
        if words[0] == "clear" and len(words) >= 2:
            item = words[2].split(":")[-1] if len(words) > 2 else None
            count = int(words[3]) if len(words) > 3 else None
            for agent in self._named(words[1]):
                if item is None:
                    agent.inventory = {}
                    agent.sword = False
                elif item in agent.inventory:  # /clear <ник> <предмет> [сколько] — бедварс: плата в магазине
                    left = 0 if count is None else agent.inventory[item] - count
                    if left > 0:
                        agent.inventory[item] = left
                    else:
                        del agent.inventory[item]
            return
        if words[0] == "mcbot" and len(words) >= 3 and words[1] == "bedwars":
            self.load_map(" ".join(words[2:]))
            return
        if words[0] == "gamemode" and len(words) == 3:
            # Бедварс: выбыл — зритель (вне игры), новая игра — снова в игре.
            for agent in self._named(words[2]):
                agent.out = words[1] == "spectator"
                if agent.out:
                    agent.body.controls, agent.pending, agent.dig = {}, None, None
            return
        if words[0] == "kill" and len(words) == 2:
            for agent in self.agents.values():
                if agent.name == words[1]:
                    self._kill(agent)
        elif words[0] == "spreadplayers":
            self.spread()
        # /team, /effect, scoreboard — пометки для глаз, в симуляции не нужны

    def _named(self, selector: str) -> list:
        return [a for a in self.agents.values() if selector in ("@a", a.name)]

    def _teleport(self, name: str, args: list) -> None:
        """/tp <ник> x y z [yaw pitch] — углы сервера в градусах (yaw 0 — на
        юг), в симуляции — конвенция mineflayer, как Geometry.serverYaw в
        плагине наоборот."""
        x, y, z = map(float, args[:3])
        for agent in self._named(name):
            agent.body.teleport(x, y, z)
            if len(args) >= 5:
                agent.body.yaw = normalize_yaw(math.pi - math.radians(float(args[3])))
                agent.body.pitch = -math.radians(float(args[4]))
            agent.route = RoutePlanner(self.world, self.config["route"])

    def _lane_start(self, agent: Agent) -> tuple:
        return self.course.start(agent.id - 1)

    def _place_randomly(self, agent: Agent) -> None:
        cx, cz = self.arena_cfg["center"]
        max_range = self.arena_cfg["size"] // 2 - 3
        x = cx + self.rng.randint(-max_range, max_range)
        z = cz + self.rng.randint(-max_range, max_range)
        agent.body.teleport(x + 0.5, self.world.top_y(x, z), z + 0.5)

    def spread(self) -> None:
        """Как /spreadplayers: всех живых — в случайные места арены, не ближе
        4 блоков друг к другу, на самый верхний блок колонки. Раунд салок
        начинается на целой арене — прошлую копку и постройку убрать."""
        self.reset_world()
        if self.course is not None:
            for agent in self.agents.values():
                if not agent.dead:
                    agent.body.teleport(*self._lane_start(agent))
                    agent.body.yaw = self.rng.uniform(-math.pi, math.pi)
                    agent.route = RoutePlanner(self.world, self.config["route"])
            return
        cx, cz = self.arena_cfg["center"]
        max_range = self.arena_cfg["size"] // 2 - 3
        placed = []
        for agent in self.agents.values():
            if agent.dead:
                continue
            for _ in range(200):
                x = cx + self.rng.randint(-max_range, max_range)
                z = cz + self.rng.randint(-max_range, max_range)
                if all(math.hypot(x - px, z - pz) >= 4 for px, pz in placed):
                    break
            placed.append((x, z))
            agent.body.teleport(x + 0.5, self.world.top_y(x, z), z + 0.5)
            agent.body.yaw = self.rng.uniform(-math.pi, math.pi)
            agent.route = RoutePlanner(self.world, self.config["route"])


def _block_count(inventory: dict) -> int:
    """Сколько строительных блоков (слитки бедварса — не блоки)."""
    return sum(count for name, count in inventory.items() if name in BLOCK_ID)


def place_front_cell(world: ArenaWorld, eye: tuple, yaw: float, pitch: float, pos) -> tuple | None:
    """Куда встал бы блок place_front (js/actions.js: placeFrontTarget): клетка
    у грани в прицеле; None — прицел пуст или дальше руки, грань неизвестна,
    это клетка самого бота."""
    hit = center_hit(world, eye, yaw, pitch)
    if hit is None or hit[4] > PLACE_REACH or hit[3] == FACE_UNKNOWN:
        return None
    dx, dy, dz = FACE_VECTORS[hit[3]]
    dest = (hit[0] + dx, hit[1] + dy, hit[2] + dz)
    feet = (math.floor(pos[0]), math.floor(pos[1]), math.floor(pos[2]))
    if dest == feet or dest == (feet[0], feet[1] + 1, feet[2]):
        return None  # в свою клетку сервер блок не поставит
    return dest


# Размеры тела для "клетка занята" — как у сервера: ширина и рост сущности —
# float (0.6f / 2 = 0.30000001), и бот, стоящий ровно впритык к клетке
# (x = 8.3, край на 8.0), для сервера её задевает — блок туда не встанет.
BUILD_HALF_WIDTH = float(np.float32(np.float32(0.6) / np.float32(2)))
BUILD_HEIGHT = float(np.float32(HEIGHT))


def free_for_block(world: ArenaWorld, cell: tuple, bodies: list) -> bool:
    """Клетка — воздух внутри мира, и ни одно тело (позиции ног) в ней не стоит."""
    x, y, z = cell
    if not world.inside(x, y, z) or world.block(x, y, z) != AIR:
        return False
    for ox, oy, oz in bodies:
        if (x < ox + BUILD_HALF_WIDTH and x + 1 > ox - BUILD_HALF_WIDTH
                and z < oz + BUILD_HALF_WIDTH and z + 1 > oz - BUILD_HALF_WIDTH
                and y < oy + BUILD_HEIGHT and y + 1 > oy):
            return False
    return True


def _shape_face(block: int, cell: list, eye: tuple, d: tuple, step: list) -> tuple | None:
    """Грань и дальность (t) попадания в коробки блока — как
    RaycastIterator.intersect prismarine-world (та же грань при равенстве)."""
    big = 1.7976931348623157e308
    inv = [1.0 / v if v != 0 else big for v in d]
    p = (eye[0] - cell[0], eye[1] - cell[1], eye[2] - cell[2])
    best, best_face = big, FACE_UNKNOWN
    for shape in SHAPES[block]:
        tmin = (shape[0 if inv[0] > 0 else 3] - p[0]) * inv[0]
        tmax = (shape[3 if inv[0] > 0 else 0] - p[0]) * inv[0]
        tymin = (shape[1 if inv[1] > 0 else 4] - p[1]) * inv[1]
        tymax = (shape[4 if inv[1] > 0 else 1] - p[1]) * inv[1]
        face = FACE_WEST if step[0] > 0 else FACE_EAST
        if tmin > tymax or tymin > tmax:
            continue
        if tymin > tmin:
            tmin = tymin
            face = FACE_BOTTOM if step[1] > 0 else FACE_TOP
        tmax = min(tmax, tymax)
        tzmin = (shape[2 if inv[2] > 0 else 5] - p[2]) * inv[2]
        tzmax = (shape[5 if inv[2] > 0 else 2] - p[2]) * inv[2]
        if tmin > tzmax or tzmin > tmax:
            continue
        if tzmin > tmin:
            tmin = tzmin
            face = FACE_NORTH if step[2] > 0 else FACE_SOUTH
        if tmin < best:
            best, best_face = tmin, face
    return None if best == big else (best_face, best)


def center_hit(world: ArenaWorld, eye: tuple, yaw: float, pitch: float, max_distance: float = VIEW_DISTANCE):
    """Луч прицела (js/vision.js: centerRaycast) до первого блока: (x, y, z,
    грань, дальность, id блока) или None. В симуляции все блоки — полные кубы:
    грань — та, через которую луч вошёл в клетку (ось последнего шага обхода,
    как RaycastIterator.intersect для куба). Обход клеток — как у js/vision.js."""
    cos_pitch = math.cos(pitch)
    d = (-math.sin(yaw) * cos_pitch, math.sin(pitch), -math.cos(yaw) * cos_pitch)
    cell = [math.floor(eye[0]), math.floor(eye[1]), math.floor(eye[2])]
    step = [(v > 0) - (v < 0) for v in d]
    big = 1e308
    t_delta = [abs(1.0 / v) if v != 0 else big for v in d]
    t_max = [abs((cell[i] + (1 if d[i] > 0 else 0) - eye[i]) / d[i]) if d[i] != 0 else big for i in range(3)]
    axis, t_entry = None, 0.0
    while True:
        block = world.block(*cell)
        if block != AIR and not PASS_THROUGH[block] and not FULL_CUBE[block]:
            # Не целый куб (карты бедварса): жидкость, табличка — попадание в
            # центр клетки, грань неизвестна; ступень, плита — по коробкам формы,
            # промах — луч идёт дальше (RaycastIterator.intersect).
            if not COLLIDES[block]:
                center = (cell[0] + 0.5 - eye[0], cell[1] + 0.5 - eye[1], cell[2] + 0.5 - eye[2])
                return cell[0], cell[1], cell[2], FACE_UNKNOWN, math.sqrt(sum(v * v for v in center)), block
            hit = _shape_face(block, cell, eye, d, step)
            if hit is not None:
                face, t = hit
                return cell[0], cell[1], cell[2], face, math.sqrt(sum((v * t) ** 2 for v in d)), block
        elif block != AIR and not PASS_THROUGH[block]:
            if axis is None:
                return cell[0], cell[1], cell[2], FACE_UNKNOWN, 0.0, block
            if axis == 0:
                face = FACE_WEST if step[0] > 0 else FACE_EAST
            elif axis == 1:
                face = FACE_BOTTOM if step[1] > 0 else FACE_TOP
            else:
                face = FACE_NORTH if step[2] > 0 else FACE_SOUTH
            return cell[0], cell[1], cell[2], face, t_entry, block
        if min(t_max) > max_distance:
            return None
        tx, ty, tz = t_max
        axis = (0 if tx < tz else 2) if tx < ty else (1 if ty < tz else 2)
        t_entry = t_max[axis]
        cell[axis] += step[axis]
        t_max[axis] += t_delta[axis]

