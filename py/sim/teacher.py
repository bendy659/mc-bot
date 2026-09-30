"""Учитель в симуляции: простые правила вместо сети — чтобы мозги сразу
видели, как надо (демонстрации для DQfD, py/train_tag_sim.py --teachers).

Учитель смотрит на то же состояние, что и сеть (точка маршрута, "удар
достанет", сетка зрения, угроза) и выбирает действия теми же макросами —
поэтому его ходы годятся мозгу как примеры "в таком состоянии делай так".
Не идеальный игрок — просто толковый: дальше мозг доучивается сам.

  - водящий (chase): развернуться к точке маршрута (ноги — по 30°, голова
    доводит по 10°), бежать; удар достаёт — бить;
  - убегающий (flee): по сетке зрения на уровне глаз — куда свободно и где
    дальше от водящего; развернуться туда и бежать. Свободное место важно:
    иначе загнал бы себя в угол.
Голова держит взгляд ровно.
"""

from __future__ import annotations

import math
import os

from config import CONFIG
from training_modules.targets import aim_errors

FINE = math.radians(5)     # точнее не надо: голова доводит шагом 10°
AROUND = math.radians(135)  # цель дальше за спиной — развернуться разом (turn_around)
COARSE = math.radians(45)  # грубее — поворот ногами (шаг 30°) на месте; мельче — голова на бегу
LEVEL = math.radians(6)    # наклон взгляда, который голова исправляет
WALL = 2.5                 # блоков: стена ближе — туда не бежать


def _steer(error: float) -> tuple[str, str]:
    """Ноги и голова, чтобы развернуться на error радиан (плюс — влево).
    Цель за спиной — разворот за одно решение; просто сбоку — ноги и голова
    разом (40° за решение): по 30° к врагу поворачивались "очень долго" (автор)."""
    if abs(error) > AROUND:
        return "turn_around", "head_idle"
    if abs(error) > COARSE:
        side = "left" if error > 0 else "right"
        return f"turn_{side}", f"look_{side}"
    if abs(error) > FINE:
        return "sprint_forward", ("look_left" if error > 0 else "look_right")
    return "sprint_forward", "head_idle"


def _level(me: dict, head: str) -> str:
    """Голова свободна — выровнять взгляд по горизонту."""
    if head == "head_idle" and abs(me["pitch"]) > LEVEL:
        return "look_down" if me["pitch"] > 0 else "look_up"
    return head


def chase_actions(state: dict, module) -> dict:
    me = state["self"]
    target = module.observe(state).get("target")  # точка маршрута (рядом — сама цель)
    hands = "attack_center" if state.get("strike") else "hands_idle"
    if target is None:
        return {"legs": "idle", "head": _level(me, "head_idle"), "hands": hands}
    legs, head = _steer(aim_errors(me, target)[0])
    return {"legs": legs, "head": _level(me, head), "hands": hands}


REACH_UP = 2.0     # цель выше на столько (блоков) — с земли не достать: наверх
# Лезть и копать за целью на столбе (1) или только догонять и бить (0) — для
# сравнения учителей (MCBOT_TEACHER_PILLAR=0 python py/train_tag_sim.py ...).
PILLAR_TACTICS = os.environ.get("MCBOT_TEACHER_PILLAR", "1") == "1"
NEAR = 3.0         # и ближе стольки по горизонтали — это она на столбе рядом
PERCHED = 1.5      # ноги выше пола арены на столько — значит, стоит на столбе
FLOOR_Y = CONFIG["server"]["arena"]["floor_y"]


DIG_STAND = 2.5    # копать блок под целью — стоя в стольких блоках от столба: вплотную
                   # луч снизу вверх упирается в столб ниже этого блока
DIG_MAX_RISE = 6.0 # цель выше — блок под ней с земли не выкопать (заслоняет сам столб)
ADJACENT = 1.5     # столб — только вплотную к столбу цели: наверху она в зоне удара
                   # (вживую боты строились в 2.5–3 блоках и не доставали — автор)
STAY_REACH = 3.5   # на своём столбе стоять, пока цель ближе стольки и не ниже — не шагать вниз
BREACH_RANGE = 8.0 # цель ближе стольки на земле, а маршрута нет (коробка, стены) — ломать стену
DIG_CLOSE = 2.0    # блок в прицеле ближе стольки — это стена вплотную, копать
LOOK_DOWN_MAX = -0.6  # ниже не смотреть, вскрывая стену (−40° — ещё не пол под ногами)
FIGHT_BACK = 2.5   # бот-цель: охотник ближе стольки — развернуться и ударить (откинуть)


def _kill_charge(sharpness: int) -> float:
    """С какого заряда удар деревянным мечом с этой остротой убивает с
    полного здоровья (20): 4·(0.2 + 0.8·c²) + (0.5·острота + 0.5)·c = 20,
    как Player.attack в игре. Нет меча — только полный заряд."""
    if sharpness <= 0:
        return 1.0
    bonus = 0.5 * sharpness + 0.5
    return (-bonus + math.sqrt(bonus * bonus + 4 * 3.2 * 19.2)) / (2 * 3.2)


# Меч в охоте — как в server/functions/hunt.mcfunction (config.json
# modules.hunt.sword_sharpness): с остротой 255 убивает удар уже с 15% заряда.
KILL_CHARGE = _kill_charge(CONFIG["modules"].get("hunt", {}).get("sword_sharpness", 0))


def _charged(state: dict, full: bool = False) -> bool:
    """Пора бить: кулаком (или блоком в руке) — только в полную силу, слабый
    удар почти ничего не снимает, а заряд сбрасывает; мечом — как только удар
    убивает (ждать полного заряда — цель уходит из зоны удара). full — только
    в полную силу и мечом: в бедварсе меч простой, а порог "убивает сразу" —
    от меча охоты (острота 255), и учитель махал впустую (автор: "удар не копят")."""
    need = 0.9
    if state.get("inventory", {}).get("held") == "tool" and not full:
        need = min(need, KILL_CHARGE + 0.1)
    return state.get("attack_charge", 1.0) >= need


def _climber(me_id: int, me: dict, goal: dict, peers: list) -> bool:
    """Лезть на столб за целью — только ближайшему к ней охотнику (при
    равенстве — с меньшим номером): когда лезли все, они толкались у столба,
    не могли поставить блок и прыгали на месте (мозг копировал эти прыжки)."""
    mine = math.hypot(goal["x"] - me["x"], goal["z"] - me["z"])
    for peer_id, other in peers:
        flat = math.hypot(goal["x"] - other["x"], goal["z"] - other["z"])
        if flat < mine - 1e-6 or (abs(flat - mine) <= 1e-6 and peer_id < me_id):
            return False
    return True


def _is_climber(state: dict, me_id: int, me: dict, goal: dict, peers: list) -> bool:
    """Лезть ли мне на столб: судья охоты (ai_loop) кладёт это в состояние —
    hunt_closest, его же видит сеть (вход closest), и учитель с сетью
    согласны, кто ближайший; без судьи — посчитать самому."""
    if "hunt_closest" in state:
        return bool(state["hunt_closest"])
    return _climber(me_id, me, goal, peers)


def _aim_head(yaw_error: float, pitch_error: float) -> str:
    if abs(pitch_error) > FINE:
        return "look_up" if pitch_error > 0 else "look_down"
    if abs(yaw_error) > FINE:
        return "look_left" if yaw_error > 0 else "look_right"
    return "head_idle"


def _dig_under(state: dict, me: dict, goal: dict) -> tuple[str, str, str]:
    """Выкопать блок прямо под целью — она спустится на блок. Стоять в
    DIG_STAND блоках от столба (ближе — луч упрётся в столб ниже), навести
    взгляд на этот блок и копать, когда прицел на нём."""
    block = {"x": math.floor(goal["x"]) + 0.5, "y": math.floor(goal["y"]) - 0.5, "z": math.floor(goal["z"]) + 0.5}
    yaw_error, pitch_error = aim_errors(me, block)
    flat = math.hypot(block["x"] - me["x"], block["z"] - me["z"])
    if abs(yaw_error) > COARSE:
        legs = "turn_left" if yaw_error > 0 else "turn_right"
    elif flat > DIG_STAND + 0.5:
        legs = "walk_forward"
    elif flat < DIG_STAND - 0.5:
        legs = "walk_back"
    else:
        legs = "idle"
    center = state.get("center_block")
    aimed = abs(yaw_error) < 0.15 and abs(pitch_error) < 0.1 and center is not None and center["distance"] <= 4.5
    return legs, _aim_head(yaw_error, pitch_error), "attack_center" if aimed else "hands_idle"


def _face(me: dict, goal: dict) -> tuple[str, str]:
    """Повернуться к цели, не сходя с места (удар — по горизонтали, ±30°)."""
    error = aim_errors(me, goal)[0]
    legs = ("turn_left" if error > 0 else "turn_right") if abs(error) > COARSE else "idle"
    head = ("look_left" if error > 0 else "look_right") if FINE < abs(error) <= COARSE else "head_idle"
    return legs, head


def _perched(me: dict) -> bool:
    """Стоит на своём столбе (ноги выше пола арены) — шагнуть = упасть."""
    return me["y"] - FLOOR_Y >= PERCHED


def _reach_up(state: dict, me: dict, goal: dict, me_id: int, peers: list) -> tuple[str, str, str]:
    """Цель на столбе рядом: ближайший охотник с блоками подходит вплотную к
    её столбу и строит свой следом (прыжок и блок под себя); на своём столбе
    без блоков — стоит и смотрит на неё (шаг — падение); остальные копают блок
    под ней с земли, а если она слишком высоко — ждут рядом."""
    flat = math.hypot(goal["x"] - me["x"], goal["z"] - me["z"])
    if state.get("inventory", {}).get("blocks", 0) > 0 and _is_climber(state, me_id, me, goal, peers):
        if not _perched(me) and flat > ADJACENT:
            legs, head = _steer(aim_errors(me, goal)[0])  # сначала вплотную к её столбу
            return legs, head, "hands_idle"
        return "jump", "head_idle", "place_below"
    if _perched(me):
        legs, head = _face(me, goal)
        return legs, head, "hands_idle"
    if goal["y"] - me["y"] <= DIG_MAX_RISE:
        return _dig_under(state, me, goal)
    legs, head = _face(me, goal)
    return legs, head, "hands_idle"


def _breach(state: dict, me: dict, goal: dict) -> tuple[str, str, str]:
    """Цель рядом на земле, а маршрута к ней нет (закрылась коробкой, за
    стеной): к ней напрямую; блок вплотную в прицеле — копать (сперва на
    уровне глаз, потом у ног); упёрся, а перед глазами пусто — опустить взгляд
    на блок у ног (не ниже LOOK_DOWN_MAX — иначе копал бы пол)."""
    yaw_error = aim_errors(me, goal)[0]
    if abs(yaw_error) > COARSE:
        return ("turn_left" if yaw_error > 0 else "turn_right"), _level(me, "head_idle"), "hands_idle"
    center = state.get("center_block")
    if center is not None and center["distance"] <= DIG_CLOSE:
        return "idle", "head_idle", "attack_center"
    if abs(me.get("move_forward", 0.0)) < 0.05 and me["pitch"] > LOOK_DOWN_MAX:
        return "idle", "look_down", "hands_idle"
    if me["pitch"] < -0.15:
        head = "look_up"
    elif abs(yaw_error) > FINE:
        head = "look_left" if yaw_error > 0 else "look_right"
    else:
        head = "head_idle"
    return "walk_forward", head, "hands_idle"


def hunt_peers(loop, session) -> list:
    """Где остальные охотники (их последние состояния) — учителю, чтобы на
    столб за целью лез только ближайший. loop — AILoop (симуляция и игра)."""
    peers = []
    for peer_id, other in loop.sessions.items():
        if peer_id == session.id or not other.hunting or other.prev_state is None:
            continue
        if peer_id == loop.hunt.target_bot:
            continue  # бот-цель — не охотник
        peers.append((peer_id, other.prev_state["self"]))
    return peers


def hunt_actions(state: dict, module, me_id: int = 0, peers: list | None = None) -> dict:
    """Охотник ("Останови меня"): к цели по маршруту и бить, как только удар
    достаёт, — но только в полную силу (взмах раньше сбросил бы заряд).
    Если в этот миг бот падает (после прыжка или уступа) — это крит (x1.5),
    и на этот тик не бежать: на бегу крита нет. Прыгать нарочно ради крита
    не стоит: пока летишь, цель убегает (так учитель не попадал вовсе).
    Цель на столбе рядом — подняться к ней (свой столб) или копать под ней;
    совсем рядом — стоять и поворачиваться к ней (со своего столба не падать)."""
    me = state["self"]
    target = module.observe(state).get("target")
    if target is None:
        return {"legs": "idle", "head": _level(me, "head_idle"), "hands": "hands_idle"}
    legs, head = _steer(aim_errors(me, target)[0])
    hands = "hands_idle"
    goal = state.get("target")  # сама цель, а не точка маршрута
    if goal is not None:
        rise = goal["y"] - me["y"]
        flat = math.hypot(goal["x"] - me["x"], goal["z"] - me["z"])
        if PILLAR_TACTICS and flat < NEAR and rise >= REACH_UP:
            legs, head, hands = _reach_up(state, me, goal, me_id, peers or [])
            if state.get("strike") and _charged(state):
                hands = "attack_center"
            return {"legs": legs, "head": head, "hands": hands}
        route = state.get("route") or {}
        if PILLAR_TACTICS and route and not route.get("complete") and abs(rise) < REACH_UP and flat < BREACH_RANGE:
            legs, head, hands = _breach(state, me, goal)
            if state.get("strike") and _charged(state):
                hands = "attack_center"
            return {"legs": legs, "head": head, "hands": hands}
        # На своём столбе, цель в зоне удара и не ниже — стоять и бить, не
        # шагать вниз (вживую боты достраивались и падали, автор). Только на
        # столбе: на полу стоять рядом с целью нельзя — бегущая цель уходит.
        if _perched(me) and flat <= STAY_REACH and rise > -1.0:
            legs, head = _face(me, goal)
    if state.get("strike") and _charged(state):
        hands = "attack_center"
        falling = not me["on_ground"] and me.get("move_up", 0.0) < 0
        if falling and legs == "sprint_forward":
            legs = "walk_forward"
    return {"legs": legs, "head": _level(me, head), "hands": hands}


def flee_actions(state: dict, config: dict, memory: dict, fight_back: bool = False) -> dict:
    """memory — своё у каждого бота (куда бежал в прошлый раз, мировой yaw):
    держаться выбранного направления, а не дёргаться между двумя равными."""
    me = state["self"]
    res_x, res_y = state["vision"]["resolution"]
    cells = state["vision"]["cells"]
    max_distance = config["vision"]["distance"] * 16
    row = res_y // 2 - 1  # луч чуть выше горизонта — на уровне глаз: стены, столбы, барьер
    threat = state.get("target")
    threat_error = aim_errors(me, threat)[0] if threat is not None else None

    best_error, best_score = 0.0, -math.inf
    for col in range(res_x):
        # Колонка 0 — прямо вперёд, дальше по кругу вправо (js/vision.js).
        error = -col * 2 * math.pi / res_x
        error = (error + math.pi) % (2 * math.pi) - math.pi
        free = cells[(row * res_x + col) * 5 + 3] * max_distance
        score = 1.2 * min(free, 12.0) / 12.0
        if free < WALL:
            score -= 2.0  # в стену — никогда: так учитель загонял себя в угол
        if threat_error is not None:
            score += math.cos(error - threat_error - math.pi)  # 1 — ровно от водящего
        if memory.get("yaw") is not None:
            score += 0.3 * math.cos(me["yaw"] + error - memory["yaw"])
        if score > best_score:
            best_error, best_score = error, score
    memory["yaw"] = me["yaw"] + best_error
    legs, head = _steer(best_error)
    hands = "hands_idle"
    # Охотник вплотную, а удар заряжен — развернуться к нему и ударить:
    # откинет, выиграет время (бот-цель в охоте; автор: "даже не пытается
    # отталкивать"). В салках убегающим бить некого — strike там всегда 0.
    if fight_back and threat is not None and _charged(state):
        if state.get("strike"):
            hands = "attack_center"
        elif math.hypot(threat["x"] - me["x"], threat["z"] - me["z"]) < FIGHT_BACK and threat_error is not None:
            legs, head = ("turn_left" if threat_error > 0 else "turn_right"), "head_idle"
    return {"legs": legs, "head": _level(me, head), "hands": hands}


BRIDGE_PITCH = math.radians(-80)  # мост: взгляд вниз — из-за края луч попадает в бок блока под ногами
BRIDGE_ROW = 2.0      # ближе к цели по z — уже в ряду острова цели (bridge_course.ISLAND): мост "Г" — вбок
BRIDGE_ABOVE = 2.5    # ближе по z и x — над островом цели (он ниже: спрыгнуть)
WALK_MARGIN = 0.65    # мост: шагом за решение проходят до 0.65 блока — столько пола сзади и нужно


def bridge_actions(state: dict) -> dict:
    """Мост (задачка bridge): спиной по оси моста, взгляд вниз на -80°, задом
    к краю, у края — блок перед собой (place_front): луч из-за края попадает
    в бок блока под ногами, и блок встаёт в пропасть. Шаг взгляда 10°:
    -70° бьёт в верх блока, -90° мимо — годится только -80°.

    Спешка (автор: "строятся медленно, шифтуют сразу, а не у края"): задом
    обычным шагом, пока сзади пола больше, чем пройдёт за решение (чувство
    пола state.ground, с поправкой на разгон); у края — крадучись (с края
    крадущегося игра не пускает). После каждого блока сзади ~0.71 пола —
    шаг с места туда влезает, и мост идёт быстрее, чем крадучись целиком.

    Виды моста (bridge_course): остров цели выше — сначала столб (place_below)
    до его уровня; ниже — дойти мостом до края над ним и шагнуть вниз; в
    стороне ("Г") — по z до ряда острова, поворот, вбок. Ось — всегда по
    миру (x или z): наискосок мост не построить. Свесился вбок (угол "Г") —
    сначала шаг вбок обратно на блок: иначе луч мимо блока."""
    me = state["self"]
    goal = state.get("target")
    if goal is None:
        return {"legs": "idle", "head": "head_idle", "hands": "hands_idle"}
    ground = state.get("ground") or [3.0, 3.0, 3.0, 3.0]  # впереди, справа, сзади, слева
    dx, dz = goal["x"] - me["x"], goal["z"] - me["z"]
    rise = goal["y"] - me["y"]

    # Ось моста: по z, пока не в ряду острова цели, потом по x (мост "Г"; у
    # прямых цель в колонне старта — всегда по z). Спиной по оси: взгляд
    # против неё, yaw = atan2(ось.x, ось.z) (конвенция mineflayer).
    axis = (0.0, math.copysign(1.0, dz)) if abs(dz) > BRIDGE_ROW else (math.copysign(1.0, dx), 0.0)
    away = math.atan2(axis[0], axis[1]) - me["yaw"]
    away = (away + math.pi) % (2 * math.pi) - math.pi  # плюс — повернуть влево
    if me["pitch"] > BRIDGE_PITCH + FINE:
        head = "look_down"
    elif me["pitch"] < BRIDGE_PITCH - FINE:
        head = "look_up"
    elif abs(away) > FINE:
        head = "look_left" if away > 0 else "look_right"
    else:
        head = "head_idle"
    turn = "turn_left" if away > 0 else "turn_right"

    # Остров цели выше — столб до его уровня (прыжок — сам place_below). Только
    # пока ниже цели: блок встаёт на третьем тике прыжка, и "ещё в прыжке" —
    # уже над новым блоком (с этим условием учитель строил столб до неба).
    if rise > 0.5:
        return {"legs": turn if abs(away) > COARSE else "idle", "head": head, "hands": "place_below"}
    # Остров цели ниже и он уже под нами — шаг с края вниз.
    if rise < -0.5 and abs(dz) <= BRIDGE_ABOVE and abs(dx) <= BRIDGE_ABOVE:
        return {"legs": "walk_back", "head": head, "hands": "hands_idle"}

    # Пятиться — только ровно спиной по оси: пятясь наискосок, бот уходил
    # вбок на полблока, мост вставал в соседний ряд, а бот — на границу рядов,
    # где луч прицела смотрит в пустоту. Пока доворачивает головой — стоит крадучись.
    if abs(away) > COARSE:
        legs = turn
    elif abs(away) > FINE:
        legs = "sneak"
    elif -0.6 < ground[1] < 0:
        legs = "strafe_right"  # свесился вбок, пол справа — обратно на блок
    elif -0.6 < ground[3] < 0:
        legs = "strafe_left"
    elif (ground[0] < 0 and not state.get("can_place") and abs(me["pitch"] - BRIDGE_PITCH) <= FINE
          and abs(me.get("move_forward", 0.0)) < 0.005):
        # Свесился с края, стоит, взгляд уже -80°, а блок не встанет
        # (state.can_place): замер на 0.284..0.2857 — чуть не дотянул до окна,
        # а крадучись дальше игра не пускает (отодвигает шагами по 0.05). Шаг
        # вперёд — и к краю заново, с другим разгоном.
        legs = "walk_forward"
    elif ground[2] > WALK_MARGIN + max(0.0, -me.get("move_forward", 0.0)):
        legs = "walk_back"
    else:
        legs = "sneak_back"
    ready = abs(away) <= FINE and abs(me["pitch"] - BRIDGE_PITCH) <= FINE
    return {"legs": legs, "head": head, "hands": "place_front" if ready else "hands_idle"}


def bridge_rescue(state: dict) -> dict | None:
    """Подсказка сети на мосту (DAgger): замерла у края — свесилась чуть
    меньше окна, стоит, блок не встаёт, — учитель на этот ход показывает
    выход (шаг вперёд и к краю заново). Свой пример учителя выхода у сети
    почти не было: учитель так застревает редко, а сеть — раз в минуту
    (2026-09-29). Остальное сеть решает сама."""
    actions = bridge_actions(state)
    return actions if actions["legs"] == "walk_forward" else None


BED_REACH = 4.0     # бедварс: кровать ближе стольки (от глаз) — наводиться и копать
LAUNCH_REACH = 1.0  # бедварс: точка неполного маршрута ближе стольки — дошёл до края, начинать мост
EDGE_WALK = 2.9     # ...а меньше стольки (чувство пола видит до 3) — к краю шагом, не бегом
STUCK_DECISIONS = 10  # мост: пятится и не двигается столько решений — упёрся, снова к краю
ENEMY_REACH = 3.5   # враг ближе — бить; дальше и за пустотой — строить мост дальше
NARROW = 1.0        # пола сбоку меньше — узкий мост: не бегать
EDGE_TURN = math.radians(20)  # у края угол до цели больше — сперва довернуть на месте


def _careful_steer(me: dict, target: dict, ground: list) -> tuple[str, str]:
    """_steer, но у края и на узком — шагом, а круто поворачивать — стоя:
    бегом с поворотом на ходу учитель срезал углы и слетал с острова
    (Egg Hunt: тропинки у точки появления, 2026-09-30)."""
    error = aim_errors(me, target)[0]
    legs, head = _steer(error)
    if ground[0] < EDGE_WALK or min(ground[1], ground[3]) < NARROW:
        if abs(error) > EDGE_TURN:
            return "idle", ("look_left" if error > 0 else "look_right")
        if legs == "sprint_forward":
            legs = "walk_forward"
    return legs, head


COVER_GIVE_UP = 40   # защитник: столько решений не вышло закрыть клетку — дальше
COVER_REACH = 4.0    # ...ставить блок, когда опора ближе стольки от глаз
COVER_TOO_CLOSE = 1.3  # ...ближе — сам стоит в клетке или вплотную: отойти
GUARD_RADIUS = 2.5   # закрыл — стоять у кровати не дальше стольки


def _cover_cells(bed: dict) -> list[tuple]:
    """Клетки вокруг кровати, которые закрывает защитник: над обеими
    половинами и по бокам (как первый слой защиты в бедварсе)."""
    halves = [tuple(bed["head"]), tuple(bed["foot"])]
    cells = [(x, y + 1, z) for x, y, z in halves]
    for x, y, z in halves:
        for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            cell = (x + dx, y, z + dz)
            if cell not in halves and cell not in cells:
                cells.append(cell)
    return cells


def bedwars_defend(state: dict, module) -> dict:
    """Защитник (роль от судья): закрыть свою кровать шерстью — сверху и с
    боков (place_front на верх кровати или пола рядом: блок встаёт в клетку
    над гранью), потом стоять у неё; враг рядом — цель от судьи, бой (это уже
    bedwars_actions). Какие клетки закрыты, учитель не видит — помнит, куда
    ставил (module.teacher_cover), а клетку, куда блок не встаёт, считает
    занятой."""
    me = state["self"]
    bed = module.own_bed
    cover = module.teacher_cover
    blocks = state.get("inventory", {}).get("blocks", 0)
    todo = [c for c in _cover_cells(bed) if cover.get(c, 0) is not True and cover.get(c, 0) < COVER_GIVE_UP]
    center = ((bed["head"][0] + bed["foot"][0]) / 2 + 0.5, bed["head"][1], (bed["head"][2] + bed["foot"][2]) / 2 + 0.5)
    if not todo or blocks <= 0:
        if math.hypot(me["x"] - center[0], me["z"] - center[2]) > GUARD_RADIUS:
            legs, head = _steer(aim_errors(me, {"x": center[0], "y": center[1], "z": center[2]})[0])
            return {"legs": "walk_forward" if legs == "sprint_forward" else legs, "head": _level(me, head),
                    "hands": "hands_idle"}
        return {"legs": "idle", "head": _level(me, "head_idle"), "hands": "hands_idle"}
    cell = todo[0]
    cover[cell] = cover.get(cell, 0) + 1
    over_bed = (cell[0], cell[1] - 1, cell[2]) in (tuple(bed["head"]), tuple(bed["foot"]))
    # Опора — верх кровати (0.5625) или верх пола под клеткой.
    point = {"x": cell[0] + 0.5, "y": cell[1] - 1 + (0.5625 if over_bed else 1.0) - 0.05, "z": cell[2] + 0.5}
    eye = (me["x"], me["y"] + 1.62, me["z"])
    distance = math.dist(eye, (point["x"], point["y"], point["z"]))
    yaw_error, pitch_error = aim_errors(me, point)
    if math.hypot(me["x"] - point["x"], me["z"] - point["z"]) < COVER_TOO_CLOSE:
        return {"legs": "walk_back", "head": _aim_head(yaw_error, pitch_error), "hands": "hands_idle"}
    if distance > COVER_REACH:
        legs, head = _steer(yaw_error)
        return {"legs": "walk_forward" if legs == "sprint_forward" else legs, "head": _level(me, head),
                "hands": "hands_idle"}
    legs = ("turn_left" if yaw_error > 0 else "turn_right") if abs(yaw_error) > COARSE else "idle"
    aimed = abs(yaw_error) < 0.12 and abs(pitch_error) < 0.12
    if aimed and state.get("can_place"):
        cover[cell] = True
        return {"legs": legs, "head": "head_idle", "hands": "place_front"}
    if aimed:
        cover[cell] = cover[cell] + 5  # прицелился, а блок не встаёт — клетка, видно, занята
    return {"legs": legs, "head": _aim_head(yaw_error, pitch_error), "hands": "hands_idle"}


ATTACK_BLOCKS = 48   # бедварс: столько шерсти набрать дома, прежде чем идти мостом к врагу
DEFEND_BLOCKS = 16   # ...защитнику — на укрытие кровати
LOW_BLOCKS = 4       # меньше — домой за шерстью
WOOL_PRICE = 4       # железа за шерсть (modules.bedwars.wool_price)


def bedwars_shopping(state: dict, module) -> dict | None:
    """Экономика (py/bedwars_game.py): у своей точки появления — генератор
    (судья отдаёт кучу железа, кто на нём стоит) и магазин (действие рук buy:
    железо -> шерсть). Дома — копить и покупать, пока шерсти меньше нужного
    (атакующему — на мост, защитнику — на укрытие); вдали без шерсти —
    домой. None — дела с магазином нет."""
    me = state["self"]
    blocks = state.get("inventory", {}).get("blocks", 0)
    iron = (state.get("inventory_items") or {}).get("iron_ingot", 0)
    need = DEFEND_BLOCKS if getattr(module, "role", None) == "defend" else ATTACK_BLOCKS
    if state.get("bedwars_shop"):
        if blocks >= need:
            return None
        hands = "buy" if iron >= WOOL_PRICE else "hands_idle"
        return {"legs": "idle", "head": _level(me, "head_idle"), "hands": hands}
    spawn = getattr(module, "own_spawn", None)
    if blocks >= LOW_BLOCKS or spawn is None:
        return None
    module.teacher_bridging = False  # домой — мост потом продолжить с его конца
    home = {"x": spawn[0] + 0.5, "y": spawn[1], "z": spawn[2] + 0.5}
    legs, head = _steer(aim_errors(me, home)[0])
    ground = state.get("ground") or [3.0, 3.0, 3.0, 3.0]
    if legs == "sprint_forward" and min(ground[1], ground[3]) < NARROW:
        legs = "walk_forward"  # по узкому мосту — шагом
    return {"legs": legs, "head": _level(me, head), "hands": "hands_idle"}


def bedwars_actions(state: dict, module) -> dict:
    """Бедварс (py/bedwars_game.py): цель от судьи — враг рядом или чужая
    кровать. Враг — к нему и бить, как только удар достаёт и заряжен. Кровать
    — если до неё можно дойти (маршрут полный), идти по маршруту; рядом —
    навести взгляд и копать (кровать ломается рукой за полсекунды); дойти
    нельзя (между островами пустота) — по маршруту к ближайшей к ней точке,
    оттуда мост, как учитель моста (ось по миру, при нужде сначала столб).
    Враг в зоне удара по пути — бить."""
    me = state["self"]
    goal = state.get("target")
    if goal is None:
        return {"legs": "idle", "head": _level(me, "head_idle"), "hands": "hands_idle"}
    strike = bool(state.get("strike")) and _charged(state, full=True)
    route = state.get("route") or {}
    shopping = None if strike else bedwars_shopping(state, module)
    if shopping is not None:
        return shopping
    if getattr(module, "role", None) == "defend" and module.own_bed and not (goal.get("h") or 0) > 0:
        actions = bedwars_defend(state, module)
        if strike:
            actions["hands"] = "attack_center"
        return actions
    ground = state.get("ground") or [3.0, 3.0, 3.0, 3.0]
    if (goal.get("h") or 0) > 0:  # враг (у кровати-точки роста нет)
        flat = math.hypot(goal["x"] - me["x"], goal["z"] - me["z"])
        bed = getattr(module, "enemy_bed", None)
        if not route.get("complete") and flat > ENEMY_REACH and bed is not None:
            # До врага не дойти (он за пустотой), а бить далеко — не бросать мост
            # и не бежать к нему в пропасть (автор: "мостостроители перестают
            # строиться и ссыкуют рядом с врагами, падают"): строить дальше к кровати.
            head_cell = bed["head"]
            state = dict(state, target={"x": head_cell[0] + 0.5, "y": head_cell[1] + 0.5,
                                        "z": head_cell[2] + 0.5, "h": 0.0})
            goal = state["target"]
        else:
            # Узкий мост, край: к врагу шагом, бегом слетал.
            legs, head = _careful_steer(me, module.observe(state)["target"], ground)
            return {"legs": legs, "head": _level(me, head), "hands": "attack_center" if strike else "hands_idle"}
    eye = (me["x"], me["y"] + 1.62, me["z"])
    if math.dist(eye, (goal["x"], goal["y"], goal["z"])) <= BED_REACH:
        yaw_error, pitch_error = aim_errors(me, goal)
        legs = ("turn_left" if yaw_error > 0 else "turn_right") if abs(yaw_error) > COARSE else "idle"
        center = state.get("center_block")
        # Кровать закрыта шерстью (защитник соперника) — прицел упирается в
        # шерсть: копать её (ломается только поставленное игроками, а шерсть
        # на карте ставят только они). Автор: "блоки не особо хотят копать".
        aimed = (center is not None and center["distance"] <= 4.5
                 and (center["name"].endswith("_bed") or center["name"].endswith("_wool")))
        return {"legs": legs, "head": _aim_head(yaw_error, pitch_error),
                "hands": "attack_center" if aimed or strike else "hands_idle"}
    waypoint = route.get("waypoint")
    at_route_end = waypoint is not None and math.hypot(waypoint["x"] - me["x"], waypoint["z"] - me["z"]) < LAUNCH_REACH
    if route.get("complete"):
        module.teacher_bridging = False
    elif not getattr(module, "teacher_bridging", False) and route and not at_route_end:
        # Дойти нельзя: маршрут ведёт к ближайшей к цели достижимой точке (край
        # своего острова или конец моста) — сперва туда, шагом (бегом
        # проскакивал в пустоту по инерции). Дошёл (точка маршрута под ногами)
        # — мост, до полного маршрута. Раньше мост начинался, когда "впереди
        # мало пола", — у сложных островов это бывало не у края (упор в стену).
        legs, head = _careful_steer(me, module.observe(state)["target"], ground)
        return {"legs": legs, "head": _level(me, head), "hands": "attack_center" if strike else "hands_idle"}
    if route.get("complete"):
        legs, head = _careful_steer(me, module.observe(state)["target"], ground)
        return {"legs": legs, "head": _level(me, head), "hands": "attack_center" if strike else "hands_idle"}
    module.teacher_bridging = True
    actions = bridge_actions(state)
    # Мост начат не у края (пол впереди кончился у препятствия, а не у пустоты)
    # и пятится в стену: стоит на месте — снова к краю по маршруту.
    moving = abs(me.get("move_forward", 0.0)) + abs(me.get("move_right", 0.0)) > 0.01
    if actions["legs"] in ("walk_back", "sneak_back") and not moving and (state.get("ground") or [0.0])[0] > 0:
        module.teacher_stuck = getattr(module, "teacher_stuck", 0) + 1
        if module.teacher_stuck >= STUCK_DECISIONS:
            module.teacher_bridging = False
            module.teacher_stuck = 0
    else:
        module.teacher_stuck = 0
    if strike:
        actions["hands"] = "attack_center"
    return actions


def teacher_actions(session, state: dict, config: dict, loop=None) -> dict | None:
    """Действия учителя для бота этой сессии (или None — пусть решает сеть).
    loop — AILoop: охотнику нужно знать, где остальные охотники."""
    if session.task_name == "chase":
        return chase_actions(state, session.module)
    if session.task_name == "bridge":
        return bridge_actions(state)
    if session.task_name == "bedwars":
        return bedwars_actions(state, session.module)
    if session.task_name == "hunt":
        return hunt_actions(state, session.module, session.id, hunt_peers(loop, session) if loop else [])
    if session.task_name == "flee":
        if not hasattr(session, "teacher_memory"):
            session.teacher_memory = {}
        # Отбиваться — только боту-цели в охоте: в салках убегающему бить некого,
        # и разворот к водящему там был бы просто потерей форы.
        return flee_actions(state, config, session.teacher_memory, fight_back=session.hunting)
    return None
