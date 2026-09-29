"""Обучение ходьбе: ядро само расставляет случайные точки в мире и
награждает бота за приближение к ним/достижение — та же логика, что раньше
жила прямо в py/reward.py (плюс цель из чата), просто вынесенная в модуль,
плюс отдельное поощрение за реакцию на препятствие по курсу (см. ниже).

Упрощение (сознательное, без pathfinding'а — его тут никто не подключал):
точки выбираются случайно по кругу вокруг текущей позиции бота на ЕГО ЖЕ
высоте Y, без проверки проходимости. Если бот не приближается к точке
дольше stuck_ticks_limit тиков подряд, она считается недостижимой
(застряли за стеной/в яме) и перевыбирается — так бот не зависает навечно
на точке, до которой физически не дойти.

Обход препятствий не реализован через pathfinding — вместо этого в награду
добавлена отдельная компонента: если по курсу (центр сетки зрения — грубое
приближение "куда иду", раз отдельного макроса взгляда не по курсу
движения тут нет) обнаружено что-то близко, turn/jump поощряются, а
продолжение ломиться вперёд — штрафуется. Сеть свободна в выборе КАК
обходить (повернуть, перепрыгнуть, обойти сбоку через несколько тиков) —
шейпинг лишь подсказывает направление обучения, не диктует действие.

Бег (sprint_forward) отдельно не поощряется — reach_goal один и тот же
независимо от числа тиков, но при gamma < 1 (см. config.json -> train.gamma)
более быстрый маршрут даёт более высокий дисконтированный возврат сам по
себе, так что явный стимул не нужен.

Два предохранителя от "тупых" смертей (падение в пещеру, пересечение
барьера мира — Minecraft не блокирует его физически, а просто наносит
урон по нарастающей):
  - health_loss_penalty — штраф сразу при потере здоровья, а не только
    отложенный -100 за саму смерть много тиков спустя;
  - границы — цели-точки никогда не выбираются у края мира. Источник
    границы: настоящий world border из пакетов сервера (Node кладёт его в
    state["world_border"], js/border.js), а если сервер его не прислал —
    ручной прямоугольник config.json -> modules.walking.bounds.

Каналы (py/protocol.py): основа награды общая для всех ("дошли до точки —
молодцы все"), плюс точечно:
  - ноги — реакция на препятствие по курсу, штраф за топтание на месте;
  - голова — штраф за задранный/опущенный взгляд и за любое верчение без
    дела (head_move_penalty): это и есть лекарство от "бесцельного
    верчения головой" — ходьбе голова не нужна, пусть учится не мешать;
  - руки — лёгкий штраф за attack_center (ходьбе махать руками незачем).
"""

from __future__ import annotations

import math
import random

from .base import TrainingModule
from .targets import times

# Экшены, которыми обходят препятствие, а не ломятся в него.
_OBSTACLE_AVOID_ACTIONS = ("turn_left", "turn_right", "jump_forward")
_OBSTACLE_BUMP_ACTIONS = ("walk_forward", "sprint_forward")


class WalkingModule(TrainingModule):
    name = "walking"

    # Голове — только выровнять наклон (крутить по сторонам незачем, курс
    # держат ноги), руки выключены.
    allowed_actions = {
        "head": ["head_idle", "look_up", "look_down"],
        "hands": ["hands_idle"],
    }

    def __init__(self, config: dict):
        super().__init__(config)
        cfg = config["modules"]["walking"]
        self.min_radius = cfg["min_radius"]
        self.max_radius = cfg["max_radius"]
        self.stuck_ticks_limit = cfg["stuck_ticks_limit"]
        self.obstacle_close_distance = cfg["obstacle_close_distance"]
        self.obstacle_avoid_bonus = cfg["obstacle_avoid_bonus"]
        self.obstacle_bump_penalty = cfg["obstacle_bump_penalty"]

        reward_cfg = config["reward"]
        self.reach_goal = reward_cfg["reach_goal"]
        self.closer = reward_cfg["closer"]
        self.idle = reward_cfg["idle"]
        self.death = reward_cfg["death"]
        self.goal_radius = reward_cfg["goal_radius"]

        # Штраф за задранную/опущенную голову: у walking нет причин смотреть
        # вверх или в пол, а залипший pitch (после look_up/look_down) странно
        # выглядит и путает attack_center при будущем переключении модуля.
        # Линейно по |pitch|, максимум при взгляде вертикально вверх/вниз.
        self.pitch_penalty = cfg.get("pitch_penalty", 0.5)

        # Мгновенный штраф за потерянное здоровье (падение в пещеру, урон
        # от барьера мира, моб, лава — что угодно). Без него единственный
        # сигнал "сюда не ходи" — это -100 за смерть спустя МНОГО тиков
        # после самого падения, и он слишком отложен и редок, чтобы бот
        # быстро связал "иду туда" с "было больно".
        self.health_loss_penalty = cfg.get("health_loss_penalty", 2.0)

        # Держит выбираемые точки внутри барьера мира — иначе walking
        # спокойно выбирает точку ЗА барьером и тупо идёт к ней. Главный
        # источник — state["world_border"] (настоящая граница от сервера),
        # а этот ручной прямоугольник — запасной вариант, если сервер
        # границу не прислал. None — проверка выключена.
        self.bounds = cfg.get("bounds")
        # Насколько отступать от настоящей границы мира при выборе точки:
        # охранник в js/bot.js начинает разворачивать бота уже за
        # world_border.margin от края, точка у самого края была бы
        # недостижима.
        self.border_target_margin = cfg.get("border_target_margin", 4.0)

        self.head_move_penalty = cfg.get("head_move_penalty", 0.05)
        self.hands_penalty = cfg.get("hands_penalty", 0.1)
        self.time_penalty = cfg.get("time_penalty", 0.05)

        # Награда за разворот к цели: разница cos(угла на цель) между тиками
        # (potential-based shaping — не меняет, какое поведение оптимально,
        # только подсказывает). Без неё поворот к цели сам по себе не давал
        # ничего, и связь "повернись -> потом иди" искалась вслепую.
        self.heading_reward = cfg.get("heading_reward", 0.5)
        # Пока бот стоит рядом с целью (например, с игроком через
        # !setTarget player) — небольшой плюс за каждый тик. Большой
        # reach_goal даётся один раз, при входе в радиус.
        self.near_goal_reward = cfg.get("near_goal_reward", 1.0)

        # Используется только в compute_reward (награда за прогресс):
        # дистанция, cos угла на цель и сама цель прошлого тика.
        self._prev_distance: float | None = None
        self._prev_heading: float | None = None
        self._prev_target: dict | None = None
        self._inside_goal = False
        self._heading_shaping = 0.0  # награда ногам за разворот к цели на этом тике
        self._progress_source = None  # чем мерили прогресс прошлый раз: "route" / "direct"
        # Используется только в on_tick (детектор "застрял") — намеренно
        # отдельная переменная от _prev_distance: on_tick и compute_reward
        # вызываются в разном порядке относительно друг друга внутри тика,
        # общая переменная между ними создала бы путаницу "кто её обновил
        # последним".
        self._stuck_reference_distance: float | None = None
        self._stuck_ticks = 0

    def reset(self, state: dict) -> None:
        self._prev_distance = None
        self._prev_heading = None
        self._prev_target = None
        self._inside_goal = False
        self._stuck_reference_distance = None
        self._stuck_ticks = 0
        self.current_target = None  # следующий on_tick выберет новую точку

    def on_tick(self, state: dict) -> None:
        if state.get("target_is_human"):
            return  # человек держит цель — не мешаем, current_target просто не используется Node

        if self.current_target is None:
            self._pick_new_target(state)
            return

        # Граница мира могла прийти от сервера позже, чем выбрали точку
        # (или сжаться командой /worldborder) — точка снаружи недостижима.
        bounds = self._effective_bounds(state)
        if bounds is not None and not _inside(self.current_target, bounds):
            self._pick_new_target(state)
            return

        distance = _distance(self.current_target, state["self"])

        if distance <= self.goal_radius:
            # Счётчик goals — в compute_reward (там же, где награда), чтобы
            # считались и цели человека через !setTarget.
            self._pick_new_target(state)
            return

        if self._stuck_reference_distance is None:
            self._stuck_reference_distance = distance
        elif distance < self._stuck_reference_distance - 0.05:
            # Реально приблизились с момента, когда начали считать "застрял" —
            # сбрасываем счётчик и точку отсчёта.
            self._stuck_reference_distance = distance
            self._stuck_ticks = 0
        else:
            self._stuck_ticks += 1

        if self._stuck_ticks >= self.stuck_ticks_limit:
            self.count("stuck")
            self._pick_new_target(state)  # похоже, точка недостижима

    def _pick_new_target(self, state: dict) -> None:
        self_pos = state["self"]
        bounds = self._effective_bounds(state)

        # Несколько попыток найти случайную точку прямо внутри границ: если
        # просто обрезать (clamp) точку по краю, то у края все цели
        # "налипают" на одну линию вдоль границы. Если за 10 попыток не
        # вышло (бот зажат в углу) — тогда уже обрезаем.
        for _ in range(10):
            angle = random.uniform(0, 2 * math.pi)
            radius = random.uniform(self.min_radius, self.max_radius)
            x = self_pos["x"] + math.sin(angle) * radius
            z = self_pos["z"] + math.cos(angle) * radius
            if bounds is None or _inside({"x": x, "z": z}, bounds):
                break
        else:
            x = min(max(x, bounds["min_x"]), bounds["max_x"])
            z = min(max(z, bounds["min_z"]), bounds["max_z"])

        self.current_target = {"x": x, "y": self_pos["y"], "z": z}
        self._prev_distance = None
        self._stuck_reference_distance = None
        self._stuck_ticks = 0

    def compute_reward(self, prev_state: dict, curr_state: dict, actions: dict) -> dict:
        """Каждому каналу — за то, на что он реально влияет (как в looking).
        Ноги — всё про движение: прогресс к цели, разворот к ней,
        препятствия, здоровье, смерть. Голова в walking/follow только
        наклоняется, на движение никак не влияет — ей лишь "держи взгляд
        ровно и не дёргайся"; раньше ей же шла и общая награда за ходьбу, а
        для неё это чистый шум (награда за то, чего она не делала)."""
        if curr_state.get("dead"):
            self._forget_target()
            self.count("deaths")
            return {"legs": self.death, "head": 0.0, "hands": 0.0}

        self._heading_shaping = 0.0
        legs = self._progress_reward(prev_state, curr_state, actions["legs"]) + self._heading_shaping

        # Голова: держать взгляд по курсу. Штраф растёт с отклонением pitch
        # от горизонта — без него бот после пары look_up может "залипнуть"
        # взглядом в небо (ходьбе это не мешает — зрение круговое, но
        # портит прицеливание в других задачах). Плюс копеечный штраф за
        # любое движение головой: ходьбе оно не нужно, и сеть головы
        # учится не вертеться без дела.
        head = -self.pitch_penalty * abs(curr_state["self"]["pitch"]) / (math.pi / 2)
        if actions["head"] != "head_idle":
            head -= self.head_move_penalty

        hands = -self.hands_penalty if actions["hands"] != "hands_idle" else 0.0
        return {"legs": legs, "head": head, "hands": hands}

    def observe(self, state: dict) -> dict:
        """Сети — ближайшая точка маршрута вместо самой цели (если маршрут
        есть): идти к ней — то же, что сеть уже умеет ("к цели"), а маршрут
        сам обходит обрывы, стены и воду (js/route.js). Рядом с целью точка
        маршрута и есть цель."""
        waypoint = (state.get("route") or {}).get("waypoint")
        if not waypoint or state.get("target") is None:
            return state
        observed = dict(state)
        observed["goal"] = state["target"]  # сама цель — сети отдельным входом (state_encoder: goal)
        observed["target"] = {**waypoint, "h": 0.0}
        return observed

    def _progress_reward(self, prev_state: dict, curr_state: dict, legs_action: str) -> float:
        """Часть ног: препятствие, здоровье, прогресс к цели."""
        reward = self._obstacle_reward(curr_state, legs_action)
        reward += self._health_loss_penalty(prev_state, curr_state)

        target = curr_state.get("target")
        if target is None:
            self._forget_target()
            # Цели нет — только мягкий штраф за бездействие, чтобы бот
            # не выучил "ничего не делать безопасно".
            return reward + (self.idle if legs_action == "idle" else 0.0)

        self_state = curr_state["self"]
        distance = _distance(target, self_state)
        # Разворот — к точке маршрута (к ней сеть и идёт), а не к цели напрямую.
        heading = _heading_cos(self_state, self.observe(curr_state)["target"])

        # Цель сменилась (модуль выбрал новую точку) — сравнивать дистанцию
        # со старой целью бессмысленно, иначе бот получил бы огромный
        # "прогресс" или "откат" за один тик. Движущийся игрок за тик
        # смещается меньше чем на пару блоков, так что его не путаем.
        if self._prev_target is not None and _distance(target, self._prev_target) > 3.0:
            self._forget_target()

        if distance <= self.goal_radius:
            if not self._inside_goal:
                self._inside_goal = True
                self.count("goals")
                reward += self.reach_goal
            else:
                reward += self.near_goal_reward
            self._remember_target(target, distance, heading)
            return reward
        if distance > self.goal_radius + 1.0:
            # Небольшой гистерезис: шаг туда-обратно на краю радиуса не
            # должен давать +reach_goal каждые два тика.
            self._inside_goal = False

        # Прогресс — по маршруту, а не по прямой: обход стены или оврага
        # сначала уводит от цели, и награда "за блок ближе по прямой" за это
        # наказывала. Если маршрута нет — по прямой, как раньше.
        progress_distance, source = self._progress_distance(curr_state, distance)
        if source != self._progress_source:
            self._prev_distance = None  # сменился способ мерить — не сравниваем
        self._progress_source = source

        # Давление времени: каждый тик вдали от цели — немного в минус.
        # Быстрее дошёл — меньше потерял; так окупается бег.
        reward -= self.time_penalty

        if self._prev_distance is not None:
            # Плотная награда за прогресс: closer за каждый блок
            # приближения (отход — такой же минус). Раньше плюс давали,
            # только если за тик бот сдвинулся больше чем на 0.5 блока в
            # сторону цели — а ходьба даёт ~0.65 блока за тик, и при
            # подходе под углом награды не было вовсе.
            delta = max(-1.0, min(1.0, self._prev_distance - progress_distance))
            reward += self.closer * delta
            self._heading_shaping = self.heading_reward * (heading - self._prev_heading)

            moved = abs(self_state["x"] - prev_state["self"]["x"]) >= 0.05 or \
                abs(self_state["z"] - prev_state["self"]["z"]) >= 0.05
            turned = abs(self_state.get("dyaw", 0.0)) > 0.01
            if not moved and not turned and legs_action not in _OBSTACLE_AVOID_ACTIONS                     and not self._standing_is_work(curr_state):
                # "Топчется на месте": не идёт и даже не поворачивается.
                # Поворот и прыжок у стены сюда не попадают — это правильная
                # реакция, а не безделье.
                reward += self.idle

        self._remember_target(target, progress_distance, heading)
        return reward

    def _progress_distance(self, state: dict, direct: float) -> tuple[float, str]:
        """Чем мерить прогресс к цели (наследники меняют: охота)."""
        return _route_distance(state, direct)

    def _standing_is_work(self, state: dict) -> bool:
        """Стоять на месте сейчас — не безделье (наследники: охота, когда
        пешком до цели не дойти — копать и строить надо стоя)."""
        return False

    def _remember_target(self, target: dict, distance: float, heading: float) -> None:
        self._prev_target = target
        self._prev_distance = distance
        self._prev_heading = heading

    def _forget_target(self) -> None:
        self._prev_target = None
        self._prev_distance = None
        self._prev_heading = None
        self._inside_goal = False

    @classmethod
    def summarize(cls, events: dict, ticks: int, bot_minutes: float) -> tuple[str, float | None]:
        goals = events.get("goals", 0)
        rate = goals / bot_minutes * 10 if bot_minutes > 0 else 0.0
        text = f"дошёл до точки {times(goals)} ({rate:.1f} за 10 мин на бота)"
        if events.get("stuck"):
            text += f", застревал {times(events['stuck'])}"
        return text, rate

    def _effective_bounds(self, state: dict) -> dict | None:
        """Прямоугольник, внутри которого можно выбирать точки: настоящая
        граница мира из state (минус отступ), иначе ручной bounds из
        конфига, иначе None (не ограничиваем). World border в Minecraft —
        квадрат, поэтому это тоже min/max по x и z."""
        border = state.get("world_border")
        if border and border.get("size"):
            half = border["size"] / 2 - self.border_target_margin
            if half > 0:
                return {
                    "min_x": border["center_x"] - half, "max_x": border["center_x"] + half,
                    "min_z": border["center_z"] - half, "max_z": border["center_z"] + half,
                }
        return self.bounds

    def _level_gaze_penalty(self, curr_state: dict) -> float:
        """Голове — штраф за задранный/опущенный взгляд (растёт с наклоном).
        В задачках с общей наградой на все каналы (салки, охота) без него
        голове было всё равно, куда смотреть: от случайных поворотов взгляд
        "уплывал" и застревал в небе (заметил автор, 2026-09-26)."""
        return -self.pitch_penalty * abs(curr_state["self"]["pitch"]) / (math.pi / 2)

    def _obstacle_reward(self, curr_state: dict, action_name: str) -> float:
        """Отдельная от прогресса-к-цели компонента: реагирует ли бот на
        препятствие ПРЯМО СЕЙЧАС, а не только задним числом через "дистанция
        не изменилась" (та проверка выше срабатывает и от банальной паузы
        между решениями, и слишком поздно — уже после столкновения)."""
        obstacle_distance = _forward_obstacle_distance(curr_state, self.config)
        if obstacle_distance is None or obstacle_distance >= self.obstacle_close_distance:
            return 0.0

        if action_name in _OBSTACLE_AVOID_ACTIONS:
            return self.obstacle_avoid_bonus
        if action_name in _OBSTACLE_BUMP_ACTIONS:
            return self.obstacle_bump_penalty
        return 0.0

    def _health_loss_penalty(self, prev_state: dict, curr_state: dict) -> float:
        """Отрицательная награда, пропорциональная реально потерянному
        здоровью за этот тик — падение, барьер мира, моб, лава и т.п.
        Срабатывает СРАЗУ, не дожидаясь итоговой смерти."""
        prev_health = prev_state.get("self", {}).get("health")
        curr_health = curr_state["self"]["health"]
        if prev_health is None or curr_health >= prev_health:
            return 0.0
        return -self.health_loss_penalty * (prev_health - curr_health)


class StillnessWatch:
    """Не застоялся ли бот: за seconds секунд (в тиках по tick_seconds) не
    отошёл дальше radius блоков от места, где начали отсчёт. update()
    возвращает True один раз — в тот тик, когда бот застоялся; дальше
    отсчёт начинается заново."""

    def __init__(self, seconds: float, radius: float, tick_seconds: float):
        self.limit_ticks = max(1, round(seconds / tick_seconds))
        self.radius = radius
        self.anchor = None
        self.ticks = 0

    def reset(self) -> None:
        self.anchor = None
        self.ticks = 0

    def update(self, self_state: dict) -> bool:
        position = (self_state["x"], self_state["y"], self_state["z"])
        if self.anchor is None or math.dist(position, self.anchor) > self.radius:
            self.anchor = position
            self.ticks = 0
            return False
        self.ticks += 1
        if self.ticks < self.limit_ticks:
            return False
        self.anchor = position
        self.ticks = 0
        return True


def _inside(point: dict, bounds: dict) -> bool:
    return bounds["min_x"] <= point["x"] <= bounds["max_x"] and bounds["min_z"] <= point["z"] <= bounds["max_z"]


def _route_distance(state: dict, direct: float) -> tuple[float, str]:
    """Сколько ещё идти: по маршруту (js/route.js), если он есть, иначе по
    прямой. Второе значение — чем мерили (менять меру посреди пути нельзя:
    разница между ними — не прогресс)."""
    route = state.get("route")
    if route and route.get("length") is not None:
        return route["length"], "route"
    return direct, "direct"


def _heading_cos(self_state: dict, target: dict) -> float:
    """cos угла между направлением взгляда (по горизонтали) и направлением
    на цель: 1 — цель прямо впереди, -1 — прямо сзади. Та же геометрия,
    что в state_encoder: вперёд f = (-sin yaw, -cos yaw) (mineflayer)."""
    dx = target["x"] - self_state["x"]
    dz = target["z"] - self_state["z"]
    length = math.hypot(dx, dz)
    if length < 1e-6:
        return 1.0
    return (dx * -math.sin(self_state["yaw"]) + dz * -math.cos(self_state["yaw"])) / length


def _distance(a: dict, b: dict) -> float:
    return math.dist((a["x"], a["y"], a["z"]), (b["x"], b["y"], b["z"]))


def _forward_obstacle_distance(curr_state: dict, config: dict) -> float | None:
    """Дистанция (в блоках) до препятствия прямо по курсу — берётся из
    центральной ячейки уже посчитанной на Node сетки зрения (центр = там,
    куда сейчас смотрит бот; отдельного понятия "курс движения" в этой
    системе нет, взгляд — лучшее доступное приближение). None, если сетка
    недоступна/рассинхронизирована или в этом направлении пусто в пределах
    дальности обзора."""
    vision = curr_state.get("vision")
    if not vision:
        return None

    cells = vision.get("cells")
    resolution = vision.get("resolution")
    if not cells or not resolution:
        return None

    res_x, res_y = resolution
    if res_x < 1 or res_y < 1:
        return None

    # Центр сетки зрения = направление взгляда. В camera-режиме это середина
    # сетки (col=resX//2), в circular — колонка 0 (она и есть "прямо вперёд").
    circular = config["vision"].get("mode") == "circular"
    col = 0 if circular else res_x // 2
    row = res_y // 2
    stride = 5  # r, g, b, d, класс блока — 5 значений на ячейку (js/state.js)
    idx = (row * res_x + col) * stride
    if idx + 3 >= len(cells):
        return None

    distance_norm = cells[idx + 3]
    if distance_norm >= 1.0:
        return None  # ничего не видно в пределах дальности

    max_view_distance = config["vision"]["distance"] * 16
    return distance_norm * max_view_distance
