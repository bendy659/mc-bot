"""Задачка "следить взглядом": держать цель в прицеле и поворачиваться за
ней, когда она двигается.

Цель:
  - если человек поставил её через !setTarget (игрок или точка) — она;
  - иначе модуль сам "цепляется" к ближайшей сущности (игрок, моб — кроме
    выпавших предметов) и держит её, пока она не пропадёт из виду. Так
    задачке не нужен человек рядом: боты роя тренируются следить друг за
    другом и за мобами.

Награда:
  - за доворот к цели — track_reward за каждый радиан, на который
    приблизился взгляд (и такой же минус за отворот). КАЖДЫЙ канал получает
    её только за СВОЙ вклад: насколько приблизился бы взгляд, если бы
    повернул только он (difference rewards). Ноги крутят корпус на 30°,
    голова — на 10°, и с общей наградой правильный доворот головы тонул в
    случайных поворотах ног (голову штрафовали за то, что сделали ноги) —
    вживую мозг за 10 минут выучил правило лишь наполовину;
  - цель в поле зрения камеры (fov) — небольшой плюс, ровно в прицеле
    (ошибка меньше center_angle) — большой плюс за каждый тик: это общий
    результат, он командный;
  - цель загорожена блоком (по сетке зрения) — плюсов за видимость нет.
Ногам — штраф за ходьбу (legs_move_penalty): задачка про взгляд, корпусом
можно только поворачиваться. Рукам — за удар (незачем).
"""

from __future__ import annotations

import math

from protocol import ACTION_ROTATION

from .base import TrainingModule
from .targets import ITEM, aim_errors, find_entity, nearest, percent, present_entities, wrap_angle

_WALKING_ACTIONS = ("walk_forward", "sprint_forward", "walk_back", "strafe_left", "strafe_right", "jump_forward")


class LookingModule(TrainingModule):
    name = "looking"

    # Задачка про взгляд: ноги только поворачивают корпус на месте, руки
    # выключены, голове — всё.
    allowed_actions = {
        "legs": ["idle", "turn_left", "turn_right"],
        "hands": ["hands_idle"],
    }

    def __init__(self, config: dict):
        super().__init__(config)
        cfg = config["modules"]["looking"]
        self.center_angle = math.radians(cfg.get("center_angle_deg", 8.0))
        self.in_view_reward = cfg["in_view_reward"]
        self.center_reward = cfg["center_reward"]
        self.track_reward = cfg.get("track_reward", 2.0)
        self.idle_penalty = cfg["idle_penalty"]
        self.occlusion_margin = cfg["occlusion_margin"]
        self.legs_move_penalty = cfg.get("legs_move_penalty", 0.1)

        # Смерть в looking по умолчанию НЕ штрафуется: ноги тут умеют только
        # поворачиваться на месте, убежать бот не может — а учить сеть бояться
        # того, на что она никак не влияет (вживую: 21 смерть от стрел
        # разбойников за 15 минут), значит только шуметь в её обучении.
        # Эпизод всё равно обрывается (done), Q не бутстрапится через смерть.
        self.death_penalty = cfg.get("death_penalty", 0.0)
        # Горизонт у looking короткий (gamma 0.9, modules.looking.gamma — его
        # читает ai_loop/demo_loader): задача "навестись" решается за секунду,
        # а с gamma 0.99 ценность "быть в прицеле" (~2 за тик на ~100 тиков
        # вперёд) достигала ~200, и разница между правильным и неправильным
        # доворотом (±0.35) тонула в шуме оценки.
        self.res_x, self.res_y = config["vision"]["resolution"]
        self.max_view_distance = config["vision"]["distance"] * 16
        self.circular = config["vision"].get("mode") == "circular"
        # "В поле зрения" — это поле зрения камеры (куда смотрит голова), а
        # не круговой радар: задачка именно про то, чтобы смотреть на цель.
        self.half_fov = math.radians(config["fov"]) / 2
        self.vertical_span = math.radians(
            config["vision"]["vertical_fov"] if self.circular else config["fov"]
        )

        self._locked_id = None

    def reset(self, state: dict) -> None:
        self._locked_id = None
        self.current_target = None

    def on_tick(self, state: dict) -> None:
        if state.get("target_is_human"):
            return  # человек держит цель — следим за ней

        # Держим уже выбранную сущность, пока она видна; иначе берём
        # ближайшую новую (кроме выпавших предметов — за ними следить
        # бессмысленно).
        if self._locked_id is None or find_entity(state, self._locked_id) is None:
            candidates = [e for e in present_entities(state) if e.get("type") != ITEM]
            chosen = nearest(candidates)
            self._locked_id = chosen["id"] if chosen else None

        self.current_target = {"entity_id": self._locked_id} if self._locked_id is not None else None

    def compute_reward(self, prev_state: dict, curr_state: dict, actions: dict) -> dict:
        if curr_state.get("dead"):
            self.count("deaths")
            return self.team(self.death_penalty)

        rewards = self.team(self._view_reward(curr_state, actions))
        for channel, value in self._tracking_rewards(prev_state, curr_state, actions).items():
            rewards[channel] += value
        if actions["legs"] in _WALKING_ACTIONS:
            rewards["legs"] -= self.legs_move_penalty
        if actions["hands"] != "hands_idle":
            rewards["hands"] -= 0.1
        return rewards

    def _tracking_rewards(self, prev_state: dict, curr_state: dict, actions: dict) -> dict:
        """Награда за доворот — каждому каналу за его собственный поворот:
        ошибка взгляда до хода минус ошибка, если бы повернул только этот
        канал (цель — в её нынешнем положении, чтобы её собственное
        движение не приписывалось боту)."""
        target = curr_state.get("target")
        if target is None or prev_state is None:
            return {}
        before = prev_state["self"]
        error_before = math.hypot(*aim_errors(before, target))
        rewards = {}
        for channel in ("legs", "head"):
            dyaw, dpitch = ACTION_ROTATION.get(actions[channel], (0.0, 0.0))
            moved = dict(before, yaw=before["yaw"] + dyaw,
                         pitch=max(-math.pi / 2, min(math.pi / 2, before["pitch"] + dpitch)))
            rewards[channel] = self.track_reward * (error_before - math.hypot(*aim_errors(moved, target)))
        return rewards

    def _view_reward(self, curr_state: dict, actions: dict) -> float:
        target = curr_state.get("target")
        self_state = curr_state["self"]

        if target is None:
            self.count("no_target")
            # Цели нет — лёгкий штраф голове за то, что даже не ищет.
            return self.idle_penalty if actions["head"] == "head_idle" else 0.0

        h_angle, v_angle = aim_errors(self_state, target)
        error = math.hypot(h_angle, v_angle)
        reward = 0.0

        in_view = abs(h_angle) <= self.half_fov and abs(v_angle) <= self.half_fov
        if not in_view:
            return reward

        distance = math.dist(
            (target["x"], target["y"], target["z"]),
            (self_state["x"], self_state["y"], self_state["z"]),
        )
        if self._is_occluded(curr_state, h_angle, v_angle, distance):
            return reward

        if error <= self.center_angle:
            self.count("centered")
            return reward + self.center_reward
        return reward + self.in_view_reward

    @classmethod
    def summarize(cls, events: dict, ticks: int, bot_minutes: float) -> tuple[str, float | None]:
        if ticks <= 0:
            return "", None
        # Доля "в прицеле" — от времени, когда цель вообще была: иначе боты,
        # которые цель не видят (далеко), тянули бы цифру вниз, хотя смотреть
        # им просто не на что.
        with_target = ticks - events.get("no_target", 0)
        if with_target <= 0:
            return "цели нет — рядом никого?", None
        centered = 100.0 * events.get("centered", 0) / with_target
        text = f"цель в прицеле {percent(centered)} времени"
        no_target = 100.0 * events.get("no_target", 0) / ticks
        if no_target >= 20:
            text += f" (а {percent(no_target)} времени цели не было — далеко или рядом никого)"
        return text, centered

    def _is_occluded(self, curr_state: dict, h_angle: float, v_angle: float, target_distance: float) -> bool:
        """Загорожена ли цель: ближайший к направлению на цель луч сетки
        зрения упёрся в блок заметно раньше, чем до цели."""
        cells = curr_state["vision"]["cells"]
        res_x, res_y = curr_state["vision"]["resolution"]
        if res_x != self.res_x or res_y != self.res_y or not cells:
            return False  # рассинхрон/пустая сетка — не режем награду зря

        # Наклон луча на цель в системе сетки: в camera-режиме сетка висит на
        # pitch бота (v_angle уже относительно него), в circular — вокруг
        # горизонта, поэтому нужен абсолютный наклон (v_angle + pitch).
        ray_pitch = v_angle if not self.circular else v_angle + curr_state["self"]["pitch"]

        # Колонки и ряды — как в js/vision.js (yaw растёт влево, pitch > 0 —
        # вверх): circular — col 0 прямо вперёд, дальше по кругу вправо;
        # camera — col 0 слева (+fov/2); ряд 0 — верх (+span/2).
        if self.circular:
            col_span = 2 * math.pi / res_x
            col = round(-wrap_angle(h_angle) / col_span) % res_x if res_x > 1 else 0
        else:
            span = 2 * self.half_fov
            col = round((self.half_fov - h_angle) / span * (res_x - 1)) if res_x > 1 else 0
        row = round((self.vertical_span / 2 - ray_pitch) / self.vertical_span * (res_y - 1)) if res_y > 1 else 0
        col = min(max(col, 0), res_x - 1)
        row = min(max(row, 0), res_y - 1)

        idx = (row * res_x + col) * 5  # r, g, b, d, класс — 5 значений на ячейку
        cell_distance = cells[idx + 3] * self.max_view_distance
        return cell_distance < target_distance - self.occlusion_margin
