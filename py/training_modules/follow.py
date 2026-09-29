"""Задачка "следовать": держаться рядом с целью-сущностью, пока она ходит.

Цель:
  - если человек поставил её через !setTarget player <ник> — он;
  - иначе модуль сам выбирает, за кем идти, по приоритету:
      1) человек (игрок, который не бот нашего роя);
      2) мирный моб (овца, корова... — враждебных не выбираем);
      3) бот нашего роя с МЕНЬШИМ номером (AI_3 может идти за AI_1 или
         AI_2, но не наоборот). Без этого правила два бота выбрали бы друг
         друга, встали рядом и получали награду "я рядом с целью", не
         делая ничего: цепочка только "вниз по номерам" исключает циклы.
    и держит выбранную сущность, пока она видна.

Награда — наследник walking (препятствия, потеря здоровья, штрафы голове
за верчение и рукам за удар — всё оттуда же), меняется только часть "к
цели":
  - дальше max_distance — плотная награда за каждый блок сближения и за
    разворот к цели (как в walking);
  - в "коридоре" [min_distance, max_distance] — in_band_reward за тик,
    стоять тут — правильно, штрафа за безделье нет;
  - ближе min_distance — небольшой штраф: не наступать цели на пятки.
"""

from __future__ import annotations

from .targets import HOSTILE, ITEM, PASSIVE, PLAYER, bot_index, find_entity, nearest, percent, present_entities
from .walking import WalkingModule, _OBSTACLE_AVOID_ACTIONS, _distance, _heading_cos, _route_distance


class FollowModule(WalkingModule):
    name = "follow"
    # allowed_actions — как у walking (наследуется): ноги — всё, голова —
    # только наклон, руки выключены.

    def __init__(self, config: dict):
        super().__init__(config)
        cfg = config["modules"].get("follow", {})
        self.min_distance = cfg.get("min_distance", 2.0)
        self.max_distance = cfg.get("max_distance", 5.0)
        self.in_band_reward = cfg.get("in_band_reward", 0.5)
        self.too_close_penalty = cfg.get("too_close_penalty", -0.2)
        self._locked_id = None
        self._in_band = False

    def reset(self, state: dict) -> None:
        super().reset(state)
        self._locked_id = None
        self._in_band = False

    def on_tick(self, state: dict) -> None:
        if state.get("target_is_human"):
            return  # человек поставил цель — идём за ней

        chosen = self._choose_target(state)
        locked = find_entity(state, self._locked_id) if self._locked_id is not None else None
        # Выбранную цель держим, пока она видна, — но появился кто-то важнее
        # (человек, а шли за ботом) — к нему. Иначе боты, сцепившиеся
        # "цепочкой" друг за другом до прихода автора, так и шли за соседом, а
        # не за ним, и трогались с опозданием (заметил автор, 2026-09-26).
        if locked is None or (chosen is not None and self._priority(chosen) < self._priority(locked)):
            self._locked_id = chosen["id"] if chosen else None
        self.current_target = {"entity_id": self._locked_id} if self._locked_id is not None else None

    def _priority(self, entity: dict) -> int:
        """Кто важнее как цель: человек (0), мирный моб (1), бот роя (2)."""
        if entity.get("type") == PLAYER:
            return 2 if bot_index(entity.get("name", ""), self.config) is not None else 0
        return 1 if entity.get("type") == PASSIVE else 3

    def _choose_target(self, state: dict) -> dict | None:
        own_index = state.get("bot_id", 1)
        humans, passive, lower_bots = [], [], []
        for entity in present_entities(state):
            kind = entity.get("type")
            if kind in (ITEM, HOSTILE):
                continue
            if kind == PLAYER:
                index = bot_index(entity.get("name", ""), self.config)
                if index is None:
                    humans.append(entity)
                elif index < own_index:
                    lower_bots.append(entity)
            elif kind == PASSIVE:
                passive.append(entity)
        for group in (humans, passive, lower_bots):
            if group:
                return nearest(group)
        return None

    @classmethod
    def summarize(cls, events: dict, ticks: int, bot_minutes: float) -> tuple[str, float | None]:
        if ticks <= 0:
            return "", None
        with_target = max(ticks - events.get("no_target", 0), 1)
        in_band = 100.0 * events.get("in_band", 0) / with_target
        text = f"рядом с целью {percent(in_band)} времени"
        if events.get("distance_dm"):
            text += f", в среднем в {events['distance_dm'] / 10 / with_target:.1f} блоках от неё"
        no_target = 100.0 * events.get("no_target", 0) / ticks
        if no_target >= 20:
            text += f" (цели нет {percent(no_target)} времени — за кем идти?)"
        return text, in_band

    def _progress_reward(self, prev_state: dict, curr_state: dict, legs_action: str) -> float:
        reward = self._obstacle_reward(curr_state, legs_action)
        reward += self._health_loss_penalty(prev_state, curr_state)

        target = curr_state.get("target")
        if target is None:
            self._forget_target()
            self._in_band = False
            self.count("no_target")
            return reward + (self.idle if legs_action == "idle" else 0.0)

        self_state = curr_state["self"]
        distance = _distance(target, self_state)
        heading = _heading_cos(self_state, self.observe(curr_state)["target"])
        # Для сводки: среднее расстояние до цели (в десятых долях блока —
        # счётчики целые). Честнее доли "рядом": видно, приближаются ли вообще.
        self.count("distance_dm", int(distance * 10))

        # Цель сменилась (перецепились на другую сущность) — без скачка награды.
        if self._prev_target is not None and _distance(target, self._prev_target) > 3.0:
            self._forget_target()
        # Прогресс — по маршруту в обход препятствий (см. walking.py).
        progress_distance, source = _route_distance(curr_state, distance)
        if source != self._progress_source:
            self._prev_distance = None
        self._progress_source = source

        if distance <= self.max_distance:
            if not self._in_band:
                self._in_band = True
                self.count("caught_up")
            self._remember_target(target, progress_distance, heading)
            if distance < self.min_distance:
                return reward + self.too_close_penalty
            self.count("in_band")
            return reward + self.in_band_reward

        # Гистерезис, как в walking: шаг туда-обратно на краю коридора не
        # должен считаться новым "догнал".
        if distance > self.max_distance + 1.0:
            self._in_band = False

        # Давление времени: цель уходит — каждый тик вдали немного в минус,
        # догнать быстрее (бегом) выгоднее.
        reward -= self.time_penalty

        if self._prev_distance is not None:
            delta = max(-1.0, min(1.0, self._prev_distance - progress_distance))
            reward += self.closer * delta
            self._heading_shaping = self.heading_reward * (heading - self._prev_heading)

            moved = abs(self_state["x"] - prev_state["self"]["x"]) >= 0.05 or \
                abs(self_state["z"] - prev_state["self"]["z"]) >= 0.05
            turned = abs(self_state.get("dyaw", 0.0)) > 0.01
            if not moved and not turned and legs_action not in _OBSTACLE_AVOID_ACTIONS:
                reward += self.idle  # цель уходит, а бот стоит

        self._remember_target(target, progress_distance, heading)
        return reward
