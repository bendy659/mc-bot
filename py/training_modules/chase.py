"""Салки, роль "водящий": догнать убегающего и ударить — осалить.

За кем бежать, говорит судья (py/tag_game.py: game_target — убегающий,
обычно ближайший). Дальше — как в walking: маршрут в обход препятствий
(js/route.js), награда за каждый блок ближе по маршруту и "давление
времени" (каждый тик погони немного в минус — бегом и сразу бить выгоднее).
Вплотную этого мало — надо ударить (attack_center: водящему судья
разрешает бить убегающих). Засчитал удар судья — модуль узнаёт об этом из
outcome перед последним переходом эпизода: +tag_reward. Салки — "заражение":
осаленный тоже водит, и последнего водящие ловят уже толпой; поймали
всех — раунд выигран, каждому водящему +round_reward (outcome "won").

Действия — полная свобода (блоки, предметы, броня), и награда общая на все
каналы: осалить помогают и ноги (догнать), и голова (навестись), и руки
(ударить), а мост или столб из блоков — тоже путь к убегающему.
"""

from __future__ import annotations

from .targets import times
from .walking import StillnessWatch, WalkingModule, _distance


def _stillness_watch(config: dict) -> tuple[StillnessWatch, float]:
    """Сторож "застоялся" для ролей салок (общий с flee.py): бот idle_seconds
    не отходил дальше idle_radius — штраф idle_penalty, а судья делает ему
    /kill (если kill_idle) — как автор наказывал таких вручную."""
    cfg = config["modules"].get("tag", {})
    tick_seconds = config["train"].get("tick_rate_ms", 150) / 1000.0
    watch = StillnessWatch(cfg.get("idle_seconds", 45.0), cfg.get("idle_radius", 2.0), tick_seconds)
    return watch, cfg.get("idle_penalty", -10.0)


class ChaseModule(WalkingModule):
    name = "chase"
    # Рукам — бить (осалить) и копать/ставить блоки (мост, столб к
    # убегающему); есть, надевать броню и выбрасывать вещи водящему незачем:
    # из 7 действий рук удар выпадал наугад редко, а "поесть" на 5 секунд
    # занимало руки (заметил автор: водящий рядом и не бьёт).
    allowed_actions = {"hands": ["hands_idle", "attack_center", "place_below", "place_front"]}

    def __init__(self, config: dict):
        super().__init__(config)
        cfg = config["modules"].get("chase", {})
        self.tag_reward = cfg.get("tag_reward", 50.0)
        self.round_reward = cfg.get("round_reward", 20.0)
        # "Дошёл до цели" из walking тут не бывает: догнать мало, надо
        # ударить. Прогресс и давление времени — до самого удара.
        self.goal_radius = 0.0
        self.reach_goal = 0.0
        self.near_goal_reward = 0.0
        self.game_target: dict | None = None  # цель от судьи, обновляется каждый тик
        self.outcome: str | None = None       # "caught" — осалил, "won" — поймали всех (от судьи)
        self.stillness, self.idle_penalty = _stillness_watch(config)
        self.unstick_requested = False        # застоялся — судья накажет (/kill)

    def reset(self, state: dict) -> None:
        super().reset(state)
        self.stillness.reset()

    def on_tick(self, state: dict) -> None:
        self.current_target = self.game_target

    def compute_reward(self, prev_state: dict, curr_state: dict, actions: dict) -> dict:
        value = super().compute_reward(prev_state, curr_state, actions)["legs"]
        if curr_state.get("dead"):
            return self.team(value)
        if self.outcome == "caught":
            value += self.tag_reward
        elif self.outcome == "won":
            value += self.round_reward
        if self.stillness.update(curr_state["self"]):
            self.count("stood_still")
            self.unstick_requested = True
            value += self.idle_penalty
        rewards = self.team(value)
        rewards["head"] += self._level_gaze_penalty(curr_state)  # наводить удар голова может сама, небо — нет
        return rewards

    def _progress_reward(self, prev_state: dict, curr_state: dict, legs_action: str) -> float:
        target = curr_state.get("target")
        if target is None:
            self.count("no_target")
        else:
            # Для сводки: как далеко в среднем держался от убегающего.
            self.count("distance_dm", int(_distance(target, curr_state["self"]) * 10))
        return super()._progress_reward(prev_state, curr_state, legs_action)

    @classmethod
    def summarize(cls, events: dict, ticks: int, bot_minutes: float) -> tuple[str, float | None]:
        tags = events.get("caught", 0)
        rate = tags / bot_minutes * 10 if bot_minutes > 0 else 0.0
        text = f"осалил {times(tags)} ({rate:.1f} за 10 мин вождения)"
        if events.get("won"):
            text += f", поймали всех {times(events['won'])}"
        if events.get("time_up"):
            text += f", не успели за раунд {times(events['time_up'])}"
        if events.get("stood_still"):
            text += f", застаивался {times(events['stood_still'])}"
        with_target = ticks - events.get("no_target", 0)
        if events.get("distance_dm") and with_target > 0:
            text += f", в среднем в {events['distance_dm'] / 10 / with_target:.1f} блоках от убегающего"
        return text, rate
