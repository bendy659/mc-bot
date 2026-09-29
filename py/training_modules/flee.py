"""Салки, роль "убегающий": как можно дольше не дать себя осалить.

От кого бежать, говорит судья (py/tag_game.py: game_target — ближайший
водящий; салки — "заражение": осаленный тоже водит, водящих всё больше).
Сеть видит его как цель — угол, расстояние, те же признаки, что у
остальных задачек, — и учится уходить от неё. Маршрут к угрозе ей не
нужен (observe не подменяет цель точкой маршрута, как walking).

Награда — "эйфория" по идее автора: чем дольше тебя не салят и чем
дальше водящий, тем лучше (общая на все каналы: убежать можно и ногами, и
столбом из блоков):
  - survive_reward за каждый тик, пока не осалили — это таймер: чем
    дольше продержался, тем больше набрал, осаливание его обрывает;
  - distance_reward * (расстояние до водящего / distance_cap);
  - "дразнить воду" (идея автора: "вода" — водящий) — tease_reward, больше
    всего в tease_peak блоках от водящего (чуть дальше, чем он салит),
    сходит на нет за tease_width в обе стороны: подойди поближе, но не
    попадись. Бонус больше, чем за расстояние, — стоять вдали уже не
    выгоднее всего;
  - осалили — caught_penalty (исход сообщает судья, эпизод на этом
    окончен: дальше этот бот водит);
  - время раунда вышло, а тебя не поймали — survivor_reward (исход
    "survived", тоже конец эпизода);
  - препятствие по курсу и потеря здоровья — как в walking;
  - застоялся (idle_seconds почти не двигался) — idle_penalty, и судья
    делает /kill. Иначе, пока водящий неумелый, убегающие выучивали
    "стоять выгоднее всего" — награда за "не осалили" капала и стоя.
"""

from __future__ import annotations

from .chase import _stillness_watch
from .targets import times
from .walking import WalkingModule, _distance


class FleeModule(WalkingModule):
    name = "flee"
    allowed_actions: dict = {}  # всё разрешено

    def __init__(self, config: dict):
        super().__init__(config)
        cfg = config["modules"].get("flee", {})
        self.survive_reward = cfg.get("survive_reward", 0.2)
        self.distance_reward = cfg.get("distance_reward", 0.3)
        self.distance_cap = cfg.get("distance_cap", 16.0)
        self.caught_penalty = cfg.get("caught_penalty", -50.0)
        self.hit_back_reward = cfg.get("hit_back_reward", 2.0)  # бот-цель в охоте ударил охотника
        self.survivor_reward = cfg.get("survivor_reward", 20.0)
        self.tease_reward = cfg.get("tease_reward", 0.5)
        self.tease_peak = cfg.get("tease_peak", 6.0)
        self.tease_width = cfg.get("tease_width", 4.0)
        self.game_target: dict | None = None  # цель от судьи, обновляется каждый тик
        self.outcome: str | None = None       # "caught" — осалили, "survived" — не поймали (от судьи)
        self.stillness, self.idle_penalty = _stillness_watch(config)
        self.unstick_requested = False        # застоялся — судья накажет (/kill)

    def reset(self, state: dict) -> None:
        super().reset(state)
        self.stillness.reset()

    def on_tick(self, state: dict) -> None:
        self.current_target = self.game_target

    def observe(self, state: dict) -> dict:
        return state  # сеть смотрит на саму угрозу, а не на маршрут к ней

    def compute_reward(self, prev_state: dict, curr_state: dict, actions: dict) -> dict:
        if curr_state.get("dead"):
            self.count("deaths")
            return self.team(self.death)

        value = self._obstacle_reward(curr_state, actions["legs"])
        value += self._health_loss_penalty(prev_state, curr_state)
        if self.outcome == "caught":
            return self.team(value + self.caught_penalty)

        value += self.survive_reward
        # Бот-цель в охоте ударил охотника (урон с силой удара, как у охотника):
        # откинул — выиграл время. В салках убегающему бить некого — тут пусто.
        for hit in curr_state.get("damage_dealt") or []:
            value += self.hit_back_reward * (0.2 + 0.8 * hit.get("charge", 1.0) ** 2)
            self.count("hit_back")
        if self.outcome == "survived":
            value += self.survivor_reward
        if self.stillness.update(curr_state["self"]):
            self.count("stood_still")
            self.unstick_requested = True
            value += self.idle_penalty
        threat = curr_state.get("target")
        if threat is None:
            distance = self.distance_cap  # водящего нет или он далеко — дальше некуда
        else:
            distance = min(_distance(threat, curr_state["self"]), self.distance_cap)
            self.count("with_threat")
            self.count("distance_dm", int(distance * 10))
            closeness = max(0.0, 1.0 - abs(distance - self.tease_peak) / self.tease_width)
            value += self.tease_reward * closeness
            if closeness >= 0.5:
                self.count("teasing")  # для сводки: "дразнил воду"
        rewards = self.team(value + self.distance_reward * distance / self.distance_cap)
        rewards["head"] += self._level_gaze_penalty(curr_state)  # смотреть в небо убегающему незачем
        return rewards

    @classmethod
    def summarize(cls, events: dict, ticks: int, bot_minutes: float) -> tuple[str, float | None]:
        if ticks <= 0:
            return "", None
        caught = events.get("caught", 0)
        text = f"осалили {times(caught)}"
        if caught:
            text += f" (в среднем через {bot_minutes * 60 / caught:.0f} с беготни)"
        if events.get("survived"):
            text += f", продержались до конца раунда {times(events['survived'])}"
        if events.get("stood_still"):
            text += f", застаивались {times(events['stood_still'])}"
        if events.get("hit_back"):
            text += f", отбивались от охотников {times(events['hit_back'])}"
        score = None
        if events.get("with_threat"):
            score = events.get("distance_dm", 0) / 10 / events["with_threat"]
            text += f", водящий в среднем в {score:.1f} блоках"
            text += f", дразнили воду {100 * events.get('teasing', 0) / events['with_threat']:.0f}% времени"
        return text, score
