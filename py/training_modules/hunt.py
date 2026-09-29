"""Задачка hunt — "Останови меня" (идея автора): догнать человека и бить,
пока не остановишь (не забьёшь).

Кого ловить и когда начинать, решает судья (py/hunt_game.py: game_target —
цель, до !start её нет и боты стоят). Бежать к цели — как водящий в
салках (py/training_modules/chase.py): маршрут в обход препятствий,
награда за каждый блок ближе по маршруту, давление времени. Сверху:
  - hit_reward за каждый удар, нанёсший цели урон, — умноженный на силу
    удара, как урон в игре: 0.2 + 0.8 * заряд² (Node сообщает, по кому урон
    и с каким зарядом, — damage_dealt). Махать каждый тик невыгодно: взмах
    сбрасывает заряд, а удар в неуязвимость цели (полсекунды после урона)
    урона не наносит вовсе (идея автора: "научить правильно махать руками");
    крит — удар в полную силу в падении после прыжка, не на бегу — x1.5, как
    в игре (автор: "удар в падении — это очень хорошо");
  - kill_reward, когда цель остановили (исход "killed" от судьи — конец
    эпизода).
Действия (автор: "пока только ходить, смотреть, бить, бегать"): ноги и
голова — все, руки — только бить.
"""

from __future__ import annotations

from .chase import ChaseModule
from .targets import times


class HuntModule(ChaseModule):
    name = "hunt"
    # Руки: удар (он же копка блока в прицеле, если бить некого), столб под
    # себя и блок перед собой — цель может забраться на столб (автор,
    # 2026-09-27: "научить их с миром взаимодействовать").
    allowed_actions = {"hands": ["hands_idle", "attack_center", "place_below", "place_front"]}

    def __init__(self, config: dict):
        super().__init__(config)
        cfg = config["modules"].get("hunt", {})
        self.hit_reward = cfg.get("hit_reward", 5.0)
        self.kill_reward = cfg.get("kill_reward", 50.0)
        # Смерть охотника (цель отбилась) — штраф свой, меньше общего
        # reward.death (-100): вживую автор с мечом убивает охотников часто
        # (~0.5 раза в минуту на бота), и с -100 за смерть против +50 за
        # остановку охотникам выгоднее было держаться от цели подальше.
        self.death_penalty = cfg.get("death_penalty", config["reward"]["death"])

    def compute_reward(self, prev_state: dict, curr_state: dict, actions: dict) -> dict:
        base = super().compute_reward(prev_state, curr_state, actions)
        value = base["legs"]
        if curr_state.get("dead"):
            return self.team(self.death_penalty)
        target = self.game_target
        for hit in curr_state.get("damage_dealt") or []:
            if target is None or hit.get("id") != target.get("entity_id"):
                continue
            charge = hit.get("charge", 1.0)
            self.count("hits")
            if charge >= 0.9:
                self.count("full_hits")
            damage = self.hit_reward * (0.2 + 0.8 * charge * charge)
            if hit.get("crit"):
                self.count("crits")
                damage *= 1.5
            value += damage
        if self.outcome == "killed":
            value += self.kill_reward
        rewards = self.team(value)
        rewards["head"] += base["head"] - base["legs"]  # штраф за наклон взгляда — как у chase
        return rewards

    # Награды охоты достались от ходьбы: прогресс — по длине маршрута, стоять
    # на месте — штраф. Когда пешком до цели не дойти (коробка, столб — маршрут
    # не полный), это наказывало ровно то, что нужно: копать и строиться стоя
    # (штраф за безделье каждый тик), подниматься на своём столбе (путь "от
    # маршрута" рос). Учитель показывал одно, награда учила другому.
    def _progress_distance(self, state: dict, direct: float) -> tuple[float, str]:
        route = state.get("route")
        if route and not route.get("complete"):
            return direct, "direct"  # по прямой в 3D: подъём к цели на столбе — прогресс
        return super()._progress_distance(state, direct)

    def _standing_is_work(self, state: dict) -> bool:
        route = state.get("route")
        return bool(route) and not route.get("complete")

    @classmethod
    def summarize(cls, events: dict, ticks: int, bot_minutes: float) -> tuple[str, float | None]:
        hits = events.get("hits", 0)
        rate = hits / bot_minutes if bot_minutes > 0 else 0.0
        text = f"попал по цели {times(hits)} ({rate:.1f} в минуту охоты"
        if hits:
            text += f", в полную силу {100 * events.get('full_hits', 0) / hits:.0f}%"
            if events.get("crits"):
                text += f", критов {100 * events['crits'] / hits:.0f}%"
        text += ")"
        if events.get("kills"):
            text += f", остановили цель {times(events['kills'])}"
        with_target = ticks - events.get("no_target", 0)
        if events.get("distance_dm") and with_target > 0:
            text += f", в среднем в {events['distance_dm'] / 10 / with_target:.1f} блоках от неё"
        if events.get("stood_still"):
            text += f", застаивались {times(events['stood_still'])}"
        return text, rate
