"""Задачка bedwars — свой бедварс для ботов: две команды на соседних
островах карты Hypixel, сломать чужую кровать и выбить соперников (судья —
py/bedwars_game.py).

Куда идти, говорит судья (game_target): враг рядом — он, иначе чужая
кровать. Дорога — как в охоте (py/training_modules/hunt.py): по маршруту,
если до цели можно дойти пешком, иначе по прямой (через пустоту — мост:
навык задачки bridge, с неё мозг и начинает, train.warm_start_from), награда
за каждый блок ближе. Сверху:
  - hit_reward за удар по врагу — умноженный на силу удара, как урон в игре;
  - bed_reward — сломал чужую кровать (судья: ближайший к ней враг);
  - kill_reward — убил врага (ударил последним);
  - bed_lost_penalty — сломали свою кровать (всей команде);
  - win_reward / lose_penalty — конец игры (исход от судьи — конец эпизода);
  - death_penalty — смерть (в бою или в пустоте).
Руки — бить (и копать: кровать, чужие блоки на пути), столб под себя, блок
перед собой; ноги и голова — все действия (мост крадучись — sneak_back).
"""

from __future__ import annotations

from .hunt import HuntModule
from .targets import times


class BedwarsModule(HuntModule):
    name = "bedwars"

    def __init__(self, config: dict):
        super().__init__(config)
        cfg = config["modules"].get("bedwars", {})
        self.hit_reward = cfg.get("hit_reward", 3.0)
        self.bed_reward = cfg.get("bed_reward", 100.0)
        self.kill_reward = cfg.get("kill_reward", 30.0)
        self.bed_lost_penalty = cfg.get("bed_lost_penalty", -20.0)
        self.win_reward = cfg.get("win_reward", 100.0)
        self.lose_penalty = cfg.get("lose_penalty", -30.0)
        self.death_penalty = cfg.get("death_penalty", -30.0)
        self.enemy_ids: set = set()      # кого можно бить (судья), для награды за удары
        self.game_events: list[str] = []  # события от судьи с прошлого перехода (bedwars_game)
        self.teacher_bridging = False     # учитель симуляции уже строит мост (py/sim/teacher.py)

    def reset(self, state: dict) -> None:
        super().reset(state)
        self.teacher_bridging = False

    def compute_reward(self, prev_state: dict, curr_state: dict, actions: dict) -> dict:
        # Прогресс к цели, давление времени, наклон взгляда — как у водящего
        # (ChaseModule); удары охоты считаем сами — по любому врагу, не только цели.
        base = super(HuntModule, self).compute_reward(prev_state, curr_state, actions)
        value = base["legs"]
        if curr_state.get("dead"):
            value = self.death_penalty
        else:
            for hit in curr_state.get("damage_dealt") or []:
                if hit.get("id") not in self.enemy_ids:
                    continue
                charge = hit.get("charge", 1.0)
                self.count("hits")
                damage = self.hit_reward * (0.2 + 0.8 * charge * charge)
                value += damage * (1.5 if hit.get("crit") else 1.0)
        events, self.game_events = self.game_events, []
        for event in events:
            if event not in ("won", "lost"):  # исходы игры считает ai_loop (_end_episode)
                self.count(event)
            value += {"bed": self.bed_reward, "kill": self.kill_reward, "bed_lost": self.bed_lost_penalty,
                      "won": self.win_reward, "lost": self.lose_penalty}.get(event, 0.0)
        rewards = self.team(value)
        if not curr_state.get("dead"):
            rewards["head"] += base["head"] - base["legs"]  # штраф за наклон взгляда — как у chase
        return rewards

    @classmethod
    def summarize(cls, events: dict, ticks: int, bot_minutes: float) -> tuple[str, float | None]:
        beds = events.get("bed", 0)
        rate = beds / bot_minutes * 10 if bot_minutes > 0 else 0.0
        text = f"сломал кроватей {beds} ({rate:.2f} за 10 мин на бота)"
        text += f", убил {times(events.get('kill', 0))}, попал {times(events.get('hits', 0))}"
        if events.get("won") or events.get("lost"):
            text += f"; игр: победа {events.get('won', 0)}, поражение {events.get('lost', 0)}"
        if events.get("time_up"):
            text += f", ничья (время) {events['time_up']}"
        return text, rate
