"""Задачка bridge — мост над пустотой (первый шаг к бедварсу, автор
2026-09-28: "так скоро можно и в бедварс научить играть").

Бот стоит на острове, цель — остров за пропастью: у каждого бота своя
дорожка трассы (py/bridge_course.py), цель кладёт судья (ai_loop._bridge_turn)
в game_target. Мост строят крадучись: спиной к пропасти, задом до края
(крадущегося игра с края не пускает), взгляд вниз, блок перед собой
(place_front) — луч из-за края попадает в бок блока под ногами.

Награда — основа ходьбы (walking): приближение к цели и давление времени,
но:
  - прогресс — по прямой в 3D (маршрута через пустоту нет, точка маршрута
    стояла бы на краю острова);
  - стоять на месте — не безделье (мост строят стоя у края);
  - голову за наклон взгляда не штрафуем (смотреть вниз — и есть работа);
  - дошёл до острова цели — cross_reward и бонус за скорость (исход
    "crossed" от судьи, конец эпизода), упал в пустоту — death_penalty.
Руки — блок перед собой, столб под себя (остров цели выше) или ничего.
Столб в маске обязателен: пример учителя с запрещённым действием мозг не
учит (dqn: demo_weight), и сеть не могла выучить "вверх" (2026-09-29).

Бонус за скорость (автор: "чем быстрее прийдут к другой платформе — тем
больше очков получат"): мосты разной длины (своя раскладка каждой попытки),
поэтому время сравнивается с нормой на этот путь — par_seconds: на
развернуться и прицелиться (par_extra) плюс на каждый блок пути
(par_per_block). Мгновенно — speed_bonus целиком, за норму — половина, за две
нормы — ноль (дальше судья считает, что застрял). Одного gamma для спешки
мало: на длинном мосту награда за переход и так далеко, и лишние секунды
почти ничего не стоили.
"""

from __future__ import annotations

import math

from .targets import times
from .walking import WalkingModule

BRIDGE = "bridge"
CROSSED = "crossed"  # исход от судьи: дошёл до острова цели
FELL = "fell"        # исход от судьи: упал ниже островов (в игре — в пустоту)
STEEP_PITCH = math.radians(-85)  # взгляд ниже — почти прямо вниз: мосту бесполезен


class BridgeModule(WalkingModule):
    name = BRIDGE
    allowed_actions = {"hands": ["hands_idle", "place_front", "place_below"]}

    def __init__(self, config: dict):
        super().__init__(config)
        cfg = config["modules"].get("bridge", {})
        self.cross_reward = cfg.get("cross_reward", 50.0)
        self.death_penalty = cfg.get("death_penalty", -50.0)
        self.speed_bonus = cfg.get("speed_bonus", 100.0)
        # Взгляд прямо вниз (-90°) мосту не нужен никогда: луч уходит мимо бока
        # блока под ногами — place_front не ставит, а столбу наклон всё равно.
        # Сеть часто "проскакивала" -80° лишним look_down и застревала у края
        # (2026-09-28/29) — голове за это свой штраф, каждый тик.
        self.steep_gaze_penalty = cfg.get("steep_gaze_penalty", 0.2)
        self.par_per_block = cfg.get("par_per_block", 1.0)
        self.par_extra = cfg.get("par_extra", 3.0)
        # Дошёл — за сколько секунд и сколько блоков был путь (от судьи, с исходом crossed).
        self.crossed_seconds = 0.0
        self.crossed_blocks = 0.0
        # "Дошёл до цели" из walking не нужен: конец попытки решает судья
        # (стоит на острове цели), награда — cross_reward.
        self.goal_radius = 0.0
        self.reach_goal = 0.0
        self.near_goal_reward = 0.0
        self.game_target: dict | None = None  # цель от судьи, каждый тик
        self.outcome: str | None = None       # "crossed" — дошёл, "fell" — упал (от судьи)

    def on_tick(self, state: dict) -> None:
        self.current_target = self.game_target

    def observe(self, state: dict) -> dict:
        # Сети — сама цель (остров за пропастью), а не точка маршрута на краю.
        observed = dict(state)
        observed["goal"] = state.get("target")
        return observed

    def par_seconds(self, blocks: float) -> float:
        """Норма времени на путь в blocks блоков (от старта до острова цели)."""
        return self.par_extra + self.par_per_block * blocks

    def speed_reward(self) -> float:
        par = self.par_seconds(self.crossed_blocks)
        return self.speed_bonus * max(0.0, 1.0 - self.crossed_seconds / (2 * par))

    def _progress_distance(self, state: dict, direct: float) -> tuple[float, str]:
        return direct, "direct"

    def _standing_is_work(self, state: dict) -> bool:
        return True

    def compute_reward(self, prev_state: dict, curr_state: dict, actions: dict) -> dict:
        if curr_state.get("dead") or self.outcome == FELL:
            self._forget_target()
            return self.team(self.death_penalty)
        value = self._progress_reward(prev_state, curr_state, actions["legs"])
        if self.outcome == CROSSED:
            value += self.cross_reward + self.speed_reward()
        rewards = self.team(value)
        if curr_state["self"]["pitch"] <= STEEP_PITCH:
            rewards["head"] -= self.steep_gaze_penalty
        return rewards

    @classmethod
    def summarize(cls, events: dict, ticks: int, bot_minutes: float) -> tuple[str, float | None]:
        crossed = events.get(CROSSED, 0)
        rate = crossed / bot_minutes if bot_minutes > 0 else 0.0
        text = f"перешёл пропасть {times(crossed)} ({rate:.2f} в минуту на бота)"
        if crossed and events.get("cross_ms"):
            text += f", в среднем за {events['cross_ms'] / crossed / 1000:.1f} с"
        if events.get(FELL):
            text += f", упал {times(events[FELL])}"
        if events.get("time_up"):
            text += f", застрял {times(events['time_up'])}"
        # По видам моста (py/bridge_course.py): дошёл / попыток.
        kinds = (("straight", "прямо"), ("up", "вверх"), ("down", "вниз"), ("turn", "Г"))
        parts = [f"{label} {events.get('crossed_' + kind, 0)}/{events['start_' + kind]}"
                 for kind, label in kinds if events.get("start_" + kind)]
        if parts:
            text += " · дошёл из попыток: " + ", ".join(parts)
        return text, rate
