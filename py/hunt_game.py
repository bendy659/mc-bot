"""Судья задачки hunt — "Останови меня" (идея автора): боты догоняют цель
и бьют её, пока не "остановят" (не забьют).

Цель:
  - человек — !start (цель — кто написал) или !start <ник>; где он, судья
    знает от ботов (state.humans — js/actions.js: visibleHumans), что он
    умер — из сообщения сервера о смерти (js/bot.js шлёт player_died);
  - бот роя — !start AI_3: он убегает своим мозгом убегающего (задачка
    flee), остальные охотятся; умер — по его же состоянию, добил тот, кто
    последним нанёс ему урон (damage_dealt);
  - !start auto — цель каждый раз случайный бот, и после "остановки" через
    restart_pause_seconds всё начинается заново само (автор: "чтоб они сами
    там развивались, бились").
Через head_start_seconds после !start охотники срываются с места — цель
успевает отбежать (бот-цель бежит сразу). До !start и после остановки
цели охотники стоят и опыта не пишут (ai_loop: как "считает до пяти" в
салках). Бить Node даёт только цель (tag_ids = [id цели]).
"""

from __future__ import annotations

import math
import random
import time

HUNT = "hunt"      # имя задачки охотников (training_modules/hunt.py)
FLEE = "flee"      # задачка бота-цели
KILLED = "killed"  # исход эпизода охотников: цель остановили
LOST_SECONDS = 5.0  # цель-бот так долго не видна (вышел) — в auto выбираем другую
CLIMBER_MARGIN = 1.0  # лезущий на столб отдаёт роль, только если другой ближе к цели на столько блоков


class HuntGame:
    def __init__(self, config: dict):
        cfg = config["modules"].get("hunt", {})
        self.head_start = cfg.get("head_start_seconds", 3.0)
        self.pause = cfg.get("restart_pause_seconds", 3.0)
        # auto: охота на бота не дольше стольких секунд — не остановили, новая
        # цель (бот-цель отбивается и лечится; иначе охота шла бы вечно).
        self.auto_seconds = cfg.get("auto_seconds", 90.0)
        self.target_name: str | None = None
        self.target_bot: int | None = None       # id сессии, если цель — бот роя
        self.auto = False                        # !start auto: цель — случайный бот, по кругу
        self.restart_at: float | None = None     # когда в auto начать следующую охоту
        self.active_from: float | None = None    # None — охоты нет (ждём !start)
        self.target: dict | None = None          # {"entity_id", "pos", "seen"} — где цель
        self.hunters: dict[int, dict] = {}       # id сессии -> {"entity_id", "pos", "seen"} — где охотники
        self.last_hitter: int | None = None      # кто последним нанёс цели урон
        self.pending_kill: set[int] = set()      # охотники, которым ещё не сказали "остановили"
        self.credited: int | None = None         # кому засчитать остановку в сводке (один раз)
        self.kills = 0
        self.climber: int | None = None          # кто из охотников лезет на столб за целью (closest)

    # --- начало и конец охоты --------------------------------------------------

    def start(self, name: str | None, now: float | None = None, bot: int | None = None, auto: bool = False) -> None:
        now = time.time() if now is None else now
        self.target_name = name
        self.target_bot = bot
        self.auto = auto
        self.target = None
        self.last_hitter = None
        self.restart_at = None
        self.active_from = now + self.head_start
        self.climber = None
        self.pending_kill.clear()

    def next_auto(self, candidates: list[int], name_of, now: float | None = None) -> int | None:
        """auto: пора — новая охота на случайного бота из candidates.
        Возвращает, кто теперь цель (или None)."""
        now = time.time() if now is None else now
        lost = (self.target_bot is not None and self.active_from is not None and self.target is not None
                and now - self.target["seen"] > LOST_SECONDS)
        timed_out = self.active_from is not None and now - self.active_from > self.auto_seconds
        if not self.auto or not candidates or not (lost or timed_out
                                                   or (self.restart_at is not None and now >= self.restart_at)):
            return None
        chosen = random.choice(candidates)
        self.start(name_of(chosen), now, bot=chosen, auto=True)
        return chosen

    def died(self, name: str, hunters: list[int], killer: int | None = None, now: float | None = None) -> bool:
        """Цель name умерла. Если это правда цель — охота окончена: всем
        охотникам исход KILLED (награда общая: остановили вместе). В сводке
        остановка засчитывается один раз — добившему (killer) или первому."""
        now = time.time() if now is None else now
        if self.active_from is None or name != self.target_name:
            return False
        self.kills += 1
        self.active_from = None
        self.target = None
        self.target_bot = None
        self.pending_kill = set(hunters)
        self.credited = killer if killer in hunters else min(hunters, default=None)
        if self.auto:
            self.restart_at = now + self.pause
        return True

    # --- кто где ---------------------------------------------------------------

    def active(self, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        return self.active_from is not None and now >= self.active_from

    def waiting(self, session_id: int, now: float | None = None) -> bool:
        """Стоит ли бот: охотники — пока охота не началась (и пока у цели
        фора), бот-цель — только когда охоты нет вовсе."""
        if session_id == self.target_bot:
            return self.active_from is None
        return not self.active(now)

    def role_of(self, session_id: int) -> str:
        return FLEE if session_id == self.target_bot else HUNT

    def see(self, humans: list[dict], now: float | None = None) -> None:
        """Сведения ботов о людях рядом: где цель-человек. Без ника в !start
        (из лаунчера) — цель тот, кого увидели первым в выживании."""
        now = time.time() if now is None else now
        if self.target_bot is not None:
            return
        for human in humans:
            if human.get("gamemode", 0) not in (0, 2):
                continue  # наблюдатель и творческий режим — не цель
            if self.target_name is None and self.active_from is not None:
                self.target_name = human["name"]
            if human["name"] == self.target_name:
                self.target = {"entity_id": human.get("id"), "pos": (human["x"], human["y"], human["z"]), "seen": now}

    def see_bot(self, session_id: int, entity_id, pos: tuple, dead: bool, now: float | None = None,
                blocks: int | None = None) -> None:
        """Где бот роя (из его же состояния): цель-бот — это цель, остальные —
        охотники (от ближайшего бот-цель убегает). blocks — сколько у него
        блоков для постройки (None — неизвестно)."""
        now = time.time() if now is None else now
        where = {"entity_id": entity_id, "pos": pos, "seen": now, "blocks": blocks}
        if session_id == self.target_bot:
            if not dead:
                self.target = where
            self.hunters.pop(session_id, None)
        elif dead:
            self.hunters.pop(session_id, None)
        else:
            self.hunters[session_id] = where

    def note_damage(self, session_id: int, damage_dealt: list[dict]) -> None:
        if self.target is not None and any(hit.get("id") == self.target["entity_id"] for hit in damage_dealt):
            self.last_hitter = session_id

    def target_for(self, now: float | None = None) -> dict | None:
        """Цель охотникам: id сущности и запасная позиция (Node подставит её,
        если сам цель не видит). Давно не видели — цели нет."""
        now = time.time() if now is None else now
        if not self.active(now) or self.target is None or now - self.target["seen"] > 3.0:
            return None
        return self._as_target(self.target)

    def threat_for(self, now: float | None = None) -> dict | None:
        """Боту-цели — ближайший охотник (от него и бежать)."""
        now = time.time() if now is None else now
        if self.target is None:
            return None
        fresh = [h for h in self.hunters.values() if now - h["seen"] <= 3.0]
        nearest = min(fresh, key=lambda h: math.dist(h["pos"], self.target["pos"]), default=None)
        return self._as_target(nearest) if nearest else None

    def hunter_ids(self, now: float | None = None) -> list[int]:
        """Кого может ударить бот-цель — охотников (отбиваться: удар откидывает)."""
        if not self.active(now):
            return []
        return [h["entity_id"] for h in self.hunters.values() if h["entity_id"] is not None]

    def closest(self, session_id: int, now: float | None = None) -> bool:
        """Лезть ли охотнику session_id на столб за целью (вход сети closest и
        учитель, sim/teacher.py): это ближайший к цели по горизонтали охотник
        из тех, у кого есть чем строить (блоки — из их же состояний: лимит
        никто не ставит, у бота то, что есть в инвентаре). Роль держится, пока
        другой не станет ближе на CLIMBER_MARGIN: иначе двое, подходя почти
        вровень, перехватывали её друг у друга каждый тик, и начатый столб
        бросался недостроенным. Охоты или цели нет — False."""
        now = time.time() if now is None else now
        target = self.target_for(now)
        if target is None or session_id == self.target_bot:
            self.climber = None
            return False
        builders = {}
        for other_id, where in self.hunters.items():
            if now - where["seen"] > 3.0 or where.get("blocks") == 0:
                continue
            builders[other_id] = math.hypot(target["x"] - where["pos"][0], target["z"] - where["pos"][2])
        if not builders:
            self.climber = None
            return False
        best = min(builders, key=lambda other_id: (builders[other_id], other_id))
        if self.climber not in builders or builders[best] < builders[self.climber] - CLIMBER_MARGIN:
            self.climber = best
        return self.climber == session_id

    def taggable_ids(self, now: float | None = None) -> list[int]:
        target = self.target_for(now)
        return [target["entity_id"]] if target and target["entity_id"] is not None else []

    @staticmethod
    def _as_target(where: dict) -> dict:
        x, y, z = where["pos"]
        return {"entity_id": where["entity_id"], "x": round(x), "y": round(y), "z": round(z)}
