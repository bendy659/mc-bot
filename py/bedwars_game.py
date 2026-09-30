"""Судья задачки bedwars — свой бедварс для ботов (идея автора: "боты
против ботов — бесконечные игры") на картах Hypixel (server/bw_maps,
описания — data/bedwars/maps/<карта>.json, py/bedwars_maps.py).

Пока — две команды (автор, 2026-09-30: "обучение и так даётся медленно"):
случайная карта и два соседних острова на одной стороне карты (мост между
ними прямой; straight_pairs: false — любые соседи, и через угол). Экономики ещё нет: у всех
меч и стопка шерсти своего цвета (start_blocks), после возрождения — снова.

Правила, как в бедварсе:
  - кровать цела — умерший возрождается у себя на острове; сломана —
    выбывает до конца игры (зритель);
  - упал в пустоту (ниже void_y карты) — умер (судья делает /kill);
  - ломать можно только поставленные игроками блоки и кровати (плагин и
    симуляция не дают ломать карту);
  - победа — у соперника выбыли все; время вышло (game_seconds) — ничья,
    новая игра.
Судья решает, куда идти каждому (target_for): враг ближе fight_radius —
он, иначе чужая кровать (пока цела), иначе ближайший враг. Бить можно только
врагов (enemy_ids — tag_ids в ответе ботам). Что случилось с ботом (сломал
кровать, убил, выбыл, победа) — события для задачки (take_events).

Команды игры — строки команд сервера (их понимает и плагин по RCON, и
симуляция: py/sim/game.py SimArena.command): загрузить карту (/mcbot
bedwars <карта> — это и сброс карты), телепорт, набор, режим игры.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

BEDWARS = "bedwars"
WON = "won"          # исход эпизода: команда победила (терминально)
LOST = "lost"        # команда проиграла — у выбывших эпизод кончился смертью
BED = "bed"          # сломал чужую кровать
BED_LOST = "bed_lost"  # сломали твою кровать
KILL = "kill"        # убил врага (последний, кто его ударил)
TIME_UP = "time_up"  # игра не кончилась за game_seconds — ничья, новая игра (обрыв, не конец эпизода)

MAPS_DIR = Path(__file__).resolve().parent.parent / "data" / "bedwars" / "maps"
READY_DISTANCE = 1.5   # бот ближе стольки к своей точке появления — телепорт дошёл
RESEND_SECONDS = 2.0   # телепорт не дошёл за столько — ещё раз
KILL_CREDIT_SECONDS = 10.0
BED_CHECK_DELAY = 3.0
JOIN_SECONDS = 2.0          # первая игра — через столько после первого бота (ждём остальных)       # секунд с начала игры, пока кровати не проверяются  # убийство засчитывается тому, кто ударил последним, не раньше стольких секунд


def clock() -> float:
    """Часы судьи: настоящее время, в симуляции — игровое (train_tag_sim
    подменяет модульный time, как у судей салок и охоты)."""
    return time.time()


class BedwarsGame:
    def __init__(self, config: dict, rng: random.Random | None = None):
        cfg = config["modules"].get(BEDWARS, {})
        self.config = config
        self.rng = rng or random.Random()
        names = sorted(p.stem for p in MAPS_DIR.glob("*.json"))
        self.maps = [m for m in cfg.get("maps", names) if m in names]
        self.game_seconds = cfg.get("game_seconds", 300.0)
        self.pause = cfg.get("restart_pause_seconds", 3.0)
        self.start_blocks = cfg.get("start_blocks", 64)
        self.fight_radius = cfg.get("fight_radius", 6.0)
        self.straight_pairs = cfg.get("straight_pairs", True)
        self.respawn_seconds = cfg.get("respawn_seconds", 5.0)
        self.world = config["bot"].get("task_worlds", {}).get(BEDWARS)
        self.map: dict | None = None      # описание карты (json)
        self.teams: list[dict] = []       # две команды: color, bed, spawn, bed_alive, members
        self.players: dict[int, dict] = {}  # id сессии -> team, entity_id, pos, dead, out, ready...
        self.started_at: float | None = None
        self.next_game_at = 0.0           # когда начать следующую игру (после паузы)
        self.first_seen: float | None = None  # когда судья впервые увидел ботов
        self.events: dict[int, list[str]] = {}
        self.commands: list[str] = []
        self.stats = {"games": 0, "decided": 0, "beds": 0, "kills": 0, "time_up": 0}

    # --- начало и конец игры ---------------------------------------------------

    def want_new_game(self, now: float | None = None) -> bool:
        """Пора начинать игру. Первую — через JOIN_SECONDS после первого бота:
        боты подключаются по одному, и игра без половины роя шла бы без них."""
        now = time.time() if now is None else now
        if self.first_seen is None:
            self.first_seen = now
        return self.started_at is None and now >= max(self.next_game_at, self.first_seen + JOIN_SECONDS)

    def new_game(self, session_ids: list[int], name_of, now: float | None = None) -> None:
        """Новая игра на случайной карте: два соседних острова, игроки — по
        очереди в команды (перемешаны). Команды сервера — в self.commands."""
        now = time.time() if now is None else now
        name = self.rng.choice(self.maps)
        self.map = json.loads((MAPS_DIR / f"{name}.json").read_text(encoding="utf-8"))
        teams = self.map["teams"]
        pairs = [(i, (i + 1) % len(teams)) for i in range(len(teams))]
        if self.straight_pairs:
            # Соседи на одной стороне карты: кровати на одной линии по x или z —
            # мост прямой. Через угол нужен мост "Г", на повороте учитель пока
            # падает (2026-09-30) — такие пары позже.
            pairs = [(i, j) for i, j in pairs
                     if min(abs(teams[i]["bed"]["head"][0] - teams[j]["bed"]["head"][0]),
                            abs(teams[i]["bed"]["head"][2] - teams[j]["bed"]["head"][2])) <= 2] or pairs
        first, second = self.rng.choice(pairs)
        pair = [teams[first], teams[second]]
        self.teams = [{"color": t["color"], "bed": t["bed"], "spawn": t["spawn"], "bed_alive": True, "members": []}
                      for t in pair]
        ids = list(session_ids)
        self.rng.shuffle(ids)
        self.players = {}
        for index, session_id in enumerate(ids):
            team = index % 2
            self.teams[team]["members"].append(session_id)
            self.players[session_id] = {"team": team, "name": name_of(session_id), "entity_id": None, "pos": None,
                                        "dead": False, "out": False, "ready": False, "sent": 0.0, "states": 0,
                                        "hit_by": None, "hit_at": 0.0}
        self.started_at = now
        self.stats["games"] += 1
        self.commands.append(f"mcbot bedwars {name}")
        for session_id in ids:
            self._spawn(session_id, now, kit=True)

    def _spawn(self, session_id: int, now: float, kit: bool) -> None:
        """На свою точку появления, лицом к центру карты; набор — заново
        (экономики пока нет: меч и шерсть своего цвета)."""
        player = self.players[session_id]
        team = self.teams[player["team"]]
        x, y, z = team["spawn"]
        # Углы сервера: взгляд (-sin yaw, 0, cos yaw), yaw 0 — на юг (+z). Кратно
        # 10°: голова поворачивает шагами по 10°, и с другого угла ровно по оси
        # моста не встать (учитель мостом дёргался между ±5°).
        yaw = 10 * round(math.degrees(math.atan2(x, -z)) / 10)
        name = player["name"]
        commands = []
        if kit:
            commands += [f"gamemode survival {name}", f"clear {name}", f"give {name} minecraft:wooden_sword 1",
                         f"give {name} minecraft:{team['color']}_wool {self.start_blocks}"]
        teleport = f"tp {name} {x + 0.5} {y} {z + 0.5} {yaw:.0f} 0"
        commands.append(f"execute in minecraft:{self.world} run {teleport}" if self.world else teleport)
        self.commands += commands
        player.update(ready=False, sent=now, states=0)

    def end_game(self, winner: int | None, now: float | None = None) -> None:
        """Конец игры: победителям — WON, проигравшим — LOST (кто ещё жив —
        тоже: игра для них кончилась), ничья — TIME_UP всем."""
        now = time.time() if now is None else now
        for session_id, player in self.players.items():
            if winner is None:
                self._event(session_id, TIME_UP)
            elif not player["out"] or player["team"] == winner:
                self._event(session_id, WON if player["team"] == winner else LOST)
        if winner is None:
            self.stats["time_up"] += 1
        else:
            self.stats["decided"] += 1
        self.started_at = None
        self.next_game_at = now + self.pause

    # --- что видно от ботов ----------------------------------------------------

    def see_bot(self, session_id: int, state: dict, now: float | None = None) -> None:
        """Ход судьи по состоянию бота: где он, не упал ли в пустоту, не умер
        ли (возрождение или выбывание), готов ли после телепорта."""
        now = time.time() if now is None else now
        player = self.players.get(session_id)
        if player is None or self.started_at is None:
            return
        me = state["self"]
        dead = bool(state.get("dead"))
        player["entity_id"] = me.get("entity_id")
        for hit in state.get("damage_dealt") or []:
            victim = self._by_entity(hit.get("id"))
            if victim is not None and self.players[victim]["team"] != player["team"]:
                self.players[victim].update(hit_by=session_id, hit_at=now)
        if player["out"]:
            return
        if dead and not player["dead"]:
            self._died(session_id, now)
        player["dead"] = dead
        if dead:
            return
        if player.get("respawn_at") is not None:
            if now < player["respawn_at"]:
                return  # ждёт возрождения (зритель)
            player["respawn_at"] = None
            self._spawn(session_id, now, kit=True)
            return
        player["pos"] = (me["x"], me["y"], me["z"])
        if not player["ready"]:
            player["states"] += 1
            spawn = self.teams[player["team"]]["spawn"]
            target = (spawn[0] + 0.5, spawn[1], spawn[2] + 0.5)
            if player["states"] >= 2 and math.dist(player["pos"], target) < READY_DISTANCE:
                player["ready"] = True
            elif now - player["sent"] > RESEND_SECONDS:
                self._spawn(session_id, now, kit=False)  # телепорт не дошёл — ещё раз
            return
        if me["y"] < self.map["void_y"]:
            self.commands.append(f"kill {player['name']}")  # в пустоту — умер, не дожидаясь дна мира

    def _died(self, session_id: int, now: float) -> None:
        player = self.players[session_id]
        killer = player["hit_by"] if now - player["hit_at"] < KILL_CREDIT_SECONDS else None
        player.update(hit_by=None, ready=False, pos=None)
        if killer is not None:
            self._event(killer, KILL)
            self.stats["kills"] += 1
        if self.teams[player["team"]]["bed_alive"]:
            # Возрождение — через respawn_seconds (как в бедварсе: до того
            # зритель), на свой остров и снова с набором (вещи при смерти
            # выпадают). Сразу назад в бой — и драка на мосту шла без конца.
            player["respawn_at"] = now + self.respawn_seconds
            self.commands.append(f"gamemode spectator {player['name']}")
            return
        player["out"] = True
        self.commands.append(f"gamemode spectator {player['name']}")

    def beds_to_check(self, now: float | None = None) -> list[tuple[int, list]]:
        """Кровати, которые ещё целы: (команда, [клетки head и foot]). Первые
        BED_CHECK_DELAY секунд игры — никаких: команда загрузить карту заново
        могла ещё не дойти до сервера, и сломанная кровать прошлой игры
        засчиталась бы этой."""
        now = time.time() if now is None else now
        if self.started_at is None or now - self.started_at < BED_CHECK_DELAY:
            return []
        return [(index, [team["bed"]["head"], team["bed"]["foot"]]) for index, team in enumerate(self.teams)
                if team["bed_alive"]]

    def bed_broken(self, team_index: int, now: float | None = None) -> None:
        """Кровать команды сломана (от мира: симуляция или RCON). Кто сломал —
        ближайший к ней живой враг (в игре сервер нам этого не говорит)."""
        now = time.time() if now is None else now
        team = self.teams[team_index]
        if not team["bed_alive"]:
            return
        team["bed_alive"] = False
        self.stats["beds"] += 1
        bed = team["bed"]["head"]
        enemies = [(math.dist(p["pos"], bed), sid) for sid, p in self.players.items()
                   if p["team"] != team_index and not p["dead"] and not p["out"] and p["pos"] is not None]
        if enemies:
            self._event(min(enemies)[1], BED)
        for session_id in team["members"]:
            self._event(session_id, BED_LOST)
            player = self.players[session_id]
            if player["dead"] or player.get("respawn_at") is not None:
                # Умер до того, как сломали кровать, и ещё не возродился — выбыл.
                player.update(out=True, respawn_at=None)
                self.commands.append(f"gamemode spectator {player['name']}")

    def check_end(self, now: float | None = None) -> None:
        """Победа (у соперника все выбыли) или время вышло."""
        now = time.time() if now is None else now
        if self.started_at is None:
            return
        alive = [any(not self.players[sid]["out"] for sid in team["members"]) for team in self.teams]
        if not alive[0] or not alive[1]:
            self.end_game(1 if alive[1] else 0 if alive[0] else None, now)
        elif now - self.started_at > self.game_seconds:
            self.end_game(None, now)

    # --- что сказать ботам -------------------------------------------------------

    def waiting(self, session_id: int) -> bool:
        """Бот стоит и опыта не пишет: игры нет (пауза), выбыл, ждёт
        возрождения, телепорт ещё не дошёл."""
        player = self.players.get(session_id)
        return (self.started_at is None or player is None or player["out"] or not player["ready"]
                or player.get("respawn_at") is not None)

    def enemy_ids(self, session_id: int) -> list[int]:
        player = self.players.get(session_id)
        if player is None:
            return []
        return [p["entity_id"] for p in self.players.values()
                if p["team"] != player["team"] and not p["out"] and p["entity_id"] is not None]

    def target_for(self, session_id: int) -> dict | None:
        """Враг ближе fight_radius — бить его; иначе чужая кровать; кровати нет —
        ближайший враг (сущность; x/y/z — запасная позиция)."""
        player = self.players.get(session_id)
        if player is None or player["pos"] is None:
            return None
        enemies = [p for p in self.players.values()
                   if p["team"] != player["team"] and not p["dead"] and not p["out"] and p["pos"] is not None
                   and p["entity_id"] is not None]
        nearest = min(enemies, key=lambda p: math.dist(p["pos"], player["pos"]), default=None)
        enemy_team = self.teams[1 - player["team"]]
        if nearest is not None and (math.dist(nearest["pos"], player["pos"]) <= self.fight_radius
                                    or not enemy_team["bed_alive"]):
            x, y, z = nearest["pos"]
            return {"entity_id": nearest["entity_id"], "x": x, "y": y, "z": z}
        if enemy_team["bed_alive"]:
            x, y, z = enemy_team["bed"]["head"]
            return {"x": x + 0.5, "y": y + 0.5, "z": z + 0.5}
        return None

    def team_of(self, session_id: int) -> int | None:
        player = self.players.get(session_id)
        return None if player is None else player["team"]

    def take_events(self, session_id: int) -> list[str]:
        return self.events.pop(session_id, [])

    def take_commands(self) -> list[str]:
        commands, self.commands = self.commands, []
        return commands

    def describe(self) -> str:
        if self.map is None:
            return "бедварс: игры ещё не было"
        teams = ", ".join(f"{t['color']} ({len(t['members'])}, кровать {'цела' if t['bed_alive'] else 'сломана'})"
                          for t in self.teams)
        return f"бедварс: {self.map['name']} — {teams}"

    # --- мелочи ---------------------------------------------------------------

    def _event(self, session_id: int, event: str) -> None:
        self.events.setdefault(session_id, []).append(event)

    def _by_entity(self, entity_id) -> int | None:
        return next((sid for sid, p in self.players.items() if p["entity_id"] == entity_id), None)
