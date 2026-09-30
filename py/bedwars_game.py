"""Судья задачки bedwars — свой бедварс для ботов (идея автора: "боты
против ботов — бесконечные игры") на картах Hypixel (server/bw_maps,
описания — data/bedwars/maps/<карта>.json, py/bedwars_maps.py).

Матч — две команды (автор, 2026-09-30: "обучение и так даётся медленно")
на двух соседних островах одной стороны карты (мост между ними прямой;
straight_pairs: false — любые соседи, и через угол). Рой делится на матчи по
team_size в команде — все на одной случайной карте, на разных парах островов.
Набор — деревянный меч; шерсть — за железо с генератора своего острова
(экономика ниже; start_blocks > 0 — старый режим, шерсть даром).

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

Сценарий каждой новой "игры" — по весам modules.bedwars.scenarios: бедварс
на карте (GAME) или упражнение (py/bedwars_drills.py; автор, 2026-09-30:
"сначала нужно было оттачивать механики"):
  - DUEL — дуэли 1 на 1 над пустотой, у каждой пары своя дорожка. Та же
    игра, только без кроватей: упал или погиб — выбыл, матч кончен (победа —
    DUEL_WON), через DUEL_PAUSE — реванш на новой площадке; раунд дольше
    duel_seconds — ничья.
  - EDGE — ходьба по краю: у каждого бота своя тропа над пустотой, цель —
    её конец (GOAL — и сразу новая тропа); упал — смерть, через
    drill_respawn_seconds — новая тропа; дольше edge_seconds — тоже новая.
Упражнение длится drill_seconds, потом — новый сценарий. Упражнения — в
своём мире (bot.task_worlds.drills; без него — только игры), дорожки строит
судья командами fill.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

from bedwars_drills import DuelLayout, EdgeLayout, clear_fill, drill_command, void_y as drill_void_y
from bridge_course import fill_command

BEDWARS = "bedwars"
WON = "won"          # исход эпизода: команда победила (терминально)
LOST = "lost"        # команда проиграла — у выбывших эпизод кончился смертью
BED = "bed"          # сломал чужую кровать
BED_LOST = "bed_lost"  # сломали твою кровать
KILL = "kill"        # убил врага (последний, кто его ударил)
BOUGHT = "bought"    # купил шерсть
TIME_UP = "time_up"  # игра не кончилась за game_seconds — ничья, новая игра (обрыв, не конец эпизода)
FELL = "fell"        # упал в пустоту (для сводки: смерть считает сама задачка)
DUEL_WON = "duel_won"    # упражнение: выиграл дуэль (терминально)
DUEL_LOST = "duel_lost"  # проиграл дуэль, оставшись живым (напарник выбыл последним; терминально)
GOAL = "goal"        # упражнение: дошёл по тропе до цели (терминально)
TERMINAL_EVENTS = (WON, LOST, DUEL_WON, DUEL_LOST, GOAL)  # исходы — конец эпизода (ai_loop._end_episode)
ATTACK = "attack"    # роль: к чужой кровати
DEFEND = "defend"    # роль: закрыть свою кровать и стоять у неё
GAME, DUEL, EDGE = "game", "duel", "edge"  # сценарии: бедварс на карте и упражнения
SCENARIO_NAMES = {GAME: "игра", DUEL: "дуэли", EDGE: "ходьба по краю"}
# Цвета команд (bedwars_maps.TEAM_COLORS) -> цвета /team сервера.
TEAM_COLOR = {"red": "red", "blue": "blue", "lime": "green", "yellow": "yellow", "cyan": "aqua", "white": "white",
              "pink": "light_purple", "gray": "gray"}
DUEL_COLORS = ("red", "blue")

MAPS_DIR = Path(__file__).resolve().parent.parent / "data" / "bedwars" / "maps"
READY_DISTANCE = 1.5   # бот ближе стольки к своей точке появления — телепорт дошёл
RESEND_SECONDS = 2.0   # телепорт не дошёл за столько — ещё раз
KILL_CREDIT_SECONDS = 10.0  # убийство засчитывается тому, кто ударил последним не раньше стольких секунд
BED_CHECK_DELAY = 3.0       # секунд с начала игры, пока кровати не проверяются
JOIN_SECONDS = 2.0          # первая игра — через столько после первого бота (ждём остальных)
DUEL_PAUSE = 1.0            # дуэль кончилась — реванш через столько


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
        self.start_blocks = cfg.get("start_blocks", 0)
        self.fight_radius = cfg.get("fight_radius", 6.0)
        self.guard_radius = cfg.get("guard_radius", 10.0)
        self.straight_pairs = cfg.get("straight_pairs", True)
        # Защитник у команды — не в каждой игре: когда он есть у обеих, атакующие
        # сходятся на мостах, победителя добивает свежий защитник, и кровати
        # не ломал никто (учителя 4 на 4, 2026-09-30: 0 кроватей за 8 минут) —
        # сеть не видела бы награды за кровать вовсе.
        self.defend_chance = cfg.get("defend_chance", 0.5)
        # Игроков в команде: рой делится на матчи — пары команд на разных парах
        # островов одной карты (команды 0-1, 2-3...). 4 на 4 на одном мосту шли
        # вничью: атакующие гуськом сходились в середине моста, и до кровати не
        # доходил никто; 2 на 2 кровати ломают (учителя, 2026-09-30).
        self.team_size = max(1, cfg.get("team_size", 2))
        self.respawn_seconds = cfg.get("respawn_seconds", 5.0)
        # Экономика: генератор у точки появления команды, магазин там же.
        self.iron_seconds = cfg.get("iron_seconds", 0.3)
        self.gold_seconds = cfg.get("gold_seconds", 4.0)
        self.iron_cap = cfg.get("iron_cap", 64)
        self.gold_cap = cfg.get("gold_cap", 16)
        self.wool_price = cfg.get("wool_price", 4)
        self.wool_amount = cfg.get("wool_amount", 16)
        self.shop_radius = cfg.get("shop_radius", 2.5)
        # Упражнения вперемешку с играми: сценарий каждой новой "игры" — по
        # весам (0 — никогда; по умолчанию — только игры на картах).
        self.scenarios = {name: weight for name, weight in cfg.get("scenarios", {GAME: 1.0}).items()
                          if name in SCENARIO_NAMES and weight > 0} or {GAME: 1.0}
        self.drill_seconds = cfg.get("drill_seconds", 120.0)
        self.duel_seconds = cfg.get("duel_seconds", 30.0)
        self.edge_seconds = cfg.get("edge_seconds", 40.0)
        self.drill_respawn_seconds = cfg.get("drill_respawn_seconds", 1.0)
        self.scenario = GAME
        worlds = config["bot"].get("task_worlds", {})
        self.world = worlds.get(BEDWARS)
        self.drill_world = worlds.get("drills")
        if self.world is not None and self.drill_world is None and set(self.scenarios) - {GAME}:
            # Без своего мира дорожки строились бы в мире карт (или в обычном).
            print("[bedwars] Упражнения выключены: нет мира упражнений (config.json bot.task_worlds.drills).")
            self.scenarios = {GAME: 1.0}
        self.map: dict | None = None      # описание карты (json)
        # Команды: color, bed, spawn, bed_alive, members, done (матч кончился);
        # соперник команды i — команда i ^ 1 (матчи — пары 0-1, 2-3...).
        self.teams: list[dict] = []
        self.players: dict[int, dict] = {}  # id сессии -> team, entity_id, pos, dead, out, ready...
        self.started_at: float | None = None
        self.next_game_at = 0.0           # когда начать следующую игру (после паузы)
        self.first_seen: float | None = None  # когда судья впервые увидел ботов
        # Команды /team прошлой игры; при первой игре — все возможные (могли
        # остаться от прошлого запуска), дальше — только свои.
        self.team_names = [f"bw_{color}" for color in TEAM_COLOR]
        self.events: dict[int, list[str]] = {}
        self.commands: list[str] = []
        self.stats = {"games": 0, "decided": 0, "beds": 0, "kills": 0, "time_up": 0, "duels": 0, "goals": 0}

    # --- начало и конец игры ---------------------------------------------------

    def want_new_game(self, now: float | None = None) -> bool:
        """Пора начинать игру. Первую — через JOIN_SECONDS после первого бота:
        боты подключаются по одному, и игра без половины роя шла бы без них."""
        now = time.time() if now is None else now
        if self.first_seen is None:
            self.first_seen = now
        return self.started_at is None and now >= max(self.next_game_at, self.first_seen + JOIN_SECONDS)

    def new_game(self, session_ids: list[int], name_of, now: float | None = None) -> None:
        """Новая "игра": сценарий по весам scenarios — бедварс на карте или
        упражнение. Команды сервера — в self.commands."""
        now = time.time() if now is None else now
        names = list(self.scenarios)
        self.scenario = self.rng.choices(names, [self.scenarios[name] for name in names])[0]
        if self.scenario == GAME:
            self._new_map_game(session_ids, name_of)
        else:
            self._new_drill(session_ids, name_of, now)
        self.started_at = now
        self.stats["games"] += 1
        self._setup_server_teams()
        for session_id in self.players:
            self._spawn(session_id, now, kit=True)

    def _new_map_game(self, session_ids: list[int], name_of) -> None:
        """Бедварс на случайной карте: пары соседних островов, игроки — по
        очереди в команды (перемешаны)."""
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
        # Матчи — на непересекающихся парах островов, сколько влезет на карту.
        wanted = max(1, len(session_ids) // (2 * self.team_size))
        self.rng.shuffle(pairs)
        chosen, used = [], set()
        for first, second in pairs:
            if len(chosen) < wanted and first not in used and second not in used:
                chosen.append((first, second))
                used.update((first, second))
        self.teams = [{"color": teams[i]["color"], "name": f"bw_{teams[i]['color']}", "bed": teams[i]["bed"],
                       "spawn": teams[i]["spawn"], "yaw": None, "bed_alive": True, "members": [], "done": False}
                      for pair in chosen for i in pair]
        ids = list(session_ids)
        self.rng.shuffle(ids)
        self.players = {}
        for index, session_id in enumerate(ids):
            team = index % len(self.teams)
            self.teams[team]["members"].append(session_id)
            self.players[session_id] = {"team": team, "role": ATTACK, "name": name_of(session_id), "entity_id": None,
                                        "pos": None, "dead": False, "out": False, "ready": False, "sent": 0.0,
                                        "states": 0, "hit_by": None, "hit_at": 0.0}
        for team in self.teams:
            # Первый в команде (если в ней двое и больше) — защитник: закрыть
            # свою кровать шерстью и стоять у неё (автор: "даже кровать не
            # защищают"); остальные — в атаку. Защитник есть не всегда
            # (defend_chance): иначе кровати почти не ломаются.
            if len(team["members"]) >= 2 and self.rng.random() < self.defend_chance:
                self.players[team["members"][0]]["role"] = DEFEND
        # Вечно загруженные чанки (прошлые forceload — например, перевод карт,
        # bedwars_maps.py --convert) переживают смену карты, и каждая загрузка
        # мира заново их грузит — сервер подвисал. Карте они не нужны.
        self.commands.append(self._in_world("forceload remove all"))
        self.commands.append(f"mcbot bedwars {name}")
        # Чанки кроватей — загружены всю игру: иначе проверка "цела ли кровать"
        # (RCON: execute if block) в незагруженном чанке не знает ответа.
        for team in self.teams:
            for x, _, z in (team["bed"]["head"], team["bed"]["foot"]):
                self.commands.append(self._in_world(f"forceload add {x} {z}"))

    def _new_drill(self, session_ids: list[int], name_of, now: float) -> None:
        """Упражнение (py/bedwars_drills.py) в мире упражнений: у каждой пары
        (дуэль) или бота (край) — своя дорожка. Команда в дуэли — один игрок
        (нечётный — вторым в последнюю команду: двое на одного), у тропы — один бот."""
        ids = list(session_ids)
        self.rng.shuffle(ids)
        lanes = max(1, len(ids) // 2) if self.scenario == DUEL else len(ids)
        self.map = {"name": SCENARIO_NAMES[self.scenario], "void_y": drill_void_y()}
        self.commands.append(drill_command(lanes))  # участок дорожек — загружен (fill — только в загруженных чанках)
        self.teams = []
        for lane in range(lanes):
            if self.scenario == DUEL:
                layout = DuelLayout(lane, self.rng)
                for side, color in enumerate(DUEL_COLORS):
                    self.teams.append(self._drill_team(color, f"bw_{color}{lane}", lane, layout, layout.spawns[side], now))
            else:
                layout = EdgeLayout(lane, self.rng)
                self.teams.append(self._drill_team("white", None, lane, layout, layout.start(), now))
            self._build_lane(lane, layout)
        self.players = {}
        for index, session_id in enumerate(ids):
            # Команд — по игроку (дуэль: две на пару); нечётный в дуэлях — вторым
            # в последнюю команду (двое на одного).
            team = min(index, len(self.teams) - 1)
            self.teams[team]["members"].append(session_id)
            self.players[session_id] = {"team": team, "role": ATTACK, "name": name_of(session_id), "entity_id": None,
                                        "pos": None, "dead": False, "out": False, "ready": False, "sent": 0.0,
                                        "states": 0, "hit_by": None, "hit_at": 0.0}

    @staticmethod
    def _drill_team(color: str, name: str | None, lane: int, layout, spawn: tuple, now: float) -> dict:
        """Команда упражнения: без кровати (выбыл — значит выбыл), своя
        дорожка и раскладка; spawn — (x, y, z, угол сервера) из раскладки."""
        x, y, z, yaw = spawn
        return {"color": color, "name": name, "bed": None, "spawn": [math.floor(x), y, math.floor(z)], "yaw": yaw,
                "bed_alive": False, "members": [], "done": False, "lane": lane, "layout": layout,
                "round_at": now, "rematch_at": None}

    def _build_lane(self, lane: int, layout) -> None:
        """Дорожка заново: всё прочь (постройки и обломки прошлого раунда),
        площадка раскладки — на место."""
        for fill in [clear_fill(lane)] + layout.fills():
            self.commands.append(fill_command(fill, self._scenario_world()))

    def _setup_server_teams(self) -> None:
        """Команды сервера (/team): ник цветом своей команды, по своим не бьёшь —
        и людям видно, кто за кого (автор: "рассыпались по командам")."""
        for name in self.team_names:  # команды прошлой игры (удалять несуществующие — шум ошибок в логе)
            self.commands.append(f"team remove {name}")
        teams = [team for team in self.teams if team["name"] is not None]
        self.team_names = [team["name"] for team in teams]
        for team in teams:
            name = team["name"]
            self.commands += [f"team add {name}", f"team modify {name} color {TEAM_COLOR[team['color']]}",
                              f"team modify {name} friendlyFire false"]
            # По одному: /team join в новых версиях берёт одно имя (или селектор).
            self.commands += [f"team join {name} {self.players[sid]['name']}" for sid in team["members"]]

    def _scenario_world(self) -> str | None:
        """Мир нынешнего сценария: карты — мир бедварса, упражнения — свой."""
        return self.world if self.scenario == GAME else self.drill_world

    def _in_world(self, command: str) -> str:
        world = self._scenario_world()
        return f"execute in minecraft:{world} run {command}" if world else command

    def _spawn(self, session_id: int, now: float, kit: bool) -> None:
        """На свою точку появления, лицом к центру карты; набор — заново (меч)."""
        player = self.players[session_id]
        team = self.teams[player["team"]]
        x, y, z = team["spawn"]
        # Углы сервера: взгляд (-sin yaw, 0, cos yaw), yaw 0 — на юг (+z). Кратно
        # 10°: голова поворачивает шагами по 10°, и с другого угла ровно по оси
        # моста не встать (учитель мостом дёргался между ±5°). На карте — лицом
        # к её середине, в упражнениях — как задала раскладка.
        yaw = team["yaw"] if team.get("yaw") is not None else 10 * round(math.degrees(math.atan2(x, -z)) / 10)
        name = player["name"]
        commands = []
        if kit:
            # Как на Hypixel: только деревянный меч; блоки — за железо с генератора
            # (автор: "насильно выдаёшь ресурсы"). start_blocks > 0 — старый режим.
            commands += [f"gamemode survival {name}", f"clear {name}", f"give {name} minecraft:wooden_sword 1"]
            if self.start_blocks > 0:
                commands.append(f"give {name} minecraft:{team['color']}_wool {self.start_blocks}")
        commands.append(self._in_world(f"tp {name} {x + 0.5} {y} {z + 0.5} {yaw:.0f} 0"))
        self.commands += commands
        player.update(ready=False, sent=now, states=0, fell=False)

    def end_match(self, team_index: int, winner: int | None, now: float | None = None) -> None:
        """Конец матча команды team_index и её соперника: победителям — WON,
        проигравшим — LOST (кто ещё жив — тоже: игра для них кончилась),
        ничья — TIME_UP. Игроки кончившегося матча ждут конца игры (waiting);
        в дуэлях — реванша через DUEL_PAUSE."""
        now = time.time() if now is None else now
        match = (team_index & ~1, team_index | 1)
        won, lost = (DUEL_WON, DUEL_LOST) if self.scenario == DUEL else (WON, LOST)
        for index in match:
            team = self.teams[index]
            team["done"] = True
            team["rematch_at"] = now + DUEL_PAUSE
            for session_id in team["members"]:
                player = self.players[session_id]
                if winner is None:
                    self._event(session_id, TIME_UP)
                elif not player["out"] or index == winner:
                    self._event(session_id, won if index == winner else lost)
                if not player["out"]:
                    # До новой игры — зритель: иначе победители стояли столбом,
                    # пока доигрывают другие матчи (автор: "просто стоят и тупят").
                    self.commands.append(f"gamemode spectator {player['name']}")
        self.stats["time_up" if winner is None else "decided"] += 1
        if self.scenario == DUEL:
            self.stats["duels"] += 1

    def end_game(self, now: float | None = None) -> None:
        """Конец игры: матчи, что ещё идут, — ничья (тропы — тоже); через
        паузу — новая игра."""
        now = time.time() if now is None else now
        if self.scenario == EDGE:
            for team in self.teams:
                for session_id in team["members"]:
                    self._event(session_id, TIME_UP)
                    self.commands.append(f"gamemode spectator {self.players[session_id]['name']}")
        else:
            for index in range(0, len(self.teams), 2):
                if not self.teams[index]["done"]:
                    self.end_match(index, None, now)
        self.started_at = None
        self.next_game_at = now + self.pause

    def _rematch(self, team_index: int, now: float) -> None:
        """Дуэль: реванш той же пары на новой площадке своей дорожки."""
        first, second = self.teams[team_index & ~1], self.teams[team_index | 1]
        layout = DuelLayout(first["lane"], self.rng)
        self._build_lane(first["lane"], layout)
        for team, spawn in ((first, layout.spawns[0]), (second, layout.spawns[1])):
            x, y, z, yaw = spawn
            team.update(spawn=[math.floor(x), y, math.floor(z)], yaw=yaw, layout=layout, done=False,
                        round_at=now, rematch_at=None)
            for session_id in team["members"]:
                self.players[session_id].update(out=False, respawn_at=None, hit_by=None)
                self._spawn(session_id, now, kit=True)

    def _new_edge_attempt(self, team_index: int, now: float, kit: bool = False) -> None:
        """Ходьба по краю: новая тропа на дорожке бота, бот — на её старт."""
        team = self.teams[team_index]
        layout = EdgeLayout(team["lane"], self.rng)
        self._build_lane(team["lane"], layout)
        x, y, z, yaw = layout.start()
        team.update(spawn=[math.floor(x), y, math.floor(z)], yaw=yaw, layout=layout, round_at=now)
        for session_id in team["members"]:
            self._spawn(session_id, now, kit=kit)

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
            if victim is not None and self.players[victim]["team"] == player["team"] ^ 1:
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
            if self.scenario == EDGE:
                self._new_edge_attempt(player["team"], now, kit=True)  # упал — новая тропа
            else:
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
        if self.scenario == EDGE and self.teams[player["team"]]["layout"].on_goal(*player["pos"]):
            self._event(session_id, GOAL)  # дошёл — и сразу новая тропа
            self.stats["goals"] += 1
            self._new_edge_attempt(player["team"], now)
            return
        if me["y"] < self.map["void_y"]:
            if not player.get("fell"):
                player["fell"] = True
                self._event(session_id, FELL)
            self.commands.append(f"kill {player['name']}")  # в пустоту — умер, не дожидаясь дна мира

    def _died(self, session_id: int, now: float) -> None:
        player = self.players[session_id]
        killer = player["hit_by"] if now - player["hit_at"] < KILL_CREDIT_SECONDS else None
        player.update(hit_by=None, ready=False, pos=None)
        if killer is not None:
            self._event(killer, KILL)
            self.stats["kills"] += 1
        if self.scenario == EDGE:
            # Ходьба по краю: упал — через drill_respawn_seconds на новую тропу.
            player["respawn_at"] = now + self.drill_respawn_seconds
            self.commands.append(f"gamemode spectator {player['name']}")
            return
        if self.teams[player["team"]]["bed_alive"]:
            # Возрождение — через respawn_seconds (как в бедварсе: до того
            # зритель), на свой остров и снова с набором (вещи при смерти
            # выпадают). Сразу назад в бой — и драка на мосту шла без конца.
            player["respawn_at"] = now + self.respawn_seconds
            self.commands.append(f"gamemode spectator {player['name']}")
            return
        player["out"] = True
        self.commands.append(f"gamemode spectator {player['name']}")

    # --- экономика: генератор и магазин -------------------------------------------

    def economy_tick(self, now: float | None = None) -> None:
        """Генераторы: у каждой команды куча железа и золота у точки появления
        растёт (iron_seconds, gold_seconds; не больше iron_cap, gold_cap)."""
        now = time.time() if now is None else now
        if self.started_at is None or self.scenario != GAME:
            return
        for team in self.teams:
            last = team.setdefault("economy_at", now)
            team["iron_timer"] = team.get("iron_timer", 0.0) + now - last
            team["gold_timer"] = team.get("gold_timer", 0.0) + now - last
            team["economy_at"] = now
            while team["iron_timer"] >= self.iron_seconds:
                team["iron_timer"] -= self.iron_seconds
                team["iron"] = min(team.get("iron", 0) + 1, self.iron_cap)
            while team["gold_timer"] >= self.gold_seconds:
                team["gold_timer"] -= self.gold_seconds
                team["gold"] = min(team.get("gold", 0) + 1, self.gold_cap)

    def at_shop(self, session_id: int) -> bool:
        """Стоит у своей точки появления — там генератор и магазин."""
        player = self.players.get(session_id)
        if (self.scenario != GAME or player is None or player["pos"] is None or player["dead"] or player["out"]
                or not player["ready"]):
            return False
        x, y, z = self.teams[player["team"]]["spawn"]
        return math.dist(player["pos"], (x + 0.5, y, z + 0.5)) <= self.shop_radius

    def collect_and_buy(self, session_id: int, state: dict, hands_action: str | None) -> None:
        """У своего генератора — забрать его кучу (give); решил купить (действие
        рук buy) и хватает железа — шерсть своего цвета за железо (clear/give).
        Магазин без меню: бот не умеет кликать по окнам, и покупка — одно действие."""
        if not self.at_shop(session_id):
            return
        player = self.players[session_id]
        team = self.teams[player["team"]]
        name = player["name"]
        iron = (state.get("inventory_items") or {}).get("iron_ingot", 0)
        player["iron"] = iron
        # Куча — тому из команды у генератора, у кого железа меньше всех: иначе
        # её каждый раз забирал один (защитник без шерсти ждал вечно).
        poorest = min((self.players[sid].get("iron", 0) for sid in team["members"] if self.at_shop(sid)), default=iron)
        if iron <= poorest:
            for item, key in (("iron_ingot", "iron"), ("gold_ingot", "gold")):
                if team.get(key, 0) > 0:
                    self.commands.append(f"give {name} minecraft:{item} {team[key]}")
                    team[key] = 0
        if hands_action == "buy" and iron >= self.wool_price:
            self.commands += [f"clear {name} minecraft:iron_ingot {self.wool_price}",
                              f"give {name} minecraft:{team['color']}_wool {self.wool_amount}"]
            self._event(session_id, BOUGHT)

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
                   if p["team"] == team_index ^ 1 and not p["dead"] and not p["out"] and p["pos"] is not None]
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
        """Победа в матче (у соперника все выбыли); все матчи кончились или
        время вышло — конец игры."""
        now = time.time() if now is None else now
        if self.started_at is None:
            return
        if self.scenario == EDGE:
            for index, team in enumerate(self.teams):
                ready = any(self.players[sid]["ready"] for sid in team["members"])
                if ready and now - team["round_at"] > self.edge_seconds:
                    for session_id in team["members"]:
                        self._event(session_id, TIME_UP)  # не дошёл за edge_seconds — новая тропа (обрыв)
                    self._new_edge_attempt(index, now)
            if now - self.started_at > self.drill_seconds:
                self.end_game(now)
            return
        alive = [any(not self.players[sid]["out"] for sid in team["members"]) for team in self.teams]
        for index in range(0, len(self.teams), 2):
            team = self.teams[index]
            if team["done"]:
                if self.scenario == DUEL and now >= team["rematch_at"]:
                    self._rematch(index, now)
                continue
            if not (alive[index] and alive[index + 1]):
                self.end_match(index, index if alive[index] else index + 1 if alive[index + 1] else None, now)
            elif self.scenario == DUEL and now - team["round_at"] > self.duel_seconds:
                self.end_match(index, None, now)  # дуэль затянулась — ничья и реванш
        if self.scenario == DUEL:
            if now - self.started_at > self.drill_seconds:
                self.end_game(now)
        elif all(team["done"] for team in self.teams) or now - self.started_at > self.game_seconds:
            self.end_game(now)

    # --- что сказать ботам -------------------------------------------------------

    def waiting(self, session_id: int) -> bool:
        """Бот стоит и опыта не пишет: игры нет (пауза), выбыл, ждёт
        возрождения, телепорт ещё не дошёл."""
        player = self.players.get(session_id)
        return (self.started_at is None or player is None or player["out"] or not player["ready"]
                or player.get("respawn_at") is not None or self.teams[player["team"]]["done"])

    def enemy_ids(self, session_id: int) -> list[int]:
        player = self.players.get(session_id)
        if player is None or self.scenario == EDGE:
            return []
        return [p["entity_id"] for p in self.players.values()
                if p["team"] == player["team"] ^ 1 and not p["out"] and p["entity_id"] is not None]

    def target_for(self, session_id: int) -> dict | None:
        """Атакующему: враг ближе fight_radius — бить его; иначе чужая кровать;
        кровати нет — ближайший враг. Защитнику: враг у своей кровати (ближе
        guard_radius) или рядом с ним — бить; иначе своя кровать (стоять у
        неё). Враг — сущность (x/y/z — запасная позиция)."""
        player = self.players.get(session_id)
        if player is None or player["pos"] is None:
            return None
        if self.scenario == EDGE:
            x, y, z = self.teams[player["team"]]["layout"].goal()  # ходьба по краю: конец тропы
            return {"x": x, "y": y, "z": z}
        enemies = [p for p in self.players.values()
                   if p["team"] == player["team"] ^ 1 and not p["dead"] and not p["out"] and p["pos"] is not None
                   and p["entity_id"] is not None]
        nearest = min(enemies, key=lambda p: math.dist(p["pos"], player["pos"]), default=None)
        enemy_team = self.teams[player["team"] ^ 1]
        own_team = self.teams[player["team"]]
        if player["role"] == DEFEND and own_team["bed_alive"]:
            bed = own_team["bed"]["head"]
            threats = [p for p in enemies if math.dist(p["pos"], bed) <= self.guard_radius
                       or math.dist(p["pos"], player["pos"]) <= self.fight_radius]
            threat = min(threats, key=lambda p: math.dist(p["pos"], player["pos"]), default=None)
            if threat is not None:
                x, y, z = threat["pos"]
                return {"entity_id": threat["entity_id"], "x": x, "y": y, "z": z}
            return {"x": bed[0] + 0.5, "y": bed[1] + 0.5, "z": bed[2] + 0.5}
        if nearest is not None and (math.dist(nearest["pos"], player["pos"]) <= self.fight_radius
                                    or not enemy_team["bed_alive"]):
            x, y, z = nearest["pos"]
            return {"entity_id": nearest["entity_id"], "x": x, "y": y, "z": z}
        if enemy_team["bed_alive"]:
            x, y, z = enemy_team["bed"]["head"]
            return {"x": x + 0.5, "y": y + 0.5, "z": z + 0.5}
        return None

    def role_of(self, session_id: int) -> str | None:
        player = self.players.get(session_id)
        return None if player is None else player["role"]

    def own_bed(self, session_id: int) -> dict | None:
        """Своя кровать ({"head", "foot"}) — защитнику (учитель закрывает её шерстью)."""
        player = self.players.get(session_id)
        return None if player is None else self.teams[player["team"]]["bed"]

    def own_spawn(self, session_id: int) -> list | None:
        """Своя точка появления (там генератор и магазин) — учителю: за шерстью домой."""
        player = self.players.get(session_id)
        return None if player is None else self.teams[player["team"]]["spawn"]

    def enemy_bed(self, session_id: int) -> dict | None:
        """Чужая кровать, пока цела, — учителю: мост строить к ней, даже когда
        цель на время — враг."""
        player = self.players.get(session_id)
        if player is None or self.scenario != GAME:
            return None
        team = self.teams[player["team"] ^ 1]
        return team["bed"] if team["bed_alive"] else None

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
        if self.scenario != GAME:
            layouts = {team["lane"]: team["layout"].describe() for team in self.teams}
            return f"бедварс, упражнение — {self.map['name']}: " + "; ".join(layouts.values())
        teams = [f"{t['color']} ({len(t['members'])}, кровать {'цела' if t['bed_alive'] else 'сломана'})"
                 for t in self.teams]
        matches = "; ".join(f"{teams[i]} против {teams[i + 1]}" for i in range(0, len(teams), 2))
        return f"бедварс: {self.map['name']} — {matches}"

    # --- мелочи ---------------------------------------------------------------

    def _event(self, session_id: int, event: str) -> None:
        self.events.setdefault(session_id, []).append(event)

    def _by_entity(self, entity_id) -> int | None:
        return next((sid for sid, p in self.players.items() if p["entity_id"] == entity_id), None)
