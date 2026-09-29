"""Салки "заражение" (идея автора): один ВОДИТ, остальные убегают. Водящий
догнал и ударил убегающего — тот тоже водит (а водящий водит дальше), и так
до последнего. Последнего осалили — "Ура, все молодцы, ещё раз!": всех
разбрасывает по арене, и снова водит один случайный.

Роли — две задачки со своими мозгами: chase (водящий) и flee
(убегающие), py/training_modules/chase.py и flee.py. TagGame — судья: кто
водит, засчитан ли удар, когда раунд кончился. Модулям он сообщает только
цель (водящему — за кем бежать, убегающему — ближайший водящий) и исход,
награды считают сами модули.

Правила:
  - раунд начинается с одного случайного водящего: он freeze_seconds
    стоит ("считает до пяти"), остальные разбегаются;
  - осалить = ударить убегающего, стоя к нему ближе tag_distance (кого
    ударил бот, Node сообщает в состоянии водящего — attacked_id);
    осаленный тоже водит и тоже сначала считает до пяти;
  - осалили последнего — раунд выигран: водящим награда (исход WON), пауза
    RESTART_PAUSE секунд (ни решений, ни опыта), всех разбрасывает по
    арене (/spreadplayers — телепорт, а не смерть: наказывать не за что),
    новый раунд;
  - раунд идёт дольше round_seconds — время вышло: кого не поймали, те
    молодцы (исход SURVIVED), дальше так же — пауза, разброс, новый раунд;
  - только что возродившийся (все возрождаются в одной точке мира)
    respawn_seconds не салит и не салится.
Сколько водящих — не настройка: сколько осалили, столько и водят.

Люди играют, если они в одной из команд сервера салок — TEAM_IT (водят)
или TEAM_RUNNERS (убегают): "/team join runners <ник>" — и в выживании или
приключении; наблюдателя и творческий режим водящие не трогают. Людей
видят боты (state.humans: ник, id сущности, позиция, режим игры, команда —
js/actions.js: visibleHumans). Роль человека — его команда, а сразу после
того, как судья сам его перевёл (осалили, новый раунд), — то, куда перевёл:
команда доходит до ботов не мгновенно. Удар бота по человеку судья видит,
как и по боту (attacked_id), а удар человека-водящего по боту — по урону у
самого бота (hurt_by: id того, кто ударил).

Водящий пишет в чат "Я догнал AI_3. Теперь и он водит!" (судья складывает
такие сообщения в messages, ai_loop отправляет). Если mark_roles — судья
ещё и помечает роли командами сервера: TEAM_IT — красный ник с приставкой
[водит], TEAM_RUNNERS — белые, боты светятся сквозь стены своим цветом
(/effect glowing) — видно, даже если бот ушёл в пещеру. С нашим
сервером-ареной команды идут по RCON сразу (take_commands); без него — от
имени бота command_bot (ему нужен op) очередью, не чаще раза в
command_interval: сервер кикает за флуд всех, кроме op.

Где кто, судья знает из состояний самих ботов (позиция, id сущности на
сервере — он у всех клиентов один и тот же), поэтому цель известна, даже
когда сервер её боту не показывает (далеко).

Смена роли применяется на СЛЕДУЮЩЕМ тике этого бота (before_tick): тогда
ai_loop закрывает последний переход старой роли и продолжает уже новой.
"""

from __future__ import annotations

import math
import random
import time
from collections import deque

from training_modules.targets import bot_name, plural

CHASE, FLEE = "chase", "flee"  # водящий, убегающий

# Исходы — чем кончился эпизод роли (награду за исход добавляет модуль).
CAUGHT = "caught"      # водящему — осалил, убегающему — осалили
WON = "won"            # водящим — осалили последнего, раунд выигран
SURVIVED = "survived"  # убегающему — время раунда вышло, а его не поймали
TIME_UP = "time_up"    # водящему — время вышло, поймали не всех
# Конец эпизода — переход закрывается без бутстрапа. TIME_UP — обрыв, а не
# конец: время раунда сети водящего не видно, и "дальше ничего не будет"
# было бы для неё неправдой.
TERMINAL_OUTCOMES = (CAUGHT, WON, SURVIVED)

# Команды сервера салок: id (его набирают: /team join runners <ник>) и
# название с цветом (видно в /team list, цвет — у ника и свечения).
TEAM_IT = "it"
TEAM_RUNNERS = "runners"
TEAMS = {TEAM_IT: ("Водящие", "red"), TEAM_RUNNERS: ("Убегающие", "white")}
TEAM_OF_ROLE = {CHASE: TEAM_IT, FLEE: TEAM_RUNNERS}
ROLE_OF_TEAM = {TEAM_IT: CHASE, TEAM_RUNNERS: FLEE}

TAG_POINTS = 10       # очков салок за осаливание (и за то, что не поймали за раунд)
RESTART_PAUSE = 3.0   # секунд между раундами: все стоят, опыта нет
# Разброс по арене — не сразу в конце раунда, а через секунду: сначала
# каждый бот закроет эпизод (тик — 150 мс), иначе телепорт попал бы в
# награду его последнего шага ("убежал на 30 блоков за тик").
SPREAD_DELAY = 1.0
SPREAD_DISTANCE = 4   # /spreadplayers: не ближе стольких блоков друг к другу
HUMAN_SEEN_SECONDS = 3.0    # человека дольше не видел ни один бот — не играет
HUMAN_ASSIGN_SECONDS = 2.0  # сколько слово судьи о роли человека главнее его команды
PLAYING_GAMEMODES = (0, 2)  # выживание, приключение
# Раз в столько секунд — свечение всем ботам заново (только с RCON):
# страховка к выдаче при входе и после смерти — команда могла прийти, пока
# бот ещё не возродился, а без свечения бота не видно сквозь стены.
GLOW_REFRESH_SECONDS = 60.0


class TagGame:
    def __init__(self, config: dict):
        self.config = config
        cfg = config["modules"].get("tag", {})
        self.tag_distance = cfg.get("tag_distance", 4.0)
        self.freeze_seconds = cfg.get("freeze_seconds", 5.0)
        # Раунд не дольше этого: пока водящие неумелые, последнего
        # убегающего могли бы не поймать никогда.
        self.round_seconds = cfg.get("round_seconds", 300.0)
        self.respawn_seconds = cfg.get("respawn_seconds", 3.0)
        # Бот не присылал состояний дольше этого — считаем, что он вышел
        # (отключился, упал): ждать, пока он поводит, бессмысленно.
        self.stale_seconds = cfg.get("stale_seconds", 5.0)
        # Водящий держится выбранного убегающего, пока другой не станет
        # ближе на столько блоков, — иначе метался бы между двумя равными.
        self.switch_margin = cfg.get("switch_margin", 3.0)
        # Застоявшегося бота (см. training_modules/chase.py: _stillness_watch)
        # судья убивает командой /kill — как автор делал вручную: смерть —
        # сильный сигнал "так не надо", и заодно бот выбирается из ямы.
        self.kill_idle = cfg.get("kill_idle", True)
        self.mark_roles = cfg.get("mark_roles", True)
        self.command_bot = cfg.get("command_bot", 1)
        self.command_interval = cfg.get("command_interval", 1.1)

        # id сессии -> {"entity_id", "pos": (x, y, z), "alive", "alive_since", "seen"}
        self.players: dict[int, dict] = {}
        # Водящие-боты: id сессии -> {"since": когда стал водить,
        # "frozen_until": до когда "считает до пяти", "chasing": за кем бежит
        # (id сессии бота или ник человека)}.
        self.its: dict[int, dict] = {}
        # Люди, которых видят боты: ник -> {"entity_id", "pos", "gamemode",
        # "team", "seen", "assigned", "assigned_until", "frozen_until"}.
        self.humans: dict[str, dict] = {}
        # Раунд: когда начался (None — не идёт: ждём игроков или пауза).
        self.round_started: float | None = None
        self.round_tags = 0
        self.paused_until: float | None = None  # пауза между раундами — до
        self.spread_at: float | None = None     # когда разбросать всех по арене
        # Исходы, которые бот ещё не "услышал": id сессии -> исход.
        self.pending: dict[int, str] = {}
        # (id бота, текст) — в чат игры от имени этого бота.
        self.messages: list[tuple[int, str]] = []
        # Команды сервера (пометка ролей, разброс, /kill) — очередь.
        self.commands: deque[str] = deque()
        self.last_command = -math.inf
        self.teams_ready = False
        # Есть наш сервер-арена (RCON): команды забирает ai_loop сразу
        # (take_commands), без очереди против флуда; можно вести и табло.
        self.direct_commands = False
        # "Очки салок" (табло в списке игроков): +TAG_POINTS за осаливание
        # и тем, кого не поймали за раунд; убегающим +1 каждые points_every секунд.
        self.points: dict[str, int] = {}  # ник -> очки
        self.points_every = cfg.get("points_every_seconds", 10.0)
        self.points_at = 0.0
        self.glow_at = 0.0
        self.tags = 0
        self.rounds = 0
        # Кого позвать в начале каждого раунда (ai_loop: набор команд задачи
        # server/functions/tag.mcfunction — почистить и пополнить инвентарь).
        self.on_round_start = None

    @property
    def it(self) -> int | None:
        """Водящий-бот (если их несколько — первый) — для тестов."""
        return next(iter(self.its), None)

    def role_of(self, session_id: int) -> str:
        return CHASE if session_id in self.its else FLEE

    def before_tick(self, session_id: int, role: str, state: dict, now: float | None = None):
        """Вызывается на каждое состояние бота-участника ДО его обработки.
        role — задачка, которую бот сейчас играет. Возвращает
        (исход прошлой роли или None, новая роль), если боту пора что-то
        менять, иначе None."""
        now = time.time() if now is None else now
        me = state["self"]
        alive = not state.get("dead")
        previous = self.players.get(session_id)
        if not alive:
            alive_since = None
        elif previous is None or previous["alive_since"] is None:
            alive_since = now  # только что вошёл в игру или возродился
        else:
            alive_since = previous["alive_since"]
        self.players[session_id] = {
            "entity_id": me.get("entity_id"),
            "pos": (me["x"], me["y"], me["z"]),
            "alive": alive,
            "alive_since": alive_since,
            "seen": now,
        }
        self._see_humans(state.get("humans") or [], now)
        if previous is None:
            self._mark_role(session_id)  # вошёл в игру
        if alive and (previous is None or previous["alive_since"] is None):
            self._glow(session_id)  # вошёл или возродился: смерть снимает эффекты
        self._drop_stale(now)

        self._run_round(now)
        if self.round_started is not None:
            if session_id in self.its:
                self._check_tag(session_id, state, now)
            else:
                self._check_hurt(session_id, state, now)
        self._award_points(now)
        self._refresh_glow(now)

        if state.get("dead"):
            return None  # мёртвому роль не меняем — дождёмся респавна
        outcome = self.pending.get(session_id)
        desired = self.role_of(session_id)
        if outcome is None and role == desired:
            return None
        self.pending.pop(session_id, None)
        return outcome, desired

    def is_frozen(self, session_id: int, now: float | None = None) -> bool:
        """Бот сейчас стоит: пауза между раундами или он водящий и "считает
        до пяти"."""
        now = time.time() if now is None else now
        if self.paused_until is not None:
            return True
        info = self.its.get(session_id)
        return info is not None and now < info["frozen_until"]

    def taggable_ids(self, session_id: int, now: float | None = None) -> list[int]:
        """Кого боту можно бить (удар = осалил): водящему — убегающих (id
        сущностей ботов и людей), остальным — никого. Node бьёт игроков
        только из этого списка (js/actions.js: isTaggable)."""
        now = time.time() if now is None else now
        if session_id not in self.its or self.is_frozen(session_id, now):
            return []
        return [runner["entity_id"] for runner in self._runners(now).values() if runner["entity_id"] is not None]

    def target_for(self, session_id: int) -> dict | None:
        """Цель для модуля: водящему — убегающий, за которым он бежит,
        убегающим — ближайший водящий. Кроме id сущности — её позиция,
        округлённая до блока: Node подставит её, если сам сущность не видит
        (js/target.js)."""
        me = self.players.get(session_id)
        if me is None or self.round_started is None:
            return None
        now = me["seen"]
        if session_id in self.its:
            other = self._choose_chased(session_id, me, now)
        else:
            its = [it for it in self._its_now(now).values() if it.get("alive", True)]
            other = min(its, key=lambda it: math.dist(it["pos"], me["pos"]), default=None)
        if other is None or other["entity_id"] is None:
            return None
        x, y, z = other["pos"]
        return {"entity_id": other["entity_id"], "x": round(x), "y": round(y), "z": round(z)}

    def punish_idle(self, session_id: int) -> None:
        """Бот застоялся (модуль роли заметил): /kill — вне очереди пометок,
        чтобы не ждать, пока разойдутся команды раскраски."""
        seconds = self.config["modules"].get("tag", {}).get("idle_seconds", 45)
        print(f"[ai] Салки: бот {session_id} стоит на месте {seconds:.0f} с"
              + (" — /kill." if self.kill_idle else "."))
        if self.kill_idle:
            self.commands.appendleft(f"/kill {self._name(session_id)}")

    def take_messages(self, now: float | None = None) -> list[tuple[int, str]]:
        """Что сказать в чат: реплики ботов — сразу, команды сервера — по
        одной, не чаще раза в command_interval (сервер кикает за флуд —
        больше ~10 сообщений подряд — всех, кроме op)."""
        now = time.time() if now is None else now
        messages, self.messages = self.messages, []
        if not self.direct_commands and self.commands and now - self.last_command >= self.command_interval:
            messages.append((self.command_bot, self.commands.popleft()))
            self.last_command = now
        return messages

    def take_commands(self) -> list[str]:
        """Все команды сервера разом — для RCON (TrainingServer)."""
        commands, self.commands = list(self.commands), deque()
        return commands

    def leave(self, session_id: int) -> None:
        """Бот вышел из салок (ему дали другую задачу)."""
        if self.players.pop(session_id, None) is not None:
            self._unmark(session_id)
        self.pending.pop(session_id, None)
        self.its.pop(session_id, None)  # водящих не осталось — _run_round назначит нового

    def describe(self, now: float | None = None) -> str:
        now = time.time() if now is None else now
        if self.paused_until is not None:
            return f"салки: раунд {self.rounds} окончен, сейчас начнётся новый"
        if self.round_started is None:
            return f"салки: ждём игроков (сейчас {len(self.players)})"
        return (f"салки: раунд {self.rounds} идёт {now - self.round_started:.0f} с, "
                f"водят {len(self._its_now(now))}, убегают {len(self._runners(now))}, осалили всего {self.tags}")

    # --- раунды ---------------------------------------------------------------

    def _run_round(self, now: float) -> None:
        """Начать раунд, когда есть с кем играть; пауза между раундами; время
        вышло; водящих не осталось (ушли из игры) — водит новый."""
        if self.paused_until is not None:
            if self.spread_at is not None and now >= self.spread_at:
                self.spread_at = None
                self._spread()
            if now >= self.paused_until:
                self.paused_until = None
                self._start_round(now)
            return
        players = len(self._participants(now))
        if self.round_started is None:
            if players >= 2:
                self._start_round(now)
            return
        if players < 2:
            print("[ai] Салки: играть не с кем — ждём игроков.")
            self._end_round(now, winners=None)
        elif now - self.round_started >= self.round_seconds:
            self._end_round(now, winners=FLEE)
        elif not self._runners(now):
            self._end_round(now, winners=None)  # последний убегающий ушёл из игры
        elif not self._its_now(now):
            # Водящие ушли из игры (бот вышел, человек стал наблюдателем или
            # сам ушёл в убегающие). Водит бот — человека, который только что
            # сам перестал водить, обратно не назначаем.
            runners = list(self._runners(now))
            new = random.choice([key for key in runners if isinstance(key, int)] or runners)
            print(f"[ai] Салки: водящих не осталось — теперь водит {self._who(new)}.")
            self._make_it(new, now)
            self._say(self._speaker(new), f"Водящих не осталось — теперь водит {self._who(new)}. "
                                          f"У вас {self._head_start()}, чтобы убежать.")

    def _start_round(self, now: float) -> None:
        """Новый раунд: все убегают, кроме одного случайного водящего."""
        participants = self._participants(now)
        if len(participants) < 2:
            self.round_started = None  # ждём игроков
            return
        self.rounds += 1
        self.round_started = now
        self.round_tags = 0
        if self.on_round_start is not None:
            self.on_round_start()
        former = list(self.its)
        self.its.clear()
        for sid in former:
            self._mark_role(sid)  # водил — теперь убегает
        for name, human in self._playing_humans(now).items():
            if self._human_role(human, now) == CHASE:
                self._assign_human(name, FLEE, now)
        alive = [key for key in participants if not isinstance(key, int) or self.players[key]["alive"]]
        first = random.choice(alive or participants)
        self._make_it(first, now)
        print(f"[ai] Салки: раунд {self.rounds} — водит {self._who(first)}, игроков {len(participants)}.")
        if isinstance(first, int):
            self._say(first, f"Я вожу! У вас {self._head_start()}, чтобы убежать.")
        else:
            self._say(self._speaker(), f"Водит {first}! У нас {self._head_start()}, чтобы убежать.")

    def _end_round(self, now: float, winners: str | None) -> None:
        """Раунд окончен: исходы ботам, пауза, разброс по арене, потом новый
        раунд. winners — CHASE (осалили последнего), FLEE (время вышло: кого
        не поймали, те молодцы) или None (играть не с кем)."""
        survivors = list(self._runners(now))
        for sid in self.players:
            if sid in self.pending:
                continue  # только что осалил или осалили — этот исход главнее
            if sid in self.its:
                self.pending[sid] = WON if winners == CHASE else TIME_UP
            elif winners == FLEE:
                self.pending[sid] = SURVIVED
        if winners == CHASE:
            self._say(self._speaker(), "Ура, все молодцы, ещё раз!")
            print(f"[ai] Салки: раунд {self.rounds} — осалили всех за {now - self.round_started:.0f} с.")
        elif winners == FLEE:
            names = ", ".join(self._who(key) for key in survivors)
            self._say(self._speaker(), f"Время вышло! Не поймали: {names} — молодцы. Ещё раз!")
            for key in survivors:
                self._add_points(self._who(key), TAG_POINTS)
            print(f"[ai] Салки: раунд {self.rounds} — время вышло, не поймали {len(survivors)}.")
        self.round_started = None
        self.paused_until = now + RESTART_PAUSE
        self.spread_at = now + SPREAD_DELAY

    def _spread(self) -> None:
        """Всех игроков салок — по арене в случайные места (телепорт, не
        смерть). Только на нашем сервере-арене: где арена — знаем из
        конфига; на чужом сервере новый раунд начинается там, где стоят."""
        arena = self.config.get("server", {}).get("arena")
        if not self.direct_commands or not arena or not arena.get("enabled", True):
            return
        cx, cz = arena["center"]
        max_range = arena["size"] // 2 - 3  # не у самых стен
        if self.config["bot"].get("body") == "plugin":
            # Раунд — на целой арене: плагин помнит, что сломали и поставили
            # за прошлый раунд, и возвращает как было (в симуляции — reset_world).
            self._command("/mcbot restore")
        # Все, кто в командах салок, кроме наблюдателей и творческого режима:
        # человек, который сейчас смотрит, а не играет, пусть стоит где стоял.
        self._command(f"/spreadplayers {cx} {cz} {SPREAD_DISTANCE} {max_range} false "
                      "@a[team=!,gamemode=!spectator,gamemode=!creative]")

    def _make_it(self, key, now: float) -> None:
        """Водит ещё один (id сессии бота или ник человека). Первые
        freeze_seconds он стоит ("считает до пяти") — пусть остальные
        разбегутся."""
        if isinstance(key, str):
            self._assign_human(key, CHASE, now)
            return
        self.its[key] = {"since": now, "frozen_until": now + self.freeze_seconds, "chasing": None}
        self._mark_role(key)  # водит — красный

    # --- удары ----------------------------------------------------------------

    def _check_tag(self, it: int, state: dict, now: float) -> None:
        """Водящий-бот кого-то ударил (attacked_id в его состоянии) — осалил ли?"""
        victim_entity = state.get("attacked_id")
        if victim_entity is None or self.is_frozen(it, now) or not self._can_play(self.players[it], now):
            return
        runners = self._runners(now)
        victim = next((key for key, runner in runners.items() if runner["entity_id"] == victim_entity), None)
        if victim is None:
            return
        if isinstance(victim, int) and not self._can_play(self.players[victim], now):
            return
        if math.dist(self.players[it]["pos"], runners[victim]["pos"]) > self.tag_distance:
            return
        self._tag(it, victim, now)

    def _check_hurt(self, runner: int, state: dict, now: float) -> None:
        """Убегающего бота ударили (hurt_by — кто): если это человек-водящий
        — осалил. Удары ботов-водящих засчитывает _check_tag."""
        attacker_entity = state.get("hurt_by")
        if attacker_entity is None or not self._can_play(self.players[runner], now):
            return
        tagger = next((name for name, human in self._playing_humans(now).items()
                       if human["entity_id"] == attacker_entity and self._human_role(human, now) == CHASE), None)
        if tagger is None or now < self.humans[tagger]["frozen_until"]:
            return
        # Дистанцию проверил сервер: урон бывает только от удара вблизи.
        self._tag(tagger, runner, now)

    def _tag(self, tagger, victim, now: float) -> None:
        """Осалил: осаленный тоже водит. Боту-водящему — награда (конец
        эпизода, но роль та же: дальше — за следующим), осаленному боту —
        штраф и новая роль."""
        self.tags += 1
        self.round_tags += 1
        self._add_points(self._who(tagger), TAG_POINTS)
        if isinstance(tagger, int):
            self.pending[tagger] = CAUGHT
            self.its[tagger]["chasing"] = None
        if isinstance(victim, int):
            self.pending[victim] = CAUGHT
        self._make_it(victim, now)
        left = len(self._runners(now))
        # Говорит водящий-бот, а если салил человек — осаленный бот.
        if isinstance(tagger, int):
            self._say(tagger, f"Я догнал {self._who(victim)}. Теперь и он водит!")
        elif isinstance(victim, int):
            self._say(victim, f"Меня догнал {tagger}. Теперь и я вожу!")
        print(f"[ai] Салки: {self._who(tagger)} осалил {self._who(victim)} — убегающих осталось {left}.")
        if left == 0:
            self._end_round(now, winners=CHASE)

    def _choose_chased(self, it: int, me: dict, now: float) -> dict | None:
        turn = self.its[it]
        runners = {key: runner for key, runner in self._runners(now).items() if runner.get("alive", True)}
        if not runners:
            turn["chasing"] = None
            return None
        distance = {key: math.dist(runner["pos"], me["pos"]) for key, runner in runners.items()}
        nearest = min(distance, key=distance.get)
        if turn["chasing"] not in runners or distance[nearest] < distance[turn["chasing"]] - self.switch_margin:
            turn["chasing"] = nearest
        return runners[turn["chasing"]]

    # --- кто в игре -----------------------------------------------------------

    def _participants(self, now: float) -> list:
        """Все в игре: id сессий ботов и ники играющих людей."""
        return list(self.players) + list(self._playing_humans(now))

    def _runners(self, now: float) -> dict:
        """Убегающие: id сессии бота или ник человека -> запись (у обоих есть
        "entity_id" и "pos")."""
        runners = {sid: player for sid, player in self.players.items() if sid not in self.its}
        runners.update({name: human for name, human in self._playing_humans(now).items()
                        if self._human_role(human, now) == FLEE})
        return runners

    def _its_now(self, now: float) -> dict:
        """Водящие — так же: боты и люди."""
        its = {sid: self.players[sid] for sid in self.its if sid in self.players}
        its.update({name: human for name, human in self._playing_humans(now).items()
                    if self._human_role(human, now) == CHASE})
        return its

    def _see_humans(self, sightings: list[dict], now: float) -> None:
        for sighting in sightings:
            human = self.humans.setdefault(sighting["name"], {"assigned": None, "assigned_until": 0.0,
                                                              "frozen_until": 0.0})
            human.update(entity_id=sighting.get("id"), pos=(sighting["x"], sighting["y"], sighting["z"]),
                         gamemode=sighting.get("gamemode", 0), team=sighting.get("team"), seen=now)

    def _human_role(self, human: dict, now: float) -> str | None:
        """Роль человека: только что назначенная судьёй, иначе — по его
        команде на сервере (None — не в салках)."""
        if now < human["assigned_until"]:
            return human["assigned"]
        return ROLE_OF_TEAM.get(human["team"])

    def _playing_humans(self, now: float) -> dict[str, dict]:
        """Люди, которые сейчас играют: их видели боты (не дольше
        HUMAN_SEEN_SECONDS назад), они в команде салок и не наблюдатели и не
        в творческом режиме."""
        return {name: human for name, human in self.humans.items()
                if now - human["seen"] <= HUMAN_SEEN_SECONDS and human["gamemode"] in PLAYING_GAMEMODES
                and self._human_role(human, now) is not None}

    def _assign_human(self, name: str, role: str, now: float) -> None:
        """Перевести человека в другую роль: командой сервера (её же потом
        увидят боты), а пока она до них не дошла — словом судьи."""
        human = self.humans[name]
        human["assigned"], human["assigned_until"] = role, now + HUMAN_ASSIGN_SECONDS
        self._ensure_teams()
        self._command(f"/team join {TEAM_OF_ROLE[role]} {name}")
        if role == CHASE:
            # Человек тоже "считает до пяти": медлительность не даёт сойти с места.
            human["frozen_until"] = now + self.freeze_seconds
            self._command(f"/effect give {name} minecraft:slowness {math.ceil(self.freeze_seconds)} 255 true")

    def _award_points(self, now: float) -> None:
        if now - self.points_at < self.points_every:
            return
        self.points_at = now
        if self.round_started is None:
            return  # между раундами очков не идёт
        for key, runner in self._runners(now).items():
            if isinstance(key, str) or self._can_play(runner, now):
                self._add_points(self._who(key), 1)

    def _add_points(self, name: str, amount: int) -> None:
        # Табло — только с RCON: через чат такой поток команд не пустить.
        if not self.direct_commands:
            return
        self.points[name] = self.points.get(name, 0) + amount
        self._command(f"scoreboard players set {name} tag_points {self.points[name]}")

    def _refresh_glow(self, now: float) -> None:
        if not self.direct_commands or now - self.glow_at < GLOW_REFRESH_SECONDS:
            return
        self.glow_at = now
        for sid, player in self.players.items():
            if player["alive"]:
                self._glow(sid)

    def _can_play(self, player: dict, now: float) -> bool:
        """Живой и не только что возродился: сразу после респавна (все
        появляются в одной точке мира) ни салить, ни быть осаленным нельзя."""
        return player["alive"] and now - player["alive_since"] >= self.respawn_seconds

    def _drop_stale(self, now: float) -> None:
        for sid in [sid for sid, p in self.players.items() if now - p["seen"] > self.stale_seconds]:
            print(f"[ai] Салки: бот {sid} давно молчит — выбывает из игры.")
            self.leave(sid)

    # --- чат и пометка ролей --------------------------------------------------

    def _name(self, session_id: int) -> str:
        return bot_name(session_id, self.config)

    def _who(self, key) -> str:
        """Ник участника: бота — по id сессии, человека — он сам."""
        return key if isinstance(key, str) else self._name(key)

    def _speaker(self, preferred=None) -> int:
        """Бот, от имени которого судья говорит в чат."""
        if isinstance(preferred, int):
            return preferred
        return self.command_bot if self.command_bot in self.players else next(iter(self.players), self.command_bot)

    def _head_start(self) -> str:
        seconds = self.freeze_seconds
        if seconds != int(seconds):
            return f"{seconds:g} секунды"  # дробное: "1.5 секунды"
        return plural(int(seconds), "секунда", "секунды", "секунд")

    def _say(self, session_id: int, text: str) -> None:
        self.messages.append((session_id, text))

    def _command(self, text: str) -> None:
        # Без RCON команды сервера — от имени одного бота с op (обычный чат
        # их не выполнит). Нет op — сервер ответит этому боту ошибкой, и только.
        if self.mark_roles:
            self.commands.append(text)

    def _ensure_teams(self) -> None:
        """Команды салок на сервере — один раз за запуск (уже есть с
        прошлого — сервер скажет "уже есть", не страшно)."""
        if self.teams_ready:
            return
        for team, (title, color) in TEAMS.items():
            self._command(f'/team add {team} "{title}"')
            self._command(f"/team modify {team} color {color}")
            # Свои своих не бьют: водящему водящего салить незачем, а урон и
            # отбрасывание от случайного удара по своим только мешают.
            self._command(f"/team modify {team} friendlyFire false")
        self._command(f'/team modify {TEAM_IT} prefix "[водит] "')
        self.teams_ready = True

    def _mark_role(self, session_id: int) -> None:
        """Цвет ника и свечения по роли. Игрок бывает только в одной команде —
        join в другую сам убирает из прежней."""
        self._ensure_teams()
        self._command(f"/team join {TEAM_OF_ROLE[self.role_of(session_id)]} {self._name(session_id)}")

    def _glow(self, session_id: int) -> None:
        """Свечение сквозь стены цветом команды — видно, где бот, даже в
        пещере. Смерть снимает эффекты, поэтому после респавна — заново."""
        self._command(f"/effect give {self._name(session_id)} minecraft:glowing infinite 0 true")

    def _unmark(self, session_id: int) -> None:
        name = self._name(session_id)
        self._command(f"/team leave {name}")
        self._command(f"/effect clear {name} minecraft:glowing")
