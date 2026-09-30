"""Наш сервер-арена (server/, Paper) — то, что лучше делает сервер, а не
сеть: мир под обучение, правила физически, показ. Всё — командами через
RCON (py/rcon.py) из фонового потока: главный цикл ИИ сервер не ждёт.

  - мир: правила игры (всегда день, без погоды и мобов, сразу респавн...),
    точка спавна, граница мира, арена с препятствиями (py/arena.py) —
    арена строится один раз на мир (метка — счёт #arena в табло mcbot).
    Уйти с арены нельзя и без штрафов: стены — барьер, за ним граница мира
    (автор: "всё равно у нас есть барьер" — штраф и возврат на спавн за
    выход убраны 2026-09-26);
  - табло "Мотивация" (сбоку) — сумма наград бота за последние
    motivation_window_seconds, то есть ровно то, за что учится сеть: видно,
    кто сейчас молодец. "Очки салок" (в списке игроков, Tab) ведёт судья
    салок (py/tag_game.py), как и пометки ролей (/team, /effect).

Не наш сервер (RCON не отвечает — чужой сервер, другой порт) —
TrainingServer.connect вернёт None, и всё работает по-старому.
"""

from __future__ import annotations

import queue
import re
import threading
import time
from collections import deque
from pathlib import Path

from arena import arena_commands
from rcon import Rcon, RconError
from training_modules.targets import bot_index

# Правила игры для обучения: ничего не отвлекает и не убивает, смерть не
# стоит минуты (сразу респавн, вещи остаются), всегда светло. В 26.x
# правила переименовали (doDaylightCycle -> advance_time, doMobSpawning ->
# spawn_mobs — подсказал автор), поэтому у каждого — имена-кандидаты:
# сначала новое, потом старое; берётся то, что примет сервер.
GAME_RULES = [
    (("advance_time", "doDaylightCycle"), "false"),
    (("advance_weather", "doWeatherCycle"), "false"),
    (("spawn_mobs", "doMobSpawning"), "false"),
    (("spawn_phantoms", "doInsomnia"), "false"),
    (("spawn_patrols", "doPatrolSpawning"), "false"),
    (("spawn_wandering_traders", "doTraderSpawning"), "false"),
    (("mob_griefing", "mobGriefing"), "false"),
    (("immediate_respawn", "doImmediateRespawn"), "true"),
    (("keep_inventory", "keepInventory"), "true"),
    (("show_advancement_messages", "announceAdvancements"), "false"),
    (("respawn_radius", "spawnRadius"), "0"),
    # Салки: удар человека-водящего по боту судья видит только по урону у
    # бота — без PvP урона нет (с 1.21.9 это правило игры, а не настройка).
    (("pvp",), "true"),
    # Арена не меняется сама: трава не расползается на кучки земли (и земля
    # под блоками не перестаёт быть травой) — иначе зрение в игре расходилось
    # с симуляцией цветом этих блоков (сверка плагина, 2026-09-27).
    (("random_tick_speed", "randomTickSpeed"), "0"),
]

# Наборы команд при запуске задачи: server/functions/<задача>.mcfunction.
FUNCTIONS_DIR = Path(__file__).resolve().parent.parent / "server" / "functions"

# Пульс судьи: ai_loop, ведущий арену, раз в publish_every секунд пишет в
# табло mcbot время (#judge). Второй ai_loop (случайный второй лаунчер,
# тест с фейковыми ботами) видит свежий пульс и к арене не подключается —
# иначе два судьи раздавали бы одним и тем же ботам роли, телепорты и
# /kill (так и было: тест назначил "второго водящего" и телепортировал ботов).
PULSE_FRESH_SECONDS = 15

# Ответы сервера, которые означают ошибку команды — их в лог.
ERROR_WORDS = ("Unknown", "Incorrect", "Invalid", "Expected", "not loaded", "No entity", "No player")


class MotivationBoard:
    """Сумма наград каждого бота за последние window секунд."""

    def __init__(self, window: float):
        self.window = window
        self.history: dict[str, deque] = {}
        self.totals: dict[str, float] = {}

    def add(self, name: str, reward: float, now: float) -> None:
        self.history.setdefault(name, deque()).append((now, reward))
        self.totals[name] = self.totals.get(name, 0.0) + reward

    def scores(self, now: float) -> dict[str, int]:
        result = {}
        for name, history in self.history.items():
            while history and history[0][0] < now - self.window:
                self.totals[name] -= history.popleft()[1]
            result[name] = round(self.totals[name])
        return result


class TrainingServer:
    def __init__(self, rcon: Rcon, config: dict):
        cfg = config["server"]
        self.config = config
        self.rcon = rcon
        self.arena = cfg["arena"]
        self.arena_enabled = self.arena.get("enabled", True)
        self.motivation = MotivationBoard(cfg.get("motivation_window_seconds", 300))
        self.publish_every = cfg.get("scoreboard_every_seconds", 5)
        self._published: dict[str, int] = {}
        self._publishes = 0
        self._last_publish = 0.0
        self._errors_shown: set[str] = set()
        self._queue: queue.Queue = queue.Queue()
        self._worker = threading.Thread(target=self._work, daemon=True)
        self._worker.start()

    @classmethod
    def connect(cls, config: dict) -> "TrainingServer | None":
        cfg = config.get("server", {}).get("rcon")
        if not cfg:
            return None
        rcon = Rcon(cfg.get("host", "127.0.0.1"), cfg["port"], cfg["password"])
        try:
            rcon.connect()
            pulse = re.search(r"has (-?\d+)", rcon.command("scoreboard players get #judge mcbot"))
        except (OSError, RconError) as err:
            print(f"[server] Наш сервер-арена не отвечает по RCON ({err}) — работаю без него "
                  "(нет арены, табло и возврата на спавн).")
            return None
        if pulse and time.time() - int(pulse.group(1)) < PULSE_FRESH_SECONDS:
            print("[server] Арену уже ведёт другой ai_loop (его пульс свежий) — этот работает без неё.")
            rcon.close()
            return None
        print("[server] Подключился к серверу-арене по RCON.")
        return cls(rcon, config)

    # --- мир ------------------------------------------------------------------

    def setup(self) -> None:
        """Правила игры, спавн, граница, табло — при каждом запуске (дёшево и
        переживает ручные правки); арена — один раз на мир."""
        cx, cz = self.arena["center"]
        y = self.arena["floor_y"]
        half = self.arena["size"] // 2
        for names, value in GAME_RULES:
            for name in names:
                reply = self.rcon.command(f"gamerule {name} {value}")
                if not any(word in reply for word in ERROR_WORDS):
                    break
            else:
                print(f"[server] Правило игры {names[0]} сервер не принял ни под одним именем {names}: {reply}")
        for command in (
            "time set day",
            "weather clear",
            f"setworldspawn {cx} {y} {cz}",
            f"worldborder center {cx} {cz}",
            f"worldborder set {self.arena['size'] + 2}",
            # Арена всегда загружена — даже когда рядом никого (боты на респавне).
            f"forceload add {cx - half - 1} {cz - half - 1} {cx + half} {cz + half}",
            "scoreboard objectives add mcbot dummy",
            'scoreboard objectives add motivation dummy "Мотивация"',
            "scoreboard objectives setdisplay sidebar motivation",
            'scoreboard objectives add tag_points dummy "Очки салок"',
            "scoreboard objectives setdisplay list tag_points",
        ):
            self.send(command)
        self._queue.join()  # дождаться, пока всё выше выполнено, — арена после forceload
        # Команды салок до "заражения" (2026-09-26) назывались иначе — убрать,
        # чтобы в /team list не висели. Ответ не важен: их может и не быть.
        # Напрямую, а не очередью, — поток очереди сейчас свободен (join выше).
        for old_team in ("vodit", "ubegaet"):
            self.rcon.command(f"team remove {old_team}")

        if self.arena_enabled and "none is set" in self.rcon.command("scoreboard players get #arena mcbot"):
            print("[server] Строю арену (один раз на мир)...")
            for command in arena_commands(self.arena):
                self.send(command)
            self.send("scoreboard players set #arena mcbot 1")
            self._queue.join()

    # --- табло "Мотивация" ------------------------------------------------------

    def add_reward(self, name: str, reward: float, now: float) -> None:
        self.motivation.add(name, reward, now)

    def publish(self, now: float) -> None:
        """Раз в publish_every секунд — счёт на табло, только изменившийся."""
        if now - self._last_publish < self.publish_every:
            return
        self._last_publish = now
        self.send(f"scoreboard players set #judge mcbot {int(now)}")  # пульс судьи
        self._publishes += 1
        if self._publishes % 12 == 0:
            # Раз в минуту — все счета заново, а не только изменившиеся:
            # табло могли удалить и создать руками, сервер счета забыл.
            self._published.clear()
        for name, score in self.motivation.scores(now).items():
            if self._published.get(name) == score:
                continue
            if name not in self._published:
                # Боковое табло Minecraft всегда сортирует по числу — поэтому
                # числом ставим место (AI_1 сверху: у него наибольшее), а саму
                # мотивацию показываем текстом (автор: "все вразброс").
                self.send(f"scoreboard players set {name} motivation {1000 - (bot_index(name, self.config) or 0)}")
            self._published[name] = score
            color = "green" if score >= 0 else "red"
            self.send(f'scoreboard players display numberformat {name} motivation fixed {{"text":"{score}","color":"{color}"}}')

    # --- команды --------------------------------------------------------------

    def send(self, command: str) -> None:
        self._queue.put(command.lstrip("/"))

    def is_bed(self, world: str, cells: list) -> bool:
        """Цела ли кровать (бедварс, py/bedwars_game.py): в клетках head и foot
        — кровать. Сразу, мимо очереди команд (RCON под замком). Сервер не
        ответил — считаем целой: сломать её зря хуже, чем заметить позже."""
        try:
            for x, y, z in cells:
                if "passed" not in self.rcon.command(f"execute in minecraft:{world} if block {x} {y} {z} #minecraft:beds"):
                    return False
        except (OSError, RconError):
            return True
        return True

    def setup_task_world(self, world: str, fills: list[str], bounds: tuple) -> None:
        """Мир задачки (плагин создаёт его пустым, config.json bot.task_worlds):
        те же правила игры, что на арене, день, участок трассы всегда
        загружен, трасса — готовыми командами fill в этом мире (fills).
        Обычный мир не трогаем (автор: задачки — в своих мирах)."""
        prefix = f"execute in minecraft:{world} run "
        for names, value in GAME_RULES:
            for name in names:
                reply = self.rcon.command(f"{prefix}gamerule {name} {value}")
                if not any(word in reply for word in ERROR_WORDS):
                    break
            else:
                print(f"[server] {world}: правило {names[0]} не принято: {reply}")
        x0, _, z0, x1, _, z1 = bounds
        for command in (f"{prefix}time set day", f"{prefix}forceload add {x0} {z0} {x1} {z1}", *fills):
            self.send(command)
        print(f"[server] Мир задачки {world}: правила и трасса ({len(fills)} команд fill).")

    def run_function(self, name: str) -> int:
        """Команды из server/functions/<name>.mcfunction (идея автора: свой
        набор команд при запуске задачи — сложность, вещи...) — по одной, по
        RCON, той же очередью. Пустые строки и # комментарии пропускаются, "/"
        в начале можно не писать. Ошибку команды покажет поток очереди
        ([server] команда -> ответ). Возвращает, сколько команд отправлено
        (0 — файла нет)."""
        path = FUNCTIONS_DIR / f"{name}.mcfunction"
        if not path.is_file():
            return 0
        commands = []
        for raw in path.read_text(encoding="utf-8-sig").splitlines():
            line = raw.strip()
            if line and not line.startswith("#"):
                commands.append(line)
        for command in commands:
            self.send(command)
        print(f"[server] {path.name}: {len(commands)} команд.")
        return len(commands)

    def close(self) -> None:
        self.send("scoreboard players set #judge mcbot 0")  # отпустить арену — перезапуск подхватит сразу
        self._queue.put(None)
        self._worker.join(timeout=5)
        self.rcon.close()

    def _work(self) -> None:
        while True:
            command = self._queue.get()
            try:
                if command is None:
                    return
                reply = self.rcon.command(command)
                if any(word in reply for word in ERROR_WORDS):
                    key = command.split(" ")[0] + " " + reply[:40]
                    if key not in self._errors_shown:  # одна и та же ошибка — один раз
                        self._errors_shown.add(key)
                        print(f"[server] {command} -> {reply}")
            except (OSError, RconError) as err:
                if "rcon" not in self._errors_shown:
                    self._errors_shown.add("rcon")
                    print(f"[server] RCON: {err}")
            finally:
                self._queue.task_done()
