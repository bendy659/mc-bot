"""Лаунчер mc-bot: одно окно вместо двух консолей.

Запуск: run_ai_bot.bat (двойным кликом) или python py/launcher.py.

Что делает:
  - поднимает py/ai_loop.py, дожидается, пока он забиндит ZMQ-порты
    ("Цикл запущен"), и только потом js/bot.js — порядок старта соблюдается
    сам, без "timeout /t 2" наугад;
  - собирает вывод обоих процессов в одно окно, строки помечены [PY] / [JS]
    и раскрашены (ошибки — красным); каждый источник можно скрыть галочкой;
  - показывает, как идут дела у каждой задачки, по строкам "[metrics] {...}",
    которые ai_loop печатает только для лаунчера (переменная окружения
    MCBOT_METRICS=1): одна фраза на задачку человеческими словами ("цель в
    прицеле 23% времени ↑") и график главной цифры каждой задачки — выше
    значит лучше;
  - "Стоп" останавливает аккуратно: сначала Node (перестаёт слать
    состояния), потом Python (сохраняет чекпоинты). Ctrl+C процессу без
    консоли не послать, поэтому оба процесса слушают строку "stop" в stdin;
  - автоперезапуск: если процесс упал сам (не по "Стоп") — поднимается
    заново через несколько секунд. Для долгого обучения без присмотра;
  - строка команд (py/command_bar.py): !task, !setTarget, !greedy... с
    автодополнением — вместо чата игры, ответы ботов приходят в лог.

Только стандартная библиотека (tkinter) — ставить ничего не нужно.
"""

import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

sys.path.insert(0, str(Path(__file__).resolve().parent))
from command_bar import CommandBar  # noqa: E402
from rcon import Rcon, RconError  # noqa: E402
from config import CONFIG  # noqa: E402
from training_modules import MIX, TAG, TASK_ORDER  # noqa: E402  (без torch/zmq — импорт лёгкий)

ROOT = Path(__file__).resolve().parent.parent

MAX_LOG_LINES = 5000
MAX_CHART_POINTS = 600
RESTART_DELAY_S = 5
PYTHON_READY_MARKER = "Цикл запущен"
PYTHON_READY_TIMEOUT_S = 120  # загрузка CUDA + демо-записей бывает небыстрой

# Цвета задачек на графике.
TASK_COLORS = {
    "walking": "#2e86de", "looking": "#e67e22", "follow": "#27ae60",
    "gathering": "#8e5a2b", "crafting": "#7f8c8d", "chase": "#c0392b", "flee": "#8e44ad",
}
ERROR_WORDS = ("Traceback", "Error", "Ошибка", "ошибка", "error", "Exception")
SETTINGS_PATH = ROOT / "data" / "launcher_settings.json"  # запоминает тему и задачу

# Цвета окна: светлая и тёмная тема (переключатель "Тёмная тема" в панели).
THEMES = {
    "light": {
        "bg": "#f0f0f0", "fg": "#000000", "field": "#ffffff", "field_fg": "#000000", "select": "#cce4ff",
        "border": "#cccccc", "muted": "#777777", "grid": "#bbbbbb", "chart_bg": "#ffffff",
        "PY": "#1f5fa8", "JS": "#1e7b34", "SRV": "#9a6a1c", "SYS": "#777777", "ERR": "#c0392b", "CMD": "#8e44ad",
        "valid": "#1e7b34", "unknown": "#c0392b",
    },
    "dark": {
        "bg": "#1e1f22", "fg": "#dcdcdc", "field": "#2b2d31", "field_fg": "#dcdcdc", "select": "#3d5a80",
        "border": "#3c3f45", "muted": "#8c8f94", "grid": "#4a4d52", "chart_bg": "#232428",
        "PY": "#6cb4ff", "JS": "#6fd08c", "SRV": "#e0b060", "SYS": "#9a9a9a", "ERR": "#ff6b6b", "CMD": "#c792ea",
        "valid": "#6fd08c", "unknown": "#ff6b6b",
    },
}


def python_executable() -> str:
    """Если лаунчер запущен через pythonw.exe (без консоли), дочерний
    ai_loop всё равно запускаем обычным python.exe — pythonw иногда
    теряет вывод."""
    exe = Path(sys.executable)
    candidate = exe.with_name("python.exe")
    return str(candidate) if exe.name.lower() == "pythonw.exe" and candidate.exists() else str(exe)


class ManagedProcess:
    """Один дочерний процесс: запуск без консольного окна, чтение вывода
    в очередь из отдельного потока, мягкая остановка через stdin."""

    def __init__(self, tag: str, command: list[str], env: dict, output: queue.Queue, cwd: Path = ROOT):
        self.tag = tag
        self.command = command
        self.env = env
        self.output = output
        self.cwd = cwd
        self.process: subprocess.Popen | None = None

    def start(self) -> None:
        # CREATE_NO_WINDOW — чтобы не всплывали те самые лишние консоли.
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.process = subprocess.Popen(
            self.command,
            cwd=self.cwd,
            env=self.env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=flags,
        )
        threading.Thread(target=self._read_output, args=(self.process,), daemon=True).start()

    def _read_output(self, process: subprocess.Popen) -> None:
        for line in process.stdout:
            self.output.put(("line", self.tag, line.rstrip("\r\n")))
        process.wait()
        self.output.put(("exit", self.tag, process.returncode))

    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def send_line(self, line: str) -> bool:
        """Строка в stdin процесса (команды для Node). False — процесс не запущен."""
        if not self.running():
            return False
        try:
            self.process.stdin.write(line + "\n")
            self.process.stdin.flush()
            return True
        except OSError:
            return False

    def request_stop(self) -> None:
        if not self.running():
            return
        try:
            self.process.stdin.write("stop\n")
            self.process.stdin.flush()
        except OSError:
            pass  # процесс уже закрывает stdin — значит, и так завершается

    def wait_or_kill(self, timeout: float) -> None:
        if self.process is None:
            return
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.output.put(("line", "SYS", f"{self.tag} не остановился за {timeout:.0f} с — завершаю принудительно."))
            self.process.kill()


class LauncherApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("mc-bot — лаунчер")
        self.geometry("1100x760")
        self.minsize(800, 500)

        self.output: queue.Queue = queue.Queue()
        self.python_proc: ManagedProcess | None = None
        self.node_proc: ManagedProcess | None = None
        self.server_proc: ManagedProcess | None = None  # сервер-арена (server/), если запущен отсюда
        self.closing = False
        self.active = False      # пользователь нажал "Старт" и ещё не нажал "Стоп"
        self.stopping = False
        self.waiting_for_python_since: float | None = None
        self.metrics_history: list[dict] = []
        self.palette = THEMES["light"]  # до apply_theme — график может перерисоваться раньше

        self._build_toolbar()
        self._build_body()
        self._set_buttons()
        self.apply_theme()

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(100, self._poll_output)

    # --- интерфейс -----------------------------------------------------------

    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self, padding=6)
        bar.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(bar, text="Задача:").pack(side=tk.LEFT)
        # Задача — та, что выбирали в прошлый раз (автор перезапустил лаунчер,
        # а задача молча вернулась на mix — и салки "пропали"); впервые — mix.
        tasks = [MIX, TAG, *TASK_ORDER]
        saved_task = self._load_settings().get("task")
        self.task_var = tk.StringVar(value=saved_task if saved_task in tasks else MIX)
        self.task_box = ttk.Combobox(bar, textvariable=self.task_var, values=tasks, state="readonly", width=12)
        self.task_box.pack(side=tk.LEFT, padx=(4, 10))
        self.task_box.bind("<<ComboboxSelected>>", lambda _event: self._save_settings(task=self.task_var.get()))

        self.start_button = ttk.Button(bar, text="▶ Старт", command=self.start)
        self.start_button.pack(side=tk.LEFT)
        self.stop_button = ttk.Button(bar, text="■ Стоп", command=self.stop)
        self.stop_button.pack(side=tk.LEFT, padx=(4, 10))

        self.restart_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Перезапускать при падении", variable=self.restart_var).pack(side=tk.LEFT)

        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        # Сервер-арена (server/, Paper): запуск и остановка ("stop" — мир
        # сохранится). Запущенный отдельно (start.bat) — останавливать
        # командой /stop в строке команд.
        self.server_button = ttk.Button(bar, text="▶ Сервер", command=self.toggle_server)
        self.server_button.pack(side=tk.LEFT)

        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        self.show_py = tk.BooleanVar(value=True)
        self.show_js = tk.BooleanVar(value=True)
        self.show_srv = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Python", variable=self.show_py, command=self._apply_filters).pack(side=tk.LEFT)
        ttk.Checkbutton(bar, text="Node", variable=self.show_js, command=self._apply_filters).pack(side=tk.LEFT)
        ttk.Checkbutton(bar, text="Сервер", variable=self.show_srv, command=self._apply_filters).pack(side=tk.LEFT)
        ttk.Button(bar, text="Очистить лог", command=self._clear_log).pack(side=tk.LEFT, padx=(10, 0))

        self.status_var = tk.StringVar(value="Остановлено")
        ttk.Label(bar, textvariable=self.status_var).pack(side=tk.RIGHT)
        self.dark_var = tk.BooleanVar(value=self._load_settings().get("dark", True))
        ttk.Checkbutton(bar, text="Тёмная тема", variable=self.dark_var,
                        command=self._on_theme_toggle).pack(side=tk.RIGHT, padx=(0, 12))

    def _build_body(self) -> None:
        panes = ttk.PanedWindow(self, orient=tk.VERTICAL)
        panes.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        log_frame = ttk.Frame(panes)
        self.log = ScrolledText(log_frame, wrap=tk.WORD, font=("Consolas", 9), state=tk.DISABLED)
        self.log.pack(fill=tk.BOTH, expand=True)
        self.command_bar = CommandBar(log_frame, self, CONFIG["bot"].get("count", 1), self.send_command,
                                      on_server=self.send_server_command)
        self.command_bar.pack(fill=tk.X)
        self.log.tag_configure("PY", foreground="#1f5fa8")
        self.log.tag_configure("JS", foreground="#1e7b34")
        self.log.tag_configure("SRV", foreground="#9a6a1c")
        self.log.tag_configure("SYS", foreground="#777777")
        self.log.tag_configure("ERR", foreground="#c0392b")
        # Команды и ответы на них — отдельным цветом, чтобы не терялись в потоке.
        # elide=False: ответы на команды видны, даже если вывод Node скрыт галочкой.
        self.log.tag_configure("CMD", foreground="#8e44ad", font=("Consolas", 9, "bold"), elide=False)
        panes.add(log_frame, weight=3)

        stats = ttk.Frame(panes)
        panes.add(stats, weight=2)

        self.stats_var = tk.StringVar(value="Как дела у задачек — появится через ~30 секунд после старта.")
        ttk.Label(stats, textvariable=self.stats_var, font=("Consolas", 9), justify=tk.LEFT).pack(
            side=tk.TOP, anchor=tk.W, pady=(4, 2))

        self.chart = tk.Canvas(stats, background="white", height=220, highlightthickness=1,
                               highlightbackground="#cccccc")
        self.chart.pack(fill=tk.BOTH, expand=True)
        self.chart.bind("<Configure>", lambda event: self._draw_chart())

    # --- тема ----------------------------------------------------------------

    def _load_settings(self) -> dict:
        try:
            return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save_settings(self, **changes) -> None:
        settings = self._load_settings()
        settings.update(changes)
        try:
            SETTINGS_PATH.parent.mkdir(exist_ok=True)
            SETTINGS_PATH.write_text(json.dumps(settings), encoding="utf-8")
        except OSError:
            pass  # не запомнится — не страшно

    def _on_theme_toggle(self) -> None:
        self.apply_theme()
        self._save_settings(dark=self.dark_var.get())

    def apply_theme(self) -> None:
        """Перекрасить всё окно: ttk-виджеты (тема clam — единственная, что
        даёт менять цвета), лог и его цвета по процессам, график, строку
        команд."""
        p = self.palette = THEMES["dark" if self.dark_var.get() else "light"]
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", background=p["bg"], foreground=p["fg"], fieldbackground=p["field"],
                        bordercolor=p["border"], lightcolor=p["bg"], darkcolor=p["bg"], troughcolor=p["field"],
                        selectbackground=p["select"], selectforeground=p["fg"], insertcolor=p["fg"])
        style.configure("TButton", background=p["field"], foreground=p["fg"])
        style.map("TButton", background=[("active", p["select"]), ("disabled", p["bg"])],
                  foreground=[("disabled", p["muted"])])
        style.configure("TCheckbutton", background=p["bg"], foreground=p["fg"], indicatorbackground=p["field"])
        style.map("TCheckbutton", background=[("active", p["bg"])], indicatorbackground=[("selected", p["select"])])
        style.configure("TCombobox", fieldbackground=p["field"], background=p["field"], foreground=p["fg"],
                        arrowcolor=p["fg"])
        style.map("TCombobox", fieldbackground=[("readonly", p["field"]), ("disabled", p["bg"])],
                  foreground=[("readonly", p["fg"]), ("disabled", p["muted"])],
                  selectbackground=[("readonly", p["field"])], selectforeground=[("readonly", p["fg"])])
        # Выпадающий список комбобоксов — обычный tk.Listbox: цвета через
        # базу опций (для ещё не открытых) и напрямую (для уже открытых).
        for option, value in (("background", p["field"]), ("foreground", p["fg"]),
                              ("selectBackground", p["select"]), ("selectForeground", p["fg"])):
            self.option_add(f"*TCombobox*Listbox.{option}", value)
        for combobox in (self.task_box, *self.command_bar.winfo_children()):
            if isinstance(combobox, ttk.Combobox):
                try:
                    listbox = self.tk.eval(f"ttk::combobox::PopdownWindow {combobox}") + ".f.l"
                    self.tk.call(listbox, "configure", "-background", p["field"], "-foreground", p["fg"],
                                 "-selectbackground", p["select"], "-selectforeground", p["fg"])
                except tk.TclError:
                    pass
        self.configure(background=p["bg"])
        self.log.configure(background=p["field"], foreground=p["field_fg"], insertbackground=p["fg"],
                           selectbackground=p["select"], selectforeground=p["field_fg"])
        for tag in ("PY", "JS", "SRV", "SYS", "ERR", "CMD"):
            self.log.tag_configure(tag, foreground=p[tag])
        self.chart.configure(background=p["chart_bg"], highlightbackground=p["border"])
        self.command_bar.apply_theme(p)
        self._draw_chart()

    def _set_buttons(self) -> None:
        self.start_button.state(["disabled"] if self.active else ["!disabled"])
        server_running = self.server_proc is not None and self.server_proc.running()
        self.server_button.configure(text="■ Сервер" if server_running else "▶ Сервер")
        self.stop_button.state(["!disabled"] if self.active and not self.stopping else ["disabled"])
        self.task_box.state(["disabled"] if self.active else ["!disabled", "readonly"])

    # --- запуск и остановка --------------------------------------------------

    def _make_python(self) -> ManagedProcess:
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", MCBOT_METRICS="1")
        command = [python_executable(), "-u", str(ROOT / "py" / "ai_loop.py"), "--task", self.task_var.get()]
        return ManagedProcess("PY", command, env, self.output)

    def _make_node(self) -> ManagedProcess | None:
        node = shutil.which("node")
        if node is None:
            self._append("SYS", "Не нашёл node в PATH — Node.js установлен?", error=True)
            return None
        return ManagedProcess("JS", [node, str(ROOT / "js" / "bot.js")], dict(os.environ), self.output)

    def start(self) -> None:
        if self.active:
            return
        self.active = True
        self.stopping = False
        self._set_buttons()
        self._start_python()

    def _start_python(self) -> None:
        self._append("SYS", f"Запускаю Python (задача {self.task_var.get()})...")
        self.python_proc = self._make_python()
        self.python_proc.start()
        # Node стартует, когда Python напечатает PYTHON_READY_MARKER (порты
        # забинжены) — см. _handle_line. Если Node уже работает (перезапуск
        # одного Python), он переподключится к ZMQ сам.
        if self.node_proc is None or not self.node_proc.running():
            self.waiting_for_python_since = time.time()
        self.status_var.set("Жду Python...")

    def _plugin_body(self) -> bool:
        """Боты — игроки сервера из плагина plugin/ (config.json: bot.body), а не Node."""
        return CONFIG.get("bot", {}).get("body") == "plugin"

    def _start_node(self) -> None:
        self.waiting_for_python_since = None
        if self._plugin_body():
            # Ботов заводит плагин сервера: Python уже слушает порты — пора.
            self._append("SYS", "Боты — в плагине сервера: /mcbot start по RCON...")
            threading.Thread(target=self._plugin_worker, args=("mcbot start",), daemon=True).start()
            self.status_var.set("Работает")
            return
        self.node_proc = self._make_node()
        if self.node_proc is None:
            return
        self._append("SYS", "Запускаю Node (рой ботов)...")
        self.node_proc.start()
        self.status_var.set("Работает")

    def stop(self) -> None:
        if not self.active or self.stopping:
            return
        self.stopping = True
        self.waiting_for_python_since = None
        self._set_buttons()
        self.status_var.set("Останавливаю...")
        self._append("SYS", "Останавливаю: сначала Node, потом Python (сохранит чекпоинты)...")
        threading.Thread(target=self._stop_worker, daemon=True).start()

    def _stop_worker(self) -> None:
        # Ожидание — в отдельном потоке, чтобы окно не зависало.
        if self._plugin_body():
            self._plugin_worker("mcbot stop")
        if self.node_proc is not None:
            self.node_proc.request_stop()
            self.node_proc.wait_or_kill(10)
        if self.python_proc is not None:
            self.python_proc.request_stop()
            self.python_proc.wait_or_kill(30)
        self.output.put(("stopped", "SYS", None))

    def toggle_server(self) -> None:
        if self.server_proc is not None and self.server_proc.running():
            self._append("SYS", "Останавливаю сервер (stop — мир сохранится)...")
            self.server_proc.request_stop()
            return
        threading.Thread(target=self._start_server_if_absent, daemon=True).start()

    def _start_server_if_absent(self) -> None:
        """Сервер уже отвечает по RCON (запущен отдельно, start.bat) — второй
        не запускаем; иначе — запускаем свой. Проверка — в потоке:
        подключение может занять секунды."""
        if self._rcon("list") is not None:
            self.output.put(("line", "SYS", "Сервер уже работает (запущен не из лаунчера). "
                                            "Остановить — /stop в строке команд."))
            return
        self.output.put(("start_server", "SYS", None))

    def _start_server(self) -> None:
        java = shutil.which("java")
        jars = sorted((ROOT / "server").glob("paper-*.jar"))
        if java is None or not jars:
            self._append("SYS", "Не нашёл java в PATH или server/paper-*.jar — см. server/README.md.", error=True)
            return
        # -Dstdout.encoding — иначе русские буквы в логе сервера превращаются в мусор.
        command = [java, "-Xms1G", "-Xmx3G", "-Dstdout.encoding=UTF-8", "-Dfile.encoding=UTF-8",
                   "-jar", str(jars[-1]), "--nogui"]
        self.server_proc = ManagedProcess("SRV", command, dict(os.environ), self.output, cwd=ROOT / "server")
        self._append("SYS", f"Запускаю сервер ({jars[-1].name}), подключение — localhost:{CONFIG['bot']['port']}...")
        self.server_proc.start()
        self._set_buttons()

    def send_server_command(self, text: str) -> None:
        """Команда серверу из строки команд ("/..."): по RCON, в потоке — ответ в лог."""
        self._append("CMD", f"/{text}  → сервер")
        threading.Thread(target=self._server_command_worker, args=(text,), daemon=True).start()

    def _server_command_worker(self, text: str) -> None:
        reply = self._rcon(text)
        if reply is None:
            self.output.put(("line", "SYS", "Сервер не отвечает по RCON — он запущен? (кнопка «Сервер»)"))
        else:
            self.output.put(("line", "CMD", reply or "(сервер выполнил молча)"))

    def _plugin_worker(self, text: str) -> None:
        """Команда плагину McBot по RCON (в потоке) — ответ или понятная ошибка в лог."""
        reply = self._rcon(text)
        if reply is None:
            self.output.put(("line", "SYS", "Сервер не отвечает по RCON — боты плагина не запущены. "
                                            "Сервер запущен? (кнопка «Сервер»)"))
        elif "Unknown or incomplete command" in reply or "Unknown command" in reply:
            self.output.put(("line", "SYS", "На сервере нет плагина McBot: plugin/gradlew deploy "
                                            "и перезапусти сервер (или bot.body = \"mineflayer\" в config.json)."))
        else:
            for line in reply.strip().splitlines():
                self.output.put(("line", "CMD", line))

    def _rcon(self, text: str) -> str | None:
        """Команда серверу по RCON; None — сервер не отвечает."""
        cfg = CONFIG.get("server", {}).get("rcon")
        if not cfg:
            return None
        try:
            connection = Rcon(cfg.get("host", "127.0.0.1"), cfg["port"], cfg["password"], timeout=3.0)
            try:
                return connection.command(text)
            finally:
                connection.close()
        except (OSError, RconError):
            return None

    def on_close(self) -> None:
        self.closing = True
        if self.active and not self.stopping:
            self.stop()
        if self.active:
            # Дождаться сохранения чекпоинтов, потом закрыть окно.
            self.after(300, self.on_close)
            return
        if self.server_proc is not None and self.server_proc.running():
            # Сервер, запущенный отсюда, без окна не остался бы сиротой: stop
            # (мир сохранится) и ждём, пока он завершится.
            if not getattr(self, "_server_stop_sent", False):
                self._server_stop_sent = True
                self._append("SYS", "Останавливаю сервер перед выходом (мир сохранится)...")
                self.server_proc.request_stop()
            self.after(300, self.on_close)
            return
        self.destroy()

    # --- вывод процессов -----------------------------------------------------

    def _poll_output(self) -> None:
        try:
            while True:
                kind, tag, payload = self.output.get_nowait()
                if kind == "line":
                    self._handle_line(tag, payload)
                elif kind == "exit":
                    self._handle_exit(tag, payload)
                elif kind == "start_server":
                    self._start_server()
                elif kind == "stopped":
                    self.active = False
                    self.stopping = False
                    self._set_buttons()
                    self.status_var.set("Остановлено")
                    self._append("SYS", "Остановлено.")
        except queue.Empty:
            pass

        if self.waiting_for_python_since is not None and \
                time.time() - self.waiting_for_python_since > PYTHON_READY_TIMEOUT_S:
            self._append("SYS", f"Python не сообщил о готовности за {PYTHON_READY_TIMEOUT_S} с — запускаю Node всё равно.")
            self._start_node()

        self.after(100, self._poll_output)

    def _handle_line(self, tag: str, line: str) -> None:
        if tag == "PY" and line.startswith("[metrics] "):
            try:
                self._on_metrics(json.loads(line[len("[metrics] "):]))
            except json.JSONDecodeError:
                pass
            return  # в лог не пишем: для этого есть читаемая строка "[ai] шаг ..."

        self._append(tag, line, error=any(word in line for word in ERROR_WORDS))

        if tag == "PY" and PYTHON_READY_MARKER in line and self.waiting_for_python_since is not None \
                and self.active and not self.stopping:
            self._start_node()

    def _handle_exit(self, tag: str, code: int) -> None:
        if tag == "SRV":
            self._append("SYS", f"Сервер завершился (код {code}).", error=code not in (0, None))
            self._set_buttons()
            return
        self._append("SYS", f"{'Python' if tag == 'PY' else 'Node'} завершился (код {code}).", error=code not in (0, None))
        if not self.active or self.stopping:
            return
        if not self.restart_var.get():
            self._append("SYS", "Автоперезапуск выключен — нажми Стоп/Старт вручную.")
            return
        self._append("SYS", f"Перезапуск через {RESTART_DELAY_S} с...")
        restart = self._start_python if tag == "PY" else self._start_node
        self.after(RESTART_DELAY_S * 1000, lambda: self._restart_if_still_needed(tag, restart))

    def _restart_if_still_needed(self, tag: str, restart) -> None:
        proc = self.python_proc if tag == "PY" else self.node_proc
        if self.active and not self.stopping and (proc is None or not proc.running()):
            restart()

    def send_command(self, to: str, text: str) -> None:
        """Команда из строки команд — в stdin Node (runConsoleCommand в js/bot.js)."""
        who = "всем" if to == "all" else f"AI_{to}"
        self._append("CMD", f"{text}  → {who}")
        line = "cmd " + json.dumps({"to": to, "text": text}, ensure_ascii=False)
        if self._plugin_body():
            # Боты в плагине: команда — серверу (/mcbot cmd), ответ Python
            # на неё придёт строкой [cmd] в вывод Python.
            threading.Thread(target=self._plugin_worker, args=(f"mcbot cmd {to} {text}",), daemon=True).start()
        elif self.node_proc is None or not self.node_proc.send_line(line):
            self._append("SYS", "Node не запущен — команду некому выполнить. Нажми «Старт».", error=True)
            return
        # Сменил задачу всему рою ("task all tag") — её же и запомнить: при
        # следующем запуске лаунчер выберет её сам (автор дважды перезапускал
        # и попадал в mix вместо салок).
        words = text.lstrip("!").split()
        if words[:1] == ["task"] and (to == "all" or "all" in words[1:-1]) and words[-1] in self.task_box["values"]:
            self.task_var.set(words[-1])
            self._save_settings(task=words[-1])

    def _append(self, tag: str, text: str, error: bool = False) -> None:
        prefix = {"PY": "[PY] ", "JS": "[JS] ", "SRV": "[SRV] ", "SYS": "[--] ", "CMD": "[>>] "}[tag]
        tags = (tag, "ERR") if error else (tag,)
        if tag == "JS" and text.startswith("[cmd]"):
            tags = (tag, "CMD")  # ответ бота на команду — тем же цветом, что и команда
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, prefix + text + "\n", tags)
        # Ограничиваем размер лога — за сутки обучения строк будет очень много.
        excess = int(self.log.index("end-1c").split(".")[0]) - MAX_LOG_LINES
        if excess > 0:
            self.log.delete("1.0", f"{excess + 1}.0")
        self.log.configure(state=tk.DISABLED)
        self.log.see(tk.END)

    def _apply_filters(self) -> None:
        # elide скрывает текст с тегом, не удаляя его — галочку можно вернуть.
        self.log.tag_configure("PY", elide=not self.show_py.get())
        self.log.tag_configure("JS", elide=not self.show_js.get())
        self.log.tag_configure("SRV", elide=not self.show_srv.get())

    def _clear_log(self) -> None:
        self.log.configure(state=tk.NORMAL)
        self.log.delete("1.0", tk.END)
        self.log.configure(state=tk.DISABLED)

    # --- метрики и график ----------------------------------------------------

    def _on_metrics(self, metrics: dict) -> None:
        self.metrics_history.append(metrics)
        del self.metrics_history[:-MAX_CHART_POINTS]

        lines = [f"Ботов: {metrics.get('bots')} · {metrics.get('states_per_sec')} состояний/с"]
        for task, m in metrics.get("tasks", {}).items():
            trend = self._trend(task, m.get("score"))
            lines.append(
                f"{task} ({m.get('bots')} бот.): {m.get('summary') or '—'} {trend}"
                f" · случайных действий {m.get('epsilon', 0):.0%}"
            )
        self.stats_var.set("\n".join(lines))
        self._draw_chart()

    def _trend(self, task: str, score) -> str:
        """Стрелка "лучше/хуже" — сравнение с несколькими прошлыми замерами
        этой же задачки (с одним — слишком шумно)."""
        if score is None:
            return ""
        previous = [
            m["tasks"][task]["score"] for m in self.metrics_history[-6:-1]
            if task in m.get("tasks", {}) and m["tasks"][task].get("score") is not None
        ]
        if not previous:
            return ""
        base = sum(previous) / len(previous)
        if score > base * 1.1 + 0.5:
            return "↑"
        if score < base * 0.9 - 0.5:
            return "↓"
        return "→"

    def _draw_chart(self) -> None:
        """Главная цифра каждой задачки (TrainingModule.summarize) во
        времени. У задачек разные единицы (проценты, штуки за 10 минут),
        поэтому каждая линия в своём масштабе: низ — ноль, верх — лучшее
        значение этой задачки за всё время. Выше — лучше."""
        canvas = self.chart
        canvas.delete("all")
        width, height = canvas.winfo_width(), canvas.winfo_height()
        history = self.metrics_history
        if len(history) < 2 or width < 50 or height < 50:
            canvas.create_text(width // 2, height // 2, text="График появится после 2-х замеров (~1 минута)",
                               fill=self.palette["muted"])
            return

        left, right, top, bottom = 20, width - 20, 30, height - 16

        def x_of(i: int) -> float:
            return left + i / (len(history) - 1) * (right - left)

        canvas.create_line(left, bottom, right, bottom, fill=self.palette["grid"])
        canvas.create_text(right, bottom + 2, text="сейчас", anchor=tk.NE, fill=self.palette["muted"], font=("Consolas", 8))

        tasks = []
        for m in history:
            for task in m.get("tasks", {}):
                if task not in tasks:
                    tasks.append(task)

        legend_x = left
        for task in tasks:
            points = [
                (i, m["tasks"][task]["score"]) for i, m in enumerate(history)
                if task in m.get("tasks", {}) and m["tasks"][task].get("score") is not None
            ]
            if not points:
                continue
            color = TASK_COLORS.get(task, "#333333")
            best = max(max(score for _, score in points), 1e-6)
            coords = []
            for i, score in points:
                coords += [x_of(i), bottom - max(score, 0.0) / best * (bottom - top)]
            if len(coords) >= 4:
                canvas.create_line(*coords, fill=color, width=2)
            else:
                canvas.create_oval(coords[0] - 3, coords[1] - 3, coords[0] + 3, coords[1] + 3, fill=color, outline="")

            label = f"{task}: {points[-1][1]:g}"
            canvas.create_line(legend_x, 12, legend_x + 16, 12, fill=color, width=3)
            canvas.create_text(legend_x + 20, 12, text=label, anchor=tk.W, font=("Consolas", 9), fill=self.palette["fg"])
            legend_x += 40 + 7 * len(label)


def main():
    app = LauncherApp()
    app.mainloop()


if __name__ == "__main__":
    main()
