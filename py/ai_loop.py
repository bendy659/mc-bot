"""Главный цикл ИИ-контроллера.

BIND-ит оба ZMQ-порта (поэтому должен запускаться раньше node js/bot.js),
затем на каждое пришедшее состояние:
  1. дёргает обучающий модуль ("задачку") этого бота (py/training_modules/):
     on_tick может обновить его цель, compute_reward считает награды за
     прошлый переход — по отдельной на каждый канал (ноги/голова/руки).
     Ядро само не знает, что такое "награда за подход к цели" — это
     ответственность модуля;
  2. если цель модуля поменялась — шлёт в Node {"type":"set_target"};
  3. кодирует наблюдение + контекст (задача, прошлые действия каналов) и
     дописывает его в стек кадров (память о движении);
  4. МОЗГ ЭТОЙ ЗАДАЧКИ (py/dqn.py: TaskBrain — свои сети каналов, своя
     память опыта, свой epsilon) выбирает действия всех каналов, они
     уходят в Node одним сообщением и выполняются одновременно;
  5. кладёт переход в n-step накопитель -> память опыта мозга задачки.

Мозги создаются по мере надобности (когда какой-то бот впервые получил
задачку) и живут в data/brains/<задачка>/<канал>.pt — переключение задачек
ничего не затирает.

Обучение — в своём потоке (_learner_loop), непрерывно и со своим
CUDA-потоком: ответы ботам его не ждут (раньше учились урывками между
тиками в том же потоке, и с 14 ботами 44% ответов опаздывали — шаг
обучения ~50 мс, а пачка состояний ждала его конца). Ответы идут по копиям
сетей (TaskBrain: actor_net), которые догоняют обучаемые каждые
actor_sync_every шагов. Каждое состояние добавляет мозгу своей задачки
learn_steps_per_tick "кредита" обучения — учимся не чаще, чем приходит
опыт; learn_duty — какую долю времени видеокарта может учиться (на ней же
игра автора). Мозги на диск пишет тоже поток обучения — по запросу.

Задача по умолчанию — флаг --task (он же старый --module), mix — разные
задачки разным ботам, tag — салки-"заражение" (py/tag_game.py: судья
выбирает водящего — задачка chase, остальные убегают — flee, осаленный
тоже водит); на лету — команда
!task <имя> / !task all <имя>.
"""

import argparse
import contextlib
import csv
import gc
import json
import os
import math
import random
import shutil
import signal
import sys
import threading
import time
from collections import deque
from pathlib import Path

import torch
import zmq

from config import CONFIG
from protocol import CHANNEL_NAMES, IDLE_ACTIONS
from state_encoder import FrameStacker, encode_state, scalar_dim, unpack_vision, vision_channels
from dqn import TaskBrain
from demo_loader import action_indices, load_demos_into
from bedwars_game import BEDWARS, LOST, WON, BedwarsGame, clock as bedwars_clock
from bridge_course import BridgeCourse, fill_command
from hunt_game import HUNT, KILLED, HuntGame
from tag_game import CHASE, FLEE, TERMINAL_OUTCOMES, TagGame
from training_server import TrainingServer
from training_modules import DEFAULT_MODULE, MIX, MODULES, TAG, create_module, task_index
from training_modules.bridge import BRIDGE, CROSSED, FELL
from training_modules.targets import bot_index, bot_name, times

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
BRAINS_DIR = DATA_DIR / "brains"
METRICS_CSV = DATA_DIR / "metrics.csv"


def task_gamma(task: str, config: dict) -> float:
    """Горизонт задачки: modules.<задачка>.gamma, иначе train.gamma."""
    return config["modules"].get(task, {}).get("gamma", config["train"]["gamma"])


def brain_dir(task: str) -> Path:
    return BRAINS_DIR / task


def bc_path(channel: str) -> Path:
    return DATA_DIR / f"bc_{channel}.pt"


class NStepAccumulator:
    """Копит последние n переходов одного бота и отдаёт в память мозга
    n-step переходы: сумма наград за n шагов (с дисконтом) + бутстрап от
    состояния через n шагов с множителем gamma^n.

    Зачем: при обычном одношаговом TD награда "+100 за цель" за один
    проход обучения доползает назад только на один шаг. С n=3 — сразу на
    три, и связь "повернул к цели -> через секунду дошёл" выучивается
    заметно быстрее."""

    def __init__(self, n: int, gamma: float):
        self.n = max(1, n)
        self.gamma = gamma
        self.items: deque = deque()  # (vision, scalars, action_indices, rewards_list, demo)

    def append(self, vision, scalars, actions: list, rewards: list, demo: bool = False) -> None:
        """demo — действие выбрал учитель (py/sim/teacher.py), а не сеть:
        такой переход идёт в демо-память мозга (DQfD: margin-loss тянет сеть
        к действию учителя, и демо не вытесняются живым опытом)."""
        self.items.append((vision, scalars, actions, rewards, demo))

    def emit_ready(self, brain, next_vision, next_scalars) -> None:
        """Если накопилось n переходов — отдать самый старый с бутстрапом
        от текущего состояния (оно ровно через n шагов после него)."""
        if len(self.items) < self.n:
            return
        self._push(brain, 0, next_vision, next_scalars, done=False)
        self.items.popleft()

    def flush(self, brain, next_vision, next_scalars, done: bool) -> None:
        """Отдать всё накопленное (конец эпизода или смена задачи). Для
        каждого перехода — возврат до конца накопленного куска; при done
        бутстрапа нет, иначе — от next (оно сразу за последним переходом)."""
        for start in range(len(self.items)):
            self._push(brain, start, next_vision, next_scalars, done)
        self.items.clear()

    def _push(self, brain, start: int, next_vision, next_scalars, done: bool) -> None:
        vision, scalars, actions, _, demo = self.items[start]
        returns = [0.0] * len(CHANNEL_NAMES)
        discount = 1.0
        for offset in range(start, len(self.items)):
            rewards = self.items[offset][3]
            for c in range(len(returns)):
                returns[c] += discount * rewards[c]
            discount *= self.gamma
        remember = brain.remember_demo if demo else brain.remember
        remember(vision, scalars, actions, returns, next_vision, next_scalars, done, discount)

    def clear(self) -> None:
        self.items.clear()


class BotSession:
    """Всё, что у каждого бота роя своё: задача (экземпляр модуля со своей
    целью и счётчиками), стек кадров, прошлый переход, n-step накопитель.
    Мозги задачек при этом ОБЩИЕ для всех ботов роя — весь опыт роя по
    задачке стекается в её обучение."""

    def __init__(self, session_id: int, task_name: str, config: dict):
        self.id = session_id
        self.config = config
        self.mix_tasks = config["train"].get("mix_tasks", ["walking", "looking", "follow", "gathering"])
        self.mix_rotate_ticks = config["train"].get("mix_rotate_ticks", 3000)
        self.stacker = FrameStacker(config)
        self.last_sent_target = None  # чтобы не слать одно и то же set_target
        # Показательный режим (!greedy): бот действует без случайных
        # действий — видно, чему мозг уже научился, пока остальные боты
        # роя продолжают исследовать. Его опыт тоже идёт в обучение
        # (DQN учится off-policy, ему всё равно, кто выбирал действия).
        self.greedy = False
        # Задачка bridge: бот на старте своей дорожки (судья его туда поставил)
        # и когда слали команды новой попытки.
        self.bridge_ready = False
        self.bridge_sent = 0.0
        self.bridge_wait = 0   # состояний с тех пор, как слали команды попытки
        self.bridge_ticks = 0  # решений в этой попытке (предел — BridgeModule.attempt_seconds)
        self.bridge_layout = None  # раскладка попытки (bridge_course.BridgeLayout): своя каждый раз
        self.assign(task_name)

    def assign(self, task_name: str) -> None:
        """Задача или режим. mix: боты стартуют с разных задач (по номеру
        бота) и каждые mix_rotate_ticks своих тиков переходят к следующей —
        за прогон каждый бот тренирует всё. tag: салки — бот начинает
        убегающим, а водящего выбирает судья (AILoop._tag_turn). hunt —
        охота: бот охотится, а если судья выбрал его целью — убегает (flee,
        AILoop._hunt_turn)."""
        self.mix = task_name == MIX
        self.tag = task_name == TAG
        self.hunting = task_name == HUNT
        if self.mix:
            self.mix_step = self.id - 1
            self.set_task(self.mix_tasks[self.mix_step % len(self.mix_tasks)])
        elif self.tag:
            self.set_task(FLEE)
        else:
            self.set_task(task_name)

    def next_mix_task(self) -> str:
        self.mix_step += 1
        return self.mix_tasks[self.mix_step % len(self.mix_tasks)]

    def set_task(self, task_name: str) -> None:
        self.task_name = task_name
        self.ticks_in_task = 0
        # n-step накопитель — со своей gamma у задачки (task_gamma). Старый
        # к этому моменту уже отдан в память прежней задачки (_switch_task).
        self.nstep = NStepAccumulator(self.config["train"].get("n_step", 3), task_gamma(task_name, self.config))
        self.task = task_index(task_name)
        self.module = create_module(task_name, self.config)
        self.clear_transition()

    def clear_transition(self) -> None:
        self.prev = None          # (stacked_vision, scalars) прошлого состояния
        self.prev_state = None
        self.prev_actions = dict(IDLE_ACTIONS)  # {канал: имя} — выбранные из prev
        self.prev_demo = False                   # их выбрал учитель, а не сеть


class TaskWindow:
    """Счётчики одной задачки за окно метрик (обнуляются после вывода)."""

    def __init__(self):
        self.ticks = 0
        self.rewards = {channel: 0.0 for channel in CHANNEL_NAMES}
        self.losses: dict[str, float] = {}
        self.learns = 0
        self.learn_seconds = 0.0
        self.events: dict[str, int] = {}
        # Какие действия выбирали каналы (отдельно — боты в !greedy): по этому
        # видно, не "зациклилась" ли выученная стратегия на одном действии.
        self.actions: dict[str, dict[str, int]] = {}
        self.greedy_actions: dict[str, dict[str, int]] = {}

    def count_actions(self, actions: dict, greedy: bool) -> None:
        table = self.greedy_actions if greedy else self.actions
        for channel, name in actions.items():
            counts = table.setdefault(channel, {})
            counts[name] = counts.get(name, 0) + 1

    def add_events(self, events: dict) -> None:
        for name, value in events.items():
            self.events[name] = self.events.get(name, 0) + value


class AILoop:
    def __init__(self, task_name: str):
        self.config = CONFIG
        self.running = True
        self.default_task = task_name

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[ai] Устройство: {self.device}")
        res_x, res_y = self.config["vision"]["resolution"]
        self.vision_shape = (vision_channels(self.config), res_y, res_x)
        self.scalar_dim = scalar_dim(self.config)
        self.brains: dict[str, TaskBrain] = {}
        self._last_save: dict[str, float] = {}
        self._warn_legacy_files()

        train_cfg = self.config["train"]
        # Может быть дробным: 0.5 = шаг обучения на каждые 2 состояния.
        # Шаг маленьких сетей стоит ~20 мс почти независимо от батча (время
        # уходит на запуск CUDA-ядер, не на математику), поэтому выгоднее
        # учиться реже, но большим батчем.
        self.learn_steps_per_tick = train_cfg.get("learn_steps_per_tick", 0.5)
        # Потолок "кредита" обучения мозга: если GPU не успевает, лишнее
        # обучение пропускается, а не копится долгом.
        self.max_learn_credit = train_cfg.get("max_learn_per_cycle", 8)
        # Автосохранение: всех мозгов — каждые N обработанных состояний, и
        # мозга задачки — на смерти бота (не чаще раза в минуту).
        self.autosave_ticks = train_cfg.get("autosave_ticks", 5000)
        self.metrics_every_seconds = train_cfg.get("metrics_every_seconds", 30)
        self.tick_seconds = train_cfg.get("tick_rate_ms", 150) / 1000.0
        self._ticks_handled = 0
        self._next_autosave = self.autosave_ticks
        # Поток обучения (_learner_loop): доля времени видеокарты под обучение,
        # его счётчики для метрик и мозги, которые он должен сохранить.
        self.learn_duty = train_cfg.get("learn_duty", 0.85)
        self._learner: threading.Thread | None = None
        self._learn_stats: dict[str, dict] = {}
        self._save_requests: set[str] = set()
        self._shared_lock = threading.Lock()  # _learn_stats и _save_requests
        # Пока отвечаем пачке ботов, обучение нового шага не начинает: два
        # потока Python делят один GIL, и шаг обучения рядом замедлял ответы.
        self._answering = threading.Event()
        # Два потока Python делят один GIL: по умолчанию поток держит его до
        # 5 мс — ответ ботам ждал бы шаг обучения. 1 мс — ответы быстрее.
        sys.setswitchinterval(0.001)

        # Сессии по bot_id — создаются лениво, при первом состоянии от бота.
        self.sessions: dict[int, BotSession] = {}
        # Судья салок — живёт всегда, играют в него только боты в режиме tag.
        self.tag_game = TagGame(self.config)
        self.tag_game.on_round_start = lambda: self._run_round_function(TAG)
        # Судья "Останови меня" (задачка hunt): кого ловить, началась ли охота.
        self.hunt = HuntGame(self.config)
        # Судья бедварса (py/bedwars_game.py): игры на картах Hypixel в мире задачки.
        self.bedwars = BedwarsGame(self.config)
        self._bed_check_at = 0.0
        self._bedwars_warned = False
        # Учитель (только в симуляции, py/train_tag_sim.py): функция (сессия,
        # состояние) -> действия или None. Если вернула действия — бот делает
        # их, а переход идёт в память как демонстрация.
        self.teacher = None
        # Наш сервер-арена (server/, py/training_server.py): RCON, арена,
        # табло. Не отвечает (чужой сервер) — работаем без него, как раньше.
        self.server = TrainingServer.connect(self.config)
        if self.server is not None:
            self.server.setup()
            self.tag_game.direct_commands = True  # команды салок — по RCON, без op и очереди
        # Набор команд задачи (server/functions/<задача>.mcfunction, идея
        # автора) — когда рой в сборе: боты заходят на сервер уже ПОСЛЕ
        # запуска Python, и @a в командах иначе не застал бы ботов.
        self._pending_function: str | None = task_name if self.server is not None else None
        self._first_join: float | None = None
        # Трасса задачки bridge (строится при первом боте на ней) и предупреждение
        # "мира задачки нет" — один раз.
        self._course: BridgeCourse | None = None
        self._course_warned = False

        # Окно метрик: копится между выводами, потом обнуляется.
        self._windows: dict[str, TaskWindow] = {}
        self._window_started = time.time()
        self._window_states = 0
        # Строка "[metrics] {json}" нужна только лаунчеру (py/launcher.py)
        # для графиков — в обычной консоли она лишний шум.
        self.emit_metrics_json = os.environ.get("MCBOT_METRICS") == "1"

        context = zmq.Context()
        self.state_socket = context.socket(zmq.PULL)
        self.state_socket.bind(f"tcp://127.0.0.1:{self.config['zmq']['node_to_py']}")
        self.action_socket = context.socket(zmq.PUSH)
        self.action_socket.bind(f"tcp://127.0.0.1:{self.config['zmq']['py_to_node']}")
        self.state_socket.set(zmq.RCVTIMEO, 200)  # чтобы цикл мог учиться и проверять running

        # Мозг задачки по умолчанию готовим сразу (загрузка весов и демо-записей
        # может занять время — лучше до того, как придут боты).
        if task_name == MIX:
            first_tasks = self.config["train"].get("mix_tasks", [])
        elif task_name == TAG:
            first_tasks = [CHASE, FLEE]
        else:
            first_tasks = [task_name]
        for task in first_tasks:
            self._brain(task)
        print(f"[ai] Задача по умолчанию: {task_name}.")
        # Всё, что создано к этому моменту (torch, библиотеки, сети), живёт
        # до конца процесса — убираем это из обхода сборщика мусора. Иначе
        # каждая полная сборка обходила бы сотни тысяч объектов (замерено:
        # ~45 мс только на библиотеки) и замораживала ответы ботам.
        gc.freeze()

    # --- мозги задачек --------------------------------------------------------

    def _brain(self, task: str) -> TaskBrain:
        if task not in self.brains:
            brain = TaskBrain(task, self.config, self.vision_shape, self.scalar_dim, self.device)
            self._load_brain(brain)
            # DQfD: демо-переходы — в память мозга до начала живого обучения,
            # награды за них считает модуль этой же задачки (demo_loader.py).
            if self.config["train"].get("demo_preload", True) and brain.learners:
                load_demos_into(brain, task, self.config)
            self.brains[task] = brain
        return self.brains[task]

    def _load_brain(self, brain: TaskBrain) -> None:
        """По каждому каналу: свой чекпоинт главнее всего (это уже обученный
        мозг этой задачки), иначе warm start — веса мозга родственной
        задачки (train.warm_start_from, например follow <- walking: ходьба
        та же), иначе веса behavior cloning, иначе — с нуля. Несовместимый
        чекпоинт (менялись действия/наблюдение) сбрасывает только свой канал."""
        task = brain.task
        loaded, warm, grown = [], [], []
        source = self.config["train"].get("warm_start_from", {}).get(task)
        for channel in brain.learners:
            path = brain_dir(task) / f"{channel}.pt"
            if path.exists():
                try:
                    notes = brain.load_channel(channel, path)
                    loaded.append(channel)
                    if notes:
                        grown.append(f"{channel}: {', '.join(notes)}")
                    continue
                except (RuntimeError, KeyError):
                    # Не удаляем: обученный мозг — это часы игры. Откладываем
                    # в data/brains_old/, вдруг настройки вернут.
                    backup = DATA_DIR / "brains_old" / f"{task}_{time.strftime('%Y%m%d_%H%M%S')}"
                    backup.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(path), backup / path.name)
                    print(f"[ai] {task}/{channel}: чекпоинт несовместим с текущей архитектурой — учится заново "
                          f"(старый отложен в {backup.relative_to(ROOT)}).")

            candidates = []
            if source:
                candidates.append((brain_dir(source) / f"{channel}.pt", f"мозг {source}"))
            candidates.append((bc_path(channel), "запись геймплея"))
            for candidate, label in candidates:
                if not candidate.exists():
                    continue
                try:
                    notes = brain.warm_start_channel(channel, candidate)
                    warm.append(f"{channel} <- {label}" + (f" (доращён: {', '.join(notes)})" if notes else ""))
                    break
                except (RuntimeError, KeyError):
                    print(f"[ai] {task}/{channel}: {candidate.name} несовместим — пропускаю.")

        channels = ", ".join(brain.learners) or "—"
        state = []
        if loaded and brain.rules_changed:
            state.append(f"загружен, но правила задачки поменялись (версия {brain.rules_version}): "
                         f"навыки оставлены, случайные действия заново")
        elif loaded:
            state.append(f"загружен ({brain.steps} шагов, epsilon {brain.epsilon():.2f})")
        if grown:
            state.append("доращён (" + "; ".join(grown) + ")")
        if warm:
            state.append("warm start: " + "; ".join(warm))
        if not loaded and not warm:
            state.append("с нуля")
        print(f"[ai] Мозг задачки {task}: сети {channels}; {', '.join(state)}.")

    def _warn_legacy_files(self) -> None:
        legacy = sorted(p.name for p in DATA_DIR.glob("net_*.pt")) + \
            sorted(p.name for p in DATA_DIR.glob("dqn_*.pt")) + \
            (["bc_weights.pt"] if (DATA_DIR / "bc_weights.pt").exists() else [])
        if legacy:
            print(
                f"[ai] Найдены веса старой архитектуры (общий мозг на все задачки): {', '.join(legacy)}. "
                "Они больше не используются — можно удалить."
            )

    # --- главный цикл ------------------------------------------------------

    def run(self):
        self._learner = threading.Thread(target=self._learner_loop, name="learner", daemon=True)
        self._learner.start()
        print("[ai] Цикл запущен, жду состояний от бота...")
        while self.running:
            try:
                raw = self.state_socket.recv()
            except zmq.Again:
                raw = None  # тишина — самое время поучиться
            if raw is not None:
                # Дочитываем всю пачку (Node шлёт состояния всех ботов подряд):
                # ответить всем ботам разом — один проход сети на мозг, —
                # учиться потом.
                messages = [json.loads(raw)]
                # Хвост пачки ждём активно, до 2 мс после последнего
                # сообщения: poll(1) в Windows спит по грубому таймеру
                # (~15 мс) — ответы ботам из-за этого опаздывали.
                tail_deadline = time.perf_counter() + 0.002
                while True:
                    if self.state_socket.poll(0):
                        messages.append(json.loads(self.state_socket.recv()))
                        tail_deadline = time.perf_counter() + 0.002
                    elif time.perf_counter() >= tail_deadline:
                        break
                self._answering.set()
                try:
                    self.handle_messages(messages)
                finally:
                    self._answering.clear()

            if self.server is not None:
                self.server.publish(time.time())  # табло "Мотивация"
            if time.time() - self._window_started >= self.metrics_every_seconds:
                self._report_metrics()

        self.shutdown()

    def _learner_loop(self) -> None:
        """Поток обучения: шаг за шагом — мозг с наибольшим кредитом, в
        своём CUDA-потоке; между шагами — сохранения по запросу."""
        stream = torch.cuda.Stream() if self.device.type == "cuda" else None
        while self.running:
            self._save_requested()
            while self._answering.is_set() and self.running:
                time.sleep(0.0005)  # ответы ботам важнее — подождать конца пачки
            with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
                spent = self.learn_one()
            if spent is None:
                time.sleep(0.005)  # учить нечего — подождать опыта
            elif self.learn_duty < 1.0:
                # Видеокарта рисует ещё и игру автора: оставить ей время.
                time.sleep(spent * (1.0 / self.learn_duty - 1.0))

    def learn_one(self) -> float | None:
        """Один шаг обучения мозга с наибольшим кредитом. Возвращает, сколько
        он длился (секунды), или None — учить нечего."""
        brain = max(list(self.brains.values()), key=lambda b: b.learn_credit, default=None)
        if brain is None or brain.learn_credit < 1.0:
            return None
        brain.learn_credit -= 1.0
        started = time.time()
        losses = brain.learn()
        if losses is None:
            brain.learn_credit = 0.0  # опыта ещё мало — копить кредит незачем
            return None
        spent = time.time() - started
        with self._shared_lock:
            stats = self._learn_stats.setdefault(brain.task, {"learns": 0, "seconds": 0.0, "losses": {}})
            stats["learns"] += 1
            stats["seconds"] += spent
            for channel, loss in losses.items():
                stats["losses"][channel] = stats["losses"].get(channel, 0.0) + loss
        return spent

    def train_while_idle(self) -> None:
        """Выучить весь накопленный кредит сразу — для тестов без потока
        обучения (в работе учится _learner_loop)."""
        while self.learn_one() is not None:
            pass

    def _save_requested(self) -> None:
        with self._shared_lock:
            tasks, self._save_requests = self._save_requests, set()
        for task in tasks:
            brain = self.brains.get(task)
            if brain is not None and brain.learners:
                brain.save(brain_dir(task))

    def handle_messages(self, messages: list[dict]) -> None:
        """Пачка сообщений от Node — обычно состояния всех ботов за тик. Для
        каждого состояния сначала всё, что до выбора действия (награда за
        прошлый переход, память), потом действия всем ботам одной задачки —
        одним проходом сети (TaskBrain.select_actions_batch): по проходу на
        бота ответы роя опаздывали, пока видеокарта занята ещё и игрой."""
        self._run_task_function_when_ready()
        # Два состояния одного бота в пачке (Python не успевал) — отвечаем
        # только на последнее: тело всё равно выполнит лишь свежий ответ
        # (plugin Swarm.receive, js/bot.js), а ответ на старое записался бы в
        # опыт действием, которого не было. Смерть не пропускаем — на ней
        # закрывается эпизод.
        last = {message.get("bot_id", 1): index for index, message in enumerate(messages)
                if message.get("type") == "state"}
        messages = [message for index, message in enumerate(messages)
                    if message.get("type") != "state" or message.get("dead")
                    or last.get(message.get("bot_id", 1)) == index]
        pending: list[tuple] = []
        for message in messages:
            if message.get("type") == "command":
                self._decide(pending)
                pending = []
                self.handle_command(message)
                continue
            session = self._session(message.get("bot_id", 1))
            if any(decision[0] is session for decision in pending):
                # Второе состояние того же бота в пачке (Python не успевал) —
                # сначала ответить на первое: второе опирается на его действия.
                self._decide(pending)
                pending = []
            decision = self._prepare(message)
            if decision is not None:
                pending.append(decision)
        self._decide(pending)

    def handle_state(self, state: dict) -> None:
        """Одно состояние (для тестов и простых случаев)."""
        self.handle_messages([state])

    def _prepare(self, state: dict) -> tuple | None:
        """Всё, что до выбора действия. Возвращает (сессия, мозг, стек кадров,
        скаляры, состояние) — боту нужно действие, или None — не нужно
        (мёртв, водящий "считает до пяти")."""
        unpack_vision(state)  # зрение байтами -> cells (до модулей: они читают клетки)
        session = self._session(state.get("bot_id", 1))
        if session.tag:
            self._tag_turn(session, state)
        elif session.hunting:
            self._hunt_turn(session, state)
        elif session.task_name == BRIDGE:
            self._bridge_turn(session, state)
        elif session.task_name == BEDWARS:
            self._bedwars_turn(session, state)
        brain = self._brain(session.task_name)
        module = session.module
        module.on_tick(state)
        self._sync_target(session)
        self._ticks_handled += 1
        self._window_states += 1

        # Контекст текущего состояния — действия, выбранные на прошлом тике
        # (они и выполнялись последние 150 мс). Сеть видит мир таким, каким
        # его показывает задачка (observe: например, цель -> точка маршрута).
        vision, scalars = encode_state(module.observe(state), self.config, session.task, session.prev_actions)

        if state.get("dead"):
            # Терминальное состояние: переход закрывается наградой за смерть
            # без выбора действия. Как next передаём прошлый стек той же
            # формы — содержимое не влияет на TD-цель (done=1 обнуляет
            # бутстрап), но размерности в памяти обязаны совпадать.
            # Node шлёт dead=1 каждый тик, пока бот не респавнится (~1 с) —
            # эпизод закрываем только на первом таком тике.
            if session.prev is None:
                return None
            self._close_transition(session, state)
            session.nstep.flush(brain, session.prev[0], session.prev[1], done=True)
            session.clear_transition()
            session.stacker.frames = []  # после респавна стек начинается заново
            module.reset(state)
            self._save_brain_throttled(brain)  # граница эпизода — надёжное место для чекпоинта
            return None

        stacked = session.stacker.push(vision)

        waiting = (session.tag and self.tag_game.is_frozen(session.id)) or \
            (session.hunting and self.hunt.waiting(session.id)) or \
            (session.task_name == BRIDGE and not session.bridge_ready) or \
            (session.task_name == BEDWARS and self.bedwars.waiting(session.id))
        if waiting:
            # Водящий "считает до пяти", пауза между раундами салок или охота
            # hunt ещё не начата (ждём !start): бот стоит, решений не
            # принимает — и опыта с этих тиков нет.
            # Незакрытый переход (раунд кончился по времени — это обрыв, а
            # не конец эпизода) уходит в память с бутстрапом от этого
            # состояния. Кадры — с чистого листа: в паузе всех разбрасывает
            # по арене, старые кадры — не про новое место.
            if session.prev is not None:
                self._close_transition(session, state)
                session.nstep.flush(brain, stacked, scalars, done=False)
            session.clear_transition()
            session.stacker.frames = []
            self._send_actions(session, dict(IDLE_ACTIONS), state)
            return None

        if session.prev is not None:
            self._close_transition(session, state)
            session.nstep.emit_ready(brain, stacked, scalars)
        return session, brain, stacked, scalars, state

    def _decide(self, pending: list[tuple]) -> None:
        """Действия всем ботам из pending: по одному проходу сети на мозг."""
        by_brain: dict[int, list[tuple]] = {}
        for decision in pending:
            by_brain.setdefault(id(decision[1]), []).append(decision)
        for group in by_brain.values():
            brain = group[0][1]
            chosen = brain.select_actions_batch(
                [decision[2] for decision in group],
                [decision[3] for decision in group],
                [decision[0].greedy for decision in group],
            )
            for decision, actions in zip(group, chosen):
                demo = False
                if self.teacher is not None:
                    taught = self.teacher(decision[0], decision[4])
                    if taught is not None:
                        actions, demo = taught, True
                self._finish(*decision, actions, demo)

    def _finish(self, session: BotSession, brain: TaskBrain, stacked, scalars, state: dict, actions: dict,
                demo: bool = False) -> None:
        """Действие выбрано: отправить боту и запомнить переход."""
        self._send_actions(session, actions, state)

        session.prev = (stacked, scalars)
        session.prev_state = state
        session.prev_actions = actions
        session.prev_demo = demo
        self._window(session.task_name).count_actions(actions, session.greedy)
        brain.learn_credit = min(brain.learn_credit + self.learn_steps_per_tick, self.max_learn_credit)

        session.ticks_in_task += 1
        if session.mix and session.ticks_in_task >= session.mix_rotate_ticks:
            next_task = session.next_mix_task()
            print(f"[ai] Бот {session.id}: mix -> {next_task}")
            self._switch_task(session, next_task, keep_mode=True)

        # По порогу, не по остатку от деления: счётчик растёт в _prepare сразу
        # на всю пачку состояний, а сюда приходит каждый бот пачки — с
        # остатком мозги сохранялись по разу на бота (12 раз подряд) и не
        # каждые autosave_ticks, а когда повезёт с кратностью размеру пачки.
        if self._ticks_handled >= self._next_autosave:
            self._next_autosave = self._ticks_handled + self.autosave_ticks
            self.save()

    def _send_actions(self, session: BotSession, actions: dict, state: dict) -> None:
        # tick — чтобы Node отличил ответ на ЭТОТ тик от запоздавшего прошлого;
        # tag_ids — кого из игроков можно бить (водящему в салках —
        # убегающих, в охоте hunt — цель, всем остальным — никого).
        if session.tag:
            taggable = self.tag_game.taggable_ids(session.id)
        elif session.hunting and session.id != self.hunt.target_bot:
            taggable = self.hunt.taggable_ids()
        elif session.hunting:
            taggable = self.hunt.hunter_ids()  # бот-цель отбивается от охотников
        elif session.task_name == BEDWARS:
            taggable = self.bedwars.enemy_ids(session.id)  # бедварс: только соперников
        else:
            taggable = []
        self.action_socket.send_json({
            "type": "action", "actions": actions, "bot_id": session.id, "tick": state.get("tick"),
            "tag_ids": taggable,
        })

    def _close_transition(self, session: BotSession, state: dict) -> None:
        """Награды за переход prev -> state и запись его в n-step накопитель."""
        rewards = session.module.compute_reward(session.prev_state, state, session.prev_actions)
        if self.server is not None:
            # Табло "Мотивация" — средняя по каналам награда бота.
            self.server.add_reward(bot_name(session.id, self.config), sum(rewards.values()) / len(rewards), time.time())
        reward_list = [rewards[channel] for channel in CHANNEL_NAMES]
        session.nstep.append(session.prev[0], session.prev[1], action_indices(session.prev_actions), reward_list,
                             session.prev_demo)

        window = self._window(session.task_name)
        window.ticks += 1
        for channel, value in rewards.items():
            window.rewards[channel] += value

    # --- салки ----------------------------------------------------------------

    def _tag_turn(self, session: BotSession, state: dict) -> None:
        """Ход судьи салок перед обработкой состояния бота: он узнаёт, где
        бот, решает, не осалили ли кого и не пора ли менять водящего, и
        говорит модулю роли, за кем бежать (или от кого). Что судья хочет
        сказать в чат ("Я догнал AI_3. Он водит.", команды пометки
        водящего) — уходит в чат игры от имени нужного бота."""
        if getattr(session.module, "unstick_requested", False):
            session.module.unstick_requested = False  # застоялся — судья накажет
            self.tag_game.punish_idle(session.id)
        change = self.tag_game.before_tick(session.id, session.task_name, state)
        if change is not None:
            outcome, role = change
            self._apply_tag_change(session, state, outcome, role)
        session.module.game_target = self.tag_game.target_for(session.id)
        for bot_id, text in self.tag_game.take_messages():
            self._chat(bot_id, text)
        if self.server is not None:
            for command in self.tag_game.take_commands():
                self.server.send(command)

    def _hunt_turn(self, session: BotSession, state: dict) -> None:
        """Ход судьи охоты: где цель и охотники (из состояния бота), умерла ли
        цель-бот, пора ли в auto начать новую охоту, какая у бота роль (цель
        убегает — flee, остальные охотятся — hunt) и цель модулю."""
        me = state["self"]
        self.hunt.see(state.get("humans") or [])
        blocks = (state.get("inventory") or {}).get("blocks")
        self.hunt.see_bot(session.id, me.get("entity_id"), (me["x"], me["y"], me["z"]), bool(state.get("dead")),
                          blocks=blocks)
        self.hunt.note_damage(session.id, state.get("damage_dealt") or [])
        hunting = [sid for sid, s in self.sessions.items() if s.hunting]
        if session.id == self.hunt.target_bot and state.get("dead"):
            name = bot_name(session.id, self.config)
            if self.hunt.died(name, [sid for sid in hunting if sid != session.id], self.hunt.last_hitter):
                print(f"[ai] Охота: бот-цель {name} остановлен.")
                self._chat(session.id, "Меня остановили!" + ("" if self.hunt.auto else " Ещё раз — !start."))
        chosen = self.hunt.next_auto(hunting, lambda sid: bot_name(sid, self.config))
        if chosen is not None:
            self._restore_arena()
            self._run_round_function(HUNT)
            print(f"[ai] Охота (auto): цель {bot_name(chosen, self.config)}.")
            self._chat(chosen, f"Теперь ловят меня! У меня {self.hunt.head_start:g} с форы.")
        role = self.hunt.role_of(session.id)
        if role != session.task_name:
            self._switch_task(session, role, keep_mode=True)
        if session.id in self.hunt.pending_kill:
            self.hunt.pending_kill.discard(session.id)
            if session.id == self.hunt.credited:
                session.module.count("kills")  # для сводки — одна остановка на всех
            self._end_episode(session, state, KILLED)
        session.module.game_target = self.hunt.threat_for() if role != HUNT else self.hunt.target_for()
        # Вход сети closest (state_encoder) и учитель: ближайшему к цели —
        # лезть за ней на столб, остальным — копать под ней.
        state["hunt_closest"] = role == HUNT and self.hunt.closest(session.id)

    def _bedwars_turn(self, session: BotSession, state: dict) -> None:
        """Ход судьи бедварса (py/bedwars_game.py): новая игра, когда прошлая
        кончилась (карта заново, команды по островам, наборы); где бот, умер
        ли (возрождение или выбывание), не в пустоте ли; целы ли кровати;
        победа. Боту — куда идти (цель от судьи), кого можно бить, события
        для наград. Играть можно только в мире задачки на нашем сервере (или
        в симуляции): обычный мир под бедварс не перестраиваем (автор)."""
        game = self.bedwars
        now = bedwars_clock()
        if game.want_new_game(now):
            players = sorted(sid for sid, other in self.sessions.items() if other.task_name == BEDWARS)
            if len(players) >= 2 and self._bedwars_world_ok():
                game.new_game(players, lambda sid: bot_name(sid, self.config), now)
                print(f"[ai] {game.describe()}")
        game.see_bot(session.id, state, now)
        game.economy_tick(now)
        # Прошлое решение рук — покупка (buy) — проводит судья; у генератора — ресурсы.
        game.collect_and_buy(session.id, state, (session.prev_actions or {}).get("hands"))
        state["bedwars_shop"] = game.at_shop(session.id)  # вход сети "shop": покупка сейчас сработает
        if game.started_at is not None and now >= self._bed_check_at:
            self._bed_check_at = now + 0.5
            for team_index, cells in game.beds_to_check(now):
                if not self.server.is_bed(game.world, cells):
                    game.bed_broken(team_index, now)
        game.check_end(now)
        for event in game.take_events(session.id):
            if event in (WON, LOST):
                session.module.game_events.append(event)
                self._end_episode(session, state, event)
                session.module.game_events.clear()  # переход уже закрыт (или его не было)
            else:
                session.module.game_events.append(event)
        session.module.game_target = game.target_for(session.id)
        session.module.enemy_ids = set(game.enemy_ids(session.id))
        session.module.role = game.role_of(session.id)      # учителю: защитник или атакующий
        session.module.own_bed = game.own_bed(session.id)
        session.module.enemy_bed = game.enemy_bed(session.id)
        session.module.own_spawn = game.own_spawn(session.id)
        for command in game.take_commands():
            self.server.send(command)

    def _bedwars_world_ok(self) -> bool:
        if self.server is not None and (getattr(self.server, "simulated", False) or self._task_world(BEDWARS)):
            return True
        if not self._bedwars_warned:
            self._bedwars_warned = True
            print("[ai] Бедварс: нужен наш сервер (RCON) с плагином и мир задачки (config.json bot.body = plugin, "
                  "bot.task_worlds.bedwars) — обычный мир не трогаю.")
        return False

    def _bridge_turn(self, session: BotSession, state: dict) -> None:
        """Судья моста (задачка bridge): у каждого бота своя дорожка трассы
        (py/bridge_course.py), цель — остров за пропастью. Каждая попытка —
        своя раскладка и свой вид моста (прямо, вверх, вниз, "Г"), старт в
        разных местах острова. Дошёл до острова цели — исход crossed (чем
        быстрее, тем больше награда: модулю — время попытки и длина пути),
        упал ниже островов — fell (конец попытки: награду или штраф даёт
        модуль). Перед каждой попыткой дорожка чистая, в инвентаре блоки,
        бот — на старте, лицом в случайную сторону; пока телепорт не дошёл,
        бот стоит (опыта нет). Упавшего возвращаем сразу, не дожидаясь смерти
        в пустоте, — без смерти и потери вещей."""
        course = self._bridge_course()
        lane = session.id - 1
        if course is None or lane >= course.lanes:
            return
        if state.get("dead"):
            session.bridge_ready = False
            return
        if not session.bridge_ready and time.time() - session.bridge_sent > 2.0:
            self._bridge_restart(session, course, lane)  # команды не дошли (или ещё не слали) — ещё раз
        layout = session.bridge_layout
        gx, gy, gz = layout.goal()
        session.module.game_target = {"x": gx, "y": gy, "z": gz}
        me = state["self"]
        if not session.bridge_ready:
            # Готов — стоит ровно на старте, и это состояние уже после команд:
            # застрявший у края бот мог оказаться рядом с новым стартом и до
            # телепорта (старт теперь каждый раз в другом месте острова).
            session.bridge_wait += 1
            if session.bridge_wait >= 2 and math.dist((me["x"], me["y"], me["z"]), layout.start()) < 0.5:
                session.bridge_ready = True
                session.bridge_ticks = 0
                session.module.reset(state)  # новая цель и место — прогресс к ней мерить заново
                session.module.count(f"start_{layout.kind}")  # для сводки: попыток по видам моста
            return
        session.bridge_ticks += 1
        if layout.on_goal_island(me["x"], me["y"], me["z"]):
            outcome = CROSSED
            seconds = session.bridge_ticks * self.tick_seconds
            session.module.crossed_seconds = seconds
            session.module.crossed_blocks = layout.blocks_to_cross()
            session.module.count("cross_ms", int(seconds * 1000))  # для сводки: сколько в среднем идёт переход
            session.module.count(f"crossed_{layout.kind}")
        elif course.fell(me["y"]):
            outcome = FELL
            session.module.count(f"fell_{layout.kind}")
        elif session.bridge_ticks > self._bridge_attempt_ticks(session):
            # Застрял (стоит у края, а луч в пустоте — не поставить и не шагнуть):
            # новая попытка. Не конец эпизода — обрыв: переход закроет ожидание
            # старта (с бутстрапом), как паузу между раундами салок.
            session.module.count("time_up")
            session.module.count(f"time_up_{layout.kind}")
            self._bridge_restart(session, course, lane)
            return
        else:
            return
        self._end_episode(session, state, outcome)
        self._bridge_restart(session, course, lane)

    def _bridge_attempt_ticks(self, session: BotSession) -> int:
        """Предел попытки: не меньше modules.bridge.attempt_seconds, а на
        длинном мосту — две нормы времени (за них и бонус за скорость
        кончается, BridgeModule.par_seconds)."""
        seconds = self.config["modules"].get(BRIDGE, {}).get("attempt_seconds", 20.0)
        seconds = max(seconds, 2 * session.module.par_seconds(session.bridge_layout.blocks_to_cross()))
        return int(seconds / self.tick_seconds)

    def _bridge_course(self) -> BridgeCourse | None:
        """Трасса моста — одна на запуск: в мире задачки (плагин создаёт его
        пустым, config.json bot.task_worlds) строится по RCON, в симуляции —
        уже в её мире. Обычный мир под мост не перестраиваем (автор): нет
        мира задачки — мост не строится."""
        if self._course is not None:
            return self._course
        simulated = getattr(self.server, "simulated", False)
        world = self._task_world(BRIDGE)
        if self.server is None or (world is None and not simulated):
            if not self._course_warned:
                self._course_warned = True
                print("[ai] Мост: нужен наш сервер-арена (RCON) с плагином и мир задачки "
                      "(config.json bot.body = plugin, bot.task_worlds.bridge) — обычный мир не трогаю.")
            return None
        lanes = max(self.config["bot"].get("count", 1), max(self.sessions, default=1))
        self._course = BridgeCourse(lanes)
        if not simulated:
            fills = self._course.clear_fills() + self._course.fills()  # сначала — мусор прошлых запусков прочь
            self.server.setup_task_world(world, [fill_command(fill, world) for fill in fills], self._course.bounds())
        return self._course

    def _task_world(self, task: str) -> str | None:
        if self.config["bot"].get("body") != "plugin":
            return None
        return self.config["bot"].get("task_worlds", {}).get(task)

    def _bridge_restart(self, session: BotSession, course: BridgeCourse, lane: int) -> None:
        """Новая попытка: своя раскладка (автор: "расстояние каждый раз
        одинаковое, хотелось бы побольше разнообразия") — дорожка заново, в
        инвентаре блоки, бот на старте, лицом в случайную сторону
        (развернуться к пропасти — тоже часть навыка)."""
        layout = course.layout(lane, randomize=True)
        session.bridge_layout = layout
        session.bridge_ready = False
        session.bridge_wait = 0
        name = bot_name(session.id, self.config)
        world = self._task_world(BRIDGE)
        blocks = self.config["modules"].get(BRIDGE, {}).get("start_blocks", 64)
        sx, sy, sz = layout.start()
        # Повернуть можно только шагами по 10° (голова) и 30° (ноги): со
        # случайного угла ровно спиной к пропасти не встать, и на длинном
        # мосту бота уводило вбок — вставал на край соседнего ряда, а луч
        # прицела уходил в пустоту. Угол старта кратен 10° — встать ровно можно.
        # Наклон взгляда на старте тоже разный (кратно 10°, от прямо вниз до
        # прямо вверх): учитель показывает, как довести взгляд до нужного, — в
        # том числе с -90° обратно на -80°. Иначе этого примера не было: учитель
        # не ошибается, а сеть у края перебирала вниз и застревала (луч — в пустоту).
        teleport = f"tp {name} {sx} {sy} {sz} {10 * random.randint(-18, 17)} {10 * random.randint(-9, 9)}"
        commands = [fill_command(fill, world) for fill in course.reset_fills(lane, layout)]
        commands += [f"clear {name}", f"give {name} minecraft:dirt {blocks}",
                     f"execute in minecraft:{world} run {teleport}" if world else teleport]
        for command in commands:
            self.server.send(command)
        session.bridge_sent = time.time()

    def _end_episode(self, session: BotSession, state: dict, outcome: str) -> None:
        """Исход задачки (осалили, раунд выигран, цель остановили...) —
        конец эпизода: последний переход закрывается ТЕРМИНАЛЬНО (без
        бутстрапа), награду или штраф за исход добавляет модуль."""
        session.module.count(outcome)
        if session.prev is None:
            return
        brain = self._brain(session.task_name)
        session.module.outcome = outcome
        self._close_transition(session, state)
        session.nstep.flush(brain, session.prev[0], session.prev[1], done=True)
        session.clear_transition()
        session.module.outcome = None
        session.module.reset(state)  # если роль та же — новый эпизод с чистого листа

    def _apply_tag_change(self, session: BotSession, state: dict, outcome, role: str) -> None:
        """Осалил, осалили, раунд выигран, продержался до конца раунда —
        конец эпизода: последний переход закрывается ТЕРМИНАЛЬНО (без
        бутстрапа), награду или штраф добавляет модуль роли по outcome.
        Время раунда вышло у водящего (TIME_UP) — не конец, а обрыв: его
        переход закроет пауза между раундами (с бутстрапом). Новая роль —
        смена задачки (как !task — накопленное уходит в память с
        бутстрапом)."""
        if outcome in TERMINAL_OUTCOMES:
            self._end_episode(session, state, outcome)
        elif outcome is not None:
            session.module.count(outcome)  # для сводки: время раунда вышло
        if role != session.task_name:
            self._switch_task(session, role, keep_mode=True)

    def handle_command(self, message: dict) -> None:
        # Откуда пришла команда: "chat" (чат игры) или "console" (строка
        # команд лаунчера) — ответ уходит туда же.
        via = message.get("via", "chat")
        if message.get("cmd") == "toggle_greedy":
            session = self._session(message.get("bot_id", 1))
            session.greedy = not session.greedy
            epsilon = self._brain(session.task_name).epsilon()
            text = (f"Показываю, чему научился ({session.task_name}): без случайных действий." if session.greedy
                    else f"Снова исследую (случайных действий {epsilon:.0%}).")
            print(f"[ai] Бот {session.id}: greedy={session.greedy}")
            self._chat(session.id, text, via)
            return
        if message.get("cmd") == "hunt_start":
            # "Останови меня": !start — охота на того, кто написал; !start <ник>
            # — на игрока или бота роя; !start auto — на случайного бота, по кругу.
            target = message.get("target")
            hunting = [sid for sid, s in self.sessions.items() if s.hunting]
            self._restore_arena()
            self._run_round_function(HUNT)
            if target == "auto" and hunting:
                chosen = random.choice(hunting)
                target = bot_name(chosen, self.config)
                self.hunt.start(target, bot=chosen, auto=True)
            else:
                self.hunt.start(target, bot=bot_index(target, self.config) if target else None)
            who = target or "первого, кого увижу в выживании"
            if hunting:
                text = f"Охота на {who}! Через {self.hunt.head_start:g} с бежим."
            else:
                text = f"Охота на {who} начнётся, когда дашь задачу: !task all {HUNT}"
            print(f"[ai] Охота (hunt): цель {who}.")
            self._chat(message.get("bot_id", 1), text, via)
            return
        if message.get("cmd") == "player_died":
            name = message.get("name")
            hunters = [sid for sid, s in self.sessions.items() if s.hunting and sid != self.hunt.target_bot]
            # Кто добил — ник бота в сообщении о смерти ("... was slain by AI_2").
            # Ник бота — отдельным словом; в плагине он в скобках после имени
            # ("Егор [AI_3]", config.json bot.names) — скобки и знаки отбросить.
            words = [word.strip("[]()<>.,:;!") for word in message.get("text", "").split()]
            killer = next((bot_index(word, self.config) for word in words
                           if bot_index(word, self.config) is not None), None)
            if self.hunt.died(name, hunters, killer):
                print(f"[ai] Охота: цель остановлена — {message.get('text', name)}.")
                self._chat(message.get("bot_id", 1), f"Остановили {name}! Ещё раз — !start.")
            return
        if message.get("cmd") != "set_task":
            print(f"[ai] Неизвестная команда: {message}")
            return

        task_name = message.get("task")
        bot_id = message.get("bot_id", 1)
        if task_name not in MODULES and task_name not in (MIX, TAG):
            self._chat(bot_id, f"Нет такой задачи '{task_name}'. Есть: {', '.join(MODULES)}, {MIX}, {TAG}", via)
            return

        if message.get("all"):
            self.default_task = task_name  # и для ботов, которые подключатся позже
            targets = list(self.sessions.values())
        else:
            targets = [self._session(bot_id)]

        for session in targets:
            left_world = self._task_world(session.task_name)
            self._switch_task(session, task_name)
            if left_world is not None and self._task_world(task_name) != left_world and self.server is not None:
                # Ушёл из мира задачки (мост) — обратно в обычный мир, на его спавн:
                # иначе бот так и стоял бы на острове над пустотой.
                self.server.send(f"mcbot goto main {bot_name(session.id, self.config)}")
            session.bridge_ready = False  # вернётся к мосту — судья поставит на старт заново
        if message.get("all") and self.server is not None:
            self._pending_function = task_name  # рой уже в игре — выполнится сразу
            self._run_task_function_when_ready()
        who = "всему рою" if message.get("all") else f"боту {bot_id}"
        print(f"[ai] Задача {task_name} -> {who}")
        self._chat(bot_id, f"Задача: {task_name} ({who}).", via)

    def _restore_arena(self) -> None:
        """Охота — на целой арене: столбы и копка прошлой охоты убраны. Плагин
        помнит изменённые блоки и возвращает их (/mcbot restore); с ботами
        mineflayer запомнить некому — арена остаётся как есть."""
        if self.server is not None and self.config["bot"].get("body") == "plugin":
            self.server.send("mcbot restore")

    def _run_round_function(self, name: str) -> None:
        """Новый круг игры (охота, раунд салок) — снова набор команд задачи
        server/functions/<задача>.mcfunction (автор: им чистят и пополняют
        инвентарь, а выполнялся он только при !task — после первой охоты у
        ботов кончались блоки). Без сервера-арены (RCON) — нечем."""
        if self.server is not None:
            self.server.run_function(name)

    def _run_task_function_when_ready(self) -> None:
        """server/functions/<задача>.mcfunction — когда зашли все bot.count
        ботов (или через 20 с после первого: кто-то не зашёл — не ждать вечно)."""
        if self._pending_function is None or not self.sessions:
            return
        if self._first_join is None:
            self._first_join = time.time()
        if len(self.sessions) >= self.config["bot"].get("count", 1) or time.time() - self._first_join > 20:
            self.server.run_function(self._pending_function)
            self._pending_function = None

    def _switch_task(self, session: BotSession, task_name: str, keep_mode: bool = False) -> None:
        """Смена задачи — не смерть: накопленные переходы отдаём в память
        мозга СТАРОЙ задачки с бутстрапом от последнего известного
        состояния. Незакрытый переход из prev выбрасываем: его награду
        считать уже нечем. События старой задачки — в её окно метрик.
        keep_mode — смена задачки внутри режима (очередная задачка mix,
        новая роль в салках), а не новая команда !task."""
        old_brain = self._brain(session.task_name)
        if session.prev is not None:
            session.nstep.flush(old_brain, session.prev[0], session.prev[1], done=False)
        session.nstep.clear()
        self._window(session.task_name).add_events(session.module.take_events())
        if keep_mode:
            session.set_task(task_name)
        else:
            was_playing_tag = session.tag
            session.assign(task_name)
            if was_playing_tag and not session.tag:
                self.tag_game.leave(session.id)
        self._brain(session.task_name)  # подготовить мозг новой задачки
        session.last_sent_target = None
        self._sync_target(session)

    def _session(self, bot_id: int) -> BotSession:
        if bot_id not in self.sessions:
            print(f"[ai] Новый бот в рое: id={bot_id}, задача {self.default_task} (сессий: {len(self.sessions) + 1})")
            self.sessions[bot_id] = BotSession(bot_id, self.default_task, self.config)
        return self.sessions[bot_id]

    def _sync_target(self, session: BotSession):
        """Шлёт в Node set_target, только если цель модуля этого бота
        реально поменялась с прошлого тика — иначе забивали бы канал
        действий тем же сообщением каждые tick_rate_ms без всякой пользы."""
        target = session.module.current_target
        if target == session.last_sent_target:
            return

        session.last_sent_target = target
        if target is not None and "entity_id" in target:
            # Позиция рядом с id сущности — запасная: Node берёт её, если
            # сам сущность не видит (салки знают, где убегающий, из его же
            # состояния, даже когда он далеко).
            fallback = {axis: target[axis] for axis in ("x", "y", "z")} if "x" in target else None
            message = {"type": "set_target", "position": fallback, "entity_id": target["entity_id"],
                       "bot_id": session.id}
        else:
            message = {"type": "set_target", "position": target, "bot_id": session.id}
        self.action_socket.send_json(message)

    def _chat(self, bot_id: int, text: str, via: str = "chat") -> None:
        self.action_socket.send_json({"type": "chat", "text": text, "bot_id": bot_id, "via": via})
        if via == "console" and self.config["bot"].get("body") == "plugin":
            # Боты — в плагине сервера: их консоль лаунчер не читает, ответ на
            # команду из строки лаунчера он увидит только в выводе Python.
            print(f"[cmd] [bot#{bot_id}] {text}")

    # --- метрики ------------------------------------------------------------

    def _window(self, task: str) -> TaskWindow:
        if task not in self._windows:
            self._windows[task] = TaskWindow()
        return self._windows[task]

    def _report_metrics(self) -> None:
        """Сводка за окно по каждой задачке: "как дела" человеческими
        словами (TrainingModule.summarize), доля случайных действий,
        средняя награда за тик и loss по каналам. Печатается в консоль,
        дописывается в data/metrics.csv — по нему видно, учится ли бот,
        даже спустя дни."""
        elapsed = max(time.time() - self._window_started, 1e-6)
        with self._shared_lock:
            learn_stats, self._learn_stats = self._learn_stats, {}
        for task, stats in learn_stats.items():
            window = self._window(task)
            window.learns += stats["learns"]
            window.learn_seconds += stats["seconds"]
            for channel, loss in stats["losses"].items():
                window.losses[channel] = window.losses.get(channel, 0.0) + loss
        bots_per_task: dict[str, int] = {}
        for session in self.sessions.values():
            bots_per_task[session.task_name] = bots_per_task.get(session.task_name, 0) + 1
            self._window(session.task_name).add_events(session.module.take_events())

        tasks = {}
        for task, window in self._windows.items():
            if window.ticks == 0 and task not in bots_per_task:
                continue
            brain = self._brain(task)
            ticks = max(window.ticks, 1)
            bot_minutes = window.ticks * self.tick_seconds / 60.0
            summary, score = MODULES[task].summarize(window.events, window.ticks, bot_minutes)
            if window.events.get("deaths"):
                summary += f"; умирал {times(window.events['deaths'])}"
            learners = list(brain.learners)
            rewards = {c: round(window.rewards[c] / ticks, 4) for c in learners}
            tasks[task] = {
                "bots": bots_per_task.get(task, 0),
                "ticks": window.ticks,
                "epsilon": round(brain.epsilon(), 4),
                "steps": brain.steps,
                "learn_steps": brain.learn_steps,
                "memory": len(brain.buffer),
                # Сколько шагов обучения в секунду успевает мозг и сколько
                # длится шаг — если шаг растёт (занята видеокарта), учимся
                # медленнее, но ответы ботам не опаздывают.
                "learn_per_sec": round(window.learns / elapsed, 1),
                "learn_ms": round(1000 * window.learn_seconds / window.learns, 1) if window.learns else None,
                # Средняя по каналам с сетями — одно число на задачку для графика.
                "reward": round(sum(rewards.values()) / max(len(rewards), 1), 4),
                "rewards": rewards,
                "loss": {c: round(v / window.learns, 4) for c, v in window.losses.items()} if window.learns else {},
                "events": window.events,
                "actions": window.actions,
                "greedy_actions": window.greedy_actions,
                "summary": summary,
                "score": None if score is None else round(score, 2),
            }

        metrics = {
            "time": round(time.time()),
            "bots": len(self.sessions),
            "states_per_sec": round(self._window_states / elapsed, 1),
            "tasks": tasks,
        }
        self._windows = {}
        self._window_started = time.time()
        self._window_states = 0

        if any(session.tag for session in self.sessions.values()):
            print(f"[ai] {self.tag_game.describe()}")
        for task, m in tasks.items():
            loss = " ".join(f"{c} {v:.3f}" for c, v in m["loss"].items()) or "—"
            speed = f"{m['learn_per_sec']} шагов/с по {m['learn_ms']} мс" if m["learn_ms"] else "—"
            print(
                f"[ai] {task} · ботов {m['bots']} · {m['summary'] or '—'} · случайных действий "
                f"{m['epsilon']:.0%} · награда {m['reward']:+.3f}/тик · обучение {speed} · loss {loss}"
            )
        if self.emit_metrics_json:
            print("[metrics] " + json.dumps(metrics, ensure_ascii=False))
        self._append_metrics_csv(metrics)

    def _append_metrics_csv(self, metrics: dict) -> None:
        fields = ["time", "task", "bots", "ticks", "steps", "learn_steps", "epsilon", "score", "reward",
                  *(f"loss_{c}" for c in CHANNEL_NAMES), "events", "summary"]
        try:
            DATA_DIR.mkdir(exist_ok=True)
            if METRICS_CSV.exists():
                with open(METRICS_CSV, encoding="utf-8") as f:
                    header = f.readline().strip()
                if header != ",".join(fields):
                    # Файл от старого формата метрик — откладываем, а не
                    # дописываем строки с другими колонками.
                    METRICS_CSV.rename(METRICS_CSV.with_name(f"metrics_old_{int(time.time())}.csv"))
            new_file = not METRICS_CSV.exists()
            with open(METRICS_CSV, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                if new_file:
                    writer.writeheader()
                for task, m in metrics["tasks"].items():
                    writer.writerow({
                        "time": metrics["time"], "task": task, "bots": m["bots"], "ticks": m["ticks"],
                        "steps": m["steps"], "learn_steps": m["learn_steps"], "epsilon": m["epsilon"],
                        "score": m["score"], "reward": m["reward"],
                        **{f"loss_{c}": m["loss"].get(c, "") for c in CHANNEL_NAMES},
                        "events": json.dumps(m["events"], ensure_ascii=False), "summary": m["summary"],
                    })
        except (OSError, ValueError) as err:
            # Файл открыт в Excel или остался от старого формата — метрики
            # не стоят падения обучения.
            print(f"[ai] Не смог дописать {METRICS_CSV.name}: {err}")

    # --- сохранение ---------------------------------------------------------

    def _save_brain_throttled(self, brain: TaskBrain) -> None:
        """На смерти бота — чекпоинт мозга, но не чаще раза в минуту: в рое
        из 7 ботов кто-нибудь умирает часто, а писать файлы каждые пару
        секунд незачем."""
        now = time.time()
        if now - self._last_save.get(brain.task, 0.0) >= 60:
            self._request_save([brain.task])
            self._last_save[brain.task] = now

    def _request_save(self, tasks: list[str]) -> None:
        """Сохранить мозги — потоком обучения, между шагами: запись на диск
        не должна задерживать ответы ботам. Поток не запущен (тесты,
        остановка) — сразу."""
        if self._learner is not None and self._learner.is_alive():
            with self._shared_lock:
                self._save_requests.update(tasks)
            return
        for task in tasks:
            brain = self.brains.get(task)
            if brain is not None and brain.learners:
                brain.save(brain_dir(task))

    def save(self):
        self._request_save(list(self.brains))
        for task in self.brains:
            self._last_save[task] = time.time()
        steps = ", ".join(f"{task} {brain.steps}" for task, brain in self.brains.items())
        print(f"[ai] Мозги сохранены (шагов: {steps}).")

    def shutdown(self):
        if self._learner is not None:
            self._learner.join(timeout=30)  # доучит текущий шаг и выйдет (running=False)
        self.save()
        if self.server is not None:
            self.server.close()
        self.state_socket.close()
        self.action_socket.close()
        print("[ai] Аккуратно остановлен.")


def parse_args():
    parser = argparse.ArgumentParser(description="ИИ-контроллер mc-bot")
    parser.add_argument(
        "--task", "--module", dest="task", default=DEFAULT_MODULE,
        help=f"Задача по умолчанию для всех ботов: {', '.join(MODULES)}, {MIX} "
             f"(разные задачи разным ботам) или {TAG} (салки) (по умолчанию: {DEFAULT_MODULE})",
    )
    return parser.parse_args()


def watch_stdin(loop: AILoop) -> None:
    """Лаунчер (py/launcher.py) не может послать Ctrl+C процессу без
    консоли, поэтому просит остановиться строкой "stop" в stdin. В обычной
    консоли stdin — терминал, и этот поток не запускается."""
    for line in sys.stdin:
        if line.strip() == "stop":
            print("[ai] Получена команда stop — останавливаюсь...")
            loop.running = False
            return


def main():
    args = parse_args()
    if args.task not in MODULES and args.task not in (MIX, TAG):
        print(f"[ai] Нет задачи '{args.task}'. Есть: {', '.join(MODULES)}, {MIX}, {TAG}")
        return 1
    loop = AILoop(args.task)
    # Учитель и вживую (py/sim/teacher.py): первых train.live_teachers ботов в
    # салках, охоте, на мосту и в бедварсе ведут простые правила, их ходы уходят в память как
    # демонстрации (DQfD) — как при обучении в симуляции. Без них живое
    # обучение за минуты размывало выученное с учителем (2026-09-27: мозги из
    # симуляции после 15 минут игры — убегающие стоят и крутятся, водящие
    # пятятся и прыгают). Бот в !greedy показывает сеть, а не учителя.
    teachers = loop.config["train"].get("live_teachers", 0)
    # Бедварс — своё число (по умолчанию все боты): сеть бедварса пока почти не
    # обучена, и под ней боты строили ерунду и не дрались (автор, 2026-09-30);
    # ходы учителя заодно учат сеть. Когда она подрастёт — уменьшить.
    bedwars_teachers = loop.config["modules"].get("bedwars", {}).get("live_teachers", 1000)
    if teachers or bedwars_teachers:
        from sim.teacher import teacher_actions
        loop.teacher = lambda session, state: (
            teacher_actions(session, state, loop.config, loop)
            if session.id <= (bedwars_teachers if session.task_name == BEDWARS else teachers)
            and not session.greedy else None)
        print(f"[ai] Учитель ведёт ботов 1–{teachers} в салках, охоте и на мосту (train.live_teachers), "
              f"в бедварсе — {'всех' if bedwars_teachers >= 1000 else f'1–{bedwars_teachers}'} "
              f"(modules.bedwars.live_teachers).")

    def on_sigint(sig, frame):
        print("\n[ai] Останавливаюсь (Ctrl+C)...")
        loop.running = False

    signal.signal(signal.SIGINT, on_sigint)
    if sys.stdin is not None and not sys.stdin.isatty():
        threading.Thread(target=watch_stdin, args=(loop,), daemon=True).start()
    loop.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
