""""Мозг" задачки: DQN-сети каналов (ноги / голова / руки, см. CHANNELS в
py/protocol.py) + своя память опыта + своё расписание epsilon.

У каждой задачки (walking, looking, follow, gathering...) — СВОЙ мозг, и
чекпоинт тоже свой: data/brains/<задачка>/<канал>.pt. Зачем: когда мозг
был один на все задачки, то, пока боты делали одну, сети переучивались
под неё, а общая память опыта заполнялась только ею — навык остальных
задачек затирался ("катастрофическое забывание": looking после смены
задачи и возврата "отупевал"). Теперь переключение задачек ничего не
затирает — каждая задачка помнит своё.

Внутри мозга каналы по-прежнему работают параллельно и "общаются" через
контекст (что делали соседи на прошлом тике, state_encoder.encode_context).
Канал, которому задачка разрешает ровно одно действие (руки в looking —
только "ничего не делать"), сети не получает вовсе: выбирать не из чего.

Поверх классической схемы (Mnih et al.) включено:
  - Double DQN — действие для TD-цели выбирает онлайн-сеть, а оценивает
    target-сеть: обычный max по target-сети систематически завышает Q;
  - n-step возвраты — переходы в памяти уже несут сумму наград за n
    шагов и множитель discount = gamma^n (см. NStepAccumulator в
    ai_loop.py), редкая награда "+100 за цель" доходит до ранних решений
    в n раз быстрее;
  - dueling-архитектура — в model.py;
  - маска действий задачки (TrainingModule.allowed_actions) — и при
    выборе действия, и в TD-цели;
  - DQfD margin-loss на демо-переходах (записи геймплея автора).
"""

import math
import os
import shutil
import threading
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from model import DQN
from protocol import CHANNELS, CHANNEL_NAMES
from replay_buffer import ReplayBuffer
from state_encoder import expand_vision, scalar_layout
from training_modules import allowed_actions


# Меньше стольки свободного места — мозг не сохраняем: запись на почти полный
# диск могла бы оборваться (у каждого канала ~17 МБ, у задачки — ~50 МБ).
MIN_FREE_BYTES = 300 * 2**20


def vision_memory_format(device) -> torch.memory_format:
    """Раскладка зрения и свёрток в памяти. На процессоре channels_last
    считает свёртки в ~1.4 раза быстрее (числа те же — меняется только
    порядок байт); на видеокарте оставляем как было — там не проверяли."""
    return torch.channels_last if torch.device(device).type == "cpu" else torch.contiguous_format


class ChannelLearner:
    """Одна сеть одного канала в мозге задачки: онлайн + target + оптимизатор."""

    def __init__(self, channel: str, allowed_names: list[str], config: dict, vision_shape: tuple,
                 scalar_dim: int, device):
        train_cfg = config["train"]
        self.channel = channel
        # Столбец этого канала в действиях/наградах перехода (порядок CHANNEL_NAMES).
        self.column = CHANNEL_NAMES.index(channel)
        self.num_actions = len(CHANNELS[channel])
        self.device = device

        # Выходов у сети всегда столько, сколько действий у канала (так веса
        # одинаковой формы у всех задачек и их можно копировать для warm
        # start), а запрещённые задачкой действия просто маскируются.
        allowed = torch.tensor([name in allowed_names for name in CHANNELS[channel]], dtype=torch.bool)
        self.allowed = allowed.to(device)
        self.allowed_indices = allowed.nonzero().flatten().tolist()

        channels, res_y, res_x = vision_shape
        self.policy_net = DQN(channels, res_y, res_x, scalar_dim, self.num_actions).to(
            device, memory_format=vision_memory_format(device))
        self.target_net = DQN(channels, res_y, res_x, scalar_dim, self.num_actions).to(
            device, memory_format=vision_memory_format(device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        # Копия для ответов ботам: учится policy_net (в потоке обучения), а
        # ответы идут по копии — не ждут шага обучения и не читают веса
        # посреди обновления. Копия догоняет policy_net каждые
        # actor_sync_every шагов (TaskBrain._sync_actors).
        self.actor_net = DQN(channels, res_y, res_x, scalar_dim, self.num_actions).to(
            device, memory_format=vision_memory_format(device))
        self.actor_net.load_state_dict(self.policy_net.state_dict())
        self.actor_net.eval()
        # fused — весь шаг Adam одним CUDA-ядром вместо десятков мелких:
        # сети маленькие, и время уходит не на математику, а на запуск ядер.
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=train_cfg["lr"], fused=(device.type == "cuda"),
        )

        self.double_dqn = train_cfg.get("double_dqn", True)
        # DQfD: Q действия эксперта должен быть хотя бы на margin выше
        # лучшего другого действия — иначе экспертное поведение
        # "размывается" TD-ошибками.
        self.demo_margin = train_cfg.get("demo_margin", 1.0)

    def act(self, inputs, epsilons: list[float]) -> list[int]:
        """Индексы действий внутри канала для батча ботов (epsilon-greedy
        среди разрешённых; epsilon у каждого бота свой — в !greedy это 0).
        Сеть считается одним проходом и только для тех ботов, кто действует
        по ней: пока случайных действий много, видеокарту почти не трогаем.
        inputs() — развёрнутые (зрение, скаляры) всего батча, считаются по
        первому требованию (select_actions_batch)."""
        chosen = [None] * len(epsilons)
        rows = []
        for i, epsilon in enumerate(epsilons):
            if torch.rand(()).item() < epsilon:
                chosen[i] = self.allowed_indices[int(torch.randint(len(self.allowed_indices), ()).item())]
            else:
                rows.append(i)
        if rows:
            vision, scalars = inputs()
            q = self.actor_net(vision[rows], scalars[rows]).masked_fill(~self.allowed, float("-inf"))
            for i, index in zip(rows, q.argmax(dim=1).tolist()):
                chosen[i] = index
        return chosen

    def learn(self, batch) -> torch.Tensor:
        """Один шаг оптимизации. batch — уже на устройстве, зрение развёрнуто.
        Возвращает loss ТЕНЗОРОМ: .item() — это ожидание GPU, его делает
        вызывающий раз на все каналы."""
        visions, scalarss, actions_all, rewards_all, next_visions, next_scalarss, dones, discounts, is_demo = batch
        actions = actions_all[:, self.column]
        rewards = rewards_all[:, self.column]
        batch_size = actions.shape[0]

        if self.double_dqn:
            # Онлайн-сеть нужна и на s (для loss), и на s' (выбрать a* для
            # Double DQN) — один проход по склеенному батчу вместо двух.
            q_both = self.policy_net(torch.cat([visions, next_visions]), torch.cat([scalarss, next_scalarss]))
            q_all, next_q_online = q_both[:batch_size], q_both[batch_size:].detach()
        else:
            q_all = self.policy_net(visions, scalarss)
        q_values = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        # TD-цель: r + discount * Q_target(s', a*), без бутстрапа на терминале.
        # a* — только среди разрешённых задачкой действий: оценивать будущее
        # по действию, которого бот никогда не сделает, бессмысленно.
        with torch.no_grad():
            next_q_target = self.target_net(next_visions, next_scalarss)
            if self.double_dqn:
                best_next = next_q_online.masked_fill(~self.allowed, float("-inf")).argmax(dim=1, keepdim=True)
                next_q = next_q_target.gather(1, best_next).squeeze(1)
            else:
                next_q = next_q_target.masked_fill(~self.allowed, float("-inf")).max(dim=1).values
            target = rewards + discounts * next_q * (1 - dones)

        loss = nn.functional.smooth_l1_loss(q_values, target)

        # DQfD margin-loss — только на демо-строках, где действие эксперта
        # задачкой разрешено (запись могла содержать, например, поворот
        # головы, а в walking голове можно только наклон). Считается на всём
        # батче и маскируется умножением — без "if есть ли демо", который
        # заставил бы CPU ждать GPU.
        demo_weight = is_demo * self.allowed[actions].to(torch.float32)
        # Лучшее ДРУГОЕ разрешённое действие: запрещённые и экспертное
        # "выключаем", берём максимум по остальным.
        q_other = q_all.masked_fill(~self.allowed, float("-inf"))
        q_other = q_other.scatter(1, actions.unsqueeze(1), float("-inf"))
        q_other = torch.where(torch.isinf(q_other), torch.full_like(q_other, -1e9), q_other)
        q_other_max = q_other.max(dim=1).values
        # Растёт, когда сеть предпочитает не-экспертное действие, и равен
        # нулю, когда экспертное и так уверенно лидирует.
        margin = nn.functional.relu(q_other_max + self.demo_margin - q_values)
        loss = loss + (margin * demo_weight).sum() / demo_weight.sum().clamp(min=1.0)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0, foreach=True)
        self.optimizer.step()
        return loss.detach()

    def sync_target(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def sync_actor(self) -> None:
        with torch.no_grad():
            for actor, policy in zip(self.actor_net.parameters(), self.policy_net.parameters()):
                actor.copy_(policy)


class TaskBrain:
    def __init__(self, task: str, config: dict, vision_shape: tuple, scalar_dim: int, device):
        train_cfg = config["train"]
        task_cfg = config["modules"].get(task, {})
        self.task = task
        self.config = config
        self.device = device
        self.vision_shape = vision_shape  # (каналы, res_y, res_x) — пишется в чекпоинт
        allowed = allowed_actions(task)
        self.learners = {
            channel: ChannelLearner(channel, names, config, vision_shape, scalar_dim, device)
            for channel, names in allowed.items() if len(names) > 1
        }
        # Каналы без выбора: единственное разрешённое действие ("ничего не
        # делать") — сеть им не нужна.
        self.fixed = {channel: names[0] for channel, names in allowed.items() if len(names) == 1}

        self.buffer = ReplayBuffer(
            train_cfg["buffer_size"],
            demo_fraction=train_cfg.get("demo_fraction", 0.25),
        )

        # Какие входы видит мозг задачки (modules.<задачка>.inputs: vision,
        # hearing, entities... — имена из state_encoder.scalar_layout; по
        # умолчанию всё). Выключенный вход подаётся нулями. Зачем: вживую
        # мозг looking, пока боты стояли на месте, заучивал МЕСТО вместо
        # правила — круговое зрение поворачивается вместе с ботом, и рельеф
        # (как и неподвижные соседи-боты) однозначно выдаёт, куда он смотрит.
        # После перемещения — 0% попаданий. Навестись на цель можно по одним
        # данным о цели, и без "ориентиров" сеть учит именно правило.
        inputs = task_cfg.get("inputs", {})
        self.use_vision = inputs.get("vision", True)
        mask = torch.ones(scalar_dim)
        for name, part in scalar_layout(config).items():
            if not inputs.get(name, True):
                mask[part] = 0.0
        self.scalar_mask = mask.to(device)

        self.batch_size = train_cfg["batch_size"]
        # Не учиться, пока опыта в памяти меньше: память при запуске пустая, и
        # первые шаги по сотне-другой примеров, которые крутятся по кругу,
        # портили обученный мозг — вживую мост падал вдвое чаще, чем без
        # обучения (2026-09-29). Демо-примеры учителя тоже считаются.
        self.learn_start = max(self.batch_size, train_cfg.get("learn_start", 0))
        self.epsilon_start = train_cfg["epsilon_start"]
        # Своя доля случайных действий у задачки (мост: случайный шаг без шифта
        # у края — падение; исследовать там дорого).
        self.epsilon_end = task_cfg.get("epsilon_end", train_cfg["epsilon_end"])
        # У задачки может быть своё расписание: простой looking учится
        # быстрее ходьбы, ему незачем часами действовать наугад.
        self.epsilon_decay_steps = task_cfg.get("epsilon_decay_steps", train_cfg["epsilon_decay_steps"])
        # Версия правил задачки (modules.<задачка>.version). Правила
        # поменялись (например, в салках стало надо ударить, а не просто
        # подбежать) — мозг под старыми правилами загружается с навыками, но
        # случайные действия начинаются заново: иначе он никогда не попробует
        # то, что раньше было бесполезно.
        self.rules_version = task_cfg.get("version", 1)
        self.rules_changed = False  # загружен мозг под старыми правилами
        # Считается в шагах ОБУЧЕНИЯ (не среды): сколько раз сеть реально
        # обновилась между синхронизациями target-сети — это и есть то, что
        # важно для стабильности TD-цели.
        self.target_update_every = train_cfg["target_update_every"]

        self.steps = 0          # шагов взаимодействия со средой (для epsilon)
        self.learn_steps = 0    # шагов оптимизации
        self.learn_credit = 0.0  # накопленный "кредит" обучения (см. ai_loop.py)
        # Ответы (поток ответов ботам) и обучение (поток обучения) — в разных
        # потоках: learn_lock — шаг обучения и запись чекпоинта не
        # пересекаются; actor_lock — короткий: ответ по копиям сетей против
        # их обновления.
        self.learn_lock = threading.Lock()
        self.actor_lock = threading.Lock()
        self.actor_sync_every = train_cfg.get("actor_sync_every", 5)

    # --- действия -----------------------------------------------------------

    def epsilon(self) -> float:
        """Линейный спад от epsilon_start к epsilon_end за decay_steps шагов."""
        frac = min(self.steps / self.epsilon_decay_steps, 1.0)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def select_actions(self, vision: torch.Tensor, scalars: torch.Tensor, greedy: bool = False) -> dict:
        """{канал: имя действия} для ВСЕХ каналов одного бота. vision —
        компактный стек кадров (FrameStacker). greedy=True — без случайных
        действий (показать, чему мозг уже научился, !greedy)."""
        return self.select_actions_batch([vision], [scalars], [greedy])[0]

    def select_actions_batch(self, visions: list, scalars: list, greedy: list[bool]) -> list[dict]:
        """То же для нескольких ботов сразу — один проход каждой сети на всех,
        а не проход на бота. На тик роя из 7 ботов это 3 вызова видеокарты на
        мозг вместо 21, а каждый вызов, пока видеокарта занята ещё и игрой,
        ждёт своей очереди — ответы ботам опаздывали (Node ждёт до 60 мс).
        Epsilon-greedy независимо по каналам: пока одна сеть исследует,
        другая может действовать жадно."""
        epsilon = self.epsilon()
        self.steps += len(visions)
        epsilons = [0.0 if flag else epsilon for flag in greedy]

        cache = []

        def inputs():
            # Развернуть батч на видеокарте — один раз и только если кому-то
            # из ботов нужна сеть.
            if not cache:
                cache.append(self._vision_input(torch.stack(visions).to(self.device)))
                cache.append(torch.stack(scalars).to(self.device) * self.scalar_mask)
            return cache[0], cache[1]

        chosen = [dict(self.fixed) for _ in visions]
        with torch.no_grad(), self.actor_lock:
            for channel, learner in self.learners.items():
                for i, index in enumerate(learner.act(inputs, epsilons)):
                    chosen[i][channel] = CHANNELS[channel][index]
        return chosen

    # --- обучение -----------------------------------------------------------

    def remember(self, vision, scalars, actions, rewards, next_vision, next_scalars, done, discount):
        """actions/rewards — списки по каналам в порядке CHANNEL_NAMES."""
        self.buffer.push(vision, scalars, actions, rewards, next_vision, next_scalars, done, discount)

    def remember_demo(self, vision, scalars, actions, rewards, next_vision, next_scalars, done, discount):
        """То же, но демонстрация (действие учителя) — в демо-память."""
        self.buffer.push_demo(vision, scalars, actions, rewards, next_vision, next_scalars, done, discount)

    def learn(self) -> dict | None:
        """Один шаг оптимизации всех сетей мозга по одному мини-батчу.
        Возвращает {канал: loss} или None, если опыта ещё мало."""
        if not self.learners or self.buffer.total() < self.learn_start:
            return None
        with self.learn_lock:
            losses = self._learn_step()
        if self.learn_steps % self.actor_sync_every == 0:
            self._sync_actors()
        return losses

    def _sync_actors(self) -> None:
        """Копии для ответов догоняют обучаемые сети. Ждём конца копирования
        на видеокарте, прежде чем отпустить замок: поток ответов работает в
        другом CUDA-потоке и иначе мог бы прочитать недописанные веса."""
        with self.actor_lock:
            for learner in self.learners.values():
                learner.sync_actor()
            if self.device.type == "cuda":
                torch.cuda.current_stream().synchronize()

    def _learn_step(self) -> dict:
        batch = [t.to(self.device, non_blocking=True) for t in self.buffer.sample(self.batch_size)]
        # Зрение разворачивается один раз на батч — для всех каналов сразу.
        batch[0] = self._vision_input(batch[0])
        batch[4] = self._vision_input(batch[4])
        batch[1] = batch[1] * self.scalar_mask
        batch[5] = batch[5] * self.scalar_mask
        losses = [learner.learn(batch) for learner in self.learners.values()]
        # Одна синхронизация с GPU на все каналы, а не по одной на каждый.
        losses = dict(zip(self.learners, torch.stack(losses).tolist()))

        self.learn_steps += 1
        if self.learn_steps % self.target_update_every == 0:
            for learner in self.learners.values():
                learner.sync_target()
        return losses

    def _vision_input(self, compact: torch.Tensor) -> torch.Tensor:
        """Компактное зрение -> вход сети (или нули, если задачке зрение
        выключено: архитектура та же, просто сигнала нет)."""
        vision = expand_vision(compact, self.config).contiguous(memory_format=vision_memory_format(self.device))
        return vision if self.use_vision else torch.zeros_like(vision)

    # --- чекпоинты: data/brains/<задачка>/<канал>.pt ------------------------

    def save(self, directory: Path) -> None:
        """Сохранить сети каналов. Каждый файл сначала пишется рядом
        (<канал>.pt.tmp) и только потом заменяет старый: если запись
        оборвётся (кончилось место на диске — было 2026-09-27, 0 байт
        свободно), прежний чекпоинт останется целым, а не обрезанным. Места
        мало (меньше MIN_FREE_BYTES) — сохранение пропускается: мозг в памяти
        цел, следующее сохранение попробует снова."""
        directory.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(directory).free
        if free < MIN_FREE_BYTES:
            print(f"[ai] Мозг {self.task}: на диске свободно {free // 2**20} МБ — сохранение пропущено "
                  f"(старые файлы целы). Освободи место.")
            return
        with self.learn_lock:
            try:
                self._save(directory)
            except OSError as err:
                print(f"[ai] Мозг {self.task}: сохранить не удалось ({err}) — старые файлы целы.")

    def _save(self, directory: Path) -> None:
        for channel, learner in self.learners.items():
            final = directory / f"{channel}.pt"
            temporary = directory / f"{channel}.pt.tmp"
            torch.save(
                {
                    "policy_net": learner.policy_net.state_dict(),
                    "target_net": learner.target_net.state_dict(),
                    "optimizer": learner.optimizer.state_dict(),
                    "steps": self.steps,
                    "learn_steps": self.learn_steps,
                    "rules_version": self.rules_version,
                    # С каким разрешением сетки зрения учился — чтобы дорастить
                    # сеть, если vision.resolution поменяют (_grow_vision).
                    "vision_resolution": [self.vision_shape[1], self.vision_shape[2]],
                },
                temporary,
            )
            os.replace(temporary, final)  # замена целиком: либо новый файл, либо старый

    def load_channel(self, channel: str, path: Path) -> list[str]:
        """Полная загрузка (веса, оптимизатор, счётчики). Если сеть в файле
        меньше (с тех пор добавились входы или действия) — она доращивается
        (_grow_state_dict), навыки сохраняются. Возвращает, что доращено.
        Бросает RuntimeError/KeyError при несовместимой архитектуре —
        вызывающий решает, что делать (ai_loop.py начинает этот канал заново)."""
        learner = self.learners[channel]
        checkpoint = torch.load(path, map_location=self.device)
        policy, grown = self._grow(checkpoint, "policy_net", learner.policy_net)
        target, _ = self._grow(checkpoint, "target_net", learner.target_net)
        learner.policy_net.load_state_dict(policy)
        learner.target_net.load_state_dict(target)
        learner.sync_actor()
        if checkpoint.get("rules_version", 1) != self.rules_version:
            # Правила задачки поменялись: навыки (веса) оставляем, а
            # счётчики — нет, и случайные действия начинаются заново.
            self.rules_changed = True
            return grown
        # Состояние оптимизатора привязано к формам весов — у доращённой сети
        # оно не подходит, Adam начинает заново (веса-то сохранены).
        if not grown:
            learner.optimizer.load_state_dict(checkpoint["optimizer"])
        # Счётчики общие на мозг; берём максимум, если каналы сохранялись в разное время.
        self.steps = max(self.steps, checkpoint.get("steps", 0))
        self.learn_steps = max(self.learn_steps, checkpoint.get("learn_steps", 0))
        return grown

    def warm_start_channel(self, channel: str, path: Path) -> list[str]:
        """Warm start: только веса сети (из behavior cloning или из мозга
        другой задачки), с доращиванием, как в load_channel. Счётчики и
        оптимизатор не трогаем — epsilon начинается с начала, и DQN
        дообучает эти веса под свою награду."""
        learner = self.learners[channel]
        checkpoint = torch.load(path, map_location=self.device)
        policy, grown = self._grow(checkpoint, "policy_net", learner.policy_net)
        learner.policy_net.load_state_dict(policy)
        learner.target_net.load_state_dict(policy)
        learner.sync_actor()
        return grown

    def _grow(self, checkpoint: dict, key: str, net: nn.Module) -> tuple[dict, list[str]]:
        return _grow_state_dict(
            checkpoint[key], net.state_dict(),
            resolution=(self.vision_shape[1], self.vision_shape[2]),
            saved_resolution=checkpoint.get("vision_resolution"),
        )


def _grow_state_dict(saved: dict, current: dict, resolution=None, saved_resolution=None) -> tuple[dict, list[str]]:
    """Дорастить сохранённую сеть до нынешних размеров, не теряя навыков.

    Проект растёт: добавляются действия (новые выходы) и входы (например,
    сводка инвентаря). Новое всегда дописывается В КОНЕЦ (protocol.CHANNELS,
    ACTION_NAMES, state_encoder.scalar_layout), поэтому старые веса остаются
    на своих местах:
      - новые входы — нулевые столбцы первого слоя скаляров: на старых
        данных сеть считает ровно то же, что и раньше;
      - новые действия — новые строки advantage-слоя, с заниженной оценкой
        (маленькие веса, bias ниже всех старых): пока сеть их не распробует,
        жадно она их не выберет, а в dueling-голове (Q = V + A - mean A)
        сдвиг mean A одинаков для всех действий — выбор среди старых не
        меняется.
    Сетку зрения сделали другого размера — см. _grow_vision (навыки,
    завязанные на картинку, переносятся приближённо).
    Любое другое расхождение форм — это уже другая архитектура:
    load_state_dict бросит RuntimeError."""
    grown_dict = dict(saved)
    notes = []

    if resolution is not None:
        grown_dict, note = _grow_vision(grown_dict, current, resolution, saved_resolution)
        if note:
            notes.append(note)
        saved = grown_dict

    weight = saved.get("scalar_net.0.weight")
    target_weight = current["scalar_net.0.weight"]
    if weight is not None and weight.shape[0] == target_weight.shape[0] and weight.shape[1] < target_weight.shape[1]:
        padded = torch.zeros_like(target_weight)
        padded[:, :weight.shape[1]] = weight
        grown_dict["scalar_net.0.weight"] = padded
        notes.append(f"входов {weight.shape[1]} -> {target_weight.shape[1]}")

    adv_weight, adv_bias = saved.get("advantage.weight"), saved.get("advantage.bias")
    target_adv_weight, target_adv_bias = current["advantage.weight"], current["advantage.bias"]
    if adv_weight is not None and adv_weight.shape[1] == target_adv_weight.shape[1] \
            and adv_weight.shape[0] < target_adv_weight.shape[0]:
        old = adv_weight.shape[0]
        new_weight = target_adv_weight.clone() * 0.1
        new_weight[:old] = adv_weight
        new_bias = torch.full_like(target_adv_bias, float(adv_bias.min()) - 2.0)
        new_bias[:old] = adv_bias
        grown_dict["advantage.weight"] = new_weight
        grown_dict["advantage.bias"] = new_bias
        notes.append(f"действий {old} -> {target_adv_weight.shape[0]}")

    return grown_dict, notes


def _grow_vision(saved: dict, current: dict, resolution, saved_resolution) -> tuple[dict, str | None]:
    """Сетку зрения сделали другого размера (vision.resolution): свёртки от
    размера картинки не зависят, меняется только слой, собирающий их выход
    (trunk.0). Его веса для новой сетки берутся с ближайшей ячейки старой
    (и делятся на число новых ячеек на одну старую, чтобы сумма не выросла),
    а смещение поправляется так, чтобы на пустой картинке сеть считала РОВНО
    то же, что раньше: задачкам без зрения (looking) она всегда пустая —
    для них ничего не меняется. Остальным — хорошее начало, дальше
    доучиваются. Старые чекпоинты разрешение не хранили — было квадратное
    (8x8)."""
    weight = saved.get("trunk.0.weight")
    target = current["trunk.0.weight"]
    if weight is None or weight.shape == target.shape or weight.shape[0] != target.shape[0]:
        return saved, None
    channels = saved["conv.3.weight"].shape[0]
    scalar_out = saved["scalar_net.0.weight"].shape[0]
    old_cells = (weight.shape[1] - scalar_out) // channels
    new_h, new_w = resolution[0] // 2, resolution[1] // 2
    if saved_resolution is not None:
        old_h, old_w = saved_resolution[0] // 2, saved_resolution[1] // 2
    else:
        old_h = old_w = round(math.sqrt(old_cells))
    if old_h * old_w != old_cells or channels * new_h * new_w + scalar_out != target.shape[1]:
        return saved, None  # не угадали, как было, — пусть будет несовместимо

    rows = weight.shape[0]
    split = channels * old_cells
    old_vision = weight[:, :split].reshape(rows, channels, old_h, old_w)
    ys = torch.arange(new_h, device=weight.device) * old_h // new_h
    xs = torch.arange(new_w, device=weight.device) * old_w // new_w
    new_vision = old_vision[:, :, ys][:, :, :, xs] * (old_cells / (new_h * new_w))
    new_vision = new_vision.reshape(rows, -1)

    grown = dict(saved)
    grown["trunk.0.weight"] = torch.cat([new_vision, weight[:, split:]], dim=1)
    blank_old = _conv_on_blank(saved, 2 * old_h, 2 * old_w)
    blank_new = _conv_on_blank(saved, 2 * new_h, 2 * new_w)
    grown["trunk.0.bias"] = saved["trunk.0.bias"] + weight[:, :split] @ blank_old - new_vision @ blank_new
    return grown, f"зрение {2 * old_w}x{2 * old_h} -> {2 * new_w}x{2 * new_h}"


def _conv_on_blank(saved: dict, height: int, width: int) -> torch.Tensor:
    """Выход свёрток сохранённой сети на пустой картинке (все нули) —
    такой её видят задачки с выключенным зрением."""
    in_channels = saved["conv.0.weight"].shape[1]
    device = saved["conv.0.weight"].device
    conv = DQN(in_channels, height, width, 1, 1).conv.to(device)
    conv.load_state_dict({key[len("conv."):]: value for key, value in saved.items() if key.startswith("conv.")})
    with torch.no_grad():
        return conv(torch.zeros(1, in_channels, height, width, device=device))[0]
