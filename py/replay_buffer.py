"""Replay buffer одной задачки ("мозга", см. py/dqn.py) — с поддержкой DQfD
и n-step.

У каждой задачки своя память опыта: если бы память была общей, то пока
боты час делают одну задачку, опыт другой полностью вытеснился бы из неё,
и навык той задачки "забылся" бы (так и было: looking после смены задачи
и возврата "отупевал").

Переход хранит наблюдение один раз, а действия и награды — по каждому
каналу (ноги/голова/руки) сразу: сети каналов учатся на одних и тех же
батчах, каждая берёт свой столбец действий/наград.

Переход: (vision, scalars, actions[C], rewards[C], next_vision,
next_scalars, done, discount). discount — множитель перед бутстрапом
Q(next): gamma^n для n-step перехода (см. NStepAccumulator в ai_loop.py),
gamma для обычного одношагового.

Хранение — заранее выделенные тензоры на всю ёмкость (по одному на поле
перехода), а не список кортежей. Раньше на 100 тысяч переходов было ~650
тысяч мелких объектов Python, и полная сборка мусора обходила их все —
232 мс заморозки процесса (замерено), за это время боты роя получали
устаревшие действия ("ИИ не успевает отвечать"). Теперь объектов — десяток,
а выборка батча — одна индексация тензоров.

Зрение хранится компактно — uint8 кадры из state_encoder.encode_state
(в сеть их разворачивает expand_vision), скаляры — в float16. Переход
занимает ~2.5 КБ: память на 100 тысяч переходов — ~250 МБ (выделяется при
первом переходе, у задачек без опыта — ничего).

Живой опыт — кольцо: при переполнении перезаписываются самые старые.
Демонстрационные переходы живут отдельно и никогда не вытесняются: их
мало, и это самый ценный опыт — суть DQfD в том, чтобы они не растворялись
в потоке собственного опыта агента.
"""

import threading

import torch


class _TensorRing:
    """Кольцо фиксированной ёмкости поверх заранее выделенных тензоров:
    storage[i] — поле i всех переходов, строка = переход."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.size = 0
        self.position = 0
        self.storage: list[torch.Tensor] | None = None

    def append(self, item: tuple) -> None:
        if self.storage is None:
            # Форма и тип каждого поля — по первому переходу.
            self.storage = [torch.empty((self.capacity, *field.shape), dtype=field.dtype) for field in item]
        for store, field in zip(self.storage, item):
            store[self.position] = field
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, k: int) -> list[torch.Tensor]:
        # С возвращением: на тысячах переходов повторы в батче из 256 редки и
        # безвредны, а выбор индексов — одна операция.
        indices = torch.randint(self.size, (k,))
        return [store[indices] for store in self.storage]

    def __len__(self) -> int:
        return self.size


class ReplayBuffer:
    def __init__(self, capacity: int, demo_fraction: float = 0.25):
        self.live = _TensorRing(capacity)
        self.demo = _TensorRing(capacity)  # демо тоже ограничим сверху
        self.demo_fraction = demo_fraction
        # Пишет поток ответов ботам, читает поток обучения (ai_loop) — без
        # замка батч мог бы захватить наполовину записанный переход.
        self.lock = threading.Lock()

    @staticmethod
    def _pack(vision, scalars, actions, rewards, next_vision, next_scalars, done, discount) -> tuple:
        return (
            vision.to(torch.uint8),
            scalars.to(torch.float16),
            torch.as_tensor(actions, dtype=torch.int64),
            torch.as_tensor(rewards, dtype=torch.float32),
            next_vision.to(torch.uint8),
            next_scalars.to(torch.float16),
            torch.tensor(float(done)),
            torch.tensor(float(discount)),
        )

    def push(self, *transition) -> None:
        packed = self._pack(*transition)
        with self.lock:
            self.live.append(packed)

    def push_demo(self, *transition) -> None:
        packed = self._pack(*transition)
        with self.lock:
            self.demo.append(packed)

    def sample(self, batch_size: int):
        """Батч на CPU: зрение — uint8 (развернуть expand_vision), скаляры —
        float32, плюс маска is_demo для margin-loss."""
        # Сколько строк берём из демо: demo_fraction от батча, но не больше,
        # чем есть в демо-буфере; остаток — из живого опыта (и наоборот).
        n_demo = min(int(batch_size * self.demo_fraction), len(self.demo))
        n_live = min(batch_size - n_demo, len(self.live))
        if n_demo + n_live < batch_size:
            # Живого опыта мало, добираем демо сверх fraction.
            n_demo = min(batch_size - n_live, len(self.demo))

        parts = []
        with self.lock:
            if n_live:
                parts.append(self.live.sample(n_live))
            if n_demo:
                parts.append(self.demo.sample(n_demo))
        fields = [torch.cat(columns) for columns in zip(*parts)]
        visions, scalarss, actions, rewards, next_visions, next_scalarss, dones, discounts = fields
        is_demo = torch.cat([torch.zeros(n_live), torch.ones(n_demo)])

        return (
            visions,
            scalarss.to(torch.float32),
            actions,                 # (B, C)
            rewards,                 # (B, C)
            next_visions,
            next_scalarss.to(torch.float32),
            dones,
            discounts,
            is_demo,
        )

    def __len__(self) -> int:
        return len(self.live)

    def total(self) -> int:
        """Живой опыт + демо — именно это нужно, чтобы набрать батч."""
        return len(self.live) + len(self.demo)
