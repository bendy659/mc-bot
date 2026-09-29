"""Базовый класс обучающих модулей ("специализаций").

Модуль решает две вещи:
  - какая сейчас у бота цель (through on_tick — может обновить
    self.current_target, а ядро само заметит изменение и пошлёт
    set_target в Node, см. AILoop._sync_target в py/ai_loop.py);
  - какая награда за переход prev_state -> curr_state под эту цель
    (compute_reward) — ОТДЕЛЬНО для каждого канала (ноги/голова/руки, см.
    py/protocol.py): у каждого канала своя сеть, и награду каждая получает
    свою.

Модуль = "задача" бота. Сети каналов общие для всех задач (задача идёт им
на вход one-hot'ом, см. state_encoder.encode_context), поэтому ноги,
научившиеся ходить в walking, остаются теми же ногами в gathering.
Задачу можно менять на лету, у каждого бота свою: чат-команда !task.

Как устроить награду: обычно основа — ОБЩАЯ командная награда (self.team:
дошли до цели — хорошо всем), плюс точечный шейпинг конкретному каналу
(голове — штраф за бесцельное верчение, рукам — за удар в воздух). Так
каналы учатся работать вместе, но каждый отвечает и за своё.

Ядро (py/ai_loop.py) ничего не знает про конкретные награды и то, как
выбираются цели — оно просто дёргает эти методы каждый тик и отдельно
занимается своим делом: ZMQ, кодирование наблюдения, DQN, replay buffer.

Как добавить свой модуль:
  1. Файл py/training_modules/<name>.py с классом, унаследованным от
     TrainingModule (name — обязательно переопределить, это же имя
     чекпоинта: data/dqn_<name>.pt).
  2. Зарегистрировать класс в MODULES в py/training_modules/__init__.py.
  3. (Опционально) секция параметров в config.json -> "modules" -> "<name>"
     — читай её в __init__ через self.config["modules"].get("<name>", {}).
  4. Добавить имя в КОНЕЦ TASK_ORDER (там же) — номер задачи идёт в
     one-hot на вход сетей, порядок менять нельзя.
  5. Запуск: python py/ai_loop.py --task <name> (или !task <name> в чате).
"""

from __future__ import annotations

from protocol import CHANNEL_NAMES


class TrainingModule:
    # Имя модуля — используется и для выбора через --module, и для
    # чекпоинта (data/dqn_<name>.pt). Обязательно переопределить.
    name = "base"

    # Какие действия задачке РАЗРЕШЕНЫ: {канал: [имена действий]}. Канала
    # нет в словаре — ему можно всё. Остальные действия сеть этого канала
    # в этой задаче не выберет вообще — ни жадно, ни случайно (маска в
    # py/dqn.py). Зачем: в "следить взглядом" ногам незачем ходить, а рукам
    # — ломать блоки; штраф за это сети выучили бы нескоро (первые часы
    # почти все действия случайные), а маска запрещает сразу. Меньше
    # вариантов — быстрее обучение. Первым в списке должно идти "ничего не
    # делать" канала (idle / head_idle / hands_idle).
    allowed_actions: dict[str, list[str]] = {}

    def __init__(self, config: dict):
        self.config = config
        # {"x":.., "y":.., "z":..} — точка, {"entity_id": id} — сущность
        # (Node сам каждый тик берёт её свежую позицию), None — цели нет.
        # Ядро сравнивает это значение с тем, что отправляло в прошлый раз,
        # и шлёт set_target в Node, только если оно реально поменялось.
        self.current_target: dict | None = None
        # Счётчики событий для метрик ({"goals": 3, "deaths": 1, ...}).
        # ai_loop.py периодически собирает их и обнуляет (take_events).
        self.events: dict[str, int] = {}

    def reset(self, state: dict) -> None:
        """Вызывается ядром на границе эпизода (после смерти и респавна).
        Тут стоит сбрасывать внутреннее состояние вроде "дистанция на
        прошлом тике" — иначе первый tick нового эпизода посчитает
        награду по дистанции из прошлой жизни бота."""

    def on_tick(self, state: dict) -> None:
        """Вызывается каждый тик, до compute_reward. Может обновить
        self.current_target (выбрать новую цель, заметить, что старая
        достигнута/недостижима, и т.п.).

        Если state["target_is_human"] истинно — человек держит цель на
        себе через !setTarget, модулю стоит не пытаться её перебить
        (это не значит "ничего не делай": current_target можно не трогать,
        Node всё равно проигнорирует его, пока человек не отпустит цель
        через !setTarget none)."""

    def compute_reward(self, prev_state: dict, curr_state: dict, actions: dict) -> dict:
        """Награды за переход prev_state -> curr_state: {канал: число} для
        КАЖДОГО канала из CHANNEL_NAMES. actions — {канал: имя действия},
        выбранные ИЗ prev_state (то есть те, что привели к curr_state).
        Обязателен к переопределению."""
        raise NotImplementedError

    @staticmethod
    def team(value: float) -> dict:
        """Одна и та же награда всем каналам — основа для compute_reward."""
        return {channel: value for channel in CHANNEL_NAMES}

    def count(self, event: str, amount: int = 1) -> None:
        self.events[event] = self.events.get(event, 0) + amount

    def take_events(self) -> dict:
        events, self.events = self.events, {}
        return events

    def observe(self, state: dict) -> dict:
        """Каким задачка показывает мир СЕТИ (награду при этом считает по
        настоящему state). По умолчанию — как есть; walking/follow, например,
        подставляют вместо цели ближайшую точку маршрута к ней."""
        return state

    @classmethod
    def summarize(cls, events: dict, ticks: int, bot_minutes: float) -> tuple[str, float | None]:
        """"Как дела" у задачки за окно метрик — одной строкой для человека
        (лаунчер показывает её как есть) плюс главное число задачки, по
        которому лаунчер рисует стрелку "лучше/хуже". events — сложенные
        счётчики всех ботов этой задачки за окно, ticks — сколько тиков они
        на ней провели, bot_minutes — то же в минутах на бота."""
        return "", None

    def describe(self) -> str:
        """Короткая строка для логов ai_loop.py."""
        target = self.current_target
        if target is None:
            target_str = "none"
        elif "entity_id" in target:
            target_str = f"сущность {target['entity_id']}"
        else:
            target_str = f"({target['x']:.1f}, {target['y']:.1f}, {target['z']:.1f})"
        return f"{self.name}: target={target_str}"
