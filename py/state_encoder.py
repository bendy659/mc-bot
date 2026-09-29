"""Кодирование JSON-состояния от Node в тензоры для сети.

Наблюдение состоит из двух веток:
  1. Визуальная: сетка лучей (resY x resX). Каждая ячейка приходит как
     5 значений (r, g, b, дистанция, класс блока). Хранится КОМПАКТНО —
     кадр (5, resY, resX) uint8: r, g, b, d квантованы в 0..255, класс —
     как есть (encode_state). В полный вид для сети — r, g, b, d в 0..1 +
     one-hot класса в block_classes каналов, (4 + block_classes, resY,
     resX) на кадр — разворачивает expand_vision, уже на видеокарте, перед
     самой сетью. Так в памяти опыта кадр занимает 320 байт вместо ~3.5 КБ,
     и у каждой задачки может быть своя память (см. py/dqn.py).
  2. Скалярная: слух (sectors значений) + данные о себе (углы, health,
     food, on_ground, дельты движения, относительная цель) + ближайшие
     сущности (9 значений на слот) + КОНТЕКСТ -> плоский вектор.

Контекст — то, чем сети каналов (ноги/голова/руки) "общаются" между собой:
  - one-hot активной задачи (walking/looking/gathering/...) — все сети
    знают, ради чего сейчас работает бот, и одни и те же веса ведут себя
    по-разному в разных задачах;
  - one-hot действий ВСЕХ каналов на прошлом тике — голова видит, что ноги
    сейчас поворачивают, руки видят, что голова навелась, и т.д.
Сети решают одновременно, поэтому "что делают соседи" — это их решение
прошлого тика (150 мс назад), а не текущего.
"""

import base64
import math

import torch

from protocol import ACTION_NAMES, CHANNEL_NAMES
from training_modules.targets import aim_errors

# Слотов под one-hot задачи — с запасом, чтобы добавление нового
# обучающего модуля (до MAX_TASKS штук) не ломало размерность входа и
# старые чекпоинты. Номер задачи — её позиция в TASK_ORDER
# (py/training_modules/__init__.py).
MAX_TASKS = 8
_ACTION_POSITION = {name: i for i, name in enumerate(ACTION_NAMES)}
INVENTORY_FEATURES = 8
# Блок context стоит в СЕРЕДИНЕ входа, и его размер менять нельзя: все входы
# после него (inventory, crosshair...) сдвинулись бы, а доращивание мозгов
# (dqn._grow_state_dict) дописывает нулевые столбцы только в конец — сеть
# молча перепутала бы входы. Поэтому в нём навсегда первые CONTEXT_ACTIONS
# имён ACTION_NAMES и MAX_TASKS задач, а всё, что добавлено позже, — во
# втором блоке context2 в КОНЦЕ входа, с запасом мест: ещё MAX_TASKS2 задач
# (bridge — первая) и EXTRA_ACTION_SLOTS действий (sneak_back — первое).
CONTEXT_ACTIONS = 22
MAX_TASKS2 = 8
EXTRA_ACTION_SLOTS = 16
assert len(ACTION_NAMES) <= CONTEXT_ACTIONS + EXTRA_ACTION_SLOTS, "кончились места под действия в context2"
_HELD_KINDS = ["block", "food", "tool", "other"]
# Прицел: класс блока строго по центру взгляда (one-hot по классам сетки
# зрения, 0 — ничего), дистанция до него и "достаёт ли рука".
REACH = 4.5  # как DIG_REACH / PLACE_REACH в js/actions.js
GROUND_RANGE = 3.0  # чувство пола меряет не дальше (js/vision.js: GROUND_RANGE)


def encode_state(state: dict, config: dict, task_index: int, prev_actions: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Возвращает (vision, scalars): компактный кадр зрения (5, H, W) uint8
    (в сеть — только через expand_vision) и скаляры float32.

    task_index — номер активной задачи (training_modules.task_index),
    prev_actions — {канал: имя действия} прошлого тика (см. контекст выше)."""
    vision = _encode_vision(state, config)
    scalars = torch.cat([
        _encode_scalars(state, config),
        encode_context(task_index, prev_actions),
        torch.tensor(_encode_inventory(state), dtype=torch.float32),
        torch.tensor(_encode_crosshair(state, config), dtype=torch.float32),
        torch.tensor(_encode_strike(state), dtype=torch.float32),
        torch.tensor([state.get("attack_charge", 1.0)], dtype=torch.float32),
        torch.tensor(_encode_goal(state), dtype=torch.float32),
        torch.tensor(_encode_route_open(state), dtype=torch.float32),
        torch.tensor([1.0 if state.get("hunt_closest") else 0.0], dtype=torch.float32),
        encode_context2(task_index, prev_actions),
        torch.tensor(_encode_ground(state), dtype=torch.float32),
        torch.tensor([1.0 if state.get("can_place") else 0.0], dtype=torch.float32),
    ])
    return vision, scalars


def encode_context(task_index: int, prev_actions: dict) -> torch.Tensor:
    """One-hot задачи (MAX_TASKS) + one-hot прошлых действий всех каналов
    (по одной единице на канал внутри общего вектора ACTION_NAMES) — только
    первые MAX_TASKS задач и CONTEXT_ACTIONS действий, остальное — в
    encode_context2 (конец входа)."""
    context = torch.zeros(MAX_TASKS + CONTEXT_ACTIONS)
    if 0 <= task_index < MAX_TASKS:
        context[task_index] = 1.0
    for channel in CHANNEL_NAMES:
        position = _ACTION_POSITION.get(prev_actions.get(channel))
        if position is not None and position < CONTEXT_ACTIONS:
            context[MAX_TASKS + position] = 1.0
    return context


def encode_context2(task_index: int, prev_actions: dict) -> torch.Tensor:
    """Второй блок контекста (в конце входа): one-hot задачи с номером от
    MAX_TASKS (bridge и дальше) и прошлые действия, появившиеся после первых
    CONTEXT_ACTIONS. У старых задач и действий здесь нули — доращённый мозг
    считает на них ровно то же, что и раньше."""
    context = torch.zeros(MAX_TASKS2 + EXTRA_ACTION_SLOTS)
    if MAX_TASKS <= task_index < MAX_TASKS + MAX_TASKS2:
        context[task_index - MAX_TASKS] = 1.0
    for channel in CHANNEL_NAMES:
        position = _ACTION_POSITION.get(prev_actions.get(channel))
        if position is not None and position >= CONTEXT_ACTIONS:
            context[MAX_TASKS2 + position - CONTEXT_ACTIONS] = 1.0
    return context


class PackedCells:
    """Сетка зрения, пришедшая байтами (js/state.js: packVision): r, g, b, d
    — 0..255, класс — число. Для модулей выглядит как прежний список cells
    (r, g, b, d в 0..1), кодировщик берёт байты напрямую."""

    __slots__ = ("raw",)

    def __init__(self, raw: bytes):
        self.raw = raw

    def __len__(self) -> int:
        return len(self.raw)

    def __getitem__(self, index: int):
        value = self.raw[index]
        return value if index % 5 == 4 else value / 255.0


def unpack_vision(state: dict) -> dict:
    """Зрение байтами (vision.packed, base64) -> vision.cells = PackedCells.
    Вызывать сразу после разбора JSON — до модулей. Старый формат (список
    cells) — как есть."""
    vision = state.get("vision")
    if vision and "packed" in vision:
        vision["cells"] = PackedCells(base64.b64decode(vision.pop("packed")))
    return state


def _encode_vision(state: dict, config: dict) -> torch.Tensor:
    res_x, res_y = state["vision"]["resolution"]
    cells = state["vision"]["cells"]
    num_classes = config["vision"]["block_classes"]
    if isinstance(cells, PackedCells) and len(cells.raw) == res_x * res_y * 5:
        # Уже байты ровно того вида, что хранит память опыта.
        compact = torch.frombuffer(bytearray(cells.raw), dtype=torch.uint8).reshape(res_y, res_x, 5)
        compact[:, :, 4].clamp_(max=num_classes - 1)
        return compact.permute(2, 0, 1).contiguous()
    # Node шлёт 5 значений на ячейку (r, g, b, d, id класса); one-hot
    # разворачивается уже здесь, на стороне Python.
    wire_stride = 5
    expected = res_x * res_y * wire_stride

    if len(cells) != expected:
        # Рассинхрон невозможен при общем config.json, но подстраховка
        # дешевле, чем молча обучаться на мусоре.
        raise ValueError(
            f"Сетка зрения пришла размером {len(cells)}, ожидалось {expected} "
            f"(resolution {res_x}x{res_y}, {wire_stride} значений на ячейку). "
            f"Проверь, что js/ и py/ читают один и тот же config.json."
        )

    raw = torch.tensor(cells, dtype=torch.float32).reshape(res_y, res_x, wire_stride)
    compact = torch.empty(res_y, res_x, wire_stride, dtype=torch.uint8)
    # r, g, b, дистанция: 0..1 -> 0..255 (точности 1/255 с запасом: цвет —
    # грубая палитра, дистанция — доли от 32 блоков, шаг ~0.13 блока).
    compact[:, :, :4] = (raw[:, :, :4].clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
    compact[:, :, 4] = raw[:, :, 4].clamp(0, num_classes - 1).to(torch.uint8)
    # (H, W, C) -> (C, H, W), как принято в свёрточных сетях.
    return compact.permute(2, 0, 1).contiguous()


def expand_vision(compact: torch.Tensor, config: dict) -> torch.Tensor:
    """Компактные кадры -> вход сети. (B, 5*K, H, W) uint8 (стек из K
    кадров, см. FrameStacker) -> (B, (4 + block_classes)*K, H, W) float32:
    r, g, b, d обратно в 0..1 и one-hot класса блока. Работает на том же
    устройстве, где лежит тензор (обычно — на видеокарте, целым батчем).

    One-hot несёт не меньше смысла, чем "сырое" число: соседние id (3 и 4 —
    земля и растения) не «ближе» друг к другу по смыслу, чем 3 и 9."""
    if compact.dim() == 3:
        compact = compact.unsqueeze(0)
    batch, channels, height, width = compact.shape
    frames = channels // 5
    num_classes = config["vision"]["block_classes"]

    x = compact.view(batch, frames, 5, height, width)
    rgbd = x[:, :, :4].to(torch.float32) / 255.0                      # (B, K, 4, H, W)
    one_hot = torch.nn.functional.one_hot(x[:, :, 4].long(), num_classes)  # (B, K, H, W, C)
    one_hot = one_hot.permute(0, 1, 4, 2, 3).to(torch.float32)         # (B, K, C, H, W)
    return torch.cat([rgbd, one_hot], dim=2).reshape(batch, frames * (4 + num_classes), height, width)


def _encode_scalars(state: dict, config: dict) -> torch.Tensor:
    features = []

    # Слух — уже нормирован 0..1 на стороне Node.
    sectors = config["hearing"]["sectors"]
    hearing = state.get("hearing") or []
    features.extend(_pad(hearing, sectors))

    self_state = state["self"]

    # Наклон головы (sin/cos — угол периодичен). АБСОЛЮТНОГО поворота (yaw)
    # на входе НЕТ сознательно: всё остальное наблюдение относительное
    # (круговое зрение поворачивается вместе с ботом, сущности и цель — в
    # его системе координат, слух тоже), а абсолютный yaw годится только
    # для заучивания сторон света. Проверено вживую: с ним looking выучил
    # "смотри на юго-запад" вместо "смотри на цель" — работало, пока боты
    # стояли на месте, и переставало, как только они сдвигались.
    features.append(math.sin(self_state["pitch"]))
    features.append(math.cos(self_state["pitch"]))

    features.append(self_state["health"] / 20.0)
    features.append(self_state.get("food", 20.0) / 20.0)
    features.append(1.0 if self_state["on_ground"] else 0.0)

    # Дельты прошлого тика — "память о движении": реально ли сдвинулся,
    # упёрся ли в стену (move_forward=0 при зажатом вперёд), падает ли.
    # Сдвиги нормируем на ~1 блок за тик (при 150 мс это быстрее ходьбы),
    # углы — на pi. Поля появлялись постепенно, поэтому .get с нулями.
    features.append(self_state.get("move_forward", 0.0))
    features.append(self_state.get("move_right", 0.0))
    features.append(self_state.get("move_up", 0.0))
    features.append(self_state.get("dyaw", 0.0) / math.pi)
    features.append(self_state.get("dpitch", 0.0) / math.pi)

    features.extend(_encode_target(state))

    features.extend(_encode_entities(state, config))

    return torch.tensor(features, dtype=torch.float32)


def _encode_inventory(state: dict) -> list[float]:
    """Сводка инвентаря от Node (js/inventory.js) -> 8 чисел: сколько блоков
    (для строительства), еды, брони в инвентаре, сколько брони надето и
    что в руке (one-hot). Нет сводки (запись геймплея) — нули."""
    inventory = state.get("inventory") or {}
    held = [1.0 if inventory.get("held") == kind else 0.0 for kind in _HELD_KINDS]
    return [
        min(inventory.get("blocks", 0), 64) / 64.0,
        min(inventory.get("food", 0), 16) / 16.0,
        min(inventory.get("armor_items", 0), 4) / 4.0,
        inventory.get("armor_worn", 0) / 4.0,
        *held,
    ]


def crosshair_features(config: dict) -> int:
    return config["vision"]["block_classes"] + 2


def _encode_crosshair(state: dict, config: dict) -> list[float]:
    """Во что бот целится: класс блока по центру прицела (one-hot), дистанция
    до него (доля дальности зрения) и достаёт ли рука (1/0). Круговая сетка
    зрения за наклоном головы не следует — без этого, глядя вниз на блок,
    бот "не видел", во что целится, и наводиться (копать, ставить) учился
    вслепую. Ничего в прицеле (или запись геймплея без класса) — класс 0."""
    classes = config["vision"]["block_classes"]
    center = state.get("center_block") or {}
    block_class = center.get("t", 0) if center else 0
    one_hot = [0.0] * classes
    one_hot[block_class if 0 <= block_class < classes else classes - 1] = 1.0
    distance = center.get("distance")
    if distance is None:
        return one_hot + [1.0, 0.0]
    max_distance = config["vision"]["distance"] * 16
    return one_hot + [min(distance / max_distance, 1.0), 1.0 if distance <= REACH else 0.0]


def _encode_goal(state: dict) -> list[float]:
    """Где сама цель, если вместо неё сеть видит точку маршрута (observe:
    walking/chase/hunt): насколько она выше (+) или ниже, в долях 8 блоков, и
    далеко ли по горизонтали, в долях 16 блоков. Для цели на столбе точка
    маршрута — у основания, на земле, и без этого сеть не видела, что цель
    высоко (охотники прыгали у столба, 2026-09-27). Цели нет — нули."""
    goal = state.get("goal") or state.get("target")
    me = state.get("self")
    if not goal or not me:
        return [0.0, 0.0]
    rise = max(-1.0, min(1.0, (goal["y"] - me["y"]) / 8.0))
    flat = min(1.0, math.hypot(goal["x"] - me["x"], goal["z"] - me["z"]) / 16.0)
    return [rise, flat]


def _encode_ground(state: dict) -> list[float]:
    """Чувство пола (js/vision.js: groundProbe): сколько пола до края впереди,
    справа, сзади и слева, в долях GROUND_RANGE, со знаком — над пустотой
    (свесился с края) минус, сколько до пола. Сетка зрения край ближе ~2.3
    блока не видит: без этого входа мост строили крадучись от самого старта
    (2026-09-28). Нет поля (запись геймплея, старое тело) — пол вокруг (1.0):
    так чаще всего и есть."""
    ground = state.get("ground")
    if not ground:
        return [1.0] * 4
    return [max(-1.0, min(1.0, value / GROUND_RANGE)) for value in ground]


def _encode_route_open(state: dict) -> list[float]:
    """Дойти до цели можно (+1), маршрута нет — цель за стенами или высоко
    (-1), маршрут не строился (0). Подсказка "пора ломать стену или строить",
    а не искать обход (цель в коробке — автор, 2026-09-28)."""
    route = state.get("route")
    if not route:
        return [0.0]
    return [1.0 if route.get("complete") else -1.0]


def _encode_strike(state: dict) -> list[float]:
    """Достанет ли удар (attack_center) кого-нибудь прямо сейчас: 1/0 от Node
    (js/bot.js: state.strike — та же проверка, что у самого удара). Нет поля
    (запись геймплея, старый Node) — 0."""
    return [1.0 if state.get("strike") else 0.0]


def _encode_target(state: dict) -> list[float]:
    """Цель — только в системе координат бота, 8 чисел:
      - есть ли цель (1/0) — иначе "цели нет" и "цель ровно в ногах" выглядели
        бы одинаково (нули);
      - насколько довернуть взгляд, чтобы смотреть на цель: по горизонтали
        (sin/cos угла — он периодичен) и по вертикали (в долях от 90°). Это
        ровно та ошибка, по которой считает награду looking
        (training_modules.targets.aim_errors), и она не зависит ни от места,
        ни от расстояния: "цель левее на 20°" — одинаковый сигнал и в 3, и в
        30 блоках. Раньше сеть должна была сама выводить угол из координат,
        делённых на 32 блока (у цели в 4 блоках — числа порядка 0.1);
      - где цель относительно тела: впереди / справа / выше и дистанция (в
        долях от 32 блоков) — для ходьбы к ней.
    Вперёд f = (-sin yaw, -cos yaw) (конвенция mineflayer, см. js/vision.js),
    вправо r = (-f.z, f.x) — та же математика, что в js/state.js и
    js/entities.js."""
    target = state.get("target")
    if target is None:
        return [0.0] * 8

    self_state = state["self"]
    dx = target["x"] - self_state["x"]
    dy = target["y"] - self_state["y"]
    dz = target["z"] - self_state["z"]
    fx = -math.sin(self_state["yaw"])
    fz = -math.cos(self_state["yaw"])
    forward = dx * fx + dz * fz
    right = dx * -fz + dz * fx
    distance = math.sqrt(dx * dx + dy * dy + dz * dz)
    h_error, v_error = aim_errors(self_state, target)
    return [
        1.0,
        math.sin(h_error), math.cos(h_error), v_error / (math.pi / 2),
        forward / 32.0, right / 32.0, dy / 32.0, distance / 32.0,
    ]


def _encode_entities(state: dict, config: dict) -> list[float]:
    """Ближайшие сущности -> 9 чисел на слот: present, forward, right, up,
    dist (всё уже нормировано на стороне Node) + one-hot типа сущности
    (0 враждебная, 1 пассивная, 2 игрок, 3 предмет). id в сеть не идёт —
    он нужен модулям награды (gathering ловит исчезновение id = убийство),
    а не сети."""
    entities = state.get("entities") or []
    out: list[float] = []
    for i in range(config["entities"]["max_tracked"]):
        if i < len(entities) and entities[i].get("present"):
            e = entities[i]
            type_one_hot = [0.0] * 4
            type_idx = int(e.get("type", 1))
            if 0 <= type_idx < 4:
                type_one_hot[type_idx] = 1.0
            out.extend([
                1.0, e.get("forward", 0.0), e.get("right", 0.0),
                e.get("up", 0.0), e.get("dist", 0.0), *type_one_hot,
            ])
        else:
            out.extend([0.0] * 9)
    return out


def _pad(values: list, size: int) -> list:
    values = list(values[:size])
    return values + [0.0] * (size - len(values))


def scalar_layout(config: dict) -> dict[str, slice]:
    """Где что лежит в скалярном векторе — в том же порядке, в каком его
    собирает encode_state. По этим именам задачка может отключить себе
    лишние входы (modules.<задачка>.inputs, см. py/dqn.py)."""
    sizes = [
        ("hearing", config["hearing"]["sectors"]),        # слух
        ("pitch", 2),                                      # sin/cos pitch (абсолютного yaw нет — см. _encode_scalars)
        ("body", 3),                                       # health, food, on_ground
        ("motion", 5),                                     # дельты: forward/right/up, dyaw, dpitch
        ("target", 8),                                     # цель: есть ли, ошибка взгляда, где она, дистанция
        ("entities", 9 * config["entities"]["max_tracked"]),  # сущности (9 значений на слот)
        ("context", MAX_TASKS + CONTEXT_ACTIONS),          # задача + прошлые действия каналов (размер — навсегда)
        # Всё новое — ТОЛЬКО В КОНЕЦ: тогда обученные сети доращиваются
        # новыми входами без потери навыков (py/dqn.py: _load_grown).
        ("inventory", INVENTORY_FEATURES),                 # блоки, еда, броня, что в руке
        ("crosshair", crosshair_features(config)),         # во что целится: класс блока, дистанция, достаёт ли
        ("strike", 1),                                     # удар сейчас достанет кого-нибудь (1/0)
        ("charge", 1),                                     # заряд удара 0..1 (нет поля — 1: полный)
        ("goal", 2),                                       # сама цель (а не точка маршрута): выше/ниже, как далеко
        ("route_open", 1),                                 # маршрут к цели: есть +1, нет -1, не строился 0
        # Охота: я ближе всех охотников к цели (ai_loop, hunt_game.closest).
        # На столб за целью лезет ближайший, остальные копают под ней — без
        # этого входа сеть не отличала одно от другого и копировала
        # большинство: стояла у столба (2026-09-28).
        ("closest", 1),
        # Задачи с номером от MAX_TASKS и действия новее первых CONTEXT_ACTIONS
        # (места с запасом: дальше они добавляются без смены размера входа).
        ("context2", MAX_TASKS2 + EXTRA_ACTION_SLOTS),
        # Чувство пола: сколько пола до края впереди/справа/сзади/слева (со
        # знаком). Край под ногами сетка зрения не видит (мост, 2026-09-29).
        ("ground", 4),
        # Блок перед собой (place_front) сейчас встанет — 1/0 от тела (как
        # strike для удара). У края моста окно для блока — 0.2857..0.3 свеса,
        # и по чувству пола "в окне" и "замерла чуть раньше" различались
        # только в третьем знаке (2026-09-29).
        ("place", 1),
    ]
    layout, start = {}, 0
    for name, size in sizes:
        layout[name] = slice(start, start + size)
        start += size
    return layout


def scalar_dim(config: dict) -> int:
    """Размер скалярного входа (вместе с контекстом) — нужен сети при
    инициализации. Обязан совпадать с тем, что собирает encode_state."""
    return max(part.stop for part in scalar_layout(config).values())


class FrameStacker:
    """Стек последних K vision-сеток: сеть видит движение "глазами",
    а не одним статичным кадром. Кадры идут дополнительными каналами.

    Один экземпляр на наблюдаемую сущность: у DQN-цикла свой, у train_bc
    — свой на каждую непрерывную запись. reset() вызывается при разрыве
    последовательности (смерть, пауза записи, новый эпизод)."""

    def __init__(self, config: dict):
        self.k = config["memory"]["frame_stack"]
        self.frames: list[torch.Tensor] = []

    def reset(self, first_vision: torch.Tensor):
        self.frames = [first_vision.clone() for _ in range(self.k)]

    def push(self, vision: torch.Tensor) -> torch.Tensor:
        if len(self.frames) < self.k:
            self.reset(vision)
        else:
            self.frames.pop(0)
            self.frames.append(vision.clone())
        return self.stacked()

    def stacked(self) -> torch.Tensor:
        # Каналы: [самый старый кадр ... последний кадр], по 5 компактных
        # каналов на кадр (expand_vision развернёт каждый в 4 + block_classes).
        return torch.cat(self.frames, dim=0)


def vision_channels(config: dict) -> int:
    """Число каналов CNN-входа (после expand_vision): (4 базовых (r,g,b,
    дистанция) + one-hot классов блоков) * глубина стека кадров."""
    return (4 + config["vision"]["block_classes"]) * config["memory"]["frame_stack"]


# Утилита для отладки/логов.
def preview(state: dict) -> str:
    res_x, res_y = state["vision"]["resolution"]
    hearing = state.get("hearing") or []
    hearing_str = " ".join(f"{v:.1f}" for v in hearing)
    return (
        f"vision {res_x}x{res_y}, hearing[{hearing_str}], "
        f"health={state['self']['health']}, food={state['self'].get('food', '?')}, "
        f"target={state.get('target_description', 'none')}"
    )
