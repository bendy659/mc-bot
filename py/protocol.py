"""Контракт сообщений между Node и Python.

Node -> Python:
  {'type': 'state', ...}                  — состояние мира на тик (js/state.js);
  {'type': 'command', 'cmd': 'set_task', 'task': ..., 'bot_id': ...}
                                          — чат-команда !task (js/bot.js).
Python -> Node:
  {'type': 'action', 'actions': {'legs': ..., 'head': ..., 'hands': ...}, 'bot_id': ...}
                                          — по одному макросу на КАЖДЫЙ канал,
                                            Node выполняет их одновременно (js/actions.js);
  {'type': 'set_target', 'position': ..., 'bot_id': ...}.

Каналы ("органы") — у каждого своя сеть, они работают параллельно:
  legs  — перемещение и поворот корпуса (он же поворот обзора: в
          Minecraft у игрока один yaw на движение и взгляд);
  head  — только взгляд: наклон и мелкий доворот;
  hands — взаимодействие с тем, что в прицеле.
Разделение и есть решение задачи "walking бесцельно вертит головой": у
сети ног физически нет действий look_up/look_down, а сеть головы
отдельно учится не дёргаться, когда задача этого не требует.

CHANNELS обязан совпадать с CHANNELS в js/actions.js (и имена действий,
и принадлежность к каналу). Порядок каналов и действий внутри канала
важен для Python (индексы выходов сетей), Node шлёт/получает имена.
НОВЫЕ ДЕЙСТВИЯ — ТОЛЬКО В КОНЕЦ своего канала (и в конец ACTION_NAMES):
тогда уже обученные сети "доращиваются" новыми выходами без потери
навыков (py/dqn.py: _grow_state_dict), а не учатся заново. Прошлые
действия после первых 22 (state_encoder.CONTEXT_ACTIONS) идут на вход во
втором блоке контекста в конце — блок context в середине входа менять
нельзя (сдвинулись бы все входы после него).
ACTION_ROTATION обязан совпадать с шагами поворота в js/actions.js
(TURN_STEP, LOOK_STEP, LOOK_YAW_STEP — оттуда же они экспортируются).
"""

import math

CHANNELS = {
    "legs": [
        "idle",            # исторически "idle" — так оно лежит в старых записях
        "walk_forward",
        "sprint_forward",  # то же движение, что walk_forward, + control state sprint
        "walk_back",
        "strafe_left",
        "strafe_right",
        "jump_forward",
        "turn_left",       # поворот корпуса на 30°
        "turn_right",
        "jump",            # прыжок на месте (без движения вперёд) — например, чтобы поставить блок под себя
        "sneak_back",      # задом крадучись: с края блока крадущийся не сходит (мост над пустотой)
        "sneak",           # стоять крадучись (у края — не сорваться)
    ],
    "head": [
        "head_idle",
        "look_up",
        "look_down",
        "look_left",       # доворот взгляда на 10° — точнее, чем turn_* ног
        "look_right",
    ],
    "hands": [
        "hands_idle",
        "attack_center",   # удар/копание того, что в центре прицела
        "place_below",     # поставить блок из инвентаря под себя (получится только в прыжке)
        "place_front",     # поставить блок на грань того, во что смотрит прицел
        "use_item",        # съесть еду (если голоден) или использовать предмет в руке
        "equip_armor",     # надеть броню из инвентаря
        "drop_item",       # выбросить то, что в руке
    ],
}

CHANNEL_NAMES = list(CHANNELS)

# "Ничего не делать" для каждого канала — стартовое значение "прошлого
# действия" и заглушка для старых записей, где было одно действие на тик.
IDLE_ACTIONS = {channel: actions[0] for channel, actions in CHANNELS.items()}

# Все имена одним списком — порядок one-hot "что делали все каналы на
# прошлом тике" (state_encoder.encode_context). Порядок — по времени
# появления, ТОЛЬКО дописывать в конец: позиция действия — это номер входа
# обученной сети, и сдвинуть его значит перепутать ей входы.
ACTION_NAMES = [
    # первое поколение
    "idle", "walk_forward", "sprint_forward", "walk_back", "strafe_left", "strafe_right",
    "jump_forward", "turn_left", "turn_right",
    "head_idle", "look_up", "look_down", "look_left", "look_right",
    "hands_idle", "attack_center",
    # 2026-09-25: салки и свобода действий
    "jump", "place_below", "place_front", "use_item", "equip_armor", "drop_item",
    # 2026-09-28: мост (bridge). Всё, что дальше CONTEXT_ACTIONS (22), сеть
    # видит во втором блоке контекста в конце входа (state_encoder.context2).
    "sneak_back", "sneak",
]
assert sorted(ACTION_NAMES) == sorted(name for actions in CHANNELS.values() for name in actions), \
    "ACTION_NAMES должен содержать ровно действия из CHANNELS"
NUM_ACTIONS = len(ACTION_NAMES)

# На сколько поворачивает взгляд каждый макрос: (dyaw, dpitch) в радианах,
# конвенция mineflayer (yaw растёт влево, pitch > 0 — вверх). Нужно модулям
# наград, чтобы оценить вклад КАЖДОГО канала в доворот к цели отдельно
# (см. looking.py: ноги и голова крутят один и тот же взгляд). Шаги — те же,
# что в js/actions.js: корпус 30°, голова 10°.
TURN_STEP = math.radians(30)
LOOK_STEP = math.radians(10)
LOOK_YAW_STEP = math.radians(10)
ACTION_ROTATION = {
    "turn_left": (TURN_STEP, 0.0),
    "turn_right": (-TURN_STEP, 0.0),
    "look_left": (LOOK_YAW_STEP, 0.0),
    "look_right": (-LOOK_YAW_STEP, 0.0),
    "look_up": (0.0, LOOK_STEP),
    "look_down": (0.0, -LOOK_STEP),
}

# Имя действия -> канал, и имя -> индекс ВНУТРИ своего канала (выход сети).
ACTION_CHANNEL = {name: channel for channel, actions in CHANNELS.items() for name in actions}
ACTION_INDEX_IN_CHANNEL = {
    name: i for actions in CHANNELS.values() for i, name in enumerate(actions)
}
