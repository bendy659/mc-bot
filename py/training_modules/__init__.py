"""Реестр обучающих модулей — см. py/training_modules/base.py про то, что
такое модуль и как добавить свой.
"""

from __future__ import annotations

from .base import TrainingModule
from .bedwars import BedwarsModule
from .bridge import BridgeModule
from .chase import ChaseModule
from .crafting import CraftingModule
from .flee import FleeModule
from .follow import FollowModule
from .gathering import GatheringModule
from .hunt import HuntModule
from .looking import LookingModule
from .walking import WalkingModule

MODULES: dict[str, type[TrainingModule]] = {
    WalkingModule.name: WalkingModule,
    LookingModule.name: LookingModule,
    GatheringModule.name: GatheringModule,
    CraftingModule.name: CraftingModule,
    FollowModule.name: FollowModule,
    ChaseModule.name: ChaseModule,
    FleeModule.name: FleeModule,
    HuntModule.name: HuntModule,
    BridgeModule.name: BridgeModule,
    BedwarsModule.name: BedwarsModule,
}

DEFAULT_MODULE = WalkingModule.name

# Номер задачи для one-hot на входе сетей (state_encoder.encode_context).
# ТОЛЬКО дописывать в конец: перестановка переименует задачи для уже
# обученных сетей. Первые state_encoder.MAX_TASKS (8, hunt — последняя) —
# в блоке context в середине входа, следующие (bridge и дальше, до
# MAX_TASKS2 штук) — во втором блоке context2 в конце входа.
TASK_ORDER = ["walking", "looking", "gathering", "crafting", "follow", "chase", "flee", "hunt",
              "bridge", "bedwars"]

# Не задача, а режим: боты роя получают РАЗНЫЕ задачи из train.mix_tasks
# и по очереди меняют их (см. BotSession в ai_loop.py). Сети общие, так
# что за один прогон учатся все навыки сразу.
MIX = "mix"

# Тоже режим: салки (py/tag_game.py). Один бот убегает (задачка flee),
# остальные догоняют (chase); роли раздаёт и меняет судья.
TAG = "tag"


def task_index(name: str) -> int:
    return TASK_ORDER.index(name)


def allowed_actions(name: str) -> dict[str, list[str]]:
    """{канал: [разрешённые действия]} задачки name — для каждого канала,
    в порядке CHANNELS (канал, не упомянутый в allowed_actions модуля, —
    все его действия). py/dqn.py строит по этому маску действий мозга."""
    from protocol import CHANNELS

    declared = MODULES[name].allowed_actions
    result = {}
    for channel, actions in CHANNELS.items():
        names = declared.get(channel)
        if names is None:
            result[channel] = list(actions)
            continue
        unknown = set(names) - set(actions)
        if unknown:
            raise ValueError(f"{name}.allowed_actions[{channel}]: нет таких действий {unknown}")
        # Порядок — как в CHANNELS (он же порядок выходов сети).
        result[channel] = [action for action in actions if action in names]
    return result


def create_module(name: str, config: dict) -> TrainingModule:
    if name not in MODULES:
        available = ", ".join(sorted(MODULES))
        raise ValueError(f"Неизвестный обучающий модуль '{name}'. Доступны: {available}")
    return MODULES[name](config)


__all__ = [
    "TrainingModule", "MODULES", "DEFAULT_MODULE", "TASK_ORDER", "MIX", "TAG",
    "create_module", "task_index", "allowed_actions",
]
