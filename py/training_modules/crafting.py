"""Заглушка модуля crafting — обучение использованию верстака/печки/сундука.

Реализация пока не нужна (PROMPT_FOR_CLAUDE.md, пункт 4 в списке модулей) —
интерфейс достаточен, чтобы дописать модуль позже, без переделок ядра.
Сейчас модуль просто не мешает жить (нулевая награда, кроме штрафа за
смерть) и не выбирает целей.

Чтобы реализовать по-настоящему, минимально понадобится:
  - новое действие-макрос канала hands вроде `use_center`
    (bot.activateBlock на верстак/печку/сундук в центре прицела) — по
    аналогии с attack_center (js/actions.js, py/protocol.py — не забыть
    добавить в оба CHANNELS синхронно);
  - наблюдение "открыт ли контейнер сейчас" и, возможно, его содержимое —
    по аналогии с center_block в js/state.js;
  - награда за успешное открытие нужного блока / получение нужного
    предмета крафтом.
"""

from __future__ import annotations

from .base import TrainingModule


class CraftingModule(TrainingModule):
    name = "crafting"

    def __init__(self, config: dict):
        super().__init__(config)
        self.death_penalty = config["reward"]["death"]

    def reset(self, state: dict) -> None:
        pass

    def on_tick(self, state: dict) -> None:
        self.current_target = None

    @classmethod
    def summarize(cls, events: dict, ticks: int, bot_minutes: float) -> tuple[str, float | None]:
        return "заглушка — пока ничему не учит", None

    def compute_reward(self, prev_state: dict, curr_state: dict, actions: dict) -> dict:
        if curr_state.get("dead"):
            self.count("deaths")
            return self.team(self.death_penalty)
        return self.team(0.0)  # TODO: пока не учит крафту, но интерфейс рабочий
