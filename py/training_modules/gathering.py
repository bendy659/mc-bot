"""Задачка "добыча": добыть норму нужного ресурса (по умолчанию 5 брёвен).

Какой ресурс и сколько — config.json -> modules.gathering:
  resource_items — регулярка по именам ПРЕДМЕТОВ в инвентаре (что считаем
                   добычей: "_log$" — любые брёвна);
  resource_class — класс БЛОКА в сетке зрения (js/vision.js,
                   BLOCK_CLASS_PATTERNS: 1 — древесина, 2 — камень,
                   6 — руда...), по нему бот "видит", куда идти;
  quota          — сколько добыть за одно задание.
Ресурс в сеть не подаётся (одна задача "gathering" на входе), поэтому
сеть учится добывать именно то, что стоит в конфиге. Сменил ресурс —
сети придётся переучиваться.

Награда (основа — командная: дойти, навестись и ударить — работа всех
трёх каналов):
  - +item_reward за каждый добытый предмет-ресурс (по инвентарю: Node
    шлёт inventory_items), +other_item_reward за прочие предметы;
  - норма выполнена — +quota_reward и сразу новая норма;
  - подсказки по пути (potential-based: за приближение плюс, за отход —
    такой же минус, "нафармить" нельзя):
      * approach_reward за каждый блок сближения с ближайшим блоком
        ресурса в сетке зрения;
      * pickup_reward за сближение с выпавшим предметом (дроп сам в
        инвентарь не прыгает — к нему надо подойти);
  - голове +aim_reward за тик, пока в прицеле блок ресурса на расстоянии
    удара; рукам +hit_reward за attack_center по нему (копка бревна
    руками идёт ~3 с — это серия ударов, а не один);
  - убийство враждебного моба рядом — kill_reward (как раньше);
  - удар в воздух — штраф рукам, все каналы стоят — штраф за безделье.
"""

from __future__ import annotations

import math
import re

from .base import TrainingModule
from .targets import HOSTILE, ITEM, present_entities

DIG_REACH = 4.5  # как DIG_REACH в js/actions.js


class GatheringModule(TrainingModule):
    name = "gathering"
    # Рукам — только бить/копать: ставить блоки, есть и выбрасывать добыче
    # не нужно (а выброшенное бревно — минус из нормы).
    allowed_actions = {"hands": ["hands_idle", "attack_center"]}

    def __init__(self, config: dict):
        super().__init__(config)
        cfg = config["modules"]["gathering"]
        self.resource_items = re.compile(cfg.get("resource_items", "_log$"))
        self.resource_class = cfg.get("resource_class", 1)
        self.quota = cfg.get("quota", 5)
        self.item_reward = cfg.get("item_reward", 10.0)
        self.other_item_reward = cfg.get("other_item_reward", 1.0)
        self.quota_reward = cfg.get("quota_reward", 100.0)
        self.approach_reward = cfg.get("approach_reward", 0.5)
        self.pickup_reward = cfg.get("pickup_reward", 0.5)
        self.aim_reward = cfg.get("aim_reward", 0.1)
        self.hit_reward = cfg.get("hit_reward", 0.2)
        self.kill_reward = cfg.get("kill_reward", 15.0)
        self.kill_radius = cfg.get("kill_radius", 4.0)
        self.miss_penalty = cfg["miss_penalty"]

        self.idle_penalty = config["reward"]["idle"]
        self.death_penalty = config["reward"]["death"]
        self.max_view_distance = config["vision"]["distance"] * 16
        self.entity_radius = config["entities"]["radius"]

        self._collected = 0          # добыто в счёт текущей нормы
        self._prev_resource_distance: float | None = None
        self._prev_item_distance: float | None = None

    def reset(self, state: dict) -> None:
        # После смерти инвентарь выпал — норма начинается заново.
        self._collected = 0
        self._prev_resource_distance = None
        self._prev_item_distance = None

    def on_tick(self, state: dict) -> None:
        # Добыча не расставляет цели-точки: ресурс бот ищет глазами.
        self.current_target = None

    def compute_reward(self, prev_state: dict, curr_state: dict, actions: dict) -> dict:
        if curr_state.get("dead"):
            self.count("deaths")
            return self.team(self.death_penalty)
        if prev_state is None:
            return self.team(0.0)

        team = 0.0

        if self._mob_killed(prev_state, curr_state):
            self.count("kills")
            team += self.kill_reward

        gained_resource, gained_other = self._inventory_gain(prev_state, curr_state)
        if gained_resource:
            self.count("gathered", gained_resource)
            team += self.item_reward * gained_resource
            self._collected += gained_resource
            if self._collected >= self.quota:
                self.count("quota_done")
                team += self.quota_reward
                self._collected = 0  # сразу новая норма
        if gained_other:
            self.count("other_items", gained_other)
            team += self.other_item_reward * gained_other

        team += self._approach_shaping(curr_state)

        all_idle = actions["legs"] == "idle" and actions["head"] == "head_idle" \
            and actions["hands"] == "hands_idle"
        if all_idle:
            team += self.idle_penalty

        rewards = self.team(team)

        # Точечно: голова навелась на ресурс в зоне удара (смотрим, куда она
        # довернулась — curr), руки ударили по ресурсу (что было в прицеле в
        # момент удара — prev).
        if self._aiming_at_resource(curr_state):
            rewards["head"] += self.aim_reward
        if actions["hands"] == "attack_center":
            if self._aiming_at_resource(prev_state):
                rewards["hands"] += self.hit_reward
            elif prev_state.get("center_block") is None and not _entity_in_front(prev_state):
                # Удар в воздух: под прицелом нет ни блока, ни сущности.
                rewards["hands"] += self.miss_penalty

        return rewards

    @classmethod
    def summarize(cls, events: dict, ticks: int, bot_minutes: float) -> tuple[str, float | None]:
        gathered = events.get("gathered", 0)
        rate = gathered / bot_minutes * 10 if bot_minutes > 0 else 0.0
        text = f"добыто ресурса: {gathered} ({rate:.1f} за 10 мин на бота)"
        if events.get("quota_done"):
            text += f", норм выполнено: {events['quota_done']}"
        if events.get("other_items"):
            text += f", прочих предметов: {events['other_items']}"
        return text, rate

    def _inventory_gain(self, prev_state: dict, curr_state: dict) -> tuple[int, int]:
        """(сколько прибавилось ресурса, сколько — прочих предметов)."""
        prev_items = prev_state.get("inventory_items") or {}
        curr_items = curr_state.get("inventory_items") or {}
        resource = other = 0
        for name, count in curr_items.items():
            gained = count - prev_items.get(name, 0)
            if gained <= 0:
                continue
            if self.resource_items.search(name):
                resource += gained
            else:
                other += gained
        return resource, other

    def _approach_shaping(self, curr_state: dict) -> float:
        reward = 0.0

        resource_distance = self._nearest_resource_in_view(curr_state)
        if resource_distance is not None and self._prev_resource_distance is not None:
            reward += self.approach_reward * max(-1.0, min(1.0, self._prev_resource_distance - resource_distance))
        self._prev_resource_distance = resource_distance

        item_distance = self._nearest_item(curr_state)
        if item_distance is not None and self._prev_item_distance is not None:
            reward += self.pickup_reward * max(-1.0, min(1.0, self._prev_item_distance - item_distance))
        self._prev_item_distance = item_distance
        return reward

    def _nearest_resource_in_view(self, state: dict) -> float | None:
        """Дистанция (блоки) до ближайшей ячейки сетки зрения с классом
        ресурса, None — ресурса не видно."""
        cells = (state.get("vision") or {}).get("cells") or []
        best = None
        for i in range(0, len(cells) - 4, 5):  # r, g, b, d, класс
            if int(cells[i + 4]) == self.resource_class:
                distance = cells[i + 3] * self.max_view_distance
                if best is None or distance < best:
                    best = distance
        return best

    def _nearest_item(self, state: dict) -> float | None:
        # Класс ITEM — всё неживое (стрелы, шары опыта тоже), а подбирать
        # стоит только выпавшие предметы — сущность "item".
        items = [e for e in present_entities(state) if e.get("type") == ITEM and e.get("name", "item") == "item"]
        if not items:
            return None
        return min(e.get("dist", 1.0) for e in items) * self.entity_radius

    def _aiming_at_resource(self, state: dict) -> bool:
        center = state.get("center_block")
        if not center:
            return False
        # Прицел — блок; ресурс ли он — по имени блока (брёвна: oak_log...).
        # Имена блока и предмета у брёвен/руды совпадают или близки, поэтому
        # та же регулярка resource_items подходит для обоих.
        return bool(self.resource_items.search(center.get("name", ""))) and \
            center.get("distance", math.inf) <= DIG_REACH

    def _mob_killed(self, prev_state: dict, curr_state: dict) -> bool:
        prev_ids = _nearby_hostile_ids(prev_state, self.kill_radius, self.config)
        if not prev_ids:
            return False
        # Моб мог бы просто выйти из радиуса — поэтому curr-радиус чуть
        # шире: исчезновение считается убийством, только если моба нет
        # и в расширенном радиусе (убегающий моб остался бы в нём).
        curr_ids = _nearby_hostile_ids(curr_state, self.kill_radius + 2.0, self.config)
        return bool(prev_ids - curr_ids)


def _nearby_hostile_ids(state: dict, radius_blocks: float, config: dict) -> set:
    """id враждебных сущностей ближе radius_blocks. В state["entities"]
    дистанция уже нормирована на entities.radius (js/entities.js)."""
    radius_norm = radius_blocks / config["entities"]["radius"]
    return {
        e.get("id") for e in present_entities(state)
        if e.get("type") == HOSTILE and e.get("dist", 1.0) <= radius_norm
    }


def _entity_in_front(state: dict) -> bool:
    """Есть ли сущность в конусе перед ботом: в передней полусфере
    (forward > 0) на дистанции ближнего боя. dist/forward нормированы
    на entities.radius (16 блоков по умолчанию)."""
    for e in present_entities(state):
        if e.get("forward", 0.0) > 0.0 and e.get("dist", 1.0) < 3.0 / 16.0:
            return True
    return False
