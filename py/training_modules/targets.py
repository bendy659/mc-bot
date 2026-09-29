"""Общая геометрия и выбор цели-сущности для задачек looking и follow.

Сущности приходят из Node (js/entities.js) уже в локальной системе бота:
forward/right/up и dist нормированы на entities.radius, есть id, тип
(0 враждебная, 1 мирная, 2 игрок, 3 неживое: выпавший предмет — имя
"item", стрела, шар опыта...), имя и высота. Модуль
"цепляется" к сущности по id (current_target = {"entity_id": id}), а Node
каждый тик подставляет в state["target"] её свежую позицию (js/target.js),
плюс state["target"]["h"] — высоту, чтобы смотреть в лицо, а не в ноги.
"""

from __future__ import annotations

import math

HOSTILE, PASSIVE, PLAYER, ITEM = 0, 1, 2, 3
EYE_HEIGHT = 1.62  # высота глаз игрока/бота над ногами


def wrap_angle(angle: float) -> float:
    """Кратчайшая угловая разница — приводит угол к (-pi, pi]."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle <= -math.pi:
        angle += 2 * math.pi
    return angle


def aim_errors(self_state: dict, target: dict) -> tuple[float, float]:
    """(ошибка по горизонтали, ошибка по вертикали) в радианах: насколько
    взгляд бота отклоняется от точки, в которую надо смотреть. Точка — на
    85% высоты цели-сущности (лицо) или сама позиция, если цель — точка.

    Геометрия как в js/vision.js (конвенция mineflayer): взгляд
    x = -sin(yaw)cos(pitch), y = sin(pitch), z = -cos(yaw)cos(pitch);
    pitch > 0 — вверх, yaw растёт влево. Как в bot.lookAt:
    yaw = atan2(-dx, -dz), pitch = atan2(dy, горизонталь)."""
    aim_y = target["y"] + 0.85 * (target.get("h") or 0.0)
    dx = target["x"] - self_state["x"]
    dy = aim_y - (self_state["y"] + EYE_HEIGHT)
    dz = target["z"] - self_state["z"]
    target_yaw = math.atan2(-dx, -dz)
    target_pitch = math.atan2(dy, math.hypot(dx, dz))
    return (
        wrap_angle(target_yaw - self_state["yaw"]),
        wrap_angle(target_pitch - self_state["pitch"]),
    )


def percent(value: float) -> str:
    """Проценты для человека: мелкие — с десятыми (в начале обучения
    "0.4%" и "0%" — большая разница), крупные — целыми."""
    return f"{value:.1f}%" if value < 10 else f"{value:.0f}%"


def plural(count: int, one: str, few: str, many: str) -> str:
    """Число со словом в нужной форме: plural(2, "секунда", "секунды",
    "секунд") -> "2 секунды"; 1 и 21 — one, 2–4 и 22–24 — few, остальное
    (и 11–14) — many."""
    if count % 10 == 1 and count % 100 != 11:
        return f"{count} {one}"
    if count % 10 in (2, 3, 4) and count % 100 not in (12, 13, 14):
        return f"{count} {few}"
    return f"{count} {many}"


def times(count: int) -> str:
    """"1 раз", "3 раза", "12 раз", "22 раза" — для сводок человеку."""
    return plural(count, "раз", "раза", "раз")


def bot_name(index: int, config: dict) -> str:
    """Ник бота роя по номеру (3 -> AI_3) — как botName в js/bot.js."""
    username = config["bot"]["username"]
    if config["bot"].get("count", 1) <= 1:
        return username
    return f"{username}{config['bot'].get('separator', '_')}{index}"


def bot_index(name: str, config: dict) -> int | None:
    """Номер бота своего роя по нику (AI_3 -> 3), иначе None — значит, это
    человек или чужой бот."""
    username = config["bot"]["username"]
    if config["bot"].get("count", 1) <= 1:
        return 1 if name == username else None
    prefix = username + config["bot"].get("separator", "_")
    if name.startswith(prefix) and name[len(prefix):].isdigit():
        return int(name[len(prefix):])
    return None


def present_entities(state: dict) -> list[dict]:
    return [e for e in state.get("entities") or [] if e.get("present")]


def find_entity(state: dict, entity_id) -> dict | None:
    for entity in present_entities(state):
        if entity.get("id") == entity_id:
            return entity
    return None


def nearest(entities: list[dict]) -> dict | None:
    return min(entities, key=lambda e: e.get("dist", 1.0), default=None)
