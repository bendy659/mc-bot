"""Загрузка демонстрационных переходов в replay buffer (часть DQfD).

Записи геймплея (data/demonstrations.jsonl от js/record.js) превращаются
в полноценные переходы для DQN: награды для каждого перехода считает
ОБУЧАЮЩИЙ МОДУЛЬ — тот самый, с которым будет работать агент. Так демо
учат не только «какие действия делал человек», но и «сколько это стоит»
в терминах конкретной специализации (walking наградит за приближение,
gathering — за добычу).

Важно: модуль создаётся свежий (не тот, что живёт в AILoop), чтобы его
внутренние счётчики (_prev_distance и т.п.) не испортились.

Порядок вызовов module.on_tick / compute_reward повторяет живой цикл
(py/ai_loop.py: сначала on_tick нового состояния, потом награда за
переход в него), чтобы демо-награды совпадали с живыми на одинаковых
состояниях.

Наблюдения собираются в стек кадров в порядке тиков; разрыв
последовательности тиков = конец куска записи (done=1) + сброс стека
и состояния модуля, как смерть в живом цикле.

Форматы строк записи (оба поддерживаются):
  {"state": ..., "actions": {"legs": ..., "head": ..., "hands": ...}} — новый,
      по действию на канал (js/record.js сейчас пишет так);
  {"state": ..., "action": "walk_forward"} — старый, одно действие на тик:
      оно уходит своему каналу, остальным каналам — "ничего не делал".
"""

import json
from pathlib import Path

from config import CONFIG
from protocol import ACTION_CHANNEL, ACTION_INDEX_IN_CHANNEL, CHANNEL_NAMES, IDLE_ACTIONS
from state_encoder import FrameStacker, encode_state, unpack_vision
from training_modules import create_module, task_index

ROOT = Path(__file__).resolve().parent.parent
DEMO_PATH = ROOT / "data" / "demonstrations.jsonl"


def demo_actions(demo: dict) -> dict | None:
    """{канал: имя действия} из строки записи любого формата, None если
    в строке неизвестное действие (записи от старого пространства действий)."""
    if "actions" in demo:
        actions = dict(IDLE_ACTIONS)
        actions.update(demo["actions"])
    else:
        name = demo.get("action")
        if name not in ACTION_CHANNEL:
            return None
        actions = dict(IDLE_ACTIONS)
        actions[ACTION_CHANNEL[name]] = name

    for channel, name in actions.items():
        if ACTION_CHANNEL.get(name) != channel:
            return None
    return actions


def is_blind(state: dict) -> bool:
    """Запись сделана, пока зрение было сломано (до 2026-09-24 raycast
    читался неправильно и ВСЕ лучи были "небо на максимальной дальности").
    На таких кадрах учиться вредно: живые боты теперь видят мир, а демо
    учили бы сети на пустой картинке. Реальный кадр, где вообще ни один
    луч ни во что не попал, практически невозможен — под ногами есть земля."""
    cells = state.get("vision", {}).get("cells") or []
    return all(cells[i + 4] == 0 and cells[i + 3] >= 1.0 for i in range(0, len(cells) - 4, 5))


def other_resolution(state: dict, config: dict) -> bool:
    """Запись снята с другим разрешением сетки зрения (vision.resolution с
    тех пор поменяли) — в память опыта её не положить: кадры другой формы."""
    return list(state.get("vision", {}).get("resolution") or []) != list(config["vision"]["resolution"])


def action_indices(actions: dict) -> list[int]:
    """Индексы действий внутри каналов, в порядке CHANNEL_NAMES — формат
    столбца действий в replay buffer."""
    return [ACTION_INDEX_IN_CHANNEL[actions[channel]] for channel in CHANNEL_NAMES]


def load_demos_into(brain, module_name: str, config: dict = CONFIG) -> int:
    """Читает записи и пушит переходы в память мозга задачки (brain.buffer,
    py/dqn.py: TaskBrain) как демо. Возвращает число загруженных переходов."""
    if not DEMO_PATH.exists():
        return 0  # записей нет — нечего и говорить (они необязательны)

    module = create_module(module_name, config)
    task = task_index(module_name)
    gamma = config["modules"].get(module_name, {}).get("gamma", config["train"]["gamma"])
    stacker = FrameStacker(config)

    prev = None  # (stacked_vision, scalars, actions, state, tick)
    prev_actions = dict(IDLE_ACTIONS)  # контекст: что делали каналы на прошлом тике
    loaded = 0
    skipped = 0
    blind = 0

    with open(DEMO_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            demo = json.loads(line)
            state = unpack_vision(demo["state"])
            actions = demo_actions(demo)
            if actions is None:
                skipped += 1
                continue
            if is_blind(state) or other_resolution(state, config):
                blind += 1
                continue

            tick = state["tick"]
            gap = prev is not None and tick != prev[4] + 1

            if gap:
                # Разрыв записи — закрываем прошлый кусок терминальным
                # переходом (next — плейсхолдер той же формы, как в живом
                # цикле при смерти: done=1 обнуляет бутстрап).
                rewards = module.compute_reward(prev[3], state, prev[2])
                brain.buffer.push_demo(
                    prev[0], prev[1], action_indices(prev[2]),
                    [rewards[c] for c in CHANNEL_NAMES], prev[0], prev[1], True, gamma,
                )
                loaded += 1
                stacker.frames = []
                module.reset(state)
                prev_actions = dict(IDLE_ACTIONS)

            vision, scalars = encode_state(state, config, task, prev_actions)
            module.on_tick(state)
            stacked = stacker.push(vision)

            if prev is not None and not gap:
                rewards = module.compute_reward(prev[3], state, prev[2])
                brain.buffer.push_demo(
                    prev[0], prev[1], action_indices(prev[2]),
                    [rewards[c] for c in CHANNEL_NAMES], stacked, scalars, False, gamma,
                )
                loaded += 1

            prev = (stacked, scalars, actions, state, tick)
            prev_actions = actions

    if skipped:
        print(f"[demo] Пропущено строк с неизвестными действиями: {skipped}")
    if blind:
        print(f"[demo] Пропущено {blind} записей со сломанным (слепым) зрением или снятых с другим "
              "разрешением сетки зрения — перезапиши геймплей: npm run record")
    print(f"[demo] Загружено демо-переходов: {loaded} (задача: {module_name})")
    return loaded
