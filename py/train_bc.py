"""Офлайн-тренировка behavior cloning по записи геймплея.

Запуск: python py/train_bc.py [--task walking]

Читает data/demonstrations.jsonl (строки записи из js/record.js, форматы —
см. demo_loader.py), кодирует состояния тем же энкодером, что и DQN
(включая стек кадров и контекст), и тренирует те же сети (model.py) как
классификаторы через cross-entropy — по сети на канал (ноги/голова/руки),
у каждой свои метки.

Готовые веса падают в data/bc_<канал>.pt — их ai_loop.py подхватывает как
warm start для тех каналов, у которых ещё нет своего чекпоинта DQN.

--task — какую задачу подставить в контекст (one-hot задачи на входе
сетей). Запись учит "как двигаться", а не конкретной задаче, поэтому
по умолчанию walking — самая близкая к обычной игре.

Наблюдения собираются в стек кадров в порядке тиков, поэтому записи должны
быть непрерывными: при разрыве (смерть, пауза записи) стек сбрасывается.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from config import CONFIG
from demo_loader import DEMO_PATH, action_indices, demo_actions, is_blind, other_resolution
from protocol import CHANNELS, CHANNEL_NAMES, IDLE_ACTIONS
from state_encoder import FrameStacker, encode_state, expand_vision, scalar_dim, vision_channels
from model import DQN
from training_modules import DEFAULT_MODULE, task_index

ROOT = Path(__file__).resolve().parent.parent


def bc_path(channel: str) -> Path:
    return ROOT / "data" / f"bc_{channel}.pt"

bc_cfg = CONFIG.get("bc", {})
EPOCHS = bc_cfg.get("epochs", 60)
LR = bc_cfg.get("lr", 0.0003)
BATCH_SIZE = bc_cfg.get("batch_size", 256)
VAL_FRACTION = 0.1
PATIENCE = 7  # сколько эпох ждать улучшения val-точности до остановки
# Раньше этого числа шагов оптимизации не останавливаемся и учим дальше,
# даже если эпох уже больше EPOCHS: на маленькой записи (пара тысяч
# примеров) в эпохе всего несколько шагов, и "7 эпох без роста" наступали
# ещё до того, как сеть вообще начинала учиться (на синтетике — 70% вместо 100%).
MIN_STEPS = 500


def load_demonstrations(config: dict, task: int):
    """Читает jsonl и собирает (stacked_vision, scalars, labels) в порядке
    тиков. labels — тензор (N, число каналов): индекс действия каждого канала."""
    stacker = FrameStacker(config)
    visions, scalarss, labels = [], [], []
    prev_tick = None
    prev_actions = dict(IDLE_ACTIONS)
    blind = 0

    with open(DEMO_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            demo = json.loads(line)
            actions = demo_actions(demo)
            if actions is None:
                continue  # действие из старого пространства действий
            if is_blind(demo["state"]) or other_resolution(demo["state"], config):
                blind += 1
                continue  # сломанное зрение или другое разрешение сетки, см. demo_loader
            tick = demo["state"]["tick"]

            # Разрыв последовательности (пауза записи, телепорт) — сброс стека.
            if prev_tick is None or tick != prev_tick + 1:
                stacker.frames = []
                prev_actions = dict(IDLE_ACTIONS)
            prev_tick = tick

            try:
                vision, scalars = encode_state(demo["state"], config, task, prev_actions)
            except ValueError as err:
                print(
                    f"[bc] Записи несовместимы с текущим наблюдением ({err}).\n"
                    "[bc] Формат состояния менялся (классы блоков, сущности?) — "
                    "перезапиши геймплей: npm run record."
                )
                return None
            stacked = stacker.push(vision)
            visions.append(stacked)
            scalarss.append(scalars)
            labels.append(action_indices(actions))
            prev_actions = actions

    if blind:
        print(f"[bc] Пропущено {blind} записей со сломанным (слепым) зрением или другим разрешением сетки — "
              "перезапиши: npm run record")
    if not labels:
        print("[bc] Годных записей нет — учить не на чем.")
        return None
    print(f"[bc] Загружено примеров: {len(labels)}")
    return (
        torch.stack(visions),
        torch.stack(scalarss),
        torch.tensor(labels, dtype=torch.int64),
    )


def accuracy(model, visions, scalarss, labels, device) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(labels), BATCH_SIZE):
            v = expand_vision(visions[i:i + BATCH_SIZE].to(device), CONFIG)
            s = scalarss[i:i + BATCH_SIZE].to(device)
            y = labels[i:i + BATCH_SIZE].to(device)
            correct += (model(v, s).argmax(dim=1) == y).sum().item()
    model.train()
    return correct / len(labels)


def train_channel(channel: str, channel_index: int, train_data, val_data, config: dict, device) -> float:
    """Классификатор одного канала. Возвращает лучшую val-точность."""
    res_x, res_y = config["vision"]["resolution"]
    model = DQN(
        vision_channels(config), res_y, res_x,
        scalar_dim(config), len(CHANNELS[channel]),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    v, s, y_all = train_data
    y = y_all[:, channel_index]
    val = (val_data[0], val_data[1], val_data[2][:, channel_index])

    # Какая доля val-примеров — просто "ничего не делал". Точность ниже этой
    # цифры значит, что сеть хуже тупого "всегда стой" — полезно видеть,
    # особенно для головы и рук, у которых idle — подавляющее большинство.
    idle_share = (val[2] == 0).float().mean().item()
    print(f"[bc] --- канал {channel}: {len(CHANNELS[channel])} действий, доля idle в val {idle_share:.1%}")

    best_val = 0.0
    epochs_since_best = 0

    steps = 0
    epoch = 0
    while epoch < EPOCHS or steps < MIN_STEPS:
        epoch += 1
        order = torch.randperm(len(y))
        epoch_loss = 0.0
        batches = 0
        for i in range(0, len(y), BATCH_SIZE):
            idx = order[i:i + BATCH_SIZE]
            logits = model(expand_vision(v[idx].to(device), config), s[idx].to(device))
            loss = loss_fn(logits, y[idx].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1
            steps += 1

        train_acc = accuracy(model, v, s, y, device)
        val_acc = accuracy(model, *val, device)
        print(
            f"[bc] {channel} эпоха {epoch:3d}: loss {epoch_loss / batches:.4f}, "
            f"train acc {train_acc:.1%}, val acc {val_acc:.1%}"
        )

        if val_acc > best_val:
            best_val = val_acc
            epochs_since_best = 0
            torch.save({"policy_net": model.state_dict()}, bc_path(channel))
        else:
            epochs_since_best += 1
            if epochs_since_best >= PATIENCE and steps >= MIN_STEPS:
                print(f"[bc] {channel}: val-точность не растёт {PATIENCE} эпох — ранняя остановка.")
                break

    return best_val


def main():
    parser = argparse.ArgumentParser(description="Behavior cloning по записям геймплея")
    parser.add_argument("--task", default=DEFAULT_MODULE, help="Задача в контексте сетей (по умолчанию walking)")
    args = parser.parse_args()

    if not DEMO_PATH.exists():
        print(f"[bc] Нет файла записей: {DEMO_PATH}. Сначала запиши геймплей: npm run record")
        return 1

    config = CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[bc] Устройство: {device}")

    loaded = load_demonstrations(config, task_index(args.task))
    if loaded is None:
        return 1  # причина уже напечатана внутри load_demonstrations
    visions, scalarss, labels = loaded

    # Перемешивание разбивает временной ряд — но наблюдения уже собраны
    # в стеки, так что для классификации это безопасно.
    perm = torch.randperm(len(labels))
    visions, scalarss, labels = visions[perm], scalarss[perm], labels[perm]
    val_size = max(1, int(len(labels) * VAL_FRACTION))
    train_data = (visions[val_size:], scalarss[val_size:], labels[val_size:])
    val_data = (visions[:val_size], scalarss[:val_size], labels[:val_size])

    results = {}
    for index, channel in enumerate(CHANNEL_NAMES):
        results[channel] = train_channel(channel, index, train_data, val_data, config, device)

    summary = ", ".join(f"{channel} {acc:.1%}" for channel, acc in results.items())
    print(f"[bc] Готово. Лучшая val-точность: {summary}. Веса: data/bc_<канал>.pt")
    print("[bc] ~60% для ног — хороший результат, дальше модель доучит DQN наградами.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
