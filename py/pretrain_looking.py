"""Предобучение задачки looking ("следить взглядом") в симуляции.

Запуск: python py/pretrain_looking.py   (минут 5 на видеокарте)

Зачем. Мозг looking видит только геометрию цели (зрение, сущности и слух
ему выключены — modules.looking.inputs), а его действия — это просто
повороты взгляда на известные углы (protocol.ACTION_ROTATION). Такой мир
легко воспроизвести без Minecraft: цель в случайном месте — близко и
далеко, выше и ниже, сзади, стоит или бежит. За несколько минут симуляция
даёт сотни тысяч разнообразных ситуаций. Вживую, где боты стоят на месте,
а цель одна и та же, такого разнообразия не набрать и за часы: проверено —
мозг, обученный вживую, доворачивал по горизонтали, но не умел смотреть
выше/ниже, потому что все боты стояли на одной высоте с целью.

Награду считает тот же модуль looking, учит тот же мозг (py/dqn.py), n-step
тот же — сохраняется в data/brains/looking/, и живое обучение продолжает с
этого места (случайных действий к концу предобучения остаётся ~5%). Если
мозг looking уже есть — предобучение его дообучит, а не начнёт заново.
"""

import argparse
import math
import random
import sys
import time
from pathlib import Path

import torch

from config import CONFIG
from protocol import ACTION_ROTATION, CHANNEL_NAMES, IDLE_ACTIONS
from state_encoder import FrameStacker, encode_state, scalar_dim, vision_channels
from dqn import TaskBrain
from ai_loop import NStepAccumulator, brain_dir, task_gamma
from demo_loader import action_indices
from training_modules import create_module, task_index

TASK = "looking"
EPISODE_TICKS = 200       # ~30 секунд игры: потом новая ситуация
PLAYER_HEIGHT = 1.8
WALK_PER_TICK = 0.65      # ходьба игрока за тик (150 мс), блоков


class SimBot:
    """Один симулированный бот: стоит на месте, крутит взглядом; цель —
    где-то вокруг, иногда бегает."""

    def __init__(self, config: dict):
        self.config = config
        self.module = create_module(TASK, config)
        self.stacker = FrameStacker(config)
        self.nstep = NStepAccumulator(config["train"].get("n_step", 3), task_gamma(TASK, config))
        res_x, res_y = config["vision"]["resolution"]
        # Зрение мозгу looking выключено, но форма входа должна быть настоящей:
        # "небо во все стороны".
        self.cells = [0.5, 0.7, 1.0, 1.0, 0] * (res_x * res_y)
        self.resolution = [res_x, res_y]
        self.reset()

    def reset(self) -> None:
        self.yaw = random.uniform(-math.pi, math.pi)
        self.pitch = random.uniform(-0.5, 0.5)
        self.dyaw = 0.0
        self.dpitch = 0.0
        self.ticks = 0
        # Цель: 10% эпизодов — цели нет (надо научиться хотя бы искать);
        # иначе — на расстоянии 2..30 блоков, на высоте от -6 до +6 блоков
        # (рельеф), любого роста от курицы до игрока.
        if random.random() < 0.1:
            self.target = None
        else:
            distance = random.choice([random.uniform(2, 8), random.uniform(8, 30)])
            angle = random.uniform(-math.pi, math.pi)
            self.target = {
                "x": math.sin(angle) * distance, "z": math.cos(angle) * distance,
                "y": 64.0 + random.uniform(-6, 6), "h": random.choice([PLAYER_HEIGHT, PLAYER_HEIGHT, 0.7, 1.4]),
            }
        # Половина целей стоит, половина ходит, иногда меняя направление.
        self.speed = 0.0 if random.random() < 0.5 else random.uniform(0.1, WALK_PER_TICK)
        self.heading = random.uniform(-math.pi, math.pi)
        self.module.reset(None)
        self.stacker.frames = []
        self.prev = None
        self.prev_state = None
        self.prev_actions = dict(IDLE_ACTIONS)

    def state(self) -> dict:
        return {
            "tick": self.ticks, "dead": False, "center_block": None, "entities": [], "hearing": [],
            "vision": {"resolution": self.resolution, "cells": self.cells},
            "self": {"x": 0.0, "y": 64.0, "z": 0.0, "yaw": self.yaw, "pitch": self.pitch,
                     "health": 20, "food": 20, "on_ground": True,
                     "move_forward": 0.0, "move_right": 0.0, "move_up": 0.0,
                     "dyaw": self.dyaw, "dpitch": self.dpitch},
            "target": dict(self.target) if self.target else None,
            "target_is_human": True,
        }

    def apply(self, actions: dict) -> None:
        """Действия ровно как в js/actions.js: yaw растёт влево, pitch > 0 —
        вверх, наклон ограничен ±90°."""
        old_yaw, old_pitch = self.yaw, self.pitch
        for channel in ("legs", "head"):
            dyaw, dpitch = ACTION_ROTATION.get(actions[channel], (0.0, 0.0))
            self.yaw += dyaw
            self.pitch = max(-math.pi / 2, min(math.pi / 2, self.pitch + dpitch))
        self.dyaw = self.yaw - old_yaw
        self.dpitch = self.pitch - old_pitch

    def move_target(self) -> None:
        if self.target is None or self.speed == 0.0:
            return
        if random.random() < 0.03:
            self.heading += random.uniform(-2, 2)
        self.target["x"] += math.sin(self.heading) * self.speed
        self.target["z"] += math.cos(self.heading) * self.speed
        # Не даём уйти слишком далеко или наступить боту на голову.
        distance = math.hypot(self.target["x"], self.target["z"])
        if distance > 30 or distance < 1.5:
            self.heading += math.pi


def run(ticks_total: int, bots: int, learn_per_step: float, greedy_eval: bool = False, brain=None):
    config = CONFIG
    sims = [SimBot(config) for _ in range(bots)]
    centered = with_target = 0
    credit = 0.0
    for _ in range(ticks_total):
        for sim in sims:
            state = sim.state()
            sim.module.on_tick(state)
            vision, scalars = encode_state(state, config, task_index(TASK), sim.prev_actions)
            stacked = sim.stacker.push(vision)
            if sim.prev is not None:
                before = sim.module.events.get("centered", 0)
                rewards = sim.module.compute_reward(sim.prev_state, state, sim.prev_actions)
                if not greedy_eval:
                    sim.nstep.append(sim.prev[0], sim.prev[1], action_indices(sim.prev_actions),
                                     [rewards[c] for c in CHANNEL_NAMES])
                    sim.nstep.emit_ready(brain, stacked, scalars)
                if state["target"] is not None:
                    with_target += 1
                    centered += sim.module.events.get("centered", 0) - before
            actions = brain.select_actions(stacked, scalars, greedy=greedy_eval)
            sim.prev = (stacked, scalars)
            sim.prev_state = state
            sim.prev_actions = actions
            sim.apply(actions)
            sim.move_target()
            sim.ticks += 1
            if sim.ticks >= EPISODE_TICKS:
                if not greedy_eval:
                    sim.nstep.flush(brain, stacked, scalars, done=False)
                sim.reset()
        if not greedy_eval:
            credit += bots * learn_per_step
            while credit >= 1.0:
                brain.learn()
                credit -= 1.0
    return 100.0 * centered / max(with_target, 1)


def main():
    parser = argparse.ArgumentParser(description="Предобучение looking в симуляции")
    parser.add_argument("--steps", type=int, default=None,
                        help="шагов симуляции (по умолчанию — как epsilon_decay_steps looking)")
    parser.add_argument("--bots", type=int, default=32, help="параллельных симулированных ботов")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res_x, res_y = CONFIG["vision"]["resolution"]
    brain = TaskBrain(TASK, CONFIG, (vision_channels(CONFIG), res_y, res_x), scalar_dim(CONFIG), device)
    directory = brain_dir(TASK)
    for channel in brain.learners:
        path = directory / f"{channel}.pt"
        if path.exists():
            brain.load_channel(channel, path)
    print(f"[sim] Мозг looking: {'дообучаю' if brain.steps else 'с нуля'} ({brain.steps} шагов), устройство {device}")

    total = args.steps or brain.epsilon_decay_steps
    ticks = max(1, total // args.bots)
    started = time.time()
    chunks = 10
    for chunk in range(chunks):
        score = run(ticks // chunks, args.bots, learn_per_step=0.25, brain=brain)
        print(f"[sim] {100 * (chunk + 1) // chunks:3d}%: цель в прицеле {score:4.1f}% времени "
              f"(случайных действий {brain.epsilon():.0%}), {time.time() - started:.0f} с")
        brain.save(directory)

    print("[sim] Проверка без случайных действий на новых ситуациях...")
    score = run(300, args.bots, learn_per_step=0.0, greedy_eval=True, brain=brain)
    print(f"[sim] Готово: цель в прицеле {score:.1f}% времени. Мозг сохранён в {directory}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
