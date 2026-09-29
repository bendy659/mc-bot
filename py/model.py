"""DQN-сеть: маленький CNN над сеткой зрения + MLP над скалярами,
с dueling-головой.

Сетка зрения — структурированные данные (соседние ячейки связаны), поэтому
CNN уместнее плоского MLP: свёртки ловят локальные паттерны вроде
"стена слева" или "ямы впереди".

Одна такая сеть — на каждый канал (ноги/голова/руки, см. py/protocol.py):
архитектура одинаковая, различается только число выходов.
"""

import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, vision_channels: int, res_y: int, res_x: int, scalar_dim: int, num_actions: int):
        super().__init__()

        # Ветка зрения. Вход (C, resY, resX). Ядро 3 "смотрит" на соседние
        # лучи, пулинг усредняет — на 8x8 это всего один MaxPool, иначе
        # сетка схлопнется в 1x1 слишком рано.
        self.conv = nn.Sequential(
            nn.Conv2d(vision_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                      # 8x8 -> 4x4
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        conv_out = 64 * (res_y // 2) * (res_x // 2)

        # Ветка скаляров: слух, углы, здоровье, цель, сущности + контекст
        # (активная задача и что делали остальные каналы на прошлом тике).
        self.scalar_net = nn.Sequential(
            nn.Linear(scalar_dim, 128),
            nn.ReLU(),
        )

        self.trunk = nn.Sequential(
            nn.Linear(conv_out + 128, 256),
            nn.ReLU(),
        )

        # Dueling (Wang et al. 2016): Q(s,a) = V(s) + A(s,a) - mean(A).
        # В большинстве состояний почти неважно, что именно делать
        # (идём по ровному полю) — V выучивается по всем переходам сразу,
        # а не по одному действию за раз, и обучение заметно стабильнее.
        self.value = nn.Linear(256, 1)
        self.advantage = nn.Linear(256, num_actions)
        self.num_actions = num_actions

    def forward(self, vision: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        v = self.conv(vision)
        s = self.scalar_net(scalars)
        features = self.trunk(torch.cat([v, s], dim=1))
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)
