from dataclasses import dataclass
from pathlib import Path
from sched import scheduler
from typing import Type

import torch
import torch.nn as nn


@dataclass
class TrainConfig:
    job_path: Path
    batch_size: int
    epochs: int
    optim: Type[torch.optim.Optimizer]
    lr: float
    decay: bool
    scheduler: str | None
    automl: bool


@dataclass
class MLPConfig:
    window: int
    hidden_dims: list[int]
    act_fun: Type[nn.Module]


@dataclass
class LSTMConfig:
    layers: int
    dropout: float
    window: int
    hidden: int


@dataclass
class TSDecoderConfig:
    window: int
    nhead: int
    d_model: int
    layers: int
    dim_forward: int
    dropout: float
