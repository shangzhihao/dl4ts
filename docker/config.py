from dataclasses import dataclass
from pathlib import Path
from sched import scheduler
import torch
import torch.nn as nn
from typing import Type

@dataclass
class MLPConfig():
    window: int
    hidden_dims: list[int]
    act_fun: Type[nn.Module]

@dataclass
class TrainConfig():
    job_path: Path
    batch_size: int
    epochs: int
    optim: Type[torch.optim.Optimizer]
    lr: float
    decay: bool
    scheduler: str | None
    automl: bool

from dataclasses import dataclass

@dataclass
class LSTMConfig:
    layers: int
    dropout: float
    input_window: int