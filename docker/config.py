from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn as nn
from typing import Type

@dataclass
class MLPConfig():
    window: int
    hidden_dims: list[int]
    act_fun: nn.Module

@dataclass
class TrainConfig():
    job_path: Path
    batch_size: int
    epochs: int
    optim: Type[torch.optim.Optimizer]
    lr: float
    automl: bool
