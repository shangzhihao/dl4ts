from dataclasses import dataclass
from pathlib import Path
import torch.nn as nn


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
    lr: float
    automl: bool
