from curses import window
from pathlib import Path
from typing import Type

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict


class TrainConfig(BaseModel):
    job_path: Path
    batch_size: int
    epochs: int
    optim: Type[torch.optim.Optimizer]
    lr: float
    decay: bool
    scheduler: str | None
    automl: bool

    model_config = ConfigDict(arbitrary_types_allowed=True)


class MLPConfig(BaseModel):
    window: int
    hidden_dims: list[int]
    act_fun: Type[nn.Module]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class LSTMConfig(BaseModel):
    layers: int
    dropout: float
    window: int
    hidden: int


class TCNConfig(BaseModel):
    channels: list[int]
    kernel_size: int
    dropout: float
    window: int


class TSDecoderConfig(BaseModel):
    window: int
    nhead: int
    d_model: int
    layers: int
    dim_forward: int
    dropout: float
