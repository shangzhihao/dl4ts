import pandas as pd
import torch.nn as nn
from config import MLPConfig


class MLP(nn.Module):
    def __init__(self, config: MLPConfig):
        super(MLP, self).__init__()
        window = config.window
        hidden_dims = config.hidden_dims
        act_fun = config.act_fun
        dims = [window, *hidden_dims, 1]
        self.layers = []
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                self.layers.append(act_fun())
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)

