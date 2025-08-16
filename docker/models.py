import pandas as pd
import torch.nn as nn
from config import MLPConfig, LSTMConfig


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

class LSTM_Model(nn.Module):
    def __init__(self, config: LSTMConfig):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=config.window,
            hidden_size=config.hidden,
            num_layers=config.layers,
            batch_first=True,
            dropout=config.dropout,
        )
        self.fc = nn.Linear(config.hidden, 1)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out)
