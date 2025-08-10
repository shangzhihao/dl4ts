import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_window, hidden_dims):
        super(MLP, self).__init__()
        dims = [input_window, ]
        dims.extend(hidden_dims)
        dims.append(1)
        self.layers = []
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                self.layers.append(nn.ReLU())
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)

