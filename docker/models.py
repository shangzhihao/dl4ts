import torch
import torch.nn as nn
from config import MLPConfig, LSTMConfig, TSDecoderConfig
import math

class MLP(nn.Module):
    def __init__(self, config: MLPConfig):
        super().__init__()
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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        L = x.size(1)
        return x + self.pe[:L].unsqueeze(0) # type: ignore

def generate_causal_mask(L, device=None):
    return torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()


class TSDecoder(nn.Module):
    def __init__(self, config: TSDecoderConfig):
        super().__init__()
        d_model = config.d_model
        self.input_proj = nn.Linear(1, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_forward,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=config.layers)

        self.memory_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.out_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        B, n = x.shape
        device = x.device

        tgt = self.input_proj(x.unsqueeze(-1))
        tgt = self.pos_enc(tgt)

        memory = self.memory_token.expand(B, 1, -1)

        # causal mask
        mask = torch.triu(torch.ones(n, n, device=device), diagonal=1).bool()

        decoded = self.decoder(tgt=tgt, memory=memory, tgt_mask=mask)  # (B, n, d_model)
        y = self.out_proj(decoded[:, -1, :])
        return y
