import torch
import torch.nn as nn
from config import MLPConfig, LSTMConfig, TSDecoderConfig, TCNConfig
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


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        if self.downsample is not None:
            nn.init.xavier_uniform_(self.downsample.weight)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    def __init__(self, conf: TCNConfig):
                 # num_channels=None, kernel_size=2, dropout=0.1):
        super(TCN, self).__init__()
        channels = conf.channels
        kernel_size = conf.kernel_size
        dropout = conf.dropout
        if channels is None:
            channels = [32, 64, 32]

        layers = []
        num_levels = len(channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = 1 if i == 0 else channels[i-1]
            out_channels = channels[i]
            padding = (kernel_size - 1) * dilation

            layers.append(
                TemporalBlock(in_channels, out_channels, kernel_size, dilation, padding, dropout)
            )

        self.network = nn.Sequential(*layers)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[-1], 1)

    def forward(self, x):
        x = x.unsqueeze(1) 
        y = self.network(x)

        y = self.global_avg_pool(y).squeeze(-1)
        output = self.fc(y)
        return output

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
