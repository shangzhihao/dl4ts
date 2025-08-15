import torch

class TSDataset(torch.utils.data.Dataset):
    def __init__(self, data: list[float], window: int):
        self.data = data 
        self.window = window

    def __len__(self)-> int:
        return len(self.data) - self.window

    def __getitem__(self, idx: int)-> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.window]
        y = self.data[idx + self.window]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)