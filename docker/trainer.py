import os
from pathlib import Path
from dataclasses import asdict
import sched
from torch.utils.data import DataLoader
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
from config import TrainConfig, MLPConfig
from models import MLP
from tsdata import TSDataset
import logging

logger = logging.getLogger(__name__)


str2act = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
}
str2opt = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "rmsprop": torch.optim.RMSprop,
}

data_path = Path(__file__).parent / "data"
envs = dict(os.environ)


def get_container_id() -> str:
    return open("/etc/hostname").read().strip()


def get_dataloader(
    job_path: Path, window: int, batch_size: int
) -> tuple[DataLoader, DataLoader]:
    df = pd.read_csv(job_path / envs["sample_file"])
    train_list = df.iloc[:, 0].dropna().tolist()
    val_list = df.iloc[:, 1].dropna().tolist()

    # Create datasets
    train_dataset = TSDataset(train_list, window)
    val_dataset = TSDataset(val_list, window)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def get_train_conf()->TrainConfig:
    job_id = envs["job_id"]
    job_path = data_path / job_id
    batch_size = int(envs["batch"])
    epochs = int(envs["epochs"])
    lr = float(envs["lr"])
    optim = str2opt.get(envs["optim"], torch.optim.Adam)
    automl = envs.get("auto", "True").lower() == "true"
    decay = envs.get("decay", "True").lower() == "true"
    scheduler = envs.get("scheduler", "none")
    train_params = {
        "job_path": job_path,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "optim": optim,
        "automl": automl,
        "scheduler": scheduler if scheduler != "none" else None,
        "decay": decay
    }
    return TrainConfig(**train_params) # type: ignore


def get_mlp_conf()-> MLPConfig:
    window = int(envs["mlp_window"])
    hidden_dims = list(map(int, envs["mlp_neurons"].split(",")))
    act_str = envs["mlp_act_fun"]
    act_fun = str2act.get(act_str, nn.GELU)
    model_params = {
        "window": window,
        "hidden_dims": hidden_dims,
        "act_fun": act_fun,
    }
    return MLPConfig(**model_params) # type: ignore


def train():
    model = None
    train_conf = get_train_conf()
    criterion = nn.MSELoss()
    model_conf = None
    if envs["model"] == "MLP":
        model_conf = get_mlp_conf()
        model = MLP(model_conf)
    else:
        raise ValueError(f"Unsupported model type: {envs['model']}")

    window = model_conf.window
    train_loader, val_loader = get_dataloader(
        train_conf.job_path, window, train_conf.batch_size
    )
    weight_decay = 1e-5 if train_conf.decay else 0
    optimizer = train_conf.optim(
        model.parameters(), lr=train_conf.lr, weight_decay=weight_decay) # type: ignore
    job_path = train_conf.job_path
    mlflow_path = job_path / envs["mlflow_dir"]
    os.makedirs(mlflow_path, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{mlflow_path.resolve()}")
    mlflow.set_experiment(envs["mlflow_exp_name"])
    with mlflow.start_run():
        mlflow.log_params(asdict(model_conf))
        mlflow.log_params(asdict(train_conf))
        epochs = train_conf.epochs
        len_train = len(train_loader.dataset)  # type: ignore
        len_val = len(val_loader.dataset)  # type: ignore
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb.unsqueeze(1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)
            avg_loss = total_loss / len_train

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    pred = model(xb)
                    loss = criterion(pred, yb.unsqueeze(1))
                    val_loss += loss.item() * xb.size(0)
            avg_val_loss = val_loss / len_val
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("progress", epoch, step=epoch)
            logger.info(
                f"Epoch {epoch+1}/{epochs},Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
            )
        sig = ModelSignature(
            inputs=Schema([TensorSpec(np.dtype("float32"), (-1, window))]),
            outputs=Schema([TensorSpec(np.dtype("float32"), (-1, 1))]),
        )
        input_np = np.random.rand(1, window)
        mlflow.pytorch.log_model( # type: ignore
            pytorch_model=model, name="model", input_example=input_np, signature=sig
        )
        # Save trained model in job_path
        torch.save(model.state_dict(), str(job_path / envs["model_state_file"]))


if __name__ == "__main__":
    train()
