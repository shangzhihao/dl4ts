import pandas as pd
import torch
import torch.nn as nn
from tsdata import TSDataset
import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
import os
import numpy as np

str2act = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
}


class MLP(nn.Module):
    def __init__(self, window, hidden_dims, act_str):
        super(MLP, self).__init__()
        dims = [
            window,
        ]
        dims.extend(hidden_dims)
        dims.append(1)
        act_fun = str2act.get(act_str, nn.GELU)
        self.layers = []
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                self.layers.append(act_fun())
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)


def train_mlp(job_path, window, hidden_dims, act_str, batch_size, epochs, lr):
    # Load data from CSV
    df = pd.read_csv(job_path / "samples.csv")
    train_list = df.iloc[:, 0].dropna().tolist()
    val_list = df.iloc[:, 1].dropna().tolist()

    # Create datasets
    train_dataset = TSDataset(train_list, window)
    val_dataset = TSDataset(val_list, window)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    # Create model
    model = MLP(window, hidden_dims, act_str)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # TODO: do not use hardcoded path
    # Set up MLflow
    mlflow_path = job_path / "mlflow_runs"
    os.makedirs(mlflow_path, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{mlflow_path.resolve()}")
    mlflow.set_experiment("dl4ts")
    with mlflow.start_run():
        # log parameters
        mlflow.log_param("window", window)
        mlflow.log_param("hidden_dims", hidden_dims)
        mlflow.log_param("act_fun", act_str)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", lr)
        # Training loop
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
            avg_loss = total_loss / len(train_loader.dataset)

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    pred = model(xb)
                    loss = criterion(pred, yb.unsqueeze(1))
                    val_loss += loss.item() * xb.size(0)
            avg_val_loss = val_loss / len(val_loader.dataset)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("progress", epoch, step=epoch)
            print(
                f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
            )

        # Log model
        sig = ModelSignature(
            inputs=Schema([TensorSpec(np.dtype("float32"), (-1, window))]),
            outputs=Schema([TensorSpec(np.dtype("float32"), (-1, 1))]),
        )
        input_np = np.random.rand(1, window)
        mlflow.pytorch.log_model(
            pytorch_model=model, name="model", input_example=input_np, signature=sig
        )
        # Save trained model in job_path
        torch.save(model.state_dict(), str(job_path / "mlp_model_state.pt"))
