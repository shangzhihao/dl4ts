import csv
import os
from MLP import train_mlp
from pathlib import Path

data_path = Path(__file__).parent / "data"
{'work_dir': '/tl4ds',
 'mlp_epochs': '10',
 'model': 'MLP',
 'mlp_lr': '0.001',
 'mlp_act_fun': 'gelu',
 'file': '/data/60869e38-3d3a-4932-9b94-d88a4b43c786/samples.csv',
 'mlp_window': '10',
 'mlp_batch': '32',
 'PWD': '/tl4ds',}

def get_container_id():
    return open("/etc/hostname").read().strip()

def get_all_envs():
    return dict(os.environ)

def train():
    envs = get_all_envs()
    job_id = envs["job_id"]
    job_path = data_path / job_id 
    if envs["model"] == "MLP":
        window = int(envs["mlp_window"])
        hidden_dims = list(map(int, envs["mlp_neurons"].split(",")))
        act_str = envs["mlp_act_fun"]
        batch_size = int(envs["mlp_batch"])
        epochs = int(envs["mlp_epochs"])
        lr = float(envs["mlp_lr"])
        train_mlp(job_path, window, hidden_dims, act_str, batch_size, epochs, lr)

if __name__ == "__main__":
    train()