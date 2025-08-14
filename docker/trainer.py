import os
from MLP import train_mlp
from pathlib import Path
data_path = Path(__file__).parent / "data"

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
        automl = envs.get("mlp_auto", "True").lower() == "true"
        train_mlp(job_path, window, hidden_dims, act_str, batch_size, epochs, lr)

if __name__ == "__main__":
    train()