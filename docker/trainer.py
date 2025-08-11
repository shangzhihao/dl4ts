import os
import torch
import pandas as pd
from models import MLP

def get_container_id():
    return open("/etc/hostname").read().strip()

def get_all_envs():
    return dict(os.environ)

def train():
    print(get_all_envs())

if __name__ == "__main__":
    train()