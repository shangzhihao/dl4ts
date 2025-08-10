
import os
import torch
import pandas as pd
from models import MLP

def get_container_id():
    return open("/etc/hostname").read().strip()

def train():
    # Get hyperparameters from environment variables
    input_window = int(os.environ.get("INPUT_WINDOW", 10))
    hidden_layers = int(os.environ.get("HIDDEN_LAYERS", 2))
    neurons = int(os.environ.get("NEURONS", 32))
    batch_size = int(os.environ.get("BATCH_SIZE", 32))
    epochs = int(os.environ.get("EPOCHS", 10))
    learning_rate = float(os.environ.get("LEARNING_RATE", 0.001))
    data_path = os.environ.get("DATA_PATH", "samples.csv")

    # Prepare hidden_dims list
    hidden_dims = [neurons] * hidden_layers

    # Load data
    df = pd.read_csv(data_path)
    X = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
    y = torch.tensor(df.iloc[:, -1].values, dtype=torch.float32).unsqueeze(1)

    # Create model
    model = MLP(input_window, hidden_dims)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # Save model
    cid = get_container_id()
    torch.save(model.state_dict(), f"{cid}/mlp_model.pt")


if __name__ == "__main__":
    train()