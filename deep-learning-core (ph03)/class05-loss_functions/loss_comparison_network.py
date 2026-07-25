import random
import torch
import torch.nn as nn

def make_circle_data(n=200, seed=42):
    random.seed(seed)
    xs, ys = [], []
    for _ in range(n):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        label = 1.0 if x * x + y * y < 1.5 else 0.0
        xs.append([x, y])
        ys.append([label])
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)


class LossComparisonNetwork(nn.Module):
    def __init__(self, hidden_size=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def train(loss_type="bce", epochs=1000, lr=0.1):
    x, y = make_circle_data()
    model = LossComparisonNetwork()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.BCELoss() if loss_type == "bce" else nn.MSELoss()

    for epoch in range(epochs):
        pred = model(x)
        loss = criterion(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            acc = ((pred >= 0.5) == (y >= 0.5)).float().mean() * 100
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"    Epoch {epoch:3d}: loss={loss.item():.4f}, accuracy={acc:.1f}%")

train(loss_type="mse")