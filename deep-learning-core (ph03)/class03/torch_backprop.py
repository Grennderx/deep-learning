import torch
import torch.nn as nn

X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype = torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype = torch.float32)

model = nn.Sequential(
    nn.Linear(2, 4),
    nn.Sigmoid(),
    nn.Linear(4, 1),
    nn.Sigmoid(),
)

optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
criterion = nn.MSELoss()

for epoch in range(10000):
    pred = model(X)
    loss = criterion(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

with torch.no_grad():
    for i in range(len(X)):
        pred = model(X[i])
        print(f" {X[i].tolist()} -> {pred.item():.4f}")