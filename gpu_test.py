import torch
import torch.nn as nn

device = "cuda"

print("GPU:", torch.cuda.get_device_name(0))
print("CUDA:", torch.version.cuda)

model = nn.Sequential(
    nn.Linear(1024, 2048),
    nn.ReLU(),
    nn.Linear(2048, 1024)
).to(device)

x = torch.randn(64, 1024, device=device)
y = torch.randn(64, 1024, device=device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("VRAM before:", round(torch.cuda.memory_allocated() / 1024**2, 1), "MB")

for step in range(100):
    optimizer.zero_grad()

    output = model(x)
    loss = ((output - y) ** 2).mean()

    loss.backward()
    optimizer.step()

torch.cuda.synchronize()

print("Loss:", round(loss.item(), 6))
print("VRAM after:", round(torch.cuda.memory_allocated() / 1024**2, 1), "MB")
print("GPU TEST: PASSED")