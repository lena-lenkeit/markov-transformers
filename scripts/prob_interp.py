import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim

n = 128
logits = torch.zeros(n, requires_grad=True)
target = torch.linspace(0.0, 1.0, n)

optimizer = optim.Adam([logits], lr=1.0)

for i in range(1024):
    loss = F.binary_cross_entropy_with_logits(logits, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.figure()
plt.plot(target, logits.detach())
plt.show()
