# Relu function
import torch
import torch.nn.functional as F

a = torch.linspace(-1, 1, 10)
print(a)
print(torch.relu(a))
print(F.relu(a))