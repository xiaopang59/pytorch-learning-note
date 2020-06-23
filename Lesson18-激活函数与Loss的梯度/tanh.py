# tanh function
import torch
import torch.nn.functional as F

a = torch.linspace(-1, 1, 10)
print(a)
print(torch.tanh(a))
print(F.tanh(a))