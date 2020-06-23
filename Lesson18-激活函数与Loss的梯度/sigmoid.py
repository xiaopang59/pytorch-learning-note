# sigmoid function
import torch
import torch.nn.functional as F

a = torch.linspace(-100, 100, 10)
print(a)
print(torch.sigmoid(a))
print(F.sigmoid(a))