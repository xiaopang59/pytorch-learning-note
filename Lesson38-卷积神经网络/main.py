import torch
import torch.nn as nn
import torch.nn.functional as F

# [input_channel, kernel_channel, 3, 3]
layer = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=0)
x = torch.rand(1, 1, 28, 28)
out = layer.forward(x)
print(out.size())

layer = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1)
out = layer.forward(x)
print(out.size())

layer = nn.Conv2d(1, 3, kernel_size=3, stride=2, padding=1)
out = layer.forward(x)
print(out.size())

# 推荐使用layer(x)
out = layer(x)  # __call__
print(out.size())

# print(layer.weight)
print(layer.weight.shape)
print(layer.bias.shape)




w = torch.rand(16, 3, 5, 5)
b = torch.rand(16)
x = torch.randn(1, 3, 28, 28)
out = F.conv2d(x, w, b, stride=1, padding=1)
print(out.size())

out = F.conv2d(x, w, b, stride=2, padding=2)
print(out.size())