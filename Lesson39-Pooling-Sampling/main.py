import torch
import torch.nn as nn
import torch.nn.functional as F

layer = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
x = torch.rand(1, 1, 28, 28)
out = layer(x)
x = out
print(x.size())



# downsampling
layer = nn.MaxPool2d(2, stride=2)
out = layer(x)
print(out.size())

out = F.avg_pool2d(x, 2, stride=2)
print(out.size())




# upsampling
x = out
out = F.interpolate(x, scale_factor=2, mode="nearest")
print(out.shape)
out = F.interpolate(x, scale_factor=3, mode="nearest")
print(out.shape)



# Relu
print(x.shape)
layer = nn.ReLU(inplace=True)
out = layer(x)
print(out.shape)
out = F.relu(x)
print(out.shape)