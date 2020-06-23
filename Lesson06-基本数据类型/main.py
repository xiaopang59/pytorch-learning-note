import torch
import numpy as np

a = torch.randn(2, 3)

print(a.type())
print(type(a))
print(isinstance(a, torch.FloatTensor))
print(isinstance(a, torch.cuda.FloatTensor))

# a = a.cuda()
# print(isinstance(a, torch.cuda.FloatTensor))

print(torch.tensor(1.))
print(torch.tensor(1.3))

# 维度为0的Tensor为标量，标量一般用在Loss这种地方
a = torch.tensor(2.2)
print(a.dim())
print(a.shape)
print(len(a.shape))
print(a.size())

# 生成dim=1的张量
print(torch.tensor([1.1]))
print(torch.tensor([1.1, 2.2])) # dim=2
print(torch.FloatTensor(1)) # dim=1, size=1，数据初始化randn
print(torch.FloatTensor(2)) # dim=2, size=2

# 通过numpy方法
data = np.ones(2)   # [1, 1]
print(data)
print(torch.from_numpy(data))


# Dim = 1，一般用于bias，或者Linear Input [28*28]=>[784]
a = torch.ones(2)
print(a.shape)

# Dim = 2，Linear Input batch 多张图片input [4, 784]
a = torch.randn(2, 3)
print(a)
print(a.size())
print(a.size(0))
print(a.size(1))
print(a.shape[1])


# Dim = 3   RNN Input Batch
a = torch.rand(2, 2, 3)
print(a)
print(a.shape)
print(a[0])
print(list(a.shape))


# Dim = 4   CNN:[b, c, h, w]=>[batch_size, channel, height, width]
a = torch.rand(2, 3, 28, 28)
print(a)
print(a.shape)


# Mixed
# numel是指tensor占用内存的数量
print(a.numel())    # 2*3*28*28
print(a.dim())
a = torch.tensor(1)
print(a.dim())