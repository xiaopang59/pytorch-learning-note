# softmax
import torch
import torch.nn.functional as F

a = torch.rand(3)
a.requires_grad_()
print(a)
p = F.softmax(a, dim=0)
print(p)
# 注意计算导数时，loss只能是一个值的，不能是有多个分量的
# 但是可以对多个参数求偏导
# 因此，只能分别对p[0]、p[1]、p[2]对应的loss求导
# 计算p对a0、a1、a2的导数，a0、a2是i != j的情况求导，得到的是负值
print(torch.autograd.grad(p[1], [a], retain_graph=True))
print(torch.autograd.grad(p[2], [a]))