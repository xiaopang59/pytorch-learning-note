# loss.backward
import torch
import torch.nn.functional as F

x = torch.ones(1)   # 默认requires_grad = False
w = torch.full([1], 2)
b = torch.ones(1)
# print(x)
# print(w)
mse = F.mse_loss(torch.ones(1), x*w+b)
print("mse: ", mse)
# Pytorch中， tensor需要定义成require_grad的类型才能求梯度
# torch.autograd.grad(mse, [w]) # RuntimeError
print(w.requires_grad_())
# 需要重新建图
# torch.autograd.grad(mse, [w]) # RuntimeError
b.requires_grad_()
mse = F.mse_loss(torch.ones(1), x*w+b)
mse.backward()
print(w.grad)
print(b.grad)
# w1.grad
# w2.grad