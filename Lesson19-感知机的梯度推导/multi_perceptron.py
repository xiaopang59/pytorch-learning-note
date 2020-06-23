# 多层感知机的梯度推导
import torch
import torch.nn.functional as F

x = torch.randn(1, 10)
w = torch.randn(2, 10, requires_grad=True)
o = torch.sigmoid(x@w.t())  # [1, 10][2, 10].T => [1, 2]
print("o:{0}, shape:{1}".format(o, o.shape))
# loss = F.mse_loss(torch.ones(1, 1), o)  # Boardcasting
loss = F.mse_loss(torch.ones(1, 2), o)
print("loss:{0}, shape:{1}".format(loss, loss.shape))
loss.backward()
print(w.grad)