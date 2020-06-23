"""
Operation
    View/reshape
    Squeeze/unsqueeze
    Transpose/t/permute
    Expand/repeat
"""

import torch
import numpy as np

# View reshape
# Lost dim information
a = torch.rand(4, 1, 28, 28)
print(a.shape)
# 保证numel()一致即可
print(a.view(4, 28*28))
print(a.view(4, 28*28).shape)
print(a.view(4*28, 28).shape)
print(a.view(4*1, 28, 28).shape)
b = a.view(4, 784)
print(b.view(4, 28, 28, 1)) # Logic Bug
# 数据的存储/维度舒徐非常重要，需要时刻记住
# print(b.view(4, 1, 28, 28))






# Flexible but prone to corrupt
# print(a.view(4, 783))   # 4 * 28 * 28





# Squeeze v.s. unsqueeze
# unsqueeze
print(a.shape)
print(a.unsqueeze(0).shape)
# print(a.unsqueeze(1).shape)
# 这里的-1是指的原来的a的最后一个维度
# index取值范围[-a.dim()-1, a.dim()+1)=>[-5, 5)
print(a.unsqueeze(-1).shape)
print(a.unsqueeze(4).shape)
print(a.unsqueeze(-4).shape)
print(a.unsqueeze(-5).shape)
a = torch.tensor([1.2, 2.3])    # shape [2]
print(a.unsqueeze(-1))
print(a.unsqueeze(0))

# For example
b = torch.rand(32)  # [32]
f = torch.rand(4, 32, 14, 14)
# [32]=>[32, 1]=>[32, 1, 1]=>[1, 32, 1, 1]
b = b.unsqueeze(1).unsqueeze(2).unsqueeze(0)
print(b.shape)
# bias相当于给每个channel上的所有像素增加一个偏置
# 扩张后面会讲

# squeeze
print(b.squeeze().shape)
print(b.squeeze(0).shape)
print(b.squeeze(-1).shape)
print(b.squeeze(1).shape)   # 无法挤压
print(b.squeeze(-4).shape)





# Expand / repeat
# Expand: broadcasting
# Repeat: memory copied
# Expand/expand_as
a = torch.rand(4, 32, 14, 14)
print(b.shape)
print(b.expand(4, 32, 14, 14).shape)
# print(b.expand(4, 28, 14, 14).shape)  # 仅限于原来的维度为1，不然会报错
print(b.expand(-1, 32, -1, -1).shape)   # -1维度保持不变
print(b.expand(-1, 32, -1, -4).shape)

# repeat
# Memory touched
print(b.shape)
# [1, 32, 1, 1] => [1*4, 32*32, 1*1, 1*1]
print(b.repeat(4, 32, 1, 1).shape)
print(b.repeat(4, 1, 1, 1).shape)
print(b.repeat(4, 1, 32, 32).shape)





# .t
a = torch.randn(3, 4)
print(a.t())
# print(b.t())  # .t方法只是用dim<=2





# Transpose
a = torch.randn(4, 3, 32, 32)
print(a.shape)
# [b, c, h, w]=>[b, w, h, c]=>[b, w*h*c]=>[b, c, w, h]
# print(a.transpose(1, 3).shape)
# a1 = a.transpose(1, 3).view(4, 3*32*32).view(4, 3, 32, 32)
# Tips: 数据的维度顺序必须和存储顺序一致
# 矩阵的转置会导致存储空间不连续，需要调用其的.contiguous方法将其转为连续
# [b, c, h, w]=>[b, w, h, c]=>[b, w*h*c]=>[b, c, w, h]
a1 = a.transpose(1, 3).contiguous().view(4, 3*32*32).view(4, 3, 32, 32)
# [b, c, h, w]=>[b, w, h, c]=>[b, w*h*c]=>[b, w, h, c]=>[b, c, h, w]
a2 = a.transpose(1, 3).contiguous().view(4, 3*32*32).view(4, 32, 32, 3).transpose(1, 3)
print(a1.shape)
print(a2.shape)
print(torch.all(torch.eq(a, a1)))
print(torch.all(torch.eq(a, a2)))
# view会导致维度顺序关系变模糊，所以需要人为地跟踪






# permute
a = torch.rand(4, 3, 28, 28)
print(a.transpose(1, 3).shape)
b = torch.rand(4, 3, 28, 32)
print(b.transpose(1, 3).shape)
# [b, c, h, w]=>[b, h, w, c]
print(b.transpose(1, 3).transpose(1, 2).shape)
print(b.permute(0, 2, 3, 1).shape)
# [b, h, w, c]是numpy存储图片的格式，需要这一步才能导出numpy
