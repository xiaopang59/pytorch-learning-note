import torch
import numpy as np

# indexing
# dim 0 first
a = torch.rand(4, 3, 28, 28)
print(a[0].shape)
print(a[0, 0].shape)
print(a[0, 0, 2, 4])
# print(a.dim())



# select first/last N
print(a.shape)
print(a[:2].shape)
print(a[:2, :1, :, :].shape)
print(a[:2, 1:, :, :].shape)
print(a[:2, -1:, :, :].shape)




# seleect by steps
print(a[:, :, 0:28:2, 0:28:2].shape)
print(a[:, :, ::2, ::2].shape)
# 0:28:等同于0:28:1
# 一种通用形式: start:end:step




# select by specific index
print(a.shape)
print(a.index_select(0, torch.tensor([0, 2])).shape)
print(a.index_select(1, torch.tensor([1, 2])).shape)
print(a.index_select(2, torch.arange(28)).shape)
print(a.index_select(2, torch.arange(8)).shape)




# ...
print(a.shape)
print(a[...].shape)
print(a[0, ...].shape)  # a[0, ...] => a[0]
print(a[:, 1, ...].shape)   # a[:, 1, ...] => a[:, 1]
print(a[..., :2].shape)     # a[..., :2] => a[:, :, :, :2]
# 当有...出现时，右边的索引需要理解为最右边
print(a[0, ..., ::2].shape) # a[0, :, :, ::2]




# select by mask
x = torch.randn(3, 4)
mask = x.ge(0.5)
print(mask)
# print(mask.type())    # torch.BoolTensor
# .masked_select()
print(torch.masked_select(x, mask))
print(torch.masked_select(x, mask).shape)
# 之所以打平是因为大于0.5的元素个数是根据内容才能确定的





# select by flatten index
src = torch.tensor([[4, 3, 5],
                    [6, 7, 8]])     # [2, 3]=>[6]
print(torch.take(src, torch.tensor([0, 2, 5])))