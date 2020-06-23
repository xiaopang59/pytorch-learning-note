"""
Merge or split
    Cat
    Stack
    Split
    Chunk
"""


import torch

"""
Cat 
    Statistics about socres
        [class1-4, students, scores]
        [class5-9, students, scores]
"""
a = torch.rand(4, 32, 8)
b = torch.rand(5, 32, 8)
print(torch.cat([a, b], dim=0).shape)
a1 = torch.rand(4, 3, 32, 32)
a2 = torch.rand(5, 3, 32, 32)
print(torch.cat([a1, a2], dim=0).shape)
a2 = torch.rand(4, 1, 32, 32)
# print(torch.cat([a1, a2], dim=0).shape)   # error 维度不匹配
print(torch.cat([a1, a2], dim=1).shape)
a1 = torch.rand(4, 3, 16, 32)
a2 = torch.rand(4, 3, 16, 32)
print(torch.cat([a1, a2], dim=2).shape)





# Stack
# create new dim
print(torch.cat([a1, a2], dim=2).shape)
print(torch.stack([a1, a2], dim=2).shape)
a = torch.rand(32, 8)
b = torch.rand(32, 8)
print(torch.stack([a, b], dim=0).shape)
c = torch.rand(64, 8)
# print(torch.stack([a, c], dim=0).shape)
# cat 拼接维度可以不一致，其他维度必须一致
# stack 所有维度都必须一致





# Cat v.s. stack
b = torch.rand([30, 8])
# print(torch.stack([a, b], dim=0).shape)   # error
print(torch.cat([a, b], dim=0).shape)






# Split: by len
b = torch.rand(32, 8)
print(a.shape)
c = torch.stack([a, b], dim=0)
print(c.shape)
aa, bb = c.split([1, 1], dim=0)
print(aa.shape)
print(bb.shape)
aa, bb = c.split(1, dim=0)
print(aa.shape)
print(bb.shape)
# aa, bb = c.split(2, dim=0)    # error: not enough values to unpack
# print(aa.shape)
# print(bb.shape)
# 只能拆成1个，所以返回1个tensor，不能用2个tensor接受






# Chunk: by num
b = torch.rand(32, 8)
print(a.shape)
c = torch.stack([a, b], dim=0)
print(c.shape)
# aa, bb = c.split(2, dim=0)  # error
# print(aa.shape)
# print(bb.shape)
aa, bb = c.chunk(2, dim=0)
print(aa.shape)
print(bb.shape)