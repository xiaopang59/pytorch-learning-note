import torch
import numpy as np

# Import from numpy
a = np.array([2, 3.3])
print(torch.from_numpy(a))  # 从NUMPY导入的FLOAT其实是DOUBLE类型
a = np.ones([2, 3])
print(torch.from_numpy(a))


# Import from List
print(torch.tensor([2., 3.2]))  # tensor接收数据，Tensor/FloatTensor接收数据维度
print(torch.FloatTensor([2., 3.2]))
print(torch.FloatTensor([[2., 3.2], [1., 22.3]]))
print(torch.Tensor(2, 3))   # torch.FloatTensor(d1, d2, d3) shoe



# uninitialized
print(torch.empty(1))
print(torch.Tensor(2, 3))
print(torch.IntTensor(2, 3))
print(torch.FloatTensor(2, 3))




# set default type
print(torch.tensor([1.2, 3]).type())
# 增强学习一般使用DOUBLE，其他一般使用FLOAT
torch.set_default_tensor_type(torch.DoubleTensor)
print(torch.tensor([1.2, 3]).type())




# rand/rand_like, randint
# [0,1]             [min, max)
print(torch.rand(3, 3))
a = torch.rand(3, 3)
print(torch.rand_like(a))
# 均匀采样0~10的TENSOR
# 要用x=10*torch.rand(d1,d2)，randint只能采样整数
print(torch.randint(1, 10, [3, 3]))




# randn
# N(0, 1)
print(torch.randn(3, 3))
# N[u, std]
print(torch.normal(mean=torch.full([10], 0), std=torch.arange(1, 0, -0.1)))




# full
print(torch.full([2, 3], 7))
print(torch.full([], 7))
print(torch.full([4], 7))




# arange/range
print(torch.arange(0, 10))
print(torch.arange(0, 10, 2))
print(torch.range(0, 10))



# linspace/logspce
print(torch.linspace(0, 10, steps=4))
# 生成0到10的10个数构成的等差数列
print(torch.linspace(0, 10, steps=10))
print(torch.linspace(0, 10, steps=11))
print(torch.linspace(0, -1, steps=10))
# 生成2的0次方为起始值，2的1次方为终止值的10个数构成的等比数列
print(torch.logspace(0, 1, steps=10, base=2))   # logspace的base参数可以设置为2, 10, e等底数




# Ones/zeros/eye
print(torch.ones(3, 3))
print(torch.zeros(3, 3))
# eye生成单位矩阵
print(torch.eye(3, 4))
print(torch.eye(3))
a = torch.zeros(3, 3)
print(torch.ones_like(a))




# randperm=>random.shuffle
print(torch.randperm(10))
a = torch.rand(2, 3)
b = torch.rand(2, 2)
idx = torch.randperm(2)
print(idx)
print(a[idx])
print(b[idx])
print(a, b)