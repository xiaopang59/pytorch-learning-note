"""
Math operation
    Add/minus/multiply/divide
    Matmul
    Pow
    Sqrt/rsqrt
    Round
"""

import torch

a = torch.rand(3, 4)
b = torch.rand(4)
# print(a.shape)
# print(b.shape)
print(a + b)
print(torch.add(a, b))
print(torch.all(torch.eq(a-b, torch.sub(a, b))))
print(torch.all(torch.eq(a*b, torch.mul(a, b))))
print(torch.all(torch.eq(a/b, torch.div(a, b))))
# 建议直接使用运算符







"""
matmul
    Torch.mm
        only for 2d
    Torch.matmul
        @  
"""
a = torch.tensor([[3., 3.], [3., 3.]])
b = torch.ones(2, 2)
print(torch.mm(a, b))
print(torch.matmul(a, b))
print(a@b)






# An example
a = torch.rand(4, 784)
x = torch.rand(4, 784)
w = torch.rand(512, 784)    # [ch-out, ch-in]
print((x@w.t()).shape)





# 2d tensor matmul?
a = torch.rand(4, 3, 28, 64)
b = torch.rand(4, 3, 64, 32)
# print(torch.mm(a, b).shape)   # error mm 只适用于2d
print(torch.matmul(a, b).shape) # 只取最后两维
# 其实就是支持多个矩阵对并行相乘
b = torch.rand(4, 1, 64, 32)
print(torch.matmul(a, b).shape) # Boardcasting: [4, 1, 64, 32]=>[4, 3, 64, 32]
b = torch.rand(4, 64, 32)
# print(torch.matmul(a, b).shape)   # Boardcasting不适用





# Power
a = torch.full([2, 2], 3)
print(a.pow(2))
print(a**2)
aa = a**2
print(aa.sqrt())
print(aa.rsqrt())
print(aa**0.5)





# Exp log
a = torch.exp(torch.ones(2, 2))
print(a)
print(torch.log(a))




"""
Approximation
    .floor() .ceil()
    .round()
    .trunc() .frac()
"""
a = torch.tensor(3.14)
print("floor:{0}, ceil:{1}, trunc:{2}, frac:{3}".format(a.floor(), a.ceil(), a.trunc(), a.frac()))
a = torch.tensor(3.499)
print(a.round())
a = torch.tensor(3.5)
print(a.round())





"""
clamp
    gradient clipping
    (min)
    (min, max)
"""
grad = torch.rand(2, 3) * 15
print("max:{0}, median:{1}, clamp:{2}".format(
    grad.max(), grad.median(), grad.clamp(10)))
print(grad)
print(grad.clamp(0, 10))