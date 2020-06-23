"""
statistics
    norm
    mean sum
    prod
    max, min, argmin, argmax
    kthvalue, topk
"""

import torch


"""
norm
    v.s. normalize, e.g. batch_norm
    matrix norm v.s. vector norm
    
norm-p
"""
a = torch.full([8], 1)
b = a.view(2, 4)
c = a.view(2, 2, 2)
print("b:{0}\n c:{1}".format(b, c))
print("a.norm(1):{0}, b.norm(1):{1}, c.norm(1):{2}".format(
    a.norm(1), b.norm(1), c.norm(1)))
print("a.norm(2):{0}, b.norm(2):{1}, c.norm(2):{2}".format(
    a.norm(2), b.norm(2), c.norm(2)))
print(b.norm(1, dim=1))
print(b.norm(2, dim=1))
print(c.norm(1, dim=0))
print(c.norm(2, dim=0))





# mean, sum, min, max, prod
a = torch.arange(8).view(2, 4).float()
print(a)
print("a.min:{0}, a.max:{1}, a.mean:{2}, a.prod:{3}"
      .format(a.min(), a.max(), a.mean(), a.prod()))
print("a.sum:{}".format(a.sum()))
print("a.argmax:{0}, a.argmin:{1}"
      .format(a.argmax(), a.argmin()))
a = torch.randn(4, 10)
print(a[0])
print(a.argmax())
print(a.argmax(dim=1))

# dim, keepdim
print(a.max(dim=1))
print(a.argmax(dim=1))
print(a.max(dim=1, keepdim=True))
print(a.argmax(dim=1, keepdim=True))






"""
Top-k or k-th
    .topk
        Largest
    kthvalue
"""
print(a.topk(3, dim=1))
print(a.topk(3, dim=1, largest=False))  # => min
print(a.kthvalue(8, dim=1))
print(a.kthvalue(3))
print(a.kthvalue(3, dim=1))





"""
compare
    >, >=, <, <=, !=, ==
    torch.eq(a, b)
        torch.equal(a, b)
"""
print(a > 0)
print(torch.gt(a, 0))
print(a != 0)
a = torch.ones(2, 3)
b = torch.randn(2, 3)
print(torch.eq(a, b))
print(torch.eq(a, a))
print(torch.equal(a, a))