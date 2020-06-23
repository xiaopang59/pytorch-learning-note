"""
Tensor advanced operation
    Where
    Gather


Gather
torch.gather()
"""

import torch

"""
where
torch.where(condition, x, y) -> Tensor
    Return a tensor of elements selected from either x or y, depending on condition.
    The operation is defined as
                    out(i) = x(i) if condition(i) otherwise yi
"""
cond = torch.tensor([[0.6769, 0.7271],
                     [0.8884, 0.4163]])
a = torch.zeros(2, 2)
b = torch.ones(2, 2)
print(torch.where(cond > 0.5, a, b))






"""
Gather
torch.gather(input, dim, index, out=None) -> Tensor
    Gather values along an axis specified by dim.
    For a 3-D tensor the output is specified by:
        out[i][j][k] = input[index[i][j][k]][i][k]  # if dim == 0
        out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
        out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2
"""
"""
retrieve global label
    argmax (pred) to get relative labeling
    On some condition, our label is distinct from relative labeling
"""
prob = torch.randn(4, 10)
idx = prob.topk(dim=1, k=3)
print(idx)
idx = idx[1]
label = torch.arange(10) + 100
print(torch.gather(label.expand(4, 10), dim=1, index=idx.long()))