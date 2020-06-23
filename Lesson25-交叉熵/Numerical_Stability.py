import torch
import torch.nn.functional as F

x = torch.randn(1, 784)
w = torch.randn(10, 784)

logits = x@w.t()
print(logits.size())

pred = F.softmax(logits, dim=1)
print(pred.size())

pred_log = torch.log(pred)
print(F.cross_entropy(logits, torch.tensor([3])))
print(F.nll_loss(pred_log, torch.tensor([3])))