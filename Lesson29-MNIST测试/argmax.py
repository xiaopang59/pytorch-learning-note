import torch
import torch.nn.functional as F

logits = torch.rand(4, 10)

pred = F.softmax(logits, dim=1)
print(pred.shape)

pred_label = pred.argmax(dim=1)
print(pred_label)

print(logits.argmax(dim=1))

label = torch.tensor([9, 3, 2, 4])
correct = torch.eq(pred_label, label)
print(correct)

# print(correct.sum())
# print(correct.sum().float())
# print(correct.sum().float().item())
print(correct.sum().float().item()/4)