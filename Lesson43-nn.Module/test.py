from torch import nn
from torch import optim

net = nn.Sequential(nn.Linear(4, 2), nn.Linear(2, 2))
# print(list(net.parameters()))
print(list(net.parameters())[0].shape)
print(list(net.parameters())[3].shape)

print(list(net.named_parameters())[0])
print(list(net.named_parameters())[1])

print(dict(net.named_parameters()).items())

optimizer = optim.SGD(net.parameters(), lr=1e-3)