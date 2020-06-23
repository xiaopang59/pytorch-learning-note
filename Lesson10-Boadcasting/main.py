"""
Broadcasting
    Expand
    without copying data

Key idea
    Insert 1 dim ahead
    Expand dims with size 1 to same size
    Feature maps: [4, 32, 14, 14]
    Bias: [32, 1, 1]=>[1, 32, 1, 1]=>[4, 32, 14, 14]

Why broadcasting
    1.for actual demanding
        [class, students, scores]
        Add bias for every students: +5 socre
        [4, 32, 8] + [4, 32, 8]
        [4, 32, 8] + [5.0]
            5.0 [1].unsqueeze(0).unsqueeze(0)=>[1, 1, 1]
                    .expand_as(A)=>[4, 32, 8]
    2.memory consumption
        [4, 32, 8] => 1024
        [5.0]=>1
理解数据的内容和数据的shape之间的区别

Is it broadcasting-able?
    Match from Last dim!
        If current dim=1, expand to same
        If either has no dim, insert one dim and expand to same
        otherwise, NOT broadcasting-able

Situation 1:
    [4, 32, 14, 14]
    [1, 32, 1, 1]=>[4, 32, 14, 14]

Situation 2:
    [4, 32, 14, 14]
    [14, 14]=>[1, 1, 14, 14]=>[4, 32, 14, 14]

Situation 3:
    [4, 32, 14, 14]
    [2, 32, 14, 14]
        Dim 0 has dim, can NOT insert and expand to same
        Dim 0 has distinct dim, NOT size 1
        NOT broadcasting-able

How to understand this behavior?
    When it has no dim
        treat it as all own the same
        [class, student, scores] + [scores]
    When it has dim of size 1
        Treat it shared by all
        [class, student, scores] + [student, 1]

match from last dim

It's effective and critically, intuitive
    [4, 3, 32, 32]
    +[32, 32]
    +[3, 1, 1]
    +[1, 1, 1, 1]
"""

import torch

a = torch.rand(4, 32, 8)
b = torch.tensor(5)
c = b.unsqueeze(0).unsqueeze(0).expand_as(a)
d = a + c
e = a + b
print(d.shape)
print(e.shape)
print(torch.all(torch.eq(d, e)))