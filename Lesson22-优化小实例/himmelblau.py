import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

x = np.arange(-6, 6, 0.1)
y = np.arange(-6, 6, 0.1)
print('x,y range: ', x.shape, y.shape)
X, Y = np.meshgrid(x, y)
print('X,Y maps: ', X.shape, Y.shape)
Z = himmelblau([X, Y])

fig = plt.figure("himmelblau")
ax = Axes3D(fig)
ax.plot_surface(X, Y, Z)
ax.view_init(60, -30) # 改变绘制图像的视角,即相机的位置,azim沿着z轴旋转，elev沿着y轴
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.show()



# [1., 0.], [-4, 0.], [4, 0.]
x = torch.tensor([4., 0.], requires_grad=True)
optimizer = torch.optim.Adam([x], lr=1e-3)
for step in range(20000):

    pred = himmelblau(x)

    optimizer.zero_grad()
    pred.backward()
    optimizer.step()

    if step % 2000 == 0:
        print("step {} : x = {}, f(x) = {}"
              .format(step, x.tolist(), pred.item()))