import matplotlib.pyplot as plt
import torch
import math
import freprecess
import matrix_rot

# plot_info = main.info()

x = torch.arange(0, 500, 0.1)

R90 = matrix_rot.yrot(- 90 * math.pi / 180)
A, B = freprecess.res(0.1, 500, 45, 10)
m0 = torch.Tensor([0,0,1]).T

M = torch.zeros(len(x), 3)
M[0] = torch.Tensor([0,0,1]).T
M[1] = M[0] @ R90.T

for i in range(2, len(x)):
    M[i] = M[i-1] @ A.T + B

plt.plot(x, M[:, 0], label='Mx')
plt.plot(x, M[:, 1], label='My')
plt.plot(x, M[:, 2], label='Mz')
plt.legend()
plt.grid()
plt.show()