import torch
import math
import matrix_rot
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 这里 t 是一个数，不是矩阵
def res(t, T1, T2, df):
    # phi = 2*math.pi*df*t/1000
    phi = df*t*2*math.pi / 1000
    E1 = math.exp(-t/T1)
    E2 = math.exp(-t/T2)
    z = matrix_rot.zrot(phi).to(device)
    A = torch.Tensor([
        [E2,0,0],
        [0,E2,0],
        [0,0,E1]
    ]).to(device) @ z
    B = torch.Tensor([0,0,1-E1]).T.to(device)
    return A, B

def mo(m_last, m0, dt, T1):
    return m0 + (m_last - m0) * math.exp(-dt/T1)