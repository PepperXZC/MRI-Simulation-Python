import math
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 默认 x 为单个列向量
# math.sin(math.pi) = 1.2246467991473532e-16
def zrot(phi):
    Rz = torch.Tensor(
        # [[math.cos(phi),-math.sin(phi),0],
        [[math.cos(phi),math.sin(phi),0],
        [-math.sin(phi),math.cos(phi),0],
        [0, 0, 1]]
        ).to(device)
    return Rz

def xrot(phi):
    Rx = torch.Tensor(
        [[1,0,0],
        [0,math.cos(phi),math.sin(phi)],
        [0, -math.sin(phi), math.cos(phi)]]
        ).to(device)
    return Rx

def yrot(phi):
    Ry = torch.Tensor(
        [[math.cos(phi),0,-math.sin(phi)],
        [0,1,0],
        [math.sin(phi), 0, math.cos(phi)]]
        ).to(device)
    return Ry

def throt(phi, theta):
    z = zrot(-theta)
    x = xrot(phi)
    return torch.linalg.inv(z) @ x @ z