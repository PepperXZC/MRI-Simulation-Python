import matplotlib.pyplot as plt
import torch

class body:
    def __init__(self, length, bandwidth, gamma) -> None:
        self.length = length
        self.data = torch.zeros(length, length, length, 3)
        # 进一步地简化：中心点初始值均为[0,0,1]

        # 正方形
        # self.lower, self.upper = length//2 - 5, length//2 + 5

        # self.data[self.lower:self.upper,self.lower:self.upper,:,2] = 1

        # 矩形
        a, b = int(length // 2), int(bandwidth // 2)
        self.lower, self.upper = a - b, a + b
        # self.data[:,self.lower:self.upper,:,2] = 1
        self.data[:,:,:,2] = 1
        # 单点
        # self.data[length//2,length//2,:,2] = 1
        # self.data[10, 10 ,:,2] = 1
        # self.data[self.length//2,self.length//2,:,2] = 1
        self.gamma = gamma # gamma / 2pi = 42.58 MHz/T

def slice_select(body, z0, thickness):
    return body.data[:,:,(z0-thickness//2):(1+z0+thickness//2)]

def get_point_index(length, bandwidth):
    # 矩形
    # TODO：生成两个 point_index 集合
    li_vassel, li_muscle = [], []
    a, b = int(length // 2), int(bandwidth // 2)
    lower, upper = a - b, a + b
    for i in range(length):
        for j in range(length):
            if j >= lower and j < upper:
                li_vassel.append((i, j))
            else:
                li_muscle.append((i, j))
    return li_vassel, li_muscle

    # return [(length//2,length//2)]