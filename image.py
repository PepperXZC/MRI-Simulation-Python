import matplotlib.pyplot as plt
import torch

class body:
    def __init__(self, length, gamma) -> None:
        self.length = length
        self.data = torch.zeros(length, length, length, 3)
        # 进一步地简化：中心点初始值均为[0,0,1]

        # 正方形
        # self.lower, self.upper = length//2 - 5, length//2 + 5

        # self.data[self.lower:self.upper,self.lower:self.upper,:,2] = 1

        # 单点
        self.data[length//2,length//2,:,2] = 1
        # self.data[self.length//2,self.length//2,:,2] = 1
        self.gamma = gamma # gamma / 2pi = 42.58 MHz/T

# kspace图像data
class kspace:
    def __init__(self, info) -> None:
        # 暂时默认delta_x, delta_y是同样大小的
        self.fov = info.fov  # cm
        # self.sample_rate = 1 / fov
        self.delta = info.delta # spatial resolution

        self.N_pe = int(self.fov / self.delta)
        self.N_read = int(self.fov / self.delta)

        self.data = torch.zeros((self.N_pe, self.N_read))



    def show(self):
        plt.imshow(self.data,cmap=plt.cm.gray_r)

def slice_select(body, z0, thickness):
    return body.data[:,:,(z0-thickness//2):(1+z0+thickness//2)]