import matplotlib.pyplot as plt
import numpy as np
import torch
import math
import freprecess
import matrix_rot
import cmath
import image
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 默认是 fa/2 - TR/2 prep
def sign(num):
    if num > 0:
        return 1
    else:
        return -1

def get_sequence_info(info, prep_num):
    fa_sequence, TR_sequence = torch.zeros(info.N_pe + prep_num), torch.ones(info.N_pe + prep_num) * info.TR
    fa_sequence[0], TR_sequence[0] = info.fa / 2, info.TR / 2
    for i in range (1, info.N_pe + + prep_num):
        fa_sequence[i] = sign((i+1) % 2) * info.fa
    return fa_sequence, TR_sequence


class sequence:
    # def __init__(self, fa, TR, TFE, point_tensor, gradient, gamma, tau_y, fov, delta, delta_t) -> None:
    def __init__(self, info, body_slice, point_index) -> None:
        self.flip_angle = info.fa * math.pi / 180 # 假设读入60，为角度制。默认60x
        self.T1 = info.T1
        self.T2 = info.T2
        self.TR = info.TR
        # self.TFE = info.TFE # 读多少条线，应该是Npe
        self.N_pe = info.N_pe
        self.N_read = info.N_read
        # 自动默认TE == TR/2
        self.TE = self.TR / 2
        self.gamma = info.gamma
        self.b0 = info.b0
        # self.w0 = info.gamma * info.b0 * 10# 已经除以2pi
        self.w0 = 0
        self.tau_y = info.tau_y
        self.delta_k = 1 / (info.fov)
        self.fov = info.fov
        self.delta = info.delta
        self.tau_x = info.tau_x
        self.tau_y = info.tau_y
        self.delta_t = info.delta_t 

        self.point_index = point_index


        self.fa_sequence = torch.zeros(self.N_pe + 1)
        self.TR_sequence = torch.ones(self.N_pe + 1) * self.TR
        self.data = body_slice.to(device)
        # Gy 默认是以Gyp生成的，根据需求更改
        # gradient = (Gx, Gyp, Gyi)
        # 是已经按照(fov,fov)生成的矩阵
        self.Gx, self.Gyp, self.Gyi = info.Gx, info.Gyp, info.Gyi

        self.ky_list = info.ky_list
        
        self.position_y = torch.Tensor([((self.fov / self.delta - 1) / 2 - m) * self.delta for m in np.arange(0, self.N_pe, 1)]).reshape(-1, 1).to(device)
        self.position_x = torch.Tensor([(- (self.fov / self.delta - 1) / 2 + m) * self.delta for m in np.arange(0, self.N_read, 1)]).to(device)
        self.kspace_img = torch.complex(torch.zeros((info.N_pe, info.N_read)), torch.zeros((info.N_pe, info.N_read))).to(device)
        
        self.temp_x = []
        self.temp_z = []
    
    def RF(self, fa):
        # fa = self.fa_sequence[num_rf]
        Rflip = matrix_rot.xrot(fa * math.pi / 180).to(device)
        self.data = self.data @ Rflip.T
        time = self.TE - self.tau_x / 2 - self.tau_y
        A, B = freprecess.res(time, self.T1, self.T2, 0 + self.w0)
        for l in range(self.point_index[0], self.point_index[1]):
            for r in range(self.point_index[0], self.point_index[1]):
                self.data[l][r] = self.data[l][r] @ A.T + B
        # print(self.data[self.point_index[0]:self.point_index[1], self.point_index[0]:self.point_index[1]])
        
        self.temp_x.append(self.data[self.point_index[0], self.point_index[0], 0, 0].cpu())
        self.temp_z.append(self.data[self.point_index[0], self.point_index[0], 0, 1].cpu())
        plt.scatter(self.temp_x, self.temp_z)
        plt.xlim((-1, 1))
        # plt.ylim((-1e-5,1e-5))
    
    def get_Gy_tensor(self, num_rf):
        G_diff = (self.Gyp - num_rf * self.Gyi) * self.gamma
        return G_diff * self.position_y # 对于正的Gy，从高到低排列

    def prep_RF(self, fa_sequence, TR_sequence, prep_num):
        # bSSFP
        for i in range(prep_num):
            fa = fa_sequence[i]
            Rflip = matrix_rot.xrot(fa * math.pi / 180).to(device)
            self.data = self.data @ Rflip.T
            TR = TR_sequence[i]
            A, B = freprecess.res(TR, self.T1, self.T2, 0 + self.w0)
            for l in range(self.point_index[0], self.point_index[1]):
                for r in range(self.point_index[0], self.point_index[1]):
                    self.data[l][r] = self.data[l][r] @ A.T + B
            self.temp_x.append(self.data[self.point_index[0], self.point_index[0], 0, 0].cpu())
            # print(self.data[0, 0, 0, 1])
            self.temp_z.append(self.data[self.point_index[0], self.point_index[0], 0, 1].cpu())
            plt.clf()
            plt.scatter(self.temp_x, self.temp_z)
            plt.xlim((-1, 1))
            plt.pause(0.1)         # 暂停一秒
            plt.ioff() 
        self.readout_num = len(fa_sequence) - prep_num

    def phase_encoding(self, num_rf):
        Gy_tensor = self.get_Gy_tensor(num_rf)
        Gx = - self.Gx
        Gx_tensor = Gx * self.gamma * self.position_x 
        # print(self.position_x[-1])
        
        # 梯度清零
        self.gradient_matrix = torch.zeros((self.N_pe, self.N_read)).to(device)
        self.gradient_matrix += Gx_tensor
        self.gradient_matrix += Gy_tensor
        self.gradient_matrix = self.gradient_matrix.to(device)
        for l in range(self.point_index[0], self.point_index[1]):
            for r in range(self.point_index[0], self.point_index[1]):
                # if self.data[i][j][0][2] != 0:
                df = self.gradient_matrix[l][r] # TODO: 考虑时间的话，这个真的对吗？
                A, B = freprecess.res(self.tau_y, self.T1, self.T2, df + self.w0)
                self.data[l][r] = self.data[l][r] @ A.T + B
        self.temp_x.append(self.data[self.point_index[0], self.point_index[0], 0, 0].cpu())
            # print(self.data[0, 0, 0, 1])
        self.temp_z.append(self.data[self.point_index[0], self.point_index[0], 0, 1].cpu())
        plt.clf()
        plt.scatter(self.temp_x, self.temp_z)
        plt.xlim((-1, 1))
        plt.pause(0.1)         # 暂停一秒
        plt.ioff() 

    def readout_encoding(self, num_rf):
        # exp_y = torch.asarray([cmath.exp(-1 * 2j*math.pi*ky*y[0]) for y in self.position_y]).reshape(-1, 1).to(device)
        # 梯度再次清零 前面的梯度已经关掉了
        # self.gradient_matrix = torch.zeros((self.data.shape[0], self.data.shape[1]))  # +Gx
        self.gradient_matrix = self.Gx * self.gamma * self.position_x
        Gx_time = torch.arange(0, self.tau_x, self.delta_t)

        now_time = 0
        # temp_data = copy.deepcopy(self.data)
        for i in range(len(Gx_time)):
            now_time += self.delta_t
            for r in range(self.point_index[0], self.point_index[1]):
                df = self.gradient_matrix[r]
                A, B = freprecess.res(self.delta_t, self.T1, self.T2, df + self.w0)
                self.data[self.point_index[0]:self.point_index[1], r] = self.data[self.point_index[0]:self.point_index[1], r] @ A.T + B
            img_matrix = torch.complex(self.data[:, :, :, 0], self.data[:, :, :, 1]).to(device)
            sample = img_matrix.sum()
            self.kspace_img[num_rf, i] = sample if num_rf % 2 == 0 else - sample

    def rewind(self, num_rf):
        Gy_tensor = - self.get_Gy_tensor(num_rf)
        Gx = - self.Gx
        Gx_tensor = Gx * self.gamma * self.position_x

        self.gradient_matrix = torch.zeros((self.data.shape[0], self.data.shape[1])).to(device)
        self.gradient_matrix += Gx_tensor
        self.gradient_matrix += Gy_tensor
        # self.gradient_matrix *= self.tau_y
        self.gradient_matrix = self.gradient_matrix.to(device)

        for l in range(self.point_index[0], self.point_index[1]):
            for r in range(self.point_index[0], self.point_index[1]):
                # if self.data[i][j][0][2] != 0:
                df = self.gradient_matrix[l][r]
                A, B = freprecess.res(self.tau_y, self.T1, self.T2, df + self.w0)
                self.data[l][r] = self.data[l][r] @ A.T + B
        
        time = self.TE - self.tau_x / 2 - self.tau_y
        A, B = freprecess.res(time, self.T1, self.T2, 0 + self.w0)
        for l in range(self.point_index[0], self.point_index[1]):
            for r in range(self.point_index[0], self.point_index[1]):
                self.data[l][r] = self.data[l][r] @ A.T + B
        
        self.temp_x.append(self.data[self.point_index[0], self.point_index[0], 0, 0].cpu())
        self.temp_z.append(self.data[self.point_index[0], self.point_index[0], 0, 1].cpu())
        plt.scatter(self.temp_x, self.temp_z)
        plt.xlim((-1, 1))

# test for watching data

# print(self.data[self.point_index[0]:self.point_index[1], self.point_index[0]:self.point_index[1]])
# print(self.data[self.point_index[0]:self.point_index[0]+5, self.point_index[0]:self.point_index[0]+5])
# ratio = self.data[self.point_index[0], self.point_index[0], 0, 0] / self.data[self.point_index[0], self.point_index[0], 0, 1]
# print(self.data[self.point_index[0], self.point_index[0], 0, 0], self.data[self.point_index[0], self.point_index[0], 0, 1], ratio)
# print("hi")

# print(self.data[self.point_index[0]:self.point_index[0]+3, self.point_index[0]:self.point_index[0]+3])

# print(self.data[self.point_index[0]:self.point_index[1], r])
# print(self.data[78, 78, 0] @ A.T + B)
# print(((self.data[49:79, 78]) @ A.T + B)[-1, -1] )
# print(self.data[78, 78, 0] @ A.T + B == (self.data[49:79, 78] @ A.T + B)[-1, -1])

# print(self.data[self.point_index[0]:self.point_index[1], self.point_index[0]:self.point_index[1]])
# print(self.data[self.point_index[0]][self.point_index[0]])