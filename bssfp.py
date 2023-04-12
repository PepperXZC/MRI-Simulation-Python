import matplotlib.pyplot as plt
import numpy as np
import torch
import math
import freprecess
import matrix_rot
import cmath
import image
import copy
from tqdm import tqdm
import new_proton
import flowpool

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
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
    def __init__(self, info, body_slice, index_list, prep_num, flow, time, frame_proton, flow_time) -> None:
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
        self.w0 = info.w0
        self.tau_y = info.tau_y
        self.delta_k = 1 / (info.fov)
        self.fov = info.fov
        self.delta = info.delta
        self.tau_x = info.tau_x
        self.tau_y = info.tau_y
        self.flow = flow
        # self.time 记录没有flow但已经frep的时间
        if self.flow == True:
            self.time = time
        self.delta_t = info.delta_t
        self.info = info
        self.prep_num = prep_num

        self.bandwidth = info.bandwidth
        
        self.flow_speed = info.flow_speed if info.flow_speed != None else None
        self.each_time_flow = info.each_time_flow

        self.index_list = index_list
        self.slice_thickness = body_slice.shape[2]

        self.data = body_slice.to(device)
        # Gy 默认是以Gyp生成的，根据需求更改
        # gradient = (Gx, Gyp, Gyi)
        # 是已经按照(fov,fov)生成的矩阵
        self.Gx, self.Gyp, self.Gyi = info.Gx, info.Gyp, info.Gyi
        
        self.position_y = torch.Tensor([((self.fov / self.delta) / 2 - m) * self.delta for m in np.arange(0, self.N_pe, 1)]).reshape(-1, 1).to(device)
        self.position_x = torch.Tensor([(- (self.fov / self.delta) / 2 + m) * self.delta for m in np.arange(0, self.N_read, 1)]).to(device)
        # y = 0 : 32
        self.kspace_img = torch.complex(torch.zeros((info.N_pe, info.N_read)), torch.zeros((info.N_pe, info.N_read))).to(device)
        
        self.temp_x = []
        self.temp_z = []
        self.alpha = []
        self.x = 0

        self.flow_time = flow_time
        # self.new_proton = torch.zeros((1, self.bandwidth, self.slice_thickness, 3)).to(device)
        self.frame_proton = frame_proton
        self.new_proton = copy.deepcopy(self.frame_proton) # 只在这里记录带有梯度的信息，output也用这个
    
    # def generate_new_point():
        
    
        
    def RF(self, fa):
        # fa = self.fa_sequence[num_rf]
        self.temp_x, self.temp_z = [], []
        Rflip = matrix_rot.xrot(fa * math.pi / 180).to(device)
        self.data = self.data @ Rflip.T
        self.new_proton.record(FA=fa)
        self.frame_proton.record(FA=fa)
        # 这里time一般都是0
        # time = self.TE - self.tau_x / 2 - self.tau_y
        # self.freeprecess(time)
        # flow的新点也受到同样的影响
    
    def get_Gy_tensor(self, num_rf):
        if self.N_pe % 2 == 0: # center 线是不可能没有的。。
            center = self.N_pe // 2
            num = center + sign((num_rf - 1) % 2) * ((num_rf + 1) // 2)
            G_diff = (self.Gyp - num * self.Gyi) * self.gamma
            return G_diff, G_diff * self.position_y, num
        # 对于正的Gy，从高到低排列

    def prep_RF(self, fa_sequence, TR_sequence, prep_num):
        # bSSFP
        for i in range(prep_num):
            fa = fa_sequence[i].item()
            Rflip = matrix_rot.xrot(fa * math.pi / 180).to(device)
            self.data = self.data @ Rflip.T
            self.new_proton.record(FA=fa)
            self.frame_proton.record(FA=fa)
            TR = TR_sequence[i].item()
            self.data, self.time = flowpool.free_flow(
                data=self.data, time=TR, new_prot=self.new_proton, frame_prot=self.frame_proton, info=self.info, flow=False,
                time_before=self.time, flow_time=self.flow_time, etf=self.each_time_flow, index_list=self.index_list
            )
            # self.temp_x.append(self.data[self.point_index[0], self.point_index[0], 0, 0].cpu())
            # # print(self.data[0, 0, 0, 1])
            # self.temp_z.append(self.data[self.point_index[0], self.point_index[0], 0, 1].cpu())
            # plt.clf()
            # plt.scatter(self.temp_x, self.temp_z)
            # plt.xlim((-1, 1))
            # plt.pause(0.1)         # 暂停一秒
            # plt.ioff() 
        self.readout_num = len(fa_sequence) - prep_num

    def phase_encoding(self, num_rf):
        G_diff, Gy_tensor, self.pe_index = self.get_Gy_tensor(num_rf)
        Gx = - self.Gx
        Gx_tensor = Gx * self.gamma * self.position_x 
        # print(self.position_x[-1])
        
        # 梯度清零
        self.gradient_matrix = torch.zeros((self.N_pe, self.N_read)).to(device)
        self.gradient_matrix += Gx_tensor
        self.gradient_matrix += Gy_tensor
        self.gradient_matrix = self.gradient_matrix.to(device)
        # assert self.gradient_matrix[32, 32] == 0
        # self.temp_x, self.temp_z = [], []
        # self.free_flow(self.tau_y, gradient=[Gx * self.gamma, G_diff])
        self.data, self.time, self.new_proton, self.flow_time = flowpool.free_flow(
            data=self.data, time=self.tau_y, flow=True,
            time_before=self.time, flow_time=self.flow_time, new_prot=self.new_proton, info=self.info,
            etf=self.each_time_flow, grad=[Gx * self.gamma, G_diff], gradient_mat=self.gradient_matrix,
            index_list=self.index_list
        )
                # self.temp_x.append(self.data[l, r, 0, 0].cpu())
                #     # print(self.data[0, 0, 0, 1])
                # self.temp_z.append(self.data[l, r, 0, 1].cpu())
        # plt.clf()   # 暂停一秒
        # plt.ioff()
        


    def readout_encoding(self, num_rf):
        # exp_y = torch.asarray([cmath.exp(-1 * 2j*math.pi*ky*y[0]) for y in self.position_y]).reshape(-1, 1).to(device)
        # 梯度再次清零 前面的梯度已经关掉了
        # self.gradient_matrix = torch.zeros((self.data.shape[0], self.data.shape[1]))  # +Gx
        # self.gradient_matrix = self.Gx * self.gamma * self.position_x
        self.gradient_matrix = torch.stack([self.Gx * self.gamma * self.position_x for _ in range(self.N_pe)])
        Gx_time = torch.arange(0, self.tau_x, self.delta_t)


        # 单点读取：
        # temp_data = copy.deepcopy(self.data)
        for i in range(len(Gx_time)):
            time = self.delta_t

            self.data, self.time, self.new_proton, self.flow_time = \
                flowpool.free_flow(
                data=self.data, time=time, new_prot=self.new_proton,info=self.info,
                time_before=self.time, flow_time=self.flow_time,flow=True,
                etf=self.each_time_flow, grad=[self.Gx * self.gamma], gradient_mat=self.gradient_matrix,
                index_list=self.index_list
            )
            # print(self.new_proton.history[-1])
            img_matrix = torch.complex(self.data[:, :, :, 0], self.data[:, :, :, 1]).to(device)
            sample = img_matrix.sum()
            # self.kspace_img[num_rf, i] = - sample if num_rf % 2 == 0 else sample
            self.kspace_img[self.pe_index, i] = - sample if num_rf % 2 == 0 else sample
            # if i == len(Gx_time) // 2:
         

    def rewind(self, num_rf):
        G_diff, Gy_tensor, _ = self.get_Gy_tensor(num_rf)
        Gy_tensor *= -1
        Gx = - self.Gx
        Gx_tensor = Gx * self.gamma * self.position_x

        self.gradient_matrix = torch.zeros((self.data.shape[0], self.data.shape[1])).to(device)
        self.gradient_matrix += Gx_tensor
        self.gradient_matrix += Gy_tensor
        # self.gradient_matrix *= self.tau_y
        self.gradient_matrix = self.gradient_matrix.to(device)

        self.data, self.time, self.new_proton, self.flow_time = \
                flowpool.free_flow(
                data=self.data, time=self.tau_y, new_prot=self.new_proton, info=self.info,
                time_before=self.time, flow_time=self.flow_time,flow=True,
                etf=self.each_time_flow, grad=[Gx * self.gamma, - G_diff], gradient_mat=self.gradient_matrix,
                index_list=self.index_list
            )
        
        # time = self.TE - self.tau_x / 2 - self.tau_y
        # self.free_flow(time)
        # print(self.data[10, 10, 0])

    def read_sequence(self, save_path, img_info:str):
        center_index = (self.fov / self.delta) // 2
        lower, upper = int(center_index - self.bandwidth // 2), int(center_index + self.bandwidth // 2)

        fa_sequence, TR_sequence = get_sequence_info(self.info, self.prep_num)
        self.prep_RF(fa_sequence, TR_sequence, self.prep_num)
        for i in tqdm(range(self.N_pe)):
            self.RF(fa_sequence[i + self.prep_num].item())
            now_etf = frame_time if i != 0 else copy.deepcopy(self.time)
            # print(self.data[:, lower:upper, 0, 1])
            self.phase_encoding(i)
            # print(self.data[:, lower:upper, 0, 1])
            self.readout_encoding(i)
            # print(self.data[:, lower:upper, 0, 1])
            self.rewind(i)
            print(self.data[:, lower:upper, 0, 1])
            frame_time = self.frame_proton.frame_time_update(now_etf, self.TR, self.each_time_flow)

        print(save_path + '\\kspace'+ img_info +'.pt')
        torch.save(self.kspace_img, save_path + '\\kspace'+ img_info +'.pt')
        self.frame_time = frame_time

        
        print(self.data[:, lower:upper, 0])
        return self.kspace_img.cpu()
        # plt.xlim((-1, 1))

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

# self.temp_x.append(self.data[self.point_index[0], self.point_index[0], 0, 0].cpu())
#             # print(self.data[0, 0, 0, 1])
# self.temp_z.append(self.data[self.point_index[0], self.point_index[0], 0, 1].cpu())
# plt.clf()
# plt.scatter(self.temp_x, self.temp_z)
# plt.xlim((-1, 1))
# plt.pause(0.1)         # 暂停一秒
# plt.ioff() 

# if flow == 1:
            #     self.flow_vassel()
        #     flow_num = int((self.time + time) // self.each_time_flow)
        #     flow = 0
        # # before_time = self.each_time_flow - self.time
        #     if (self.time + time) % self.each_time_flow == (self.time + time):
        #         rest_time = time
        #         self.time += time
        #     else:
        #         flow = 1
        #         self.flow_time += 1
        #         rest_time = (self.time + time) % self.each_time_flow
        #         self.time = rest_time
            # for n in range(flow_num):
            #     t = self.each_time_flow - self.time if n == 0 else self.each_time_flow
            #     self.readout_relax(t, lower, upper, length)
            #     self.flow_vassel()
            # self.readout_relax(time, lower, upper, length)

    # def readout_relax(self, time, lower, upper, length):
    #     # 仅针对竖直方向血管
    #     for r in range(lower, upper):
    #         df = self.gradient_matrix[r]
    #         A, B = freprecess.res(time, self.T1[0], self.T2[0], df + self.w0)
    #         m0 = torch.Tensor([-1, 0, 0]).T
    #         print(m0 @ A.T + B)
    #         self.data[0, 0, :] = self.data[0, 0, :] @ A.T + B
    #     # muscle
    #     for r in range(lower):
    #         df = self.gradient_matrix[r]
    #         A, B = freprecess.res(time, self.T1[1], self.T2[1], df + self.w0)
    #         self.data[:, r, :] = self.data[:, r, :] @ A.T + B
    #     for r in range(upper, length):
    #         df = self.gradient_matrix[r]
    #         A, B = freprecess.res(time, self.T1[1], self.T2[1], df + self.w0)
    #         self.data[:, r, :] = self.data[:, r, :] @ A.T + B

        #     length, bandwidth = self.data.shape[0], self.bandwidth
        # a, b = int(length // 2), int(bandwidth // 2)
        # x_min, x_max = 0, self.data.shape[0]
        # lower, upper = a - b, a + b