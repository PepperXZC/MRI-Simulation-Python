import torch
import matrix_rot
import torch
import freprecess
import math
import sys
import copy
import numpy as np

# 默认全部都是 180y, 10y
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# point: m*3，m为抽样的点的个数
def rot(point:torch.Tensor, phi = torch.pi):
    # 返回 m*3，每条m仍为每个样本不变
    return point @ matrix_rot.zrot(phi).T

def relax(x, time, point_tensor, result, A, B, info, rest_time):
    # for _ in range(time):
    a, b = int(point_tensor.shape[0]) // 2, int(info.bandwidth) // 2
    lower, upper = a - b, a + b
    point = point_tensor[0,0]
    accumu_t = rest_time
    for t in range(int(time / info.dt)):
        point_tensor = point_tensor @ A.T + B

        accumu_t = round(accumu_t + info.dt, len(str(info.dt))-2)
        if accumu_t == info.each_time:
            vassel = copy.deepcopy(point_tensor[:, lower:(upper+1)])
            accumu_t = 0
            vassel = torch.roll(vassel, 1, 0)
            for i in range(len(vassel[0])):
                # if random.randint(1, 10) < self.half:
                vassel[0][i] = info.m0
            point_tensor[:, lower:(upper+1)] = vassel

        point = point @ A.T + B
            # result[N_dt * index + dp + 1] = point[2]
            # dp += 1
        result.append(point)
        # result = [temp_list + [point] for temp_list in result]
        if info.dt < 1:
            x.append(round(x[-1] + info.dt, len(str(info.dt))-2))
        else:
            x.append(x[-1] + info.dt)
    return x, point_tensor, result, accumu_t

class naive_molli:
    def __init__(self, info) -> None:
        # self.point = torch.Tensor([0,0,1]).to(device).T # 初始化
        self.point = info.m0
        self.m0 = info.m0
        self.info = info
        self.Rflip_10 = [matrix_rot.yrot( angle ) for angle in info.fa_10] # list
        self.Rflip_180 = matrix_rot.yrot(torch.pi)
        self.readout_Mxy = torch.zeros(len(self.info.TI_5) + len(self.info.TI_3), self.info.pool_length, self.info.pool_length)
        self.readout_Mz = torch.zeros(len(self.info.TI_5) + len(self.info.TI_3), self.info.pool_length, self.info.pool_length)
    # 因为考虑到 10y 中有RF，需要同时制作时刻表 x_time
        self.dt = info.dt
        self.pool_size = (self.info.pool_length, self.info.pool_length, len(self.m0))
        self.N_TI_5 = [int(num) for num in info.TI_5]
        self.N_TI_3 = [int(num) for num in info.TI_3]
        self.N_per_ext_5 = int(info.total_time[0])
        self.N_per_ext_3 = int(info.total_time[1])
        if len(self.N_TI_5) != 0:
            self.N_5_rest = int(self.N_per_ext_5 - self.N_TI_5[-1] - info.TR * ((info.rep_time // 2) + 1))
        # else:
        #     self.N_5_rest = int(self.N_per_ext_5 - info.TR * info.rep_time)
    
        if len(self.N_TI_3) != 0:
            self.N_3_rest = int(self.N_per_ext_3 - self.N_TI_3[-1] - info.TR * info.rep_time)
        # else:
        #     self.N_3_rest = int(self.N_per_ext_3 - info.TR * info.rep_time)
    # num_excitation = 1
        self.result = [[self.point] for _ in range(len(self.Rflip_10))]
        # self.result.append(self.point)
        # self.x_time = []
        # self.x_time.append(0)
        self.x_time = [[0] for _ in range(len(self.Rflip_10))]
        # 设置时间戳，记录 readout 所在的时间点 目前只设置 Rflip_10 只有一个角度的情况
        # 目前假设 TR * rep_time 之后得到一张图
        self.readout_time = []
        # 血池
        # self.res_pool_tensor = torch.zeros(size=(len(self.N_TI_5) + len(self.N_TI_3), self.info.fov, self.info.fov, len(self.m0)))
        self.pool_tensor = torch.zeros(size=self.pool_size)
        # 默认 m0 = [0,0,1]
        self.pool_tensor[:,:,2] = 1
        self.each_time = self.info.real_length / self.info.roll_rate
        self.vassel = torch.zeros(self.info.pool_length, self.info.bandwidth)
        self.rest_time = 0
        
    
    def insert_points(self, pulse_index, now_point, now_time, new_point, new_time):
        n = int(self.info.TR / self.dt)
        point_interval = (new_point - now_point) / n
        temp_point, temp_time = copy.deepcopy(now_point), copy.deepcopy(now_time)
        # temp_point_list, temp_time_list = [now_point], [now_time]
        for _ in range(n-1):
            # self.x_time[pulse_index], self.point, self.result[pulse_index] = relax(self.x_time[pulse_index], self.dt, self.point, self.result[pulse_index], A, B, self.dt)
            temp_point += point_interval
            temp_time += self.dt
            if self.dt < 1:
                temp_time = round(float(temp_time), len(str(self.dt)) - 2)
            self.x_time[pulse_index].append(copy.deepcopy(temp_time))
            self.result[pulse_index].append(copy.deepcopy(temp_point))
        try:
            assert (now_point[0] + n * point_interval[0]) == new_point[0]
        except AssertionError:
            print(now_point[0], now_point[0] + n * point_interval[0], new_point[0])
        self.x_time[pulse_index].append(new_time)
        self.result[pulse_index].append(new_point)
    
    # def roll(self, rest_time):
    #     each_time = self.info.real_length / self.info.roll_rate

    
    def ti_relax(self, pulse_index, ti_info, A, B):
        '''
        只考虑了一次 Rflip-10 的情形，因为我觉得如果对rep_time中每条线都来一遍10的话数据会差很多。。。
        '''
        for i in range(len(ti_info)):
            if i == 0:
                rest = ti_info[0] - (self.info.read_time - 1) * self.info.TR
                self.x_time[pulse_index], self.pool_tensor, self.result[pulse_index], self.rest_time = relax(
                    self.x_time[pulse_index], rest, self.pool_tensor, self.result[pulse_index], A, B, self.info, self.rest_time
                    )
            
            readout = []
            # read_time_record = []
            for t in range(self.info.rep_time):
                # now_point = self.point              
                self.pool_tensor = self.pool_tensor @ self.Rflip_10[pulse_index].T
                # self.result[pulse_index].append(self.pool_tensor[0,0,1])
                now_time = self.x_time[pulse_index][-1]
                # after_read_time = now_time + self.info.TR * self.info.rep_time
                # after_read_time = now_time + self.info.TR
                after_TR_time = round(now_time + self.info.TR, len(str(self.dt))-2)
                after_TE_time = round(now_time + self.info.TE, len(str(self.dt))-2)
                
                # self.x_time[pulse_index].append(after_read_time)
                # self.insert_points(pulse_index, now_point, now_time, self.point, after_TR_time)
                self.x_time[pulse_index], self.pool_tensor, self.result[pulse_index], self.rest_time = relax(
                    self.x_time[pulse_index], self.info.TE, self.pool_tensor, self.result[pulse_index], A, B, self.info, self.rest_time
                    )
                # 对于每个(kspace)点，它们的 Readout 时间点都是不同的。但是对于(fov, fov)中的点，它们是同时被一起检测的。
                # 所以这里我暂时是将这些所有同时取得的点求均值处理
                # self.readout_time.append(after_TE_time)
                # now_z = self.pool_tensor[:,:,2].mean()
                # print(self.pool_tensor[:,:,2].mean().reshape(-1)[0])
                if t == self.info.read_time:
                    # self.readout_Mz.append(self.pool_tensor[:,:,2])
                    if len(ti_info) > 4:
                        # print(self.pool_tensor[:,:,2])
                        self.readout_Mxy[i,0,0] = torch.abs(self.pool_tensor[0,0,0])
                        self.readout_Mz[i,:,:] = self.pool_tensor[:,:,2]
                        self.readout_time.append(after_TE_time)
                    else:
                        self.readout_Mxy[i + 5,0,0] = torch.abs(self.pool_tensor[0,0,0])
                        self.readout_Mz[i + 5,:,:] = self.pool_tensor[:,:,2]
                        self.readout_time.append(after_TE_time - self.info.total_time[0])
                    # self.readout_time.append(after_TE_time)

                self.x_time[pulse_index], self.pool_tensor, self.result[pulse_index], self.rest_time = relax(
                    self.x_time[pulse_index], self.info.TR - self.info.TE, self.pool_tensor, self.result[pulse_index], A, B, self.info, self.rest_time
                    )
                # 这里是一般 Gz 考虑的方式 但暂时不用 假设 xy 方向被理想地全部清掉.
                # for m in range(self.info.fov):
                #     for n in range(self.info.fov):
                #         self.pool_tensor[m][n] = self.pool_tensor[m][n] @ matrix_rot.zrot(self.info.Gz[m][n]).T
                
                # 理想
                self.pool_tensor[0,0,0] = 0
                assert len(self.x_time[0]) == len(self.result[0])
            
            assert len(self.x_time[0]) == len(self.result[0])
            # n_interval 表示每两个 TI 之间的时间间隔.
            if i != len(ti_info) - 1:
                n_interval = int((ti_info[i + 1] - ti_info[i] - self.info.TR * self.info.rep_time))
            else:
                n_interval = self.N_5_rest

            self.x_time[pulse_index], self.pool_tensor, self.result[pulse_index], self.rest_time = relax(
                self.x_time[pulse_index], n_interval, self.pool_tensor, self.result[pulse_index], A, B, self.info, self.rest_time
                )

            assert len(self.x_time[0]) == len(self.result[0])
        if len(ti_info) == 0:
            self.x_time[pulse_index], self.pool_tensor, self.result[pulse_index], self.rest_time = relax(
                self.x_time[pulse_index], self.N_5_rest, self.pool_tensor, self.result[pulse_index], A, B, self.info, self.rest_time
                )

    def inversion_relax(self, pulse_index):
        # 单点旋转
        # self.point = self.point @ self.Rflip_180.T
        # 血池旋转
        self.pool_tensor = self.pool_tensor @ self.Rflip_180.T
        self.result[pulse_index].append(self.pool_tensor[0,0])
        # 因为 result 是多维的列表所以原本的 append 操作要按照下面这个来写
        # self.result = [temp_list + [self.point] for temp_list in self.result]
        self.x_time[pulse_index].append(self.x_time[pulse_index][-1] + self.dt)

    def simulation(self):
        A, B = freprecess.res(self.dt, self.info.T1, self.info.T2, self.info.df)
        for pulse_index in range(len(self.Rflip_10)):
            self.point = self.m0
            for _ in range(self.info.num_excitation):
                self.inversion_relax(pulse_index)
                # self.x_time = [temp_list + [temp_list[-1] + self.dt] for temp_list in self.x_time]
                # 假设 180y 是下一个dt完成，也就是瞬间完成
                assert len(self.x_time[0]) == len(self.result[0])                            
                self.ti_relax(pulse_index, self.info.TI_5, A, B)
                self.inversion_relax(pulse_index)
                assert len(self.x_time[0]) == len(self.result[0])

                self.ti_relax(pulse_index, self.info.TI_3, A, B)
        self.readout_Mxy = self.readout_Mxy[self.info.readout_index]
        self.readout_Mz = self.readout_Mz[self.info.readout_index]
        self.readout_time = torch.Tensor(self.readout_time)[self.info.readout_index]
        # self.readout_Mz = torch.Tensor(self.readout_Mz)
                
        