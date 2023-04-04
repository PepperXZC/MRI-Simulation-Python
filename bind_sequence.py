import numpy as np
import matplotlib.pyplot as plt
import torch
import copy
import math
import freprecess
import bssfp
import matrix_rot
import image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: 每当进行有关时间的操作，都需要考虑血流移动的问题。
# TODO: 目前考虑 data 里所有的点都有质子.
# TODO：其它的都没问题，考虑蒙特卡洛扔质子。

class bSSFP_MOLLI:
    def __init__(self, info, data, li_vassel, li_muscle) -> None:
        self.m0 = info.m0
        self.time = 0   # 积累起来的时间
        # self.each_time_flow = info.each_time # TODO：这里计算好：每过 each_time 时间就移动1个矩阵上的位置。
        self.data = data.to(device)  # (fov, fov, fov, 3)
        # self.vassel_width = info.bandwidth
        self.fov = info.fov
        self.HR = info.HR  # bpm
        self.li_vassel = li_vassel
        self.li_muscle = li_muscle

        self.TI_5 = info.TI_5
        self.TI_3 = info.TI_3
        self.T1 = info.T1
        self.T2 = info.T2
        self.TR = info.TR

        # 这里默认 prep_num = 1
        self.TR_time = self.TR * (info.N_pe + 0.5) # 读取 整整一张图 的时间
        # self.before_lines = (info.N_pe  // 2 - 1) if info.N_pe % 2 == 0 else (info.N_pe  // 2)
        self.before_lines = 0 # 第一根线就是中心线

        self.center_line_time = info.TR * self.before_lines # 从读取开始，到中心线的时间
        self.Rflip_180 = matrix_rot.xrot(torch.pi)

        self.info = info
    
    def input_FA(self, fa_list, prep_num):
        self.fa_list = fa_list
        self.prep_num = prep_num
    
    def flow(self):
        center_index = (self.fov + 1) // 2
        lower, upper = center_index - self.vassel_width // 2, center_index + self.vassel_width // 2
        vassel = copy.deepcopy(self.data[:, lower:(upper+1)])
        vassel = torch.roll(vassel, 1, 0)
        for i in range(len(vassel[0])):
            # if random.randint(1, 10) < self.half:
            vassel[0][i] = self.m0
        self.data[:, lower:(upper+1)] = vassel
        self.time -= self.each_time_flow
    
    def check_flow(self):
        if self.time >= self.each_time_flow:
            return True
        else:
            return False
    
    # 考虑在读取的过程中已经将x-y矢量打掉。
    # 在a/2 - T/R的收尾，可以考虑最后打一个 -a/2
    def relax(self, time):
        A, B = freprecess.res(time, self.T1[0], self.T2[0], 0)
        for (i, j) in self.li_vassel:
            self.data[i, j, :] = self.data[i, j, :] @ A.T + B
        A, B = freprecess.res(time, self.T1[1], self.T2[1], 0)
        for (i, j) in self.li_muscle:
            self.data[i, j, :] = self.data[i, j, :] @ A.T + B
    
    def get_time(self):
        ms_per_beat = 60 * 1e3 / self.HR
        # self.TI_5_list = [self.TI_5 + i * ms_per_beat for i in range(strategy[0])]
        # self.TI_3_list = [self.TI_3 + i * ms_per_beat for i in range(strategy[2])]
        self.TI_5_before,  self.TI_3_before = self.TI_5 - self.center_line_time, self.TI_3 - self.center_line_time
        self.interval = ms_per_beat - self.TR_time - self.center_line_time # 读取完上一张图，到读取下一张图之间的时间差

        self.before_time = [self.TI_5_before, self.TI_3_before]
        self.inversion_interval = 3 * ms_per_beat

    def inversion_pulse(self):
        # for (i, j) in self.li_vassel + self.li_muscle:
        self.data = self.data @ self.Rflip_180.T # 现在所有点都有数据
    
    def plot(self):
        for index in range(len(self.img_list)):
            plt.subplot(2,4,index + 1)
            ft_mat = torch.fft.ifft2(self.img_list[index])
            ft_mat = torch.fft.ifftshift(ft_mat)
            res = abs(ft_mat).cpu().numpy()
            plt.imshow(res, cmap=plt.cm.gray)
        plt.show()

    def protocol(self):
        # 5
        self.get_time()
        self.img_list = []
        self.data = image.slice_select(self.data, 32, 5)
        for i in range(len(self.before_time)):
            self.inversion_pulse()
            print(self.data[10, 32, 0])
            self.relax(self.before_time[i])
            print(self.data[10, 32, 0])
            if i == 0: # 表示 TI_5
                for j in range(5):
                    # 开销时间：(num_N_pe + 1 / 2) * TR
                    img_plot = bssfp.sequence(self.info, self.data, self.li_vassel, self.li_muscle, prep_num=1)
                    img = img_plot.read_sequence(img_info='TI5_' + str(j))
                    # temp = abs(img).numpy()
                    # plt.subplot(1,2,1)
                    # # plt.imshow(temp, cmap=plt.cm.gray)
                    # plt.imshow(temp, cmap=plt.cm.gray)
                    # ft_mat = torch.fft.ifft2(img)
                    # ft_mat = torch.fft.ifftshift(ft_mat)
                    # res = abs(ft_mat).numpy()
                    # plt.subplot(1,2,2)
                    # plt.imshow(res, cmap=plt.cm.gray)
                    # plt.show()

                    self.img_list.append(img)
                    self.data = img_plot.data
                    self.relax(self.interval)
            elif i == 1:
                for j in range(3):
                    # 开销时间：(num_N_pe + 1 / 2) * TR
                    img_plot = bssfp.sequence(self.info, self.data, self.li_vassel, self.li_muscle, prep_num=1)
                    img = img_plot.read_sequence(img_info='TI3_' + str(j))
                    self.img_list.append(img)
                    self.data = img_plot.data
                    self.relax(self.interval)
        self.plot()
    
