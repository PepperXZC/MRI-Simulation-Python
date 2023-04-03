import numpy as np
import matplotlib.pyplot as plt
import torch
import copy
import math
import freprecess
import bssfp
import matrix_rot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: 每当进行有关时间的操作，都需要考虑血流移动的问题。
# TODO: 目前考虑 data 里所有的点都有质子.
# TODO：其它的都没问题，考虑蒙特卡洛扔质子。

class bSSFP_MOLLI:
    def __init__(self, info, data, point_index) -> None:
        self.m0 = info.m0
        self.time = 0   # 积累起来的时间
        self.each_time_flow = info.each_time # TODO：这里计算好：每过 each_time 时间就移动1个矩阵上的位置。
        self.data = data  # (fov, fov, fov, 3)
        self.vassel_width = info.bandwidth
        self.fov = info.fov
        self.HR = info.HR  # bpm
        self.point_index = point_index

        self.TI_5 = info.TI_5
        self.TI_3 = info.TI_3
        self.T1 = info.T1
        self.T2 = info.T2
        self.TR = info.TR

        self.TR_time = self.TR * self.info.N_pe # 读取整整一张图的时间
        self.before_lines = (info.N_pe  // 2 - 1) if info.N_pe % 2 == 0 else (info.N_pe  // 2)
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
        A, B = freprecess.res(time, self.T1, self.T2, 0)
        self.data = self.data @ A.T + B
    
    def get_time(self):
        ms_per_beat = 60 * 1e3 / self.HR
        # self.TI_5_list = [self.TI_5 + i * ms_per_beat for i in range(strategy[0])]
        # self.TI_3_list = [self.TI_3 + i * ms_per_beat for i in range(strategy[2])]
        self.TI_5_before,  self.TI_3_before = self.TI_5 - self.center_line_time, self.TI_3 - self.center_line_time
        self.interval = ms_per_beat - self.TR_time // 2 - self.center_line_time # 读取完上一张图，到读取下一张图之间的时间差

        self.inversion_interval = 3 * ms_per_beat

    def inversion_pulse(self):
        self.data = self.data @ self.Rflip_180.T
    
    def protocol(self):
        # 5
        self.inversion_pulse()
        self.relax(self.TI_5_before)
        img_plot = bssfp.sequence(self.info, self.data, self.point_index, prep_num=1)
        img = img_plot.read_sequence()
        