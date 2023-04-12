import freprecess
import torch
import math
import numpy as np
import matrix_rot
import freprecess
import copy

class NewProton:
    def __init__(self, info, body_slice, device, now_tensor) -> None:
        self.T1 = info.T1[0] # 只记录 vassel 的T1
        self.T2 = info.T2[0]
        self.TR = info.TR

        self.N_pe = info.N_pe
        self.tau_y = info.tau_y
        self.fov = info.fov
        self.delta = info.delta # 在 body 中相邻两个位置的实际对应距离 (cm)
        self.tau_x = info.tau_x
        self.tau_y = info.tau_y
        self.gamma = info.gamma
        self.delta_t = info.delta_t
        self.w0 = info.w0
        self.slice_thickness = body_slice.shape[2]
        self.device = device
        self.bandwidth = info.bandwidth

        # self.new_proton = torch.zeros((1, self.bandwidth, self.slice_thickness, 3)).to(device)

        # self.y_max = (self.fov / self.delta) / 2
        self.y_max = self.fov  / 2
        self.position_x = torch.Tensor([(- (self.fov / self.delta) / 2 + m) * self.delta for m in np.arange(0, info.N_read, 1)]).to(device)

        center_index = (self.fov / self.delta) // 2
        self.lower, self.upper = int(center_index - self.bandwidth // 2), int(center_index + self.bandwidth // 2)
        self.tensor = now_tensor

        self.history = [] # 列表集合. 每个列表的第一个值为当前操作的种类. 最后一个值表示所处的位置. 从 1 开始(y0 + delta_y)

    def check(self):
        li1, li2 = self.history[-1], self.history[-2]
        if li1[0] == li2[0] and li1[-1] == li2[-1]:
            t = li1[-2]
            self.history.pop()
            self.history[-1][-2] += t
    
    def record(self, t=None, Gx=None, Gy=None, FA=None, n=None):
        # 默认输入进来的G都已经乘上了gamma
        if FA != None: # 只有fa
            self.history.append(['fa', FA])
        elif t != None and Gx == None and Gy == None: # 只有时间
            self.history.append(['free_time', t])
        elif Gy != None and Gx != None and t != None:
            self.history.append(['t_xy', Gx, Gy, t, n]) # Gx Gy 一起开
        elif Gx != None and t != None:
            self.history.append(['t_x', Gx, t, n]) # 只有 Gx 开
        if len(self.history) >= 2:
            self.check()
    
    def fa_operation(self, fa, tensor):
        Rflip = matrix_rot.xrot(fa * math.pi / 180).to(self.device)
        for j in range(self.bandwidth):
            tensor[0, j, :] = tensor[0, j, :] @ Rflip.T
        return tensor

    def free_operation(self, t, tensor):
        A, B = freprecess.res(t, self.T1, self.T2, 0)
        for j in range(self.bandwidth):
           tensor[0, j, :] = tensor[0, j, :] @ A.T + B
        return tensor

    def Gxy_operation(self, li, tensor):
        # li = ['t_xy', Gx, Gy, t, n]
        pos_y = self.y0 - (li[4] - 1) * self.delta
        Gy = li[2] * pos_y
        G_tensor = self.position_x[self.lower:self.upper] * li[1] + Gy
        
            
        for j in range(len(G_tensor)):
            A, B = freprecess.res(li[3], self.T1, self.T2, G_tensor[j] + self.w0)
            tensor[0, j, :] = tensor[0, j, :] @ A.T + B
        return tensor
    
    def Gx_operation(self, li, tensor):
        # li = ['t_xy', Gx, t, n]
        G_tensor = self.position_x[self.lower:self.upper] * li[1]
        for j in range(len(G_tensor)):
            A, B = freprecess.res(li[2], self.T1, self.T2, G_tensor[j] + self.w0)
            tensor[0, j, :] = tensor[0, j, :] @ A.T + B
        return tensor

    def output(self, tensor, N=None):
        # 这里 n 表示这是第n个需要给出的proton组
        if N != None:
            self.y0 = self.y_max + N * self.delta
        tensor = copy.deepcopy(self.tensor)
        for li in self.history:
            if li[0] == 'fa':
                tensor = self.fa_operation(li[1], tensor)
            elif li[0] == 'free_time':
                tensor = self.free_operation(li[1], tensor)
            elif li[0] == 't_xy':
                tensor = self.Gxy_operation(li, tensor)
            elif li[0] == 't_x':
                tensor = self.Gx_operation(li, tensor)
        return tensor
    
    def frame_time_update(self, st, time, etf):
        new = 0
        if (st + time) % etf == (st + time):
            rest_time = time
            st += time
        else:
            rest_time = (st + time) % etf
            new_time_before = rest_time
            new = 1
        self.record(t=time)
        return new_time_before if new == 1 else st
    
    def now_update(self):
        self.tensor = self.output(self.tensor)
        self.history = [] # 把之前所有的记录全部保存，只对frame_proton做！