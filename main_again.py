import torch
import math
import sequence
import matrix_rot
import freprecess
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import image
import gradient
import numpy as np
import bssfp
import cv2

import random
def randomcolor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class info:
    def __init__(self, 
        T1_generate = 800, 
        T2 = 200, 
        TR = 2.7,
        # TFE = 49,
        fa = 60,
        b0 = 1.5, # Tesla
        # tau_x = 1.0, # 原本应该是 receiver bandwidth， 但因为序列设计
        receiver_bandwidth = 83.3, # khz
        gamma = 4258, # Hz / G
        delta = 0.1, # spatial resolution delta
        fov = 12.8,  # cm
        # tau_y = 0.29
        
    ) -> None:
        '''
        在这里，整体的时间如下(按顺序)：
        total_time = (180y + TI_5[0] + TR * rep_time + TI_5[1] - ...)
        其中 TE 为每个 TR 中 10y时刻 到 readout 梯度的中点
        '''

        self.T1 = T1_generate # float64
        self.T2 = T2
        self.TE = TR / 2
        # self.TFE = TFE
        self.TR = TR
        self.fov = fov
        self.gamma = gamma
        # self.Gx = Gx
        
        self.b0 = b0
        self.fa = fa
        self.delta = delta

        self.w0 = self.gamma * 1e-4 * self.b0 # MHz
        self.N_pe = int(self.fov / self.delta)
        self.N_read = int(self.fov / self.delta)

        self.delta_k = 1 / (self.fov)

        self.BW = receiver_bandwidth # khz
        # self.delta_t = self.tau_x / self.N_read
        # self.delta_t = 1 / (2 * self.BW) # sampling period, ms
        # self.tau_x = self.N_read * self.delta_t
        # self.Gx = 2 * self.BW * 1e3 / (self.gamma * self.fov)
        

        # self.tau_y = tau_y # ms
        self.tau_y = 1/4 * (self.TR)
        self.tau_x = 2 * self.tau_y 
        self.delta_t = self.tau_x / self.N_read 

        self.Gx = 1e3 / (self.gamma * self.delta_t * self.fov)
        self.delta_ky = 1 / (self.fov) #cm-1
        # self.k_max = 1 / (2 * self.delta)
        self.ky_max = 0.5 * (self.N_pe - 1) * self.delta_ky
        self.ky_list = [((self.N_pe - 1) / 2 - m) * self.delta_ky for m in np.arange(0, self.N_pe, 1)]
        self.Gyp = self.ky_max * 1e3/ (self.gamma * self.tau_y)
        self.Gyi = self.delta_ky * 1e3/ (self.gamma * self.tau_y)
        # self.Gx = Gx
        
        # self.tau_x = tau_x
        
        # self.Gx = 1e5 * self.k_max / (self.gamma * self.tau_x / 2) # ms
        # self.delta_t = 1e3 / (self.fov * 1e-2 * self.gamma * self.Gx) # sampling period: s
        
        # self.Gyp = 2 / (self.gamma * self.delta * 1e-2 * self.tau_y * 1e-3)
        # self.Gyp = 100 * self.k_max / (self.gamma * self.tau_y * 1e-3) # G/cm
        
        # self.Gyi = 1 / (self.gamma * self.fov * 1e-2  * self.tau_y * 1e-3) # G/cm

        

if __name__ == "__main__":
    test_info = info()
    gamma = 4258
    body = image.body(128, test_info.gamma)

    # 这是个范围
    # point_index = (body.length // 2, body.length // 2 + 1)
    point_index = (body.lower, body.upper)
    # point_index = [(i, j) for (i, j) in ]
    slice_data = image.slice_select(body, 32, 1)

    data = slice_data[point_index[0], point_index[1]]
    print(slice_data[point_index[0], point_index[0]])
    # print(gradient.get_Gx())

    seq = bssfp.sequence(test_info, slice_data, point_index)
    prep_num = 1
    fa_sequence, TR_sequence = bssfp.get_sequence_info(test_info, prep_num)
    
    # seq.RF(fa_sequence[0])
    # A, B = freprecess.res(TR_sequence[0], test_info.T1, test_info.T2, 0)
    plt.ion()
    seq.prep_RF(fa_sequence, TR_sequence, prep_num)
    from tqdm import tqdm
    
    for i in tqdm(range(test_info.N_pe)):
        plt.clf()
        seq.RF(fa_sequence[i + prep_num])
        plt.pause(0.1)         # 暂停一秒
        plt.ioff()
        seq.phase_encoding(i)
        seq.readout_encoding(i)
        seq.rewind(i)
        plt.pause(0.1)         # 暂停一秒
        plt.ioff() 
    seq.kspace_img = seq.kspace_img.cpu()
    plt.clf()
    torch.save(seq.kspace_img, 'kspace.pt')
    res = abs(seq.kspace_img)
    plt.imshow(res, cmap=plt.cm.gray)
    plt.show()
    

    # test
    # A, B = freprecess.res(10, 1000, 45, 20)
    # m0 = torch.Tensor([0,0,1]).T
    # m0 = m0 @ matrix_rot.xrot(60 * math.pi / 180).T
    # m1 = m0 @ A.T + B
    # print(m1)
    # A, B = freprecess.res(20, 1000, 45, -20)
    # m2 = m1 @ A.T + B
    # print(m2)
    # A, B = freprecess.res(10, 1000, 45, 20)
    # m3 = m2 @ A.T + B
    # print(m3)

    # cv2.imshow(res)
    # cv2.waitKey(0)
    # print("?")