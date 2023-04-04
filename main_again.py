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
import bind_sequence

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
        T1_generate = [1400, 1000], # vassel, muscle
        T2 = [50, 50], 
        TR = 5,
        TI_5 = 120,
        TI_3 = 300,
        HR = 80,
        # TFE = 49,
        fa = 60,
        b0 = 1.5, # Tesla
        # tau_x = 1.0, # 原本应该是 receiver bandwidth， 但因为序列设计
        receiver_bandwidth = 83.3, # khz
        gamma = 4258, # Hz / G
        delta = 0.1, # spatial resolution delta
        fov = 6.4,  # cm
        # tau_y = 0.29
        
    ) -> None:
        '''
        在这里，整体的时间如下(按顺序)：
        total_time = (180y + TI_5[0] + TR * rep_time + TI_5[1] - ...)
        其中 TE 为每个 TR 中 10y时刻 到 readout 梯度的中点
        '''

        self.T1 = T1_generate # float64
        self.T2 = T2
        self.TI_5 = TI_5
        self.TI_3 = TI_3
        self.TE = TR / 2
        # self.TFE = TFE
        self.TR = TR
        self.fov = fov
        self.gamma = gamma
        # self.Gx = Gx
        
        self.b0 = b0
        self.fa = fa
        self.HR = HR
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
        self.ky_max = 0.5 * self.N_pe * self.delta_ky
        # self.ky_list = [((self.N_pe - 1) / 2 - m) * self.delta_ky for m in np.arange(0, self.N_pe, 1)]
        self.Gyp = self.ky_max * 1e3 / (self.gamma * self.tau_y)
        self.Gyi = self.delta_ky * 1e3 / (self.gamma * self.tau_y)
        self.m0 = torch.Tensor([0,0,1]).to(device).T
        # self.Gx = Gx
        print("hi")
        
        # self.tau_x = tau_x
        
        # self.Gx = 1e5 * self.k_max / (self.gamma * self.tau_x / 2) # ms
        # self.delta_t = 1e3 / (self.fov * 1e-2 * self.gamma * self.Gx) # sampling period: s
        
        # self.Gyp = 2 / (self.gamma * self.delta * 1e-2 * self.tau_y * 1e-3)
        # self.Gyp = 100 * self.k_max / (self.gamma * self.tau_y * 1e-3) # G/cm
        
        # self.Gyi = 1 / (self.gamma * self.fov * 1e-2  * self.tau_y * 1e-3) # G/cm

        
def test_plot(test_info, slice_data, li_vassel, li_muscle):
    seq = bssfp.sequence(test_info, slice_data, li_vassel, li_muscle, 1)
    prep_num = 1
    fa_sequence, TR_sequence = bssfp.get_sequence_info(test_info, prep_num)
    
    # seq.RF(fa_sequence[0])
    # A, B = freprecess.res(TR_sequence[0], test_info.T1, test_info.T2, 0)
    # plt.ion()
    seq.prep_RF(fa_sequence, TR_sequence, prep_num)
    from tqdm import tqdm
    
    for i in tqdm(range(test_info.N_pe)):
        plt.clf()
        seq.RF(fa_sequence[i + prep_num])
        # plt.pause(0.1)         # 暂停一秒
        # plt.ioff()
        seq.phase_encoding(i)
        seq.readout_encoding(i)
        seq.rewind(i)
        # plt.pause(0.1)         # 暂停一秒
        # plt.ioff() 
    seq.kspace_img = seq.kspace_img.cpu()
    plt.clf()
    torch.save(seq.kspace_img, 'kspace.pt')
    res = abs(seq.kspace_img)
    plt.subplot(1,2,1)
    plt.imshow(res, cmap=plt.cm.gray)

    ft_mat = torch.fft.ifft2(seq.kspace_img)
    ft_mat = torch.fft.ifftshift(ft_mat)
    # ft_mat = torch.fft.fftshift(ft_mat)
    res = abs(ft_mat).numpy()
    plt.subplot(1,2,2)
    plt.imshow(res, cmap=plt.cm.gray)
    plt.show()


if __name__ == "__main__":
    test_info = info()
    gamma = 4258
    body = image.body(64, 20, test_info.gamma)

    # 这是个范围
    # point_index = (body.length // 2, body.length // 2 + 1)
    # point_index = (body.lower, body.upper)

    # 矩形
    li_vassel, li_muscle = image.get_point_index(64, 20)
    

    # 单点时使用这个：
    # point_index = [(10, 10)]
    # point_index = [(i, j) for (i, j) in ]
    # slice_thickness = 5
    # slice_data = image.slice_select(body, 32, 5)
    # test_plot(test_info, slice_data, li_vassel, li_muscle)

    # data = slice_data[point_index[0], point_index[1]]
    # slice_index = (slice_lower, slice_upper)
    # print(slice_data[point_index[0], point_index[0]])
    # print(gradient.get_Gx())

    # test_plot(test_info, slice_data, point_index)

    bS_molli = bind_sequence.bSSFP_MOLLI(test_info, body.data, li_vassel, li_muscle)
    bS_molli.protocol()


    # cv2.imshow(res)
    # cv2.waitKey(0)
    # print("?")