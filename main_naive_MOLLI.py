import torch
import matrix_rot
import math
import sequence
import freprecess
import matplotlib.pyplot as plt
import experiment
from scipy.optimize import curve_fit

import random
def randomcolor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 默认在[0, 10000] 内寻找 T1
# 默认 TI 是 torch.tensor
class info:
    def __init__(
        self, 
        T1_generate = 1000, 
        T2 = 45, 
        TR = 5,
        TE = 1.1,
        rep_time = 49, # 假设每个RF给出 rep_time 个读取次数 
        # TI_5 = [120, 870, 1620, 2370 , 3120],
        TI_5 = [120, 1120, 2120, 3120 , 4120],
        # TI_5 = [],
        # TI_3 = [320, 1070, 1820],
        TI_3 = [320, 1320, 2320],
        readout_index = [0,5,1,6,2,7,3,4],
        # readout_index = [0,5,1,6,2],
        # TI_3 = [],
        # 因为不知道第一个read开始之前有多少时间间隙
        FA_pulse = math.pi,
        FA_small =  [5 * math.pi / 180],
        # FA_small = torch.arange(2, 12, 2) * math.pi / 180,
        # fa_slice = 10,
        df = 0,
        t_interval = 30,
        # total_time = [3120+2*750, 5000], # 5-3-3
        total_time = [4120+3*1000, 5000], # 5-3-3
        num_excitation = 1,
        fov = 320,
        pool_length = 128,
        bandwidth = 3,
        dt = 0.1,
        roll_rate = 0.014,
        c = 2,
        gamma = 42.58 * 1e4,# MHz/G
        Gz = None,
        tau_x = 1
    ) -> None:
        '''
        在这里，整体的时间如下(按顺序)：
        total_time = (180y + TI_5[0] + TR * rep_time + TI_5[1] - ...)
        其中 TE 为每个 TR 中 10y时刻 到 readout 梯度的中点
        '''
        self.TI_5 = TI_5 # 每个 molli 中有两个TI
        self.TI_3 = TI_3 # 每个 molli 中有两个TI
        self.T1 = T1_generate # float64
        self.TE = TE
        self.rep_time = rep_time
        self.read_time = (self.rep_time + 1) / 2
        self.fa_10 = FA_small
        self.TR = TR
        self.df = df
        self.gamma = gamma
        self.total_time = total_time
        self.T2 = T2
        self.pool_length = pool_length
        self.fa_180 = FA_pulse
        self.t_interval = t_interval
        self.num_excitation = num_excitation
        self.m0 = torch.Tensor([0,0,1]).to(device).T
        self.fov = fov
        self.bandwidth = bandwidth
        self.dt = dt
        # 这个量代表 每 1 ms，血流流动 roll_rate 厘米
        self.roll_rate = roll_rate
        # 这个量代表画框下每1cm对应实际心肌部位的 c:real_length厘米
        self.real_length = c
        self.each_time = round(self.real_length / self.roll_rate, len(str(self.dt))-2) 
        self.readout_index = readout_index
        self.Gz = torch.arange(- self.fov * self.fov / 2 , self.fov * self.fov / 2, 1).reshape((self.fov, self.fov)) * torch.pi / self.fov*self.fov
        # self.fa_slice = fa_slice # 由 Slice profile 给出, 
        # 是个 5 维向量，因为有5个fa
        # 先不管它的 sub-slice 版本！只是一个数

        # self.theExp = torch.exp(-TI / T1_generate)
        

        # # 转化为 norm，我也不知道为什么非得这么做，先试试吧，可以删掉
        # self.theExp = self.theExp / torch.linalg.norm(self.theExp)

    def get_readout_time(self):
        return (self.TI_5 + self.TI_3)

def index_find(arr1, arr2):
    '''
    假设arr1, arr2中都没有重复值，且arr1的值在arr2中只会出现小于等于1次，给出arr2各值在arr1中的索引
    '''
    assert len(arr2.shape) == 2
    if isinstance(arr1, list):
        arr1 = torch.Tensor(arr1)
    # print("now.", arr2[:,:,None])
    # print("hi", arr2 * 10)
    # return torch.where(arr1==arr2[:,:,None])[2].reshape(arr2.shape)
    return (arr2 * 10).type(torch.long)
    # return torch.where(arr1==arr2)[2].reshape(arr2.shape)

def main():
    # test_info = info()

    # m0 = torch.Tensor([0,0,1]).to(device).T
    # program = sequence.molli(test_info)
    # program.simulation()
    # x = torch.Tensor(program.readout_time)
    # for t in program.x_time:

    # 画图
    # for index in range(len(test_info.fa_10)):
    #     angle = int(test_info.fa_10[index] * 180 / math.pi)
    #     plt.plot(program.x_time[0], [key[2] for key in program.result[index]], color=randomcolor(), label='Mz')
    
    # # # # # plt.show()
    # plt.plot(program.x_time[0], [key[0] for key in program.result[0]], color='r', label='Mx')
    # plt.plot(program.x_time[0], [key[1] for key in program.result[0]], color='g', label='My')
    # # plt.scatter(x, program.readout_Mz)
    # plt.legend()
    # plt.show()
    
    # scipy-curve fit 尝试
    from scipy.optimize import curve_fit
    import numpy as np
    def model(x, A, B, t1):
        return abs(A - B * np.exp(- x / t1))
    res = []
    # a, b = int(program.readout_Mxy.shape[0]) // 2, int(test_info.bandwidth) // 2
    # lower, upper = a - b, a + b
    # plt.ion()
    from tqdm import tqdm
    # for i in tqdm(range(test_info.pool_length)):
    #     for j in range(test_info.pool_length):
    #     # for j in range(test_info.bandwidth):
    #         plt.clf()
    # T1_list = torch.arange(100, 1500, 200)
    T1_list = [1000]
    T1_res = []
    for T1 in T1_list:
        test_info = info(T1_generate=T1)
        program = sequence.naive_molli(test_info)
        program.simulation()
        x = torch.Tensor(program.readout_time)
        param, param_cov = curve_fit(model, x.numpy(), program.readout_Mxy[:,0,0], p0=[0.5,1,T1], maxfev = 10000000)
        res = param[2] * (param[1] / param[0] - 1)
        accuracy = (res - T1) / T1 * 100
        T1_res.append(accuracy)
        print(accuracy)
    plt.plot(T1_list, T1_res,marker='s')
    plt.ylim((-25,0))
    plt.xlabel("Reference T1 (ms)")
    plt.ylabel("percentage error (%)")
    plt.xlim((0, T1_list.max()+100))
    plt.grid()
    plt.show()
    # t = np.arange(x.min(), x.max(), 0.1)
    # # print("T1:", param[2] * (param[1] / param[0] - 1))
    # res.append(param[2] * (param[1] / param[0] - 1))
    #         # if (param[2] * (param[1] / param[0] - 1) > 1000):
    #         #     y = model(t, param[0], param[1], param[2])
    #         #     plt.plot(t,y)
    #         #     plt.scatter(x.numpy(), program.readout_Mxy[:,i,j])
    #         #     plt.show()
    # # print(torch.Tensor(res).mean(), torch.Tensor(res).var())
    # res = torch.Tensor(res).reshape(test_info.pool_length, test_info.pool_length).detach()
    # res = res[:, 63]
    # np.savetxt('res_pool.csv',res,delimiter=',')


    # max_value_index = torch.where(res == res.max())[1].data[0].numpy()
    # print(max_value_index)
    # np.savetxt('res.txt',res,delimiter=',')

    # 热力图生成
    # import seaborn as sns
    # sns.set(font_scale=1.5)
    # fig = sns.heatmap(data=res,square=True, cmap="RdBu_r")
    # fig = fig.get_figure()
    # fig.savefig('fig_pool.png',dpi=400)

main()