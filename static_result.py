import numpy as np
import torch
import main_again
import pandas as pd
import image
import bind_sequence
import copy
import os
from scipy.optimize import curve_fit

device = 'cpu'

T1_muscle = np.arange(400, 1800, 100)
T1_vassel = np.arange(400, 1800, 100)
# data = np.zeros((len(T1_vassel), len(T1_muscle)))
data = np.array([ 1 + 1j for _ in range(len(T1_vassel) * len(T1_vassel))]).reshape((len(T1_vassel), len(T1_muscle)))
df = pd.DataFrame(data, columns=T1_muscle, index=T1_vassel)
# print(df)

def model(x, A, B, t1):
    return A - B * torch.exp(- x / t1)

for i in range(len(T1_vassel)):
    for j in range(len(T1_muscle)):
        T1 = [T1_vassel[i], T1_muscle[j]]
        info = main_again.info(T1_generate=T1)
        body = image.body(info.length, info.bandwidth, info.gamma)
        li_vassel, li_muscle = image.get_point_index(info.length, info.bandwidth)
        path = str(T1_vassel[i] + T1_muscle[j] * 1j)
        if not os.path.exists(path):
            os.makedirs(path)
        bS_molli = bind_sequence.bSSFP_MOLLI(info, body.data, li_vassel, li_muscle, save_path=path)
        x, ksp_list = bS_molli.protocol() # 获得的是kspace
        img_list = np.array([torch.fft.ifftshift(torch.fft.ifft2(ksp_list[i])).abs().numpy() for i in info.index_list])
        


        accuracy_vassel, accuracy_muscle = [], []
        for (l, r) in li_vassel:
            y = copy.deepcopy(img_list[:, l, r])
            y[0] *= -1
            y[1] *= -1
            param, param_cov = curve_fit(model, x, y, p0=[0.5,1,T1[0]], maxfev = int(1e8))
            res = param[2] * (param[1] / param[0] - 1)
            accuracy = 1e2 * (res - T1[0]) / T1[0]
            accuracy_vassel.append(accuracy)
        for (l, r) in li_muscle:
            y = copy.deepcopy(img_list[:, l, r])
            y[0] *= -1
            y[1] *= -1
            param, param_cov = curve_fit(model, x, y, p0=[0.5,1,T1[1]], maxfev = int(1e8))
            res = param[2] * (param[1] / param[0] - 1)
            accuracy = 1e2 * (res - T1[1]) / T1[1]
            accuracy_muscle.append(accuracy)

        here = np.array(accuracy_vassel).mean() + np.array(accuracy_muscle).mean() * 1j
        df[i, j] = here

df.to_csv("result.csv")