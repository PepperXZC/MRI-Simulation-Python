#  打草稿 别在意
import matplotlib.pyplot as plt
import math
import copy
import numpy as np
import torch
from PIL import Image
from scipy.optimize import curve_fit
import main_again
import image
import pandas as pd
from tqdm import tqdm

# ax = []                    # 定义一个 x 轴的空列表用来接收动态的数据
# ay = []                    # 定义一个 y 轴的空列表用来接收动态的数据
# plt.ion()                  # 开启一个画图的窗口
# for i in range(100):       # 遍历0-99的值
# 	ax.append(i)           # 添加 i 到 x 轴的数据中
# 	ay.append(math.sin(i))        # 添加 i 的平方到 y 轴的数据中
# 	plt.clf()              # 清除之前画的图
# 	plt.plot(ax,ay)        # 画出当前 ax 列表和 ay 列表中的值的图形
# 	plt.pause(0.1)         # 暂停一秒
# 	plt.ioff()             # 关闭画图的窗口


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_fft():
    
    mat = torch.load("E:\Study\毕业设计\MRI-simulation\kspaceTI5_0.pt").cpu()
    temp = abs(mat).numpy()
    plt.subplot(1,2,1)
    # plt.imshow(temp, cmap=plt.cm.gray)
    plt.imshow(temp)
    length = 64
    ft_mat = torch.fft.ifft2(mat)
    ft_mat = torch.fft.ifftshift(ft_mat)
    res = abs(ft_mat).numpy()
    plt.subplot(1,2,2)
    # plt.imshow(res, cmap=plt.cm.gray)
    plt.imshow(res)
    # print(res)
    plt.show()

def model(x, A, B, t1):
    return A - B * np.exp(- x / t1)

def eight_imgs_data(T1) -> np.ndarray:
    x = torch.Tensor([ 120,  320, 1120, 1320, 2120, 2320, 3120, 4120]).numpy()
    # name = 'E:\Study\毕业设计\MRI-simulation\(1100+1200j)\kspace'

    # name = 'E:\Study\毕业设计\MRI-simulation\data\(' + str(T1[0]) + '+' + str(T1[1]) + 'j)\\kspace'
    # name = 'E:\\Study\\flow_result\\' + str(T1[0]) + 'and' + str(T1[1]) + '\\kspace'
    name = 'E:\\kspace'
    li = []
    for key in range(len([5, 3])):
        if key == 0:
            for i in range(5):
                li.append(name + "TI5_" + str(i) + ".pt")
        else:
            for i in range(3):
                li.append(name + "TI3_" + str(i) + ".pt")
    data_li = []
    j = 0
    index = [0,5,1,6,2,7,3,4]
    # for path in li:
    for i in index:
        data = torch.load(li[i]).cpu()
        # print(li[i])
        ft_mat = torch.fft.ifft2(data)
        ft_mat = torch.fft.ifftshift(ft_mat)
        plt.subplot(2, 4, j + 1)
        plt.imshow(ft_mat.abs().numpy())
        # print(ft_mat)
        j += 1
        data_li.append(ft_mat.abs().numpy())
    data_list = np.array(data_li)
    plt.show()
    return x, data_list

def fit_one(T1_info):
    test_info = main_again.info(T1_generate=T1_info)
    x, data_list = eight_imgs_data(T1_info)
    # if (T1_info[0] == T1_info[1]):
    #     for p in range(len(data_list)):
    #         plt.subplot(2, 4, p + 1)
    #         plt.imshow(data_list[p])
    #     plt.show()
    li_vassel, li_muscle = image.get_point_index(test_info.length, test_info.bandwidth)
    accuracy_vassel, accuracy_muscle = [], []
    res_vassel, res_muscle = [], []
    for (i, j) in li_vassel:
        y = copy.deepcopy(data_list[:, i, j])
        y[0] *= -1
        y[1] *= -1
        param, param_cov = curve_fit(model, x, y, p0=[0.5,1,test_info.T1[0]], maxfev = int(1e8))
        res = param[2] * (param[1] / param[0] - 1)
        res_vassel.append(res)
        T1 = test_info.T1[0]
        accuracy = 1e2 * (res - T1) / T1
        # print(res, T1, accuracy)
        accuracy_vassel.append(accuracy)
    for (i, j) in li_muscle:
        y = copy.deepcopy(data_list[:, i, j])
        y[0] *= -1
        y[1] *= -1
        param, param_cov = curve_fit(model, x, y, p0=[0.5,1,test_info.T1[1]], maxfev = int(1e8))
        res = param[2] * (param[1] / param[0] - 1)
        res_muscle.append(res)
        T1 = test_info.T1[1]
        accuracy = 1e2 * (res - T1) / T1
        accuracy_muscle.append(accuracy)
    return res_vassel, res_muscle, accuracy_vassel, accuracy_muscle


def fit_8():
    
    
    # plt.scatter(x, y)
    T1_muscle = np.arange(400, 1800, 100)
    T1_vassel = np.arange(400, 1800, 100)
    # data = np.zeros((len(T1_vassel), len(T1_muscle)))
    # data = np.array([ 1 + 1j for _ in range(len(T1_vassel) * len(T1_vassel))]).reshape((len(T1_vassel), len(T1_muscle)))
    for m in range(len(T1_muscle)):
        path = "E:\\Study\\flow_result\\" + str(T1_vassel[m]) + ".csv"
        # vassel muscle accuracy_vassel accuracy_muscle
        # 固定 muscle 对不同 vassel
        zeros = np.zeros((6, len(T1_vassel)))
        for v in tqdm(range(len(T1_vassel))):
            
            T1_info = [T1_vassel[v], T1_muscle[m]]
            res_vassel, res_muscle, accuracy_vassel, accuracy_muscle = fit_one(T1_info)
    # df = pd.DataFrame(data, columns=T1_muscle, index=T1_vassel)
            
            # print(param[0], param[1], param[2], res, accuracy)
            zeros[0, v] = np.array(res_vassel).mean()
            zeros[1, v] = np.array(accuracy_vassel).mean()
            zeros[2, v] = np.array(res_vassel).std()

            zeros[3, v] = np.array(res_muscle).mean()
            zeros[4, v] = np.array(accuracy_muscle).mean()
            zeros[5, v] = np.array(res_muscle).std()
        df = pd.DataFrame(zeros, columns=T1_muscle, index=['vassel', 'accuracy_vassel','std_vassel', 'muscle', 'accuracy_muscle', 'std_muscle'])
        print(df)
        df.to_csv(path)
            
            # df.iloc[0, 0] = np.array(res_vassel).mean() + np.array(res_muscle).mean() * 1j
    df.to_csv("result.csv")
    # print((961 - 1000) / 1000)
    # x_temp = np.arange(x.min(), x.max(), 1)
    # y = model(x_temp, param[0], param[1], param[2])
    # plt.plot(x_temp, y)
    # plt.show()

def T1_contrast():
    data_list = eight_imgs_data()


# fit_8()
res_vassel, res_muscle, accuracy_vassel, accuracy_muscle = fit_one([800, 600])
print(np.array(res_vassel).mean(), np.array(accuracy_vassel).mean(), np.array(res_vassel).std())
print(np.array(res_muscle).mean(), np.array(accuracy_muscle).mean(), np.array(res_muscle).std())








y = torch.Tensor([-0.7816, -0.6536,  0.2293,  0.2763,  0.5942,  0.6116,  0.7318,  0.7828])
# x = torch.arange(0, 8, 1)
x = torch.Tensor([ 15.1000, 115.1000, 215.1000, 315.1000, 415.1000, 575.2000, 675.2000,
        775.2000])

A = torch.randn(1, requires_grad=True)
B = torch.randn(1, requires_grad=True)
# t1 = torch.ones(1, requires_grad=True)
t1 = torch.ones(1) * 200
t1.requires_grad_(True)
# print(t1)

def model(x, A, B, t1):
    return A - B * torch.exp(- x / t1)

def squared_loss(y_hat, y):  #@save
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):  #@save
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
            
import matplotlib.pyplot as plt

# annotation for regression

# lr = 0.1
# num_epochs = 300
# net = model
# loss = squared_loss
# # trainer = torch.optim.Adam([A, B, t1], lr=lr)
# trainer = sgd
# plt.ion()
# for epoch in range(num_epochs):
#     # for X, y in data_iter(batch_size, features, labels):
#     l = loss(net(x, A, B, t1), y)  # X和y的小批量损失
#     # trainer.zero_grad()
#     # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
#     # 并以此计算关于[w,b]的梯度
#     l.mean().backward()
#     # trainer.step()
#     # trainer([A, B, t1], lr=lr, batch_size=1)
#     # y_to_plot = net(x, A, B, t1)
     
#     sgd([A, B, t1], lr, batch_size=1)  # 使用参数的梯度更新参数
#     with torch.no_grad():
#         train_l = loss(net(x, A, B, t1), y)
#         print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}, A {float(A):f}, B {float(B):f}, t1 {float(t1):f}')
#         plt.clf()              # 清除之前画的图
#         plt.plot(x,net(x, A, B, t1).detach().numpy())        # 画出当前 ax 列表和 ay 列表中的值的图形
#         plt.scatter(x, y)
#         plt.pause(0.1)         # 暂停一秒
#         plt.ioff()   
		# plt.plot(x,net(x, A, B, t1).detach().numpy())        # 画出当前 ax 列表和 ay 列表中的值的图形
		# plt.scatter(x, y)
# plt.show()