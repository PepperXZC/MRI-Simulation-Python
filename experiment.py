# 基于256*256*3的矩阵进行逐点弛豫变化操作
# 如果是每个时刻t的点，应该是很难算。。而且目前也不考虑 Gradient spoil
# 那么目前就只考虑做一个【具有广播机制的】矩阵
import torch
import matrix_rot
import torch
import freprecess
import random
import math
import sequence
import matplotlib.pyplot as plt
from torch import nn
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class pool:
    def __init__(self, info) -> None:
        self.info = info
        self.pool = torch.zeros(self.info.fov, self.info.fov)
        self.vassel = torch.zeros(self.info.fov, self.info.bandwidth)
        # 设定随机阈值
        self.half = 9
    
    
    def roll(self, t):
        # 规定：每1ms，血流数组滚动1个单元
        each_time = self.info.real_length / self.info.roll_rate
        rest_time = t
        now_time = 0
        # 各管各的变化。先变化pool：
        self.pool += t
        # 然后只管 vassel：
        if rest_time - each_time < 0:
            self.vassel += rest_time
        while (rest_time - each_time >= 0):
            self.vassel += each_time
            rest_time -= each_time
            # 默认向下流动
            self.vassel = torch.roll(self.vassel, 1, 0)
            for i in range(len(self.vassel[0])):
                # if random.randint(1, 10) < self.half:
                self.vassel[0][i] = 0
            # print(self.vassel)
        
        a, b = int(self.info.fov) // 2, int(self.info.bandwidth) // 2
        lower, upper = a - b, a + b
        # self.pool[:, lower:upper+1] = self.vassel

class Model(nn.Module):
    def __init__(self, shape) -> None:
        super(Model, self).__init__()
        self.A = nn.Parameter(torch.ones(shape, requires_grad=True))
        self.B = nn.Parameter(torch.ones(shape, requires_grad=True))
        self.t1 = nn.Parameter(torch.ones(shape) * 2000)
        self.t1.requires_grad_(True)
        # self.t1 = nn.Parameter(torch.normal(0, 0.01, size=shape, requires_grad=True))
    
    def forward(self, t):
        # return torch.stack([(self.A - self.B * torch.exp(- t / self.t1)) for t in t_sequence])
        return self.A - self.B * torch.exp(- t / self.t1)


def model_t(t, A, B, t1):
    return A - B * torch.exp(-t / t1)

def train(num_epochs = 10000000, lr = 0.01, data = None, y_true = None, info = None):
    # 训练，顺带看看训练效果
    X, y = data, y_true
    # 以y_true中的[0][0]点
    # plot_true_y = y_true[:, 0, 0]
    plot_true_y = y_true
    print(plot_true_y)
    # model = Model(shape=(info.fov, info.fov))
    model = Model(shape=1)
    model.to(device)
    loss = nn.MSELoss(reduction='none')
    trainer = torch.optim.Adam(model.parameters(), lr=lr)

    plt.ion()
    for epoch in range(num_epochs):
        y_pred = model(X)
        y_to_plot = y_pred.detach().numpy()
        y_true_plot = model_t(X, model.A, model.B, 2000).detach().numpy()
        # print(y_pred.shape)
        # y_pred.requires_grad_(True)
        # model.A.requires_grad_(True)
        # model.B.requires_grad_(True)
        # model.t1.requires_grad_(True)
        l = loss(y_pred, y)
        # l = torch.mean(l, dim=0)
        # l = l.mean()
        trainer.zero_grad()
        l.sum().backward()
        trainer.step()
        plt.clf()              # 清除之前画的图
        plt.scatter(X,plot_true_y)        # 画出当前 ax 列表和 ay 列表中的值的图形
        plt.plot(X,y_to_plot, color='b')
        plt.plot(X, y_true_plot, color='r')
        # plt.plot(X,y_to_plot[:, 2, 2])
        # plt.plot(X,y_to_plot[:, 4, 4])
        plt.pause(0.1)         # 暂停一秒
        plt.ioff()
        if (epoch + 1) % 100 == 0:
            # print(epoch + 1, l[0][0][0])
            # print(f'epoch {epoch + 1}, loss {float(l.sum()):f}, A {float(model.A[0][0]):f}, B {float(model.B[0][0]):f}, t1star {float(model.t1[0][0]):f}, t1[0][0][0] {float(model.t1[0][0] * ((model.B[0][0] / model.A[0][0]) - 1)):f}')
            print(f'epoch {epoch + 1}, loss {float(l.sum()):f}, A {float(model.A):f}, B {float(model.B):f}, t1star {float(model.t1):f}, t1 {float(model.t1 * ((model.B / model.A) - 1)):f}')
    # T1 = model.t1 * ((model.B / model.A) - 1)
    # print(T1)