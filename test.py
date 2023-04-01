#  打草稿 别在意
import matplotlib.pyplot as plt
import math

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

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from PIL import Image
mat = torch.load("E:\Study\毕业设计\MRI-simulation\kspace.pt").cpu()
temp = abs(mat).numpy()
plt.subplot(1,2,1)
plt.imshow(temp, cmap=plt.cm.gray)
# print(mat.shape)
import cv2

# tensor([[ 1.-5.5951e-06j],        [-1.-3.3139e-06j],        [ 1.-3.0360e-06j],        [-1.-5.8729e-06j],        [ 1.-4.7700e-07j],        [-1.+6.8269e-06j],        [ 1.+2.0820e-06j],        [-1.+4.2678e-06j],        [ 1.+4.6411e-06j],        [-1.+1.7088e-06j],        [ 1.+7.2001e-06j],        [-1.-1.6109e-05j],        [ 1.+9.7591e-06j],        [-1.-3.4093e-06j],        [ 1.-2.9406e-06j],        [-1.-5.9683e-06j],        [ 1.-3.8160e-07j],        [-1.+6.7315e-06j],        [ 1.-1.3081e-05j],        [-1.+4.1724e-06j],        [ 1.+4.7365e-06j],        [-1.+1.6134e-06j],        [ 1.-7.9633e-06j],        [-1.+1.4313e-05j],        [ 1.+2.2251e-06j],        [-1.+4.1247e-06j],        [ 1.+4.7842e-06j],        [-1.+1.5657e-06j],        [ 1.+7.3432e-06j],        [-1.-9.9332e-07j],        [ 1.+2.2728e-06j],        [-1.-3.5523e-06j],        [ 1.-2.7975e-06j],        [-1.+1.5180e-06j],        [ 1.-2.3850e-07j],        [-1.-1.0410e-06j],        [ 1.+2.3205e-06j],        [-1.-3.6000e-06j],        [ 1.-2.7498e-06j],        [-1.+1.4703e-06j],        [ 1.-1.9080e-07j],        [-1.+6.5407e-06j],        [ 1.+2.3682e-06j],        [-1.+3.9816e-06j],        [ 1.+1.1126e-06j],        [-1.+1.4226e-06j],        [ 1.-1.4310e-07j],        [-1.-1.1364e-06j],        [ 1.-1.3988e-06j],        [-1.+1.1925e-07j],        [ 1.+1.1603e-06j],        [-1.+1.3749e-06j],        [ 1.-9.5399e-08j],        [-1.-1.1841e-06j],        [ 1.+5.5628e-07j],        [-1.+7.1549e-08j],        [ 1.-6.9938e-07j],        [-1.-5.8013e-07j],        [ 1.-4.7700e-08j],        [-1.-2.7814e-07j],        [ 1.-3.4969e-07j],        [-1.+2.3850e-08j],        [ 1.-6.5168e-07j],        [-1.+3.2584e-07j],        [ 1.+0.0000e+00j],        [-1.-8.7423e-08j],        [ 1.+1.7485e-07j],        [-1.-2.3850e-08j],        [ 1.+3.4969e-07j],        [-1.+2.7814e-07j],        [ 1.+4.7700e-08j],        [-1.+5.8013e-07j],        [ 1.+6.9938e-07j],        [-1.-7.1549e-08j],        [ 1.-5.5628e-07j],        [-1.+1.1841e-06j],        [ 1.+9.5399e-08j],        [-1.+2.4398e-06j],        [ 1.-1.1603e-06j],        [-1.-1.1925e-07j],        [ 1.+1.3988e-06j],        [-1.+1.1364e-06j],        [ 1.+1.4310e-07j],        [-1.+2.3921e-06j],        [ 1.-1.1126e-06j],        [-1.+3.6477e-06j],        [ 1.-2.3682e-06j],        [-1.+1.0887e-06j],        [ 1.+1.9080e-07j],        [-1.-1.4703e-06j],        [ 1.+2.7498e-06j],        [-1.+3.6000e-06j],        [ 1.+5.3089e-06j],        [-1.+1.0410e-06j],        [ 1.+2.3850e-07j],        [-1.-1.5180e-06j],        [ 1.+2.7975e-06j],        [-1.+3.5523e-06j],        [ 1.-2.2728e-06j],        [-1.+9.9332e-07j],        [ 1.+2.8620e-07j],        [-1.-1.5657e-06j],        [ 1.-4.7842e-06j],        [-1.-4.1247e-06j],        [ 1.-2.2251e-06j],        [-1.+9.4562e-07j],        [ 1.-7.2955e-06j],        [-1.-1.6134e-06j],        [ 1.-4.7365e-06j],        [-1.-4.1724e-06j],        [ 1.-2.1774e-06j],        [-1.+8.5273e-06j],        [ 1.+3.8160e-07j],        [-1.+5.9683e-06j],        [ 1.+2.9406e-06j],        [-1.-1.1850e-05j],        [ 1.+5.4997e-06j],        [-1.+8.5022e-07j],        [ 1.-7.2001e-06j],        [-1.-1.7088e-06j],        [ 1.-4.6411e-06j],        [-1.+1.0991e-05j],        [ 1.-1.7341e-05j],        [-1.-6.8269e-06j],        [ 1.+4.7700e-07j],        [-1.+5.8729e-06j],        [ 1.-1.2223e-05j],        [-1.+1.8573e-05j]], device='cuda:0')
y = torch.ones(3,3)
x = torch.randn(3,3)
print(torch.mul(x, y))
# mat = torch.zeros((128, 128))
length = 128


ft_mat = torch.fft.fft2(mat)
ft_mat = torch.fft.fftshift(ft_mat)
# ft_mat = torch.fft.fftshift(ft_mat)
# # ft_mat = torch.fft.fftshift(mat)
res = abs(ft_mat).numpy()

# print(res.shape)
# lower, upper = length // 2 - 15, length//2 + 15
# mat[lower:upper, :] = 0
plt.subplot(1,2,2)
plt.imshow(res, cmap=plt.cm.gray)
# print(res)
plt.show()


# img = Image.fromarray(res)
# img = img.convert("L")
# img.show()


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