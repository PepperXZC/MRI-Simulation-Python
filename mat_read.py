import numpy as np
import torch
import matplotlib.pyplot as plt

def read_mat(path):
    f = open(path)
    lines = f.readlines()
    x, y = 0, 0
    mat = torch.zeros((128, 128))
    for l in lines:
        le = l.strip('\n').split(' ')
        for s in le:
            if len(s) != 0:
                mat[x, y] = float(s)
                y += 1
        x += 1
        y = 0
    return mat

for i in range(8):
    r_path, i_path = "Molli/real_mat_" +  str(i) + ".txt", "Molli/img_mat_" +  str(i) + ".txt"
# r_path, i_path = "Molli/real_mat_1.txt", "Molli/img_mat_1.txt"
    real, img = read_mat(r_path), read_mat(i_path)
    mat = torch.complex(real, img)
    res = torch.fft.ifft2(mat)
    res = torch.fft.ifftshift(res).abs().numpy()
    plt.subplot(2, 4, i + 1)
    plt.imshow(res)
    plt.colorbar()
plt.show()