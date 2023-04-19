import numpy as np

f = open("mat.txt")
lines = f.readlines()
for l in lines:
    le = l.strip('\n').split(' ')
    
columns = len(le)-1

A = np.zeros((lines, columns), dtype=float)
A_row = 0
for lin in lines:
    list = lin.strip('\n').split(' ')
    A[A_row:] = list[0:columns]
    A_row += 1
print(A)
