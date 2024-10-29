import numpy as np

A = np.array([[30, 34, 19],
            [31.4, 35.4, 20],
            [24, 28, 13]], dtype=float)
alpha = 0.05
beta = 0.1

print(f'det = {np.linalg.det(A)} => A is invertible\n------------')

mn1 = np.inf
mx1 = -np.inf
for i in range(512):
    new_A = A * np.array([1+(1-2*int(j))*alpha/100 for j in bin(i)[2:].zfill(9)]).reshape(3,3)
    new_det = np.linalg.det(new_A)
    mn1 = min(new_det, mn1)
    mx1 = max(new_det, mx1)

print(f'alpha = {alpha} \n{mn1} <= new det <= {mx1}')
if mn1 <= 0 and 0 <= mx1:
    print('matr is not invertible')
else:
    print('matr is invertible')
print(f'relative error in det = {(abs(mn1-np.linalg.det(A)) + abs(mx1-np.linalg.det(A)))/(2*abs(np.linalg.det(A)))}\n------------')

mn2 = np.inf
mx2 = -np.inf
for i in range(512):
    new_A = A * np.array([1+(1-2*int(j))*beta/100 for j in bin(i)[2:].zfill(9)]).reshape(3,3)
    new_det = np.linalg.det(new_A)
    mn2 = min(new_det, mn2)
    mx2 = max(new_det, mx2)
print(f'beta = {beta} \n{mn2} <= new det <= {mx2}')
if mn2 <= 0 and 0 <= mx2:
    print('matr is not invertible')
else:
    print('matr is invertible')
print(f'relative error in det = {(abs(mn2-np.linalg.det(A)) + abs(mx2-np.linalg.det(A)))/(2*abs(np.linalg.det(A)))}\n------------')