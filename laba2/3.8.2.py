import numpy as np
import matplotlib.pyplot as plt

def solve(A, b):
    if len(A.shape)!=2:
        raise Exception
    if len(b.shape)!=1:
        raise Exception
    if A.shape[0]!=b.shape[0]:
        raise Exception
    A = A.copy()
    b = b.copy()
    n = A.shape[0]
    b_order = []
    for i in range(n):
        max_value, max_i, max_j = abs(A[i, 0]), i, 0
        for j in range(i, n):
            for k in range(0, n):
                if abs(A[j, k])> max_value:
                    max_value, max_i, max_j = abs(A[j, k]), j, k
        b_order.append(max_j)
        if i!=max_i:
            tmp = A[i].copy()
            A[i] = A[max_i]
            A[max_i] = tmp
            tmp = b[i]
            b[i] = b[max_i]
            b[max_i] = tmp
        for k in range(i + 1, n):
            b[k] -= b[i] * A[k,max_j]/A[i,max_j]
            A[k] -= A[i] * A[k,max_j]/A[i,max_j]
    result = b.copy()
    for i in range(n-1, -1, -1):
        j = b_order[i]
        b[i]/=A[i,j]
        result[j] = b[i]
        for k in range(0, i):
            b[k] -= b[i] * A[k,j]
    return result

def find_b(x):
    b = np.zeros(40)
    for i in range(40):
        b[i] = np.linalg.norm(x - 40/10) * i * np.sin(x)
    return b


x = np.linspace(-5, 5, 1000)

A = np.zeros((40, 40))
q = 1.001 - 2*2*10**(-3)
for i in range(40):
    for j in range(40):
        if i == j:
            A[i, j] = (q - 1)**(i+j)
        else:
            A[i, j] = q**(i+j) + 0.1*(j - i)

b = np.zeros(40)
y = []
for x_i in x:
    y.append(np.sum(solve(A, find_b(x_i))))

plt.plot(x, y)
plt.show()