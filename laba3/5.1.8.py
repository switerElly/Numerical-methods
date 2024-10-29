import numpy as np

A = np.array([
    [118.8, -14, -5, -89.1],
    [-59.4, 194, 5, 128.7],
    [148.5, 12, -310, 148.5],
    [0, 18.5, 90, -108.9]
])
b = np.array([-92.5, -340.1, -898, 184.1])

def zeid(A, b, x):
    B = np.empty((len(A),len(A)))
    c = np.empty((len(A),1))
    for i in range(len(A)):
        B[i] = -A[i]/A[i][i]
        c[i] = b[i]/A[i][i]
    B1 = np.tril(B, -1)
    B2 = np.triu(B, 1)

    x_new = np.zeros(len(A))
    for i in range(len(x)):
        x_new[i] = np.dot(B1[i], x_new) + np.dot(B2[i], x) + c[i]
    return x_new

gauss = np.linalg.solve(A, b)

B = np.empty((len(A),len(A)))
c = np.empty((len(A),1))
for i in range(len(A)):
    B[i] = -A[i]/A[i][i]
    c[i] = b[i]/A[i][i]
B1 = np.tril(B, -1)
B2 = np.triu(B, 1)
print(f"Sufficient convergence condition is completed: {np.linalg.norm(B2, np.inf)} < 1")
print(f"------------------")

eps = np.linalg.norm(((1 - np.linalg.norm(B, np.inf))/np.linalg.norm(B2, np.inf)))*10**(-6)

num = 0
x = [0.6, 2.3, -2.8, -1.35]
print(f"Start approach: {x}")
while num <= 10:
    x = zeid(A, b, x)
    num += 1
print(f"Method Gauss solution:     {gauss}")
print(f"Solution of zeid function (10 iterations): {x}")
print(f"Difference in solution: {np.linalg.norm(gauss - x, np.inf)}")
print(f"------------------")

print(f"Start approach: {x}")
x_prev = np.array(x)
x_real = zeid(A, b, x)
count = 0
while np.linalg.norm(x_prev - x_real, np.inf) >= eps:
    count += 1
    x_prev = x_real
    x_real = zeid(A, b, x_real)
print(f"Method Gauss solution:                          {gauss}")
print(f"Solution of zeid function (epsilon with norm) : {x_real} was reached by {count} iterations")
print(f"Difference in solution: {np.linalg.norm(gauss - x_real, np.inf)}")
print(f"------------------")

num = 0
x = [10, 10, 10, 10]
print(f"Start approach: {x}")
while num <= 10:
    x = zeid(A, b, x)
    num += 1
print(f"Method Gauss solution:     {gauss}")
print(f"Solution of zeid function: {x}")
print(f"Difference in solution: {np.linalg.norm(gauss - x, np.inf)}")
print(f"------------------")

x = [10, 10, 10, 10]
print(f"Start approach: {x}")
x_prev = np.array(x)
x_real = zeid(A, b, x)
count = 0
while np.linalg.norm(x_prev - x_real, np.inf) >= eps:
    count += 1
    x_prev = x_real
    x_real = zeid(A, b, x_real)
print(f"Method Gauss solution:                          {gauss}")
print(f"Solution of zeid function (epsilon with norm) : {x_real} was reached by {count} iterations")
print(f"Difference in solution: {np.linalg.norm(gauss - x_real, np.inf)}")
print(f"------------------")