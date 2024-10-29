import numpy as np
import matplotlib.pyplot as plt

A = np.array([
    [3.5, -1, 0.9, 0.2, 0.1],
    [-1, 7.3, 2, 0.3, 2],
    [0.9, 2, 4.9, -0.1, 0.2],
    [0.2, 0.3, -0.1, 5, 1.2],
    [0.1, 2, 0.2, 1.2, 7]
])
b = np.array([1, 2, 3, 4, 5])

def zeid_mod(A, b, x, omega):
    B = np.empty((len(A), len(A)))
    c = np.empty((len(A),1))
    for i in range(len(A)):
        B[i] = -A[i]/A[i][i]
        c[i] = b[i]/A[i][i]
    B1 = np.tril(B, -1)
    B2 = np.triu(B, 1)

    x_new = np.zeros(len(A))
    for i in range(len(x)):
        x_h = np.zeros(len(A))
        x_new[i] = (1 - omega)*x[i] + omega*(np.dot(B1[i], np.hstack((x_new[:i], x_h[i:]))) + np.dot(B2[i], np.hstack((x_h[:i+1], x[i+1:]))) + c[i])
    return x_new

print(f"Is matrix A symmetrical: {np.all(A==A.T)}")
print(f"Is matrix A positive: {np.all(np.linalg.eigvals(A) > 0)}")
print("-----------")

x0 = [0, 0, 0, 0, 0]
print(f"Start approach: {x0}")
print("-----------")
eps = 10**(-5)
num_of_it = []
omega = [om for om in np.arange(0.2, 2, 0.2)]
for om in omega:
    x_prev = np.array(x0)
    x_real = zeid_mod(A, b, x0, 0.2)
    count = 0
    while np.linalg.norm(x_prev - x_real, np.inf) >= eps:
        count += 1
        x_prev = x_real
        x_real = zeid_mod(A, b, x_real, om)
    num_of_it.append(count)

print(f"Method Gauss solution:         {np.linalg.solve(A, b)}")
print(f"Solution of relaxation method: {x_real}")
print(f"Difference in solutions: {np.linalg.norm(np.linalg.solve(A, b) - x_real, np.inf)}")
print("-----------")
print(f"Relaxation parameter for minimum iterations: {omega[num_of_it.index(min(num_of_it))]}")
print(f"Number of iterations for omega: {num_of_it}")

plt.plot(omega, num_of_it)
plt.xlabel("Relaxation parameter")
plt.ylabel("Number of iterations")
plt.show()