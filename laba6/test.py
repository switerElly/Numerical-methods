import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

A = np.array([[-0.717, -23.827], [114.483, -640.393]], dtype=np.double)
Y0 = np.array([1., 2.], dtype=np.double)
B = np.array([[-1.905, -0.015], [-0.13, -2.295]], dtype=np.double)
Z0 = np.array([1., 0.], dtype=np.double)

def eyler(f, y0, t0, T, h=0.1):
    t = np.arange(t0, T + h /2, h)
    y = np.empty((len(t),  y0.shape[0] if isinstance(y0, np.ndarray) else 1), dtype=np.double)
    y[0, :] = y0
    for i, t_i in enumerate(t[:-1]):
        y[i + 1, :] = y[i, :] + h * f(y[i, :])
    return t, y

def implicit_eyler(f, y0, t0, T, h=0.1):
    t = np.arange(t0, T + h /2, h)
    y = np.empty((len(t),  y0.shape[0] if isinstance(y0, np.ndarray) else 1), dtype=np.double)
    y[0, :] = y0
    for i, t_i in enumerate(t[:-1]):
        y[i + 1, :] = fsolve(lambda yn: (yn - y[i, :])/h - f(y[i, :])/2 -f(yn)/2, y[i, :])
    return t, y

def f1(y):
    return A @ y

def f2(y):
    return B @ y

t0 = 0
T = 1
h = 0.01
t_euler_A, y_euler_A = eyler(f1, Y0, t0, T, h)
t_euler_B, y_euler_B = eyler(f2, Z0, t0, T, h)
plt.plot(t_euler_A, y_euler_A[:, 0], label = "$y_1$")
plt.plot(t_euler_A, y_euler_A[:, 1], label = "$y_2$")
plt.ylim(-10, 10)
plt.legend()
plt.xlabel("t")
plt.ylabel("y")
plt.show()
plt.plot(t_euler_B, y_euler_B)
plt.plot(t_euler_B, y_euler_B[:, 0], label = "$z_1$")
plt.plot(t_euler_B, y_euler_B[:, 1], label = "$z_2$")
plt.ylim(-10, 10)
plt.legend()
plt.xlabel("t")
plt.ylabel("z")
plt.show()
l_A = np.linalg.eigvals(A)
l_B = np.linalg.eigvals(B)
with np.printoptions(5):
    print("Eigenvalues of A :", l_A)
    a_stiff = np.max(np.abs(l_A))/np.min(np.abs(l_A))
    print(f"Stiffness coef of A : {a_stiff:.5f}")
    print('---------------------')
    print("Eigenvalues of B :", l_B)
    b_stiff = np.max(np.abs(l_B))/np.min(np.abs(l_B))
    print(f"Stiffness coef of B : {b_stiff:.5f}")
print('---------------------')
print("A is stiffer" if a_stiff > b_stiff else "B is stiffer")
print(f"h* theoretical: {2/np.max(np.abs(l_A)):.5f}")
print(f"h visial: {h/3.7:.5f}")
t_ieuler_A, y_ieuler_A = implicit_eyler(f1, Y0, t0, T, h)
plt.plot(t_ieuler_A, y_ieuler_A[:, 0], label = "$y_1$")
plt.plot(t_ieuler_A, y_ieuler_A[:, 1], label = "$y_2$")
plt.legend()
plt.xlabel("t")
plt.ylabel("y")
plt.show()
t_euler_A, y_euler_A = eyler(f1, Y0, t0, T, h/3.7)
plt.plot(t_euler_A, y_euler_A[:, 0], label = "$y_1$")
plt.plot(t_euler_A, y_euler_A[:, 1], label = "$y_2$")
plt.legend()
plt.xlabel("t")
plt.ylabel("y")
plt.show()