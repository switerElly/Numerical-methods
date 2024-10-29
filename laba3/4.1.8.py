import numpy as np
import matplotlib.pyplot as plt
import sympy as sym 
import scipy.optimize

def find_point(arr):
    x_1 = arr[0]
    x_2 = arr[1]
    x1, x2 = sym.symbols("x1, x2")
    y = sym.cos(x1 + x2) + 2*x2
    z = x1 + sym.sin(x2) - 0.6
    df = np.array([[float(sym.diff(y, x1).subs([(x1,x_1), (x2,x_2)])), float(sym.diff(y, x2).subs([(x1,x_1), (x2,x_2)]))], 
                  [float(sym.diff(z, x1).subs([(x1,x_1), (x2,x_2)])), float(sym.diff(z, x2).subs([(x1,x_1), (x2,x_2)]))]])
    f = np.array([float(y.subs([(x1,x_1), (x2,x_2)])), float(z.subs([(x1,x_1), (x2,x_2)]))])
    return np.array([x_1 + np.linalg.solve(df, -f)[0], x_2 + np.linalg.solve(df, -f)[1]])

def func_arr(x):
    return np.array([np.cos(x[0] + x[1]) + 2*x[1], x[0] + np.sin(x[1]) - 0.6])


x0 = [-3.48, 2.49]
eps = 10**(-6)
x_prev = np.array(x0)
x_real = find_point(x0)
x_new = np.array([])
count = 1

print(f'Start point: {x_prev}')
print('---------------')
while np.linalg.norm(np.linalg.norm(find_point(x_real) - x_real, 2)/(1 - np.linalg.norm(find_point(x_real) - x_real, 2)/np.linalg.norm(x_real - x_prev, 2))) >= eps:
    x_prev = x_real
    x_real = find_point(x_real)
    print(f'{count} approximation: {x_prev}')
    print('---------------')
    count += 1

print(f'Result of presented function:  {x_prev}')
print(f'Result of build-in function:   {scipy.optimize.fsolve(func_arr, np.array(x0))}')

print(f'Difference in solutions: {np.linalg.norm((x_prev - scipy.optimize.fsolve(func_arr, np.array(x0))).astype(np.float32))}')

x_1, x_2 = np.meshgrid(np.arange(-2, 2, 0.005), np.arange(-2, 2, 0.005))
fig, ax = plt.subplots(figsize=(6, 5))
contour1 = plt.contour(x_1, x_2, np.cos(x_1 + x_2) + 2*x_2, [0], colors=['blue'])
contour2 = plt.contour(x_1, x_2, x_1 + np.sin(x_2) - 0.6, [0], colors=['green'])
ax.scatter(*x_prev, c="#ff7f0e", alpha=1, zorder=2)
ax.scatter(*x_prev, c="#ff7f0e", alpha=1, zorder=2)
f1,_ = contour1.legend_elements()
f2,_ = contour2.legend_elements()
ax.legend([f1[0], f2[0]], ["cos(x_1 + x_2) + 2*x_2", "x_1 + sin(x_2) - 0.6"])
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
plt.show()