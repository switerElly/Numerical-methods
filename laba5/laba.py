import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def fibonacci(n):
    if n in (0, 1):
        return 1
    return fibonacci(n - 1) + fibonacci(n - 2)

def f(x, t):
    return np.exp(2*x+t) - np.exp(t**2) - 2*np.cos(x)

def x(t):
    return  fsolve(lambda x: f(x, t), 1)[0]

def fib_meth(f, a, b, n, eps=1e-6):
    x1, x2 = a + (b - a)*fibonacci(n-2)/fibonacci(n), a + (b - a)*fibonacci(n-1)/fibonacci(n)
    y1, y2 = f(x1), f(x2)
    for ind in range(1, n, -1):
        if y1 > y2:
            a = x1
            x1 = x2
            x2 = b - (x1 - a)
            y1 = y2
            y2 = f(x2)
        else:
            b = x2
            x2 = x1
            x1 = a + (b - x2)
            y2 = y1
            y1 = f(x1)
    return (x1 + x2)/2

t1, t2 =  0, 2
x1, x2 =  0, 2
t_points = np.linspace(t1, t2, 100)
x_points = np.linspace(x1, x2, 100)
X, T = np.meshgrid(x_points, t_points)
contour1 = plt.contour(T, X, f(X, T),[0], colors='b')
h1,_ = contour1.legend_elements()

eps = 10**(-6)
n = 0
while 2/fibonacci(n + 1) >= eps:
    n += 1
print(f'Accuracy achives with N = {n}: {2/fibonacci(31)} < {eps}')
print('-----------------')

t_min = fib_meth(x, t1, t2, n)
print(f"t_min = {t_min:.4f}")
print(f"x(t_min) = {x(t_min):.4f}")
print('-----------------')
t_max = [t1, t2][np.argmax([x(t1), x(t2)])]
print(f"t_max = {t_max:.4f}")
print(f"x(t_max) = {x(t_max):.4f}")
p1 = plt.scatter(t_max, x(t_max), c="r", s=49, zorder=10, label="$t_{max}$")
p2 = plt.scatter(t_min, x(t_min), c="g", s=49, zorder=10, label="$t_{min}$")
plt.legend([h1[0], p1, p2], ['$F(x,t)=0$', '$t_{max}$', "$t_{min}$"])
plt.xlabel("$t$")
plt.ylabel("$x$")
plt.show()