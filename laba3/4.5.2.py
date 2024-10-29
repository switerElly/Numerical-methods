import numpy as np
import matplotlib.pyplot as plt

N = 2
a1 = 8.5 - N * 0.25
a2 = 2.3 + N * 0.3
a3 = 4 + N * 0.1
P1 = np.array([16, 5.8, 11.879])
P2 = np.array([8.485, 5.328, 8.91])
P3 = np.array([15, 3.139, 5.25])

def coordinates(q):
    phi = q[0]
    theta = q[1]
    return np.array([a1*np.sin(phi)*np.sin(theta), a2*np.cos(phi)*np.sin(theta), a3*np.cos(phi)])

def f(point):
    def func(q):
        return np.sum((coordinates(q) - point) ** 2)
    return func

def gradient(f, q, h = 1e-8):
    n = len(q)
    grad = np.empty_like(q)
    for i in range(n):
        delta = np.zeros_like(q)
        delta[i] = h
        grad[i] = (f(q + delta) - f(q-delta)) / 2 / h
    return grad

def hesse(f, q, h = 1e-8):
    n = len(q)
    hesse_matrix = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        delta_i = np.zeros_like(q)
        delta_i[i] = h
        for j in range(n):
            delta_j = np.zeros_like(q)
            delta_j[j] = h
            hesse_matrix[i, j] = (f(q + delta_i + delta_j) - f(q + delta_i - delta_j) - 
            f(q - delta_i + delta_j) + f(q - delta_i - delta_j))/(4 * h * h)
    return hesse_matrix

def p(x2, x1, x0):
    norm_x2_x1 = np.max(np.abs(x2 - x1))
    norm_x1_x0 = np.max(np.abs(x1 - x0))
    return np.abs(norm_x2_x1 / (1 - norm_x2_x1/ norm_x1_x0))


def newton_method(f, x0,  eps = 1e-6):
    x1 = x0 + np.linalg.solve(hesse(f, x0), -gradient(f, x0))
    x2 = x1 + np.linalg.solve(hesse(f, x1), -gradient(f, x1))
    iter_cnt = 2
    while p(x2, x1, x0) > eps:
        x0 = x1
        x1 = x2
        try:
            x2 = x2 + np.linalg.solve(hesse(f, x2), -gradient(f, x2))
        except np.linalg.LinAlgError:
            return x2, iter_cnt
        iter_cnt += 1
    return x2, iter_cnt

def optimal(point, q0):
    func = f(point)
    q1,_ = newton_method(func, q0)
    dist = np.sqrt(func(q1))
    return dist

d1, d2, d3 = optimal(P1, [np.pi/6, np.pi/4]), optimal(P2, [np.pi/3, np.pi/4]), optimal(P3, [np.pi/4, np.pi/5])

print(f"Point: {P1}")
print(f"Distance to P1 = {d1}")
print("-------------")
print(f"Point: {P2}")
print(f"Distance to P2 = {d2}")
print("-------------")
print(f"Point: {P3}")
print(f"Distance to P3 = {d3}")
print("-------------")
print(f"The closest point is P{np.argmin([d1, d2, d3]) + 1}")
print(f"The furthest point is P{np.argmax([d1, d2, d3]) + 1}")

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection='3d')

phi = np.linspace(0, 3 * np.pi, 100)
theta = np.linspace(0, 3 * np.pi, 100)
x = a1 * np.outer(np.cos(theta), np.sin(phi))
y = a2 * np.outer(np.sin(theta), np.sin(phi))
z = a3 * np.outer(np.ones_like(theta), np.cos(phi))

ax.plot_surface(x, y, z,  rstride=4, cstride=4, alpha = 0.5)
ax.scatter(*P1, marker='*', label = "P1")
ax.scatter(*P2, marker='*', label = "P2")
ax.scatter(*P3, marker='*', label = "P3")
ax.view_init(30, 135, 0)
ax.set_xlim(-14, 14)
ax.set_ylim(-14, 14)
ax.set_zlim(-2, 14)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.legend(loc="upper left")
plt.show()