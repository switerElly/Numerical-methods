import numpy as np
import matplotlib.pyplot as plt

b = np.full(6, fill_value=8, dtype=float)

A = np.zeros((6, 6))
C = np.zeros((6, 6))
for i in range(6):
    for j in range(6):
        C[i, j] = 0.1 * 8 * (i + 1) * (j + 1)
        A[i, j] = 1 / (C[i,j]**2 + 0.58 * C[i, j])

x = np.linalg.solve(A, b)
cond_value = np.linalg.cond(np.abs(A), p=np.inf)

delta = 0.08
new_x = np.empty((6, 6))
for i in range(6):
    new_b = b.copy()
    new_b[i] += delta
    new_x[i] = np.linalg.solve(A, new_b)

d = []
for i in new_x:
    d.append(np.linalg.norm(x - i, ord=np.inf) / np.linalg.norm(x, ord=np.inf))

plt.figure(figsize=(6, 6))
plt.bar(range(1, 7), d)
plt.show()

new_b = b.copy()
new_b[np.argmax(d)] += delta

delta_b = (np.linalg.norm(new_b - b, ord=np.inf) / np.linalg.norm(b, ord=np.inf))
print(f'The greatest influence on the error has b_{np.argmax(d) + 1}')
print(f'Vector d = {d}')
print(f'The inequality delta(x^m) <= cond(A) * delta(b^m) is fulfilled: {d[np.argmax(d)]} <= {delta_b * cond_value}')