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
cond_value = np.linalg.cond(A, p=np.inf)

delta = 0.08
new_x = {}
for i in range(6):
    for j in range(6):
        new_A = A.copy()
        new_A[i, j] += delta
        new_x[(i, j)] = np.linalg.solve(new_A, b)

d = {key: np.linalg.norm(x - x_i, ord=np.inf) / np.linalg.norm(x, ord=np.inf) for key, x_i in new_x.items()}

plt.figure(figsize=(10, 5))
plt.bar([str(i) for i in d.keys()], d.values())
plt.xticks(rotation=90)
plt.show()

d_i, d_j = max(d, key=d.get)
new_A = A.copy()
new_A[d_i, d_j] += delta

delta_A = (np.linalg.norm(new_A - A, ord=np.inf) / np.linalg.norm(A, ord=np.inf))
print(f'The greatest influence on the error has element on position: {d_i, d_j}')
print(f'The inequality delta(x^*) <= cond(A) * delta(A^*) is fulfilled: {d[(d_i, d_j)]} <= {delta_A * cond_value}')