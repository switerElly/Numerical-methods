import matplotlib.pyplot as plt

analit_s = 8
S = {}
d = {}
M = dict()
for N in [10, 100, 1000, 10000, 100000]:
    s = 0
    M[N] = 0
    for n in range(N):
        s += 32/(n**2 + 9*n + 20)
    S[N] = s
    d[N] = abs(analit_s - s)
    num = 0
    while num > -20:
        if d[N] <= 10 ** num:
            M[N] += 1
            num -= 1
        else:
            break
    print(f'S({N}): {S[N]}, d({N}): {d[N]}, M({N}): {M[N]} \n-------------------')

bars = plt.bar(list(map(str, S.keys())), list(M.values()))
plt.show()