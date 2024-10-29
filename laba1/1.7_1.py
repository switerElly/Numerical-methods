import numpy as np

types =  {
    'float': np.single,
    'double': np.double,
    'long double': np.longdouble
    }
for t in types:
    n_t = types[t]
    с = 0
    n = n_t(1)
    while n != np.inf:
        n = (n*2).astype(n_t)
        с += 1
    print(f'Infinity for {t} is 2^{с}')
    с = 0
    n = n_t(1)
    while n != 0:
        n = (n/2).astype(n_t)
        с += 1
    print(f'Zero for {t} is 2^-{с}')
    с = 0
    n = n_t(1)
    while n_t(1) + n > n_t(1):
        n = (n/2).astype(n_t)
        с += 1
    print(f'Epsilon for {t} is 2^-{с} \n')
