import numpy as np

true_coefficients = np.array([1, 27.4, 187.65])
errors = [0.1, 10**-2, 10**-3, 10**-4, 10**-5, 10**-6, 10**-7, 10**-8, 10**-9, 10**-10, 10**-11, 10**-12, 10**-13, 10**-14, 10**-15]
for err in errors:
    actual_coefficients = true_coefficients + np.array([0, err, 0])
    print('error:', err)
    print("x1={:.5f}, x2={:.5f}".format(*np.roots(actual_coefficients)))