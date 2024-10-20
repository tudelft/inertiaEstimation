import numpy as np
import math

m = 3
def derivativeCoefficients(n, f):
    T = np.zeros(n * n).reshape(n, n)
    res = np.zeros(n)
    res[1] = 1

    for y in range(n):
        for x in range(n):
            if y == 0:
                T[y, x] = 1
            elif x == 0:
                T[y, x] = 0
            else:
                T[y, x] = (-x) ** y / math.factorial(y)
    res = np.flip(np.linalg.solve(T, res))

    deriv_coefs_kernel = np.zeros(len(res) * f)
    for i, c in enumerate(res):
        deriv_coefs_kernel[i * f] = c
    return deriv_coefs_kernel * m / ((m + 1) * f)