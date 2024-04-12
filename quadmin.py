"""
    Minimizer search sample for quadratic function
"""

import numpy as np
from numoptim import SD, CG, CGNR, CGR, param
def An(x, y):
    n = len(x)
    A = [[0]*n for _ in range(n)]
    for i in range(n):
        A[i][i] = y**i
    return A


if __name__ == "__main__":
    parameters = param()
    parameters.tol = 1e-6

    x_in = np.zeros(3)
    x = x_in[:]
    b = [-1 for _ in range(len(x_in))]

    for y in [1, 2, 5, 10, 20, 50]:
        print('-'*75)
        print('\u03B3 =', y, '\n')
        A = An(x_in, y)
        results = [
            SD(A, b, 0., x, parameters).__str__(),
            CG(A, b, 0., x, parameters).__str__(),
            CGNR(A, b, 0., x, parameters).__str__(),
            CGR(A, b, 0., x, parameters).__str__()
        ]
        print("\n\n".join(results))
        print('-'*75)