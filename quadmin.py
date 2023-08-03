"""
    Minimizer search sample for quadratic function
"""

import numpy as np
from numoptim import quadminsearch
import numoptim

def An(x, y):
    n = len(x)
    A = [[0]*n for _ in range(n)]
    for i in range(n):
        A[i][i] = y**i
    return A


if __name__ == "__main__":
    parameters = numoptim.param()
    parameters.tol = 1e-6
    methods = ['sd', 'cg', 'cgnr', 'cgr']

    x_in = np.zeros(3)
    x = x_in[:]
    b = [-1 for _ in range(len(x_in))]

    for y in [1, 2, 5, 10, 20, 50]:
        print('-'*75)
        print('\u03B3 =', y, '\n')
        for method in methods:
            A = An(x_in, y)
            x, funval, grad_norm, k = quadminsearch(A, b, 0, x, method, parameters)
            print("Method                  :", method.upper())
            print(f"Initial Iterate         : {x_in}")
            print(f"Approximate minimizer   : {x}")
            print(f"Function Value          : {funval}")
            print(f"Gradient Norm           : {grad_norm}")
            print(f"Number of Iterations    : {k}")
            print(f"Condition Number        : {y**2}")
            if method == "sd":
                x = [-1, -y, -y**2]
                eps = np.finfo(float).eps
                k = np.ceil((np.log(parameters.tol) - (np.log(y * np.sqrt(1+y**2+y**4)))) / (np.log(y**2-1+eps) - np.log(y**2+1)))
                print(f"Required Iterations     : {k}")
            print("")
            x = x_in[:]
        print('-'*75)