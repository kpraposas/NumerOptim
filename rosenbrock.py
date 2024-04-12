"""
Steepest Descent Method with Armijo Rule and Backtracking
"""

import numpy as np
from numoptim import steepest_descent, options, param
from time import time

def rosenbrock(x):
    """
    Parameter
    ---------
        x : list
            input n-dimensional vector
            
    Returns
    -------
        float : value of the Rosenbrock function at x
    """
    x = np.array(x)
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def grad_rosenbrock(x):
    """
    Parameter
    ---------
        x : list
            input n-dimensional vector
            
    Returns
    -------
        list : value of the gradient of Rosenbrock function at x
    """
    n = len(x)
    grad = np.zeros(n)

    grad[0] = -400. * (x[1] - x[0] ** 2) * x[0] - 2. * (1 - x[0])
    grad[1:-1] = 200. * (x[1:-1] - x[:-2] ** 2)\
        - 400. * (x[2:] - x[1:-1] ** 2) * x[1:-1] - 2. * (1 - x[1:-1])
    grad[-1] = 200. * (x[-1] - x[-2] ** 2)

    return grad

def hess_rosenbrock(x):
    """
    Parameter
    ---------
        x : list
            input 2d vector
            
    Returns
    -------
    list : value of the Hessian of the Rosenbrock function at x
    """
    n = len(x)
    hess = np.zeros((n, n))
    hess[0][0] = -400.*x[1] + 1200.*x[0]**2 + 2
    hess[0][1] = -400.*x[0]
    for k in range(1, n - 1):
        hess[k][k-1] = -400.*x[k-1]
        hess[k][k] = 202 - 400.*x[k+1] + 1200.*x[k]**2
        hess[k][k+1] = -400.*x[k]
    hess[n-1][n-2] = -400.*x[n-2]
    hess[n-1][n-1] = 200.
    return np.array(hess)
    
if __name__ == "__main__":
    
    start_time = time()
    methods = ["backtrack", "polyinterp"]
    criteria = ["armijo", "goldstein", "wolfe", "strongwolfe"]
    
    parameters = param()
    parameters.tol = 1e-6
    parameters.maxit = 5*1e4
    parameters.iter_hist = False
    parameters.rho = 0.5
    parameters.rho_r = 0.5
    parameters.alpha_in = 1.
    parameters.maxback = 30
    parameters.c1 = 1e-4
    parameters.abstol = 1e-12
    parameters.reltol = 1e-8
    
    N = 100
    x_in = np.zeros(N)
    
    print("MINIMIZER SEARCH ON THE ROSENBROCK FUNCTION USING STEEPEST DESCENT")
    print('-'*109)
    print("{}\t\t{}\t\t{}\t{}\t\t{}\t\t\t{}".format("SSM", "SSC", "NUMIT",
                                              "RELATIVE ERR", "FUNVAL",
                                              "GRADNORM"))
    print('-'*109)
    for method in methods:
        for criterion in criteria:
            options["ssm"], options["ssc"] = method, criterion
            result = steepest_descent(rosenbrock, grad_rosenbrock,
                                      np.array(x_in), options, parameters)
            error = np.linalg.norm(result.x - np.ones(N))
            print("{}\t{:<8}\t{}\t{}\t{}\t{}".format(method.capitalize(),
                                                  criterion.capitalize(),
                                                  result.it,
                                                  error, result.fx,
                                                  result.grad_norm))
    print('-'*109)
    
    stop_time = time()
    elapsed_time = stop_time - start_time
    print(f"This program ran for {elapsed_time} seconds.")