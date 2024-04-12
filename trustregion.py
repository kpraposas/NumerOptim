"""
Minimizer search of Rosenbrock function using Newton Methods
"""

import numpy as np
from numoptim import dogleg, param
from rosenbrock import rosenbrock, grad_rosenbrock
from time import time

if __name__ == "__main__":
    
    start_time = time()
    
    parameters = param()
    parameters.tol = 1e-6
    parameters.maxit = 5*1e4
    parameters.iter_hist = False
    parameters.rho = 0.5
    parameters.alpha_in = 1.
    parameters.maxback = 30
    parameters.c1 = 1e-4
    parameters.abstol = 1e-12
    parameters.reltol = 1e-8
    
    N = 5
    x_in = np.zeros(N)
    # print("MINIMIZER SEARCH ON THE ROSENBROCK FUNCTION USING NEWTON's METHOD")
    # print('-'*94)
    # print("{}\t\t{}\t{}\t\t{}\t\t\t{}".format("", "NUMIT", "RELATIVE ERR",
    #                                                 "FUNVAL", "GRADNORM"))
    # print('-'*94)
    # for ver in versions:
    #     options["ver"] = ver
    #     result = unidir(rosenbrock, grad_rosenbrock, x_in, parameters)
    #     error = np.linalg.norm(result.x - np.ones(N))
    #     print("{:<8}\t{}\t{}\t{}\t{}".format(result.method.upper(),
    #                                             result.it,
    #                                             error, result.fx,
    #                                             result.grad_norm))
    # print('-'*94)
    print(dogleg(rosenbrock, grad_rosenbrock, x_in, parameters))
    
    stop_time = time()
    elapsed_time = stop_time - start_time
    print(f"This program ran for {elapsed_time} seconds.")