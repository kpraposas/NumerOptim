"""
Minimizer search of Rosenbrock function using nonlinear conjugate gradient
    method
"""

import numpy as np
from numoptim import nonlinearcg, options, param
from rosenbrock import rosenbrock, grad_rosenbrock
from time import time

if __name__ == "__main__":
    
    start_time = time()
    
    options["ssm"] = "backtrack"
    options["ssc"] = "armijo"
    methods = ["fr", "pr", "frpr", "prgn", "hs", "dy", "hz"]
    
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
    print("MINIMIZER SEARCH ON THE ROSENBROCK FUNCTION USING NONLINEAR " +
            "CONJUGATE GRADIENT METHOD WITH\n\tBACKTRACKING AS " +
            "LINE SEARCH METHOD AND ARMIJO RULE AS CRITERTION")
    print('-'*86)
    print("{}\t{}\t{}\t\t{}\t\t\t{}".format("", "NUMIT", "RELATIVE ERR",
                                                    "FUNVAL", "GRADNORM"))
    print('-'*86)
    for method in methods:
        options["beta"] = method
        result = nonlinearcg(rosenbrock, grad_rosenbrock,
                                    np.array(x_in), options, parameters)
        error = np.linalg.norm(result.x - np.ones(N))
        print("{}\t{}\t{}\t{}\t{}".format(method.upper(),
                                                result.it,
                                                error, result.fx,
                                                result.grad_norm))
    print('-'*86)
    
    stop_time = time()
    elapsed_time = stop_time - start_time
    print(f"This program ran for {elapsed_time} seconds.")