"""
Minimizer search of Rosenbrock function using Barzilai Borwein Methods
"""

import numpy as np
from numoptim import barzilaiborwein, options, param
from rosenbrock import rosenbrock, grad_rosenbrock
from time import time

if __name__ == "__main__":
    
    start_time = time()
    options["ssm"] = "polyinterp"
    options["ssc"] = "armijo"
    versions = ["v1", "v2", "v3", "v4"]
    
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
    
    N = 100
    x_in = np.zeros(N)
    print("MINIMIZER SEARCH ON THE ROSENBROCK FUNCTION USING " +
            "BARZILAI-BORWEIN's METHOD")
    print('-'*94)
    print("{}\t\t{}\t{}\t\t{}\t\t\t{}".format("", "NUMIT", "RELATIVE ERR",
                                                    "FUNVAL", "GRADNORM"))
    print('-'*94)
    for ver in versions:
        options["alpha"] = ver
        method = f"Version {ver[1]}"
        result = barzilaiborwein(rosenbrock, grad_rosenbrock, np.array(x_in), 
                                 options, parameters)
        error = np.linalg.norm(result.x - np.ones(N))
        print("{}\t{}\t{}\t{}\t{}".format(method,
                                                result.it,
                                                error, result.fx,
                                                result.grad_norm))
    print('-'*94)
    
    stop_time = time()
    elapsed_time = stop_time - start_time
    print(f"This program ran for {elapsed_time} seconds.")