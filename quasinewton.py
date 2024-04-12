"""
Minimizer search of Rosenbrock function using Quasi Newton Methods
"""

import numpy as np
from numoptim import quasinewton, options, param
from rosenbrock import rosenbrock, grad_rosenbrock
from time import time

if __name__ == "__main__":
    
    start_time = time()
    options["ssm"] = "polyinterp"
    options["ssc"] = "wolfe"
    types = ["hess", "inv_hess"]
    updates = [
        ["sr1", "nsr1", "dfp", "bfgs", "spb", "bfgsdfp"],
        ["sr1", "nsr1", "dfp", "bfgs", "bfgsdfp", "gr", "ol", "ss",
                 "ho", "mh", "ssvm"]
        ]
    
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
    print("MINIMIZER SEARCH ON THE ROSENBROCK FUNCTION USING " +
            "QUASI_NEWTON's METHOD BOTH WITH HESSIAN AND INVERSE HESSIAN")
    i = 0
    for type in types:
        options["qn_type"] = type
        print('-'*86)
        print("{}\t{}\t{}\t\t{}\t\t\t{}".format("", "NUMIT", "RELATIVE ERR",
                                                    "FUNVAL", "GRADNORM"))
        print('-'*86)
        for update in updates[i]:
            options["qn_update"] = update
            method = f"{update}".upper()
            result = quasinewton(rosenbrock, grad_rosenbrock, np.array(x_in), 
                                    options, parameters)
            error = np.linalg.norm(result.x - np.ones(N))
            print("{}\t{}\t{}\t{}\t{}".format(method,
                                                    result.it,
                                                    error, result.fx,
                                                    result.grad_norm))
        print('-'*86 + "\n")
        i += 1
    
    stop_time = time()
    elapsed_time = stop_time - start_time
    print(f"This program ran for {elapsed_time} seconds.")