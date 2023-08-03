"""
Steepest Descent Method with Armijo Rule and Backtracking
"""

import numpy as np
import numoptim
from numoptim import steepest_descent

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
    n = len(x)
    s = 0
    for k in range(n - 1):
        s = s + 100*(x[k+1] - x[k]**2)**2  + (1 - x[k])**2
    return s

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
    grad = []
    grad.append(-400.*(x[1]-x[0]**2)*x[0] - 2.*(1-x[0]))
    for k in range(1, n - 1):
        grad.append(200.*(x[k]-x[k-1]**2) - 400.*(x[k+1]-x[k]**2)*x[k] - 2.*(1-x[k]))
    grad.append(200.*(x[n-1] - x[n-2]**2))
    return np.array(grad)

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
    hess = [[0]* n for i in range(n)]
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
    
    options = numoptim.options
    methods = ["backtrack", "polyinterp"]
    criteria = ["armijo", "goldstein", "wolfe", "strongwolfe"]
    
    parameters = numoptim.param()
    parameters.tol = 1e-6
    parameters.maxit = 5*1e4
    parameters.iter_hist = False
    parameters.rho_r = 0.5
    
    N = 100
    x_in = np.zeros(N)
    for method in methods:
        for criterion in criteria:
            options["ssm"], options["ssc"] = method, criterion
            x, funval, grad_norm, it, num_feval, num_geval, num_heval, term_flag, elapsed_time = \
                steepest_descent(rosenbrock, grad_rosenbrock, np.array(x_in), options, parameters)
            print('-'*72)
            print(f"METHOD                          : Steepest Descent")
            print("STEPSIZE SELECTION METHOD       :", options["ssm"].capitalize())
            print("STEPSIZE SELECTION CRITERION    :", options["ssc"].capitalize())
            print(f"INITIAL ITERATE                 : \n{x_in}")
            print(f"APPROXIMATE MINIMIZER           : \n{x}")
            print(f"FUNCTION VALUE                  : {funval}")
            print(f"GRADIENT NORM                   : {grad_norm}")
            print(f"TOLERANCE                       : {parameters.tol}")
            print(f"NITERS                          : {it}\t\tMAXIT: {parameters.maxit}")
            print(f"TERMINATION                     : {term_flag}")
            print(f"NFEVALS                         : {num_feval}")
            print(f"NGEVALS                         : {num_geval}")
            print(f"NHEVALS                         : {num_heval}")
            print(f"ELAPSED TIME                    : {elapsed_time} seconds")
            print('-'*72, "\n")
        