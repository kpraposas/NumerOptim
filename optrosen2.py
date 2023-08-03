"""
Minimizer search of Rosenbrock function using nonlinear cg, quasi-newton and trust region
"""

import numpy as np
import numoptim
from numoptim import steepest_descent, nonlinear_cg, barzilai_borwein, newton, quasi_newton, trust_region

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
    options["ssm"] = "backtrack"
    options["ssc"] = "armijo"
    options["qn_type"] = "inv_hess"
    options["qn_update"] = "bfgs"
    options["beta"] = "hz"
    options["alpha"] = "v2"
    
    parameters = numoptim.param()
    parameters.tol = 1e-6
    parameters.maxit = 5*1e4
    parameters.rho = 0.5
    parameters.alpha_in = 1.
    parameters.maxback = 30
    parameters.c1 = 1e-4
    parameters.abstol = 1e-12
    parameters.reltol = 1e-8
    
    N = 100
    x_in = np.zeros(N)
    sol = barzilai_borwein(rosenbrock, grad_rosenbrock, x_in, options, parameters)
    print(sol)