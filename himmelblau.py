"""
Steepest Descent Method with Armijo Rule and Backtracking
"""

import numpy as np
from numoptim import steepest_descent
import numoptim 
from matplotlib import pyplot as plt
from matplotlib import cm

def himmelblau(x):
    """
    Parameter
    ---------
        x : list
            input 2d vector
            
    Returns
    -------
    float : value of the Himmelblau function at x
    """
    a = (x[0]**2 + x[1] - 11.0)**2 
    b = (x[0] + x[1]**2 - 7.0)**2
    return a + b

def grad_himmelblau(x):
    """
    Parameter
    ---------
        x : list
            input 2d vector
            
    Returns
    -------
    list : the gradient of the Himmelblau function at x
    """
    a = x[0]**2 + x[1] - 11.0
    b = x[0] + x[1]**2 - 7.0
    dx0 = 4.0*a*x[0] + 2.0*b
    dx1 = 2.0*a + 4.0*b*x[1]
    return np.array([dx0, dx1])

def hess_himmelblau(x):
    """
    Parameters
    ----------
    list : the gradient of the Himmelblau function at x
    
    Returns 
    -------
    list : the hessian of the Himmelblau function at x
    """
    a = x[0]**2 + x[1] - 11.0
    b = x[0] + x[1]**2 - 7.0
    return 

# Plotting function
def plot_surface(f, xlim, ylim, gridsize):
    """
    Plots the surface corresponding to the function f on a rectangular domain
    
    Parameters
    ----------
        f : callable
            Input function
        xlim : 2-list of floats
            Limits for x coordinates
        ylim : 2-list of float
            Limits for y coordinates
        gridsize : float
            Resolution of meshgrid
    """
    x = np.linspace(xlim[0], xlim[1], gridsize)
    y = np.linspace(ylim[0], ylim[1], gridsize)
    X, Y = np.meshgrid(x, y)
    Z = f(np.array([X, Y]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap=cm.twilight)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    return ax
    
if __name__ == "__main__":
    # compute minimizers of the Himmelblau function starting at the points
    # (+/- 2. +/- 2) using the steepest descent method with Armijo rule step
    # size criterion and backtracking step size method
    list_minimizers = []
    parameters = numoptim.param()
    options = numoptim.options
    options["ssm"] = "backtrack"
    options["ssc"] = "goldstein"
    for x_in in [[2., 2.], [2., -2.], [-2., -2.], [-2., 2.]]:
        result = steepest_descent(himmelblau, grad_himmelblau, np.array(x_in), options, parameters)
        print(f"Initial Iterate         : {x_in}")
        print(f"Approximate minimizer   : {result[0]}")
        print(f"Function Value          : {result[1]}")
        print(f"Gradient Norm           : {result[2]}")
        print(f"Number of Iterations    : {result[3]}")
        print("")
        list_minimizers.append(result[0])
    
    # plot surface corresponding to the Himmelblau function
    # and scatter plot of the computed minimizers
    ax = plot_surface(himmelblau, xlim=[-5, 5], ylim=[-5, 5], gridsize=100)
    for minimizer in list_minimizers:
        ax.scatter(minimizer[0], minimizer[1], s=20, marker='x', color='red')
    plt.show()