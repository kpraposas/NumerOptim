"""
Python module for Math 197 - Numerical Optimization
"""

import numpy as np
from time import time

# Class Parameters Initialization
class param():
    """
    Class for parameters in minimization algorithms
    
    Attributes
    ----------
        c1 : float
            Armijo parameter, default at 1e-4
        c : float
            Goldstein parameter, default at 0.25
        c2 : float
            Wolfe parameter, default at 0.9
        rho : float
            Backtracking parameter, default at 0.5
        rho_l : float
            Lower bound of backtracking projection, default at 0.1
        rho_r : float
            Upper bound of backtracking projection, default at 0.1
        alpha_in : float
            initial step length, default at 1.0 
        maxback : int
            maximum number of backtracking steps, default at 30
        tol : float
            tolerance of the method, default is 1e-10
        maxit : int 
            maximum numbers of iteration, default is 1e3
        iter_hist : bool
            prints function value and gradient at current iterate and
                number of backtracks/interpolations, default as False
    """
    def __init__(self, c1=1e-4, c=0.25, c2=0.9, rho=0.5, rho_l=0.1, rho_r=0.9, alpha_in=1.,
                 tol=1e-10, abstol=1e-12, reltol=1e-8, eps=np.finfo(float).eps, maxback=30, theta=0.5, gamma=1.5, maxit=1e3, iter_hist=False):
        """Class Initialization"""
        self.c1 = c1
        self.c = c
        self.c2 = c2
        self.rho = rho
        self.rho_l = rho_l
        self.rho_r = rho_r
        self.alpha_in = alpha_in
        self.maxback = maxback
        self.eps = eps
        self.theta = theta
        self.gamma = gamma
        self.tol = tol
        self.abstol = abstol
        self.reltol = reltol
        self.maxit = maxit
        self.iter_hist = iter_hist


# Options Dictionary Initialization
options = dict()
"""
    Dictionary for stepsize selection methods and criterion
        ssm         : backtrack, polyinterp
            stepsize selection method, backtrack as default
        ssc         : armijo, goldstein, wolfe, strongwolfe
            stepsize selection criterion, armijo as default
        beta        : fr, pr, frpr, prgn, hs, dy, hz
            beta formulas for nonlinear cg method, fr as default
        alpha       : v1, v2, v3, v4
            alpha formulas for barzilai-borwein, default is v1
        qn_type     : hess, inv_hess
            hessian or inverse hessian to be used in quasi newton, default at inv_hess
        qn_update   : sr1, nsr1, dfp, bfgs, bfgs-dfp, gr, ol, ssbfgs, ho, mh, ssvm
            update of hess or inverse hessian at each iterate, default is bfgs
"""
options = {"ssm"        : "backtrack",
           "ssc"        : "armijo",
           "beta"       : "fr",
           "alpha"      : "v1",
           "qn_type"    : "inv_hess",
           "qn_update"  : "bfgs"}


def minsearch(fun, grad, x, method, options, parameters):
    """
    
    """
 
# Methods
def steepest_descent(fun, grad, x, options, parameters):
    """
    Parameters
    ----------
        fun : callable
            objective function
        grad : callable
            gradient of the objective function
        x : list
            initial point
        options : dict
            method and criterion for stepsize strategy
        parameters : class 'numoptim.py'
            parameters to be passed for the method
    
    Returns
    -------
        tuple (x, fun, grad_norm, it, num_feval, num_geval, num_heval, 
            term_flag, elapsed_time)
            x : list
                approximate local minimizer or last iterate
            fun(x) : float
                function value of the approximate minimizer 
            grad_norm : float
                norm of the gradient at x
            it : int
                number of iterations
            num_feval : int
                number of function evaluations
            num_geval : int
                number of gradient evaluations
            num_heval : int
                number of function evaluations
            term_flag : str 
                termination flag of method
            elapsed_time : float
                time method took to execute
    """
    term_flag = "Success"
    start_time = time()
    it = 0
    gradx = grad(x)
    grad_norm = np.linalg.norm(gradx)
    if parameters.iter_hist == True:
        print("NUMIT\tFUN_VAL\tGRAD_NORM\tBACKTRACK")
    # Initialize Evaluation counters
    num_feval = 0
    num_geval = 1
    num_heval = 0
    while grad_norm > parameters.tol and it < parameters.maxit:
        d = -gradx
        fx = fun(x)
        alpha, alpha_it = stepsize_strategy(fun, grad, x, fx, -d, d, options, parameters)
        # Update evaluation counter from stepsize selection method
        if options["ssc"] == "armijo" or options["ssc"] == "goldstein":
            num_feval = num_feval + alpha_it
        if options["ssc"] == "wolfe" or options["ssc"] == "strongwolfe":
            num_feval = num_feval + alpha_it
            num_geval = num_geval + alpha_it
        if options["ssm"] == "polyinterp":
            num_feval = num_feval + 1 + alpha_it
        x = x + alpha * d
        gradx = grad(x)
        grad_norm = np.linalg.norm(gradx)
        # Update evaluation counter inside loop
        num_feval = num_feval + 1
        num_geval = num_geval + 1
        # Print Iteration
        if parameters.iter_hist == True:
            print(it, "\t", "{:.15e}".format(fx),
                  "\t", "{:.15e}".format(grad_norm), "\t", alpha_it)  
        it = it + 1
    if grad_norm > parameters.tol and it == parameters.maxit:
        term_flag = "Fail"
    stop_time = time()
    elapsed_time = stop_time - start_time
    return x, fun(x), grad_norm, it, num_feval, num_geval, num_heval, term_flag, elapsed_time

# newton cg damped newton
def newton(fun, grad, hess, x, options, parameters):
    """
    Parameters
    ----------
        fun : callable
            objective function
        grad : callable
            gradient of the objective function
        hess : callable
            hessian of the objective function
        x : list
            initial point
        parameters : class 'numoptim.py'
            parameters to be passed for the method
    
    Returns
        tuple (x, fun(x), grad_norm, it)
            x : list
                approximate local minimizer or last iterate
            fun(x) : float
                function value of the approximate minimizer
            grad_norm : float
                gradient norm of the approximate minimizer
            it : int
                number of iterations
    """
    it = 0
    if grad == None:
        gradx = num_grad(fun, x)
    else:
        gradx = grad(x)
    grad_norm = np.linalg.norm(gradx)
    while grad_norm > parameters.tol and it < parameters.maxit:
        if hess == None:
            hessx = num_hess(fun, x)
        else:
            hessx = hess(x)
        d = np.linalg.solve(hessx, -gradx)
        x = x + d
        if grad == None:
            gradx = num_grad(fun, x)
        else:
            gradx = grad(x)
        grad_norm = np.linalg.norm(gradx)
        it += 1
    return x, fun(x), grad_norm, it

def nonlinear_cg(fun, grad, x, options, parameters):
    """
    """
    fx = fun(x)
    gradx = grad(x)
    grad_norm = np.linalg.norm(gradx)
    d = -gradx
    it = 0
    while grad_norm > parameters.tol and it < parameters.maxit:
        alpha = stepsize_strategy(fun, grad, x,
            fx, gradx, d, options, parameters)[0]
        x = x + alpha*d
        gradx_old = gradx
        grad_norm_old = grad_norm
        gradx = grad(x)
        grad_norm = np.linalg.norm(gradx)
        beta = beta_formula(gradx_old, gradx, grad_norm_old, grad_norm, d, options)
        fx = fun(x)
        d = -gradx + beta*d
        it += 1
    return x, fun(x), grad_norm, it

def barzilai_borwein(fun, grad, x, options, parameters):
    """
    """
    gradx_old = grad(x)
    x_old = x
    fx = fun(x)
    d = -gradx_old
    alpha = stepsize_strategy(fun, grad, x,
        fx, -d, d, options, parameters)[0]
    x = x + alpha*d
    gradx = grad(x)
    grad_norm = np.linalg.norm(gradx)
    it = 1
    while grad_norm > parameters.tol and it < parameters.maxit:
        s = x - x_old
        y = gradx - gradx_old
        alpha = alpha_formula(s, y, it, options)
        x_old = x
        x = x - alpha*gradx
        gradx_old = gradx
        gradx = grad(x)
        grad_norm = np.linalg.norm(gradx)
        it += 1
    return x, fun(x), grad_norm, it

def quasi_newton(fun, grad, x, options, parameters):
    """
    """
    gradx = grad(x)
    grad_norm = np.linalg.norm(gradx)
    hess = np.eye(len(x))
    inv_hess = np.eye(len(x))
    it = 0
    while grad_norm > parameters.tol and it < parameters.maxit:
        if options["qn_type"] == "hess":
            d = np.linalg.solve(hess, -gradx)
        if options["qn_type"] == "inv_hess":
            d = -np.dot(inv_hess, gradx)
        fx = fun(x)
        alpha = stepsize_strategy(fun, grad, x, fx, gradx, d, options, parameters)[0]
        s = alpha*d
        x = x + s
        gradx_old = gradx
        gradx = grad(x)
        grad_norm = np.linalg.norm(gradx)
        y = gradx - gradx_old
        rho = np.dot(s, y)
        if rho < parameters.reltol*np.linalg.norm(s)*np.linalg.norm(y) or rho < parameters.abstol:
            if options["qn_type"] == "hess":
                hess = np.eye(len(x))
            if options["qn_type"] == "inv_hess":
                inv_hess = np.eye(len(x))
        else:
            if options["qn_type"] == "hess":
                hess = hess_update(hess, inv_hess, s, y, rho, options, parameters)
            if options["qn_type"] == "inv_hess":
                inv_hess = inv_hess_update(hess, inv_hess, s, y, rho, options, parameters)
        it += 1
    return x, fun(x), grad_norm, it

def hess_update(hess, inv_hess, s, y, rho, options, parameters):
    """
    """
    if options["qn_update"] == "bfgs":
        w = np.dot(hess, s)
        hess = hess - np.outer(w, w) / np.dot(s, w) + np.outer(y, y) / rho
    if options["qn_update"] == "sr1":
        w = y - np.dot(hess, s)
        hess = hess + np.outer(w, w) / np.dot(w, s)
    if options["qn_update"] == "nsr1":
        w = y - np.dot(hess, s)
        hess = hess + np.outer(w, s) / np.dot(s, s)
    if options["qn_update"] == "dfp":
        I = np.eye(len(s))
        w = np.outer(y, s)/rho
        hess = np.dot(I-w, np.dot(hess, I-w.T)) + np.outer(y, y) / rho
    if options["qn_update"] == "spb":
        w = y - np.dot(hess, s)
        hess = hess + (np.outer(w, s) + np.outer(s, w)) / np.dot(s, s) \
            - np.dot(w, s) / np.dot(s, s)**2 * np.outer(s, s)
    if options["qn_update"] == "bfgsdfp":
        temp = hess[:]
        options["qn_update"] = "bfgs"
        hess = parameters.theta*hess_update(hess, hess, s, y, rho, options, parameters)
        options["qn_update"] = "dfp"
        hess = hess + (1-parameters.theta)*hess_update(temp, inv_hess, s, y, rho, options, parameters)
        options["qn_update"] = "bfgsdfp"
    if options["qn_update"] == "ol":
        temp = hess[:]
        options["qn_update"] = "bfgs"
        hess = parameters.theta*parameters.gamma*hess_update(hess, hess, s, y, rho, options, parameters)
        options["qn_update"] = "dfp"
        hess = hess + (1-parameters.theta)*parameters.gamma*hess_update(hess, hess, s, y, rho, options, parameters)
        options["qn_update"] = "ol"
    return hess
    

def inv_hess_update(hess, inv_hess, s, y, rho, options, parameters):
    """
    """
    if options["qn_update"] == "bfgs":
        I = np.eye(len(s))
        w = np.outer(s, y)/rho
        inv_hess = np.dot(np.dot(I-w, inv_hess), I - w.T) + np.outer(s, s)/rho
    if options["qn_update"] == "sr1":
        w = s - np.dot(inv_hess, y)
        inv_hess = inv_hess + np.outer(w, w) / np.dot(w, y)
    if options["qn_update"] == "nsr1":
        w = s - np.dot(inv_hess, y)
        inv_hess = inv_hess + np.outer(w, y) / np.dot(y, y)
    if options["qn_update"] == "dfp":
        w = np.dot(inv_hess, y)
        inv_hess = inv_hess - np.outer(w, w)/ np.dot(y, w) + np.dot(s, s)/ rho
    if options["qn_update"] == "gr":
        w = s - np.dot(inv_hess, y)
        inv_hess = inv_hess + (np.outer(w, y) + np.outer(y, w)) / np.dot(y, y) \
            - np.dot(w, y) / np.dot(y, y)**2 * np.outer(y, y)
    if options["qn_update"] == "bfgsdfp":
        temp = inv_hess[:]
        options["qn_update"] = "bfgs"
        inv_hess = parameters.theta*inv_hess_update(hess, inv_hess, s, y, rho, options, parameters)
        options["qn_update"] = "dfp"
        inv_hess = inv_hess + (1-parameters.theta)*inv_hess_update(hess, temp, s, y, rho, options, parameters)
        options["qn_update"] = "bfgsdfp"
    if options["qn_update"] == "ol":
        I = np.eye(len(s))
        w1 = np.outer(s, y)/rho
        w2 = np.dot(inv_hess, y)
        inv_hess = parameters.theta*parameters.gamma*np.dot(np.dot(I-w1, inv_hess), I-w1.T) \
            + (1-parameters.theta)*parameters.gamma*(inv_hess - np.outer(w2, w2) / np.dot(y, w2)) \
            + np.outer(s,s) / rho
    if options["qn_update"] == "ssbfgs":
        w1 = np.dot(inv_hess, y)
        w2 = np.dot(y, w1)
        v = s / rho - w1 / w2
        hess = hess_update(hess, inv_hess, s, y, rho, options, parameters)
        inv_hess = rho / np.dot(y, np.dot(hess, y))*(inv_hess - np.outer(w1, w1) / w2 + w2*np.outer(v, v)) \
            + np.dot(s, s) / rho
    if options["qn_update"] == "ho":
        w1 = np.dot(inv_hess, y)
        w2 = np.dot(y, w1)
        v = s / rho - w1 / w2
        inv_hess = inv_hess + np.outer(s, s) + rho*w2 / (rho + w2) * np.outer(v, v) \
            - np.outer(w1, w1) / w2
    if options["qn_update"] == "mh":
        w = np.dot(inv_hess, y)
        u = s + w
        v = 0.1*s + 0.1*w
        inv_hess = inv_hess + np.outer(u, s) / np.dot(u, y) - np.outer(v, w) / np.dot(v, y)
    return inv_hess

def hess_action(grad, x, gradx, d, parameters):
    """
    """
    d_norm = np.linalg.norm(d)
    if d_norm == 0:
        z = np.zeros(len(x))
    else:
        gradx_temp = grad(x + parameters.eps/d_norm*d)
        z = (gradx_temp-gradx)/parameters.eps
    return z

def hess_approx(grad, x, gradx, parameters):
    """
    """
    n = len(x)
    hess = np.zeros([n,n])
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1
        hess[:, j] = hess_action(grad, x, gradx, e, parameters)
    return (hess + hess.T)/2

def trust_region(fun, grad, x, options, parameters):
    """
    """
    fx = fun(x)
    gradx = grad(x)
    grad_norm = np.linalg.norm(gradx)
    delta = min(grad_norm, 100)
    NewPointFlag = True
    it = 0
    while grad_norm > parameters.tol and it < parameters.maxit:
        print(grad_norm)
        it += 1
        if NewPointFlag == True:
            hess = hess_approx(grad, x, gradx, parameters)
        mu = np.dot(gradx, np.dot(hess, gradx))
        grad_norm = np.linalg.norm(gradx)
        mu1 = grad_norm**2
        TrialPointType = "Cauchy"
        BoundaryPointFlag = "False"
        if mu <= 0:
            x_cauchy = x - delta/grad_norm*gradx
            BoundaryPointFlag = True
        else:
            ratio1 = delta/grad_norm
            ratio2 = mu1/mu
            sigma = min(ratio1, ratio2)
            if ratio2 >= ratio1:
                BoundaryPointFlag = True
            x_cauchy = x - sigma*gradx
            inv_hess = np.linalg.inv(hess)
            d_newton = np.dot(inv_hess, gradx)
            if np.dot(d_newton, gradx) >= 0:
                pass
            else:
                x_newton = x + d_newton
                d_norm = np.linalg.norm(d_newton)
                if d_norm <= delta:
                    TrialPointType = "Newton"
                    if d_norm >= delta - 1e-6:
                        BoundaryPointFlag = True
                else:
                    if BoundaryPointFlag == True:
                        pass
                    else:
                        d_cauchy = -sigma*gradx
                        d = d_newton - d_cauchy
                        a, b, c = np.linalg.norm(d)**2, 2*np.dot(d_cauchy, d), np.linalg.norm(d_cauchy)**2 - delta**2
                        xi = (-b+np.sqrt(b**2 - 4*a*c))/(2*a)
                        x_cauchy = x_cauchy + xi*(x_newton - x_cauchy)
                        BoundaryPointFlag = True
                        TrialPointType = "Dogleg"
        if TrialPointType == "Newton":
            x_trial = x_newton
        else:
            x_trial = x_cauchy
        d_trial = x_trial - x
        f_trial = fun(x_trial)
        actual_reduction = f_trial - fx
        predicted_reduction = np.dot(gradx, d_trial) + 0.5*np.dot(d_trial, np.dot(hess, d_trial))
        rho = actual_reduction / predicted_reduction
        if rho >= 1e-4:
            x = x_trial
            NewPointFlag = True
        else:
            delta = 0.5*min(delta, np.linalg.norm(d_trial))
            NewPointFlag = False
        if rho > 0.75 and BoundaryPointFlag == True:
            delta = min(2*delta, 1000)
        if NewPointFlag == True:
            fx = fun(x)
            gradx = grad(x)
            grad_norm = np.linalg.norm(gradx)
    return x, fx, grad_norm, it


# Armijo Stepsize Selection Method
def stepsize_strategy(fun, grad, x, fx, gradx, d, options, parameters):
    """
    Parameters
    ----------
        fun : callable
            objective function
        grad : callable
            gradient of objective function
        x : list
            initial point
        fx : float
            function value at x
        gradx : list
            gradient at x
        d : list
            current search direction
        options : dict
            method and criterion for stepsize strategy
        parameters : class
            parameters to be passed for the method
    
    Returns
    -------
        stepsize strategy executed
    """
    if options["ssm"] == "backtrack":
        return backtrack(fun, grad, x, fx, gradx, d, options, parameters)
    if options["ssm"] == "polyinterp":
        return polyinterp(fun, grad, x, fx, gradx, d, options, parameters)
    
def backtrack(fun, grad, x, fx, gradx, d, options, parameters):
    """
    Returns
    -------
        tuple (alpha, j)
        alpha : float
            steplength satisfying the specified rule or the last steplength
        j : int
            number of backtracking steps
    """
    alpha = parameters.alpha_in
    q = np.dot(gradx, d)
    j = 0
    while stopping_criterion(fun, grad, x, fx, d, alpha, q, options, parameters) and j < parameters.maxback:
        alpha = parameters.rho * alpha
        j = j + 1
    return alpha, j

def polyinterp(fun, grad, x, fx, gradx, d, options, parameters):
    """
    Returns
    -------
    tuple (alpha, j)
        alpha : float
            steplength satisfying the specified rule or the last steplength
        j : int
            number of interpolations
    """
    alpha = parameters.alpha_in
    q = np.dot(gradx, d)
    f = fun(x + alpha*d)
    j = 0
    while stopping_criterion(fun, grad, x, fx, d, alpha, q, options, parameters) and j < parameters.maxback:
        if j == 0:
            alpha_new = -q*alpha**2 / (2*(f - fx - alpha*q))
        else:
            A = [[alpha_old**2, alpha_old**3], [alpha**2, alpha**3]]
            b = [f_old - fx - alpha_old*q, f - fx - alpha*q]
            c0, c1 = np.linalg.solve(A, b)
            if c0**2 - 3*c1*q >= 0 and abs(c1) > 1e-10:
                alpha_new = (-c0 + np.sqrt(c0**2 - 3*c1*q)) / (3*c1)
            else:
                alpha_new = alpha
        alpha_old = alpha
        f_old = f
        alpha = max(parameters.rho_l*alpha, min(alpha_new, parameters.rho_r*alpha))
        f = fun(x + alpha*d)
        j += 1
    return alpha, j


# Stepsize Selection Criterion
def stopping_criterion(fun, grad, x, fx, d, alpha, q, options, parameters):
    """
    Parameters
    ----------
        fun : callable
            objective function
        grad : callable
            gradient of objective function
        x : list
            current point
        fx : float
            function value at x
        d : list
            current search direction
        alpha : float
            current step length
        q : float
            dot product of gradient at x and current direction
        options : dict
            method and criterion for stepsize strategy
        parameters : class 'numoptim.py'
            parameters to be passed for the method

    Returns
    -------
        either set stepsize criterion satisfied or not
    """
    if options["ssc"] == "armijo":
        return Rule_Armijo(fun, x, fx, d, alpha, q, parameters)
    if options["ssc"] == "goldstein":
        return Rule_Goldstein(fun, x, fx, d, alpha, q, parameters)
    if options["ssc"] == "wolfe":
        return Rule_Wolfe(fun, grad, x, fx, d, alpha, q, parameters)
    if options["ssc"] == "strongwolfe":
        return Rule_StrongWolfe(fun, grad, x, fx, d, alpha, q, parameters)

def Rule_Armijo(fun, x, fx, d, alpha, q, parameters):
    """
    Returns
    -------
        bool, if Armijo rule is satisfied or not
    """
    LHS = fun(x + alpha*d)
    RHS = fx + parameters.c1*alpha*q
    return LHS > RHS

def Rule_Goldstein(fun, x, fx, d, alpha, q, parameters):
    """
    Returns
    -------
        bool, if Goldstein rule is satisfied or not
    """
    LHS = fx + (1 - parameters.c)*alpha*q
    Ctr = fun(x + alpha*d)
    RHS = fx + parameters.c*alpha*q
    return LHS > Ctr or Ctr > RHS

def Rule_Wolfe(fun, grad, x, fx, d, alpha, q, parameters):
    """
    Returns
    -------
        bool, if Wolfe rule is satisfied or not
    """
    LHS = np.dot(grad(x + alpha*d), d)
    RHS = parameters.c2*q
    return Rule_Armijo(fun, x, fx, d, alpha, q, parameters) or LHS < RHS

def Rule_StrongWolfe(fun, grad, x, fx, d, alpha, q, parameters):
    """
    Returns
    -------
        bool, if Strong Wolfe rule is satisfied or not
    """
    LHS = abs(np.dot(grad(x + alpha*d), d))
    RHS = -parameters.c2*q
    return Rule_Armijo(fun, x, fx, d, alpha, q, parameters) or LHS > RHS


# Numerical gradient and hessian for Newton
def num_grad(fun, x, h=1e-6):
    """
    Parameters
    ----------
        fun : callable
            objective function
        x : list
            initial point
        h : float
            step size h from x
            
    Returns
    -------
        num_grad : list
            numerical gradient of objective function at x
    """
    n = len(x)
    num_grad = []
    for i in range(n):
        stepsize = h*np.eye(1, n, i)
        df_xi = (fun(x + stepsize) - fun(x - stepsize))/(2*h)
        num_grad.append(df_xi)
    return num_grad

def num_hess(fun, x, h=1e-6):
    """
    Parameters
    ----------
    fun : callable
            objective function
        x : list
            initial point
    """
    n = len(x)
    num_hess = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            if i == j: 
                stepsize = 2.0*h*np.eye(1, n, j)
                df_xixi = (fun(x + stepsize) - 2.0*fun(x) + fun(x - stepsize))/ (4*h*h)
                num_hess[i][j] = df_xixi
            else:
                stepsize1 = h*(np.eye(1, n, i) + np.eye(1, n, j))
                stepsize2 = h*(np.eye(1, n, i) - np.eye(1, n, j))
                df_xixj = (fun(x + stepsize1) - fun(x + stepsize2) - fun(x - stepsize2) + fun(x - stepsize1)) / (4*h*h)
                num_hess[i][j], num_hess[j][i] = df_xixj, df_xixj
    return num_hess


# Beta computations
def beta_formula(gradx_old, gradx, grad_norm_old, grad_norm, d, options):
    """
    """
    if options['beta'] == 'fr':
        beta = grad_norm**2 / grad_norm_old**2
    if options['beta'] == 'pr':
        beta = np.dot(gradx, gradx - gradx_old) / grad_norm_old
    if options['beta'] == 'prgn':
        beta = np.dot(gradx, gradx - gradx_old) / grad_norm_old
        beta = max(0, beta)
    if options['beta'] == 'frpr':
        beta_pr = np.dot(gradx, gradx - gradx_old) / grad_norm_old
        beta_fr = grad_norm**2 / grad_norm_old**2
        if abs(beta_pr) <= beta_fr:
            beta = beta_pr
        elif beta_pr < -beta_fr:
            beta = -beta_fr
        elif beta_pr > beta_fr:
            beta = beta_fr
    if options['beta'] == 'hs':
        y = gradx - gradx_old
        beta = np.dot(gradx, y) / np.dot(d, y)
    if options['beta'] == 'dy':
        beta = grad_norm**2/np.dot(d, gradx - gradx_old)
    if options['beta'] == 'hz':
        y = gradx - gradx_old
        w = np.dot(d, y)
        beta = np.dot(y - 2.*(np.linalg.norm(y)**2/w)*d, gradx/w)
    return beta


# Alpha computations for Barzilai-Borwein
def alpha_formula(s, y, k, options):
    if options['alpha'] == 'v1':
        alpha = np.dot(s, y)/np.linalg.norm(y)**2
    if options['alpha'] == 'v2':
        alpha = np.linalg.norm(s)**2/np.dot(s, y)
    if options['alpha'] == 'v3':
        if 0 == k%2:
            alpha = np.dot(s, y)/np.linalg.norm(y)**2
        else:
            alpha = np.linalg.norm(s)**2/np.dot(s, y)
    if options['alpha'] == 'v4':
        if 0 == k%2:
            alpha = np.linalg.norm(s)**2/np.dot(s, y)
        else:
            alpha = np.dot(s, y)/np.linalg.norm(y)**2
    return alpha


"""
    Minimizer search for quadratic functions f: R^n -> R defined by
        f(x) = 0.5x^TAx - b^Tx + c
        where A is n-by-n SPD matrix, b is n-dimensional vector and c is scalar
"""

def quadminsearch(A, b, c, x, method, parameters):
    """
    Minimizer search for quadratic functions
    
    Parameters
    ----------
        A : 2d array
            SPD matrix
        b : 1d array
            n-dim vector
        c : float
            constant term
        x : list
            initial iterate
        method : str
            method to be used, accepted str are 
                sd, cg, cgnr, and cgr
        parameters : class 'numoptim.py'
            parameters to be passed for the method
            
    Returns
    -------
        Executes and returns the result of the method given
    """
    if method == 'sd':
        return steep_descent(A, b, c, x, parameters)
    if method == 'cg':
        return CG(A, b, c, x, parameters)
    if method == 'cgnr':
        return CGNR(A, b, c, x, parameters)
    if method == 'cgr':
        return CGR(A, b, c, x, parameters)

def steep_descent(A, b, c, x, parameters):
    """
    Steepest Descent Method
    
    Parameters
    ----------
        A : 2d array
            SPD matrix
        b : 1d array
            n-dim vector
        x : list
            initial iterate    
    
    Returns
    -------

    """
    k = 0
    gradx = np.dot(A, x) - b
    grad_norm = np.linalg.norm(gradx)
    while grad_norm > parameters.tol:
        d = -gradx
        alpha = np.dot(d, d) / np.dot(d, np.dot(A, d))
        x = x + alpha*d
        gradx = np.dot(A, x) - b
        grad_norm = np.linalg.norm(gradx)
        k += 1
    funval = 0.5*np.dot(x, np.dot(A, x)) - np.dot(b, x) + c
    return x, funval, grad_norm, k

def CG(A, b, c, x, parameters):
    """
    Linear Conjugate Gradient Method
    
    Parameters
    ----------
        A : 2d array
            SPD matrix
        b : 1d array
            n-dim vector
        x : list
            initial iterate    
    
    Returns
    -------
    """
    k = 0
    gradx = np.dot(A, x) - b
    grad_norm = np.linalg.norm(gradx)
    d = -gradx
    while grad_norm > parameters.tol:
        w = np.dot(A, d)
        alpha = grad_norm**2/(np.dot(d, w))
        x = x + alpha*d
        grad_norm_old = grad_norm
        gradx = gradx + alpha*w
        grad_norm = np.linalg.norm(gradx)
        beta = grad_norm**2/grad_norm_old**2
        d = -gradx + beta*d
        k += 1
    funval = 0.5*np.dot(x, np.dot(A, x)) - np.dot(b, x) + c
    return x, funval, grad_norm, k

def CGNR(A, b, c, x, parameters):
    """
    Conjugate Gradient Normal Residual Method
    
    Parameters
    ----------
        A : 2d array
            SPD matrix
        b : 1d array
            n-dim vector
        x : list
            initial iterate    
    
    Returns
    -------
    """
    k = 0
    gradx = np.dot(A, x) - b
    grad_norm = np.linalg.norm(gradx)
    d = -np.dot(A, gradx)
    z = -d
    while grad_norm > parameters.tol:
        w = np.dot(A, d)
        alpha = np.dot(z, z)/(np.dot(w, w))
        x = x + alpha*d
        gradx = gradx + alpha*w
        grad_norm = np.linalg.norm(gradx)
        z_old = z
        z = np.dot(A, gradx)
        beta = np.dot(z, z)/np.dot(z_old, z_old)
        d = -z + beta*d
        k += 1
    funval = 0.5*np.dot(x, np.dot(A, x)) - np.dot(b, x) + c
    return x, funval, grad_norm, k

def CGR(A, b, c, x, parameters):
    """
    Conjugate Gradient Residual Method
    
    Parameters
    ----------
        A : 2d array
            SPD matrix
        b : 1d array
            n-dim vector
        x : list
            initial iterate    
    
    Returns
    -------
    """
    k = 0
    gradx = np.dot(A, x) - b
    grad_norm = np.linalg.norm(gradx)
    d = -gradx
    while grad_norm > parameters.tol:
        w = np.dot(A, d)
        alpha = -np.dot(gradx, w)/(np.dot(w, w))
        x = x + alpha*d
        gradx = gradx + alpha*w
        grad_norm = np.linalg.norm(gradx)
        beta = np.dot(np.dot(gradx, A), w)/np.dot(w, w)
        d = -gradx + beta*d
        k += 1
    funval = 0.5*np.dot(x, np.dot(A, x)) - np.dot(b, x) + c
    return x, funval, grad_norm, k