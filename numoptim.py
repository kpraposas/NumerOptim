"""
Python module - Numerical Optimization
"""

import numpy as np
from time import time

# Class parameter Initialization
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
            Upper bound of backtracking projection, default at 0.5
        alpha_in : float
            initial step length, default at 1.0 
        maxback : int
            maximum number of backtracking steps, default at 30
        tol : float
            tolerance of the method, default is 1e-10
        abstol : float
        
        reltol : float
        
        theta : float
        
        gamma : float
        
        eps : float
            step size for the Hessian approximate, default is 1e-6
        delta : float
            maximum radius of the trust region minimizer search, default at
        eta : float
            minimum ratio of actual to predicted reduction to update iterate,
            default at
        mu_lo : float
            maximum ratio of actual to predicted reduction to increase search
            radius, default at
        mu_hi : float
            minimum ratio of actual to predicted reduction to decrease search
            radius, default at
        sigma_lo : float
            scale the next search radius by this factor 
        sigma_hi : float
            dilate the next search radiues by this factor
        maxit : int 
            maximum numbers of iteration, default is 1e3
        iter_hist : bool
            prints function value and gradient at current iterate and
                number of backtracks/interpolations, default as False
    """
    def __init__(self, c1=1e-4, c=0.25, c2=0.9, rho=0.5, rho_l=0.1, rho_r=0.9,
                 alpha_in=1., maxback=30, tol=1e-10, abstol=1e-12, reltol=1e-8,
                 theta=0.5, gamma=1.5, eps=1e-6, delta=1000, eta=1e-4,
                 mu_lo=0.25, mu_hi=0.75, sigma_lo=0.5, sigma_hi=2., maxit=1e3,
                 iter_hist=False):
        """Class Initialization"""
        self.c1 = c1
        self.c = c
        self.c2 = c2
        self.rho = rho
        self.rho_l = rho_l
        self.rho_r = rho_r
        self.alpha_in = alpha_in
        self.maxback = maxback
        self.tol = tol
        self.abstol = abstol
        self.reltol = reltol
        self.theta = theta
        self.gamma = gamma
        self.eps = eps
        self.delta = delta
        self.eta = eta
        self.mu_lo = mu_lo
        self.mu_hi = mu_hi
        self.sigma_lo = sigma_lo
        self.sigma_hi = sigma_hi
        self.maxit = maxit
        self.iter_hist = iter_hist


# Options Dictionary Initialization
options = dict()
"""
    Dictionary for stepsize selection methods and criterion
        iter_hist : bool
            prints function value and gradient at current iterate and
                number of backtracks/interpolations, default as False
        ssm         : backtrack, polyinterp
            stepsize selection method, backtrack as default
        ssc         : armijo, goldstein, wolfe, strongwolfe
            stepsize selection criterion, armijo as default
        beta        : fr, pr, frpr, prgn, hs, dy, hz
            beta formulas for nonlinear cg method, fr as default
        ver         : inv, damp, cg, nh, ngnh
            version of Newton Method to be used, cg as default
        alpha       : v1, v2, v3, v4
            alpha formulas for barzilai-borwein, default is v1
        qn_type     : hess, inv_hess
            hessian or inverse hessian to be used in quasi newton,
                default at inv_hess
        qn_update   : sr1, nsr1, dfp, bfgs, bfgs-dfp, gr, ol, ssbfgs, ho, mh, ssvm
            update of hess or inverse hessian at each iterate, default is bfgs
"""
options = {
    "ssm"        : "backtrack",
    "ssc"        : "armijo",
    "beta"       : "fr",
    "alpha"      : "v1",
    "ver"        : "cg",
    "qn_type"    : "inv_hess",
    "qn_update"  : "bfgs"}
    
class minsearch():
    """
    Class for the result of the minimizer search algorithm
    
    Attributes
    ----------
        x               : list
            approximate minimizer of the function
        fx              : float
            function value of the minimizer
        grad_norm       : float
            norm of the gradient of x under the function
        it              : int
            number of iterations the method took
        num_evals       : list
            triple consisting of number of function, gradient, and hessian evals
        term_flag       : str
            whether the method is successful in approximating a minimizer
        method          : str
            method used to search for a minimizer
        options         : dict
            stepsize selection criterion and method used by the method
        parameter       : param
            parameters used by the method
        elapsed_time    : float
            time in seconds the method took    
    """
    def __init__(self, x, fx, grad_norm, it, num_evals, term_flag, method,
                 options, param, elapsed_time):
        self.x = x
        self.fx = fx
        self.grad_norm = grad_norm
        self.it = it
        self.num_evals = num_evals
        self.term_flag = term_flag
        self.method = method
        self.options = options
        self.param = param
        self.elapsed_time = elapsed_time
        
    def __str__(self):
        list_str_repr = [
            "METHOD                          : {}".format(self.method),
            "STEPSIZE SELECTION METHOD       : {}".format(self.options["ssm"].upper()),
            "STEPSIZE SELECTION CRITERION    : {}".format(self.options["ssc"].upper()),
            "APPROXIMATE MINIMIZER           : {}".format(self.x),
            "FUNCTION VALUE                  : {}".format(self.fx),
            "GRADIENT NORM OF MINIMIZER      : {}".format(self.grad_norm),
            "TOLERANCE                       : {}".format(self.param.tol),
            "NITERS                          : {}".format(self.it),
            "TERMINATION                     : {}".format(self.term_flag),
            "NUM_FEVALS                      : {}".format(self.num_evals[0]),
            "NUM_GEVALS                      : {}".format(self.num_evals[1]),
            "NUM_HEVALS                      : {}".format(self.num_evals[2]),
            "ELAPSED TIME                    : {} seconds".format(self.elapsed_time)
        ]
        return "\n".join(list_str_repr)
        
 
# Methods
def steepest_descent(fun: callable, grad: callable, x: list, options: dict,
                     parameter: param) -> minsearch:
    """
    Steepest Descent method to approximate the minimizer of objective function
    
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
        parameter : param
            parameter to be passed for the method
    
    Returns
    -------
        minsearch
            Result of the method
    """
    term_flag = "Success"
    start_time = time()
    it = 0
    # If iterations are to be printed
    if parameter.iter_hist == True:
        print("NUMIT\tFUN_VAL\tGRAD_NORM\tBACKTRACK")
    
    gradx = grad(x)
    grad_norm = np.linalg.norm(gradx)
    
    # Initialize Evaluation counters
    num_evals = [0, 1, 0]
    
    while grad_norm > parameter.tol and it < parameter.maxit:
        d = -gradx
        fx = fun(x)
        alpha, alpha_it = stepsize_strategy(fun, grad, x, fx, -d, d,
                                            options["ssm"], options["ssc"],
                                            parameter)
        # Update evaluation counter from stepsize selection method
        match options["ssc"]:
            case "armijo", "goldstein":
                num_evals[0] += alpha_it
            case "wolfe", "strongwolfe":
                num_evals[0] += alpha_it
                num_evals[1] += alpha_it
        match options["ssm"]:
            case "polyinterp":
                num_evals[0] += 1 + alpha_it
                
        x = x + alpha * d
        gradx = grad(x)
        grad_norm = np.linalg.norm(gradx)
        # Update evaluation counter inside loop
        num_evals[0] += 1
        num_evals[1] += 1
        # Print Iteration
        if parameter.iter_hist == True:
            print("{}\t{:.15e}\t{:.15e}\t{}".format(it, fx, grad_norm, alpha_it))
        it += + 1
    if grad_norm > parameter.tol and it == parameter.maxit:
        term_flag = "Fail"
    stop_time = time()
    elapsed_time = stop_time - start_time
    return minsearch(x, fun(x), grad_norm, it, num_evals, term_flag,
                     "STEEPEST DESCENT", options, parameter, elapsed_time)

def nonlinearcg(fun: callable, grad: callable, x: list, options: dict,
                parameter: param) -> minsearch:
    """
    Extension of Conjugate Gradient method to nonlinear functions on
        approximating minimizer
    
    Parameters
    ----------
    fun : callable
        objective function
    grad : callable
        gradient of the objective function
    x : list
        initial point
    options : dict
        stepsize selection criterion and method used by the method
    parameter : param
        parameters used by the method

    Returns
    -------
    minsearch
        Result of the method
    """
    term_flag = "Success"
    start_time = time()
    fx = fun(x)
    gradx = grad(x)
    grad_norm = np.linalg.norm(gradx)
    d = -gradx
    it = 0
    
    # If iterations are to be printed
    if parameter.iter_hist:
        print("NUMIT\tFUN_VAL\tGRAD_NORM\tBACKTRACK")
    # Initialize evaluation counters
    num_evals = [1, 1, 0]
    
    while grad_norm > parameter.tol and it < parameter.maxit:
        alpha, alpha_it = stepsize_strategy(fun, grad, x, fx, gradx, d,
                                            options["ssm"], options["ssc"],
                                            parameter)
        
        # Update evaluation counter from stepsize selection method
        match options["ssc"]:
            case "armijo", "goldstein":
                num_evals[0] += alpha_it
            case "wolfe", "strongwolfe":
                num_evals[0] += alpha_it
                num_evals[1] += alpha_it
        match options["ssm"]:
            case "polyinterp":
                num_evals[0] += 1 + alpha_it
        
        x = x + alpha*d
        gradx_old = gradx
        grad_norm_old = grad_norm
        fx = fun(x)
        gradx = grad(x)
        grad_norm = np.linalg.norm(gradx)
        beta = beta_formula(gradx_old, gradx, grad_norm_old, grad_norm, d,
                            options["beta"])
        d = -gradx + beta*d
        # Update evaluation counter inside loop
        num_evals[0] += 1
        num_evals[1] += 1
        # Print Iteration
        if parameter.iter_hist == True:
            print("{}\t{:.15e}\t{:.15e}\t{}".format(it, fx, grad_norm, alpha_it))
        it += 1
    if grad_norm > parameter.tol and it == parameter.maxit:
        term_flag = "Fail"
    stop_time = time()
    elapsed_time = stop_time - start_time
    return minsearch(x, fx, grad_norm, it, num_evals, term_flag,
                     f"NLCG - {options["beta"]}", options, param, elapsed_time)

def newton(fun: callable, x: list, options: dict, parameter: param,*,
           grad: callable=None, hess: callable=None,) -> minsearch:
    """
    Newton's method to numerical optimization
    
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
    options : dict
        stepsize selection criterion and method used by the method
    parameter : param
        parameter to be passed for the method

    Result
    -------
    minsearch
        Result of the Method
    """
    term_flag = "Success"
    start_time = time()
    fx = fun(x)

    it = 0
    if grad == None:
        gradx = num_grad(fun, x)
        num_evals = [1, 0, 0]
    else:
        gradx = grad(x)
        num_evals = [1, 1, 0]
    grad_norm = np.linalg.norm(gradx)
    
    # If iterations are to be printed
    if parameter.iter_hist:
        print("NUMIT\tFUN_VAL\tGRAD_NORM")
    
    while grad_norm > parameter.tol and it < parameter.maxit:
        if hess == None:
            hessx = num_hess(fun, x)
        else:
            hessx = hess(x)
            num_evals[2] += 1
        match options["ver"]:
            case "inv":
                d = -np.dot(np.linalg.inv(hessx), gradx)
                method = "NEWTON"
            case "damp":
                d = np.dot(np.linalg.inv(hessx), gradx)
                alpha, alpha_it = stepsize_strategy(fun, grad, x, fx, gradx,
                                                    -d, options["ssm"],
                                                    options["ssc"], parameter)
                d = -alpha*d
                method = "DAMPED NEWTON"
                # Update evaluation counter from stepsize selection method
                match options["ssc"]:
                    case "armijo", "goldstein":
                        num_evals[0] += alpha_it
                    case "wolfe", "strongwolfe":
                        num_evals[0] += alpha_it
                        num_evals[1] += alpha_it
                match options["ssm"]:
                    case "polyinterp":
                        num_evals[0] += 1 + alpha_it
            case "cg":
                d = np.linalg.solve(hessx, -gradx)
                method = "NEWTON-CG"
            case "nh":
                d = np.linalg.solve(hessx, -gradx)
                method = "NEWTON-NH"
            case "ngnh":
                d = np.linalg.solve(hessx, -gradx)
                method = "NEWTON-NGNH"
        x = x + d
        fx = fun(x)
        num_evals[0] += 1
        if grad == None:
            gradx = num_grad(fun, x)
        else:
            gradx = grad(x)
            num_evals[1] += 1
        grad_norm = np.linalg.norm(gradx)
        it += 1
        if parameter.iter_hist == True:
            print("{}\t{:.15e}\t{:.15e}".format(it, fx, grad_norm))
    if grad_norm > parameter.tol and it == parameter.maxit:
        term_flag = "Fail"
    stop_time = time()
    elapsed_time = stop_time - start_time
    return minsearch(x, fx, grad_norm, it, num_evals, term_flag, method,
                     options, param, elapsed_time)

def barzilaiborwein(fun: callable, grad: callable, x: list, options: dict,
                    parameter: param) -> minsearch:
    """
    Gradienth method that aims to approximate the action of the Hessian

    Parameters
    ----------
    fun : callable
        objective function
    grad : callable
        gradient of the objective function
    x : list
        initial point
    options : dict
        stepsize selection criterion and method used by the method
    parameter : param
        parameters used by the method

    Returns
    -------
    minsearch
        Result of the method
    """
    term_flag = "Success"
    start_time = time()
    
    fx = fun(x)
    gradx_old = grad(x)
    x_old = x
    d = -gradx_old
    alpha, alpha_it = stepsize_strategy(fun, grad, x, fx, -d, d, options["ssm"],
                                        options["ssc"], parameter)
    x = x + alpha*d
    gradx = grad(x)
    grad_norm = np.linalg.norm(gradx)
    it = 1
    
    # Initialize evaluation counters
    num_evals = [1, 2, 0]
    # Update evaluation counter from stepsize selection method
    match options["ssc"]:
        case "armijo", "goldstein":
            num_evals[0] += alpha_it
        case "wolfe", "strongwolfe":
            num_evals[0] += alpha_it
            num_evals[1] += alpha_it
    match options["ssm"]:
        case "polyinterp":
            num_evals[0] += 1 + alpha_it
    # If iterations are to be printed
    if parameter.iter_hist:
        print("NUMIT\tGRAD_NORM")
        print("{}\t{:.15e}\t{:.15e}\t{}".format(it, fx, grad_norm, alpha_it))
    
    while grad_norm > parameter.tol and it < parameter.maxit:
        deltax = x - x_old
        deltag = gradx - gradx_old
        alpha = alpha_formula(deltax, deltag, it, options["alpha"])
        x_old = x
        x = x - alpha*gradx
        gradx_old = gradx
        gradx = grad(x)
        grad_norm = np.linalg.norm(gradx)
        it += 1
        num_evals[1] += 1
        if parameter.iter_hist == True:
            print("{}\t{:.15e}".format(it, grad_norm))
    fx = fun(x)
    num_evals[0] += 1
    if grad_norm > parameter.tol and it == parameter.maxit:
        term_flag = "Fail"
    stop_time = time()
    elapsed_time = stop_time - start_time
    return minsearch(x, fx, grad_norm, it, num_evals, term_flag,
                     f"NLCG - BW {options["alpha"].capitalize()}",
                     options, param, elapsed_time)

def quasinewton(fun: callable, grad: callable, x: list, options: dict,
                 parameter: param) -> minsearch:
    """
    Damped Newton Method with Hessian or its inverse approximated

    Parameters
    ----------
    fun : callable
        objective function
    grad : callable
        gradient of the objective function
    x : list
        initial point
    options : dict
        stepsize selection criterion and method used by the method
    parameter : param
        parameters used by the method

    Returns
    -------
    minsearch
        Result of the method
    """
    term_flag = "Success"
    start_time = time()
    fx = fun(x)
    gradx = grad(x)
    grad_norm = np.linalg.norm(gradx)
    match options["qn_type"]:
        case "hess":
            hess = np.eye(len(x))
        case "inv_hess":
            inv_hess = np.eye(len(x))
    it = 0
    
    # Initialize evaluation counters
    num_evals = [1, 1, 0]
    # If iterations are to be printed
    if parameter.iter_hist:
        print("NUMIT\tFUN_VAL\tGRAD_NORM\tBACKTRACK")
    
    while grad_norm > parameter.tol and it < parameter.maxit:
        match options["qn_type"]:
            case "hess":
                d = np.linalg.solve(hess, -gradx)
            case "inv_hess":
                d = -np.dot(inv_hess, gradx)
        alpha, alpha_it = stepsize_strategy(fun, grad, x, fx, gradx, d,
                                            options["ssm"], options["ssc"],
                                            parameter)
        s = alpha*d
        x = x + s
        gradx_old = gradx
        fx = fun(x)
        gradx = grad(x)
        grad_norm = np.linalg.norm(gradx)
        y = gradx - gradx_old
        rho = np.dot(s, y)
        if rho < parameter.reltol*np.linalg.norm(s)*np.linalg.norm(y)\
            or rho < parameter.abstol:
            match options["qn_type"]:
                case "hess":
                    hess = np.eye(len(x))
                case "inv_hess":
                    inv_hess = np.eye(len(x))
        else:
            match options["qn_type"]:
                case "hess":
                    hess = hess_update(hess, s, y, rho, options["qn_update"],
                                       parameter)
                case "inv_hess":
                    inv_hess = inv_hess_update(inv_hess, s, y, rho,
                                               options["qn_update"], parameter)
        it += 1
        # Update evaluation counter from stepsize selection method
        match options["ssc"]:
            case "armijo", "goldstein":
                num_evals[0] += alpha_it
            case "wolfe", "strongwolfe":
                num_evals[0] += alpha_it
                num_evals[1] += alpha_it
        match options["ssm"]:
            case "polyinterp":
                num_evals[0] += 1 + alpha_it
        num_evals[0] += 1
        num_evals[1] += 1
        if parameter.iter_hist == True:
            print("{}\t{:.15e}\t{:.15e}".format(it, fx, grad_norm))
    if grad_norm > parameter.tol and it == parameter.maxit:
        term_flag = "Fail"
    stop_time = time()
    elapsed_time = stop_time - start_time
    return minsearch(x, fx, grad_norm, it, num_evals, term_flag,
                     f"QUASI-NEWTON {options["qn_update"]}",
                     options, param, elapsed_time)

def trustregion(fun: callable, grad: callable, x: list, options: dict,
                 parameter: param) -> minsearch:
    """
    Trust Region Method
    
    Parameters
    ----------
    fun : callable
        objective function
    grad : callable
        gradient of the objective function
    x : list
        initial point
    options : dict
        stepsize selection criterion and method used by the method
    parameter : param
        parameters used by the method

    Returns
    -------
    minsearch
        Result of the method
    """
    # fx = fun(x)
    # gradx = grad(x)
    # grad_norm = np.linalg.norm(gradx)
    # delta = min(grad_norm, parameter.Delta)
    # NewPointFlag = True
    # it = 0
    # while grad_norm > parameter.tol and it < parameter.maxit:
    #     it += 1
    #     if NewPointFlag == True:
    #         hess = hess_approx(grad, x, gradx, parameter.eps)
    #     w = np.dot(gradx, np.dot(hess, gradx))
    #     grad_norm = np.linalg.norm(gradx)
    #     mu1 = grad_norm**2
    #     TrialPointType = "Cauchy"
    #     BoundaryPointFlag = "False"
    #     if w <= 0:
    #         x_cauchy = x - delta/grad_norm*gradx
    #         BoundaryPointFlag = True
    #     else:
    #         ratio1 = delta/grad_norm
    #         ratio2 = mu1/w
    #         sigma = min(ratio1, ratio2)
    #         if ratio2 >= ratio1:
    #             BoundaryPointFlag = True
    #         x_cauchy = x - sigma*gradx
    #         inv_hess = np.linalg.inv(hess)
    #         d_newton = np.dot(inv_hess, gradx)
    #         if np.dot(d_newton, gradx) >= 0:
    #             pass
    #         else:
    #             x_newton = x + d_newton
    #             d_norm = np.linalg.norm(d_newton)
    #             if d_norm <= delta:
    #                 TrialPointType = "Newton"
    #                 if d_norm >= delta - 1e-6:
    #                     BoundaryPointFlag = True
    #             else:
    #                 if BoundaryPointFlag == True:
    #                     pass
    #                 else:
    #                     d_cauchy = -sigma*gradx
    #                     d = d_newton - d_cauchy
    #                     a, b, c = np.linalg.norm(d)**2, 2*np.dot(d_cauchy, d), np.linalg.norm(d_cauchy)**2 - delta**2
    #                     xi = (-b+np.sqrt(b**2 - 4*a*c))/(2*a)
    #                     x_cauchy = x_cauchy + xi*(x_newton - x_cauchy)
    #                     BoundaryPointFlag = True
    #                     TrialPointType = "Dogleg"
    #     if TrialPointType == "Newton":
    #         x_trial = x_newton
    #     else:
    #         x_trial = x_cauchy
    #     d_trial = x_trial - x
    #     f_trial = fun(x_trial)
    #     actual_reduction = f_trial - fx
    #     predicted_reduction = np.dot(gradx, d_trial) + 0.5*np.dot(d_trial, np.dot(hess, d_trial))
    #     rho = actual_reduction / predicted_reduction
    #     if rho >= 1e-4:
    #         x = x_trial
    #         NewPointFlag = True
    #     else:
    #         delta = 0.5*min(delta, np.linalg.norm(d_trial))
    #         NewPointFlag = False
    #     if rho > 0.75 and BoundaryPointFlag == True:
    #         delta = min(2*delta, 1000)
    #     if NewPointFlag == True:
    #         fx = fun(x)
    #         gradx = grad(x)
    #         grad_norm = np.linalg.norm(gradx)
    term_flag = "Success"
    start_time = time()
    
    fx = fun(x)
    gradx = grad(x)
    grad_norm = np.linalg.norm(gradx)
    delta = min(grad_norm, parameter.delta)
    it = 0
    
    while grad_norm > parameter.tol and it < parameter.maxit:
        it += 1
        hessx = hess_approx(grad, x, gradx, parameter.eps)
        d_trial = NotImplemented
        x_trial = x + d_trial
        d_trial = x_trial - x
        actual_reduction = fun(x_trial) - fx
        predicted_reduction = np.dot(gradx, d_trial) \
            + 0.5*np.dot(d_trial, np.dot(hessx, d_trial))
        rho = actual_reduction / predicted_reduction
        if rho >= parameter.eta:
            x = x_trial
        else:
            delta = parameter.sigma_lo*min(delta, np.linalg.norm(d_trial))
        if rho > 0.75:
            delta = min(parameter.sigma_hi*delta, parameter.delta)
        fx = fun(x)
        gradx = grad(x)
        grad_norm = np.linalg.norm(gradx)
    if grad_norm > parameter.tol and it == parameter.maxit:
        term_flag = "Fail"
    stop_time = time()
    elapsed_time = stop_time - start_time
    return minsearch(x, fx, grad_norm, it, [], term_flag,
                     f"{options["region"]}", options, param, elapsed_time)

def unidir(fun, grad, x, parameter):
    k = 0
    fx = fun(x)
    gradx = grad(x)
    grad_norm = np.linalg.norm(gradx)
    delta = min(grad_norm, parameter.delta)
    while grad_norm > parameter.tol and k < parameter.maxit:
        k += 1
        hessx = hess_approx(grad, x, gradx, parameter.eps)
        w = np.dot(gradx, np.dot(hessx, gradx))
        TrialPointType = "Cauchy"
        BoundaryPointFlag = False
        if w <= 0.:
            x_trial = x - delta/grad_norm*gradx
            BoundaryPointFlag = True
        else: 
            ratio1 = delta/grad_norm
            ratio2 = grad_norm**2/w
            alpha = min(ratio1, ratio2)
            if ratio2 >= ratio1:
                BoundaryPointFlag = True
            x_trial = x - alpha*gradx
        d_trial = x_trial - x
        actual_reduction = fun(x_trial) - fx
        predicted_reduction = np.dot(gradx, d_trial) \
            + 0.5*np.dot(d_trial, np.dot(hessx, d_trial))
        rho = actual_reduction/predicted_reduction
        if rho >= parameter.eta:
            x = x_trial
            fx = fun(x)
            gradx = grad(x)
            grad_norm = np.linalg.norm(gradx)
        else:
            delta = parameter.sigma_lo*min(delta, np.linalg.norm(d_trial))
        if rho > parameter.mu_hi and BoundaryPointFlag:
            delta = min(parameter.sigma_hi*delta, parameter.delta)
    return k, x, fx, grad_norm

def dogleg(fun, grad, x, parameter):
    k = 0
    fx = fun(x)
    gradx = grad(x)
    grad_norm = np.linalg.norm(gradx)
    delta = min(grad_norm, parameter.delta)
    while grad_norm > parameter.tol and k < parameter.maxit:
        k += 1
        hessx = hess_approx(grad, x, gradx, parameter.eps)
        w = np.dot(gradx, np.dot(hessx, gradx))
        TrialPointType = "Cauchy"
        BoundaryPointFlag = False
        if w <= 0.:
            x_cauchy = x - delta/grad_norm*gradx
            BoundaryPointFlag = True
        else:
            ratio1 = delta/grad_norm
            ratio2 = grad_norm**2/w
            alpha = min(ratio1, ratio2)
            if ratio2 >= ratio1:
                BoundaryPointFlag = True
            x_cauchy = x - alpha*gradx
            d_newton = np.linalg.solve(hessx, -gradx)
            if np.dot(d_newton, gradx) < 0.:
                x_newton = x + d_newton
                d_norm = np.linalg.norm(d_newton)  
                if d_norm <= delta:
                    TrialPointType = "Newton"
                    if d_norm >= delta - 1e-6:
                        BoundaryPointFlag = True
                else:
                    if not BoundaryPointFlag:
                        d_cauchy = -alpha*gradx
                        d = d_newton - d_cauchy
                        a, b, c = np.linalg.norm(d)**2, 2.*np.dot(d_cauchy, d), \
                            np.linalg.norm(d_cauchy)**2 - delta**2
                        xi = (-b + np.sqrt(b**2 - 4.*a*c)) / (2.*a)
                        x_cauchy = x_cauchy + xi*(x_newton - x_cauchy)
                        BoundaryPointFlag = True
                        TrialPointType = "Dogleg"
        if TrialPointType == "Newton":
            x_trial = x_newton
        else:
            x_trial = x_cauchy
        d_trial = x_trial - x
        actual_reduction = fun(x_trial) - fx
        predicted_reduction = np.dot(gradx, d_trial) \
            + 0.5*np.dot(d_trial, np.dot(hessx, d_trial))
        rho = actual_reduction/predicted_reduction
        if rho >= parameter.eta:
            x = x_trial
            fx = fun(x)
            gradx = grad(x)
            grad_norm = np.linalg.norm(gradx)
        else:
            delta = parameter.sigma_lo*min(delta, np.linalg.norm(d_trial))
        if rho > parameter.mu_hi and BoundaryPointFlag:
            delta = min(parameter.sigma_hi*delta, parameter.delta)
            
    return k, x, fx, grad_norm
        
def unidir(fun, grad, x, parameter):
    k = 0
    fx = fun(x)
    gradx = grad(x)
    grad_norm = np.linalg.norm(gradx)
    delta = min(grad_norm, parameter.delta)
    while grad_norm > parameter.tol and k < parameter.maxit:
        k += 1
        hessx = hess_approx(grad, x, gradx, parameter.eps)
        w = np.dot(gradx, np.dot(hessx, gradx))
        BoundaryPointFlag = False
        if w <= 0.:
            x_trial = x - delta/grad_norm*gradx
            BoundaryPointFlag = True
        else: 
            ratio1 = delta/grad_norm
            ratio2 = grad_norm**2/w
            alpha = min(ratio1, ratio2)
            if ratio2 >= ratio1:
                BoundaryPointFlag = True
            x_trial = x - alpha*gradx
        d_trial = x_trial - x
        actual_reduction = fun(x_trial) - fx
        predicted_reduction = np.dot(gradx, d_trial) \
            + 0.5*np.dot(d_trial, np.dot(hessx, d_trial))
        rho = actual_reduction/predicted_reduction
        if rho >= parameter.eta:
            x = x_trial
            fx = fun(x)
            gradx = grad(x)
            grad_norm = np.linalg.norm(gradx)
        else:
            delta = parameter.sigma_lo*min(delta, np.linalg.norm(d_trial))
        if rho > parameter.mu_hi and BoundaryPointFlag:
            delta = min(parameter.sigma_hi*delta, parameter.delta)
    return k, x, fx, grad_norm

# Stepsize Selection Method
def stepsize_strategy(fun: callable, grad: callable, x: list, fx: float,
                      gradx: list, d: list, ssm: str, ssc: str,
                      parameter: param) -> tuple[float, int]:
    """
    Helper function for the stopping criteria of minimizer search algorithms.
        Method is to be indicated at ssm variable with valid values
        backtrack and polyinterp for Backtracking and Polynomial 
        Interpolation, respectively.
    
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
        ssm : str
            method for stepsize strategy
        ssc : str
            criterion for stepsize strategy
        parameter : param
            parameter to be passed for the method
    
    Returns
    -------
        stepsize strategy executed
    """
    match ssm:
        case "backtrack":
            return backtrack(fun, grad, x, fx, gradx, d, ssc, parameter)
        case "polyinterp":
            return polyinterp(fun, grad, x, fx, gradx, d, ssc, parameter)
    
def backtrack(fun: callable, grad: callable, x: list, fx: float, gradx: list,
              d: list, ssc: str, parameter: param) -> tuple[float, int]:
    """
    Returns
    -------
    tuple[float, int]
        alpha : float
            steplength satisfying the specified rule or the last steplength
        j : int
            number of backtracking steps
    """
    alpha = parameter.alpha_in
    q = np.dot(gradx, d)
    j = 0
    while stopping_criterion(fun, grad, x, fx, d, alpha, q, ssc, parameter)\
            and j < parameter.maxback:
        alpha = parameter.rho * alpha
        j += 1
    return alpha, j

def polyinterp(fun: callable, grad: callable, x: list, fx: float, gradx: list,
               d: list, ssc: str, parameter: param) -> tuple[float, int]:
    """
    Returns
    -------
    tuple (alpha, j)
        alpha : float
            steplength satisfying the specified rule or the last steplength
        j : int
            number of interpolations
    """
    alpha = parameter.alpha_in
    q = np.dot(gradx, d)
    f = fun(x + alpha*d)
    j = 0
    while stopping_criterion(fun, grad, x, fx, d, alpha, q, ssc, parameter)\
            and j < parameter.maxback:
        if j == 0:
            temp = -q*alpha**2 / (2.*(f - fx - alpha*q))
        else:
            A = [[alpha_old**2, alpha_old**3], [alpha**2, alpha**3]]
            b = [f_old - fx - alpha_old*q, f - fx - alpha*q]
            c0, c1 = np.linalg.solve(A, b)
            if c0**2 - 3.*c1*q >= 0 and abs(c1) > 1e-10:
                temp = (-c0 + np.sqrt(c0**2 - 3.*c1*q)) / (3.*c1)
            else:
                temp = alpha
        alpha_old = alpha
        f_old = f
        alpha = max(parameter.rho_l*alpha, min(temp, parameter.rho_r*alpha))
        f = fun(x + alpha*d)
        j += 1
    return alpha, j


# Stepsize Selection Criterion
def stopping_criterion(fun: callable, grad: callable, x: list, fx: float,
                       d: list, alpha: float, q: float, ssc: str,
                       parameter: param) -> bool:
    """
    Helper function for the line search step of minimizer search algorithms.
        Criterion is to be indicated at ssc variable with valid values
        armijo, goldstein, wolfe, and strongwolfe for Armijo Rule, Goldstein 
        Rule, Wolfe and Strong Wolfe Rule, respectively.

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
    ssc : str
        criterion for stepsize strategy
    parameter : param
        parameter to be passed for the method

    Returns
    -------
    bool
        either set stepsize criterion satisfied or not  
    """
    match ssc:
        case "armijo":
            return armijo(fun, x, fx, d, alpha, q, parameter.c1)
        case "goldstein":
            return goldstein(fun, x, fx, d, alpha, q, parameter.c)
        case "wolfe":
            return wolfe(fun, grad, x, fx, d, alpha, q, parameter.c1,
                         parameter.c2)
        case "strongwolfe":
            return strongwolfe(fun, grad, x, fx, d, alpha, q, parameter.c1,
                         parameter.c2)

def armijo(fun: callable, x: list, fx: float, d: list, alpha: float, q: float,
           c1: float) -> bool:
    """
    Returns
    -------
        bool, if Armijo rule is satisfied return false, otherwise true
    """
    LHS = fun(x + alpha*d)
    RHS = fx + c1*alpha*q
    return LHS > RHS

def goldstein(fun: callable, x: list, fx: float, d: list, alpha: float,
              q: float, c: float) -> bool:
    """
    Returns
    -------
        bool, if Goldstein rule is satisfied return false, otherwise true
    """
    LHS = fx + (1 - c)*alpha*q
    Ctr = fun(x + alpha*d)
    RHS = fx + c*alpha*q
    return LHS > Ctr or Ctr > RHS

def wolfe(fun: callable, grad: callable, x: list, fx: float, d: list,
          alpha: float, q: float, c1: float, c2: float) -> bool:
    """
    Returns
    -------
        bool, if Wolfe rule is satisfied return false, otherwise true
    """
    LHS = np.dot(grad(x + alpha*d), d)
    RHS = c2*q
    return armijo(fun, x, fx, d, alpha, q, c1) or LHS < RHS

def strongwolfe(fun, grad, x, fx, d, alpha, q, c1: float, c2: float) -> bool:
    """
    Returns
    -------
        bool, if Strong Wolfe rule is satisfied return false, otherwise true
    """
    LHS = abs(np.dot(grad(x + alpha*d), d))
    RHS = -c2*q
    return armijo(fun, x, fx, d, alpha, q, c1) or LHS > RHS


# Numerical gradient and hessian for Newton
def num_grad(fun: callable, x: list, h: float=1e-6) -> list:
    """
    Numerical Gradient of the objective function fun at point x using 
        centered finite difference with stepsize h
          
    Parameters
    ----------
    fun : callable
        objective function
    x : list
        initial point
    h : float, optional
        step size h from x, by default 1e-6

    Returns
    -------
    list
        numerical gradient of objective function at x
    """
    x = np.array(x)
    n = len(x)
    num_grad = np.zeros(n)
    for i in range(n):
        stepsize = np.zeros(n)
        stepsize[i] = h
        df_xi = (fun(x + stepsize) - fun(x - stepsize)) / (2.*h)
        num_grad[i] = df_xi
    return num_grad

def num_hess(fun: callable, x: list, h: float=1e-6) -> list:
    """
    Numerical Hessian of the objective function fun at point x using 
        centered finite difference with stepsize h
          
    Parameters
    ----------
    fun : callable
        objective function
    x : list
        initial point
    h : float, optional
        step size h from x, by default 1e-6

    Returns
    -------
    list
        numerical hessian of objective function at x
    """
    n = len(x)
    num_hess = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if i == j: 
                stepsize = 2.*h*np.eye(n)[i]
                num_hess[i][j] = (fun(x + stepsize) - 2.*fun(x)\
                    + fun(x - stepsize)) / (4.*h**2)
            else:
                stepsize1 = h*(np.eye(n)[i] + np.eye(n)[j])
                stepsize2 = h*(np.eye(n)[i] - np.eye(n)[j])
                df_xixj = (fun(x + stepsize1) - fun(x + stepsize2)\
                    - fun(x - stepsize2) + fun(x - stepsize1)) / (4.*h**2)
                num_hess[i][j] = num_hess[j][i] = df_xixj
    return num_hess


# Beta computations for Nonlinear Conjugate Gradient
def beta_formula(gradx_old: list, gradx: list, grad_norm_old: float,
                 grad_norm: float, d: list, method: str= "fr") -> float:
    """
    Beta formulas for Nonlinear Conjugate Gradient Methods

    Parameters
    ----------
    gradx_old : list
        gradient at the previous iterate
    gradx : list
        gradient at the current iterate
    grad_norm_old : float
        norm of the gradient at the previous iterate
    grad_norm : float
        norm of the gradient at the current iterate
    d : list
        current direction
    method : str, optional
        Formula to be used for the update of the direction, by default "fr"

    Returns
    -------
    float
        scalar multiple of the current direction for the update of direction
    """
    match method:
        case "fr":
            beta = grad_norm**2 / grad_norm_old**2
        case "pr":
            beta = np.dot(gradx, gradx - gradx_old) / grad_norm_old
        case "prgn":
            beta = np.dot(gradx, gradx - gradx_old) / grad_norm_old
            beta = max(0, beta)
        case "frpr":
            beta_pr = np.dot(gradx, gradx - gradx_old) / grad_norm_old
            beta_fr = grad_norm**2 / grad_norm_old**2
            if abs(beta_pr) <= beta_fr:
                beta = beta_pr
            elif beta_pr < -beta_fr:
                beta = -beta_fr
            elif beta_pr > beta_fr:
                beta = beta_fr
        case "hs":
            y = gradx - gradx_old
            beta = np.dot(gradx, y) / np.dot(d, y)
        case "dy":
            beta = grad_norm**2/np.dot(d, gradx - gradx_old)
        case "hz":
            y = gradx - gradx_old
            w = np.dot(d, y)
            beta = np.dot(y - 2.*(np.linalg.norm(y)**2/w)*d, gradx/w)
    return beta


# Alpha computations for Barzilai-Borwein
def alpha_formula(s: list, y: list, k: int, ver: str) -> float:
    """
    Alpha formulas for Barzilai-Borwein Nonlinear Conjugate Gradient Methods
    
    Parameters
    ----------
    s : list
        change in iterates
    y : list
        change in the gradient of iterates
    k : int
        current number of iterations
    ver : str
        version of the method
        
    Returns
    -------
    float
        step size of the method
    """
    match ver:
        case "v1":
            alpha = np.dot(s, y)/(np.linalg.norm(y)**2)
        case "v2":
            alpha = np.linalg.norm(s)**2/np.dot(s, y)
        case "v3":
            if k%2:
                alpha = np.dot(s, y)/(np.linalg.norm(y)**2)
            else:
                alpha = np.linalg.norm(s)**2/np.dot(s, y)
        case "v4":
            if k%2:
                alpha = np.linalg.norm(s)**2/np.dot(s, y)
            else:
                alpha = np.dot(s, y)/(np.linalg.norm(y)**2)
    return alpha

# Update for the Approximate Hessian or Inverse of the Hessian for Quasi-Newton
def hess_update(hess: list, s: list, y: list, rho: float, method: str,
                parameter: param) -> list:
    """
    Update for the Hessian of the Objective Function at the current iterate

    Parameters
    ----------
    hess : list
        current hessian approximation
    s : list
        change in the iterates
    y : list
        change in the gradient of the iterates
    rho : float
        dot product of the s and y
    method : str
        Method to use in updating the hessian
    parameter : param
        parameters used by the method

    Returns
    -------
    list
        Next Approximate Hessian 
    """
    match method:
        case "bfgs":
            w = np.dot(hess, s)
            hess = hess - np.outer(w, w) / np.dot(s, w) + np.outer(y, y) / rho
        case "sr1":
            w = y - np.dot(hess, s)
            hess = hess + np.outer(w, w) / np.dot(w, s)
        case "nsr1":
            w = y - np.dot(hess, s)
            hess = hess + np.outer(w, s) / np.dot(s, s)
        case "dfp":
            I = np.eye(len(s))
            w = np.outer(y, s)/rho
            hess = np.dot(I-w, np.dot(hess, I-w.T)) + np.outer(y, y) / rho
        case "spb":
            w = y - np.dot(hess, s)
            hess = hess + (np.outer(w, s) + np.outer(s, w)) / np.dot(s, s) \
                - np.dot(w, s) / np.dot(s, s)**2 * np.outer(s, s)
        case "bfgsdfp":
            temp = hess
            hess = parameter.theta*hess_update(hess, s, y, rho, "bfgs",
                                               parameter)
            hess = hess + (1-parameter.theta)*hess_update(temp, s, y, rho,
                                                          "dfp", parameter)
    return hess

def inv_hess_update(inv_hess: list, s: list, y: list, rho: float, method: str,
                    parameter: param) -> list:
    """
    Update for the Inverser of the Hessian of the Objective Function
        at the current iterate

    Parameters
    ----------
    inv_hess : list
        inverse of the current hessian approximation 
    s : list
        change in the iterates
    y : list
        change in the gradient of the iterates
    rho : float
        dot product of the s and y
    method : str
        Method to use in updating the inverse of the hessian
    parameter : param
        parameters used by the method

    Returns
    -------
    list
        Next Inverse of the Approximate Hessian
    """
    match method:
        case "bfgs":
            I = np.eye(len(s))
            w = np.outer(s, y) / rho
            inv_hess = np.dot(np.dot(I - w, inv_hess), I - w.T) \
                + np.outer(s, s) / rho
        case "sr1":
            w = s - np.dot(inv_hess, y)
            inv_hess = inv_hess + np.outer(w, w) / np.dot(w, y)
        case "nsr1":
            w = s - np.dot(inv_hess, y)
            inv_hess = inv_hess + np.outer(w, y) / np.dot(y, y)
        case "dfp":
            w = np.dot(inv_hess, y)
            inv_hess = inv_hess - np.outer(w, w) / np.dot(y, w) \
                + np.dot(s, s) / rho
        case "gr":
            w = s - np.dot(inv_hess, y)
            inv_hess = inv_hess\
                + (np.outer(w, y) + np.outer(y, w)) / np.dot(y, y)\
                - np.dot(w, y) / np.dot(y, y)**2 * np.outer(y, y)
        case "bfgsdfp":
            temp = inv_hess
            inv_hess = parameter.theta*inv_hess_update(inv_hess, s, y, rho,
                                                       "bfgs", parameter)
            inv_hess = inv_hess + (1-parameter.theta)*inv_hess_update(temp, s,
                                                                      y, rho,
                                                                      "dfp",
                                                                      parameter)
        case "ol":
            I = np.eye(len(s))
            w1 = np.outer(s, y) / rho
            w2 = np.dot(inv_hess, y)
            inv_hess = parameter.theta*parameter.gamma \
                * np.dot(np.dot(I - w1, inv_hess), I - w1.T) \
                + (1 - parameter.theta)*parameter.gamma * \
                (inv_hess - np.outer(w2, w2) / np.dot(y, w2)) \
                + np.outer(s,s) / rho
        case "ss":
            w1 = np.dot(inv_hess, y)
            w2 = np.dot(y, w1)
            v = s / rho - w1 / w2
            inv_hess = rho / w2 \
                * (inv_hess - np.outer(w1, w1) / w2 + w2 * np.outer(v, v)) \
                + np.dot(s, s) / rho
        case "ho":
            w1 = np.dot(inv_hess, y)
            w2 = np.dot(y, w1)
            v = s / rho - w1 / w2
            inv_hess = inv_hess + np.outer(s, s) + (rho * w2) / (rho + w2) \
                * np.outer(v, v) - np.outer(w1, w1) / w2
        case "mh":
            t = np.dot(inv_hess, y)
            u = s + t
            w = 0.1*s + 0.1*t
            inv_hess = inv_hess + np.outer(u, s) / np.dot(u, y) \
                - np.outer(w, t) / np.dot(w, y)
        case "ssvm":
            u = np.dot(inv_hess, y)
            w = np.linalg.solve(inv_hess, s)
            if rho / np.dot(y, np.dot(inv_hess, y)) > 1:
                phi, omega = 1., 0.
            elif np.dot(s, w) / rho < 1:
                phi, omega = 0., 1.
            else:
                phi = omega = rho*(np.dot(y, u) - rho) / \
                    ((np.dot(s, w))*(np.dot(y, u)) - rho**2)
            gamma = (1 - omega)*rho / (np.dot(y, u)) \
                + omega*np.dot(s, w) / rho
            inv_hess = gamma * (inv_hess - np.outer(u, u) / np.dot(y, u)) \
                + phi * (np.dot(y, u)) * np.outer(u, u) + np.outer(s, s) / rho
    return inv_hess

# Alternative Hessian Approximation for Trust Region Methods
def hess_action(grad: callable, x: list, gradx: float, d: list,
                eps: float=1e-6) -> float:
    """
    Approximate the action of the Hessian at point x in the direction of d

    Parameters
    ----------
    grad : callable
        gradient of the objective function
    x : list
        current point
    gradx : float
        gradient of the objective function at the current point
    d : list
        current direction
    eps : float
        step size

    Returns
    -------
    float
        approximate value of the hessian at x
    """
    d_norm = np.linalg.norm(d)
    if d_norm == 0:
        z = np.zeros(len(x))
    else:
        gradx_temp = grad(x + eps*d/d_norm)
        z = (gradx_temp - gradx)/eps
    return z

def hess_approx(grad: callable, x: list, gradx: list, eps: float=1e-6) -> list:
    """
    Approximate the Hessian at point x in the direction of d

    Parameters
    ----------
    grad : callable
        gradient of the objective function
    x : list
        current point
    gradx : float
        gradient of the objective function at the current point
    d : list
        current direction
    eps : float
        step size

    Returns
    -------
    list
        Approximate Hessian at x
    """
    n = len(x)
    hess = np.zeros([n,n])
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1
        hess[:, j] = hess_action(grad, x, gradx, e, eps)
    return (hess + hess.T)/2

"""
    Minimizer search for quadratic functions f: R^n -> R defined by
        f(x) = 0.5x^TAx - b^Tx + c
        where A is n-by-n SPD matrix, b is n-dimensional vector and c is scalar
"""
class quadminsearch:
    """
    Class for the result of the minimizer search algorithm for quadratic
        functions of the form x^TAx + b^Tx + c
        where A is an n by n SPD matrix, b is an n dimensional vector and c
        is a real number  
    
    Attributes
    ----------
        x               : list
            approximate minimizer of the function
        fx              : float
            function value of the minimizer
        grad_norm       : float
            norm of the gradient of x under the function
        it              : int
            number of iterations the method took
        method          : str
            method used to search for a minimizer
        parameter       : param
            parameters used by the method
        elapsed_time    : float
            time in seconds the method took    
    """
    def __init__(self, x, fx, grad_norm, it, method, param, elapsed_time):
        self.x = x
        self.fx = fx
        self.grad_norm = grad_norm
        self.it = it
        self.method = method
        self.param = param
        self.elapsed_time = elapsed_time
        
    def __str__(self):
        list_str_repr = [
            "METHOD                          : {}".format(self.method),
            "APPROXIMATE MINIMIZER           : {}".format(self.x),
            "FUNCTION VALUE                  : {}".format(self.fx),
            "GRADIENT NORM OF MINIMIZER      : {}".format(self.grad_norm),
            "TOLERANCE                       : {}".format(self.param.tol),
            "NITERS                          : {}".format(self.it),
            "ELAPSED TIME                    : {} seconds".format(self.elapsed_time)
        ]
        return "\n".join(list_str_repr)
    
def SD(A: list, b: list, c: float, x: list, parameter: param) -> quadminsearch:
    """Steepest Descent Method for quadratic functions

    Parameters
    ----------
    A : list
        SPD matrix
    b : list
        n-dim vector
    c : float
        constant term
    x : list
        initial iterate 
    parameter : param
        Parameters of the method

    Returns
    -------
    quadminsearch
        Result of the method
    """
    start_time = time()
    k = 0
    gradx = np.dot(A, x) - b
    grad_norm = np.linalg.norm(gradx)
    while grad_norm > parameter.tol:
        d = -gradx
        alpha = np.dot(d, d) / np.dot(d, np.dot(A, d))
        x = x + alpha*d
        gradx = np.dot(A, x) - b
        grad_norm = np.linalg.norm(gradx)
        k += 1
    funval = 0.5*np.dot(x, np.dot(A, x)) - np.dot(b, x) + c
    stop_time = time()
    elapsed_time = stop_time - start_time
    return quadminsearch(x, funval, grad_norm, k, "Steepest Descent", parameter,
                         elapsed_time)

def CG(A: list, b: list, c: float, x: list, parameter: param) -> quadminsearch:
    """Conjugate Gradient method for quadratic functions

    Parameters
    ----------
    A : list
        SPD matrix
    b : list
        n-dim vector
    c : float
        constant term
    x : list
        initial iterate 
    parameter : param
        Parameters of the method

    Returns
    -------
    quadminsearch
        Result of the method
    """
    start_time = time()
    k = 0
    gradx = np.dot(A, x) - b
    grad_norm = np.linalg.norm(gradx)
    d = -gradx
    while grad_norm > parameter.tol:
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
    stop_time = time()
    elapsed_time = stop_time - start_time
    return quadminsearch(x, funval, grad_norm, k, "Conjugate Gradient",
                         parameter, elapsed_time)
    
def CGNR(A: list, b: list, c: float, x: list, parameter: param) -> quadminsearch:
    """Conjugate Gradient Normal Residual method for quadratic functions

    Parameters
    ----------
    A : list
        SPD matrix
    b : list
        n-dim vector
    c : float
        constant term
    x : list
        initial iterate 
    parameter : param
        Parameters of the method

    Returns
    -------
    quadminsearch
        Result of the method
    """
    start_time = time()
    k = 0
    gradx = np.dot(A, x) - b
    grad_norm = np.linalg.norm(gradx)
    d = -np.dot(A, gradx)
    z = -d
    while grad_norm > parameter.tol:
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
    stop_time = time()
    elapsed_time = stop_time - start_time
    return quadminsearch(x, funval, grad_norm, k, "Conjugate Normal", parameter,
                         elapsed_time)
    
def CGR(A: list, b: list, c: float, x: list, parameter: param) -> quadminsearch:
    """Conjugate Gradient Residual method for quadratic functions

    Parameters
    ----------
    A : list
        SPD matrix
    b : list
        n-dim vector
    c : float
        constant term
    x : list
        initial iterate 
    parameter : param
        Parameters of the method

    Returns
    -------
    quadminsearch
        Result of the method
    """
    start_time = time()
    k = 0
    gradx = np.dot(A, x) - b
    grad_norm = np.linalg.norm(gradx)
    d = -gradx
    while grad_norm > parameter.tol:
        w = np.dot(A, d)
        alpha = -np.dot(gradx, w)/(np.dot(w, w))
        x = x + alpha*d
        gradx = gradx + alpha*w
        grad_norm = np.linalg.norm(gradx)
        beta = np.dot(np.dot(gradx, A), w)/np.dot(w, w)
        d = -gradx + beta*d
        k += 1
    funval = 0.5*np.dot(x, np.dot(A, x)) - np.dot(b, x) + c
    stop_time = time()
    elapsed_time = stop_time - start_time
    return quadminsearch(x, funval, grad_norm, k, "Conjugate Residual",
                         parameter, elapsed_time)