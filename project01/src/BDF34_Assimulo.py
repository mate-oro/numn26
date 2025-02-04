from assimulo.explicit_ode import Explicit_ODE # type: ignore
from assimulo.ode import NORMAL, ID_PY_OK # type: ignore
from assimulo.exception import Explicit_ODE_Exception
import numpy as np
from scipy.optimize import fsolve
from scipy.linalg import norm
from typing import Any

class BDF_34(Explicit_ODE):
    """
    BDF-3 & 4
    """
    tol = 1.e-6  
    maxit = 100     
    maxsteps = int(20000)
    errorest = []
    
    def __init__(self, problem):
        Explicit_ODE.__init__(self, problem) #Calls the base class
        
        #Solver options
        self.options["h"] = 0.01
        self.options["ord"] = 3
        
        #Statistics
        self.statistics["nsteps"] = 0
        self.statistics["nfcns"] = 0
    
    @property
    def h(self) -> float:
        return self.options["h"]

    @h.setter
    def h(self, h: float):
            self.options["h"] = float(h)
    
    @property
    def ord(self) -> int:
        return self.options["ord"]
    
    @ord.setter
    def ord(self, ord:int):
            if not ((ord == 3) or (ord == 4)):
                raise ValueError("The parameter 'ord' must be int(3) or int(4)!")
            self.options["ord"] = int(ord)

        
    def integrate(self, t: float, y: np.ndarray, tf: float, opts) -> tuple[Any, list[float], list[np.ndarray]]:
        """
        _integrates (t,y) values until t > tf
        """
        h = self.options["h"]
        h = min(h, abs(tf-t))
        ord = self.options["ord"]
        y = np.atleast_1d(y)
        
        #Lists for storing the result
        tres = []
        yres = []

        #Lists for storing tempoary data
        T = [t]
        Y = [y]
        
        if ord == 3:
            for i in range(self.maxsteps):
                if t >= tf:
                    break
                self.statistics["nsteps"] += 1
                
                if i == 0:  # initial step
                    t_np1, y_np1, f_n = self.step_EE(t, y, h)
                    Y.insert(0, y_np1)
                elif i == 1: # getting the second initial step
                    t_np1, y_np1, f_n = self.step_BDF2(t, Y, h)
                    Y.insert(0, y_np1)
                    F = [f_n]
                else: 
                    t_np1, y_np1, f_n = self.step_BDF3(t, Y, h, F) # type: ignore
                    Y = [y_np1] + Y[:2]
                    F = [f_n]
                t = t_np1

                tres.append(t)
                yres.append(y_np1.copy())
            
                h = min(self.h, np.abs(tf-t))
            else:
                raise Explicit_ODE_Exception('Final time not reached within maximum number of steps') # type: ignore
        else:
            for i in range(self.maxsteps):
                if t >= tf:
                    break
                self.statistics["nsteps"] += 1
                
                if i == 0:  # initial step
                    t_np1, y_np1, f_n = self.step_EE(t, y, h)
                    Y.insert(0, y_np1)
                elif i == 1: # getting the second initial step
                    t_np1, y_np1, f_n = self.step_BDF2(t, Y, h)
                    Y.insert(0, y_np1)
                    F = [f_n]
                elif i == 2:
                    t_np1, y_np1, f_n = self.step_BDF3(t, Y, h, F)
                    Y.insert(0, y_np1)
                    F.insert(0, f_n)
                else:   
                    t_np1, y_np1, f_n = self.step_BDF4(t, Y, h, F)
                    Y = [y_np1] + Y[:3]
                    F = [f_n] + F[:1]
                t = t_np1

                tres.append(t)
                yres.append(y_np1.copy())
            
                h = min(self.h, np.abs(tf-t))
            else:
                raise Explicit_ODE_Exception('Final time not reached within maximum number of steps') # type: ignore
        
        return ID_PY_OK, tres, yres
    
    def step_EE(self, t: float, y: np.ndarray, h: float) -> tuple[float, np.ndarray, np.ndarray]:
        """
        This calculates the next step in the integration with explicit Euler.
        """
        self.statistics["nfcns"] += 1
        
        f = self.problem.rhs
        f_n = f(t, y)

        return t + h, y + h * f_n, f_n

    def step_AB2(self, t: float, y: np.ndarray, F: list[np.ndarray], h: float) -> tuple[float, np.ndarray, np.ndarray]:
        """
        Adams-Bashforth 2 method for explicit predictor

        """
        
        # Parsing input
        if len(F) != 1:
            raise ValueError(f"Expected F to have 1 elements, but got {len(F)}")
        
        f_nm1 = F[0]

        self.statistics["nfcns"] += 1
        f = self.problem.rhs
        f_n = f(t, y)

        return t + h, y + h * (3/2 * f_n - 1/2 * f_nm1), f_n
    
    def step_AB3(self, t: float, y: np.ndarray, F: list[np.ndarray], h: float) -> tuple[float, np.ndarray, np.ndarray]:
        """
        Adams-Bashforth 3 method for explicit predictor

        """
        
        # Parsing input
        if len(F) != 2:
            raise ValueError(f"Expected F to have 2 elements, but got {len(F)}")
        
        f_nm1, f_nm2 = F

        self.statistics["nfcns"] += 1
        f = self.problem.rhs
        f_n = f(t, y)

        return t + h, y + h * (23/12 * f_n - 4/3 * f_nm1 + 5/12 * f_nm2), f_n
    
    def step_AB4(self, t: float, y: np.ndarray, F: list[np.ndarray], h: float) -> tuple[float, np.ndarray, np.ndarray]:
        """
        Adams-Bashforth 4 method for explicit predictor

        """
        
        # Parsing input
        if len(F) != 3:
            raise ValueError(f"Expected F to have 3 elements, but got {len(F)}")
        
        f_nm1, f_nm2, f_nm3 = F

        self.statistics["nfcns"] += 1
        f = self.problem.rhs
        f_n = f(t, y)

        return t + h, y + h * (55/24 * f_n - 59/24 * f_nm1 + 37/24 * f_nm2 - 3/8 * f_nm3), f_n

    def step_BDF2(self, t: float, Y: list[np.ndarray], h: float) -> tuple[float, np.ndarray, np.ndarray]:
        """
        BDF-2 with Fixed Point Iteration and Zero order predictor

        """
        alpha = [3./2., -2., 1./2.]
        f = self.problem.rhs
        
        # Parsing inputs
        t_n = t
        y_n, y_nm1 = Y

        # predictor
        t_np1, y_np1, f_n = self.step_EE(t_n, y_n, h)

        # corrector with newton iteration (fsolve)
        residue = lambda y_np1: alpha[0] * y_np1 + alpha[1] * y_n + alpha[2] * y_nm1 - h * f(t_np1, y_np1)
        solution, infodict, ier, _ = fsolve(residue, x0=y_np1, full_output=True, maxfev=np.size(y_n)*self.maxit, xtol=self.tol)
        self.statistics["nfcns"] += infodict["nfev"]

        # Checking the result
        if ier == 1:
            self.errorest.append(norm(solution - y_np1))
            return t_np1, solution, f_n
        else:
            raise Explicit_ODE_Exception('Corrector could not converge within the set iterations') # type: ignore
        
    def step_BDF3(self, t: float, Y: list[np.ndarray], h: float, F: list[np.ndarray] | None = None) -> tuple[float, np.ndarray, np.ndarray]:
        """
        BDF-3 with Fixed Point Iteration and Zero order predictor

        """
        alpha = [11./6., -3., 3./2., -1./3.]
        f = self.problem.rhs
        
        # Parsing inputs
        t_n = t
        y_n, y_nm1, y_nm2 = Y

        if F is None:
            t_np1, y_np1, f_n = self.step_EE(t_n, y_n, h)
        else:
            t_np1, y_np1, f_n = self.step_AB2(t_n, y_n, F, h)

        # corrector with newton iteration (fsolve)
        residue = lambda y_np1: alpha[0] * y_np1 + alpha[1] * y_n + alpha[2] * y_nm1 + alpha[3] * y_nm2 - h * f(t_np1, y_np1)
        solution, infodict, ier, _ = fsolve(residue, x0=y_np1, full_output=True, maxfev=np.size(y_n)*self.maxit, xtol=self.tol)
        self.statistics["nfcns"] += infodict["nfev"]

        # Checking the result
        if ier == 1:
            self.errorest.append(norm(solution - y_np1))
            return t_np1, solution, f_n
        else:
            raise Explicit_ODE_Exception('Corrector could not converge within the set iterations') # type: ignore

    def step_BDF4(self, t: float, Y: list[np.ndarray], h: float, F: list[np.ndarray] | None = None) -> tuple[float, np.ndarray, np.ndarray]:
        """
        BDF-4 with Fixed Point Iteration and Zero order predictor

        """
        alpha = [25./12., -4., 3., -4./3., 1./4.]
        f = self.problem.rhs
        
        # Parsing inputs
        t_n = t
        y_n, y_nm1, y_nm2, y_nm3 = Y

        # predictor
        if F is None:
            t_np1, y_np1, f_n = self.step_EE(t_n, y_n, h)
        else:
            t_np1, y_np1, f_n = self.step_AB3(t_n, y_n, F, h)

        # corrector with newton iteration (fsolve)
        residue = lambda y_np1: alpha[0] * y_np1 + alpha[1] * y_n + alpha[2] * y_nm1 + alpha[3] * y_nm2 + alpha[4] * y_nm3 - h * f(t_np1, y_np1)
        solution, infodict, ier, _ = fsolve(residue, x0=y_np1, full_output=True, maxfev=np.size(y_n)*self.maxit, xtol=self.tol)
        self.statistics["nfcns"] += infodict["nfev"]

        # Checking the result
        if ier == 1:
            self.errorest.append(norm(solution - y_np1))
            return t_np1, solution, f_n
        else:
            raise Explicit_ODE_Exception('Corrector could not converge within the set iterations') # type: ignore
            
    def print_statistics(self, verbose=NORMAL): # type: ignore
        self.log_message('Final Run Statistics            : {name} \n'.format(name=self.problem.name),        verbose)
        self.log_message(' Step-length                    : {stepsize} '.format(stepsize=self.options["h"]), verbose)
        self.log_message(' Number of Steps                : '+str(self.statistics["nsteps"]),          verbose)               
        self.log_message(' Number of Function Evaluations : '+str(self.statistics["nfcns"]),         verbose)
            
        self.log_message('\nSolver options:\n',                                    verbose)
        self.log_message(' Solver            : BDF2',                     verbose)
        self.log_message(' Solver type       : Fixed step\n',                      verbose)
