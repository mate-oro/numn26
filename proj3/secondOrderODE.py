import assimulo.problem as ap
from assimulo.exception import Explicit_ODE_Exception
from assimulo.explicit_ode import Explicit_ODE # type: ignore
from assimulo.ode import NORMAL, ID_PY_OK # type: ignore
import numpy as np
import scipy.sparse.linalg as ssl
from scipy.optimize import fsolve
from scipy.sparse import sparray, csr_array, eye_array
from collections.abc import Callable
from warnings import warn

class Explicit_Problem_2nd_nonlin(ap.Explicit_Problem): # type: ignore
    """
    A class capable of expressing second order IVPs of the form
    u'' = f(t, u, u')   for t in [t0, tf]
    u(t0)   = u0
    u'(t0)  = up0

    With damping turned on f(t, u, u') = f(t, u) is assumed.
    """
    def __init__(self,
                 ndofs: int,
                 f: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
                 u0: np.ndarray,
                 up0: np.ndarray,
                 damping: bool = True,
                 t0: float = 0.,
                 name: str = "Explicit Second Order Problem"):
        self.ndofs = ndofs
        self.f = f
        self.u0 = np.atleast_1d(u0)
        self.up0 = np.atleast_1d(up0)
        self.damping = damping
        self.t0 = t0
        self.name = name

        # the "y" notation is kept for traditional solvers
        self.y0 = np.hstack((self.u0, self.up0))
    
    def rhs(self, t, y):
        yp = y[self.ndofs:]
        ypp = self.f(t, y[:self.ndofs], y[self.ndofs:])
        return np.hstack((yp, ypp))

class Second_Order(Explicit_ODE):
    
    def __init__(self, problem: Explicit_Problem_2nd_nonlin):
        super().__init__(problem)
        
        # Solver options
        self.options["h"] = 0.01
        self.options["damping"] = problem.damping
        self.options["maxsteps"] = int(500000)
        
        # Statistics
        self.statistics["nsteps"] = 0
        self.statistics["nfcns"] = 0
    
    @property
    def h(self) -> float:
        return self.options["h"]
    @h.setter
    def h(self, h: float):
            self.options["h"] = float(h)
    
    @property
    def maxsteps(self) -> int:
        return self.options["maxsteps"]
    @maxsteps.setter
    def maxsteps(self, maxsteps: int):
            self.options["maxsteps"] = maxsteps

    @property
    def damping(self) -> bool:
        return self.options["damping"]
    @damping.setter
    def damping(self, damping: bool):
            self.options["damping"] = damping

class Newmark(Second_Order):
    def __init__(self, problem):
        super().__init__(problem)

        self.options["beta"] = 0.25
        self.options["gamma"] = 0.5

    # Setting up the method parameters
    @property
    def beta(self) -> float:
        return self.options["beta"]
    @beta.setter
    def beta(self, beta: float):
            if beta < 0 or beta > 1/2:
                 raise ValueError("\"beta\" must be in the range [0, 1/2]")
            self.options["beta"] = beta

    @property
    def gamma(self) -> float:
        return self.options["gamma"]
    @gamma.setter
    def gamma(self, gamma: float):
            if gamma < 0 or gamma > 1.:
                 raise ValueError("\"gamma\" must be in the range [0, 1]")
            self.options["gamma"] = gamma
    
    def integrate(self, t: float, y: np.ndarray, tf: float, opts) -> tuple[ID_PY_OK, list[float], list[np.ndarray]]:
        """
        integrates (t,y) values until t > tf
        """
        h = self.h
        h = min(h, abs(tf - t))
        y = np.atleast_1d(y)
        f = self.problem.f

        # Separating components
        p_n = y[:self.problem.ndofs]
        v_n = y[self.problem.ndofs:]

        # Computing initial a
        a_n = f(t, p_n, v_n)
        self.statistics["nfcns"] += 1

        #Lists for storing the result
        tres = []
        yres = []

        for i in range(self.maxsteps):
            if t >= tf:
                break
            self.statistics["nsteps"] += 1

            t_np1, p_np1, v_np1, a_np1 = self.NewmarkStep(t, p_n, v_n, a_n, h)

            y_np1 = np.hstack((p_np1, v_np1))
            yres.append(y_np1)
            tres.append(t_np1)

            t, p_n, v_n, a_n = t_np1, p_np1, v_np1, a_np1
            h = min(self.h, np.abs(tf - t))
        else:
            raise Explicit_ODE_Exception(f'Final time not reached within maximum number of steps at t={t}')

        return ID_PY_OK, tres, yres
    
    def NewmarkStep(self, t_n: float, p_n: np.ndarray, v_n: np.ndarray, a_n: np.ndarray, h:float)\
                    -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:

        f = self.problem.f
        ga = self.gamma
        be = self.beta
        t_np1 = t_n + h
        
        if self.damping is False and be == 0:
            self.statistics["nfcns"] += 1
            p_np1 = p_n + h * v_n + 0.5 * h ** 2 * a_n
            a_np1 = f(t_np1, p_np1, v_n)
            v_np1 = v_n + h * ((1 - ga) * a_n + ga * a_np1)
            return t_np1, p_np1, v_np1, a_np1
        else:
            # Set up equations for v_np1 and a_np1 in terms of p_np1
            eq_a = lambda p_np1: (1 - 1 / (2 * be)) * a_n - (1 / (be * h)) * v_n  + (1 / (be * h ** 2)) * (p_np1 - p_n)
            eq_v = lambda p_np1: v_n + h * ((1 - ga) * a_n + ga * eq_a(p_np1))

            # Now set up the equation that can be solved for p_np1 and then solve it
            p_guess = p_n + h * v_n + 0.5 * h ** 2 * a_n
            p_np1, infodict, ier, _  = fsolve(lambda x: eq_a(x) - f(t_np1, x, eq_v(x)), p_guess, full_output=True)
            self.statistics["nfcns"] += infodict["nfev"]
            
            if ier != 1:
                raise Explicit_ODE_Exception("Corrector could not converge within the set iterations")

            # Now compute a_np1 and v_np1 explicitly
            a_np1 = eq_a(p_np1)
            v_np1 = eq_v(p_np1)

            return t_np1, p_np1, v_np1, a_np1
    
    def print_statistics(self, verbose=NORMAL): # type: ignore
        self.log_message('\nFinal Run Statistics             : {name}'.format(name=self.problem.name),        verbose)
        self.log_message(' Step-length                     : {stepsize} '.format(stepsize=self.options["h"]), verbose)
        self.log_message(' Number of Steps                 : '+str(self.statistics["nsteps"]),          verbose)               
        self.log_message(' Number of Function Evaluations  : '+str(self.statistics["nfcns"]),           verbose)

        self.log_message('\nSolver options:',                                  verbose)
        self.log_message(' Solver            : Newmark',                        verbose)
        self.log_message(f' Damping           : {self.damping}',                 verbose)
        self.log_message(f' beta:             : {self.beta}',                    verbose)
        self.log_message(f' gamma:            : {self.gamma}\n' ,                  verbose)

class HHTalpha(Second_Order):
    def __init__(self, problem):
        super().__init__(problem)

        self.options["alpha"] = 0
        self.options["beta"] = 0.25
        self.options["gamma"] = 0.5

    # Setting up the method parameters
    @property
    def alpha(self) -> float:
        return self.options["alpha"]
    @alpha.setter
    def alpha(self, alpha: float):
            if alpha < -1/3 or alpha > 0:
                 raise ValueError("\"alpha\" must be in the range [-1/3, 0]")
            self.options["alpha"] = alpha
            self.options["beta"] = ((1 - alpha) / 2) ** 2
            self.options["gamma"] = 1 / 2 - alpha

    @property
    def beta(self) -> float:
        return self.options["beta"]
    @beta.setter
    def beta(self, beta: float):
        warn("\"beta\" should not be set manually. It is fully determined by \"alpha\"")
    
    @property
    def gamma(self) -> float:
        return self.options["gamma"]
    @gamma.setter
    def gamma(self, gamma: float):
        warn("\"gamma\" should not be set manually. It is fully determined by \"alpha\"")

    
    def integrate(self, t: float, y: np.ndarray, tf: float, opts) -> tuple[ID_PY_OK, list[float], list[np.ndarray]]:
        """
        integrates (t,y) values until t > tf
        """
        h = self.options["h"]
        h = min(h, abs(tf - t))
        y = np.atleast_1d(y)
        f = self.problem.f

        # Separating components
        p_n = y[:self.problem.ndofs]
        v_n = y[self.problem.ndofs:]

        # Computing initial a
        a_n = f(t, p_n, v_n)
        self.statistics["nfcns"] += 1

        #Lists for storing the result
        tres = []
        yres = []

        for i in range(self.maxsteps):
            if t >= tf:
                break
            self.statistics["nsteps"] += 1

            t_np1, p_np1, v_np1, a_np1 = self.HHTalphaStep(t, p_n, v_n, a_n, h)

            y_np1 = np.hstack((p_np1, v_np1))
            yres.append(y_np1)
            tres.append(t_np1)

            t, p_n, v_n, a_n = t_np1, p_np1, v_np1, a_np1
            h = min(self.h, np.abs(tf - t))
        else:
            raise Explicit_ODE_Exception(f'Final time not reached within maximum number of steps at t={t}')

        return ID_PY_OK, tres, yres
    
    def HHTalphaStep(self, t_n: float, p_n: np.ndarray, v_n: np.ndarray, a_n: np.ndarray, h:float)\
                    -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:

        f = self.problem.f 
        t_np1 = t_n + h
        
        al = self.alpha
        be = self.beta
        ga = self.gamma

        # Set up equations for v_np1 and a_np1 in terms of p_np1
        eq_a = lambda p_np1: (1 - 1 / (2 * be)) * a_n - (1 / (be * h)) * v_n  + (1 / (be * h ** 2)) * (p_np1 - p_n)
        eq_v = lambda p_np1: v_n + h * ((1 - ga) * a_n + ga * eq_a(p_np1))

        # Now set up the equation that can be solved for p_np1 and then solve it
        p_guess = p_n + h * v_n + 0.5 * h ** 2 * a_n
        p_np1, infodict, ier, _  = fsolve(lambda x: eq_a(x) - (1 + al) * f(t_np1, x, eq_v(x)) + al * f(t_n, p_n, v_n), p_guess, full_output=True)
        self.statistics["nfcns"] += infodict["nfev"]
        
        if ier != 1:
            raise Explicit_ODE_Exception(f"Corrector could not converge within the set iterations at t={t_n}")

        # Now compute a_np1 and v_np1 explicitly
        a_np1 = eq_a(p_np1)
        v_np1 = eq_v(p_np1)

        return t_np1, p_np1, v_np1, a_np1
    
    def print_statistics(self, verbose=NORMAL): # type: ignore
        self.log_message('\nFinal Run Statistics             : {name}'.format(name=self.problem.name),        verbose)
        self.log_message(' Step-length                     : {stepsize} '.format(stepsize=self.options["h"]), verbose)
        self.log_message(' Number of Steps                 : '+str(self.statistics["nsteps"]),          verbose)               
        self.log_message(' Number of Function Evaluations  : '+str(self.statistics["nfcns"]),           verbose)

        self.log_message('\nSolver options:',                                  verbose)
        self.log_message(' Solver            : HHT-alpha',                        verbose)
        self.log_message(f' Damping           : {self.damping}',                 verbose)
        self.log_message(f' aplha             : {self.alpha}',                    verbose)


class Linear_ODE_2nd_Order:
    """
    A class capable of expressing second order linear ODEs of the form
    M*y'' + C*y' + K*y  = f(t)

    All matrices have to be of shape (n,n) and f has to of compatible shape.
    """
    def __init__(self,
                 ndofs: int,
                 M: np.ndarray | csr_array,
                 K: np.ndarray | csr_array,
                 force: Callable[[float], np.ndarray],
                 C: np.ndarray | csr_array | None = None):
        
        if len(M.shape) != 2 or M.shape[0] != ndofs or M.shape[1] != ndofs:                         # type: ignore
            raise ValueError(f"M is not of the correct shape, it is {M.shape}, but should be ({ndofs, ndofs})!")
        if len(K.shape) != 2 or K.shape[0] != ndofs or K.shape[1] != ndofs:                         # type: ignore
            raise ValueError(f"K is not of the correct shape, it is {K.shape}, but should be ({ndofs, ndofs})!")
        if C is not None and (len(C.shape) != 2 or C.shape[0] != ndofs or C.shape[1] != ndofs):     # type: ignore
            raise ValueError(f"C is not of the correct shape, it is {C.shape}, but should be ({ndofs, ndofs})!")

        self.ndofs = ndofs
        self.M = M
        self.K = K
        self.force = force
        self.damping = C is not None
        if C is None:
            self.C = 0 * eye_array(self.ndofs)
        else:
            self.C = C
        

    def rhs(self, t, y):
        yp = y[self.ndofs:]
        b = -self.K @ y[:self.ndofs] + self.force(t)
        if self.C is not None:
            b -= self.C @ y[self.ndofs:]
        ypp = ssl.spsolve(self.M, b)

        return np.hstack((yp, ypp)) # type: ignore

class Explicit_Problem_2nd_lin(ap.Explicit_Problem): # type: ignore
    """
    A class capable of expressing second order IVPs of the form
    M*y'' + C*y' + K*y  = f(t)   for t in [t0, tf]
    y(t0)   = y0
    y'(t0)  = yp0
    """
    def __init__(self,
                 linODE: Linear_ODE_2nd_Order,
                 u0: np.ndarray,
                 up0: np.ndarray,
                 t0: float = 0.,
                 name: str = "Explicit Second Order Problem"):
        
        # ODE info
        self.ndofs = linODE.ndofs
        self.Mass_mat = linODE.M
        self.Stiff_mat = linODE.K
        self.Damp_mat = linODE.C
        self.force = linODE.force
        self.damping = linODE.damping
        
        self.u0 = np.atleast_1d(u0)
        self.up0 = np.atleast_1d(up0)
        self.t0 = t0
        self.name = name

        # the "y" notation is kept for traditional solvers
        self.y0 = np.hstack((self.u0, self.up0))

        self.rhs = linODE.rhs

class Second_Order_lin(Explicit_ODE):
    
    def __init__(self, problem: Explicit_Problem_2nd_lin):
        super().__init__(problem)
        
        # Solver options
        self.options["h"] = 0.01
        self.options["damping"] = problem.damping
        self.options["maxsteps"] = int(50000)
        
        # Statistics
        self.statistics["nsteps"] = 0
        self.statistics["nfcns"] = 0
    
    @property
    def h(self) -> float:
        return self.options["h"]
    @h.setter
    def h(self, h: float):
            self.options["h"] = float(h)
    
    @property
    def maxsteps(self) -> int:
        return self.options["maxsteps"]
    @maxsteps.setter
    def maxsteps(self, maxsteps: int):
            self.options["maxsteps"] = maxsteps

    @property
    def damping(self) -> bool:
        return self.options["damping"]
    @damping.setter
    def damping(self, damping: bool):
        warn("\"beta\" should not be set manually. It is fully determined by \"alpha\"")

class Newmark_lin(Second_Order_lin):
    def __init__(self, problem):
        super().__init__(problem)

        self.options["beta"] = 0.25
        self.options["gamma"] = 0.5

    # Setting up the method parameters
    @property
    def beta(self) -> float:
        return self.options["beta"]
    @beta.setter
    def beta(self, beta: float):
        if beta < 0 or beta > 1/2:
            raise ValueError("\"beta\" must be in the range [0, 1/2]")
        if self.problem.damping is True and beta == 0:
            raise ValueError("Witht a dmapening term present \"beta\" cannot be zero, must be in the range (0, 1/2]")
        self.options["beta"] = beta

    @property
    def gamma(self) -> float:
        return self.options["gamma"]
    @gamma.setter
    def gamma(self, gamma: float):
            if gamma < 0 or gamma > 1.:
                 raise ValueError("\"gamma\" must be in the range [0, 1]")
            self.options["gamma"] = gamma
    
    def integrate(self, t: float, y: np.ndarray, tf: float, opts) -> tuple[ID_PY_OK, list[float], list[np.ndarray]]:
        """
        integrates (t,y) values until t > tf
        """
        h = self.options["h"]
        h = min(h, abs(tf - t))
        y = np.atleast_1d(y)

        # Separating components
        p_n = y[:self.problem.ndofs]
        v_n = y[self.problem.ndofs:]

        # Computing initial a
        M = self.problem.Mass_mat
        K = self.problem.Stiff_mat
        F = self.problem.force
        f_n = F(t)
        if self.damping is False and self.beta == 0:
            a_n = ssl.spsolve(M, f_n - K @ p_n)
        else:
            C = self.problem.Damp_mat
            a_n = ssl.spsolve(M, F(t) - C @ v_n -  K @ p_n)
        a_n = np.array(a_n)
        self.statistics["nfcns"] += 1

        #Lists for storing the result
        tres = []
        yres = []

        for i in range(self.maxsteps):
            if t >= tf:
                break
            self.statistics["nsteps"] += 1

            t_np1, p_np1, v_np1, a_np1, f_np1 = self.NewmarkStep(t, p_n, v_n, a_n, h, f_n)

            y_np1 = np.hstack((p_np1, v_np1))
            yres.append(y_np1)
            tres.append(t_np1)

            t, p_n, v_n, a_n, f_n = t_np1, p_np1, v_np1, a_np1, f_np1
            h = min(self.h, np.abs(tf - t))
        else:
            raise Explicit_ODE_Exception('Final time not reached within maximum number of steps')

        return ID_PY_OK, tres, yres
    
    def NewmarkStep(self, t_n: float, p_n: np.ndarray, v_n: np.ndarray, a_n: np.ndarray, h:float, f_n: np.ndarray)\
                    -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        M = self.problem.Mass_mat
        K = self.problem.Stiff_mat
        F = self.problem.force
        ga = self.gamma
        t_np1 = t_n + h
        f_np1 = F(t_np1)
        
        if self.damping is False and self.beta == 0:
            self.statistics["nfcns"] += 1
            p_np1 = p_n + h * v_n + 0.5 * h ** 2 * a_n
            a_np1 = np.array(ssl.spsolve(M, f_n - K @ p_n))
            v_np1 = v_n + h * ((1 - ga) * a_n + ga * a_np1)
            return t_np1, p_np1, v_np1, a_np1, f_np1
        else:
            C = self.problem.Damp_mat
            be = self.beta

            # Set up the linear equation to be solved for p_np1
            A = (1 / (be * h ** 2)) * M + (ga / (be * h)) * C + K
            b = f_n\
                + M @ ((1 / (be * h ** 2)) * p_n + (1 / (be * h)) * v_n + (1 / (2 * be) - 1) * a_n)\
                + C @ ((ga / (be * h)) * p_n - (1 - ga / be) * v_n - (1 - ga / (2 * be)) * h * a_n)
            
            # Set up equations for a_np1 and v_np1 in terms of the previously calculated terms
            eq_a = lambda p_np1: (1 - 1 / (2 * be)) * a_n - (1 / (be * h)) * v_n  + (1 / (be * h ** 2)) * (p_np1 - p_n)
            eq_v = lambda a_np1: v_n + h * ((1 - ga) * a_n + ga * a_np1)
            
            # Solve the system
            self.statistics["nfcns"] += 1
            p_np1 = np.array(ssl.spsolve(A, b))
            a_np1 = eq_a(p_np1)
            v_np1 = eq_v(a_np1)

            return t_np1, p_np1, v_np1, a_np1, f_np1
    
    def print_statistics(self, verbose=NORMAL): # type: ignore
        self.log_message('\nFinal Run Statistics             : {name}'.format(name=self.problem.name),        verbose)
        self.log_message(' Step-length                     : {stepsize} '.format(stepsize=self.options["h"]), verbose)
        self.log_message(' Number of Steps                 : '+str(self.statistics["nsteps"]),          verbose)               
        self.log_message(' Number of Function Evaluations  : '+str(self.statistics["nfcns"]),           verbose)

        self.log_message('\nSolver options:',                                  verbose)
        self.log_message(' Solver            : Newmark',                        verbose)
        self.log_message(f' Damping           : {self.damping}',                 verbose)
        self.log_message(f' beta:             : {self.beta}',                    verbose)
        self.log_message(f' gamma:            : {self.gamma}\n' ,                  verbose)

class HHTalpha_lin(Second_Order_lin):
    def __init__(self, problem):
        super().__init__(problem)

        self.options["alpha"] = 0
        self.options["beta"] = 0.25
        self.options["gamma"] = 0.5

    # Setting up the method parameters
    @property
    def alpha(self) -> float:
        return self.options["alpha"]
    @alpha.setter
    def alpha(self, alpha: float):
            if alpha < -1/3 or alpha > 0:
                 raise ValueError("\"alpha\" must be in the range [-1/3, 0]")
            self.options["alpha"] = alpha
            self.options["beta"] = ((1 - alpha) / 2) ** 2
            self.options["gamma"] = 1 / 2 - alpha

    @property
    def beta(self) -> float:
        return self.options["beta"]
    @beta.setter
    def beta(self, beta: float):
        warn("\"beta\" should not be set manually. It is fully determined by \"alpha\"")
    
    @property
    def gamma(self) -> float:
        return self.options["gamma"]
    @gamma.setter
    def gamma(self, gamma: float):
        warn("\"gamma\" should not be set manually. It is fully determined by \"alpha\"")
    
    def integrate(self, t: float, y: np.ndarray, tf: float, opts) -> tuple[ID_PY_OK, list[float], list[np.ndarray]]:
        """
        integrates (t,y) values until t > tf
        """
        h = self.options["h"]
        h = min(h, abs(tf - t))
        y = np.atleast_1d(y)

        # Separating components
        p_n = y[:self.problem.ndofs]
        v_n = y[self.problem.ndofs:]

        # Computing initial a
        M = self.problem.Mass_mat
        K = self.problem.Stiff_mat
        F = self.problem.force
        f_n = F(t)
        if self.damping is False and self.beta == 0:
            a_n = ssl.spsolve(M, f_n - K @ p_n)
        else:
            C = self.problem.Damp_mat
            a_n = ssl.spsolve(M, F(t) - C @ v_n -  K @ p_n)
        a_n = np.array(a_n)
        self.statistics["nfcns"] += 1

        #Lists for storing the result
        tres = []
        yres = []

        for i in range(self.maxsteps):
            if t >= tf:
                break
            self.statistics["nsteps"] += 1

            t_np1, p_np1, v_np1, a_np1, f_np1 = self.HHTalphaStep(t, p_n, v_n, a_n, h, f_n)

            y_np1 = np.hstack((p_np1, v_np1))
            yres.append(y_np1)
            tres.append(t_np1)

            t, p_n, v_n, a_n, f_n = t_np1, p_np1, v_np1, a_np1, f_np1
            h = min(self.h, np.abs(tf - t))
        else:
            raise Explicit_ODE_Exception('Final time not reached within maximum number of steps')

        return ID_PY_OK, tres, yres
    
    def HHTalphaStep(self, t_n: float, p_n: np.ndarray, v_n: np.ndarray, a_n: np.ndarray, h:float, f_n: np.ndarray)\
                    -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        M = self.problem.Mass_mat
        K = self.problem.Stiff_mat
        F = self.problem.force
        C = self.problem.Damp_mat
        al = self.alpha
        be = self.beta
        ga = self.gamma
        t_np1 = t_n + h
        f_np1 = F(t_np1)

        # Set up the linear equation to be solved for p_np1
        A = (1 / (be * h ** 2)) * M + (ga / (be * h)) * C + (1 + al) * K
        b = f_n\
            + M @ ((1 / (be * h ** 2)) * p_n + (1 / (be * h)) * v_n + (1 / (2 * be) - 1) * a_n)\
            + C @ ((ga / (be * h)) * p_n - (1 - ga / be) * v_n - (1 - ga / (2 * be)) * h * a_n)\
            + al * K @ p_n
        
        # Set up equations for a_np1 and v_np1 in terms of the previously calculated terms
        eq_a = lambda p_np1: (1 - 1 / (2 * be)) * a_n - (1 / (be * h)) * v_n  + (1 / (be * h ** 2)) * (p_np1 - p_n)
        eq_v = lambda a_np1: v_n + h * ((1 - ga) * a_n + ga * a_np1)
        
        # Solve the system
        self.statistics["nfcns"] += 1
        p_np1 = np.array(ssl.spsolve(A, b))
        a_np1 = eq_a(p_np1)
        v_np1 = eq_v(a_np1)

        return t_np1, p_np1, v_np1, a_np1, f_np1
    
    def print_statistics(self, verbose=NORMAL): # type: ignore
        self.log_message('\nFinal Run Statistics             : {name}'.format(name=self.problem.name),        verbose)
        self.log_message(' Step-length                     : {stepsize} '.format(stepsize=self.options["h"]), verbose)
        self.log_message(' Number of Steps                 : '+str(self.statistics["nsteps"]),          verbose)               
        self.log_message(' Number of Function Evaluations  : '+str(self.statistics["nfcns"]),           verbose)

        self.log_message('\nSolver options:',                                  verbose)
        self.log_message(' Solver            : HHT-alpha',                        verbose)
        self.log_message(f' Damping           : {self.damping}',                 verbose)
        self.log_message(f' alpha:            : {self.alpha}',                    verbose)