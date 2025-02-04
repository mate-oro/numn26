import numpy as np
import matplotlib.pyplot as plt
from assimulo.solvers import CVode # type: ignore
from assimulo.problem import Explicit_Problem # type: ignore
from BDF34_Assimulo import BDF_34

def runtask1():
    
    # Define problem parameters
    k = 1.e+3
    def lamb(y: np.ndarray) -> float:
        return k * ((np.linalg.norm(y) - 1) / np.linalg.norm(y)).astype(float)

    # Define the rhs
    def f(t,y):
        yd_0 = y[2]
        yd_1 = y[3]
        yd_2 = -y[0] * lamb(y[:2])
        yd_3 = -y[1] * lamb(y[:2]) - 1
        
        return np.array([yd_0, yd_1, yd_2, yd_3])
    
    # Define an Assimulo problem
    y0 = np.array([1.1, 0, 0, 0])
    pend_mod = Explicit_Problem(f, y0, name = "Elastic Pendulum")

    # Define an explicit solver
    pend_sim = BDF_34(pend_mod) #Create a CVode solver
    pend_sim.h = 1.e-2
    pend_sim.ord = 4

    # Simulate
    t, y = pend_sim.simulate(5)

    # Plot the components
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    axs[0].plot(t, y)
    axs[0].set_title("all y components")
    axs[0].set_xlabel("Time (t)")
    axs[0].set_ylabel("y values")

    # Plot the movement
    axs[1].plot(y[:, 0], y[:, 1])
    axs[1].set_title("space movement")
    axs[1].set_xlabel("y1")
    axs[1].set_ylabel("y2")
    
    plt.title("Elastic Penduum")
    plt.tight_layout()

    plt.figure()
    plt.semilogy(t[2:], pend_sim.errorest)
    plt.title("Error Estimate")
    plt.show()

if __name__ == "__main__":
    runtask1()