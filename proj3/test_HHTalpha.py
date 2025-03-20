import numpy as np
from secondOrderODE import Explicit_Problem_2nd_nonlin as EP2, HHTalpha
from assimulo.solvers import CVode # type: ignore
import matplotlib.pyplot as plt

# Define problem parameters
k = 1.e+3
def lamb(y: np.ndarray) -> float:
    return k * ((np.linalg.norm(y) - 1) / np.linalg.norm(y)).astype(float)

# Define the rhs
def f(t, u, up):
    upp_0 = -u[0] * lamb(u)
    upp_1 = -u[1] * lamb(u) - 1
    
    return np.array([upp_0, upp_1])

# Initial condition
y0 = np.array([1.1, 0])
yp0 = np.array([0, 0])

# Define the problem
pend_mod = EP2(2, f, y0, yp0, damping = False, t0 = 0, name = "Elastic Pendulum")

# Define an solver
pend_sim = HHTalpha(pend_mod)
pend_sim.alpha = -0.2
pend_sim.h = 1e-2

# Simulate
t, y = pend_sim.simulate(5)

# Plot the components
fig, axs = plt.subplots(2, 1, figsize=(10, 10))
axs[0].grid()
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
plt.show()