from squeezer import Seven_bar_mechanism_3 as SBM3
from squeezer2 import Seven_bar_mechanism_2 as SBM2
from assimulo.solvers import IDA # type: ignore
import numpy as np
import matplotlib.pyplot as plt


model = SBM3()

algvar = np.zeros(20)
algvar[:14] = 1
model.algvar = algvar

sim = IDA(model)
sim.suppress_alg = True

atol = 1.e-6 * np.ones(20)
atol[7:] = 1e-1
sim.atol = atol

t, y, yd = sim.simulate(0.03)

# Plotting
ymod = np.mod(y + np.pi/2, np.pi) - np.pi/2
jump0 = np.where(np.abs(np.diff(ymod[:, 0])) > np.pi/2)[0] + 1
jump1 = np.where(np.abs(np.diff(ymod[:, 1])) > np.pi/2)[0] + 1


# Insert NaN values after each jump so that the line is broken there.
y_nan = np.copy(ymod)

t0 = np.insert(t, jump0, np.nan)
t1 = np.insert(t, jump1, np.nan)
y0 = np.insert(y_nan[:, 0], jump0, np.nan)
y1 = np.insert(y_nan[:, 1], jump1, np.nan)

plt.plot(t0, y0)
plt.plot(t1, y1)
plt.plot(t, y_nan[:, 2:7])
plt.show()

plt.plot(t, y[:, -6:])
plt.title("Lagrange multipliers")
plt.show()