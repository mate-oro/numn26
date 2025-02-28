from squeezer import Seven_bar_mechanism_3 as SBM3
from squeezer2 import Seven_bar_mechanism_2 as SBM2
from squeezer2 import Seven_bar_mechanism_1 as SBM1
from assimulo.solvers import IDA # type: ignore
import numpy as np
import matplotlib.pyplot as plt


algvar = np.zeros(20)
algvar[:14] = 1
atol = 1.e-6 * np.ones(20)
atol[7:] = 1e-5

# Index 1 formulation
model1 = SBM1()
model1.algvar = algvar
sim1 = IDA(model1)
sim1.suppress_alg = True
sim1.atol = atol
t1, y1, yd1 = sim1.simulate(0.03)


plt.plot(t1, y1[:, -6:])
plt.title("Lagrange multipliers")
plt.show()