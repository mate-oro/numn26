from squeezer import Seven_bar_mechanism_3 as SBM3
from squeezer2 import Seven_bar_mechanism_2 as SBM2
from assimulo.solvers import IDA # type: ignore
import numpy as np
import matplotlib.pyplot as plt


algvar = np.zeros(20)
algvar[:14] = 1
atol = 1.e-6 * np.ones(20)
atol[7:] = 1e-0

# Index 3 formulation
model3 = SBM3()
model3.algvar = algvar
sim3 = IDA(model3)
sim3.suppress_alg = True
sim3.atol = atol
t3, y3, yd3 = sim3.simulate(0.03)

# Index 2 formulation
model2 = SBM2()
model2.algvar = algvar
sim2 = IDA(model2)
sim2.suppress_alg = True
sim2.atol = atol
t2, y2, yd2 = sim2.simulate(0.03)


plt.plot(t3, y3[:, -6:])
plt.plot(t2, y2[:, -6:], linestyle="--")
plt.title("Lagrange multipliers")
plt.show()