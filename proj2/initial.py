import numpy as np
from scipy.optimize import fsolve

# Geometry
xa,  ya = -.06934, -.00227
xb, yb = -0.03635, .03273
xc, yc = .014, .072
d, da, e, ea = 28.e-3, 115.e-4, 2.e-2, 1421.e-5
rr, ra = 7.e-3, 92.e-5
ss, sa, sb, sc, sd = 35.e-3, 1874.e-5, 1043.e-5, 18.e-3, 2.e-2
ta, tb = 2308.e-5, 916.e-5
u, ua, ub = 4.e-2, 1228.e-5, 449.e-5
zf, zt = 2.e-2, 4.e-2
fa = 1421.e-5

def gz(q: np.ndarray) -> np.ndarray:
    theta = 0
    beta, gamma, phi, delta, omega, epsilon = q[0:6]
    sibe, siga, siph, side, siom, siep = np.sin(q[0:6])
    cobe, coga, coph, code, coom, coep = np.cos(q[0:6])
    #sibeth = np.sin(beta+theta);    cobeth = np.cos(beta+theta)
    sibeth = sibe;    cobeth = cobe
    siphde = np.sin(phi+delta);     cophde = np.cos(phi+delta)
    siomep = np.sin(omega+epsilon); coomep = np.cos(omega+epsilon)
    rhs = np.array([xb, yb, xa, ya, xa, ya])

    lhs = np.zeros(6)
    lhs[0] = rr * cobe - d * cobeth - ss * siga
    lhs[1] = rr * sibe - d * sibeth + ss * coga
    lhs[2] = rr * cobe - d * cobeth - e * siphde - zt * code
    lhs[3] = rr * sibe - d * sibeth + e * cophde - zt * side
    lhs[4] = rr * cobe - d * cobeth - zf * coomep - u * siep
    lhs[5] = rr * sibe - d * sibeth - zf * siomep + u * coep
    return lhs - rhs

q0 = np.array([  
            -0.0617138900142764496358948458001, # beta
            0.,                                 # theta
            0.455279819163070380255912382449,   # gamma
            0.222668390165885884674473185609,   # phi
            0.487364979543842550225598953530,   # delta
            -0.222668390165885884674473185609,  # Omega
            1.23054744454982119249735015568])   # epsilon

x0 = np.zeros(6)
x, infodict, ier, _ = fsolve(gz, x0, full_output=True, xtol=1e-12)

q_est = np.zeros(7)
q_est[[0, 2, 3, 4, 5, 6]] = x

for v in q_est:
    print(v)

print(f"\n ||g(q_est) - rhs||: {np.linalg.norm(gz(x))}")
print("\n distance:", np.linalg.norm(q0 - q_est))