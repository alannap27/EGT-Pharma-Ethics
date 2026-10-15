import lyapynov
import numpy as np

from model import *

x0 = np.array([0.5,0.5,0.5])
t0 = 0
dt = 0.001
timeEnd = 10
continuous_system = lyapynov.ContinuousDS(x0, t0, fun, jac, dt)
#lyap_exponents = lyapynov.mLCE(continuous_system, 10000, 10000, False)
lyap_exponents = lyapynov.LCE(continuous_system, 3, 0, int(timeEnd/dt), False)
print(lyap_exponents)
