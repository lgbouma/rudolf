import numpy as np
from astropy import units as u, constants as c

# M = 50 * u.Msun
# sigma_v = 0.5*u.km/u.s
# r = 10 * u.pc

M = 2*80 * u.Msun
sigma_v = 0.5*u.km/u.s
r = 7 * u.pc

T = 0.5*M*sigma_v**2
U = - c.G * M**2 / r

print(f"{T.cgs:.2e}")
print(f"{U.cgs:.2e}")
print(f"{T.cgs/U.cgs:.2f}")
