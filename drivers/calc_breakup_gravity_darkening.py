import numpy as np
from astropy import units as u, constants as c

P_rot = 2.642*u.day

Rstar = 0.881*u.Rsun
Mstar = 0.953*u.Msun

P_break = (
    2*np.pi*Rstar**(3/2) / ((c.G*Mstar)**(1/2))
).to(u.day)

print(f'P_break: {P_break:.3f} days')

frac = ((P_break/P_rot)**2).cgs

print(f'frac transit change: {frac*100:.4f}%')

delta = 1800 # ppm
delta_GD = 1800*frac

print(f'delta_GD: {delta_GD:.2f} ppm')

delta_obs = 100 # 0.1 ppt = 0.1 e-3 = 100 e-6
print(f'delta_obs: {delta_obs:.2f} ppm')
