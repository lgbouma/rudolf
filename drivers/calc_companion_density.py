from astropy import units as u, constants as c
import numpy as np

M = 0.33*u.Msun
R = 0.465*u.Rsun
# from http://perso.ens-lyon.fr/isabelle.baraffe/BHAC15dir/BHAC15_tracks+structure
# using 0.300 Msun, logt 7.571495, teff 3411 and rounding up

V = (4/3)*np.pi*R**3

rho = M/V

print(rho.cgs)
