import numpy as np

#
# Note: the names of these parameters do matter for posterior_table creation.
# So does the extra comma in singletons.
#

# normal: mu, sd
# uniform: lower, upper, testval
# ImpactParameter: testval

priordict = {
'period': ('Normal', 24.2783801, 0.01), # Q1-Q17 DR25 table
't0': ('Normal', 2454987.513-2454833, 0.05), # Q1-Q17 DR25 table  (smaller unc maybe needed?)
# #Classical Parametrization: fit log(Rp/R*), b[0,1+Rp/R*]
'log_ror': ('Uniform', np.log(5e-3), np.log(1), np.log(0.023)),
'b': ('ImpactParameter', 0.5),
# # Alternative paramterization: fit log(depth), b[0,1]. 
#'log_depth': ('Normal', np.log(1.7e-3), 2),
#'b': ('Uniform', 0, 1),
'u_star': ('QuadLimbDark',),
#'u[0]': ('Uniform', 0.51-0.2, 0.51+0.2, 0.51), # Claret+2011, Teff 5500K, logg 4.5, solar metallicity, V-band
#'u[1]': ('Uniform', 0.24-0.2, 0.24+0.2, 0.240),
'r_star': ('Normal', 0.790, 0.049), # Cluster isochrone (MIST+PARSEC)
'logg_star': ('Normal', 4.523, 0.043), # Cluster isochrone (MIST+PARSEC)
'mean': ('Normal', 0, 0.1),
'ecc': ('EccentricityVanEylen19',),
'omega': ('Uniform', 0, 2*np.pi),
'log_jitter': ('Normal', r"\log\langle \sigma_f \rangle", 2.0),
# SHO term
#'rho': ("InverseGamma", 0.5, 2),
# 'rho': ("Uniform", 1, 10),
# 'sigma': ("InverseGamma", 1, 5),
# Rotation term
'sigma_rot': ("InverseGamma", 1, 5),
'log_prot': ('Normal', np.log(3.399078), 0.024), # Measured 3.399 +/- 0.042 from Kepler window slider.   (2% unc on logProt assumed)
'log_Q0': ('Normal', 0, 2),
'log_dQ': ('Normal', 0, 2),
#'f': ('Uniform', 0.01, 1),
'log_f': ('Uniform', -10, 0, -5),
}
