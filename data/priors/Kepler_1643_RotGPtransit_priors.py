import numpy as np

#
# Note: the names of these parameters do matter for posterior_table creation.
# So does the extra comma in singletons.
#

# normal: mu, sd
# uniform: lower, upper, testval
# ImpactParameter: testval

#FIXME FIXME ALL PRIORS!
priordict = {
'period': ('Normal', 5.34264143, 0.01), # Q1-Q17 DR25 table
't0': ('Normal', 2454967.381-2454833, 0.02), # Q1-Q17 DR25 table
# #Classical Parametrization: fit log(Rp/R*), b[0,1+Rp/R*]
'log_ror': ('Uniform', np.log(1e-3), np.log(1), np.log(0.0236)),
'b': ('ImpactParameter', 0.5),
# # Alternative paramterization: fit log(depth), b[0,1]. 
#'log_depth': ('Normal', np.log(1.7e-3), 2),
#'b': ('Uniform', 0, 1),
'u_star': ('QuadLimbDark',),
#'u[0]': ('Uniform', 0.51-0.2, 0.51+0.2, 0.51), # Claret+2011, Teff 5500K, logg 4.5, solar metallicity, V-band
#'u[1]': ('Uniform', 0.24-0.2, 0.24+0.2, 0.240),
'r_star': ('Normal', 0.855, 0.044), # Cluster isochrone (MIST+PARSEC)
'logg_star': ('Normal', 4.502, 0.035), # Cluster isochrone (MIST+PARSEC)
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
'log_prot': ('Normal', np.log(5.1064), 0.032), # 5.1064 +/- 0.0426 d. (2% unc on logProt assumed)
'log_Q0': ('Normal', 0, 2),
'log_dQ': ('Normal', 0, 2),
'f': ('Uniform', 0.01, 1),
}
