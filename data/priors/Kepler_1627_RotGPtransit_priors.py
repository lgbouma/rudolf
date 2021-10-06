import numpy as np

#
# Note: the names of these parameters do matter for posterior_table creation.
# So does the extra comma in singletons.
#

# normal: mu, sd
# uniform: lower, upper, testval
# ImpactParameter: testval

priordict = {
'period': ('Normal', 7.20280608, 0.01), # Holczer+16
't0': ('Normal', 120.790531, 0.02), # Holczer+16, 2454953.790531 - 2454833 for Kepler timestamp
#'log_r': ('Uniform', np.log(1e-2), np.log(1), np.log(0.0433)),
#'b': ('ImpactParameter', 0.5),
'log_depth': ('Normal', np.log(1.8e-3), 2),
'b': ('Uniform', 0, 1),
'u_star': ('QuadLimbDark',),
#'u[0]': ('Uniform', 0.51-0.2, 0.51+0.2, 0.51), # Claret+2011, Teff 5500K, logg 4.5, solar metallicity, V-band
#'u[1]': ('Uniform', 0.24-0.2, 0.24+0.2, 0.240),
'r_star': ('TruncatedNormal', 0.881, 0.018), # Cluster isochrone (MIST+PARSEC)
'logg_star': ('Normal', 4.53, 0.05), # Cluster isochrone (MIST+PARSEC)
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
'log_prot': ('Normal', np.log(2.606418), 0.02), # LGB LombScargle, +/-2% sys (on the log) assumed.
'log_Q0': ('Normal', 0, 2),
'log_dQ': ('Normal', 0, 2),
'f': ('Uniform', 0.01, 1),
}
