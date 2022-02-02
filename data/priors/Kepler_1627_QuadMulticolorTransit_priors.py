import numpy as np

#
# Note: the names of these parameters do matter for posterior_table creation.
# So does the extra comma in singletons.
#

# normal: mu, sd
# uniform: lower, upper, testval
# ImpactParameter: testval

priordict = {
'period': ('Normal', 7.20280608, 7.4e-6), # Holczer+16
#'t0': ('Normal', 2454953.790531, 9.5e-4), # Holczer+16, 2454953.790531
#'period': ('Normal', 7.20280608, 7.4e-6), # Holczer+16
't0': ('Normal', 2459433.93591276, 30/(60*24)), # say 30 min t_tra uncert
'log_ror': ('Uniform', np.log(1e-2), np.log(1e-1), np.log(0.0433)),
'b': ('ImpactParameter', 0.5),
#'u[0]': ('Uniform', 0.51-0.2, 0.51+0.2, 0.51), # Claret+2011, Teff 5500K, logg 4.5, solar metallicity, V-band
#'u[1]': ('Uniform', 0.24-0.2, 0.24+0.2, 0.240),
'r_star': ('TruncatedNormal', 0.881, 0.018), # Cluster isochrone (MIST+PARSEC)
'logg_star': ('Normal', 4.53, 0.05), # Cluster isochrone (MIST+PARSEC)
#'ecc': ('EccentricityVanEylen19',),
#'omega': ('Uniform', 0, 2*np.pi),
#'mean': ('Normal', 0.5, 0.1),
#'log_jitter': ('Normal', r"\log\langle \sigma_f \rangle", 2.0),
# SHO term
#'rho': ("InverseGamma", 0.5, 2),
# 'rho': ("Uniform", 1, 10),
# 'sigma': ("InverseGamma", 1, 5),
# Rotation term
# 'sigma_rot': ("InverseGamma", 1, 5),
# 'log_prot': ('Normal', np.log(2.606418), 0.02), # LGB LombScargle, +/-2% sys (on the log) assumed.
# 'log_Q0': ('Normal', 0, 2),
# 'log_dQ': ('Normal', 0, 2),
# 'f': ('Uniform', 0.01, 1),
}

bandpasses = 'g,r,i,z'.split(',')
for bp in bandpasses:
    priordict[f'muscat3_{bp}_mean'] = ('Normal', 1.0, 0.1)
    priordict[f'muscat3_{bp}_a1'] = ('Uniform', -0.2, 0.2, 0)
    priordict[f'muscat3_{bp}_a2'] = ('Uniform', -0.2, 0.2, 0)
    priordict[f'muscat3_{bp}_log_jitter'] = ('Normal', r"\log\langle \sigma_f \rangle", 2.0)
    priordict[f'muscat3_{bp}_u_star'] = ('QuadLimbDark',)



