import numpy as np

priordict = {
'period': ('Normal', 24.2783801, 0.01),  # Q1-Q17 DR25 table
't0': ('Normal', 2454987.513-2454833, 0.05),
'log_ror': ('Uniform', np.log(5e-3), np.log(1), np.log(0.023)),
'b': ('ImpactParameter', 0.5),
'u_star': ('QuadLimbDark',),
'r_star': ('Normal', 0.79, 0.049), # Cluster isochrone (MIST+PARSEC)
'logg_star': ('Normal', 4.523, 0.043), # Cluster isochrone (MIST+PARSEC)
'log_jitter': ('Normal', '\\log\\langle \\sigma_f \\rangle', 2.0),
'kepler_mean': ('Normal', 0, 0.1),
}
