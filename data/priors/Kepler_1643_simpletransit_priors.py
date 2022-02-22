import numpy as np

priordict = {
'period': ('Normal', 5.34264143, 0.01), # Q1-Q17 DR25 table
't0': ('Normal', 2454967.381-2454833, 0.02), # Q1-Q17 DR25 table
'log_ror': ('Uniform', np.log(2e-3), np.log(1), np.log(0.0236)),
'b': ('ImpactParameter', 0.5),
'u_star': ('QuadLimbDark',),
'r_star': ('Normal', 0.855, 0.044), # Cluster isochrone (MIST+PARSEC)
'logg_star': ('Normal', 4.502, 0.035), # Cluster isochrone (MIST+PARSEC)
'log_jitter': ('Normal', '\\log\\langle \\sigma_f \\rangle', 2.0),
'kepler_mean': ('Normal', 1, 0.1),
}
