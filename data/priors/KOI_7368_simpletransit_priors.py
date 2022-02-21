import numpy as np

priordict = {
'period': ('Normal', 6.842939, 0.01), # Q1-Q17 DR25 table
't0': ('Normal', 2454970.06-2454833, 0.02), # Q1-Q17 DR25 table
'log_ror': ('Uniform', np.log(1e-2), np.log(1), np.log(0.0433)),
'b': ('ImpactParameter', 0.5),
'u_star': ('QuadLimbDark',),
'r_star': ('Normal', 0.876, 0.035), # Cluster isochrone (MIST+PARSEC)
'logg_star': ('Normal', 4.499, 0.03), # Cluster isochrone (MIST+PARSEC)
'log_jitter': ('Normal', '\\log\\langle \\sigma_f \\rangle', 2.0),
'kepler_mean': ('Normal', 0, 0.1),
}
