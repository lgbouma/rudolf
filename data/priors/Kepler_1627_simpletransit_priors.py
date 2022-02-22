import numpy as np

priordict = {
'period': ('Normal', 7.20280608, 0.01), # Holczer+16
't0': ('Normal', 120.790531, 0.02), # Holczer+16, 2454953.790531 - 2454833 for Kepler timestamp
'log_ror': ('Uniform', np.log(5e-3), np.log(1), np.log(0.039)),
'b': ('ImpactParameter', 0.5),
'u_star': ('QuadLimbDark',),
'r_star': ('Normal', 0.881, 0.018), # Cluster isochrone (MIST+PARSEC)
'logg_star': ('Normal', 4.53, 0.05), # Cluster isochrone (MIST+PARSEC)
'log_jitter': ('Normal', '\\log\\langle \\sigma_f \\rangle', 2.0),
'kepler_mean': ('Normal', 1, 0.1),
}
