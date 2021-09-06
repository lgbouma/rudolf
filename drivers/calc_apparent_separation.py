import numpy as np
rho_arcsec = 0.16
e_rho_arcsec = 0.02
d_pc = 329.5
e_d_pc = 3.5

rho_AU = rho_arcsec*d_pc
upper_rho_AU = (rho_arcsec+e_rho_arcsec)*(d_pc+e_d_pc)
lower_rho_AU = (rho_arcsec-e_rho_arcsec)*(d_pc-e_d_pc)
e_rho_AU = np.max([upper_rho_AU-rho_AU, rho_AU-lower_rho_AU])

print(f'{rho_AU:.2f} +/- {e_rho_AU:.2f} AU')
