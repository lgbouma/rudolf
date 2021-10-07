"""
Compare fitted rp value with lit values
"""
import numpy as np

# Me from table 2
Rp_nocorr = 3.789
eRp_nocorr = 0.157

factor = 1.015**0.5

Rp_1 = factor * Rp_nocorr
eRp_1 = factor * eRp_nocorr
print('Bouma+21')
print(f'{Rp_1:.3f} +/- {eRp_1:.3f} Re')

# Morton+16
Rp_0 = 3.69
eRp_0 = 0.45

e = (eRp_1**2 + eRp_0**2)**(1/2)

N_sigma = (Rp_1 - Rp_0)/e
print('Morton+16')
print(f'Difference = {Rp_1 - Rp_0:.3f} +/- {e:.3f} Re ({N_sigma:.1f}σ)')

##########################################

# Berger+18
Rp_0 = 3.760
eRp_0 = 0.303

e = (eRp_1**2 + eRp_0**2)**(1/2)

N_sigma = (Rp_1 - Rp_0)/e
print('Berger+18')
print(f'Difference = {Rp_1 - Rp_0:.3f} +/- {e:.3f} Re ({N_sigma:.1f}σ)')


