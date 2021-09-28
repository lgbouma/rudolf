"""
Compare fitted rp value with lit values
"""
import numpy as np

# Morton+16
Rp_0 = 3.69
eRp_0 = 0.45

# Me 3.340 +/ 0.179
Rp_1 = 3.340
eRp_1 = 0.179

e = (eRp_1**2 + eRp_0**2)**(1/2)

N_sigma = (Rp_1 - Rp_0)/e
print('Morton+16')
print(f'Difference = {Rp_1 - Rp_0:.3f} +/- {e:.3f} Re ({N_sigma:.1f}σ)')

##########################################

# Berger+18
Rp_0 = 3.760
eRp_0 = 0.303

# Me 3.340 +/ 0.179
Rp_1 = 3.340
eRp_1 = 0.179

e = (eRp_1**2 + eRp_0**2)**(1/2)

N_sigma = (Rp_1 - Rp_0)/e
print('Berger+18')
print(f'Difference = {Rp_1 - Rp_0:.3f} +/- {e:.3f} Re ({N_sigma:.1f}σ)')


