"""
Compare fitted value from MUSCAT routines
"""
import numpy as np

Tdur_0 = 2.538
eTdur_0 = 0.030

Tdur_1 = 2.826
eTdur_1 = 0.057

e = (eTdur_1**2 + eTdur_0**2)**(1/2)

N_sigma = (Tdur_1 - Tdur_0)/e
print(f'Difference = {Tdur_1 - Tdur_0:.3f} +/- {e:.3f} hr ({N_sigma:.1f}σ)')


colors = 'g,r,i,z'.split(',')
rps = [0.315, 0.291, 0.307, 0.270]
rp_uncs = [0.025, 0.021, 0.024, 0.020]

rp_0 = 0.298
erp_0 = 0.016

for _r, _er, c in zip(rps, rp_uncs, colors):

    e = (erp_0**2 + _er**2)**(1/2)

    N_sigma = np.abs((_r - rp_0)/e)
    print(f'{c}-band: Difference = {_r - rp_0:.3f} +/- {e:.3f} Rjup ({N_sigma:.1f}σ)')


