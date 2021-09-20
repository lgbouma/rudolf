Tdur_0 = 2.538
eTdur_0 = 0.030

Tdur_1 = 2.826
eTdur_1 = 0.057

e = (eTdur_1**2 + eTdur_0**2)**(1/2)

N_sigma = (Tdur_1 - Tdur_0)/e
print(f'Difference = {Tdur_1 - Tdur_0:.3f} +/- {e:.3f} hr ({N_sigma:.1f}σ)')
