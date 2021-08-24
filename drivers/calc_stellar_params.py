import numpy as np

# ported from plot_hr.py with the interpolation done
paramd = {
"Mstar_mist": [0.93608091, 0.95293971, 0.96948655],
"Teff_mist": [5461.14115145, 5497.82250237, 5526.99810925],
"logg_mist": [4.51654624, 4.52809482, 4.53703309],
"Rstar_mist": [0.88392901, 0.88007385, 0.87859386],
"Rho_mist": [1.91078807, 1.97087632, 2.01524843],
"Mstar_parsec": [0.939, 0.947],
"Teff_parsec": [5501.73920511, 5524.58932613],
"logg_parsec": [4.54, 4.54],
"Rstar_parsec": [0.86172087, 0.86538388],
"Rho_parsec": [2.06879308, 2.06003624],
}

params = 'Mstar,Teff,logg,Rstar,Rho'.split(',')

for p in params:
    k_mist = p+'_mist'
    k_parsec = p+'_parsec'

    print(10*'-')
    val = np.mean(paramd[k_mist])
    upper = np.max(paramd[k_mist])
    lower = np.min(paramd[k_mist])
    s = f'{k_mist}: {val:.3f} +{upper-val:.3f} -{val-lower:.3f}'
    print(s)

    val_p = np.mean(paramd[k_parsec])
    upper_p = np.max(paramd[k_parsec])
    lower_p = np.min(paramd[k_parsec])
    s = f'{k_parsec}: {val_p:.3f} +{upper_p-val_p:.3f} -{val_p-lower_p:.3f}'
    print(s)

    s = f'{p}: MIST-PARSEC/MIST: {100*(val-val_p)/val:.1f}%'
    print(s)

    # quadrature sum of statistical and systematic
    rel_unc = np.sqrt(
        (np.max(np.abs([upper-val,val-lower]))/val)**2
        +
        ((val-val_p)/val)**2
    )

    s = f'{p} uncertainty (stat+sys): +/-{val*rel_unc:.3f}'
    print(s)
