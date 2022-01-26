import numpy as np

# ported from plot_hr.py with the interpolation done
paramd = {
"Mstar_mist": [0.8295773, 0.84509594, 0.8616312],
"Teff_mist":  [4754.70855616, 4875.08224191, 5005.29750439],
"logg_mist":  [4.50513235, 4.50170668, 4.49823669],
"Rstar_mist": [0.84313297, 0.85434539, 0.86611623],
"Rho_mist":   [1.95128156, 1.91054322, 1.86958055],
"Mstar_parsec": [0.803, 0.83,  0.848],
"Teff_parsec":  [4769.79814119, 4953.36122316, 5087.44819645],
"logg_parsec":  [4.545, 4.535, 4.529],
"Rstar_parsec": [0.79230352, 0.81484098, 0.82933833],
"Rho_parsec":   [2.27610387, 2.16277233, 2.09581008],
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
