import numpy as np

# ported from plot_hr.py with the interpolation done
paramd = {
"Mstar_mist": [0.74038145, 0.76002731, 0.77927976],
"Teff_mist":  [4176.12554997, 4282.60043351, 4400.1054509],
"logg_mist":  [4.52962853, 4.52292936, 4.51753755],
"Rstar_mist": [0.77436785, 0.79064902, 0.80558568],
"Rho_mist":   [2.24783736, 2.16785028, 2.1014036],
"Mstar_parsec": [0.737, 0.75],
"Teff_parsec":  [4295.36426765, 4454.5106336],
"logg_parsec":  [4.558, 4.574],
"Rstar_parsec": [0.74776917, 0.74056715],
"Rho_parsec":   [2.48494114, 2.60327003],
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
