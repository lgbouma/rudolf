import numpy as np

# ported from plot_hr.py with the interpolation done
paramd = {
"Mstar_mist": [0.8616312,  0.87855967, 0.89627916],
"Teff_mist":  [5005.29750439,  5136.09301364,  5263.80006698],
"logg_mist":  [4.50170668,  4.49823669,  4.49552493,  4.49560307,  4.50294799],
"Rstar_mist": [0.86611623,  0.87731788,  0.88604123],
"Rho_mist":   [1.86958055,  1.83422079,  1.81648911],
"Mstar_parsec": [0.85,   0.879,  0.898],
"Teff_parsec":  [5101.52477386,  5257.75082257,  5335.80561846],
"logg_parsec":  [4.529,  4.527,  4.531],
"Rstar_parsec": [0.83031575,  0.84630762,  0.85147516],
"Rho_parsec":   [2.09334296,  2.04435076,  2.0507451],
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
