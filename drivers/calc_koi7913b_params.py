import numpy as np

# ported from plot_hr.py with the interpolation done
paramd = {
"Mstar_mist": [0.67537384, 0.69827908, 0.71963715],
"Teff_mist":  [3933.17105347, 4003.26878352, 4080.71818549],
"logg_mist":  [4.55914526, 4.54876685, 4.53813507],
"Rstar_mist": [0.71488026, 0.73563927, 0.75600221],
"Rho_mist":   [2.60612727, 2.47278077, 2.3479869],
"Mstar_parsec": [0.652, 0.659, 0.691],
"Teff_parsec":  [3619.09513884, 3644.18150789, 3773.98352537],
"logg_parsec":  [4.499, 4.498, 4.496],
"Rstar_parsec": [0.75276187, 0.75766377, 0.7776297],
"Rho_parsec":   [2.1548947,  2.13602899, 2.07162349],
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
