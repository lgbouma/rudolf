"""
Functions for using the VESPA validation code.

Contents:
    _write_vespa
"""
###########
# imports #
###########
import os
import numpy as np, pandas as pd, matplotlib.pyplot as plt

from rudolf.paths import DRIVERDIR
from astrobase.lcmath import phase_magseries_with_errs

from betty.plotting import doublemean, doublepctile

##########
# config #
##########

VESPADRIVERDIR = os.path.join(DRIVERDIR, 'vespa_drivers')
if not os.path.exists(VESPADRIVERDIR):
    os.mkdir(VESPADRIVERDIR)

def _write_vespa(data, soln, staridentifier,
                 N_hours_from_transit=4, make_plot=False):
    """
    Given the data and m.trace.posterior ("data" and "soln"),  get *detrended*
    (i.e., rotation signal removed) light curve and its ephemeris, and
    phase-fold it into the format that vespa needs.

    Args:

        data (OrderedDict): e.g., data['tess'] = (time, flux, flux_err, t_exp)

        soln (arviz.data.inference_data.InferenceData): posterior's trace
        (m.trace.posterior) from PyMC3.

        staridentifier: str used in the output csv file name.

        N_hours_from_transit: number of hours +/- the transit mid-point that
        are used.  Better to not use all available data because otherwise
        vespa's MCMC is slow.
    """

    # get time/flux/error/period/t0
    assert len(data.keys()) == 1
    name = list(data.keys())[0]
    x,y,yerr,texp = data[name]
    t0, period = np.nanmedian(soln["t0"]), np.nanmedian(soln["period"])

    assert len(soln["gp_pred"].shape) == 3, 'ncores X nchains X time'
    medfunc, pctfunc = doublemean, doublepctile
    gp_mod = medfunc(soln["gp_pred"]) + medfunc(soln["mean"])
    _yerr = np.sqrt(yerr**2 + np.exp(2*medfunc(soln["log_jitter"])))

    time = x
    flux = 1 + (y-gp_mod) - np.nanmedian(y-gp_mod)
    flux_err = _yerr

    # make relevant directories and paths
    outdir = os.path.join(VESPADRIVERDIR, f'{staridentifier}')
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    outpath = os.path.join(outdir, f'{staridentifier}_vespa_lc.csv')

    # phase-fold and take points +/- N hours from transit
    orb_d = phase_magseries_with_errs(
        time, flux, flux_err, period, t0, wrap=True, sort=True
    )

    t = orb_d['phase']*period*24
    f = orb_d['mags']
    e = orb_d['errs']

    s = (t > -N_hours_from_transit) & (t < N_hours_from_transit)

    t,f,e = t[s],f[s],e[s]

    outdf = pd.DataFrame({'t':t, 'f':f, 'e':e})

    outdf.to_csv(outpath, index=False, header=False)

    print(f'Wrote vespa lc to {outpath}')

    if make_plot:
        fig, ax = plt.subplots(figsize=(8,6))

        ax.errorbar(t, f, yerr=e,
                    color="darkgray", label="data", fmt='.', elinewidth=0.2,
                    capsize=0, markersize=1, rasterized=True, zorder=-1,
                    alpha=0.6)

        ax.set_xlabel('hours around mid-transit')
        ax.set_ylabel('relative flux')

        outpath = os.path.join(outdir, f'{staridentifier}_vespa_lc.png')

        fig.savefig(outpath, bbox_inches='tight', dpi=400)
        print(f'Wrote {outpath}')
