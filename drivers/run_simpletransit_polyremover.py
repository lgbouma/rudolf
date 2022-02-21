"""
Fit the Kepler-1627 / Kepler-1643 / KOI-7368 / KOI-7913 datasets, using a local
polynomial for the stellar variability and a transit for the planet.
"""

import numpy as np, matplotlib.pyplot as plt, pandas as pd, pymc3 as pm
import os
from os.path import join
from collections import OrderedDict
from importlib.machinery import SourceFileLoader
from copy import deepcopy

from astrobase.lcmath import find_lc_timegroups

import betty.plotting as bp
from betty.posterior_table import make_posterior_table
from betty.modelfitter import ModelFitter
from betty.paths import BETTYDIR
from betty.helpers import _subset_cut, _quicklcplot
from betty.io import given_priordict_make_priorfile

from rudolf.helpers import (
    get_manually_downloaded_kepler_lightcurve
)

from rudolf.paths import DATADIR, RESULTSDIR

EPHEMDICT = {
    'KOI_7368': {'t0': 2454970.06-2454833, 'per': 6.842939, 'tdur':2.8/24, 'n_tdurs':3.5},
    'KOI_7913': {'t0': 2454987.513-2454833, 'per': 24.2783801, 'tdur':4.564/24, 'n_tdurs':2.5},
}

# NOTE: change starid as desired based on the dataset to use.
#def run_RotGPtransit(starid='Kepler_1627_Q15slc', N_samples=2000):
#def run_RotGPtransit(starid='Kepler_1627', N_samples=2000):
def run_RotGPtransit(starid='KOI_7368', N_samples=2000):
#def run_RotGPtransit(starid='KOI_7913', N_samples=2000):
#def run_RotGPtransit(starid='Kepler_1643', N_samples=1000):

    assert starid in ['Kepler_1627', 'KOI_7368', 'KOI_7913', 'Kepler_1643']

    modelid = 'simpletransit'
    norm_zero = False # normalize LC around 1, not 0

    PLOTDIR = os.path.join(RESULTSDIR, f'run_{modelid}_{starid}')
    if not os.path.exists(PLOTDIR):
        os.mkdir(PLOTDIR)

    datasets = OrderedDict()
    if starid == 'Kepler_1627':
        lctype = 'longcadence'
    elif starid in ['KOI_7368', 'KOI_7913', 'Kepler_1643']:
        lctype = starid
    else:
        raise NotImplementedError

    time, flux, flux_err, qual, texp = (
        get_manually_downloaded_kepler_lightcurve(
            lctype=starid, norm_zero=norm_zero
        )
    )

    # NOTE: we have an abundance of data -> drop all non-zero quality flags.
    sel = (qual == 0)
    time, flux, flux_err, texp = time[sel], flux[sel], flux_err[sel], texp

    from cdips.lcproc.detrend import transit_window_polynomial_remover
    outpath = join(PLOTDIR, f'{starid}_{modelid}.png')
    d = transit_window_polynomial_remover(
        time, flux, flux_err, EPHEMDICT[starid]['t0'],
        EPHEMDICT[starid]['per'], EPHEMDICT[starid]['tdur'],
        n_tdurs=EPHEMDICT[starid]['n_tdurs'],
        method='poly_4', plot_outpath=outpath
    )

    datasets = OrderedDict()

    time = np.hstack([d[f'time_{ix}'] for ix in range(d['ngroups'])])
    flux = np.hstack([d[f'flat_flux_{ix}'] for ix in range(d['ngroups'])])
    flux_err = np.hstack([d[f'flux_err_{ix}'] for ix in range(d['ngroups'])])
    texp = np.nanmedian(np.diff(time))

    datasets['kepler'] = [time, flux, flux_err, texp]

    # get prior
    priorpath = join(DATADIR, 'priors', f'{starid}_{modelid}_priors.py')
    if not os.path.exists(priorpath):
        raise FileNotFoundError(f'need to create {priorpath}')
    priormod = SourceFileLoader('prior', priorpath).load_module()
    priordict = priormod.priordict

    pklpath = join(BETTYDIR, f'run_{starid}_{modelid}.pkl')

    m = ModelFitter(modelid, datasets, priordict, plotdir=PLOTDIR,
                    pklpath=pklpath, overwrite=0, N_samples=N_samples,
                    N_cores=os.cpu_count())

    print(pm.summary(m.trace, var_names=list(priordict)))

    summdf = pm.summary(m.trace, var_names=list(priordict), round_to=10,
                        kind='stats', stat_funcs={'median':np.nanmedian},
                        extend=True)

    localpolyindivpanels = 1
    phaseplot = 1
    posttable = 1
    cornerplot = 1
    writevespa = 1
    getbecclimits = 1

    if localpolyindivpanels:
        outpath = join(
            PLOTDIR, f'{starid}_{modelid}_localpolyindivpanels_resids.png'
        )
        bp.plot_localpolyindivpanels(
            d, m, summdf, outpath, modelid=modelid, plot_resids=True
        )
        outpath = join(
            PLOTDIR, f'{starid}_{modelid}_localpolyindivpanels.png'
        )
        bp.plot_localpolyindivpanels(
            d, m, summdf, outpath, modelid=modelid
        )

    if phaseplot:
        outpath = join(PLOTDIR, f'{starid}_{modelid}_phaseplot.png')
        if starid == 'KOI_7368':
            ylimd = {'A':[-2, 1.5], 'B':[-0.19,0.19]}
        elif starid == 'KOI_7913':
            ylimd = {'A':[-1.5, 0.5], 'B':[-0.05,0.05]}
        else:
            ylimd = {'A':[-3.5, 2.5], 'B':[-0.19,0.19]}
        bp.plot_phasefold(m, summdf, outpath, modelid=modelid, inppt=1,
                          ylimd=ylimd, binsize_minutes=15,
                          singleinstrument='kepler')

    if posttable:
        outpath = join(PLOTDIR, f'{starid}_{modelid}_posteriortable.tex')
        make_posterior_table(pklpath, priordict, outpath, modelid, makepdf=1)

    if cornerplot:
        outpath = join(PLOTDIR, f'{starid}_{modelid}_trimmed_cornerplot.png')
        bp.plot_cornerplot(list(priordict), m, outpath)

    if writevespa:
        from rudolf.vespa import _write_vespa
        staridentifier = f'{starid}_{modelid}'
        _write_vespa(datasets, m.trace.posterior, staridentifier,
                     N_hours_from_transit=4, make_plot=True)

    if getbecclimits:
        from rudolf.helpers import get_becc_limits
        get_becc_limits(
            datasets, m.trace.posterior
        )


if __name__ == "__main__":
    run_RotGPtransit()
