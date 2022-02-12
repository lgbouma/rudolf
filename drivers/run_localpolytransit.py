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
    'KOI_7913': {'t0': 2454987.513-2454833, 'per': 24.2783801, 'tdur':4.564/24},
}

# NOTE: change starid as desired based on the dataset to use.
#def run_RotGPtransit(starid='Kepler_1627_Q15slc', N_samples=2000):
#def run_RotGPtransit(starid='Kepler_1627', N_samples=2000):
#def run_RotGPtransit(starid='KOI_7368', N_samples=2000):
def run_RotGPtransit(starid='KOI_7913', N_samples=2000):
#def run_RotGPtransit(starid='Kepler_1643', N_samples=1000):

    assert starid in ['Kepler_1627', 'Kepler_1627_Q15slc', 'KOI_7368',
                      'KOI_7913', 'Kepler_1643']

    modelid = 'localpolytransit'

    PLOTDIR = os.path.join(RESULTSDIR, f'run_{modelid}_{starid}')
    if not os.path.exists(PLOTDIR):
        os.mkdir(PLOTDIR)

    datasets = OrderedDict()
    if starid == 'Kepler_1627':
        time, flux, flux_err, qual, texp = (
            get_manually_downloaded_kepler_lightcurve(lctype='longcadence')
        )
    elif starid == 'Kepler_1627_Q15slc':
        time, flux, flux_err, qual, texp = (
            get_manually_downloaded_kepler_lightcurve(lctype='shortcadence')
        )
    elif starid in ['KOI_7368', 'KOI_7913', 'Kepler_1643']:
        time, flux, flux_err, qual, texp = (
            get_manually_downloaded_kepler_lightcurve(lctype=starid)
        )
    else:
        raise NotImplementedError

    # NOTE: we have an abundance of data -> drop all non-zero quality flags.
    sel = (qual == 0)
    time, flux, flux_err, texp = time[sel], flux[sel], flux_err[sel], texp

    # trim
    outpath = join(PLOTDIR, f'{starid}_rawlc.png')
    _quicklcplot(time, flux, flux_err, outpath)
    n_tdurs = 4.0
    time, flux, flux_err = _subset_cut(
        time, flux, flux_err, n=n_tdurs, t0=EPHEMDICT[starid]['t0'],
        per=EPHEMDICT[starid]['per'], tdur=EPHEMDICT[starid]['tdur']
    )
    outpath = join(PLOTDIR, f'{starid}_rawtrimlc.png')
    _quicklcplot(time, flux, flux_err, outpath)

    # build datasets
    mingap = EPHEMDICT[starid]['per'] - 3*n_tdurs*EPHEMDICT[starid]['tdur']
    assert mingap > 0
    ngroups, groupinds = find_lc_timegroups(time, mingap=mingap)

    for ix, g in enumerate(groupinds):
        tess_texp = np.nanmedian(np.diff(time[g]))
        datasets[f'kepler_{ix}'] = [time[g], flux[g], flux_err[g], tess_texp]

    # get / make prior
    priorpath = join(DATADIR, 'priors', f'{starid}_{modelid}_priors.py')
    if not os.path.exists(priorpath):
        raise FileNotFoundError(f'need to create {priorpath}')
    priormod = SourceFileLoader('prior', priorpath).load_module()
    _init_priordict = priormod.priordict
    priordict = deepcopy(_init_priordict)

    if modelid == 'localpolytransit' and 'kepler_0_mean' not in priordict.keys():
        for n, (name, (x, y, yerr, texp)) in enumerate(datasets.items()):
            # mean + a1*(time-midtime) + a2*(time-midtime)^2.
            priordict[f'{name}_mean'] = ('Normal', 0, 0.1)
            priordict[f'{name}_a1'] = ('Uniform', -0.1, 0.1)
            priordict[f'{name}_a2'] = ('Uniform', -0.1, 0.1)
    given_priordict_make_priorfile(priordict, priorpath)

    pklpath = join(BETTYDIR, f'run_{starid}_{modelid}.pkl')

    m = ModelFitter(modelid, datasets, priordict, plotdir=PLOTDIR,
                    pklpath=pklpath, overwrite=0, N_samples=N_samples,
                    N_cores=os.cpu_count())

    print(pm.summary(m.trace, var_names=list(priordict)))

    summdf = pm.summary(m.trace, var_names=var_names, round_to=10,
                        kind='stats', stat_funcs={'median':np.nanmedian},
                        extend=True)

    writevespa = 1
    phaseplot = 1
    cornerplot = 1
    posttable = 1
    phasedsubsets = 1
    getbecclimits = 1
    phasesampleplot = 1

    if phaseplot:
        outpath = join(PLOTDIR, f'{starid}_{modelid}_phaseplot.png')
        bp.plot_phasefold(m, summdf, outpath, modelid=modelid, inppt=1,
                          binsize_minutes=15, singleinstrument='tess')

    if fitindivpanels:
        outpath = join(PLOTDIR, f'{starid}_{modelid}_fitindivpanels.png')
        bp.plot_fitindivpanels(m, summdf, outpath, modelid=modelid,
                               singleinstrument='tess')

    if posttable:
        outpath = join(PLOTDIR, f'{starid}_{modelid}_trimmed_posteriortable.tex')
        make_posterior_table(pklpath, _init_priordict, outpath, modelid, makepdf=1)

    if posttable:
        outpath = join(PLOTDIR, f'{starid}_{modelid}_posteriortable.tex')
        make_posterior_table(pklpath, priordict, outpath, modelid, makepdf=1)

    if cornerplot:
        outpath = join(PLOTDIR, f'{starid}_{modelid}_trimmed_cornerplot.png')
        bp.plot_cornerplot(list(_init_priordict), m, outpath)

    if writevespa:
        # TODO TODO TODO FIXME FIXME: needs re-implementation with this
        # localpolytransit model...
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
