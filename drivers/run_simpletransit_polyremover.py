"""
Fit the Kepler-1627 / Kepler-1643 / KOI-7368 / KOI-7913 datasets, using a local
polynomial for the stellar variability and a transit for the planet.

(NOTE: this was the model that was actually adopted for the Cep-Her paper!)
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
    'KOI_7368': {'t0': 2454970.06-2454833, 'per': 6.842939, 'tdur':2.54/24, 'n_tdurs':3.},
    'KOI_7913': {'t0': 2454987.513-2454833, 'per': 24.2783801, 'tdur':4.564/24, 'n_tdurs':2.5},
    'Kepler_1627': {'t0': 120.790531, 'per': 7.20280608, 'tdur':2.841/24, 'n_tdurs':3.5},
    'Kepler_1643': {'t0': 2454967.381-2454833, 'per': 5.34264143, 'tdur':2.401/24, 'n_tdurs':3.5},
}

def run_simpletransit_polyremover(starid='Kepler_1643', N_samples=2000, N_cores=32):

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
            lctype=lctype, norm_zero=norm_zero
        )
    )

    # NOTE: we have an abundance of data -> drop all non-zero quality flags.
    sel = (qual == 0)
    time, flux, flux_err, texp = time[sel], flux[sel], flux_err[sel], texp

    from cdips.lcproc.detrend import transit_window_polynomial_remover
    outpath = join(PLOTDIR, f'{starid}_{modelid}.png')

    drop_badtransits = {'min_pts_in_transit':2, 'drop_worst_rms_percentile':90}
    if starid == 'KOI_7368':
        drop_badtransits = {
            'min_pts_in_transit':2, 'drop_worst_rms_percentile':85,
            'badtimewindows':[(6.5,7.5), (260,261), (383,384), (458.3, 459.3),
                              (1094.7, 1095.7), (1368.5, 1369.5)],
            'x0': 123.0316919518882
        }

    d = transit_window_polynomial_remover(
        time, flux, flux_err, EPHEMDICT[starid]['t0'],
        EPHEMDICT[starid]['per'], EPHEMDICT[starid]['tdur'],
        n_tdurs=EPHEMDICT[starid]['n_tdurs'],
        method='poly_4', plot_outpath=outpath,
        drop_badtransits=drop_badtransits
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
                    N_cores=N_cores)

    print(pm.summary(m.trace, var_names=list(priordict)))

    summdf = pm.summary(m.trace, var_names=list(priordict), round_to=10,
                        kind='stats', stat_funcs={'median':np.nanmedian},
                        extend=True)

    localpolyindivpanels = 0
    phaseplot = 0
    spphaseplot = 1
    posttable = 0
    cornerplot = 0
    writevespa = 0
    getbecclimits = 0

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
        ylimds = {
            'KOI_7368':{'A':[-2, 1.5], 'B':[-0.39,0.39]},
            'KOI_7913':{'A':[-2, 1.5], 'B':[-0.39,0.39]},
            'Kepler_1627':{'A':[-3.5, 1.5], 'B':[-0.39,0.39]},
            'Kepler_1643':{'A':[-2, 1.5], 'B':[-0.39,0.39]},
        }
        bp.plot_phasefold(m, summdf, outpath, modelid=modelid, inppt=1,
                          ylimd=ylimds[starid], binsize_minutes=20,
                          singleinstrument='kepler')

    if spphaseplot:
        # sub-options!
        darkcolors = 0
        showresid = 1
        s = ''
        if darkcolors:
            s += '_darkcolors'
        if not showresid:
            s += '_noresid'
        outpath = join(PLOTDIR, f'{starid}_{modelid}_singlepanelphaseplot{s}.png')
        if showresid:
            ylimds = {
                'KOI_7368':[-4, 1.5],
                'KOI_7913':[-5.5, 1.5],
                'Kepler_1627':[-6.5, 1.5],
                'Kepler_1643':[-4, 1.5],
            }
        else:
            ylimds = {
                'KOI_7368':[-3.5, 1.5],
                'KOI_7913':[-3.5, 1.5],
                'Kepler_1627':[-3.5, 1.5],
                'Kepler_1643':[-3.5, 1.5],
            }
        dyfactor = 4.2 if starid != 'Kepler_1627' else 2.8
        bp.plot_singlepanelphasefold(
            m, summdf, outpath, dyfactor=dyfactor, txt=starid.replace('_','-'),
            modelid=modelid, inppt=1, ylims=ylimds[starid], binsize_minutes=20,
            singleinstrument='kepler', darkcolors=darkcolors,
            showresid=showresid
        )

    if posttable:
        outpath = join(PLOTDIR, f'{starid}_{modelid}_posteriortable.tex')
        make_posterior_table(pklpath, priordict, outpath, modelid, makepdf=1)

    if cornerplot:
        outpath = join(PLOTDIR, f'{starid}_{modelid}_trimmed_cornerplot.png')
        bp.plot_cornerplot(list(priordict), m, outpath)

    if writevespa:
        from rudolf.vespa import _write_vespa
        _write_vespa(datasets, m.trace.posterior, starid, modelid,
                     N_hours_from_transit=4, make_plot=True)

    if getbecclimits:
        from rudolf.helpers import get_becc_limits
        get_becc_limits(
            datasets, m.trace.posterior
        )


if __name__ == "__main__":
    for starid in ['Kepler_1627', 'KOI_7368', 'KOI_7913', 'Kepler_1643']:
        run_simpletransit_polyremover(
            starid=starid, N_samples=2000, N_cores=os.cpu_count()
        )
