"""
Fit the Kepler-1627 / Kepler-1643 / KOI-7368 / KOI-7913 datasets, using a GP
for the stellar variability and a transit for the planet.
"""

import numpy as np, matplotlib.pyplot as plt, pandas as pd, pymc3 as pm
import os
from os.path import join
from collections import OrderedDict
from importlib.machinery import SourceFileLoader

try:
    import betty.plotting as bp
except ModuleNotFoundError as e:
    print(f'WRN! {e}')
    pass

from betty.posterior_table import make_posterior_table
from betty.modelfitter import ModelFitter
from betty.paths import BETTYDIR

from rudolf.helpers import (
    get_manually_downloaded_kepler_lightcurve
)
from rudolf.paths import DATADIR, RESULTSDIR

# NOTE: change starid as desired based on the dataset to use.
#def run_RotGPtransit(starid='Kepler_1627_Q15slc', N_samples=2000):
#def run_RotGPtransit(starid='Kepler_1627', N_samples=2000):
#def run_RotGPtransit(starid='KOI_7368', N_samples=2000):
def run_RotGPtransit(starid='KOI_7913', N_samples=2000):
#def run_RotGPtransit(starid='Kepler_1643', N_samples=1000):

    assert starid in ['Kepler_1627', 'Kepler_1627_Q15slc', 'KOI_7368',
                      'KOI_7913', 'Kepler_1643']

    # this line ensures I use the right python environment on my system
    assert os.environ['CONDA_DEFAULT_ENV'] == 'py38'

    modelid = 'RotGPtransit'

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

    datasets['keplerllc'] = [time[sel], flux[sel], flux_err[sel], texp]

    priorpath = join(DATADIR, 'priors', f'{starid}_{modelid}_priors.py')
    assert os.path.exists(priorpath)
    priormod = SourceFileLoader('prior', priorpath).load_module()
    priordict = priormod.priordict

    if 'log_depth' in priordict.keys():
        parametrization = 'log_depth_and_b'
    elif 'log_ror' in priordict.keys():
        parametrization = 'log_ror_and_b'
    else:
        raise NotImplementedError

    pklpath = join(BETTYDIR, f'run_{starid}_{modelid}.pkl')

    s = '' if starid == 'Kepler_1627' else '_'+starid
    PLOTDIR = os.path.join(RESULTSDIR, 'run_'+modelid+s)
    if not os.path.exists(PLOTDIR):
        os.mkdir(PLOTDIR)

    #
    # values: transit, gpJitterMean,  RotationTerm, JitterMean, LimbDark,
    # bonus, FullOptimize.
    #
    # #works
    # map_optimization_method = 'RotationTerm_transit'
    # #does not work (produces grazing transit)
    # map_optimization_method = 'transit_gpJitterMean_FullOptimize'
    # #works for Kepler_1627, KOI_7368
    map_optimization_method = 'RotationTerm_transit_FullOptimize'
    m = ModelFitter(modelid, datasets, priordict, plotdir=PLOTDIR,
                    pklpath=pklpath, overwrite=0, N_samples=N_samples,
                    N_cores=os.cpu_count(),
                    map_optimization_method=map_optimization_method)

    if parametrization == 'log_depth_and_b':
        var_names = [
            'mean','logg_star','r_star','t0','period', 'log_depth', 'b', 'ecc',
            'omega','u_star', 'log_jitter', 'log_prot','log_Q0','log_dQ',
            'sigma_rot', 'prot', 'f', 'rho_star', 'depth', 'ror', 'r_planet',
            'a_Rs', 'cosi', 'sini','T_14','T_13'
        ]
    elif parametrization == 'log_ror_and_b':
        # if in log_ror + b parametrization (to explore grazing), depth is not derived
        var_names = [
            'mean','logg_star','r_star','t0','period', 'log_ror', 'b', 'ecc',
            'omega','u_star', 'log_jitter', 'log_prot','log_Q0','log_dQ',
            'sigma_rot', 'prot', 'f', 'rho_star', 'ror', 'r_planet', 'a_Rs',
            'cosi', 'sini','T_14','T_13'
        ]

    print(pm.summary(m.trace, var_names=var_names))

    summdf = pm.summary(m.trace, var_names=var_names, round_to=10,
                        kind='stats', stat_funcs={'median':np.nanmedian},
                        extend=True)

    writevespa = 1
    phaseplot = 1
    cornerplot = 1
    posttable = 1
    phasedsubsets = 1
    getbecclimits = 1

    if cornerplot:
        outpath = join(PLOTDIR, f'{starid}_{modelid}_cornerplot.png')
        if starid == 'KOI_7913':
            # don't plot T_13 due to NaN-based convergence issues.
            corner_var_names = var_names[:-1]
        else:
            corner_var_names = var_names
        bp.plot_cornerplot(corner_var_names, m, outpath)

    if posttable:
        outpath = join(PLOTDIR, f'{starid}_{modelid}_posteriortable.tex')
        make_posterior_table(pklpath, priordict, outpath, modelid, makepdf=1,
                             var_names=var_names)

    if writevespa:
        from rudolf.vespa import _write_vespa
        staridentifier = f'{starid}_{modelid}'
        _write_vespa(datasets, m.trace.posterior, staridentifier,
                     N_hours_from_transit=4, make_plot=True)

    if phaseplot:
        outpath = join(PLOTDIR, f'{starid}_{modelid}_posterior_phaseplot.png')
        if starid == 'KOI_7368':
            ylimd = {'A':[-2, 1.5], 'B':[-0.19,0.19]}
        elif starid == 'KOI_7913':
            ylimd = {'A':[-1, 0.5], 'B':[-0.05,0.05]}
        else:
            ylimd = {'A':[-3.5, 2.5], 'B':[-0.19,0.19]}
        alpha = 1 # 3e-1 looks good
        bp.plot_phased_light_curve(datasets, m.trace.posterior, outpath,
                                   from_trace=True, ylimd=ylimd,
                                   map_estimate=m.map_estimate,
                                   do_hacky_reprerror=True,
                                   binsize_minutes=15, alpha=alpha)
        outpath = join(PLOTDIR,
                       f'{starid}_{modelid}_posterior_phaseplot_fullxlim.png')
        alpha = 1 #2e-2 looked good
        bp.plot_phased_light_curve(datasets, m.trace.posterior, outpath,
                                   from_trace=True, ylimd=ylimd,
                                   map_estimate=m.map_estimate, fullxlim=True,
                                   BINMS=0.5, do_hacky_reprerror=True,
                                   binsize_minutes=15, alpha=alpha)


    if getbecclimits:
        from rudolf.helpers import get_becc_limits
        get_becc_limits(
            datasets, m.trace.posterior
        )

    if phasedsubsets:
        outpath = join(PLOTDIR, f'{starid}_{modelid}_phasedsubsets_yearchunk.png')
        timesubsets = [(50,450,'Y1'), (450,850,'Y2'), (850,1250,'Y3'),
                       (1250,1650,'Y4') ]
        ylimd = {'0':[-4.5,0.5], '1':[-4.5,0.5]}
        bp.plot_phased_subsets(
            datasets, m.trace.posterior, outpath, timesubsets, from_trace=True,
            map_estimate=m.map_estimate, yoffsetNsigma=5, ylimd=ylimd,
            inch_per_subset=0.75, binsize_minutes=20
        )

        timepath = os.path.join(
            DATADIR,'phot','time_to_quarter_conversion.csv'
        )
        tdf = pd.read_csv(timepath)
        outpath = join(PLOTDIR, f'{starid}_{modelid}_phasedsubsets_quarterchunk.png')
        timesubsets = [(r['tstart'], r['tstop'], 'Q'+str(int(r['quarter'])))
                       for _,r in tdf.iterrows() if int(r['quarter'])!=0]
        ylimd = {'0':[-10.5,0.5], '1':[-10.5,0.5]}
        bp.plot_phased_subsets(
            datasets, m.trace.posterior, outpath, timesubsets, from_trace=True,
            map_estimate=m.map_estimate, yoffsetNsigma=3.5, ylimd=ylimd,
            inch_per_subset=0.35, binsize_minutes=20
        )


if __name__ == "__main__":
    run_RotGPtransit()
