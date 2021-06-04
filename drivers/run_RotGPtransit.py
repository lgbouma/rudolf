"""
Fit the Kepler 1627 dataset, using a GP for the stellar variability and a
transit for the planet.
"""

import numpy as np, matplotlib.pyplot as plt, pandas as pd, pymc3 as pm
import pickle, os, corner, pytest
from collections import OrderedDict
import exoplanet as xo

from os.path import join
from importlib.machinery import SourceFileLoader

try:
    import betty.plotting as bp
except ModuleNotFoundError as e:
    print(f'WRN! {e}')
    pass

from rudolf.helpers import get_kep1627_kepler_lightcurve
from betty.posterior_table import make_posterior_table
from betty.modelfitter import ModelFitter

from rudolf.paths import DATADIR, RESULTSDIR
from betty.paths import BETTYDIR

def run_RotGPtransit(starid='Kepler_1627', N_samples=2000):

    # this line ensures I use the right python environment on my system
    assert os.environ['CONDA_DEFAULT_ENV'] == 'py38'

    modelid = 'RotGPtransit'

    datasets = OrderedDict()
    if starid == 'Kepler_1627':
        time, flux, flux_err, qual, texp = get_kep1627_kepler_lightcurve()
    else:
        raise NotImplementedError

    # NOTE: we have an abundance of data -> drop all non-zero quality flags.
    sel = (qual == 0)

    datasets['keplerllc'] = [time[sel], flux[sel], flux_err[sel], texp]

    priorpath = join(DATADIR, 'priors', f'{starid}_{modelid}_priors.py')
    assert os.path.exists(priorpath)
    priormod = SourceFileLoader('prior', priorpath).load_module()
    priordict = priormod.priordict

    pklpath = join(BETTYDIR, f'run_{starid}_{modelid}.pkl')

    PLOTDIR = os.path.join(RESULTSDIR, 'run_'+modelid)
    if not os.path.exists(PLOTDIR):
        os.mkdir(PLOTDIR)

    m = ModelFitter(modelid, datasets, priordict, plotdir=PLOTDIR,
                    pklpath=pklpath, overwrite=0, N_samples=N_samples,
                    N_cores=os.cpu_count())

    var_names = [
        'mean','logg_star','t0','period','log_r','log_jitter',
        'log_prot','log_Q0','log_dQ','r_star','rho_star','u_star', 'r',
        'b', 'ecc', 'omega', 'sigma_rot', 'prot', 'f',
        'r_planet', 'a_Rs', 'cosi', 'sini','T_14','T_13'
    ]

    print(pm.summary(m.trace, var_names=var_names))

    summdf = pm.summary(m.trace, var_names=var_names, round_to=10,
                        kind='stats', stat_funcs={'median':np.nanmedian},
                        extend=True)

    phaseplot = 1
    cornerplot = 1
    posttable = 1

    if phaseplot:
        outpath = join(PLOTDIR, f'{starid}_{modelid}_posterior_phaseplot.png')
        ylimd = {'A':[-3.5, 2.5], 'B':[-0.5,0.5]}
        bp.plot_phased_light_curve(datasets, m.trace.posterior, outpath,
                                   from_trace=True, ylimd=ylimd,
                                   map_estimate=m.map_estimate)
        outpath = join(PLOTDIR,
                       f'{starid}_{modelid}_posterior_phaseplot_fullxlim.png')
        ylimd = {'A':[-3.5, 2.5], 'B':[-1,1]}
        bp.plot_phased_light_curve(datasets, m.trace.posterior, outpath,
                                   from_trace=True, ylimd=ylimd,
                                   map_estimate=m.map_estimate, fullxlim=True,
                                   BINMS=0.5)

    if posttable:
        outpath = join(PLOTDIR, f'{starid}_{modelid}_posteriortable.tex')
        make_posterior_table(pklpath, priordict, outpath, modelid, makepdf=1,
                             var_names=var_names)

    if cornerplot:
        outpath = join(PLOTDIR, f'{starid}_{modelid}_cornerplot.png')
        bp.plot_cornerplot(var_names, m, outpath)


if __name__ == "__main__":
    run_RotGPtransit()
