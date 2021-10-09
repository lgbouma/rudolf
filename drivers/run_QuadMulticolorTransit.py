"""
Fit a single transit + quadratic trend with multiple simultaneous bandpasses
(presumably a ground-based transit).
"""
import numpy as np, matplotlib.pyplot as plt, pandas as pd, pymc3 as pm
import pickle, os, corner, pytest
from collections import OrderedDict
from pymc3.backends.tracetab import trace_to_dataframe
import exoplanet as xo

from os.path import join
from importlib.machinery import SourceFileLoader

try:
    import betty.plotting as bp
except ModuleNotFoundError as e:
    print(f'WRN! {e}')
    pass

from betty.posterior_table import make_posterior_table
from betty.modelfitter import ModelFitter
from betty.paths import BETTYDIR

from rudolf.helpers import get_kep1627_muscat_lightcuve
from rudolf.paths import DATADIR, RESULTSDIR

def run_QuadMulticolorTransit(starid='Kepler_1627', N_samples=2000):

    assert starid in ['Kepler_1627']

    # this line ensures I use the right python environment on my system
    assert os.environ['CONDA_DEFAULT_ENV'] == 'py38'

    modelid = 'QuadMulticolorTransit'
    datasets = get_kep1627_muscat_lightcuve()

    priorpath = join(DATADIR, 'priors', f'{starid}_{modelid}_priors.py')
    assert os.path.exists(priorpath)
    priormod = SourceFileLoader('prior', priorpath).load_module()
    priordict = priormod.priordict

    pklpath = join(BETTYDIR, f'run_{starid}_{modelid}.pkl')

    s = '' if starid == 'Kepler_1627' else '_'+starid
    PLOTDIR = os.path.join(RESULTSDIR, 'run_'+modelid+s)
    if not os.path.exists(PLOTDIR):
        os.mkdir(PLOTDIR)

    map_optimization_method = None
    m = ModelFitter(modelid, datasets, priordict, plotdir=PLOTDIR,
                    pklpath=pklpath, overwrite=0, N_samples=N_samples,
                    N_cores=os.cpu_count(),
                    map_optimization_method=map_optimization_method)

    var_names = [
        'logg_star','t0','period','dur', 'log_dur', 'log_ror', 'r_star','rho_star', 'ror',
        'b', 'r_planet', 'a_Rs', 'cosi', 'sini','T_14','T_13'
    ]
    bandpasses = 'g,r,i,z'.split(',')
    for bandpass in bandpasses:
        var_names.append(f'muscat3_{bandpass}_mean')
        var_names.append(f'muscat3_{bandpass}_a1')
        var_names.append(f'muscat3_{bandpass}_a2')
        var_names.append(f'muscat3_{bandpass}_log_jitter')
        var_names.append(f'muscat3_{bandpass}_u_star')

    print(pm.summary(m.trace, var_names=var_names))

    summdf = pm.summary(m.trace, var_names=var_names, round_to=10,
                        kind='stats', stat_funcs={'median':np.nanmedian},
                        extend=True)

    multicolorplot = 1
    cornerplot = 1

    # not implemented
    fitindiv = 0
    phaseplot = 0
    posttable = 0

    if multicolorplot:
        pass

    if cornerplot:
        outpath = join(PLOTDIR, f'{starid}_{modelid}_cornerplot.png')
        bp.plot_cornerplot(var_names, m, outpath)

    if posttable:
        outpath = join(PLOTDIR, f'{starid}_{modelid}_posteriortable.tex')
        make_posterior_table(pklpath, priordict, outpath, modelid, makepdf=1)

    if phaseplot:
        outpath = join(PLOTDIR, f'{starid}_{modelid}_phaseplot.png')
        bp.plot_phasefold(m, summdf, outpath, modelid=modelid, inppt=1)

    if fitindiv:
        outpath = join(PLOTDIR, f'{starid}_{modelid}_fitindiv.png')
        bp.plot_fitindiv(m, summdf, outpath, modelid=modelid)


if __name__ == "__main__":
    run_QuadMulticolorTransit()
