"""
Fit the Kepler 1627 dataset, using a GP for the stellar variability and a
transit for the planet.
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

from rudolf.helpers import (
    get_kep1627_kepler_lightcurve
)
from betty.helpers import _subset_cut
from betty.posterior_table import make_posterior_table
from betty.modelfitter import ModelFitter

from rudolf.paths import DATADIR, RESULTSDIR
from betty.paths import BETTYDIR

EPHEMDICT = {
    'Kepler_1627': {'t0': 1355.1845, 'per': 1.338231466, 'tdur':2.5/24},
}


def run_gptransit(starid='Kepler_1627', N_samples=1000):

    modelid = 'gptransit'

    datasets = OrderedDict()
    if starid == 'Kepler_1627':
        time, flux, flux_err, qual, texp = get_kep1627_kepler_lightcurve()
    else:
        raise NotImplementedError

    # NOTE: might want this, since it's 50k data points...
    # time, flux, flux_err = _subset_cut(
    #     time, flux, flux_err, n=2.0, t0=EPHEMDICT[starid]['t0'],
    #     per=EPHEMDICT[starid]['per'], tdur=EPHEMDICT[starid]['tdur']
    # )

    # NOTE: we have an abundance of data. so... drop all non-zero quality
    # flags.
    sel = (qual == 0)

    datasets['keplerllc'] = [time[sel], flux[sel], flux_err[sel], texp]

    priorpath = join(DATADIR, 'priors', f'{starid}_priors.py')
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

    import IPython; IPython.embed()
    # NOTE: care about the "ecc" and "omega" results here too... 

    print(pm.summary(m.trace, var_names=list(priordict)))

    summdf = pm.summary(m.trace, var_names=list(priordict), round_to=10,
                        kind='stats', stat_funcs={'median':np.nanmedian},
                        extend=True)

    fitindiv = 1
    phaseplot = 1
    cornerplot = 1
    posttable = 1

    if posttable:
        outpath = join(PLOTDIR, f'{starid}_{modelid}_posteriortable.tex')
        make_posterior_table(pklpath, priordict, outpath, modelid, makepdf=1)

    if phaseplot:
        outpath = join(PLOTDIR, f'{starid}_{modelid}_phaseplot.png')
        bp.plot_phasefold(m, summdf, outpath, modelid=modelid, inppt=1)

    if fitindiv:
        outpath = join(PLOTDIR, f'{starid}_{modelid}_fitindiv.png')
        bp.plot_fitindiv(m, summdf, outpath, modelid=modelid)

    if cornerplot:
        outpath = join(PLOTDIR, f'{starid}_{modelid}_cornerplot.png')
        bp.plot_cornerplot(list(priordict), m, outpath)



if __name__ == "__main__":
    run_gptransit()
