import os
import numpy as np, pandas as pd
from numpy import array as nparr
from rudolf.helpers import get_kep1627_kepler_lightcurve
from rudolf.iterative_bls import run_iterative_bls
from rudolf.paths import DATADIR, RESULTSDIR

outdir = os.path.join(RESULTSDIR, 'iterative_bls')
# made by plot_keplerlc.py, in a dumper under _plot_zoom_light_curve
incsv = os.path.join(outdir, 'gptransit_dtr_20210518.csv')
df = pd.read_csv(incsv)

x,y,yerr,gp_mod = nparr(df.x), nparr(df.y), nparr(df.yerr), nparr(df.gp_mod)

for pmax in [100,300]:
    run_iterative_bls(
        x, y-gp_mod, outdir, pmin=0.5, pmax=pmax, y_alreadysmoothed=True
    )
    run_iterative_bls(
        x, y, outdir, pmin=0.5, pmax=pmax, y_alreadysmoothed=False
    )

