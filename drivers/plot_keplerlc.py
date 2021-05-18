import os
import rudolf.plotting as rp
from rudolf.paths import RESULTSDIR
import numpy as np

PLOTDIR = os.path.join(RESULTSDIR, 'keplerlc')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

N_samples=500
for t0 in np.arange(0,1500,100):
    t1 = t0+100
    rp.plot_keplerlc(PLOTDIR, N_samples, [t0,t1])
