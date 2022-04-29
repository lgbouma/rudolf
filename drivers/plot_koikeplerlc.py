import os
import rudolf.plotting as rp
from rudolf.paths import RESULTSDIR
import numpy as np

PLOTDIR = os.path.join(RESULTSDIR, 'koikeplerlc')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

rp.plot_koikeplerlc(PLOTDIR, [550,600])
for t0 in np.arange(0,1500,50):
    t1 = t0+50
    rp.plot_koikeplerlc(PLOTDIR, [t0,t1])
