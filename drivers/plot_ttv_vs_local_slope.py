import os
import rudolf.plotting as rp
from rudolf.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'ttv_vs_local_slope')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

rp.plot_ttv_vs_local_slope(PLOTDIR)
