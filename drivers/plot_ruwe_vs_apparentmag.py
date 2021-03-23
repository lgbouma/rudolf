import os
import rudolf.plotting as rp
from rudolf.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'gaia_ruwe_vs_apparentmag')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

rp.plot_ruwe_vs_apparentmag(PLOTDIR)
rp.plot_ruwe_vs_apparentmag(PLOTDIR)
