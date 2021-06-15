import os
import rudolf.plotting as rp
from rudolf.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'ttv')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

rp.plot_ttv(PLOTDIR, narrowylim=1)
rp.plot_ttv(PLOTDIR)
