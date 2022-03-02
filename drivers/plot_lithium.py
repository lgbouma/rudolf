import os
import rudolf.plotting as rp
from rudolf.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'lithium')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

rp.plot_lithium(PLOTDIR)
