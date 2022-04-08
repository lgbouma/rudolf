import os
import rudolf.plotting as rp
from rudolf.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'halpha')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

rp.plot_halpha(PLOTDIR, reference='TucHor')

