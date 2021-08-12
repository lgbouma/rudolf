import os
import rudolf.plotting as rp
from rudolf.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'RM')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

rp.plot_RM(PLOTDIR)

rp.plot_RM_and_phot(PLOTDIR)
