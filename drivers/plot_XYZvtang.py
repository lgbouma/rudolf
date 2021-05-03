import os
import rudolf.plotting as rp
from rudolf.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'XYZvtang')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

rp.plot_XYZvtang(PLOTDIR)
rp.plot_XYZvtang(PLOTDIR, show1627=1)
