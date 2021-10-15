import os
import rudolf.plotting as rp
from rudolf.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'rotation_period_windowslider')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

rp.plot_rotation_period_windowslider(PLOTDIR, koi7368=1)
rp.plot_rotation_period_windowslider(PLOTDIR)
