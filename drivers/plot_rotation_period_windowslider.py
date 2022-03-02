import os
import rudolf.plotting as rp
from rudolf.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'rotation_period_windowslider')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

rp.plot_rotation_period_windowslider(PLOTDIR, 'KOI_7913_B')
rp.plot_rotation_period_windowslider(PLOTDIR, 'KOI_7913')
rp.plot_rotation_period_windowslider(PLOTDIR, 'Kepler_1643')
rp.plot_rotation_period_windowslider(PLOTDIR, 'KOI_7368')
rp.plot_rotation_period_windowslider(PLOTDIR, 'Kepler_1627')
