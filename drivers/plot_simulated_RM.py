import os
import rudolf.plotting as rp
from rudolf.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'simulated_RM')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

rp.plot_simulated_RM(PLOTDIR, 'prograde')
rp.plot_simulated_RM(PLOTDIR, 'retrograde')
rp.plot_simulated_RM(PLOTDIR, 'polar')
