import os
import rudolf.plotting as rp
from rudolf.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'flares')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

rp.plot_flare_checker(PLOTDIR, method='itergp')

# NOTE: decent
#rp.plot_flare_checker(PLOTDIR, method='gp')

# # tried these; didn't look great
# rp.plot_flare_checker(PLOTDIR, method='rspline')
# rp.plot_flare_checker(PLOTDIR, method='pspline')
