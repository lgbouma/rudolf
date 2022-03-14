import os
import rudolf.plotting as rp
from rudolf.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'CepHer_XYZvtang_sky')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

rp.plot_CepHer_XYZvtang_sky(
    PLOTDIR
)
