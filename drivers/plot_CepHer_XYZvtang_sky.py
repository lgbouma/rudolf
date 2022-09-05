import os
import rudolf.plotting as rp
from rudolf.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'CepHer_XYZvtang_sky')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

rp.plot_CepHer_XYZvtang_sky(
    PLOTDIR, showgroups=1, lsrcorr=1
)

for showgroups in [1,0]:
    rp.plot_CepHer_XYZvtang_sky(
        PLOTDIR, showgroups=showgroups
    )
