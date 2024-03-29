import os
import rudolf.plotting as rp
from rudolf.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'lithium')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

rp.plot_lithium(PLOTDIR, reference='Pleiades_Randich01_TucHorK14M08')
assert 0
rp.plot_lithium(PLOTDIR, reference='Pleiades')
rp.plot_lithium(PLOTDIR, reference='Randich01_TucHorK14M08')
rp.plot_lithium(PLOTDIR, reference='Randich01')
rp.plot_lithium(PLOTDIR, reference='Randich18')
