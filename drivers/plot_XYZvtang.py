import os
import rudolf.plotting as rp
from rudolf.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'XYZvtang')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

rp.plot_XYZvtang(PLOTDIR, show_1627=0, show_comovers=0, show_sun=1,
                 show_7368=0, show_allknown=1, show_rsg5=1,
                 show_set1=1, orientation='square')

rp.plot_XYZvtang(PLOTDIR, show_1627=1, show_comovers=0, show_sun=1,
                 show_7368=0, show_allknown=1, show_rsg5=1, orientation='square')

rp.plot_XYZvtang(PLOTDIR, show_1627=1, show_comovers=0, show_sun=1,
                 show_7368=0, show_allknown=1, show_rsg5=1, orientation='portrait')

rp.plot_XYZvtang(PLOTDIR, show_1627=0, show_comovers=0, show_sun=1,
                 show_7368=0, show_allknown=1, orientation='portrait')

rp.plot_XYZvtang(PLOTDIR, show_1627=1, show_comovers=1, show_sun=1)
rp.plot_XYZvtang(PLOTDIR, show_1627=1, show_comovers=1, show_sun=1,
                 show_7368=1)
rp.plot_XYZvtang(PLOTDIR, show_1627=1, show_comovers=1, show_sun=1,
                 orientation='portrait')
rp.plot_XYZvtang(PLOTDIR, show_1627=1, show_comovers=1)
rp.plot_XYZvtang(PLOTDIR, show_1627=1)
rp.plot_XYZvtang(PLOTDIR)
