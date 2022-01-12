import os
import rudolf.plotting as rp
from rudolf.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'skychart')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

rp.plot_skychart(PLOTDIR, narrowlims=0, showkepler=1, showtess=0,
                 shownakedeye=0, showcomovers=0, showkepstars=0,
                 showkepclusters=1, showdellyrcluster=0)

rp.plot_skychart(PLOTDIR, narrowlims=0, showkepler=1, showtess=0,
                 shownakedeye=0, showcomovers=1, showkepstars=0,
                 showkepclusters=1)

rp.plot_skychart(PLOTDIR, narrowlims=0, showkepler=1, showtess=0,
                 shownakedeye=0, showcomovers=1, showkepstars=0)


rp.plot_skychart(PLOTDIR, narrowlims=0, showkepler=1, showtess=0,
                 shownakedeye=0, showcomovers=1, showkepstars=1)

rp.plot_skychart(PLOTDIR, narrowlims=0, showkepler=1, showtess=1,
                 shownakedeye=0, showcomovers=1)

rp.plot_skychart(PLOTDIR, narrowlims=0, showkepler=1, showtess=1,
                 shownakedeye=1, showcomovers=1)
rp.plot_skychart(PLOTDIR, narrowlims=1, showkepler=1, showtess=1,
                 shownakedeye=1, showcomovers=1)
rp.plot_skychart(PLOTDIR, narrowlims=1, showkepler=1, showtess=1,
                 shownakedeye=1)

rp.plot_skychart(PLOTDIR, showkepler=1, showtess=1, shownakedeye=1)
rp.plot_skychart(PLOTDIR, narrowlims=1, showkepler=1, showtess=1)
rp.plot_skychart(PLOTDIR, showkepler=1, showtess=1)
rp.plot_skychart(PLOTDIR)
rp.plot_skychart(PLOTDIR, narrowlims=1)
rp.plot_skychart(PLOTDIR, narrowlims=1, showkepler=1)

