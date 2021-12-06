import os
import rudolf.plotting as rp
from rudolf.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'galex')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

for xval in ['bpmrp0', 'jmk']:
    rp.plot_galex(
        PLOTDIR, overplotkep1627=1,
        clusters=['Pleiades', 'δ Lyr cluster'],
        extinctionmethod='gaia2018', xval=xval
    )
