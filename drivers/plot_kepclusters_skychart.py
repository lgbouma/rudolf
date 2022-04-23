import os
import rudolf.plotting as rp
from rudolf.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'kepclusters_skychart')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

for showplanets in [1,0]:
    rp.plot_kepclusters_skychart(PLOTDIR)
    rp.plot_kepclusters_skychart(PLOTDIR, showplanets=showplanets, clusters=['Theia-520'])
    rp.plot_kepclusters_skychart(PLOTDIR, showplanets=showplanets, clusters=['Melange-2'])
    rp.plot_kepclusters_skychart(PLOTDIR, showplanets=showplanets, clusters=['Cep-Her'])
    rp.plot_kepclusters_skychart(PLOTDIR, showplanets=showplanets, clusters=['Cep-Her', 'δ Lyr', 'RSG-5', 'CH-2'])
    rp.plot_kepclusters_skychart(PLOTDIR, showplanets=showplanets, clusters=['Theia-520', 'Melange-2', 'Cep-Her', 'δ Lyr', 'RSG-5', 'CH-2'])
