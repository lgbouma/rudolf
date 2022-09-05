import os
import rudolf.plotting as rp
from rudolf.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'kepclusters_skychart')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

for showplanets in [1,0]:
    ## poster
    #rp.plot_kepclusters_skychart(PLOTDIR, showplanets=showplanets,
    #                             clusters=['Cep-Her'], showkepclusters=0,
    #                             darkcolors=1)

    # AAS240
    rp.plot_kepclusters_skychart(PLOTDIR, showplanets=showplanets,
                                 clusters=['Cep-Her'], showkepclusters=0,
                                 hideaxes=1)

    # Exoplanets-4
    #rp.plot_kepclusters_skychart(PLOTDIR, showplanets=showplanets, clusters=['Theia-520'])
    #rp.plot_kepclusters_skychart(PLOTDIR, showplanets=showplanets, clusters=['Theia-520', 'Melange-2'])
    #rp.plot_kepclusters_skychart(PLOTDIR, showplanets=showplanets, clusters=['Theia-520', 'Melange-2', 'Cep-Her'])

    # rp.plot_kepclusters_skychart(PLOTDIR)
    # rp.plot_kepclusters_skychart(PLOTDIR, showplanets=showplanets, clusters=['Cep-Her'])
    # rp.plot_kepclusters_skychart(PLOTDIR, showplanets=showplanets, clusters=['Cep-Her', 'δ Lyr', 'RSG-5', 'CH-2'])
    # rp.plot_kepclusters_skychart(PLOTDIR, showplanets=showplanets, clusters=['Theia-520', 'Melange-2', 'Cep-Her', 'δ Lyr', 'RSG-5', 'CH-2'])
