import os
import rudolf.plotting as rp
from rudolf.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'kepclusters_skychart')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

for showplanets in [1,0]:

    # COO 23B
    rp.plot_kepclusters_skychart(
        PLOTDIR, showplanets=showplanets,
        clusters=['Cep-Her', 'δ Lyr keck', 'RSG-5 keck', 'ch01', 'ch03', 'ch04', 'ch06'],
        showkepclusters=0,
        style='clean',
        factor=0.5,
        cepher_alpha=0.5,
        figx=16/2,
        figy=7/2
    )


    ## poster
    #rp.plot_kepclusters_skychart(PLOTDIR, showplanets=showplanets,
    #                             clusters=['Cep-Her'], showkepclusters=0,
    #                             darkcolors=1)

    ## AAS240
    #rp.plot_kepclusters_skychart(PLOTDIR, showplanets=showplanets,
    #                             clusters=['Cep-Her'], showkepclusters=0,
    #                             hideaxes=1)

    # Exoplanets-4
    #rp.plot_kepclusters_skychart(PLOTDIR, showplanets=showplanets, clusters=['Theia-520'])
    #rp.plot_kepclusters_skychart(PLOTDIR, showplanets=showplanets, clusters=['Theia-520', 'Melange-2'])
    #rp.plot_kepclusters_skychart(PLOTDIR, showplanets=showplanets, clusters=['Theia-520', 'Melange-2', 'Cep-Her'])

    # rp.plot_kepclusters_skychart(PLOTDIR)
    # rp.plot_kepclusters_skychart(PLOTDIR, showplanets=showplanets, clusters=['Cep-Her'])
    # rp.plot_kepclusters_skychart(PLOTDIR, showplanets=showplanets, clusters=['Cep-Her', 'δ Lyr', 'RSG-5', 'CH-2'])
    # rp.plot_kepclusters_skychart(PLOTDIR, showplanets=showplanets, clusters=['Theia-520', 'Melange-2', 'Cep-Her', 'δ Lyr', 'RSG-5', 'CH-2'])
