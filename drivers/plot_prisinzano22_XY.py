import os
from rudolf.paths import DATADIR, RESULTSDIR
from rudolf.plotting import plot_prisinzano22_XY

outdir = os.path.join(RESULTSDIR, 'prisinzano22_XY')
if not os.path.exists(outdir):
    os.mkdir(outdir)

plot_prisinzano22_XY(outdir, colorkey=None, show_realrandommw=1,
                     show_randommw=0, show_CepHer=0, noaxis=1,
                     hide_prisinzano=1)

#plot_prisinzano22_XY(outdir, colorkey=None, show_randommw=1, show_CepHer=0,
#                     noaxis=1, hide_prisinzano=1)
plot_prisinzano22_XY(outdir, colorkey=None, show_CepHer=1, noaxis=1)
