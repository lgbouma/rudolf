import os
from rudolf.paths import DATADIR, RESULTSDIR
from rudolf.plotting import plot_kerr21_XY

outdir = os.path.join(RESULTSDIR, 'kerr21_XY')
if not os.path.exists(outdir):
    os.mkdir(outdir)

for tablenum in [1,2]:
    plot_kerr21_XY(outdir, tablenum=tablenum, colorkey=None, dcut=200)
    plot_kerr21_XY(outdir, tablenum=tablenum, colorkey=None)
    assert 0
    if tablenum == 1:
        plot_kerr21_XY(outdir, tablenum=tablenum, colorkey=None, show_CepHer=1)
        plot_kerr21_XY(outdir, tablenum=tablenum, colorkey='Age')
        plot_kerr21_XY(outdir, tablenum=tablenum, colorkey='P')
        plot_kerr21_XY(outdir, tablenum=tablenum, colorkey='plx')
    if tablenum == 2:
        plot_kerr21_XY(outdir, tablenum=tablenum, colorkey='Weight')
