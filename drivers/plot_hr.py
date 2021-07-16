import os
import rudolf.plotting as rp
from rudolf.paths import RESULTSDIR

isochrones = [None]#, 'parsec', 'mist'] # could be None, mist, parsec.
colors = ['phot_bp_mean_mag', 'phot_g_mean_mag']

clusters = ['δ Lyr cluster', 'IC 2602', 'Pleiades', 'μ Tau', 'UCL']

PLOTDIR = os.path.join(RESULTSDIR, 'hr')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

for isochrone in isochrones:
    for c in colors:
        for o in [1]:
            for reddening_corr in [1,0]:
                rp.plot_hr(PLOTDIR, isochrone=isochrone, color0=c, show100pc=o,
                           clusters=clusters, reddening_corr=reddening_corr)
