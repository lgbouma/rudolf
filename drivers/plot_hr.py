import os
import rudolf.plotting as rp
from rudolf.paths import RESULTSDIR

isochrones = [None]#, 'parsec', 'mist'] # could be None, mist, parsec.
colors = ['phot_bp_mean_mag']#, 'phot_g_mean_mag']

#clusters = ['δ Lyr cluster', 'IC 2602', 'Pleiades', 'μ Tau', 'UCL']

PLOTDIR = os.path.join(RESULTSDIR, 'hr')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

rp.plot_hr(PLOTDIR, isochrone=None, color0='phot_bp_mean_mag', show100pc=1,
           clusters=['δ Lyr cluster', 'Set1', 'IC 2602', 'Pleiades'], reddening_corr=1,
           overplotkep1627=1, show_allknown=1, smalllims=1)
assert 0

# as in ms, but with "Set 1" (KOI 7368)
rp.plot_hr(PLOTDIR, isochrone=None, color0='phot_bp_mean_mag', show100pc=1,
           clusters=['δ Lyr cluster', 'Set1', 'IC 2602', 'Pleiades'], reddening_corr=1,
           overplotkep1627=1, overplotkoi7368=1)
rp.plot_hr(PLOTDIR, isochrone=None, color0='phot_bp_mean_mag', show100pc=1,
           clusters=['δ Lyr cluster', 'Set1', 'IC 2602', 'Pleiades'], reddening_corr=1,
           smalllims=1, overplotkep1627=1, overplotkoi7368=1)
for iso in [None, 'mist', 'parsec']:
    rp.plot_hr(PLOTDIR, isochrone=iso, color0='phot_bp_mean_mag', show100pc=1,
               clusters=['δ Lyr cluster', 'Set1', 'IC 2602', 'Pleiades'],
               reddening_corr=1, overplotkep1627=1, overplotkoi7368=1, getstellarparams=1)

# actually used in kep-1627 manuscript
rp.plot_hr(PLOTDIR, isochrone=None, color0='phot_bp_mean_mag', show100pc=1,
           clusters=['δ Lyr cluster', 'IC 2602', 'Pleiades'], reddening_corr=1,
           overplotkep1627=1)
rp.plot_hr(PLOTDIR, isochrone=None, color0='phot_bp_mean_mag', show100pc=1,
           clusters=['δ Lyr cluster', 'IC 2602', 'Pleiades'], reddening_corr=1,
           smalllims=1, overplotkep1627=1)

# useful isochrone fitting tests
for iso in [None, 'mist', 'parsec']:
    rp.plot_hr(PLOTDIR, isochrone=iso, color0='phot_bp_mean_mag', show100pc=1,
               clusters=['δ Lyr cluster', 'IC 2602', 'Pleiades'],
               reddening_corr=1, overplotkep1627=1, getstellarparams=1)

# subsets actually used in paper with & w/out reddening
for r in [1,0]:
    rp.plot_hr(PLOTDIR, isochrone=None, color0='phot_bp_mean_mag', show100pc=0,
               clusters=['δ Lyr cluster'], reddening_corr=r)
    rp.plot_hr(PLOTDIR, isochrone=None, color0='phot_bp_mean_mag', show100pc=1,
               clusters=['δ Lyr cluster', 'IC 2602', 'Pleiades'], reddening_corr=r)
    rp.plot_hr(PLOTDIR, isochrone=None, color0='phot_bp_mean_mag', show100pc=1,
               clusters=['δ Lyr cluster', 'IC 2602', 'Pleiades'], reddening_corr=r,
               smalllims=1)
    rp.plot_hr(PLOTDIR, isochrone=None, color0='phot_bp_mean_mag', show100pc=1,
               clusters=['δ Lyr cluster', 'IC 2602', 'Pleiades', 'UCL'],
               reddening_corr=r, smalllims=1)

#BPMG checks
for r in [1,0]:
    rp.plot_hr(PLOTDIR, isochrone=None, color0='phot_bp_mean_mag', show100pc=1,
               clusters=['δ Lyr cluster', 'IC 2602', 'Pleiades', 'UCL', 'BPMG'],
               reddening_corr=r, smalllims=1)
