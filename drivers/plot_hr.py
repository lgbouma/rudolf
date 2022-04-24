import os
import rudolf.plotting as rp
from rudolf.paths import RESULTSDIR

isochrones = [None]#, 'parsec', 'mist'] # could be None, mist, parsec.
colors = ['phot_bp_mean_mag']#, 'phot_g_mean_mag']

#clusters = ['δ Lyr cluster', 'IC 2602', 'Pleiades', 'μ Tau', 'UCL']

PLOTDIR = os.path.join(RESULTSDIR, 'hr')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

# show all the known KOIs that are ~40 Myr old
for smalllims in [1,0]:

    rp.plot_hr(PLOTDIR, isochrone=None, color0='phot_bp_mean_mag', show100pc=1,
               clusters=['δ Lyr cluster', 'CH-2'], reddening_corr=1,
               overplotkep1627=0, show_allknown=0, overplotkoi7368=1,
               overplotkoi7913=1, smalllims=smalllims)

    rp.plot_hr(PLOTDIR, isochrone=None, color0='phot_bp_mean_mag', show100pc=1,
               clusters=['δ Lyr cluster', 'RSG-5'], reddening_corr=1,
               overplotkep1627=0, show_allknown=0, overplotkep1643=1,
               smalllims=smalllims)

assert 0

# get the isochrones for KOI-7913 and Kepler-1643
for iso in ['mist', 'parsec']:
    rp.plot_hr(PLOTDIR, isochrone=iso, color0='phot_bp_mean_mag', show100pc=1,
               clusters=['δ Lyr cluster', 'IC 2602', 'Pleiades'],
               reddening_corr=1, overplotkep1643=1, getstellarparams=1)
    rp.plot_hr(PLOTDIR, isochrone=iso, color0='phot_bp_mean_mag', show100pc=1,
               clusters=['δ Lyr cluster', 'IC 2602', 'Pleiades'],
               reddening_corr=1, overplotkoi7913=1, getstellarparams=1)

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
