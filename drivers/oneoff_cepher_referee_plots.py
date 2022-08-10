import os
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from cdips.utils.gaiaqueries import given_source_ids_get_gaia_data

df = pd.read_csv('../papers/paper/tab_supp_CepHer_X_Kepler.csv')

gdf = given_source_ids_get_gaia_data(
    np.array(df.dr3_source_id).astype(np.int64), 'ch_kepler_gaia_source',
    n_max=10000, overwrite=False, enforce_all_sourceids_viable=False,
    savstr='', whichcolumns='*', table_name='gaia_source',
    gaia_datarelease='gaiadr3'
)

plt.scatter(
    gdf.bp_rp, gdf.grvs_mag, s=10, c='k', marker='.', linewidths=0, zorder=1
)

sel = ~pd.isnull(gdf.radial_velocity)
plt.scatter(
    gdf[sel].bp_rp, gdf[sel].grvs_mag, s=10, c='C1', marker='.', linewidths=0,
    zorder=2
)

plt.xlabel('BP-RP')
plt.ylabel('G_RVS')
outdir = '../results/cepher_referee_plots'
if not os.path.exists(outdir): os.mkdir(outdir)
outpath = os.path.join(outdir, 'DR3_RV_availability.png')
plt.savefig(outpath, bbox_inches='tight', dpi=350)


assert len(gdf) == len(df)
plt.close('all')
plt.scatter(
    gdf.bp_rp, gdf.radial_velocity, s=40, c=np.log10(df.weight),
    cmap='viridis', marker='.', linewidths=0, zorder=1,
    vmin=-1.2, vmax=-0.5
)
plt.colorbar(label='log10(weight)', extend='both')

plt.xlabel('BP-RP')
plt.ylabel('RV [km/s]')
outpath = os.path.join(outdir, 'RV_vs_BP-RP.png')
plt.savefig(outpath, bbox_inches='tight', dpi=350)

# for comparison against Psc-Eri counts
plt.close('all')
plt.hist(
    gdf.phot_g_mean_mag, bins=20
)
plt.xlabel('g')
plt.xticks(np.arange(6,18,1))
outpath = os.path.join(outdir, 'hist_g.png')
plt.savefig(outpath, bbox_inches='tight', dpi=350)
