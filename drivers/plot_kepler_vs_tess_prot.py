import os
import numpy as np, matplotlib.pyplot as plt, pandas as pd
from rudolf.paths import DATADIR, RESULTSDIR
from rudolf.helpers import (
    get_autorotation_dataframe, get_clean_gaia_photometric_sources
)

# GET DATA
auto_df, base_df = get_autorotation_dataframe(
    'deltaLyrCluster', cleaning='curtiscleaning'
)

csvpath = os.path.join(DATADIR, 'rotation',
                       'Theia73-Prot_Results-Prelim-v2.csv')
curtis_df = pd.read_csv(csvpath)

# kinematic bit
csvpath = os.path.join(
    RESULTSDIR, 'tables', 'deltalyr_kc19_cleansubset_withreddening.csv'
)
kinematic_df = pd.read_csv(csvpath)
N_kin = len(kinematic_df)

mdf0 = curtis_df.merge(base_df, how='inner', left_on='Gaia_DR2_Source',
                      right_on='dr2_source_id')

mdf = kinematic_df.merge(mdf0, how='left', left_on='source_id',
                         right_on='dr3_source_id')


selcols = ['source_id', 'phot_bp_mean_mag_corr',
           'phot_rp_mean_mag_corr', 'phot_g_mean_mag_corr']
kinematic_df = kinematic_df[get_clean_gaia_photometric_sources(kinematic_df)]
mdf1 = pd.DataFrame(kinematic_df[selcols]).merge(
    auto_df, left_on='source_id', right_on='dr3_source_id',
    how='inner'
)

# make plot

plt.figure(figsize=(6,6))
plt.scatter(mdf.Prot_Kepler_y, mdf.Prot_LS_Auto_y, s=1, c='k')
_ = np.linspace(0,40,1000)
plt.plot(_,_,ls='-',zorder=-1,color='gray',lw=0.5,
         label='Prot TESS = Prot Kepler')
plt.plot(_,0.5*_,ls=':',zorder=-1,color='gray',lw=0.5,
         label='Prot TESS = 0.5*Prot Kepler')
plt.plot(_,2*_,ls='--',zorder=-1,color='gray',lw=0.5,
         label='Prot TESS = 2*Prot Kepler')
plt.legend(loc='best', fontsize='x-small')
plt.xlabel("Prot Kepler [d]")
plt.ylabel("Prot TESS [d]")
plt.xscale('log'); plt.yscale('log')
plt.ylim([0.1,50])
plt.xlim([0.1,50])
outdir = os.path.join(RESULTSDIR,'kepler_vs_tess_prot')
if not os.path.exists(outdir):
    os.mkdir(outdir)

outpath = os.path.join(outdir, 'kepler_vs_tess_prot.png')
plt.savefig(outpath, dpi=400)
