import pandas as pd
import os
from rudolf.helpers import (
    get_autorotation_dataframe, get_clean_gaia_photometric_sources
)

from rudolf.paths import DATADIR, RESULTSDIR

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

print('How many stars are there?')
print(f'N_KC19_deltaLyrCluster: {len(base_df)}')

print('How many stars have TESS data?')
N_TESS = len(base_df[base_df.TESS_Data == 'Yes'])
print(f'N_with_TESS: {N_TESS}')

print('How many stars have finite rotation periods?')
N_finite_Prot = len(base_df[base_df.period > 0])
print(f'N_finite_Prot: {N_finite_Prot}')

print('How many kinematically selected stars have TESS data?')
N_kin_with_tess_data = len(mdf[mdf['TESS_Data_x']=='Yes'])
print(f'N_kin_with_tess_data: {N_kin_with_tess_data}')

print('How many kinematically selected stars have TESS data and meet the crowding cutoff? (nequal<=0, nclose<=1)')
N_kin_with_tess_data_and_crowding = len(mdf[
    (mdf['TESS_Data_x']=='Yes')
    &
    (mdf['nequal']<=0)
    &
    (mdf['nclose']<=1)
])
print(f'N_kin_with_tess_data_and_crowding: {N_kin_with_tess_data_and_crowding}')

print('How many of these meet the "defaultcleaning" criteria (LSP>0.2, Prot<15, and crowding cut)?')
N_kin_defaultcleaning = len(mdf1)
print(f'N_kin_defaultcleaning: {N_kin_defaultcleaning}')


print('How many of the "defaultcleaning" stars fall below slow sequence?')
N_outliers = 10 # TODO FIXME MANUAL NUMBEr
N_rot_good = len(mdf1) - N_outliers
print(f'N_rot_good: {N_rot_good}')

print(f'frac: {N_rot_good/N_kin_with_tess_data_and_crowding}')
