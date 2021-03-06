"""
Write the table that cross-matches plausible Cep-Her stars against stars
observed by Kepler.

These are the candidates with weights > 0.02.  There are 338 objects in the
match.  (The planetary detection fraction of ~1% is about what one would
expect!) Cross-matching against, say, Santos et al 2021, (which we note uses a
different color selection function!) one can see plausible rotation periods are
most often detected when the weights exceed ~=0.05.
"""
import os
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from rudolf.paths import DATADIR, RESULTSDIR, TABLEDIR, LOCALDIR
from rudolf.helpers import get_ronan_cepher_augmented

# __df: after Ronan's flags
# df: after Ronan's + my flags

df, __df, _ = get_ronan_cepher_augmented()
fig1_grey = (df.strengths > 0.02)
fig1_black = (df.strengths > 0.10)
sdf = df[fig1_grey]

# crossmatch with Kepler objects for "publishable Cep-Her subset" 
fitspath = os.path.join(LOCALDIR, "kepler_dr2_1arcsec.fits")
hdulist = fits.open(fitspath)
kdf = Table(hdulist[1].data).to_pandas()
hdulist.close()

# Get DR2 source_ids for all of Ronan's objects, since this is the
# relevant key for Bedell's lists.  Take the closest (proper motion
# and epoch-corrected) angular distance as THE single match.
from cdips.utils.gaiaqueries import given_dr3_sourceids_get_dr2_xmatch
dr3_source_ids = np.array(sdf.source_id.astype(str)).astype(np.int64)
dr2_x_edr3_df = given_dr3_sourceids_get_dr2_xmatch(
    dr3_source_ids, 'CepHer_weight_gt0pt02', overwrite=False
)
get_dr2_xm = lambda _df: (
	_df.sort_values(by='angular_distance').
	drop_duplicates(subset='dr3_source_id', keep='first')
)
s_dr2 = get_dr2_xm(dr2_x_edr3_df)
dr2_source_ids = np.array(s_dr2.dr2_source_id).astype(np.int64)

s_dr2['dr2_source_id'] = s_dr2.dr2_source_id.astype(str)
s_dr2['dr3_source_id'] = s_dr2.dr3_source_id.astype(str)
sdf = sdf.rename({'source_id':'dr3_source_id'}, axis='columns')
sdf['dr3_source_id'] = sdf.dr3_source_id.astype(str)
kdf['source_id'] = kdf.source_id.astype(str)

sdf = sdf.merge(s_dr2[['dr2_source_id', 'dr3_source_id', 'magnitude_difference']], how='inner')

# Merge Kepler (DR2; Bedell) and Cep-Her source lists

mdf = sdf.merge(
    kdf, left_on='dr2_source_id', right_on='source_id', how='inner',
    suffixes=('_EDR3', '_KIC_GDR2')
)

selcols = (
    "dr2_source_id,dr3_source_id,kepid,ra_EDR3,dec_EDR3,"
    "strengths,v_l*,v_b,x_pc,y_pc,z_pc,"
    "kepler_gaia_ang_dist,magnitude_difference"
)

smdf = mdf[selcols.split(',')]

rename_dict = {
    "strengths":"weight",
    "v_l*":"v_l",
    "kepler_gaia_ang_dist":"kic_dr2_ang_dist",
    "magnitude_difference":"edr3_dr2_mag_diff"
}
smdf = smdf.rename(rename_dict, axis='columns')

round_dict = {
    'ra_EDR3': 5,
    'dec_EDR3': 5,
    'l_EDR3': 4,
    'b_EDR3': 4,
    'parallax_EDR3': 4,
    'ruwe_EDR3': 3,
    'weight': 3,
    'v_l': 2,
    'v_b': 2,
    'x_pc': 1,
    'y_pc': 1,
    'z_pc': 1,
    'M_G': 3,
    'bp_rp_EDR3': 3,
    'kic_dr2_ang_dist': 3,
    'edr3_dr2_mag_diff': 3
}
smdf = smdf.round(round_dict)

outpath = os.path.join(TABLEDIR, 'CepHer_X_Kepler.csv')
smdf.to_csv(outpath, index=False)
print(f'Wrote {outpath}')

# Do the obvious match against Santos+21

# get data
csvpath = os.path.join(
    DATADIR, "rotation", "Santos_2021_apjsac033ft1_mrt.txt"
)
t = Table.read(csvpath, format='cds')
s21_df = t.to_pandas()

s21_df['KIC'] = s21_df.KIC.astype(str)
smdf['kepid'] = smdf.kepid.astype(str)
mmdf = smdf.merge(s21_df, left_on='kepid', right_on='KIC', how='left')

outpath = os.path.join(TABLEDIR, 'CepHer_X_Kepler_X_Santos21.csv')
mmdf.to_csv(outpath, index=False)
print(f'Wrote {outpath}')

# glue exploration suggests 0.05 to 0.10 is a fine cut.
