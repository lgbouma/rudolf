"""
Write the RSG-5 and CH-2 membership + rotation period table.
"""
import os
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from rudolf.paths import DATADIR, RESULTSDIR, TABLEDIR, LOCALDIR
from rudolf.helpers import get_ronan_cepher_augmented

# __df: after Ronan's flags
# df: after Ronan's + my flags

df, __df = get_ronan_cepher_augmented()
fig1_grey = (df.strengths > 0.02)
fig1_black = (df.strengths > 0.10)
sdf = df[fig1_grey]

tabpath = os.path.join(TABLEDIR, "RSG-5_auto_withreddening_gaia2018.csv")
_df1 = pd.read_csv(tabpath)
tabpath = os.path.join(TABLEDIR, "CH-2_auto_withreddening_gaia2018.csv")
_df2 = pd.read_csv(tabpath)
df1 = pd.concat((_df1,_df2))


csvpath = os.path.join(DATADIR, "rotation",
                       "CepHer_RSG5_EDR3_v2_Curtis_20220326.csv")
rot_df = pd.read_csv(csvpath)

rot_df['Source_Input'] = rot_df.Source_Input.astype(str)
df1['source_id'] = df1.source_id.astype(str)

mdf = rot_df.merge(df1, left_on='Source_Input', right_on='source_id',
                   suffixes=('_rot', '_EDR3'))

mdf = mdf.rename({'source_id_EDR3':'dr3_source_id'}, axis='columns')

reddening_corr = 1
cstr = '_corr' if reddening_corr else ''
color0 = 'phot_bp_mean_mag'
get_yval = (
    lambda _df: np.array(
        _df['phot_g_mean_mag'+cstr] + 5*np.log10(_df['parallax_EDR3']/1e3) + 5
    )
)
get_xval = (
    lambda _df: np.array(
        _df[color0+cstr] - _df['phot_rp_mean_mag'+cstr]
    )
)

mdf['(BP-RP)0'] = get_xval(mdf)
mdf['(M_G)0'] = get_yval(mdf)

selcols = (
    "dr3_source_id,ra_EDR3,dec_EDR3,parallax_EDR3,ruwe_EDR3,"
    "strengths,v_l*,v_b,x_pc,y_pc,z_pc,(BP-RP)0,(M_G)0,"
    "Cluster,period,Prot_TESS,Prot_ZTF,Prot_Confused"
)

smdf = mdf[selcols.split(',')]

rename_dict = {
    "strengths":"weight",
    "v_l*":"v_l",
    "Cluster":"cluster",
    "period":"Prot_Adopted"
}
smdf = smdf.rename(rename_dict, axis='columns')

round_dict = {
    'ra_EDR3': 4,
    'dec_EDR3': 4,
    'parallax_EDR3': 4,
    'ruwe_EDR3': 3,
    'weight': 5,
    'v_l': 4,
    'v_b': 4,
    'x_pc': 2,
    'y_pc': 2,
    'z_pc': 2,
    '(M_G)0': 3,
    '(BP-RP)0': 3,
    'Prot_Adopted': 3,
    'Prot_TESS': 3,
    'Prot_ZTF': 3,
}
smdf = smdf.round(round_dict)

smdf.loc[smdf.cluster=='CepHer', 'cluster'] = 'CH-2'
smdf.loc[smdf.cluster=='RSG5', 'cluster'] = 'RSG-5'

smdf = smdf.sort_values(by=['cluster','(BP-RP)0'])

outpath = os.path.join(TABLEDIR, 'RSG5_CH2_Prot_cleaned.csv')
smdf.to_csv(outpath, index=False)
print(f'Wrote {outpath}')

assert len(smdf) == 141+37
import IPython; IPython.embed()
