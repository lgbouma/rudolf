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

df, __df, _ = get_ronan_cepher_augmented()
fig1_grey = (df.strengths > 0.02)
fig1_black = (df.strengths > 0.10)
sdf = df[fig1_grey]

tabpath = os.path.join(TABLEDIR, "RSG-5_auto_withreddening_gaia2018.csv")
_df1 = pd.read_csv(tabpath)
tabpath = os.path.join(TABLEDIR, "CH-2_auto_withreddening_gaia2018.csv")
_df2 = pd.read_csv(tabpath)
df1 = pd.concat((_df1,_df2))


# the default 37 CH-2 and 141 RSG-5
csvpath = os.path.join(DATADIR, "rotation",
                       "CepHer_RSG5_EDR3_v2_Curtis_20220326.csv")
rot_df = pd.read_csv(csvpath)
# 32 new RSG-5 objects
csvpath = os.path.join(DATADIR, "rotation",
                       "20220424_Curtis_RSG5-new-Prot_Results.csv")
rot_df1 = pd.read_csv(csvpath)
csvpath = os.path.join(RESULTSDIR, "CepHer_XYZvtang_sky",
                       "new_RSG-5_auto_XYZ_vl_vb_cut_20220422.csv")
_rdf1 = pd.read_csv(csvpath)

# note: formally incorrect.  however we have 32 sources, and they all happen to
# have identifical EDR3 to DR2 source_id matches.
assert np.all(rot_df1.DR2Name.isin(_rdf1.source_id))
rot_df1 = rot_df1.merge(_rdf1, left_on='DR2Name', right_on='source_id',
                        how='inner')

rot_df['Source_Input'] = rot_df.Source_Input.astype(str)
rot_df1['DR2Name'] = rot_df1['DR2Name'].astype(str)
rot_df1['Cluster'] = 'RSG-5'
df1['source_id'] = df1.source_id.astype(str)

mdf = rot_df.merge(df1, left_on='Source_Input', right_on='source_id',
                   suffixes=('_rot', '_EDR3'), how='inner')
mdf1 = rot_df1.merge(df1, left_on='DR2Name', right_on='source_id',
                   suffixes=('_rot', '_EDR3'), how='inner')

mdf = mdf.rename({'source_id_EDR3':'dr3_source_id'}, axis='columns')
mdf1 = mdf1.rename({'source_id_EDR3':'dr3_source_id', 'parallax':'parallax_EDR3',
                    'ra':'ra_EDR3', 'dec':'dec_EDR3', 'ruwe':'ruwe_EDR3',
                    'TESS_Prot':'Prot_TESS', 'ZTF_Prot':'Prot_ZTF'}, axis='columns')

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
mdf1['(BP-RP)0'] = get_xval(mdf1)
mdf1['(M_G)0'] = get_yval(mdf1)
mdf1['Prot_Confused'] = 0

selcols = (
    "dr3_source_id,ra_EDR3,dec_EDR3,parallax_EDR3,ruwe_EDR3,"
    "strengths,v_l*,v_b,x_pc,y_pc,z_pc,(BP-RP)0,(M_G)0,"
    "Cluster,period,Prot_TESS,Prot_ZTF,Prot_Confused"
)

smdf = mdf[selcols.split(',')]
smdf1 = mdf1[selcols.split(',')]
smdf = pd.concat((smdf, smdf1))

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
smdf.loc[smdf.Prot_TESS==-1, 'Prot_TESS'] = np.nan
smdf.loc[smdf.Prot_ZTF==-1, 'Prot_ZTF'] = np.nan

smdf = smdf.sort_values(by=['cluster','(BP-RP)0'])

outpath = os.path.join(TABLEDIR, 'RSG5_CH2_Prot_cleaned.csv')
smdf.to_csv(outpath, index=False)
print(f'Wrote {outpath}')

assert len(smdf) == 141+37+32
import IPython; IPython.embed()
