import numpy as np, pandas as pd
from rudolf.paths import DATADIR, LOCALDIR, RESULTSDIR
import os
from astropy.table import Table

# Ronan Kerr's Cep-Her X [Bedell 1 arcsec list X EDR3]
csvpath = os.path.join(DATADIR, 'Cep-Her', 'kepgaiafun_X_cepher.csv')
df_cep_her = pd.read_csv(csvpath)
df_cep_her['dr2_source_id'] = df_cep_her.dr2_source_id.astype(str)

# Bedell 1 arcsec list (has the KOI columns)
fitspath = '/Users/luke/local/kepler_gws/kepler_dr2_1arcsec.fits'
df_kep = Table.read(fitspath, format='fits').to_pandas()
df_kep['dr2_source_id'] = df_kep['source_id'].astype(str)

mdf = df_cep_her.merge(
    df_kep, how='left', on='dr2_source_id'
)

mdf['planet?'] = mdf['planet?'].str.decode('utf-8')

outdf = mdf[(mdf.nkoi > 0)]

selcols = ['kepid', 'dr2_source_id', 'dr3_source_id', 'weight', 'bmr',
           'TOP LEVEL', 'EOM', 'LEAF',
            'kepmag', 'phot_g_mean_mag',
           'planet?', 'nkoi', 'nconfp', 'kepler_gaia_ang_dist',
           'magnitude_difference', 'angular_distance']

outdir = os.path.join(RESULTSDIR, 'Cep-Her')

outpath = os.path.join(outdir,
                       '20211022_kepgaiafun_X_CepHer_allmatches_cutcolumns.csv')

outdf = mdf[selcols].sort_values(by='weight', ascending=False)
outdf.to_csv(outpath, index=False)
print(f'Made {outpath}')

outpath = os.path.join(outdir,
                       '20211022_kepgaiafun_X_CepHer_KOIs_only_cutcolumns.csv')

outdf = outdf[selcols].sort_values(by='weight', ascending=False)

outdf.to_csv(outpath, index=False)
print(f'Made {outpath}')
