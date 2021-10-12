import os
from numpy import array as nparr
import numpy as np, pandas as pd
from rudolf.paths import RESULTSDIR, PAPERDIR, DATADIR

df = pd.read_csv("/Users/luke/local/cdips/catalogs/cdips_targets_v0.6_nomagcut_gaiasources.csv")

outdf = pd.DataFrame({
    'dr2_source_id': df.source_id.astype(str),
    'dr2_ra': np.round(nparr(df['ra']), 5),
    'dr2_dec': np.round(nparr(df['dec']), 5),
    'dr2_parallax': np.round(nparr(df['parallax']), 4),
    'dr2_parallax_error': np.round(nparr(df['parallax_error']), 4),
    'dr2_pmra': np.round(nparr(df['pmra']), 5),
    'dr2_pmdec': np.round(nparr(df['pmdec']), 5),
    'dr2_phot_g_mean_mag': np.round(nparr(df['phot_g_mean_mag']), 3),
    'dr2_phot_rp_mean_mag': np.round(nparr(df['phot_rp_mean_mag']), 3),
    'dr2_phot_bp_mean_mag': np.round(nparr(df['phot_bp_mean_mag']), 3),
    'cluster': df.cluster.astype(str),
    'age': df.age.astype(str),
    'mean_age': df.mean_age.astype(str),
    'reference_id': df.reference_id.astype(str),
    'reference_bibcode': df.reference_bibcode.astype(str),
})

outdf = outdf.sort_values(by=['dr2_parallax','dr2_ra'], ascending=False)

outpath = os.path.join(PAPERDIR, 'cdips_targets_v0.6_nomagcut_gaiasources_table.csv')

outdf.to_csv(outpath, index=False)
print(f"Wrote {outpath}")
