"""
Write the header for the v05 CDIPS target list.
"""
import os
from numpy import array as nparr
import numpy as np, pandas as pd

from cdips.utils.catalogs import get_cdips_pub_catalog
from rudolf.paths import RESULTSDIR

df = get_cdips_pub_catalog(ver=0.6)

#
# make the header table for the paper
#
namedesc_dict = {
'source_id': "Gaia DR2 source identifier.",
'ra': "Gaia DR2 right ascension [deg].",
'dec': "Gaia DR2 declination [deg].",
'parallax': "Gaia DR2 parallax [mas].",
'parallax_error': "Gaia DR2 parallax uncertainty [mas].",
'pmra': r"Gaia DR2 proper motion $\mu_\alpha \cos \delta$ [mas$\,$yr$^{-1}$].",
'pmdec': "Gaia DR2 proper motion $\mu_\delta$ [mas$\,$yr$^{-1}$].",
'phot_g_mean_mag': "Gaia DR2 $G$ magnitude.",
'phot_bp_mean_mag': "Gaia DR2 $G_\mathrm{BP}$ magnitude.",
'phot_rp_mean_mag': "Gaia DR2 $G_\mathrm{RP}$ magnitude.",
'cluster': "Comma-separated cluster or group name.",
'age': r"Comma-separated logarithm (base-10) of reported$^{\rm a}$ age in years.",
'mean_age': r"Mean (ignoring NaNs) of $\texttt{age}$ column.",
'reference_id': "Comma-separted provenance of group membership.",
'reference_bibcode': r"ADS bibcode corresponding to $\texttt{reference\_id}$.",
}

keys = list(namedesc_dict.keys())
keys = ["\\texttt{"+k.replace("_", "\_")+"}" for k in keys]

#sel = df.source_id == np.int64(3311804515502788352)
sel = df.source_id == np.int64(1709456705329541504)
_vals = df[sel].T.values.flatten()
vals = []
for v in _vals:
    if isinstance(v, float):
        v = np.round(v,3)
    elif isinstance(v, str):
        v = str(v).replace("_", "\_").replace("%","\%")
    else:
        v = v
    vals.append(v)

descrs = list(namedesc_dict.values())

df_totex = pd.DataFrame({
    'Parameter': keys,
    'Example Value': vals,
    'Description': descrs
})

outpath = os.path.join(RESULTSDIR, 'tables', 'v06_main_tableheader.tex')

pd.set_option('display.max_colwidth',100)
# escape=False fixes "textbackslash"
df_totex.to_latex(outpath, index=False, escape=False)
print(f'Wrote {outpath}')
