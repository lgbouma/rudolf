import os
from numpy import array as nparr
import numpy as np, pandas as pd
from rudolf.paths import RESULTSDIR, PAPERDIR
from rudolf.helpers import get_kep1627_muscat_lightcuve

datasets = get_kep1627_muscat_lightcuve()

bps = 'g,r,i,z'.split(',')

dfs = [
    pd.DataFrame({
        'BJDTDB': datasets[f"muscat3_{bp}"][0],
        'FLUX': datasets[f"muscat3_{bp}"][1],
        'E_FLUX': datasets[f"muscat3_{bp}"][2],
        'BANDPASS': str(bp)
    }) for bp in bps
]

df = pd.concat(dfs)

outpath = os.path.join(PAPERDIR, 'muscat_phot_table.csv')

df.to_csv(outpath, index=False)
print(f"Wrote {outpath}")
