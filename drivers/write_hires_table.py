import os
from numpy import array as nparr
import numpy as np, pandas as pd
from rudolf.paths import RESULTSDIR, PAPERDIR, DATADIR

rvpath = os.path.join(DATADIR, 'spec', '20210809_rvs_template_V1298TAU.csv')
df = pd.read_csv(rvpath)

outdf = pd.DataFrame({
    'BJDTDB': np.round(nparr(df['bjd']), 6),
    'RV': np.round(nparr(df['rv']), 2),
    'E_RV': np.round(nparr(df['e_rv']), 2),
    'S_VALUE': np.round(nparr(df['svalue']), 4)
})

outpath = os.path.join(PAPERDIR, 'hires_rv_table.csv')

outdf.to_csv(outpath, index=False)
print(f"Wrote {outpath}")
