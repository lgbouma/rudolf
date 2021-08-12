import lightkurve as lk
from rudolf.paths import DATADIR, RESULTSDIR
import os
import pandas as pd, numpy as np
from astropy import units as u, constants as c
from datetime import datetime

csvpath = os.path.join(DATADIR, 'gaia', 'stephenson1_kc19_dr2.csv')

df = pd.read_csv(csvpath)

N_res = []
names = []
for ix, r in df.iterrows():
    n = r['designation']
    print(f'{datetime.utcnow().isoformat()}: {ix}/{len(df)}', n)

    ra = r['ra']*u.deg
    dec = r['dec']*u.deg

    radecstr = f'{ra.value} {dec.value:+}'

    res = lk.search_lightcurve(radecstr, radius=0.1*u.arcsec, mission='Kepler')
    N_res.append(len(res))
    names.append(n)

outdf = pd.DataFrame({
    'source_id': df['source_id'],
    'designation': names,
    'n_kepler_lcs': N_res
})
outpath = os.path.join(RESULTSDIR, 'check_delta_lyra_kepler_lc_count',
                       'result.csv')
outdf.to_csv(outpath, index=False)
import IPython; IPython.embed()
