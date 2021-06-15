import os
import pandas as pd, numpy as np
from rudolf.paths import RESULTSDIR
from rudolf.plotting import _get_detrended_flare_data

flaredir = os.path.join(RESULTSDIR, 'flares')

# read data
method = 'itergp'
cachepath = os.path.join(flaredir, f'flare_checker_cache_{method}.pkl')
c = _get_detrended_flare_data(cachepath, method)
flpath = os.path.join(flaredir, f'fldict_{method}.csv')
df = pd.read_csv(flpath)

FL_AMP_CUTOFF = 5e-3
sel = df.ampl_rec > FL_AMP_CUTOFF
sdf = df[sel]

tot_dur = np.sum(sdf.dur)

N_flares = len(sdf)
N_flares_with_successor = len(sdf[sdf.has_Porb_successor])
N_flares_with_candsuccessor = len(sdf[sdf.has_Porb_candsuccessor])

t0,t1 = min(c.time), max(c.time)
tobs = t1-t0

print(f'In {tobs:.1f} days observed...')
print(f'Saw {N_flares} flares > {FL_AMP_CUTOFF:.2e}, spanning total of {tot_dur:.3f} days.')
print(f'->Duty cycle: {100*tot_dur/tobs:.3f}% of time there is a flare above this amplitude.')
print(f'Of these, {N_flares_with_successor} with Porb successor')
print(f'... and {N_flares_with_candsuccessor} with Porb cand successor')
