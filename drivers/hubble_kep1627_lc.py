import numpy as np, pandas as pd, matplotlib.pyplot as plt
from aesthetic.plot import set_style, savefig

import os
from glob import glob

from rudolf.helpers import get_manually_downloaded_kepler_lightcurve

t, f, ferr, q, texp = get_manually_downloaded_kepler_lightcurve(lctype='longcadence')

n = lambda x: 1e2 * ( x - np.nanmean(x) )
nx = lambda x: x - np.nanmin(x)

set_style('science')
fa = 0.7
fig, ax = plt.subplots(figsize=(fa*4.,fa*1))
ax.scatter(nx(t), n(f), c='k', linewidths=0, s=fa*0.5)
ax.set_xlim([500, 600])
ax.set_ylim([-5.5, 5.5])
ax.set_xticklabels([])
ax.set_yticklabels([])
#ax.set_xlabel('Time [days]')
#ax.set_ylabel('Δ Flux [%]')

outpath = '../results/hubble_kep1627/kep1627lc.pdf'
savefig(fig, outpath)
