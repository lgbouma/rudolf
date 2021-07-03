import os
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from aesthetic.plot import set_style, savefig
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

# get pairwise distances...
t_arr = np.array(sdf.tstart)
dists = np.abs(t_arr - t_arr[:, None])

# remove duplicates
uniq_dists = np.unique(np.round(dists,3).flatten())

plt.close('all')
set_style()
fig, ax = plt.subplots(figsize=(4,3))
bins = np.logspace(-1, 2, 100)
ax.hist(uniq_dists, bins=bins, cumulative=False, color='k',
        fill=False, histtype='step', linewidth=0.5)

P_orb = 7.2028041 # pm 0.0000074 
P_rot = 2.642 # pm 0.042
P_syn = (1/P_rot - 1/P_orb)**(-1) # ~=4.172 day

ylim = ax.get_ylim()
ax.vlines(P_orb, min(ylim), max(ylim), ls='--', lw=0.5, colors='C0',
          zorder=-1, label='P$_\mathrm{orb}$'+f' ({P_orb:.3f} d)')

#ax.vlines(2*P_orb, min(ylim), max(ylim), ls='--', lw=0.5, colors='C0',
#          zorder=-1, label=r'2$\times$P$_\mathrm{orb}$')
ax.vlines(P_syn, min(ylim), max(ylim), ls='--', lw=0.5, colors='C1',
          zorder=-1, label='P$_\mathrm{syn}$'+f' ({P_syn:.3f} d)')
ax.vlines(2*P_syn, min(ylim), max(ylim), ls='--', lw=0.5, colors='C1',
          zorder=-1, label=r'2$\times$P$_\mathrm{syn}$')

ax.vlines(P_orb+P_syn, min(ylim), max(ylim), ls='--', lw=0.5,
          colors='C2',
          zorder=-1, label='P$_\mathrm{orb}$+P$_\mathrm{syn}$')
ax.vlines(P_orb+2*P_syn, min(ylim), max(ylim), ls='--', lw=0.5,
          colors='C2',
          zorder=-1, label=r'P$_\mathrm{orb}$+2$\times$P$_\mathrm{syn}$')

#ax.vlines(3*P_syn, min(ylim), max(ylim), ls='--', lw=0.5, colors='C1',
#          zorder=-1, label=r'3$\times$P$_\mathrm{syn}$')
ax.set_ylim(ylim)

ax.legend(loc='upper left', fontsize='x-small')
ax.set_xscale('log')

ax.set_xlabel('Flare pair separation [days]')
ax.set_ylabel('Count')
outpath = '../results/flares/flare_pair_separation_histogram.png'
savefig(fig, outpath, dpi=400)
