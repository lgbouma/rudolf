import os
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from aesthetic.plot import set_style, savefig
from rudolf.helpers import get_flare_df
from rudolf.plotting import plot_flare_pair_time_distribution

#
# SANITY CHECK: what does the interarrival time distirbution look like on the
# actual data?
# the inter-arrival time distribution
#

sdf = get_flare_df()
t_arr = np.array(sdf.tstart)
t_arr_diff = np.diff(np.sort(t_arr))

plt.hist(t_arr_diff, bins=15)
plt.savefig('../results/flares/poisson_comparison_temp.png')

# ok... so guess the rate as the time difference divided by the number of
# events... NB the interarrival time distriubtion for a poisson process is an
# EXOPONENTIAL distribution.
# 100 days total observed... #TODO GET THE EXACT NUMBER, TO GET AN EXACT RATE...
lambda_0 = 100 / len(t_arr)
print(f'λ0: {lambda_0:.3f}')

np.random.seed(3)
samples_0 = np.random.exponential(scale=lambda_0, size=len(t_arr_diff))

plt.close('all')
plt.hist(samples_0, bins=15)
plt.savefig('../results/flares/poisson_comparison_samples0.png')

t0 = min(t_arr)

t_arr_0 = t0 + np.hstack([0, np.cumsum(samples_0)])
assert len(t_arr_0) == len(t_arr)
dists_0 = np.abs(t_arr_0 - t_arr_0[:, None])
uniq_distances_0 = np.unique(np.round(dists_0,3).flatten())

outpath = '../results/flares/flare_pair_separation_histogram_poisson_sampled0.png'

plot_flare_pair_time_distribution(uniq_distances_0, outpath)

# TODO OKAY, NOW MAKE IT BIGGER. REPEAT THIS ACROSS LIKE SAY 1000 RANDOM SEEDS.
# HOW OFTEN DO YOU GET YOUR COINCIDENCE?

# FIXME FIXME FIXME VECTORIZE!

