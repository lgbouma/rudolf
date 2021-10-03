import os
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from aesthetic.plot import set_style, savefig
from rudolf.helpers import get_flare_df
from rudolf.plotting import plot_flare_pair_time_distribution

#
# SANITY CHECK: what does the interarrival time distribution look like on the
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
t_shortcadence = 97.667479  # max-min from get_kep1627_kepler_lightcurve
lambda_0 = t_shortcadence / len(t_arr)
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
uniq_distances_0 = np.unique(np.round(dists_0,4).flatten())
print(len(uniq_distances_0)) # should be N choose 2, or 276, for N=24

outpath = '../results/flares/flare_pair_separation_histogram_poisson_sampled0.png'
plot_flare_pair_time_distribution(uniq_distances_0, outpath)


bins = np.logspace(-1, 2, 100)
histograms, uniq_distances = [], []
N_samples = 1000
for i in range(0,N_samples):
    if i % 100 == 0:
        print(f'{i}...')
    np.random.seed(i)
    samples_i = np.random.exponential(scale=lambda_0, size=len(t_arr_diff))

    t_arr_i = t0 + np.hstack([0, np.cumsum(samples_i)])
    assert len(t_arr_i) == len(t_arr)
    dists_i = np.abs(t_arr_i - t_arr_i[:, None])
    uniq_distances_i = np.unique(np.round(dists_i,4).flatten())

    h_i, x_i = np.histogram(uniq_distances_i, bins=bins)
    histograms.append(h_i)

    uniq_distances.append(uniq_distances_i)

outpath = f'../results/flares/flare_pair_separation_histogram_poisson_{N_samples}samples.png'
plot_flare_pair_time_distribution(uniq_distances, outpath)


# plot just the samples
# N_seeds X N_bins
hist_arr = np.array(histograms)
avg_hist = np.mean(hist_arr, axis=0)
std_hist = np.std(hist_arr, axis=0)

outpath = f'../results/flares/flare_pair_separation_histogram_poisson_{N_samples}sample_average.png'
plot_flare_pair_time_distribution(None, outpath, hists=(avg_hist, std_hist, x_i))


# now overplot the real distribution

sdf = get_flare_df()
t_arr = np.sort(np.array(sdf.tstart))
dists = np.abs(t_arr - t_arr[:, None])
uniq_dists = np.unique(np.round(dists,4).flatten())

outpath = f'../results/flares/flare_pair_separation_histogram_poisson_real_and_{N_samples}sample_average.png'
plot_flare_pair_time_distribution(uniq_dists, outpath, hists=(avg_hist, std_hist, x_i))
