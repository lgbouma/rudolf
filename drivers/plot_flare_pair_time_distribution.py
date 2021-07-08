import os
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from aesthetic.plot import set_style, savefig
from rudolf.helpers import get_flare_df
from rudolf.plotting import plot_flare_pair_time_distribution

sdf = get_flare_df()

# get pairwise distances, and remove duplicates (the pairwise distance matrix
# is symmetric). np sorting is optional.
t_arr = np.sort(np.array(sdf.tstart))
dists = np.abs(t_arr - t_arr[:, None])
uniq_dists = np.unique(np.round(dists,4).flatten())

outpath = '../results/flares/flare_pair_separation_histogram.png'

plot_flare_pair_time_distribution(uniq_dists, outpath)
