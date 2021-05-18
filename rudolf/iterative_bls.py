import os
import numpy as np, pandas as pd
from astropy.timeseries import BoxLeastSquares
from scipy.signal import savgol_filter
import matplotlib as mpl
import matplotlib.pyplot as plt
from aesthetic.plot import savefig
from copy import deepcopy

from astrobase.lcmath import sigclip_magseries

def run_iterative_bls(x, y, outdir, pmin=0.5, pmax=100,
                      y_alreadysmoothed=False):
    """
    y: if y_alreadysmoothed is False, then this is a detrended flux vector.
    Otherwise, it's the original flux vector, and it gets Savitzky-Golay
    smoothed.
    """

    mpl.rcParams['agg.path.chunksize'] = 10000

    period_grid = np.exp(np.linspace(np.log(pmin), np.log(pmax), 50000))

    bls_results = []
    periods = []
    t0s = []
    depths = []

    if not y_alreadysmoothed:
        _filter = savgol_filter(y, 51, polyorder=3)
        y_search = y - _filter
    else:
        y_search = y

    y_search -= np.nanmedian(y_search)
    orig_x = deepcopy(x)

    x, y_search, _ = sigclip_magseries(
        x, y_search, y_search*1e-4, magsarefluxes=True, sigclip=[999, 3.]
    )

    # verify the data being searched are correct
    plt.close('all')
    f,axs = plt.subplots(nrows=2, figsize=(20,4), sharex=True)
    axs[0].scatter(orig_x, y, c='k', s=0.1)
    axs[1].scatter(x, y_search, c='k', s=0.1)
    s = 'savgol' if not y_alreadysmoothed else 'gptransit'
    outpath = os.path.join(outdir, f'verify_{s}.png')
    savefig(f, outpath, dpi=400)
    plt.close('all')

    # Compute the periodogram for each planet by iteratively masking out
    # transits from the higher signal to noise planets. Here we're assuming
    # that we know that there are exactly two planets.
    m = np.zeros(len(x), dtype=bool)
    for i in range(5):
        bls = BoxLeastSquares(x[~m], y_search[~m])
        bls_power = bls.power(period_grid, 0.1, oversample=20)
        bls_results.append(bls_power)

        # Save the highest peak as the planet candidate
        index = np.argmax(bls_power.power)
        periods.append(bls_power.period[index])
        t0s.append(bls_power.transit_time[index])
        depths.append(bls_power.depth[index])

        # Mask the data points that are in transit for this candidate
        m |= bls.transit_mask(x, periods[-1], 0.5, t0s[-1])


    fig, axes = plt.subplots(len(bls_results), 2, figsize=(10, 10))

    for i in range(len(bls_results)):
        # Plot the periodogram
        ax = axes[i, 0]
        ax.axvline(np.log10(periods[i]), color="C1", lw=5, alpha=0.8)
        ax.plot(np.log10(bls_results[i].period), bls_results[i].power, "k")
        ax.annotate(
            "period = {0:.4f} d".format(periods[i]),
            (0, 1),
            xycoords="axes fraction",
            xytext=(5, -5),
            textcoords="offset points",
            va="top",
            ha="left",
            fontsize=12,
        )
        ax.set_ylabel("bls power")
        #ax.set_yticks([])
        ax.set_xlim(np.log10(period_grid.min()), np.log10(period_grid.max()))
        if i < len(bls_results) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("log10(period)")

        # Plot the folded transit
        ax = axes[i, 1]
        p = periods[i]
        x_fold = (x - t0s[i] + 0.5 * p) % p - 0.5 * p
        m = np.abs(x_fold) < 0.4
        ax.plot(x_fold[m], y_search[m], ".k")

        # Overplot the phase binned light curve
        bins = np.linspace(-0.41, 0.41, 32)
        denom, _ = np.histogram(x_fold, bins)
        num, _ = np.histogram(x_fold, bins, weights=y_search)
        denom[num == 0] = 1.0
        ax.plot(0.5 * (bins[1:] + bins[:-1]), num / denom, color="C1")

        ax.set_xlim(-0.4, 0.4)
        ax.set_ylabel("relative flux [ppt]")
        if i < len(bls_results) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("time since transit")

    _ = fig.subplots_adjust(hspace=0.02)

    s = ''
    if y_alreadysmoothed:
        s += '_on_gptransit_smoothed'
    else:
        s += '_on_savgol_smoothed'

    outpath = os.path.join(outdir, f'iterative_bls{s}_{pmin}_{pmax}.png')
    savefig(fig, outpath, dpi=400)
