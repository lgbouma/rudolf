"""
Due to the on-sky orientation of KOI-7913 A and KOI-7913 B, the default
``optimal aperture'' selected in Q3, Q7, Q11, and Q15 in fact included both
stars, while for the remaining quarters KOI-7913 B was (correctly) excluded
from the optimal aperture (see pages 35 through 71 of the Data Validation
Report.)

We propose a new aperture for these quarters: one with only two pixels,
designed to omit the light from KOI 7913 B which including the light from KOI
7913 A.
"""
import os
from astropy.io import fits
from astropy.time import Time
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from rudolf.paths import DATADIR, RESULTSDIR
from os.path import join
from glob import glob
from itertools import product
from matplotlib import patches
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib.ticker import (
    MultipleLocator, FormatStrFormatter, AutoMinorLocator
)

datadir = join(
    DATADIR, 'KOI_7913', 'phot',
    'MAST_2022-01-25T1331/Kepler/kplr008873450_lc_Q011111111111111111'
)
dataglob = join(
    datadir, 'kplr*-targ.fits.gz'
)

tpf_paths = np.sort(glob(dataglob))
assert len(tpf_paths) == 17

outdir = os.path.join(RESULTSDIR, 'koi7913_aperture_analysis')
if not os.path.exists(outdir):
    os.mkdir(outdir)

for tpf_path in tpf_paths:

    hl = fits.open(tpf_path)

    quarter_num = hl[0].header['QUARTER']

    # sec 2.3.2, Kepler_Archive_Manual_KDMC_100008.pdf :
    # 0: data not collected
    # 1: data collected
    # 3: collected and in optimal aperture
    tpf = hl[1].data
    aperture_mask = hl[2].data

    # the "lowest row, lowest column" count starts at (ccd_col, ccd_row), and
    # ascends.
    ccd_col = hl[1].header['1CRV4P']
    ccd_row = ccd_row = hl[1].header['2CRV4P']

    # plot the aperture polygon
    fig, ax = plt.subplots()

    # extent: For origin == 'lower' the default is (-0.5, numcols-0.5, -0.5, numrows-0.5)
    numrows = aperture_mask.shape[0]
    numcols = aperture_mask.shape[1]

    extent = (-0.5+ccd_col, -0.5+ccd_col+numcols,
              -0.5+ccd_row, -0.5+ccd_row+numrows)

    # show the mean image
    flux = tpf['FLUX']
    mean_flux = np.nanmean(flux, axis=0)
    cset0 = ax.imshow(mean_flux, origin='lower', extent=extent, cmap='viridis')

    # show the aperture mask
    for r,c in product(range(numrows),range(numcols)):
        x,y = c+ccd_col, r+ccd_row

        # FIXME TODO Q3, Q7, Q11, and Q15
        if int(quarter_num) in [3,7,11,15]:
            if ( (x == 439) and (y==234) ) or ( (x == 440) and (y==234) ):
                print(f"Q{quarter_num}: row {r} column {c}, ccdrow {x}, ccdcol {y}")
                ax.add_patch(
                    patches.Rectangle(
                        (x-.5, y-.5), 1, 1, hatch='+', fill=False, snap=False,
                        linewidth=0., zorder=3, alpha=1, rasterized=True,
                        color='red'
                    )
                )

        # missing
        if aperture_mask[r,c] == 0:
            ax.add_patch(
                patches.Rectangle(
                    (x-.5, y-.5), 1, 1, hatch='o', fill=False, snap=False,
                    linewidth=0., zorder=2, alpha=1, rasterized=True,
                    color='black'
                )
            )
        # collected
        if aperture_mask[r,c] == 1:
            ax.add_patch(
                patches.Rectangle(
                    (x-.5, y-.5), 1, 1, hatch='..', fill=False, snap=False,
                    linewidth=0., zorder=2, alpha=1, rasterized=True,
                    color='white'
                )
            )
        # selected aperture
        if aperture_mask[r,c] == 3:
            ax.add_patch(
                patches.Rectangle(
                    (x-.5, y-.5), 1, 1, hatch='+', fill=False, snap=False,
                    linewidth=0., zorder=2, alpha=1, rasterized=True,
                    color='white'
                )
            )

    ##########################################
    from KeplerRaDec2Pix import raDec2Pix
    from KeplerRaDec2Pix.paths import DATADIR
    rdp = raDec2Pix.raDec2PixClass(DATADIR)

    # BJD = BKJD + 2454833
    mean_time_bjd = np.nanmean(tpf['TIME']) + hl[1].header['BJDREFI']
    mean_time = Time(mean_time_bjd, format='jd', scale='tdb')

    # target (KOI-7913 A)
    starids = ['KOI-7913 A', 'KOI-7913 B']
    ras = [hl[0].header['RA_OBJ'], 286.74755600036]
    decs = [hl[0].header['DEC_OBJ'], 45.15801695893]
    colors = ['C0', 'C1']
    markers = ['o', 'X']

    for starid, ra, dec, _c, _m in zip(starids, ras, decs, colors, markers):

        m, o, r, c = rdp.ra_dec_2_pix(ra, dec, mean_time.mjd)
        txt = (
            f'{starid}: quarter {quarter_num}, module {m}, '
            f'output {o}, row {r}, column {c}'
        )
        print(txt)

        x_star, y_star = c, r
        ax.scatter(
            x_star, y_star, c=_c, zorder=100, s=40, rasterized=True,
            linewidths=0.5, marker=_m, edgecolors='k'
        )

    ax.update({
        'xlabel': 'CCD Column',
        'ylabel': 'CCD Row'
    })

    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb = fig.colorbar(cset0, cax=cax, extend='neither')
    cb.ax.tick_params(labelsize='x-small')
    cb.ax.yaxis.set_ticks_position('right')
    cb.ax.yaxis.set_label_position('right')
    cb.set_label('Flux [e$^{-}$/s]', fontsize='x-small')

    qstr = str(quarter_num).zfill(2)

    ax.set_title('Q'+qstr)

    outpath = os.path.join(outdir, f'Q{qstr}_scene_optimal_apertures.png')
    fig.tight_layout()

    plt.savefig(outpath, bbox_inches='tight', dpi=350)
