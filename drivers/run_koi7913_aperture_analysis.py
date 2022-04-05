from astropy.io import fits
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from rudolf.paths import DATADIR
from os.path import join
from glob import glob

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

for tpf_path in tpf_paths:

    hl = fits.open(tpf_path)

    quarter_num = hl[0].header['QUARTER']
    print(quarter_num)

    # sec 2.3.2, Kepler_Archive_Manual_KDMC_100008.pdf :
    # 0: data not collected
    # 1: data collected
    # 3: collected and in optimal aperture
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

    # purple: not collected. blue: collected. yellow: optimal aperture.
    ax.imshow(aperture_mask, origin='lower', extent=extent)

    ##########################################
    # TODO: need steve bryson's ra to pixel coordinate converter.
    x_star,y_star = 434.8, 238.3
    ax.scatter(
        x_star, y_star, zorder=100
    )
    ##########################################

    ax.update({
        'xlabel': 'CCD Column',
        'ylabel': 'CCD Row'
    })

    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

    plt.show()

    import IPython; IPython.embed()
    assert 0
