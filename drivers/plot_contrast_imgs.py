import os, pickle
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from numpy import array as nparr
from aesthetic.plot import savefig, format_ax, set_style
import matplotlib as mpl
from astropy import units as u, constants as c
import matplotlib.patheffects as pe
import matplotlib.image as mpimg
from rudolf.paths import DATADIR, RESULTSDIR

def plot_zorro_speckle():

    datapaths = [
        os.path.join(DATADIR, 'imaging', 'Alopeke_562_deltaM_vs_sep.csv'),
        os.path.join(DATADIR, 'imaging', 'Alopeke_832_deltaM_vs_sep.csv'),
        os.path.join(DATADIR, 'imaging', 'NIRC2_deltaM_vs_sep.csv'),
    ]
    labels = [
        '`Alopeke 0.56$\,\mu$m',
        '`Alopeke 0.83$\,\mu$m',
        'NIRC2 2.12$\,\mu$m',
    ]
    colors = [
        'dodgerblue',
        'tomato',
        'darkred'
    ]

    fig, ax = plt.subplots(figsize=(4,3))
    set_style()

    for d,l,c in zip(datapaths, labels, colors):
        df = pd.read_csv(d)
        #sel = df.sep>0.15
        ax.plot(
            df.sep, df.deltamag, color=c, label=l
        )
        #ax.plot(
        #    df[~sel].sep, df[~sel].deltamag, color=c, ls=':'
        #)

    xf = np.linspace(0,0.15,100)
    ylim = ax.get_ylim()

    #ax.fill_between(xf, min(ylim),
    #                max(ylim), alpha=0.3,
    #                color='gray', lw=0,
    #                label=r'Diffraction limit',zorder=-10)
    ax.set_ylim(ylim[::-1])


    leg = ax.legend(
        loc='upper center', handletextpad=0.2, fontsize='x-small',
        framealpha=0.9, bbox_to_anchor=(0.4,0.9)
    )

    img = mpimg.imread(
        os.path.join(DATADIR, 'imaging',
                     'NIRC2_img_2015stack_rot.png')
    )

    # [left, bottom, width, height]
    inset = fig.add_axes([0.66, 0.63, .3, .3])
    inset.imshow(img)
    inset.axis('off')
    plt.setp(inset, xticks=[], yticks=[])

    ax.set_ylabel('$\Delta$mag')
    ax.set_xlabel('Angular separation [arcsec]')


    ax.set_xlim([0,1.25])

    fig.tight_layout(h_pad=0, w_pad=0)

    figpath = (
        os.path.join(RESULTSDIR, 'imaging', 'imaging_summary.png')
    )
    savefig(fig, figpath)


if __name__ == "__main__":

    plot_zorro_speckle()
