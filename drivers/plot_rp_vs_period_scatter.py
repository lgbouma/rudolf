'''
Make scatter plot of mass versus period. Optionally, color by discovery method.
Optionally, overplot archetype names.
'''

import matplotlib as mpl
import matplotlib.pyplot as plt

import pandas as pd, numpy as np
import os

from astropy import units as u
from cdips.utils import today_YYYYMMDD
from cdips.utils.catalogs import get_nasa_exoplanet_archive_pscomppars
from aesthetic.plot import savefig, format_ax, set_style

VER = '20210702' # could be today_YYYYMMDD()

def arr(x):
    return np.array(x)

def plot_rp_vs_period_scatter(
    showlegend=1, colorbydisc=1, showarchetypes=1, showss=1, colorbyage=0,
    verbose=0, add_kep1627=0, add_allkep=0
):

    set_style()

    #
    # columns described at
    # https://exoplanetarchive.ipac.caltech.edu/docs/API_exoplanet_columns.html
    #
    ea_df = get_nasa_exoplanet_archive_pscomppars(VER)

    #
    # get systems with finite ages (has a value, and +/- error bar)
    #
    has_rp_value = ~pd.isnull(ea_df['pl_rade'])
    has_rp_errs  = (~pd.isnull(ea_df['pl_radeerr1'])) & (~pd.isnull(['pl_radeerr2']))

    has_mp_value = ~pd.isnull(ea_df['pl_bmasse'])
    has_mp_errs = (~pd.isnull(ea_df['pl_bmasseerr1'])) & (~pd.isnull(ea_df['pl_bmasseerr2']))

    rp_gt_0 = (ea_df['pl_rade'] > 0)
    mp_gt_0 = (ea_df['pl_bmassj'] > 0)

    transits = (ea_df['tran_flag']==1)

    has_age_value = ~pd.isnull(ea_df['st_age'])
    has_age_errs  = (~pd.isnull(ea_df['st_ageerr1'])) & (~pd.isnull(ea_df['st_ageerr2']))

    if not colorbyage:
        sel = (
            # has_mp_value & has_mp_errs & mp_gt_0
            has_rp_value & has_rp_errs & transits & rp_gt_0
        )
    else:
        # NOTE: not requiring age uncertainties, b/c TOI451 doesn't quote them.
        sel = (
            has_age_value & has_rp_value & has_rp_errs & transits
            & rp_gt_0
        )
        hardexclude = (
            (ea_df.pl_name == 'CoRoT-18 b') # this is an isochrone age? G9V? wat?
            |
            (ea_df.pl_name == 'Qatar-4 b') # age uncertainties understated. Rotn good. Li is not. isochrones who knows.
        )

        sel &= (~hardexclude)

        has_factorN_age = (
            (ea_df['st_age'] / np.abs(ea_df['st_ageerr1']) > 3)
            &
            (ea_df['st_age'] / np.abs(ea_df['st_ageerr2']) > 3)
        )

        #sel &= has_factorN_age

    sdf = ea_df[sel]

    if verbose:
        sdf_young = (
            sdf[(sdf['st_age']*u.Gyr < 0.3*u.Gyr) & (sdf['st_age']*u.Gyr > 0*u.Gyr)]
        )
        scols = ['pl_name', 'pl_rade', 'pl_orbper', 'st_age', 'st_ageerr1',
                 'st_ageerr2']
        print(42*'-')
        print('Age less than 0.3 Gyr, S/N>3')
        print(sdf_young[scols].sort_values(by='st_age'))
        print(42*'-')
        sdf_young_hj = (
            sdf[(sdf['st_age']*u.Gyr < 0.5*u.Gyr) &
                (sdf['st_age']*u.Gyr > 0*u.Gyr) &
                (sdf['pl_rade'] > 8) &
                (sdf['pl_orbper'] < 10)
               ]
        )
        scols = ['pl_name', 'pl_rade', 'pl_orbper', 'st_age', 'st_ageerr1',
                 'st_ageerr2']
        print('Age less than 0.5 Gyr, S/N>3, Rp>7Re, P<10d')
        print(sdf_young_hj[scols].sort_values(by='st_age'))
        print(42*'-')


    #
    # read params
    #
    mp = sdf['pl_bmasse']
    rp = sdf['pl_rade']
    age = sdf['st_age']*1e9
    period = sdf['pl_orbper']
    discoverymethod = sdf['discoverymethod']

    #
    # plot age vs rp. (age is on y axis b/c it has the error bars, and I at
    # least skimmed the footnotes of Hogg 2010)
    #
    fig,ax = plt.subplots(figsize=(1.3*4,1.3*3))

    if not colorbydisc and not colorbyage:
        ax.scatter(period, rp, color='darkgray', s=3, zorder=1, marker='o',
                   linewidth=0, alpha=1, rasterized=True)

    if colorbyage:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        s0 = (~has_factorN_age) | (age > 1e9)
        s1 = (has_factorN_age) & (age <= 1e9)

        ax.scatter(period[s0], rp[s0],
                   color='darkgray', s=1.5, zorder=1, marker='o', linewidth=0,
                   alpha=1, rasterized=True)

        # draw the colored points
        axins1 = inset_axes(ax, width="3%", height="33%", loc='lower right')

        cmapname = 'viridis'
        cmap = mpl.cm.viridis
        bounds = np.arange(6.9,9.1,0.01)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='neither')

        _p = ax.scatter(
            period[s1], rp[s1],
            c=np.log10(age[s1]), alpha=1, zorder=2, s=10, edgecolors='k',
            marker='o', cmap=cmap, linewidths=0.3, norm=norm
        )

        if add_kep1627:
            ax.scatter(
                7.2028, 3.52,
                c=np.log10(3.5e7), alpha=1, zorder=2, s=90, edgecolors='k',
                marker='*', cmap=cmap, linewidths=0.3, norm=norm,
            )

        if add_allkep:
            # Kepler-52 and Kepler-968
            namelist = ['Kepler-52', 'Kepler-968', 'Kepler-1627']
            ages = [3.5e8, 3.5e8, 3.5e7]

            for n, a in zip(namelist, ages):
                sel = ea_df.hostname == n

                _sdf = ea_df[sel]
                _rp = _sdf.pl_rade
                _per= _sdf.pl_orbper
                _age = np.ones(len(_sdf))*a

                ax.scatter(
                    _per, _rp,
                    c=np.log10(_age), alpha=1, zorder=2, s=90, edgecolors='k',
                    marker='*', cmap=cmap, linewidths=0.3, norm=norm
                )

        cb = fig.colorbar(_p, cax=axins1, orientation="vertical",
                          extend="neither", ticks=[7,7.5,8,8.5,9],
                          norm=norm)
        cb.ax.set_yticklabels([7,7.5,8,8.5,9])
        cb.ax.tick_params(labelsize='x-small')
        cb.ax.tick_params(size=0, which='both') # remove the ticks

        cb.ax.yaxis.set_ticks_position('left')
        cb.ax.yaxis.set_label_position('left')

        cb.set_label("log$_{10}$(age [years])", fontsize='x-small')

    if colorbydisc:
        dmethods = np.unique(discoverymethod)
        dmethods = [
            'Radial Velocity', 'Transit', 'Imaging', 'Microlensing'
        ]
        for ix,dm in enumerate(dmethods):
            print(f'{dm}...')

            sel = discoverymethod == dm
            ax.scatter(period[sel], rp[sel], color=f'C{ix}', s=3, zorder=1,
                       marker='o', linewidth=0.1, label=dm, alpha=1,
                       edgecolors='k', rasterized=True)

    if showarchetypes:
        # P, Mp, Rp
        ARCHETYPEDICT = {
            'hot Jupiters': [3, 3*380, 2*1*11.8],
            'cold Jupiters': [365*1.6, 3*380, 2*1*11.8],
            'super-Earths': [8, 1.5, 1.25],
            'mini-Neptunes': [25, 10, 3]
        }

        for k,v in ARCHETYPEDICT.items():
            txt = k
            _x,_y = v[0], v[2]
            bbox = dict(facecolor='white', alpha=0.95, pad=0, edgecolor='white')
            ax.text(_x, _y, txt, va='center', ha='center',
                    color='k', bbox=bbox,
                    fontsize='xx-small')

    if showss:

        # P[day], Mp[Me], Rp[Re]
        SSDICT = {
            'Jupiter': [4332.8, 317.8, 11.21, 'J'],
            'Saturn': [10755.7, 95.2, 9.151, 'S'],
            'Neptune': [60190.0, 17.15, 3.865, 'N'],
            'Uranus': [30687, 14.54, 3.981, 'U'],
            'Earth': [365.3, 1, 1, 'E'],
            'Venus': [224.7, 0.815, 0.950, 'V'],
            'Mars': [687.0, 0.1074, 0.532, 'M'],
        }

        for k,v in SSDICT.items():
            txt = k
            _x,_y = v[0], v[2]
            _m = v[3]

            ax.scatter(_x, _y, color='k', s=9, zorder=1000,
                       marker='$\mathrm{'+_m+'}$',
                       linewidth=0, alpha=1)
            ax.scatter(_x, _y, color='white', s=22, zorder=999,
                       marker='o', edgecolors='k',
                       linewidth=0.2, alpha=1)

            #bbox = dict(facecolor='white', alpha=0., pad=0, edgecolor='white')
            #ax.text(_x, _y, txt, va='top', ha='center',
            #        color='k', bbox=bbox,
            #        fontsize='xx-small')


    # flip default legend order
    if showlegend:
        handles, labels = ax.get_legend_handles_labels()
        leg = ax.legend(handles, labels, loc='lower right',
                        borderpad=0.2, handletextpad=0.1, fontsize='xx-small',
                        framealpha=1)

        leg.get_frame().set_linewidth(0.5)

    ax.set_xlabel('Orbital period [days]')
    ax.set_ylabel('Planet radius [Earths]')

    ax.set_xlim([0.1, 110000])
    #ax.set_ylim([0.0001*318, 100*318])
    #ax.set_ylim([0.0001*318, 100*318])
    format_ax(ax)

    ax.set_yscale('log')

    ax.set_xscale('log')

    s = ''
    if showlegend:
        s += '_yeslegend'
    if colorbydisc:
        s += '_colorbydisc'
    if showss:
        s += '_showss'
    if showarchetypes:
        s += '_showarchetypes'
    if colorbyage:
        s += '_colorbyage'
    if add_kep1627:
        s += '_showkep1627'
    if add_allkep:
        s += '_showallkep'

    outdir = '../results/rp_vs_period_scatter/'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    outpath = (
        os.path.join(
            outdir, f'rp_vs_period_scatter_{VER}{s}.png'
        )
    )

    savefig(fig, outpath, writepdf=1, dpi=400)


if __name__=='__main__':

    plot_rp_vs_period_scatter(
        showlegend=0, colorbydisc=0, showarchetypes=0, showss=1, colorbyage=1,
        verbose=1, add_kep1627=1
    )

    plot_rp_vs_period_scatter(
        showlegend=0, colorbydisc=0, showarchetypes=0, showss=1, colorbyage=1,
        verbose=0, add_kep1627=0, add_allkep=1
    )

    plot_rp_vs_period_scatter(
        showlegend=0, colorbydisc=0, showarchetypes=0, showss=1, colorbyage=1,
        verbose=0
    )
