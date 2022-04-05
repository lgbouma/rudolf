"""
Make scatter plot of mass versus period. Optionally, color by discovery method.
Optionally, overplot archetype names.
"""

# Standard imports
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd, numpy as np
import os
from astropy import units as u

# If you want to run the code, you'll need to do:
# `git clone https://github.com/lgbouma/cdips; cd cdips; python setup.py install`
# `git clone https://github.com/lgbouma/aesthetic; cd aesthetic; python setup.py install`
from cdips.utils import today_YYYYMMDD
from cdips.utils.catalogs import get_nasa_exoplanet_archive_pscomppars
from aesthetic.plot import savefig, format_ax, set_style

# This "VER" string caches the NASA exoplanet archive `ps` table at a
# particular date, in the YYYYMMDD format.
VER = '20210915' # kepler-1627 paper version
VER = '20220107' # could be today_YYYYMMDD()
VER = '20220227'
VER = '20220405'

def plot_rp_vs_period_scatter(
    showlegend=1, colorbydisc=1, showarchetypes=1, showss=1, colorbyage=0,
    verbose=0, add_kep1627=0, add_allkep=0, add_CepHer=0, add_plnames=0
):
    """
    Plot planetary parameters versus ages. By default, it writes the plots to
    '../results/rp_vs_period_scatter/' from wherever you put this script.
    (See `outdir` parameter below).

    Options (all boolean):

        showlegend: whether to overplot a legend.

        colorbydisc: whether to color by the discovery method.

        showarchetypes: whether to show "Hot Jupiter", "Cold Jupiter" etc
        labels for talks.

        showss: whether to show the solar system planets.

        colorbyage: whether to color the points by their ages.

        verbose: if True, prints out more information about the youngest
        planets from the NASA exoplanet archive.

        add_kep1627: adds a special star for Kepler 1627.

        add_allkep: adds special symbols for the recent Kepler systems in open
        clusters: 'Kepler-52', 'Kepler-968', 'Kepler-1627', 'KOI-7368'

        add_plnames: if True, shows tiny texts for the age-dated planets.
    """

    set_style()

    #
    # Columns are described at
    # https://exoplanetarchive.ipac.caltech.edu/docs/API_exoplanet_columns.html
    #
    ea_df = get_nasa_exoplanet_archive_pscomppars(VER)

    #
    # In most cases, we will select systems with finite ages (has a value, and
    # +/- error bar). We may also want to select on "is transiting", "has
    # mass", etc.
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
            #has_rp_value & has_rp_errs & transits & rp_gt_0
            has_age_value & has_rp_value & has_rp_errs & transits & rp_gt_0
        )

    else:

        # We will be coloring the points by their ages.

        sel = (
            has_age_value & has_rp_value & has_rp_errs & transits & rp_gt_0
        )

        # Do not show these planets.
        HARDEXCLUDE = (
            (ea_df.pl_name == 'CoRoT-18 b') # this is an isochrone age? G9V? wat?
            |
            (ea_df.pl_name == 'Qatar-3 b') # age uncertainties understated.
            |
            (ea_df.pl_name == 'Qatar-5 b') # age uncertainties understated.
            |
            (ea_df.pl_name == 'Qatar-4 b') # age uncertainties understated. Rotn good. Li is not. isochrones who knows.
        )

        # Show these planets, because they have good ages.
        EXCEPTIONS = (
            ea_df.pl_name.str.contains('TOI-451')
        )

        sel &= (~HARDEXCLUDE)

        # This additional selection on the age uncertainties will be imposed
        # further below.
        has_factorN_age = (
            (
                (ea_df['st_age'] / np.abs(ea_df['st_ageerr1']) > 3)
                &
                (ea_df['st_age'] / np.abs(ea_df['st_ageerr2']) > 3)
            )
            |
            (EXCEPTIONS)
        )

    # Impose the selection function defined above.
    sdf = ea_df[sel]

    if verbose:

        # Print stuff about the young transiting planet sample.

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
    # Read parameters.
    #
    mp = sdf['pl_bmasse']
    rp = sdf['pl_rade']
    age = sdf['st_age']*1e9
    period = sdf['pl_orbper']
    discoverymethod = sdf['discoverymethod']
    pl_name = sdf['pl_name']

    #
    # Plot age vs rp. (age is on y axis b/c it has the error bars, and I at
    # least skimmed the footnotes of Hogg 2010).
    #
    fig,ax = plt.subplots(figsize=(1.3*4,1.3*3))

    if not colorbydisc and not colorbyage:
        ax.scatter(period, rp, color='darkgray', s=2.5, zorder=1, marker='o',
                   linewidth=0, alpha=1, rasterized=True)

    if colorbyage:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        s0 = (~has_factorN_age) | (age > 1e9)
        s1 = (has_factorN_age) & (age <= 1e9)

        ax.scatter(period[s0], rp[s0],
                   color='darkgray', s=2.5, zorder=1, marker='o', linewidth=0,
                   alpha=1, rasterized=True)

        # draw the colored points
        axins1 = inset_axes(ax, width="3%", height="33%", loc='lower right',
                            borderpad=0.7)

        #cmap = mpl.cm.magma_r
        cmap = mpl.cm.Spectral
        cmap = mpl.cm.get_cmap('magma_r', 8)
        bounds = np.arange(7.0,9.0,0.01)
        norm = mpl.colors.LogNorm(vmin=1e7, vmax=1e9)

        _p = ax.scatter(
            period[s1], rp[s1],
            c=age[s1], alpha=1, zorder=2, s=32, edgecolors='k',
            marker='o', cmap=cmap, linewidths=0.3, norm=norm
        )

        if add_plnames:
            bbox = dict(facecolor='white', alpha=0.7, pad=0, edgecolor='white',
                       lw=0)
            for _x,_y,_s in zip(period[s1],rp[s1],pl_name[s1]):
                ax.text(_x, _y, _s, ha='right', va='bottom', fontsize=2,
                        bbox=bbox, zorder=49)


        if add_kep1627:
            namelist = ['Kepler-1627']
            ages = [3.8e7]

            # NOTE: Rp is a bit off in this...
            for n, a in zip(namelist, ages):
                sel = ea_df.hostname == n

                _sdf = ea_df[sel]
                _rp = _sdf.pl_rade
                _per= _sdf.pl_orbper
                _age = np.ones(len(_sdf))*a

                if n == 'Kepler-1627':
                    _rp = (0.338*(1.015)**(0.5)*u.Rjup).to(u.Rearth).value

                ax.scatter(
                    _per, _rp,
                    c=_age, alpha=1, zorder=2, s=120, edgecolors='k',
                    marker='P', cmap=cmap, linewidths=0.3, norm=norm
                )

                if add_plnames:
                    ax.text(_per, _rp, 'Kepler-1627 b', ha='right',
                            va='bottom', fontsize=2, bbox=bbox, zorder=49)

        if add_allkep:
            namelist = ['Kepler-52', 'Kepler-968', 'Kepler-1627', 'KOI-7368',
                        'KOI-7913', 'Kepler-1643']
            ages = [3e8, 3e8, 3.8e7, 3.8e7, 3.8e7, 3.8e7]
            markers = ['o','d','P', 'v', 'X', 's']
            sizes = [80, 80, 120, 120, 120, 120]

            for n, a, m, _s in zip(namelist, ages, markers, sizes):
                sel = ea_df.hostname == n

                _sdf = ea_df[sel]
                _rp = _sdf.pl_rade
                _per= _sdf.pl_orbper
                if n == 'KOI-7368':
                    del _sdf
                    _rp = [2.67]
                    _per = [6.84]
                    _sdf = pd.DataFrame({'pl_name':'KOI-7368'}, index=[0])

                if n == 'Kepler-1627':
                    _rp = [(0.338*(1.015)**(0.5)*u.Rjup).to(u.Rearth).value]

                if n == 'KOI-7913':
                    del _sdf
                    _rp = [2.39]
                    _per = [24.0]
                    _sdf = pd.DataFrame({'pl_name':'KOI-7913'}, index=[0])

                _age = np.ones(len(_sdf))*a

                ax.scatter(
                    _per, _rp,
                    c=_age, alpha=1, zorder=2, s=_s, edgecolors='k',
                    marker=m, cmap=cmap, linewidths=0.3, norm=norm
                )

                if add_plnames:
                    print(np.array(_sdf.pl_name), np.array(_per),
                          np.array(_rp))
                    for __n, __per, __rp in zip(
                        np.array(_sdf.pl_name), np.array(_per), np.array(_rp)
                    ):
                        ax.text(__per, __rp, __n, ha='right', va='bottom',
                                fontsize=2, bbox=bbox, zorder=49)


        if add_CepHer:
            # Kepler-52 and Kepler-968
            namelist = ['Kepler-1627', 'KOI-7368', 'KOI-7913', 'Kepler-1643']
            ages = [3.8e7, 3.8e7, 3.8e7, 3.8e7]
            markers = ['P', 'v', 'X', 's']
            sizes = [120, 120, 120, 120]

            for n, a, m, _s in zip(namelist, ages, markers, sizes):
                sel = ea_df.hostname == n

                _sdf = ea_df[sel]
                _rp = _sdf.pl_rade
                _per= _sdf.pl_orbper
                if n == 'KOI-7368':
                    del _sdf
                    _rp = [2.67]
                    _per = [6.84]
                    _sdf = pd.DataFrame({'pl_name':'KOI-7368'}, index=[0])
                if n == 'Kepler-1627':
                    del _sdf
                    _rp = [(0.338*(1.015)**(0.5)*u.Rjup).to(u.Rearth).value]
                    _per = [7.2028]
                    _sdf = pd.DataFrame({'pl_name':'Kepler-1627'}, index=[0])
                if n == 'Kepler-1643':
                    del _sdf
                    _rp = [2.32]
                    _per = [5.34]
                    _sdf = pd.DataFrame({'pl_name':'Kepler-1643'}, index=[0])
                if n == 'KOI-7913':
                    del _sdf
                    _rp = [2.34]
                    _per = [24.0]
                    _sdf = pd.DataFrame({'pl_name':'KOI-7913'}, index=[0])

                _age = np.ones(len(_sdf))*a

                ax.scatter(
                    _per, _rp,
                    c=_age, alpha=1, zorder=2, s=_s, edgecolors='k',
                    marker=m, cmap=cmap, linewidths=0.3, norm=norm
                )

                if add_plnames:
                    print(np.array(_sdf.pl_name), np.array(_per),
                          np.array(_rp))
                    for __n, __per, __rp in zip(
                        np.array(_sdf.pl_name), np.array(_per), np.array(_rp)
                    ):
                        ax.text(__per, __rp, __n, ha='right', va='bottom',
                                fontsize=2, bbox=bbox, zorder=49)



        cb = fig.colorbar(_p, cax=axins1, orientation="vertical",
                          extend="neither", norm=norm)

        cb.set_ticks([1e7,1e8,1e9])
        cb.set_ticklabels(['$10^7$','$10^8$','$10^9$'])

        cb.ax.tick_params(labelsize='x-small')
        cb.ax.tick_params(size=0, which='both') # remove the ticks

        cb.ax.yaxis.set_ticks_position('left')
        cb.ax.yaxis.set_label_position('left')

        cb.set_label("Age [years]", fontsize='x-small')

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

    # flip default legend order
    if showlegend:
        handles, labels = ax.get_legend_handles_labels()
        leg = ax.legend(handles, labels, loc='lower right',
                        borderpad=0.2, handletextpad=0.1, fontsize='xx-small',
                        framealpha=1)

        leg.get_frame().set_linewidth(0.5)

    ax.set_xlabel('Orbital period [days]')
    ax.set_ylabel('Planet radius [Earths]')

    if showss:
        ax.set_xlim([0.1, 110000])
    else:
        ax.set_xlim([0.1, 1100])
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
    if add_CepHer:
        s += '_showCepHer'
    if add_plnames:
        s += '_showplnames'

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
        showlegend=0, colorbydisc=0, showarchetypes=0, showss=0, colorbyage=1,
        verbose=1, add_kep1627=0, add_CepHer=1, add_plnames=0
    )
    plot_rp_vs_period_scatter(
        showlegend=0, colorbydisc=0, showarchetypes=0, showss=0, colorbyage=1,
        verbose=1, add_kep1627=0, add_CepHer=1, add_plnames=1
    )

    plot_rp_vs_period_scatter(
        showlegend=0, colorbydisc=0, showarchetypes=0, showss=0, colorbyage=1,
        verbose=1, add_kep1627=0, add_allkep=1, add_plnames=1
    )
    plot_rp_vs_period_scatter(
        showlegend=0, colorbydisc=0, showarchetypes=0, showss=0, colorbyage=1,
        verbose=1, add_kep1627=0, add_allkep=1, add_plnames=0
    )

    plot_rp_vs_period_scatter(
        showlegend=0, colorbydisc=0, showarchetypes=0, showss=0, colorbyage=1,
        verbose=1, add_kep1627=0, add_plnames=1
    )
    plot_rp_vs_period_scatter(
        showlegend=0, colorbydisc=0, showarchetypes=0, showss=0, colorbyage=1,
        verbose=1, add_kep1627=1, add_plnames=1
    )

    plot_rp_vs_period_scatter(
        showlegend=0, colorbydisc=0, showarchetypes=0, showss=0, colorbyage=0,
        verbose=0, add_kep1627=0, add_CepHer=0, add_plnames=0
    )

    for showss in [0,1]:
        plot_rp_vs_period_scatter(
            showlegend=0, colorbydisc=0, showarchetypes=0, showss=showss, colorbyage=1,
            verbose=1, add_kep1627=0
        )
        plot_rp_vs_period_scatter(
            showlegend=0, colorbydisc=0, showarchetypes=0, showss=showss, colorbyage=1,
            verbose=1, add_kep1627=1
        )
        plot_rp_vs_period_scatter(
            showlegend=0, colorbydisc=0, showarchetypes=0, showss=showss, colorbyage=1,
            verbose=0, add_kep1627=0, add_allkep=1
        )

    plot_rp_vs_period_scatter(
        showlegend=0, colorbydisc=0, showarchetypes=0, showss=1, colorbyage=1,
        verbose=0
    )
