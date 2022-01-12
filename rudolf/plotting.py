"""
plot_TEMPLATE

Gaia (CMDs, RUWEs)
    plot_ruwe_vs_apparentmag
    plot_skychart
    plot_XYZvtang
    plot_hr

Gaia + TESS + Kepler:
    plot_rotationperiod_vs_color

GALEX:
    plot_galex

Kepler phot:
    plot_keplerlc
        _plot_zoom_light_curve
    plot_flare_checker
        _get_detrended_flare_data
    plot_ttv
    plot_ttv_vs_local_slope
    plot_rotation_period_windowslider
    plot_flare_pair_time_distribution
    plot_phasedlc_quartiles

RM:
    plot_simulated_RM
    plot_RM
    plot_RM_and_phot
    plot_rvactivitypanel
    plot_rv_checks

General for re-use:
    multiline: iterate through a colormap with many lines.
    truncate_colormap: extract a subset of a colormap
"""
import os, corner, pickle, inspect
from glob import glob
from datetime import datetime
import numpy as np, matplotlib.pyplot as plt, pandas as pd, pymc3 as pm
from numpy import array as nparr
from collections import Counter, OrderedDict
from importlib.machinery import SourceFileLoader

import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.collections import LineCollection
from matplotlib.ticker import FormatStrFormatter

from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter

from astropy import units as u, constants as const
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.table import Table

import matplotlib.patheffects as pe
from matplotlib.ticker import MaxNLocator

from aesthetic.plot import savefig, format_ax, set_style

from astrobase.services.identifiers import gaiadr2_to_tic
from astrobase.lcmath import (
    phase_magseries, phase_magseries_with_errs, phase_bin_magseries,
    phase_bin_magseries_with_errs, sigclip_magseries, find_lc_timegroups,
    phase_magseries_with_errs, time_bin_magseries
)

from cdips.utils.gaiaqueries import (
    given_source_ids_get_gaia_data, parallax_to_distance_highsn
)
from cdips.utils.tapqueries import given_source_ids_get_tic8_data
from cdips.utils.plotutils import rainbow_text
from cdips.utils.mamajek import (
    get_SpType_BpmRp_correspondence, get_SpType_GmRp_correspondence
)

from rudolf.paths import DATADIR, RESULTSDIR
from rudolf.helpers import (
    get_deltalyr_kc19_gaia_data, get_simulated_RM_data,
    get_keplerfieldfootprint_dict, get_deltalyr_kc19_comovers,
    get_deltalyr_kc19_cleansubset, get_kep1627_kepler_lightcurve,
    get_set1_koi7368,
    get_gaia_catalog_of_nearby_stars, get_clustermembers_cg18_subset,
    get_mutau_members, get_ScoOB2_members,
    get_BPMG_members,
    supplement_gaia_stars_extinctions_corrected_photometry,
    get_clean_gaia_photometric_sources, get_galex_data, get_2mass_data,
    get_rsg5_kc19_gaia_data
)
from rudolf.extinction import (
    retrieve_stilism_reddening, append_corrected_gaia_phot_Gagne2020
)

def plot_TEMPLATE(outdir):

    # get data

    # make plot
    plt.close('all')
    set_style()

    fig, ax = plt.subplots(figsize=(4,3))
    fig = plt.figure(figsize=(4,3))
    axd = fig.subplot_mosaic(
        """
        AB
        CD
        """#,
        #gridspec_kw={
        #    "width_ratios": [1, 1, 1, 1]
        #}
    )


    # set naming options
    s = ''

    bn = inspect.stack()[0][3].split("_")[1]
    outpath = os.path.join(outdir, f'{bn}{s}.png')
    savefig(fig, outpath, dpi=400)


def plot_ruwe_vs_apparentmag(outdir):

    set_style()

    _, _, trgt_df = get_deltalyr_kc19_gaia_data()
    df_edr3 = get_deltalyr_kc19_cleansubset()

    plt.close('all')

    f, ax = plt.subplots(figsize=(4,3))

    ykey = 'ruwe'
    get_yval = (
        lambda _df: np.array(
            _df[ykey]
        )
    )
    get_xval = (
        lambda _df: np.array(
            _df['phot_g_mean_mag']
        )
    )

    ax.scatter(
        get_xval(df_edr3), get_yval(df_edr3), c='k', alpha=0.9,
        zorder=4, s=12, rasterized=True, linewidths=0,
        label='$\delta$ Lyr candidates', marker='.'
    )
    ax.plot(
        get_xval(trgt_df), get_yval(trgt_df), alpha=1, mew=0.7,
        zorder=8, label='Kepler 1627A', markerfacecolor='yellow',
        markersize=16, marker='*', color='black', lw=0
    )
    print('Kepler 1627A RUWE')
    print(get_yval(trgt_df))
    print('Delta Lyr cluster RUWE mean+/-stdev')
    print(f'{np.nanmean(get_yval(df_edr3)):.3f} +/- {np.nanstd(get_yval(df_edr3)):.3f}')
    print('80,90,95,98,99 pctile')
    print(f'{np.nanpercentile(get_yval(df_edr3),80):.3f}, '+
          f'{np.nanpercentile(get_yval(df_edr3),90):.3f}, '+
          f'{np.nanpercentile(get_yval(df_edr3),95):.3f}, '+
          f'{np.nanpercentile(get_yval(df_edr3),98):.3f}, '+
          f'{np.nanpercentile(get_yval(df_edr3),99):.3f}, '
         )

    leg = ax.legend(loc='upper left', handletextpad=0.1, fontsize='small',
                    framealpha=0.9)

    ax.set_xlabel('G [mag]')
    ax.set_ylabel('RUWE')
    ax.set_yscale('log')
    ax.set_xlim([4.9,19.1])

    s = ''
    outpath = os.path.join(outdir, f'ruwe_vs_apparentmag{s}.png')

    savefig(f, outpath, dpi=400)


def plot_simulated_RM(outdir, orientation, N_mcmc=10000):

    import rmfit # Gudmundur Stefansson's RM fitting package
    assert orientation in ['prograde', 'retrograde', 'polar']
    set_style()

    cachepath = os.path.join(outdir, f'rmfit_cache_{orientation}.pkl')
    medpath = os.path.join(
        outdir, f'rmfit_median_posterior_values_{orientation}.csv'
    )

    if not os.path.exists(cachepath):

        times, rv_true = get_simulated_RM_data(orientation)
        np.random.seed(42+1)
        noise_rv = 10 # m/s internal, from Andrew Howard.

        rv_obs = rv_true + np.random.normal(loc=0, scale=noise_rv, size=len(times))

        df = pd.DataFrame({
            'bjd':times,
            'rv':rv_obs,
            'e_rv':np.ones_like(times)*noise_rv,
            'inst':np.repeat('HIRES', len(times)),
            'name':np.repeat('Kepler1627', len(times))
        })
        csvpath = os.path.join(DATADIR, 'simulated', f'Kepler1627_{orientation}.csv')
        df.to_csv(csvpath, index=False)

        # read priors from file
        priorpath = os.path.join(DATADIR, 'simulated', f'Kepler1627_priors.dat')
        L = rmfit.rmfit.LPFunction(df.bjd.values, df.rv.values, df.e_rv.values, priorpath)
        TF = rmfit.rmfit.RMFit(L)
        # max likelihood fit
        TF.minimize_PyDE(mcmc=False)

        # plot best-fit
        outpath = os.path.join(outdir, f'rmfit_maxlikelihood_{orientation}.pdf')
        TF.plot_fit(TF.min_pv)
        plt.savefig(outpath)
        plt.close('all')

        # N=1000 mcmc iterations
        L = rmfit.rmfit.LPFunction(df.bjd.values,df.rv.values,df.e_rv.values, priorpath)
        TF = rmfit.rmfit.RMFit(L)
        TF.minimize_PyDE(mcmc=True, mc_iter=N_mcmc)

        # Plot the median MCMC fit
        outpath = os.path.join(outdir, f'rmfit_mcmcbest_{orientation}.pdf')
        TF.plot_mcmc_fit()
        plt.savefig(outpath)
        plt.close('all')

        # The min values are recorded in the following attribute
        print(TF.min_pv_mcmc)

        # plot chains
        rmfit.mcmc_help.plot_chains(TF.sampler.chain,labels=TF.lpf.ps_vary.labels)
        outpath = os.path.join(outdir, f'rmfit_chains_{orientation}.pdf')
        plt.savefig(outpath)
        plt.close('all')

        # Make flatchain and posteriors
        burnin_index = 200
        chains_after_burnin = TF.sampler.chain[:,burnin_index:,:]
        flatchain = chains_after_burnin.reshape((-1,len(TF.lpf.ps_vary.priors)))
        df_post = pd.DataFrame(flatchain,columns=TF.lpf.ps_vary.labels)
        print(df_post)

        # Assess convergence, should be close to 1 (usually within a few percent, if not, then rerun MCMC with more steps)
        # This example for example would need a lot more steps, but keeping steps fewer for a quick minimal example
        # Usually good to let it run for 10000 - 20000 steps for a 'production run'
        print(42*'.')
        print('Gelman Rubin statistic, Rhat. Near 1?')
        print(rmfit.mcmc_help.gelman_rubin(chains_after_burnin))
        print(42*'.')

        # Plot corner plot
        fig = rmfit.mcmc_help.plot_corner(chains_after_burnin,
                                          show_titles=True,labels=np.array(TF.lpf.ps_vary.descriptions),title_fmt='.1f',xlabcord=(0.5, -0.2))
        outpath = os.path.join(outdir, f'rmfit_corner_full_{orientation}.png')
        savefig(fig, outpath, dpi=100)
        plt.close('all')

        # Narrow down on the lambda and vsini
        import corner
        fig = corner.corner(df_post[['lam_p1','vsini']],show_titles=True,quantiles=[0.18,0.5,0.84])
        outpath = os.path.join(
            outdir, f'rmfit_corner_lambda_vsini_{orientation}.png'
        )
        savefig(fig, outpath, dpi=300)
        plt.close('all')

        # Print median values
        df_medvals = TF.get_mean_values_mcmc_posteriors(df_post.values)
        df_medvals.to_csv(medpath, index=False)

        with open(cachepath, "wb") as f:
            pickle.dump({'TF':TF, 'flatchain':flatchain}, f)

    with open(cachepath, "rb") as f:
        d = pickle.load(f)
    TF = d['TF']
    flatchain = d['flatchain']
    df_medvals = pd.read_csv(medpath)

    ##########################################
    # The Money Figure
    # needs: TF, flatchain, df_medvals
    ##########################################
    TITLE = 'Kepler1627b (simulated)'
    NUMMODELS = 400
    shadecolor="black"
    T0 = 2454953.790531

    scale_x = lambda x : (x-T0)*24

    times1 = np.linspace(TF.lpf.data['x'][0]-0.02,TF.lpf.data['x'][-1]+0.02,500)
    pv_50 = np.percentile(flatchain,[50],axis=0)[0]
    t1_mod = np.linspace(times1.min()-0.02,times1.max()+0.02,300)
    rv_50 = TF.lpf.compute_total_model(pv_50,t1_mod)

    plt.close('all')
    mpl.rcParams.update(mpl.rcParamsDefault)
    set_style()
    fig, ax = plt.subplots(figsize=(4,3))

    # first data
    ax.errorbar(
        scale_x(TF.lpf.data['x']), TF.lpf.data['y'], TF.lpf.data['error'],
        marker='o', elinewidth=0.5, capsize=4, lw=0, mew=0.5, color='k',
        markersize=3, zorder=5
    )

    ax.plot(scale_x(t1_mod), rv_50, color="crimson", lw=2., zorder=4)

    # then models
    mmodel1 = []
    for i in range(NUMMODELS):
        if i%100 == 0: print("Sampling, i=",i)
        idx = np.random.randint(0, flatchain.shape[0])
        m1 = TF.lpf.compute_total_model(flatchain[idx],times=t1_mod)
        mmodel1.append(m1)
    mmodel1 = np.array(mmodel1)

    ax.fill_between(scale_x(t1_mod), np.quantile(mmodel1,0.16,axis=0),
                    np.quantile(mmodel1,0.84,axis=0), alpha=0.1,
                    color=shadecolor, lw=0, label='1$\sigma$',zorder=-1)
    ax.fill_between(scale_x(t1_mod), np.quantile(mmodel1,0.02,axis=0),
                    np.quantile(mmodel1,0.98,axis=0), alpha=0.1,
                    color=shadecolor, lw=0, label='2$\sigma$', zorder=-1)
    ax.fill_between(scale_x(t1_mod), np.quantile(mmodel1,0.0015,axis=0),
                    np.quantile(mmodel1,0.9985,axis=0), alpha=0.1,
                    color=shadecolor, lw=0, label='3$\sigma$', zorder=-1)

    sdf = df_medvals[df_medvals.Labels == 'lam_p1']
    from rudolf.helpers import ORIENTATIONTRUTHDICT
    txt = (
        '$\lambda_\mathrm{inj}=$'+f'{ORIENTATIONTRUTHDICT[orientation]:.1f}'+'$\!^\circ$'
        '\n$\lambda_\mathrm{fit}=$'+f'{str(sdf["values"].iloc[0])}'+'$^\circ$'
    )
    ax.text(0.03,0.03,txt,
            transform=ax.transAxes,
            ha='left',va='bottom', color='crimson')



    ax.set_xlabel('Time from transit [hours]')
    ax.set_ylabel('RV [m/s]')

    outpath = os.path.join(outdir, f'rmfit_moneyplot_{orientation}.png')
    savefig(fig, outpath, dpi=400)
    plt.close('all')


def plot_skychart(outdir, narrowlims=0, showkepler=0, showtess=0,
                  shownakedeye=0, showcomovers=0, showkepstars=0):

    set_style()

    df_dr2, df_edr3, trgt_df = get_deltalyr_kc19_gaia_data()
    if showcomovers:
        df_edr3 = get_deltalyr_kc19_cleansubset()
        #df_edr3 = get_deltalyr_kc19_comovers()

    plt.close('all')
    f, ax = plt.subplots(figsize=(3,2.5), constrained_layout=True)

    xkey, ykey = 'ra', 'dec'
    get_yval = (
        lambda _df: np.array(
            _df[ykey]
        )
    )
    get_xval = (
        lambda _df: np.array(
            _df[xkey]
        )
    )

    if not showtess:
        ax.scatter(
            get_xval(df_edr3), get_yval(df_edr3), c='k', alpha=0.9,
            zorder=4, s=8, rasterized=True, linewidths=0,
            label='$\delta$ Lyr candidates', marker='.'
        )
        ax.plot(
            get_xval(trgt_df), get_yval(trgt_df), alpha=1, mew=0.5,
            zorder=8, label='Kepler 1627', markerfacecolor='yellow',
            markersize=15, marker='*', color='black', lw=0
        )
    if showkepstars:
        _df = pd.read_csv('/Users/luke/Dropbox/proj/cdips_followup/results/KNOWNPLANET_YOUNGSTAR_XMATCH/20210916_kepgaiafun_X_cdips0pt6/20210916_kepgaiafun_X_cdipsv0pt6_short.csv')
        sel = ~pd.isnull(_df.cluster)
        _sdf = _df[sel]
        sel = (_sdf.cluster.str.contains('Stephenson'))
        _sdf0 = _sdf[sel]
        sel = (df_edr3.source_id.astype(str).isin(_sdf0.source_id.astype(str)))
        ax.scatter(
            get_xval(df_edr3[sel]), get_yval(df_edr3[sel]), c='yellow', alpha=1,
            zorder=5, s=4, rasterized=True, linewidths=0.1,
            label='... with Kepler data', marker='o', edgecolors='k'
        )
        print(f'N comovers with Kepler data: {len(df_edr3[sel])}')

    if showkepler:
        kep_d = get_keplerfieldfootprint_dict()
        for mod in np.sort(list(kep_d.keys())):
            for op in np.sort(list(kep_d[mod].keys())):
                this = kep_d[mod][op]
                ra, dec = nparr(this['corners_ra']), nparr(this['corners_dec'])
                ax.fill(ra, dec, c='lightgray', alpha=0.95, lw=0,
                        rasterized=True, zorder=-1)

    if showtess:

        _s = '' if not showcomovers else '_comovers'
        outcsv = os.path.join(outdir, f'showtess_stephenson1_cache{_s}.csv')

        if not os.path.exists(outcsv):

            from tess_stars2px import tess_stars2px_function_entry
            ra, dec = get_xval(df_edr3), get_yval(df_edr3)
            source_id = nparr(df_edr3.source_id) # actual ID doesnt matter

            out_tuple = (
                 tess_stars2px_function_entry(source_id, ra, dec)
             )

            (outID, outEclipLong, outEclipLat, outSec,
             outCam, outCcd, outColPix, outRowPix, scinfo) = out_tuple

            tdf = pd.DataFrame({
                'source_id': outID, 'elon': outEclipLong, 'elat': outEclipLat,
                'sec': outSec, 'cam': outCam, 'ccd': outCcd, 'col_px':
                outColPix, 'row_px': outRowPix, 'scinfo': scinfo
            })

            # Require within first two years
            stdf = tdf[tdf.sec <= 26]

            # now count
            res = Counter(nparr(stdf.source_id))
            tess_df = pd.DataFrame({
                "source_id":list(res.keys()), "n_tess_sector": list(res.values())}
            )
            mdf = df_edr3.merge(tess_df, on="source_id", how='left')
            assert len(mdf) == len(df_edr3)

            mdf.to_csv(outcsv, index=False)

        mdf = pd.read_csv(outcsv)

        cmap = mpl.cm.viridis
        cmap = truncate_colormap(cmap, 0., 0.75)
        bounds = np.arange(-0.5, 3.5, 1)
        ticks = (np.arange(-1,3)+1)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        cax = ax.scatter(
            mdf.ra, mdf.dec, c=mdf.n_tess_sector, cmap=cmap,
            alpha=0.9, zorder=40, s=5.5, rasterized=True, linewidths=0,
            marker='.', norm=norm,
        )

        trgt_id = "2103737241426734336" # Kepler 1627
        trgt_mdf = mdf[mdf.source_id.astype(str) == trgt_id]

        _p = ax.scatter(
            trgt_mdf.ra, trgt_mdf.dec, c=trgt_mdf.n_tess_sector, cmap=cmap,
            alpha=1, zorder=42, s=80, linewidths=0.2,
            marker='*', norm=norm, edgecolors='k'
        )

        bbox = dict(facecolor='white', alpha=0.9, pad=0, edgecolor='white')
        deltay=1.2
        ax.text(trgt_mdf.ra, trgt_mdf.dec+deltay, 'Kepler 1627', ha='center',
                va='bottom', fontsize='xx-small', bbox=bbox, zorder=49)


        # standard colorbar...
        # cb = f.colorbar(cax, extend='max', ticks=ticks)
        # cb.ax.minorticks_off()
        # cb.set_label("TESS Sectors", rotation=270, labelpad=10)

        x0,y0,dx,dy = 0.037, 0.05, 0.03, 0.38

        axins1 = inset_axes(ax, width="100%", height="100%",
                            # x0,y0, dx, dy
                            bbox_to_anchor=(x0,y0,dx,dy),
                            loc='lower left',
                            bbox_transform=ax.transAxes)
        cb = f.colorbar(_p, cax=axins1, orientation="vertical",
                        ticks=ticks)
        cb.ax.minorticks_off()
        cb.ax.tick_params(labelsize='x-small')
        cb.ax.set_title("$N_{\mathrm{TESS}}$", fontsize='x-small')

        # add white background
        import matplotlib.patches as patches
        rect = patches.Rectangle((x0,y0), 3*dx, dy+0.1, linewidth=1,
                                 edgecolor='white', facecolor='white',
                                 transform=ax.transAxes,
                                 zorder=99, alpha=0.9)
        ax.add_patch(rect)


        #axd['B'].text(0.725,0.955, '$t$ [days]',
        #        transform=axd['B'].transAxes,
        #        ha='right',va='top', color='k', fontsize='xx-small')
        cb.ax.tick_params(size=0, which='both') # remove the ticks
        #axins1.xaxis.set_ticks_position("bottom")


    if narrowlims:
        dx,dy = 30,20
        x0,y0 = float(get_xval(trgt_df)),float(get_yval(trgt_df))
        xmin,xmax = x0-dx,x0+dx
        ymin,ymax = y0-dy,y0+dy
        ax.set_xlim([xmin,xmax])
        ax.set_ylim([ymin,ymax])

    if showcomovers:
        x,y = get_xval(df_edr3), get_yval(df_edr3)
        xmin,xmax = np.max(x)+1, np.min(x)-1
        ymin,ymax = np.min(y)-1, np.max(y)+1
        ax.set_xlim([xmin,xmax])
        ax.set_xlim([304,274])
        ax.set_ylim([np.min([ymin,8]),ymax])
        ax.set_ylim([26,49])


    if shownakedeye:

        plot_starnames = 1

        from earhart.skyfield_helpers import (
            get_hygdata, get_asterisms, get_constellation_boundaries,
            get_messier_data, radec_to_lb, get_const_names
        )
        limiting_magnitude = 6.5
        _, stars = get_hygdata()
        asterisms = get_asterisms()
        messiers = get_messier_data()
        const_names = get_const_names()

        # the stars
        magnitude = stars['mag']
        marker_size = (0.5 + limiting_magnitude - magnitude) ** 1.5
        ras, decs = 360/24*nparr(stars['ra']), nparr(stars['dec'])
        stars['ra'] = ras
        stars['dec'] = decs
        ax.scatter(ras, decs, s=marker_size, alpha=0.5, lw=0, c='k')

        bbox = dict(facecolor='white', alpha=0.9, pad=0, edgecolor='white')
        if plot_starnames:
            #stars_names = stars[pd.notnull(stars['proper'])]
            #sel_names = ['Vega', 'Altair', 'Deneb', '12Del2Lyr']
            sel_names = ['Vega', 'Deneb', '12Del2Lyr']
            stars_names = stars[
                (stars.proper.astype(str).str.contains('|'.join(sel_names)))
                |
                (stars.bf.astype(str).str.contains('|'.join(sel_names)))
            ]
            stars_names = stars_names[
                ~stars_names.proper.astype(str).str.contains('Denebola')
            ]
            # delta lyra fine tuning
            sel = stars_names.bf.astype(str).str.contains('Del2Lyr')
            stars_names.loc[sel, 'proper'] = ''
            deltay=0.6
            for index, row in stars_names.iterrows():
                ax.text(row['ra'], row['dec']+deltay, row['proper'], ha='center',
                        va='bottom', fontsize='xx-small', bbox=bbox,
                        zorder=49)

            magnitude = stars_names['mag']
            marker_size = (0.5 + limiting_magnitude - magnitude) ** 1.5
            ax.scatter(stars_names['ra'], stars_names['dec'], s=1.05*marker_size,
                       alpha=1, lw=0, c='C2', zorder=50)

            radec = 283.62618-0.5, 36.898613
            delradec = -8, -4
            ax.annotate(
                'δ Lyr', radec, nparr(radec) + nparr(delradec),
                transform=ax.transData, ha='center', va='center',
                fontsize='xx-small',
                bbox=bbox,
                zorder=50,
                arrowprops=dict(
                    arrowstyle='->', shrinkA=0.0, shrinkB=0.0,
                    connectionstyle='angle3', linewidth=1
                )
            )

    if not showtess:
        leg = ax.legend(loc='lower left', handletextpad=0.1,
                        fontsize='x-small', framealpha=0.9)

    ax.set_xlabel(r'$\alpha$ [deg]', fontsize='large')
    ax.set_ylabel(r'$\delta$ [deg]', fontsize='large')

    s = ''
    if narrowlims:
        s += '_narrowlims'
    if showkepler:
        s += '_showkepler'
    if showkepstars:
        s += '_showkepstars'
    if showtess:
        s += '_showtess'
    if shownakedeye:
        s += '_shownakedeye'
    if showcomovers:
        s += '_showcomovers'

    bn = inspect.stack()[0][3].split("_")[1]
    outpath = os.path.join(outdir, f'{bn}{s}.png')
    savefig(f, outpath, dpi=400)


def plot_XYZvtang(outdir, show_1627=0, save_candcomovers=1, save_allphys=1,
                  show_comovers=0, show_sun=0, orientation=None, show_7368=0,
                  show_allknown=0, show_rsg5=0):

    plt.close('all')
    set_style()

    # NOTE: assumes plot_XYZvtang has already been run
    df_sel = get_deltalyr_kc19_cleansubset()

    _, df_edr3, trgt_df = get_deltalyr_kc19_gaia_data()
    if show_7368:
        # keys: KOI-7368, KOI-7913, Kepler-1643
        _, _, _, koi_df_dict = get_deltalyr_kc19_gaia_data(return_all_targets=1)
        set1_df = get_set1_koi7368()
    if show_allknown:
        _, _, _, koi_df_dict = get_deltalyr_kc19_gaia_data(return_all_targets=1)

    if show_rsg5:
        df_rsg5_dr2, df_rsg5_edr3 = get_rsg5_kc19_gaia_data()

    # set "dr2_radial_velocity" according to Andrew Howard HIRES recon
    # spectrum. agrees with -16.9km/s+/-0.5km/s TRES.
    trgt_df.dr2_radial_velocity = -16.7

    from earhart.physicalpositions import append_physicalpositions
    df_edr3 = append_physicalpositions(df_edr3, trgt_df)
    df_sel = append_physicalpositions(df_sel, trgt_df)
    trgt_df = append_physicalpositions(trgt_df, trgt_df)
    if show_7368:
        koi_df_dict['KOI-7368'] = append_physicalpositions(koi_df_dict['KOI-7368'], trgt_df)
        set1_df = append_physicalpositions(set1_df, trgt_df)
    if show_allknown:
        for k,v in koi_df_dict.items():
            koi_df_dict[k] = append_physicalpositions(v, trgt_df)
    if show_rsg5:
        df_rsg5_edr3 = append_physicalpositions(df_rsg5_edr3, trgt_df)

    if save_candcomovers:
        sel = (
            (df_edr3.delta_pmdec_prime_km_s > -5)
            &
            (df_edr3.delta_pmdec_prime_km_s < 2)
            &
            (df_edr3.delta_pmra_prime_km_s > -4)
            &
            (df_edr3.delta_pmra_prime_km_s < 2)
        )
        cm_df_edr3 = df_edr3[sel]

        _outpath = os.path.join(RESULTSDIR, 'tables',
                                'stephenson1_edr3_XYZvtang_candcomovers.csv')
        cm_df_edr3.to_csv(_outpath, index=False)

    if save_allphys:
        _outpath = os.path.join(RESULTSDIR, 'tables',
                                'stephenson1_edr3_XYZvtang_allphysical.csv')
        df_edr3.to_csv(_outpath, index=False)


    # use plx S/N>20 to get good XYZ.
    sdf = df_edr3[df_edr3.parallax_over_error > 20]

    #scmdf = cm_df_edr3[cm_df_edr3.parallax_over_error > 20]
    scmdf = df_sel[df_sel.parallax_over_error > 20]

    plt.close('all')

    if orientation is None:
        factor=1
        fig = plt.figure(figsize=(factor*7,factor*3))
        axd = fig.subplot_mosaic(
            """
            ABBCC
            A.DD.
            """,
            gridspec_kw={
                "width_ratios": [6,1,1,1,1],
            }
        )
    elif orientation == 'portrait':
        factor=1
        fig = plt.figure(figsize=(factor*3,factor*6.5))
        axd = fig.subplot_mosaic(
            """
            A
            B
            C
            D
            """
            #"""
            #AAAA
            #BBCC
            #DDDD
            #"""
            ,
            gridspec_kw={
                #"height_ratios": [2,1,1],
                "height_ratios": [2.5,1,1,2.5],
            }
        )
    elif orientation == 'square':
        fig = plt.figure(figsize=(6,6))
        axd = fig.subplot_mosaic(
            """
            AB
            CD
            """
            #gridspec_kw={
                #"height_ratios": [2,1,1],
                #"height_ratios": [2.5,1,1,2.5],
            #}
        )


    xydict = {
        "A":('x_pc', 'y_pc'),
        "B":('x_pc', 'z_pc'),
        "C":('y_pc', 'z_pc'),
        "D":('delta_pmra_prime_km_s', 'delta_pmdec_prime_km_s')
    }
    ldict = {
        'x_pc':'X [pc]',
        'y_pc':'Y [pc]',
        'z_pc':'Z [pc]',
        'delta_pmra_prime_km_s': r"$\Delta \mu_{{\alpha'}}^{*}$ [km$\,$s$^{-1}$]",
        'delta_pmdec_prime_km_s': r"$\Delta \mu_{\delta}^{*}$ [km$\,$s$^{-1}$]"
    }
    sun = {
        'x_pc':-8122,
        'y_pc':0,
        'z_pc':20.8
    }

    for k,v in xydict.items():

        xv, yv = v[0], v[1]

        c = 'k' if not show_comovers else 'darkgray'
        axd[k].scatter(
            sdf[xv], sdf[yv], c=c, alpha=1, zorder=7, s=2, edgecolors='none',
            rasterized=True, marker='.', label='KC19 Stephenson_1'
        )
        if show_comovers:
            axd[k].scatter(
                scmdf[xv], scmdf[yv], c='k', alpha=1, zorder=8, s=2,
                edgecolors='none', rasterized=True, marker='.'
            )

        if show_1627:
            axd[k].plot(
                trgt_df[xv], trgt_df[yv], alpha=1, mew=0.5,
                zorder=42, label='Kepler-1627', markerfacecolor='yellow',
                markersize=12, marker='*', color='black', lw=0
            )

        if show_7368:
            axd[k].plot(
                koi_df_dict['KOI-7368'][xv], koi_df_dict['KOI-7368'][yv], alpha=1, mew=0.5,
                zorder=42, label='KOI-7368', markerfacecolor='lime',
                markersize=12, marker='*', color='black', lw=0
            )
            if show_comovers:
                axd[k].scatter(
                    set1_df[set1_df.parallax_over_error>20][xv],
                    set1_df[set1_df.parallax_over_error>20][yv], c='lime', alpha=1,
                    zorder=8, s=2, edgecolors='none', rasterized=True, marker='.'
                )

        if show_allknown:
            mfcs = ['lime', 'salmon', 'magenta']
            for mfc, (name,_df) in zip(mfcs, koi_df_dict.items()):
                axd[k].plot(
                    _df[xv], _df[yv], alpha=1, mew=0.5,
                    zorder=42, label=name, markerfacecolor=mfc,
                    markersize=12, marker='*', color='black', lw=0
                )

        if show_sun and '_pc' in xv:
            axd[k].scatter(
                sun[xv], sun[yv], c='k', alpha=1, zorder=10, s=10,
                edgecolors='k', marker='.'
            )
            axd[k].plot(
                sun[xv], sun[yv], c='k', alpha=1, zorder=10, ms=10,
                marker='o', markerfacecolor="None", markeredgecolor='k',
                markeredgewidth=0.5
            )

        if show_rsg5:
            axd[k].scatter(
                df_rsg5_edr3[xv], df_rsg5_edr3[yv], c='C0', alpha=1, zorder=6,
                s=2, edgecolors='none', rasterized=True, marker='.',
                label='KC19 RSG_5'
            )

        # update x/ylabels
        axd[k].update({'xlabel': ldict[xv], 'ylabel': ldict[yv]})

        # add orientation arrows
        if k == 'A':
            delta_x = 0.1
            delta_y = 0.2
            axd['A'].arrow(0.73, 0.07, delta_x, 0, length_includes_head=True,
                           head_width=1e-2, head_length=1e-2,
                           transform=axd['A'].transAxes)
            axd['A'].text(0.73+delta_x/2, 0.085, 'Galactic center',
                          va='bottom', ha='center',
                          transform=axd['A'].transAxes, fontsize='xx-small')
            axd['A'].arrow(0.07, 0.68, 0, delta_y, length_includes_head=True,
                           head_width=1e-2, head_length=1e-2,
                           transform=axd['A'].transAxes)
            axd['A'].text(0.085, 0.73+delta_x/2, 'Galactic rotation',
                          va='center', ha='left', transform=axd['A'].transAxes,
                          fontsize='xx-small', rotation=90)

        elif k == 'B':
            delta_x = 0.1
            x0 = 0.15
            axd['B'].arrow(x0, 0.69, delta_x, 0, length_includes_head=True,
                           head_width=1e-2, head_length=1e-2,
                           transform=axd['B'].transAxes)
            axd['B'].text(x0+delta_x/2, 0.71, 'Galactic\ncenter',
                          va='bottom', ha='center',
                          transform=axd['B'].transAxes, fontsize='xx-small')

        elif k == 'C':
            delta_x = 0.1
            x0 = 0.15
            axd['C'].arrow(x0, 0.69, delta_x, 0,
                         length_includes_head=True, head_width=1e-2,
                         head_length=1e-2,
                         transform=axd['C'].transAxes)
            axd['C'].text(x0+delta_x/2, 0.71, 'Galactic\nrotation',
                          va='bottom', ha='center',
                          transform=axd['C'].transAxes, fontsize='xx-small')

    #axd['C'].update({'ylabel': '', 'yticklabels':[]})
    axd['A'].update({'xlim': [-8275, -7450], 'ylim': [-25, 525]})
    axd['B'].update({'xlim': [-8275, -7450]})
    axd['C'].update({'xlim': [-25, 525]})

    if orientation == 'square':
        axd['A'].update({'xlim': [-8275, -7825], 'ylim': [-25, 525]})
        axd['B'].update({'xlim': [-8275, -7825], 'ylim': [-25, 175]})
        axd['C'].update({'xlim': [-25, 525], 'ylim': [-25, 175]})

    for _,ax in axd.items():
        format_ax(ax)

    if orientation == 'portrait':
        #axd['C'].set_ylabel('')
        #axd['C'].set_yticklabels([])
        fig.tight_layout(w_pad=0.4, h_pad=0.4)
    else:
        fig.tight_layout(w_pad=0.2)

    if show_allknown and orientation in ['portrait', 'square']:
        axd['D'].legend(loc='best', fontsize=4)

    s = ''
    if show_1627:
        s += "_show1627"
    if show_7368:
        s += "_show7368"
    if show_allknown:
        s += "_showallknown"
    if show_rsg5:
        s += "_showrsg5"
    if show_comovers:
        s += "_showcomovers"
    if show_sun:
        s += "_showsun"
    if orientation:
        s += f"_{orientation}"

    bn = inspect.stack()[0][3].split("_")[1]
    outpath = os.path.join(outdir, f'{bn}{s}.png')
    savefig(fig, outpath, dpi=400)


def plot_keplerlc(outdir, N_samples=500, xlim=[200,300]):
    """
    mosaic format:
    AAAA
    BBBB
    CCCC
    A: full Kepler time-series
    B: 100 day slice
      [nix: CDE: 1 day slices centered on individual transit events.]
    C: data - GP on same 100 day slice.  see residual transits IN THE DATA.
    """

    ##########################################
    # BEGIN-COPYPASTE FROM run_RotGPtransit.PY
    from betty.paths import BETTYDIR
    from betty.modelfitter import ModelFitter

    # get data
    modelid, starid = 'RotGPtransit', 'Kepler_1627'
    datasets = OrderedDict()
    if starid == 'Kepler_1627':
        time, flux, flux_err, qual, texp = get_kep1627_kepler_lightcurve()
    else:
        raise NotImplementedError

    # NOTE: we have an abundance of data. so... drop all non-zero quality
    # flags.
    sel = (qual == 0)

    datasets['keplerllc'] = [time[sel], flux[sel], flux_err[sel], texp]

    priorpath = os.path.join(DATADIR, 'priors', f'{starid}_{modelid}_priors.py')
    assert os.path.exists(priorpath)
    priormod = SourceFileLoader('prior', priorpath).load_module()
    priordict = priormod.priordict

    pklpath = os.path.join(BETTYDIR, f'run_{starid}_{modelid}.pkl')
    PLOTDIR = outdir

    m = ModelFitter(modelid, datasets, priordict, plotdir=PLOTDIR,
                    pklpath=pklpath, overwrite=0, N_samples=N_samples,
                    N_cores=os.cpu_count())

    # END-COPYPASTE FROM run_RotGPtransit.PY
    ##########################################

    # make plot
    plt.close('all')
    set_style()

    fig = plt.figure(figsize=(1.22*5,1.22*3))
    axd = fig.subplot_mosaic(
        """
        B
        C
        """
    )

    _plot_zoom_light_curve(datasets, m.trace.posterior, axd, fig, xlim=xlim)

    # set naming options
    s = f'_Nsample{N_samples}_xlim{str(xlim[0]).zfill(4)}_{str(xlim[1]).zfill(4)}'

    bn = inspect.stack()[0][3].split("_")[1]
    outpath = os.path.join(outdir, f'{bn}{s}.png')
    savefig(fig, outpath, dpi=400)



def _plot_zoom_light_curve(data, soln, axd, fig, xlim=[200,300], mask=None,
                           dumpmodel=True):
    # plotter for kepler 1627

    from betty.plotting import doublemedian

    assert len(data.keys()) == 1
    name = list(data.keys())[0]
    x,y,yerr,texp = data[name]
    if mask is None:
        mask = np.ones(len(x), dtype=bool)
    gp_mod = doublemedian(soln["gp_pred"]) + doublemedian(soln["mean"])

    if dumpmodel:
        # Dump the model data for iterative bls searching
        outdf = pd.DataFrame({
            'x': x,
            'y': y,
            'yerr': yerr,
            'gp_mod': gp_mod
        })
        from cdips.utils import today_YYYYMMDD
        outpath = os.path.join(
            RESULTSDIR, 'iterative_bls', f'gptransit_dtr_{today_YYYYMMDD()}.csv'
        )
        outdf.to_csv(outpath, index=False)
        print(f'Wrote {outpath}')

    if len(x[mask]) > int(2e4):
        # see https://github.com/matplotlib/matplotlib/issues/5907
        mpl.rcParams['agg.path.chunksize'] = 10000

    x0 = np.nanmin(x[mask])

    from astrobase.lcmath import find_lc_timegroups
    ngroups, groups = find_lc_timegroups(x[mask], mingap=3/24)

    for g in groups:

        # B: 100 day slice
        #   [nix: CDE: 1 day slices centered on individual transit events.]
        ax = axd['B']
        ax.scatter(x[mask][g]-x0, 1e3*(y[mask][g]-1), c="k", s=0.5, rasterized=True,
                   linewidths=0, zorder=42)
        ax.plot(x[mask][g]-x0, 1e3*(gp_mod[g]-1), color="C2", zorder=41, lw=0.5)
        ax.set_xlim(xlim)
        ax.set_xticklabels([])
        ax.set_ylim([-60,60])

        # C: data - GP on same 100 day slice.  see residual transits IN THE DATA.
        ax = axd['C']
        ax.axhline(0, color="#aaaaaa", lw=0.5, zorder=-1)
        ax.scatter(x[mask][g]-x0, 1e3*(y[mask][g]-gp_mod[g]), c="k", s=0.5,
                   rasterized=True,
                   label="$f_{\mathrm{obs}} - f_{\mathrm{mod},\star}$",
                   linewidths=0, zorder=42)
        # NOTE: could plot the transit model
        # for i, l in enumerate("b"):
        #     mod = soln["light_curves"][:, i]
        #     ax.plot(x[mask], mod, label="planet {0}".format(l))

        ax.set_xlim(xlim)

        ax.set_ylim(
            (-3, 3)
        )

    fig.text(-0.01,0.5, 'Relative flux [ppt]', va='center',
             rotation=90)

    ax.set_xlabel('Days from start')

    fig.tight_layout()


class containerclass(object):
    pass


def _get_detrended_flare_data(cachepath, method):

    if not os.path.exists(cachepath):
        # get 1min data
        time, flux, flux_err, qual, texp = (
            get_kep1627_kepler_lightcurve(lctype='shortcadence')
        )
        sel = (qual == 0)
        time, flux, flux_err = time[sel], flux[sel], flux_err[sel]

        # identify flare times.
        # before fitting out the rotation signal, clip out points +/-3*MAD above a
        # sliding median window of 4 hours.  (less than typical flare time)
        # then detrend each segment using a spline with knots separated by 0.3 days.
        from wotan import flatten
        P_rotation = 2.6064
        if method == 'pspline':
            flat_flux, trend_flux, nsplines = flatten(
                time, flux, method='pspline', max_splines=300, edge_cutoff=0.5,
                return_trend=True, return_nsplines=True, verbose=False
            )
            print(f'Split LC into {nsplines} segments.')
        elif method == 'rspline':
            flat_flux, trend_flux = flatten(
                time, flux, method='rspline', edge_cutoff=0.5,
                window_length=P_rotation/7, return_trend=True
            )
        elif method == 'gp':
            from betty.mapfitroutines import flatten_starspots
            flat_flux, trend_flux, _ = flatten_starspots(
                time, flux, flux_err, P_rotation
            )
        elif method == 'itergp':
            from betty.mapfitroutines import flatten_starspots
            flat_flux, trend_flux, _ = flatten_starspots(
                time, flux, flux_err, P_rotation, flare_iterate=True
            )
        else:
            raise NotImplementedError(f'Got unknown method {method}.')

        #  then, find positive outliers using 
        resid = (flat_flux - np.nanmedian(flat_flux))
        mad = np.nanmedian(np.abs(resid))

        is_gtNmad = resid > 7*mad

        c = containerclass()
        c.time = time
        c.flux = flux
        c.flux_err = flux_err
        c.flat_flux = flat_flux
        c.trend_flux = trend_flux
        c.is_gtNmad = is_gtNmad
        c.mad = mad

        with open(cachepath, "wb") as f:
            pickle.dump(c, f)

    with open(cachepath, "rb") as f:
        c = pickle.load(f)

    return c


def plot_flare_checker(outdir, method=None):

    assert method in ['gp','pspline','rspline','itergp']

    cachepath = os.path.join(outdir, f'flare_checker_cache_{method}.pkl')
    c = _get_detrended_flare_data(cachepath, method)

    if len(c.time) > int(2e4):
        # see https://github.com/matplotlib/matplotlib/issues/5907
        mpl.rcParams['agg.path.chunksize'] = 10000

    # check the phase
    from complexrotators.plotting import plot_phased_light_curve
    t0 = 2454953.790531 # , 0.000769,  (-2454833)
    period = 7.20280608 # , 0.00000088
    _outpath = os.path.join(outdir, f'kepler1627_dtr-{method}_flare_phase.png')
    plot_phased_light_curve(c.time, c.flat_flux, t0, period, _outpath,
                            titlestr=None, alpha1=0, c0='k', alpha0=1,
                            phasewrap=False,
                            plotnotscatter=True)

    #
    # first-case flare checker plot
    # A: raw LC, +trend
    # B: flat LC, + flares in red
    #
    plt.close('all')
    set_style()
    fig, axs = plt.subplots(nrows=2, figsize=(9,3), sharex=True)

    axs[0].scatter(
        c.time, 1e3*(c.flux-1), c='k', s=0.5, rasterized=True, linewidths=0,
        zorder=1
    )
    axs[0].plot(
        c.time, 1e3*(c.trend_flux-1), c='C0', zorder=2, lw=0.5
    )
    axs[1].scatter(
        c.time, 1e3*(c.flat_flux-1), c='k', s=0.5, rasterized=True,
        linewidths=0, zorder=1
    )
    #axs[1].scatter(
    #    c.time[c.is_gtNmad], 1e3*(c.flat_flux[c.is_gtNmad]-1), c='red', s=0.5,
    #    rasterized=True, linewidths=0, zorder=3, label=r'$>7\times$MAD'
    #)
    #axs[1].legend(loc='best', fontsize='xx-small')
    axs[1].axhline(0, color="C0", lw=0.5, zorder=4)

    fig.text(-0.01,0.5, 'Relative flux [ppt]', va='center', rotation=90)
    axs[1].set_xlabel('Days from start')
    fig.tight_layout()

    # naming options
    s = ''
    s += "_"+method
    bn = inspect.stack()[0][3].split("_")[1]
    outpath = os.path.join(outdir, f'{bn}{s}.png')
    savefig(fig, outpath, dpi=400)

    #
    # altai-pony
    #
    plt.close('all')

    from altaipony.flarelc import FlareLightCurve

    flc = FlareLightCurve(time=c.time, flux=c.flux, flux_err=c.flux_err)
    window_length = 241 # 240 minutes
    flcd = flc.detrend("savgol", window_length=window_length)
    #kwargs passed to find_flares_in_cont_obs_period
    # N1: N1*sigma above median
    # N2: N2*sigma above detrended flux error
    flcd = flcd.find_flares(20, **{'N1':7,'N2':7,'N3':2,'sigma':c.mad})
    fl_df = flcd.flares

    fig = plt.figure(figsize=(6,4))
    axd = fig.subplot_mosaic(
        """
        A
        B
        """,
        gridspec_kw={
            "height_ratios": [1,1]
        }
    )

    fig, axd['A'] = plot_phased_light_curve(
        c.time, c.flat_flux, t0, period, None, savethefigure=False,
        titlestr=None, alpha1=0, c0='k', alpha0=1, phasewrap=False,
        plotnotscatter=True, fig=fig, ax=axd['A'], showtext=False
    )
    txt = f'$t_0$ [BJD]: {t0:.6f}\n$P$: {period:.6f} d'
    axd['A'].text(0.95,0.95,txt,
            transform=axd['A'].transAxes,
            ha='right',va='top', color='k', fontsize='xx-small')


    _pd = phase_magseries(nparr(fl_df.tstart), nparr(fl_df.ampl_rec), period,
                          t0, wrap=False, sort=False)
    x_fold = _pd['phase']
    y_fold = _pd['mags']

    axd['A'].set_xlabel('')
    axd['A'].set_xticklabels([])

    #axd['B'].vlines(
    #    x_fold, 0, 1, ls='-', lw=0.5, colors='k'
    #)
    _p = axd['B'].scatter(
        x_fold, y_fold, s=15, rasterized=True, linewidths=0.5,
        edgecolors='k',
        zorder=1, #c='k',
        c=nparr(fl_df.tstart)-min(c.time), cmap='viridis'
    )

    fig.tight_layout()

    axins1 = inset_axes(axd['B'], width="25%", height="5%", loc='upper right')
    cb = fig.colorbar(_p, cax=axins1, orientation="horizontal")
    cb.ax.tick_params(labelsize='xx-small')
    #cb.ax.set_title('Days from start', fontsize='x-small')
    axd['B'].text(0.725,0.955, '$t$ [days]',
            transform=axd['B'].transAxes,
            ha='right',va='top', color='k', fontsize='xx-small')
    cb.ax.tick_params(size=0, which='both') # remove the ticks
    #axins1.xaxis.set_ticks_position("bottom")

    #axd['B'].set_ylim([0,1])
    axd['B'].set_yscale('log')
    axd['B'].set_ylabel('Ampl.')
    axd['B'].set_xlabel('Phase')
    axd['B'].set_ylim([1e-3,1e-1])
    #axd['B'].set_yticklabels([])

    axd['A'].set_xlim([-0.1,1.1])
    axd['B'].set_xlim([-0.1,1.1])

    outpath = os.path.join(outdir, f'flarephase_{method}_altailabel.png')
    savefig(fig, outpath, dpi=400)

    #
    # analyze...
    #
    fl_df = fl_df.sort_values(by='ampl_rec', ascending=False)

    P_dict = {
        'Porb': 7.20283653,
        'Prot': 2.6064
    }

    # a "successor" event follows at <1% of the proposed periodicity.
    # check if each flare has a successor event.
    for k,P in P_dict.items():
        for cstr,_eps in zip(['','cand'],[0.01, 0.02]):
            eps = P * _eps

            has_successors = []
            for ix, r in fl_df.iterrows():
                tstart = float(r['tstart'])
                has_successor = np.any(
                    np.abs( (tstart + P) - (fl_df.tstart)) < eps
                )
                has_successors.append(has_successor)

            fl_df[f'has_{k}_{cstr}successor'] = has_successors

    outpath = os.path.join(outdir, f'fldict_{method}.csv')
    fl_df.to_csv(outpath, index=False)
    print(f'Saved {outpath}')

    #
    # the publication-ready thing.
    #
    fig = plt.figure(figsize=(6,4))
    axd = fig.subplot_mosaic(
        """
        AA
        BB
        CD
        """,
        #gridspec_kw={
        #    "height_ratios": [3, 1.5]
        #}
    )

    # A: flux+model
    # B: flux resid
    # lower left: zooms on 1420 thru 1430,  and 1457 thru 1467
    # lower right: phase-fold and ampl vs phase.

    axd['A'].scatter(
        c.time, 1e2*(c.flux-1), c='k', s=0.5, rasterized=True, linewidths=0,
        zorder=1
    )
    from astrobase.lcmath import find_lc_timegroups
    ngroups, groups = find_lc_timegroups(c.time, mingap=1/24)
    for g in groups:
        axd['A'].plot(
            c.time[g], 1e2*(c.trend_flux[g]-1), c='C0', zorder=2, lw=0.3
        )
    axd['A'].set_xticklabels([])

    axd['B'].scatter(
        c.time, 1e2*(c.flat_flux-1), c='k', s=0.5, rasterized=True,
        linewidths=0, zorder=1
    )
    axd['B'].axhline(0, color="C0", lw=0.3, zorder=4)

    axd['C'].scatter(
        c.time, 1e2*(c.flat_flux-1), c='k', s=0.5, rasterized=True,
        linewidths=0, zorder=1
    )
    axd['C'].axhline(0, color="C0", lw=0.3, zorder=4)
    yoffset=-0.6
    t0 = 1420.106829+0.026562/2
    axd['C'].plot([t0,t0+P_dict['Porb']],[yoffset,yoffset], c='C2',
                  zorder=3, lw=0.5, ls='--')
    txt = '$P_\mathrm{orb}$='+f'{P_dict["Porb"]:.3f} d'
    axd['C'].text(t0+P_dict['Porb']/2, 1.2*yoffset, txt,
                  ha='center', va='top', color='C2', fontsize='small')
    axd['C'].set_xlim([t0-2, t0+P_dict['Porb']+2])


    axd['D'].scatter(
        c.time, 1e2*(c.flat_flux-1), c='k', s=0.5, rasterized=True,
        linewidths=0, zorder=1
    )
    axd['D'].axhline(0, color="C0", lw=0.5, zorder=4)
    t0 = 1458.767766+0.01907/2
    axd['D'].plot([t0,t0+P_dict['Porb']],[yoffset,yoffset], c='C2',
                  zorder=3, lw=0.5, ls='--')
    axd['D'].text(t0+P_dict['Porb']/2, 1.2*yoffset, txt,
                  ha='center', va='top', color='C2', fontsize='small')
    axd['D'].set_xlim([t0-2, t0+P_dict['Porb']+2])
    axd['D'].set_yticklabels([])
    for a in [axd['C'],axd['D']]:
        a.set_ylim([-2.5,5])
        a.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.text(-0.01,0.5, r'Relative flux [$\times10^{-2}$]', va='center', rotation=90)
    fig.text(0.5,-0.01, 'Days from start', ha='center', va='center')

    fig.tight_layout(h_pad=0.2)
    outpath = os.path.join(outdir, f'flarezoom_{method}.png')
    savefig(fig, outpath, dpi=400)


def plot_ttv(outdir, narrowylim=0):

    # get data
    fitspath = os.path.join(DATADIR, 'ttv', 'Holczer_2016_koi5245.fits')
    hl = fits.open(fitspath)
    d = hl[1].data
    df = Table(d).to_pandas()

    # print
    sdf = df[np.abs(df["O-C"]<30)]
    print(f'Mean stdev TTV: {np.mean(sdf["O-C"]):.3f} +/- {np.std(sdf["O-C"]):.3f} min')
    # TDV is fractional, they found 3.2 hour (!) transit, convert to minutes
    print(f'Mean stdev TDV: {3.229*60*np.mean(sdf["TDV"]):.3f} +/- {3.229*60*np.std(sdf["TDV"]):.3f} min')

    # make plot
    plt.close('all')
    set_style()

    fig, ax = plt.subplots(figsize=(4,3))

    ax.errorbar(
        df['N'], df['O-C'], df['e_O-C'],
        marker='o', elinewidth=0.5, capsize=4, lw=0, mew=0.5, color='k',
        markersize=3, zorder=5
    )

    ax.set_xlabel('Transit index')
    ax.set_ylabel('O-C [min.]')

    # set naming options
    s = ''
    if narrowylim:
        s += '_narrowylim'
        ax.set_ylim([-25, 25])

    bn = inspect.stack()[0][3].split("_")[1]
    outpath = os.path.join(outdir, f'{bn}{s}.png')
    savefig(fig, outpath, dpi=400)


def _linear_model(xdata, m, b):
    return m*xdata + b


def plot_ttv_vs_local_slope(outdir):

    DO_METHOD1 = 0
    DO_METHOD2 = 1

    if DO_METHOD1:
        datadir = os.path.join(DATADIR, 'ttv', 'Masuda', 'K05245.01/')
        tnum, tc, tc_err, tc_lin = np.loadtxt(datadir+"tc.dat").T
        ttv = tc - tc_lin
        slopes = []
        for _tnum in tnum:
            _t, _f, _ferr, _fmodel = np.loadtxt(datadir+"transit_%04d.dat"%int(_tnum)).T
            coeffs = np.polyfit(_t, _f, deg=1)
            slopes.append(coeffs[0])
        slopes = np.array(slopes)
        ttv_mean, ttv_std = np.mean(ttv), np.std(ttv)
        ttv_cut = 0.02 # doesn't mean match
        sel = (np.abs(ttv)<ttv_cut)

        print(f'N_transits originally: {len(ttv)}')
        print(f'N_transits after |TTV| < 0.02 days: {len(ttv[sel])}')

        ttv = ttv[sel]
        slopes = slopes[sel]
        tc = tc[sel]
        tc_err = tc_err[sel]

        # 1000 will give ppt per day...
        # for context, the light curve changes by ~50 ppt per half rotation period
        # (which is ~1.3 days).
        XMULT = 1e3
        YMULT = 24*60

    elif DO_METHOD2:

        datadir = os.path.join(DATADIR, 'ttv', 'Masuda_20211009')
        _df = pd.read_csv(os.path.join(datadir, 'coeffs_w_variability.csv'))

        slopes = nparr(_df.dfdt) # ppt/day already
        ttv = nparr(_df.ttv)
        tc_err = nparr(_df.tc_err)
        tc = nparr(_df.tc)
        sd = nparr(_df.localSD)
        p2p = nparr(_df.localP2P)

        ttv_cut = 0.02*24*60 # 4 outliers > 30 minutes out
        sel = (np.abs(ttv)<ttv_cut)

        print(f'N_transits originally: {len(ttv)}')
        print(f'N_transits after |TTV| < 0.02 days: {len(ttv[sel])}')

        slopes = slopes[sel]
        ttv = ttv[sel]
        tc_err = tc_err[sel]
        tc = tc[sel]
        sd = sd[sel]
        p2p = p2p[sel]

        print(f'Mean stdev TTV: {np.mean(ttv):.3f} +/- {np.std(ttv):.3f} min')
        print(f'Mean stdev df/dt: {np.mean(slopes):.3f} +/- {np.std(slopes):.3f} ppt/day')

        XMULT = 1
        YMULT = 1

    slope_guess = -5e-2
    intercept_guess = 0

    # ignore the error bars... because they are probably wrong.
    p_opt, p_cov = curve_fit(
        _linear_model, slopes, ttv,
        p0=(slope_guess, intercept_guess)#, sigma=tc_err
    )
    lsfit_slope = p_opt[0]
    lsfit_slope_err = p_cov[0,0]**0.5
    lsfit_int = p_opt[1]
    lsfit_int_err = p_cov[1,1]**0.5

    # make plot
    plt.close('all')
    set_style()

    fig, ax = plt.subplots(figsize=(4,3))

    ax.set_ylabel("TTV [minutes]")
    ax.set_xlabel("Local light curve slope [ppt$\,$day$^{-1}$]")

    ax.errorbar(
        XMULT*slopes, ttv*YMULT, tc_err*YMULT,
        marker='o', elinewidth=0.5, capsize=0, lw=0, mew=0.5, color='k',
        markersize=3, zorder=5, alpha=1
    )
    ax.axhline(0, color="C4", lw=1, ls='--', zorder=-5)

    if DO_METHOD1:
        t0 = np.linspace(-0.2, 0.2, 100)
    elif DO_METHOD2:
        t0 = np.linspace(-0.2*1e3, 0.2*1e3, 100)

    #ax.plot(24*t0, z1(t0)*24*60, '-', label='linear fit', color='C1', alpha=0.6)

    ax.plot(XMULT*t0, _linear_model(t0, lsfit_slope, lsfit_int)*YMULT, '-',
            label='Best fit', color='C2', alpha=1, lw=1)

    ax.fill_between(XMULT*t0,
                    _linear_model(t0, lsfit_slope-lsfit_slope_err,
                                  lsfit_int)*YMULT,
                    _linear_model(t0, lsfit_slope+lsfit_slope_err,
                                  lsfit_int)*YMULT,
                    alpha=0.6,
                    color='C2', lw=0, label='1$\sigma$',zorder=-1)

    ax.fill_between(XMULT*t0,
                    _linear_model(t0, lsfit_slope-2*lsfit_slope_err,
                                  lsfit_int)*YMULT,
                    _linear_model(t0, lsfit_slope+2*lsfit_slope_err,
                                  lsfit_int)*YMULT,
                    alpha=0.2,
                    color='C2', lw=0, label='2$\sigma$',zorder=-2)

    print(f'Slope: {lsfit_slope:.4f} +/- {lsfit_slope_err:.4f} ppt/day')
    print(f'implies {abs(lsfit_slope/lsfit_slope_err):.2f}σ different from zero.')
    print(f'Intercept: {lsfit_int*YMULT:.5f} +/- {lsfit_int_err*YMULT:.5f} minutes')

    #ax.set_ylim([-0.03*24*60,0.03*24*60])
    ax.set_ylim([-29,29])
    ax.set_xlim([-0.149*1e3,0.149*1e3])

    ax.legend(fontsize='x-small')

    # set naming options
    s = ''

    outpath = os.path.join(outdir, f'ttv_vs_local_slope{s}.png')
    savefig(fig, outpath, dpi=400)

    # TODO: BIC difference for line versus flat.
    x = slopes
    y = ttv
    y_err = tc_err

    ax.errorbar(
        XMULT*slopes, ttv*YMULT, tc_err*YMULT,
        marker='o', elinewidth=0.5, capsize=0, lw=0, mew=0.5, color='k',
        markersize=3, zorder=5, alpha=1
    )
    ax.axhline(0, color="C4", lw=1, ls='--', zorder=-5)

    ax.plot(XMULT*t0, _linear_model(t0, lsfit_slope, lsfit_int)*YMULT, '-',
            label='Best fit', color='C2', alpha=1, lw=1)

    # MODEL#1: a line
    # number data points
    n = len(y)
    # number free parameters
    k = 2
    fudge = 1
    chi_sq_1 = np.sum(
        (y - _linear_model(x, lsfit_slope, lsfit_int))**2 /
        np.nanmedian(y_err)**2
    )
    BIC_1 = chi_sq_1 + k*np.log(n)

    # MODEL#2: the line, with slope=0.  i.e., remove a free parameter.
    k = 1
    chi_sq_2 = np.sum(
        (y - _linear_model(x, 0, lsfit_int))**2 / np.nanmedian(y_err)**2
    )
    BIC_2 = chi_sq_2 + k*np.log(n)

    chisq_ratio = chi_sq_1 / chi_sq_2
    delta_BIC = BIC_2 - BIC_1

    # https://statswithr.github.io/book/bayesian-model-selection.html
    # BF ~= exp(ΔBIC/2)
    # ΔBIC ~= 2*log(BF)
    approx_bayes_factor = np.exp(0.5*delta_BIC)

    print(f'Model 1: a line')
    print(f'n={n}')
    print(f'k={k}')
    print(f'chisq={chi_sq_1:.1f}')
    print(f'BIC={BIC_1:.1f}')

    print(f'Model 2: a constant')
    print(f'n={n}')
    print(f'k={k}')
    print(f'chisq={chi_sq_2:.1f}')
    print(f'BIC={BIC_2:.1f}')

    print(f'chi_sq_1/chi_sq_2 = {chisq_ratio:.4f}')
    print(f'deltaBIC = BIC_2 - BIC_1 = {delta_BIC:.1f}')
    print(f'approx BF = exp(0.5 deltaBIC) = {approx_bayes_factor:.3f}')




def plot_rotation_period_windowslider(outdir, koi7368=0):

    from timmy.rotationperiod import measure_rotation_period_and_unc

    # get data
    datasets = OrderedDict()
    if not koi7368:
        modelid, starid = 'gptransit', 'Kepler_1627'
        time, flux, flux_err, qual, texp = (
            get_kep1627_kepler_lightcurve(lctype='longcadence_byquarter')
        )
    else:
        modelid, starid = 'gptransit', 'KOI_7368'
        time, flux, flux_err, qual, texp = (
            get_kep1627_kepler_lightcurve(lctype='koi7368_byquarter')
        )

    N_quarters = len(time)

    periods, period_uncs = [], []
    for i in range(N_quarters):
        print(i)

        t,f = time[i],flux[i]

        plotpath = os.path.join(outdir, f'{starid}_rotationperiodslider_quarter_ix{i}.png')
        p, p_unc = measure_rotation_period_and_unc(t, f, 1, 10,
                                                   period_fit_cut=0.5, nterms=1,
                                                   samples_per_peak=50,
                                                   plotpath=plotpath)

        periods.append(p)
        period_uncs.append(p_unc)

    outdf = pd.DataFrame(
        {'period': periods, 'period_unc': period_uncs}
    )
    outpath = os.path.join(outdir, f'{starid}_rotation_period_windowslider_QUARTERS.csv')
    outdf.to_csv(outpath, index=False)

    print(42*'-')
    print(f'P: {np.mean(outdf.period):.4f}, scatter: {np.std(outdf.period):.4f}')
    print(outdf)
    print(42*'-')


def plot_flare_pair_time_distribution(uniq_dists, outpath, ylim=[0,18],
                                      hists=None):
    # uniq_dists: distribution of inter-flare arrival times.

    P_orb = 7.2028041 # pm 0.0000074 
    P_rot = 2.642 # pm 0.042
    P_syn = (1/P_rot - 1/P_orb)**(-1) # ~=4.172 day

    plt.close('all')
    set_style()
    fig, ax = plt.subplots(figsize=(4,3))
    bins = np.logspace(-1, 2, 100)

    if isinstance(uniq_dists, np.ndarray):
        label = None if hists is None else 'Observed'
        ax.hist(uniq_dists, bins=bins, cumulative=False, color='k',
                fill=False, histtype='step', linewidth=0.5, label=label)
    elif isinstance(uniq_dists, list):
        # list of uniq distances
        for u in uniq_dists:
            ax.hist(u, bins=bins, cumulative=False, color='k',
                    fill=False, histtype='step', linewidth=0.5, alpha=0.1)

    if isinstance(hists, tuple):
        avg_hist = hists[0]
        std_hist = hists[1]
        x = hists[2]
        # midpoints of logspaced values
        midway = np.exp((np.log(x[0:-1])+np.log(x[1:]))/2)
        #ax.errorbar(
        #    midway, avg_hist, yerr=std_hist, ls='none',
        #    color='gray', elinewidth=0.5, capsize=0.5, marker='.', markersize=0,
        #    zorder=-1, alpha=1
        #)
        from scipy.ndimage import gaussian_filter1d
        fn = lambda x: gaussian_filter1d(x, sigma=1)
        ax.fill_between(midway, fn(avg_hist-std_hist),
                        fn(avg_hist+std_hist), alpha=1,
                        color='darkgray', lw=0,
                        label='Poisson $\pm$1$\sigma$',zorder=-1)
        ax.fill_between(midway, fn(avg_hist-2*std_hist),
                        fn(avg_hist+2*std_hist), alpha=1,
                        color='gainsboro', lw=0,
                        label='Poisson $\pm$2$\sigma$',zorder=-2)


    ax.set_ylim(ylim)
    ylim = ax.get_ylim()
    ax.vlines(P_orb, min(ylim), max(ylim), ls='--', lw=0.5, colors='C0',
              zorder=-1, label='P$_\mathrm{orb}$'+f' ({P_orb:.3f} d)')
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


    ## correct the "stepfill" box legend label...
    #from matplotlib.lines import Line2D
    #handles, labels = ax.get_legend_handles_labels()
    #new_handles = [Line2D([], [], c=h.get_edgecolor()) for h in handles]
    #ax.legend(labels=labels, handles=new_handles, loc='upper left',
    #          fontsize='x-small')

    ax.legend(loc='upper left', fontsize='x-small')

    ax.set_xscale('log')

    ax.set_xlim([1e-1, 1e2])
    ax.set_ylim(ylim)
    ax.set_xlabel('Flare pair separation [days]')
    ax.set_ylabel('Count')
    savefig(fig, outpath, dpi=400)


def plot_hr(
    outdir, isochrone=None, color0='phot_bp_mean_mag', rasterized=False,
    show100pc=0, clusters=['$\delta$ Lyr cluster'], reddening_corr=0,
    cleanhrcut=1, extinctionmethod='gaia2018', smalllims=0,
    overplotkep1627=0, overplotkoi7368=0, getstellarparams=0
):
    """
    clusters: ['$\delta$ Lyr cluster', 'IC 2602', 'Pleiades']
    """

    set_style()

    # Kinematic (and spatially) selected subgroups of KC19's δ Lyr cluster
    #df = get_deltalyr_kc19_comovers()
    outpath = os.path.join(
        RESULTSDIR, 'tables', 'deltalyr_kc19_cleansubset_withreddening.csv'
    )
    if not os.path.exists(outpath):
        df = get_deltalyr_kc19_cleansubset()
        df = supplement_gaia_stars_extinctions_corrected_photometry(df)
        df.to_csv(outpath, index=False)
    df = pd.read_csv(outpath)
    if cleanhrcut:
        df = df[get_clean_gaia_photometric_sources(df)]
    if reddening_corr:
        print('delta Lyr cluster')
        print(df['reddening[mag][stilism]'].describe())

    if show100pc:
        import mpl_scatter_density # adds projection='scatter_density'
        df_bkgd = get_gaia_catalog_of_nearby_stars()

    if isochrone in ['mist', 'parsec']:
        # see /doc/20210816_isochrone_selection_notes.txt
        if isochrone == 'mist':
            from timmy.read_mist_model import ISOCMD
            isocmdpath = os.path.join(DATADIR, 'isochrones',
                                      #'MIST_iso_611aab73dae40.iso.cmd'
                                      #'MIST_iso_611a9964c4fe7.iso.cmd'
                                      #'MIST_iso_611ab68ad5d3c.iso.cmd'
                                      'MIST_iso_611b0094a1199.iso.cmd'
                                     )
            # relevant params: star_mass log_g log_L log_Teff Gaia_RP_DR2Rev
            # Gaia_BP_DR2Rev Gaia_G_DR2Rev
            isocmd = ISOCMD(isocmdpath)
            assert len(isocmd.isocmds) > 1

        elif isochrone == 'parsec':
            # v0, v1, v2, v3......
            isopath = os.path.join(DATADIR, 'isochrones',
                                   #'output200759820480.dat'
                                   #'output108191896536.dat'
                                   #'output412021529701.dat'
                                   #'output511503265080.dat'
                                   #'output830354854305.dat'
                                   #'output841190949156.dat'
                                   #'output225137455229.dat'
                                    'output32383660626.dat'
                                  )
            iso_df = pd.read_csv(isopath, delim_whitespace=True, comment='#')

    ##########

    plt.close('all')

    if not show100pc:
        f, ax = plt.subplots(figsize=(1.5*2,1.5*3))
    else:
        f = plt.figure(figsize=(1.5*2,1.5*3))
        ax = f.add_subplot(1, 1, 1, projection='scatter_density')

    cstr = '_corr' if reddening_corr else ''
    get_yval = (
        lambda _df: np.array(
            _df['phot_g_mean_mag'+cstr] + 5*np.log10(_df['parallax']/1e3) + 5
        )
    )
    get_xval = (
        lambda _df: np.array(
            _df[color0+cstr] - _df['phot_rp_mean_mag'+cstr]
        )
    )

    s = 4
    if smalllims:
        s = 6.5

    # mixed rasterizing along layers b/c we keep the loading times nice
    l0 = '$\delta$ Lyr candidates'
    ax.scatter(
        get_xval(df), get_yval(df), c='k', alpha=1, zorder=3,
        s=s, rasterized=False, linewidths=0.1, label=l0, marker='o',
        edgecolors='k'
    )

    if overplotkep1627:
        sel = (df.source_id == 2103737241426734336)
        sdf = df[sel]
        ax.plot(
            get_xval(df[sel]), get_yval(df[sel]),
            alpha=1, mew=0.5,
            zorder=9001, label='Kepler 1627',
            markerfacecolor='yellow', markersize=11, marker='*',
            color='black', lw=0
        )

    if 'UCL' in clusters:
        outpath = os.path.join(
            RESULTSDIR, 'tables', f'UCL_withreddening_{extinctionmethod}.csv'
        )
        if not os.path.exists(outpath):
            _df = get_ScoOB2_members()
            _df = supplement_gaia_stars_extinctions_corrected_photometry(
                _df, extinctionmethod=extinctionmethod,
                savpath=os.path.join(RESULTSDIR,'tables','UCL_stilism.csv')
            )
            _df.to_csv(outpath, index=False)
        _df = pd.read_csv(outpath)
        if cleanhrcut:
            _df = _df[get_clean_gaia_photometric_sources(_df)]
        if reddening_corr:
            print('UCL')
            print(_df['reddening[mag][stilism]'].describe())

        ax.scatter(
            get_xval(_df), get_yval(_df), c='purple', alpha=1, zorder=10,
            s=s, rasterized=False, label='UCL', marker='o',
            edgecolors='k', linewidths=0.1
        )

    if 'Set1' in clusters:
        # KOI 7368 alternative cluster definition
        outpath = os.path.join(
            RESULTSDIR, 'tables',
            'set1_koi7368_kc19_cleansubset_{extinctionmethod}.csv'
        )
        if not os.path.exists(outpath):
            _df = get_set1_koi7368()
            _df = supplement_gaia_stars_extinctions_corrected_photometry(
                _df, extinctionmethod=extinctionmethod,
                savpath=os.path.join(RESULTSDIR,'tables','set1_stilism.csv')
            )
            _df.to_csv(outpath, index=False)
        _df = pd.read_csv(outpath)
        if cleanhrcut:
            _df = _df[get_clean_gaia_photometric_sources(_df)]
        if reddening_corr:
            print('Set 1')
            print(_df['reddening[mag][stilism]'].describe())

        ax.scatter(
            get_xval(_df), get_yval(_df), c='lime', alpha=1, zorder=100000,
            s=s, rasterized=False, label='KOI 7368 vicinity (Set 1)', marker='D',
            edgecolors='k', linewidths=0.1
        )

        if overplotkoi7368:
            sel = (_df.source_id == 2128840912955018368)
            _sdf = _df[sel]
            ax.plot(
                get_xval(_sdf), get_yval(_sdf),
                alpha=1, mew=0.5,
                zorder=9001, label='KOI 7368',
                markerfacecolor='lime', markersize=11, marker='*',
                color='black', lw=0
            )


    if 'BPMG' in clusters:
        outpath = os.path.join(
            RESULTSDIR, 'tables', f'BPMG_withreddening_{extinctionmethod}.csv'
        )
        if not os.path.exists(outpath):
            _df = get_BPMG_members()
            _gdf = given_source_ids_get_gaia_data(
                nparr(_df['source_id']).astype(np.int64), 'BPMG_rudolf',
                n_max=10000, overwrite=False,
                enforce_all_sourceids_viable=True, savstr='', whichcolumns='*',
                gaia_datarelease='gaiadr2'
            )
            assert len(_df) == len(_gdf)
            del _df
            _df = supplement_gaia_stars_extinctions_corrected_photometry(
                _gdf, extinctionmethod=extinctionmethod,
                savpath=os.path.join(RESULTSDIR,'tables','BPMG_stilism.csv')
            )
            _df.to_csv(outpath, index=False)
        _df = pd.read_csv(outpath)
        if cleanhrcut:
            _df = _df[get_clean_gaia_photometric_sources(_df)]
        if reddening_corr:
            print('BPMG')
            print(_df['reddening[mag][stilism]'].describe())

        ax.scatter(
            get_xval(_df), get_yval(_df), c='yellow', alpha=1, zorder=11,
            s=s, rasterized=False, label='BPMG', marker='o',
            edgecolors='k', linewidths=0.1
        )

    if 'IC 2602' in clusters:
        outpath = os.path.join(
            RESULTSDIR, 'tables', f'IC_2602_withreddening_{extinctionmethod}.csv'
        )
        if not os.path.exists(outpath):
            _df = get_clustermembers_cg18_subset('IC_2602')
            _gdf = given_source_ids_get_gaia_data(
                nparr(_df['source_id']).astype(np.int64), 'IC_2602_rudolf',
                n_max=10000, overwrite=False,
                enforce_all_sourceids_viable=True, savstr='', whichcolumns='*',
                gaia_datarelease='gaiadr2'
            )
            assert len(_df) == len(_gdf)
            del _df
            _df = supplement_gaia_stars_extinctions_corrected_photometry(
                _gdf, extinctionmethod=extinctionmethod,
                savpath=os.path.join(RESULTSDIR,'tables','IC_2602_stilism.csv')
            )
            _df.to_csv(outpath, index=False)
        _df = pd.read_csv(outpath)
        if cleanhrcut:
            _df = _df[get_clean_gaia_photometric_sources(_df)]
        if reddening_corr:
            print('IC2602')
            print(_df['reddening[mag][stilism]'].describe())

        ax.scatter(
            get_xval(_df), get_yval(_df), c='orange', alpha=1, zorder=10,
            s=s, rasterized=False, label='IC 2602', marker='o',
            edgecolors='k', linewidths=0.1
        )

    if 'Pleiades' in clusters:
        outpath = os.path.join(
            RESULTSDIR, 'tables', f'Pleiades_withreddening_{extinctionmethod}.csv'
        )
        if not os.path.exists(outpath):
            _df = get_clustermembers_cg18_subset('Melotte_22')
            _gdf = given_source_ids_get_gaia_data(
                nparr(_df['source_id']).astype(np.int64), 'Melotte_22_rudolf',
                n_max=10000, overwrite=False,
                enforce_all_sourceids_viable=True, savstr='', whichcolumns='*',
                gaia_datarelease='gaiadr2'
            )
            assert len(_df) == len(_gdf)
            del _df
            _df = supplement_gaia_stars_extinctions_corrected_photometry(
                _gdf, extinctionmethod=extinctionmethod,
                savpath=os.path.join(RESULTSDIR,'tables','Pleiades_stilism.csv')
            )
            _df.to_csv(outpath, index=False)
        _df = pd.read_csv(outpath)
        if cleanhrcut:
            _df = _df[get_clean_gaia_photometric_sources(_df)]
        if reddening_corr:
            print('Pleiades')
            print(_df['reddening[mag][stilism]'].describe())

        ax.scatter(
            get_xval(_df), get_yval(_df), c='deepskyblue', alpha=1, zorder=1,
            s=s, rasterized=False, label='Pleiades', marker='o',
            edgecolors='k', linewidths=0.1
        )


    if 'μ Tau' in clusters:
        outpath = os.path.join(
            RESULTSDIR, 'tables', f'muTau_withreddening_{extinctionmethod}.csv'
        )
        if not os.path.exists(outpath):
            _df = get_mutau_members()
            _df = _df[~pd.isnull(_df['Gaia'])]
            _gdf = given_source_ids_get_gaia_data(
                nparr(_df['Gaia']).astype(np.int64), 'muTau_rudolf',
                n_max=10000, overwrite=False,
                enforce_all_sourceids_viable=True, savstr='', whichcolumns='*',
                gaia_datarelease='gaiadr2'
            )
            assert len(_df) == len(_gdf)
            del _df
            _df = supplement_gaia_stars_extinctions_corrected_photometry(
                _gdf, extinctionmethod=extinctionmethod,
                savpath=os.path.join(RESULTSDIR,'tables','muTau_stilism.csv')
            )
            _df.to_csv(outpath, index=False)
        _df = pd.read_csv(outpath)
        if reddening_corr:
            print('muTAU')
            print(_df['reddening[mag][stilism]'].describe())

        ax.scatter(
            get_xval(_df), get_yval(_df), c='limegreen', alpha=1, zorder=4,
            s=s, rasterized=False, label='μ Tau', marker='o',
            edgecolors='k', linewidths=0.1
        )



    if show100pc:
        from matplotlib.colors import LinearSegmentedColormap
        # "Viridis-like" colormap with white background
        white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
            (0, '#ffffff'),
            (1e-20, '#440053'),
            (0.2, '#404388'),
            (0.4, '#2a788e'),
            (0.6, '#21a784'),
            (0.8, '#78d151'),
            (1, '#fde624'),
        ], N=256)

        get_xval_no_corr = (
            lambda _df: np.array(_df[color0] - _df['phot_rp_mean_mag'])
        )
        get_yval_no_corr = (
            lambda _df: np.array(
                _df['phot_g_mean_mag'] + 5*np.log10(_df['parallax']/1e3) + 5
            )
        )

        _x = get_xval_no_corr(df_bkgd)
        _y = get_yval_no_corr(df_bkgd)
        s = np.isfinite(_x) & np.isfinite(_y)

        # add log stretch...
        from astropy.visualization import LogStretch
        from astropy.visualization.mpl_normalize import ImageNormalize
        norm = ImageNormalize(vmin=1., vmax=1000,
                              stretch=LogStretch())

        density = ax.scatter_density(_x[s], _y[s], cmap='Greys', norm=norm)

        ax.scatter(-99,-99,marker='s',c='gray',label='Field',s=5)

        # NOTE: looks fine, but really not needed.
        # # default is 72 dots per inch on the map.
        # axins1 = inset_axes(ax, width="20%", height="2%", loc='upper right')
        # cb = f.colorbar(density, cax=axins1, orientation="horizontal",
        #                 extend="neither")
        # cb.ax.tick_params(labelsize='xx-small')
        # cb.ax.tick_params(size=0, which='both') # remove the ticks
        # cb.ax.yaxis.set_ticks_position('default')
        # #cb.ax.yaxis.set_label_position('default')
        # cb.set_label("Stars per pixel", fontsize='xx-small')


    if isochrone:

        if isochrone == 'mist':

            ages = [31.6,39.8,50.1]
            logages = [7.5,7.6,7.7]
            N_ages = len(ages)
            colors = plt.cm.cool(np.linspace(0,1,N_ages))[::-1]

            for i, (a, c) in enumerate(zip(ages, colors)):
                mstar = isocmd.isocmds[i]['star_mass']
                sel = (mstar < 8)

                __sel = df.parallax_over_error > 40
                avg_distance = (
                    parallax_to_distance_highsn(np.mean(df[__sel].parallax))
                )

                corr = 5*np.log10(avg_distance) - 5
                offset = 0.05
                _yval = (
                    isocmd.isocmds[i]['Gaia_G_DR2Rev'][sel] +
                    5*np.log10(np.nanmedian(df['parallax']/1e3)) + 5
                    #+ AVG_AG
                    + corr
                    + offset
                )

                if color0 == 'phot_bp_mean_mag':
                    _c0 = 'Gaia_BP_DR2Rev'
                elif color0 == 'phot_g_mean_mag':
                    _c0 = 'Gaia_G_DR2Rev'
                else:
                    raise NotImplementedError

                _xval = (
                    isocmd.isocmds[i][_c0][sel]-isocmd.isocmds[i]['Gaia_RP_DR2Rev'][sel]
                    #+ AVG_EBpmRp
                )

                # from rudolf.extinction import given_EBmV_and_BpmRp_get_A_X
                # AG = given_EBmV_and_BpmRp_get_A_X(
                #     AVG_EBmV, _xval, bandpass='G'
                # )
                # ABP = given_EBmV_and_BpmRp_get_A_X(
                #     AVG_EBmV, _xval, bandpass='BP'
                # )
                # ARP = given_EBmV_and_BpmRp_get_A_X(
                #     AVG_EBmV, _xval, bandpass='RP'
                # )
                # (Bp-Rp)corr = (Bp-A_Bp) - (Rp-A_Rp)
                # so
                # (Bp-Rp)corr = (Bp-Rp) - A_Bp + A_Rp
                #EBpmRP = -ABP + ARP

                ax.plot(
                    _xval,
                    _yval,
                    c=c, alpha=1., zorder=9001, label=f'{a} Myr', lw=0.5
                )

                if getstellarparams and i == 1:

                    print("MIST")
                    if overplotkoi7368:
                        sel = (mstar > 0.83) & (mstar < 0.93)
                    else:
                        sel = (mstar > 0.93) & (mstar < 1.0)

                    print(f'Mstar {mstar[sel]}')
                    teff = 10**isocmd.isocmds[i]['log_Teff']
                    print(f'Teff {teff[sel]}')
                    logg = isocmd.isocmds[i]['log_g']
                    print(f'logg {logg[sel]}')
                    rstar = ((( (10**logg)*u.cm/(u.s*u.s)) /
                              (const.G*mstar*u.Msun))**(-1/2)).to(u.Rsun)
                    print(f'Rstar {rstar[sel]}')
                    rho = (mstar*u.Msun/(4/3*np.pi*rstar**3)).cgs
                    print(f'Rho [cgs] {rho[sel]}')

                    _xval = (
                        isocmd.isocmds[i][_c0][sel]-isocmd.isocmds[i]['Gaia_RP_DR2Rev'][sel]
                        #+ AVG_EBpmRp
                    )
                    _yval = (
                        isocmd.isocmds[i]['Gaia_G_DR2Rev'][sel] +
                        5*np.log10(np.nanmedian(df['parallax']/1e3)) + 5
                        #+ AVG_AG
                        + corr
                        + offset
                    )

                    ax.scatter(
                        _xval,
                        _yval,
                        color='k', alpha=1., zorder=9000000009, s=0.5, marker=".", linewidths=0
                    )

        elif isochrone == 'parsec':

            #ages = [25.1,31.6,39.8]
            #logages = [7.4,7.5,7.6]
            ages = [31.6,39.8,50.1]
            logages = [7.5,7.6,7.7]
            N_ages = len(ages)
            #colors = plt.cm.cividis(np.linspace(0,1,N_ages))
            colors = plt.cm.spring(np.linspace(0,1,N_ages))
            colors = ['red','gold','lime']

            for i, (a, la, c) in enumerate(zip(ages, logages, colors)):

                sel = (
                    (np.abs(iso_df.logAge - la) < 0.01) &
                    (iso_df.Mass < 7)
                )

                __sel = df.parallax_over_error > 40
                avg_distance = (
                    parallax_to_distance_highsn(np.mean(df[__sel].parallax))
                )
                corr = 5*np.log10(avg_distance) - 5
                offset = 0.0

                _yval = (
                    iso_df[sel]['Gmag'] +
                    5*np.log10(np.nanmedian(df['parallax']/1e3)) + 5
                    #+ AVG_AG
                    + corr
                    + offset
                )
                sel2 = (_yval < 12) # numerical issue
                _yval = _yval[sel2]

                if color0 == 'phot_bp_mean_mag':
                    _c0 = 'G_BPmag'
                elif color0 == 'phot_g_mean_mag':
                    _c0 = 'Gmag'

                #+ AVG_EBpmRp  # NOTE: reddening already included!
                _xval = (
                    iso_df[sel][sel2][_c0]-iso_df[sel][sel2]['G_RPmag']
                )

                # ax.plot(
                #     _xval, _yval,
                #     c=c, alpha=1., zorder=7, label=f'{a} Myr', lw=0.5
                # )

                late_mdwarfs = (_xval > 2.2) & (_yval > 5)
                ax.plot(
                    _xval[~late_mdwarfs], _yval[~late_mdwarfs],
                    c=c, ls='-', alpha=1., zorder=9001, label=f'{a} Myr', lw=0.5
                )
                ax.plot(
                    _xval, _yval,
                    c=c, ls='--', alpha=9000, zorder=6, lw=0.5
                )

                if getstellarparams and i == 1:

                    if overplotkoi7368:
                        sel = (
                            (np.abs(iso_df.logAge - la) < 0.01) &
                            (iso_df.Mass > 0.83) &
                            (iso_df.Mass < 0.93)
                        )
                    else:
                        sel = (
                            (np.abs(iso_df.logAge - la) < 0.01) &
                            (iso_df.Mass < 1.0) &
                            (iso_df.Mass > 0.90)
                        )
                    mstar = np.array(iso_df.Mass)

                    print(42*'#')
                    print("PARSEC")

                    print(f'{_c0} - Rp')
                    print(f'Mstar {mstar[sel]}')
                    teff = np.array(10**iso_df['logTe'])
                    print(f'Teff {teff[sel]}')
                    logg = np.array(iso_df['logg'])
                    print(f'logg {logg[sel]}')
                    rstar = ((( (10**logg)*u.cm/(u.s*u.s)) /
                              (const.G*mstar*u.Msun))**(-1/2)).to(u.Rsun)
                    print(f'Rstar {rstar[sel]}')
                    rho = (mstar*u.Msun/(4/3*np.pi*rstar**3)).cgs
                    print(f'Rho [cgs] {rho[sel]}')

                    _yval = (
                        iso_df[sel]['Gmag'] +
                        5*np.log10(np.nanmedian(df['parallax']/1e3)) + 5
                        #+ AVG_AG
                        + corr
                        + offset
                    )

                    ax.scatter(
                        iso_df[sel][_c0]-iso_df[sel]['G_RPmag'],
                        _yval,
                        c='k', alpha=1., zorder=9000000009, s=2, marker=".", linewidths=0
                    )

    ax.set_ylabel('Absolute $\mathrm{M}_{G}$ [mag]', fontsize='medium')
    if reddening_corr:
        ax.set_ylabel('Absolute $\mathrm{M}_{G,0}$ [mag]', fontsize='medium')
    if color0 == 'phot_bp_mean_mag':
        ax.set_xlabel('$G_{\mathrm{BP}}-G_{\mathrm{RP}}$ [mag]',
                      fontsize='medium')
        if reddening_corr:
            ax.set_xlabel('$(G_{\mathrm{BP}}-G_{\mathrm{RP}})_0$ [mag]',
                          fontsize='medium')
        c0s = '_Bp_m_Rp'
    elif color0 == 'phot_g_mean_mag':
        ax.set_xlabel('$G-G_{\mathrm{RP}}$ [mag]',
                      fontsize='medium')
        c0s = '_G_m_Rp'
    elif color0 == 'phot_bp_mean_mag_corr':
        ax.set_xlabel('$(G_{\mathrm{BP}}-G_{\mathrm{RP}})_0$ [mag]',
                      fontsize='medium')
        c0s = '_Bp_m_Rp_corr'
    elif color0 == 'phot_g_mean_mag_corr':
        ax.set_xlabel('$(G-G_{\mathrm{RP}})_0$ [mag]',
                      fontsize='medium')
        c0s = '_G_m_Rp_corr'
    else:
        raise NotImplementedError

    ylim = ax.get_ylim()
    ax.set_ylim((max(ylim),min(ylim)))

    if len(clusters) > 1:
        ax.legend(fontsize='xx-small', loc='upper right', handletextpad=0.1)

    if show100pc and 'phot_bp_mean_mag' in color0:
        ax.set_xlim([-1,4.5])
        ax.set_ylim((16, -3))
    elif show100pc and 'phot_g_mean_mag' in color0:
        ax.set_xlim([-0.2,2.0])
        ax.set_ylim((16, -3))

    if smalllims and 'phot_bp_mean_mag' in color0:
        ax.set_xlim([0.85,3.45])
        ax.set_ylim([12.5,5.0])
    elif smalllims and 'phot_bp_mean_mag' not in color0:
        raise NotImplementedError

    format_ax(ax)
    ax.tick_params(axis='x', which='both', top=False)


    #
    # append SpTypes (ignoring reddening)
    #
    from rudolf.priors import AVG_EBpmRp
    tax = ax.twiny()
    tax.set_xlabel('Spectral Type')

    xlim = ax.get_xlim()
    getter = (
        get_SpType_BpmRp_correspondence
        if 'phot_bp_mean_mag' in color0 else
        get_SpType_GmRp_correspondence
    )
    if not smalllims:
        sptypes, xtickvals = getter(
            ['A0V','F0V','G0V','K2V','K5V','M0V','M2V','M4V']
        )
    else:
        sptypes, xtickvals = getter(
            ['K2V','K5V','M0V','M2V','M3V','M4V']
        )
    print(sptypes)
    print(xtickvals)

    xvals = np.linspace(min(xlim), max(xlim), 100)
    tax.plot(xvals, np.ones_like(xvals), c='k', lw=0) # hidden, but fixes axis.
    tax.set_xlim(xlim)
    ax.set_xlim(xlim)

    tax.set_xticks(xtickvals+AVG_EBpmRp)
    tax.set_xticklabels(sptypes, fontsize='x-small')

    tax.xaxis.set_ticks_position('top')
    tax.tick_params(axis='x', which='minor', top=False)
    tax.get_yaxis().set_tick_params(which='both', direction='in')

    if not isochrone:
        s = ''
    else:
        s = '_'+isochrone
    if isochrone is not None:
        c0s += f'_{isochrone}'
    if show100pc:
        c0s += f'_show100pc'
    if reddening_corr:
        c0s += f'_redcorr'
    if len(clusters) > 1:
        c0s += '_'+'_'.join(clusters).replace(' ','_')
    if cleanhrcut:
        c0s += f'_cleanhrcut'
    if extinctionmethod:
        c0s += f'_{extinctionmethod}'
    if smalllims:
        c0s += '_smalllims'
    if overplotkep1627:
        c0s += '_overplotkep1627'
    if overplotkoi7368:
        c0s += '_overplotkoi7368'

    outpath = os.path.join(outdir, f'hr{s}{c0s}.png')

    savefig(f, outpath, dpi=500)


def plot_rotationperiod_vs_color(outdir, runid, yscale='linear', cleaning=None,
                                 emph_binaries=False, refcluster_only=False,
                                 talk_aspect=0, xval_absmag=0,
                                 kinematic_selection=0,
                                 overplotkep1627=0):
    """
    Plot rotation periods that satisfy the automated selection criteria
    (specified in helpers.get_autorotation_dataframe)
    """

    set_style()

    from rudolf.paths import DATADIR
    rotdir = os.path.join(DATADIR, 'rotation')

    # make plot
    plt.close('all')

    if not talk_aspect:
        f, ax = plt.subplots(figsize=(4,5))
    else:
        f, ax = plt.subplots(figsize=(4,3))

    classes = ['pleiades', 'praesepe', f'{runid}']
    colors = ['gray', 'gray', 'k']
    zorders = [-2, -3, -1]
    markers = ['X', '+', 'o']
    lws = [0., 0.1, 0.1]
    mews= [0., 0.5, 0.5]
    _s = 3 if runid != 'VelaOB2' else 1.2
    ss = [15, 12, 8]
    if runid == 'deltaLyrCluster':
        l = '$\delta$ Lyr candidates'
    else:
        raise NotImplementedError
    labels = ['Pleiades', 'Praesepe', l]

    # plot vals
    for _cls, _col, z, m, l, _lw, s, mew in zip(
        classes, colors, zorders, markers, labels, lws, ss, mews
    ):

        if f'{runid}' not in _cls:
            t = Table.read(
                os.path.join(rotdir, 'Curtis_2020_apjabbf58t5_mrt.txt'),
                format='cds'
            )
            if _cls == 'pleiades':
                df = t[t['Cluster'] == 'Pleiades'].to_pandas()
            elif _cls == 'praesepe':
                df = t[t['Cluster'] == 'Praesepe'].to_pandas()
            else:
                raise NotImplementedError

        else:
            from rudolf.helpers import get_autorotation_dataframe
            auto_df, base_df = get_autorotation_dataframe(
                runid, cleaning=cleaning
            )

            # match the rotation dataframe against kinematically selected
            # cluster members, made by plot_hr. apply cleaning criteria to
            # these
            csvpath = os.path.join(
                RESULTSDIR, 'tables', 'deltalyr_kc19_cleansubset_withreddening.csv'
            )
            df0 = pd.read_csv(csvpath)
            df0 = df0[get_clean_gaia_photometric_sources(df0)]

            if kinematic_selection:
                # nb df0 has EDR3 source_ids...
                selcols = ['source_id', 'phot_bp_mean_mag_corr',
                           'phot_rp_mean_mag_corr', 'phot_g_mean_mag_corr']
                df = pd.DataFrame(df0[selcols]).merge(
                    auto_df, left_on='source_id', right_on='dr3_source_id',
                    how='inner'
                )
                cstr = '_corr'
                # NOTE: 
                # 2021/07/29: 157 of the 278 auto-selected are in the kinematic subset
            else:
                df = auto_df
                cstr = ''
            # exclude photometric non-members
            df = df[~df.is_phot_nonmember]


        if f'{runid}' not in _cls:
            key = '(BP-RP)0' if not xval_absmag else 'MGmag'
            xval = df[key]

        else:
            if not xval_absmag:
                xval = (
                    df['phot_bp_mean_mag'+cstr] - df['phot_rp_mean_mag'+cstr]
                )
            else:
                get_xval = lambda _df: np.array(
                    _df['phot_g_mean_mag'] + 5*np.log10(_df['parallax']/1e3) + 5
                )
                xval = get_xval(df)

        ykey = 'Prot' if f'{runid}' not in _cls else 'period'

        if refcluster_only and f'{runid}' in _cls:
            pass

        else:
            ax.scatter(
                xval,
                df[ykey],
                c=_col, alpha=1, zorder=z, s=s, edgecolors='k',
                marker=m, linewidths=_lw, label=f"{l.replace('_',' ')}"
            )

        if emph_binaries and f'{runid}' in _cls:

            #
            # photometric binaries
            #
            ax.scatter(
                xval[df.is_phot_binary],
                df[df.is_phot_binary][ykey],
                c='orange', alpha=1, zorder=10, s=s, edgecolors='k',
                marker='o', linewidths=_lw, label="Photometric binary"
            )

            #
            # astrometric binaries
            #
            is_astrometric_binary = (df.ruwe > 1.2)
            df['is_astrometric_binary'] = is_astrometric_binary

            ax.scatter(
                nparr(xval)[nparr(df.is_astrometric_binary)],
                df[nparr(df.is_astrometric_binary)][ykey],
                c='red', alpha=1, zorder=9, s=s, edgecolors='k',
                marker='o', linewidths=_lw, label="RUWE>1.2"
            )

    if overplotkep1627:
        sel = (df.dr2_source_id == 2103737241426734336)
        sdf = df[sel]
        ax.plot(
            xval[sel], df[sel]['period'], alpha=1, mew=0.5,
            zorder=9001, label='Kepler 1627',
            markerfacecolor='yellow', markersize=14, marker='*',
            color='black', lw=0
        )

    ax.set_ylabel('Rotation Period [days]', fontsize='medium')

    if not xval_absmag:
        ax.set_xlabel('($G_{\mathrm{BP}}-G_{\mathrm{RP}}$)$_0$ [mag]',
                      fontsize='medium')
        if 'deltaLyrCluster' in runid:
            ax.set_xlim((0.2, 2.4))
        else:
            ax.set_xlim((0.2, 3.6))
    else:
        ax.set_xlabel('Absolute $\mathrm{M}_{G}$ [mag]', fontsize='medium')
        ax.set_xlim((1.5, 10))

    if yscale == 'linear':
        ax.set_ylim((0,13))
    elif yscale == 'log':
        ax.set_ylim((0.05,13))
    else:
        raise NotImplementedError
    ax.set_yscale(yscale)

    format_ax(ax)

    #
    # twiny for the SpTypes
    #
    tax = ax.twiny()
    tax.set_xlabel('Spectral Type')

    xlim = ax.get_xlim()
    if 'deltaLyrCluster' in runid:
        splist = ['F0V','F5V','G0V','K0V','K3V','K5V','K7V','M0V','M1V','M2V']
    else:
        splist = ['F0V','F5V','G0V','K0V','K3V','K6V','M0V','M1V','M3V','M4V']
    sptypes, BpmRps = get_SpType_BpmRp_correspondence(splist)
    print(sptypes)
    print(BpmRps)

    xvals = np.linspace(min(xlim), max(xlim), 100)
    tax.plot(xvals, np.ones_like(xvals), c='k', lw=0) # hidden, but fixes axis.
    tax.set_xlim(xlim)
    ax.set_xlim(xlim)

    tax.set_xticks(BpmRps)
    tax.set_xticklabels(sptypes, fontsize='x-small')

    tax.xaxis.set_ticks_position('top')
    tax.tick_params(axis='x', which='minor', top=False)
    tax.get_yaxis().set_tick_params(which='both', direction='in')

    # fix legend zorder
    loc = 'upper left' if yscale == 'linear' else 'lower right'
    leg = ax.legend(loc=loc, handletextpad=0.1, fontsize='x-small',
                    framealpha=1.0)


    outstr = '_vs_BpmRp'
    if emph_binaries:
        outstr += '_emphbinaries'
    if talk_aspect:
        outstr += '_talkaspect'
    if xval_absmag:
        outstr += '_xvalAbsG'
    if kinematic_selection:
        outstr += '_kinematicspatialselected'
    else:
        outstr += '_allKC19'
    outstr += f'_{yscale}'
    outstr += f'_{cleaning}'
    if overplotkep1627:
        outstr += '_overplotkep1627'
    if refcluster_only:
        outstr += '_refclusteronly'
    outpath = os.path.join(outdir, f'{runid}_rotation{outstr}.png')
    savefig(f, outpath)



def plot_RM(outdir, N_mcmc=20000, model=None):

    import rmfit # Gudmundur Stefansson's RM fitting package
    assert isinstance(model, str)
    outdir = os.path.join(outdir, model)
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    if model in ['quadratic', 'quadraticfreejitter']:
        k = 7
    elif model in ['quadraticprograde', 'quadraticretrograde',
                   'quadraticprogradefreejitter',
                   'quadraticretrogradefreejitter']:
        k = 6
    elif model in ['linear', 'linearfreejitter']:
        k = 6
    elif model in ['trendonly', 'trendonlyfreejitter']:
        k = 4 # sigma_rv, gamma, gammadot, gammadotdot
    else:
        raise NotImplementedError
    # 10 free parameters in the quadratic case
    # 9 in the linear

    rvpath = os.path.join(DATADIR, 'spec', '20210809_rvs_template_V1298TAU.csv')
    df = pd.read_csv(rvpath)

    n = len(df) # number of data points

    # quick check
    plt.close('all')
    set_style()
    fig, ax = plt.subplots(figsize=(4,3))

    # first data
    ax.errorbar(
        df.bjd - 2459433, df.rv, df.e_rv,
        marker='o', elinewidth=0.5, capsize=4, lw=0, mew=0.5, color='k',
        markersize=3, zorder=5
    )


    # t0 = 2454953.790531
    # p = 7.20280608
    # epoch 622, 
    #
    # so transit hapepned at t_622= 2459433.93591276
    # with a formal uncertainty of ~1.1 minutes. Boost it to ~2.2.

    ax.set_xlabel('JD - 2459433 [days]')
    ax.set_ylabel('RV [m/s]')
    outpath = os.path.join(outdir, f'kepler_1627_20210809_rvs.png')
    savefig(fig, outpath, dpi=400)
    plt.close('all')

    cachepath = os.path.join(outdir, f'rmfit_cache_Kepler1627.pkl')
    medpath = os.path.join(
        outdir, f'rmfit_median_posterior_values_Kepler1627.csv'
    )

    if not os.path.exists(cachepath):

        # read priors from file
        priorpath = os.path.join(outdir, f'Kepler1627_priors.dat')
        L = rmfit.rmfit.LPFunction(df.bjd.values, df.rv.values, df.e_rv.values, priorpath)
        TF = rmfit.rmfit.RMFit(L)
        # max likelihood fit
        TF.minimize_PyDE(mcmc=False, k=k, n=n)

        # plot best-fit
        outpath = os.path.join(outdir, f'rmfit_maxlikelihood.pdf')
        TF.plot_fit(TF.min_pv)
        plt.savefig(outpath)
        plt.close('all')

        # N=1000 mcmc iterations
        L = rmfit.rmfit.LPFunction(df.bjd.values,df.rv.values,df.e_rv.values, priorpath)
        TF = rmfit.rmfit.RMFit(L)
        TF.minimize_PyDE(mcmc=True, mc_iter=N_mcmc, k=k, n=n)

        # Plot the median MCMC fit
        outpath = os.path.join(outdir, f'rmfit_mcmcbest.pdf')
        TF.plot_mcmc_fit()
        plt.savefig(outpath)
        plt.close('all')

        # The min values are recorded in the following attribute
        print(TF.min_pv_mcmc)

        # plot chains
        rmfit.mcmc_help.plot_chains(TF.sampler.chain, labels=TF.lpf.ps_vary.labels)
        outpath = os.path.join(outdir, f'rmfit_chains.pdf')
        plt.savefig(outpath)
        plt.close('all')

        # Make flatchain and posteriors
        burnin_index = 200
        chains_after_burnin = TF.sampler.chain[:,burnin_index:,:]
        flatchain = chains_after_burnin.reshape((-1,len(TF.lpf.ps_vary.priors)))
        df_post = pd.DataFrame(flatchain,columns=TF.lpf.ps_vary.labels)
        print(df_post)

        # Assess convergence, should be close to 1 (usually within a
        # few percent, if not, then rerun MCMC with more steps) This
        # example for example would need a lot more steps, but keeping
        # steps fewer for a quick minimal example Usually good to let
        # it run for 10000 - 20000 steps for a 'production run'
        print(42*'.')
        print('Gelman Rubin statistic, Rhat. Near 1?')
        print(rmfit.mcmc_help.gelman_rubin(chains_after_burnin))
        print(42*'.')

        # Plot corner plot
        fig = rmfit.mcmc_help.plot_corner(
            chains_after_burnin, show_titles=True,
            labels=np.array(TF.lpf.ps_vary.descriptions),
            title_fmt='.1f', xlabcord=(0.5, -0.2)
        )

        outpath = os.path.join(outdir, f'rmfit_corner_full.png')
        savefig(fig, outpath, dpi=100)
        plt.close('all')

        if model not in [
            'trendonly','quadraticprograde','quadraticretrograde',
            'trendonlyfreejitter','quadraticprogradefreejitter','quadraticretrogradefreejitter'
        ]:
            # Narrow down on the lambda and vsini
            import corner
            fig = corner.corner(df_post[['lam_p1','vsini']],show_titles=True,quantiles=[0.18,0.5,0.84])
            outpath = os.path.join(
                outdir, f'rmfit_corner_lambda_vsini.png'
            )
            savefig(fig, outpath, dpi=300)
            plt.close('all')

        # Print median values
        df_medvals = TF.get_mean_values_mcmc_posteriors(df_post.values)
        df_medvals.to_csv(medpath, index=False)

        with open(cachepath, "wb") as f:
            pickle.dump({'TF':TF, 'flatchain':flatchain}, f)

    with open(cachepath, "rb") as f:
        d = pickle.load(f)

    TF = d['TF']
    flatchain = d['flatchain']
    df_medvals = pd.read_csv(medpath)

    ##########################################
    # The Money Figure
    # needs: TF, flatchain, df_medvals
    ##########################################
    TITLE = 'Kepler 1627 Ab'
    NUMMODELS = 400
    shadecolor="black"
    # 2459433
    #T0 = 2454953.790531
    tmid = 2459433.93591276

    #scale_x = lambda x : (x-tmid)*24  # if in hours
    scale_x = lambda x : (x-int(tmid))

    times1 = np.linspace(TF.lpf.data['x'][0]-0.02,TF.lpf.data['x'][-1]+0.02,500)
    pv_50 = np.percentile(flatchain,[50],axis=0)[0]
    t1_mod = np.linspace(times1.min()-0.02,times1.max()+0.02,300)
    rv_50 = TF.lpf.compute_total_model(pv_50,t1_mod)

    plt.close('all')
    mpl.rcParams.update(mpl.rcParamsDefault)
    set_style()
    fig, ax = plt.subplots(figsize=(4,3))

    # first data
    ax.errorbar(
        scale_x(TF.lpf.data['x']), TF.lpf.data['y'], TF.lpf.data['error'],
        marker='o', elinewidth=0.5, capsize=4, lw=0, mew=0.5, color='k',
        markersize=3, zorder=5
    )

    ax.plot(scale_x(t1_mod), rv_50, color="crimson", lw=2., zorder=4)

    # then models
    mmodel1 = []
    for i in range(NUMMODELS):
        if i%100 == 0: print("Sampling, i=",i)
        idx = np.random.randint(0, flatchain.shape[0])
        m1 = TF.lpf.compute_total_model(flatchain[idx],times=t1_mod)
        mmodel1.append(m1)
    mmodel1 = np.array(mmodel1)

    ax.fill_between(scale_x(t1_mod), np.quantile(mmodel1,0.16,axis=0),
                    np.quantile(mmodel1,0.84,axis=0), alpha=0.1,
                    color=shadecolor, lw=0, label='1$\sigma$',zorder=-1)
    ax.fill_between(scale_x(t1_mod), np.quantile(mmodel1,0.02,axis=0),
                    np.quantile(mmodel1,0.98,axis=0), alpha=0.1,
                    color=shadecolor, lw=0, label='2$\sigma$', zorder=-1)
    ax.fill_between(scale_x(t1_mod), np.quantile(mmodel1,0.0015,axis=0),
                    np.quantile(mmodel1,0.9985,axis=0), alpha=0.1,
                    color=shadecolor, lw=0, label='3$\sigma$', zorder=-1)

    sdf = df_medvals[df_medvals.Labels == 'lam_p1']
    #from rudolf.helpers import ORIENTATIONTRUTHDICT
    if model not in [
        'trendonly','quadraticprograde','quadraticretrograde',
        'trendonlyfreejitter','quadraticprogradefreejitter','quadraticretrogradefreejitter'
    ]:
        txt = (
            #'$\lambda_\mathrm{inj}=$'+f'{ORIENTATIONTRUTHDICT[orientation]:.1f}'+'$\!^\circ$'
            '$\lambda_\mathrm{fit}=$'+f'{str(sdf["values"].iloc[0])}'+'$^\circ$'
        )
        ax.text(0.03,0.03,txt,
                transform=ax.transAxes,
                ha='left',va='bottom', color='crimson')

    ax.set_xlabel(f'JD - {int(tmid)}')
    #ax.set_xlabel(f'Time from transit [hours]')
    ax.set_ylabel('RV [m/s]')

    outpath = os.path.join(outdir, f'rmfit_moneyplot.png')
    savefig(fig, outpath, dpi=400)
    plt.close('all')


def plot_RM_and_phot(outdir, model=None, showmodelbands=0, showmodel=0):
    # NOTE: assumes plot_RM has been run

    import rmfit # Gudmundur Stefansson's RM fitting package
    assert isinstance(model, str)

    rvpath = os.path.join(DATADIR, 'spec', '20210809_rvs_template_V1298TAU.csv')
    outdir = os.path.join(RESULTSDIR, 'RM', model)
    df = pd.read_csv(rvpath)

    cachepath = os.path.join(outdir, f'rmfit_cache_Kepler1627.pkl')
    medpath = os.path.join(
        outdir, f'rmfit_median_posterior_values_Kepler1627.csv'
    )

    with open(cachepath, "rb") as f:
        d = pickle.load(f)
    TF = d['TF']
    flatchain = d['flatchain']
    df_medvals = pd.read_csv(medpath)

    ##########################################
    # The Money Figure
    # needs: TF, flatchain, df_medvals
    ##########################################
    np.random.seed(42)
    NUMMODELS = 400
    shadecolor="black"

    tmid = 2459433.93591276
    t_ing = tmid - (0.5*2.823/24)
    t_egr = tmid + (0.5*2.823/24)

    scale_x = lambda x : (x-int(tmid))

    times1 = np.linspace(TF.lpf.data['x'][0]-0.02,TF.lpf.data['x'][-1]+0.02,500)
    pv_50 = np.percentile(flatchain,[50],axis=0)[0]
    t1_mod = np.linspace(times1.min()-0.02,times1.max()+0.02,300)
    rv_50 = TF.lpf.compute_total_model(pv_50,t1_mod)

    plt.close('all')
    mpl.rcParams.update(mpl.rcParamsDefault)
    set_style()
    factor = 1.2
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(factor*2.75,factor*5.5), sharex=True)

    ax = axs[0]

    ## first data
    # ax.errorbar(
    #     scale_x(TF.lpf.data['x']), TF.lpf.data['y'], TF.lpf.data['error'],
    #     marker='o', elinewidth=0.5, capsize=4, lw=0, mew=0.5, color='k',
    #     markersize=3, zorder=5
    # )
    # NOTE: just show the jitter included
    sigma_rv = float(df_medvals[df_medvals.Labels == 'sigma_rv'].medvals)
    ax.errorbar(
        scale_x(TF.lpf.data['x']), TF.lpf.data['y'],
        np.sqrt(TF.lpf.data['error']**2),
        #np.sqrt(TF.lpf.data['error']**2 + sigma_rv**2),
        marker='o', elinewidth=0.5, capsize=4, lw=0, mew=0.5, color='k',
        markersize=3, zorder=5
    )

    ax.plot(scale_x(t1_mod), rv_50, color="crimson", lw=1., zorder=4)

    # then models
    mmodel1 = []
    for i in range(NUMMODELS):
        if i%100 == 0: print("Sampling, i=",i)
        idx = np.random.randint(0, flatchain.shape[0])
        m1 = TF.lpf.compute_total_model(flatchain[idx],times=t1_mod)
        mmodel1.append(m1)
    mmodel1 = np.array(mmodel1)

    #ax.fill_between(scale_x(t1_mod), np.quantile(mmodel1,0.16,axis=0),
    #                np.quantile(mmodel1,0.84,axis=0), alpha=0.1,
    #                color=shadecolor, lw=0, label='1$\sigma$',zorder=-1)
    ax.fill_between(scale_x(t1_mod), np.quantile(mmodel1,0.02,axis=0),
                    np.quantile(mmodel1,0.98,axis=0), alpha=0.1,
                    color=shadecolor, lw=0, label='2$\sigma$', zorder=-1)
    #ax.fill_between(scale_x(t1_mod), np.quantile(mmodel1,0.0015,axis=0),
    #                np.quantile(mmodel1,0.9985,axis=0), alpha=0.1,
    #                color=shadecolor, lw=0, label='3$\sigma$', zorder=-1)

    sdf = df_medvals[df_medvals.Labels == 'lam_p1']
    #from rudolf.helpers import ORIENTATIONTRUTHDICT
    if model not in [
        'trendonly','quadraticprograde','quadraticretrograde',
        'trendonlyfreejitter','quadraticprogradefreejitter','quadraticretrogradefreejitter'
    ]:
        txt = (
            #'$\lambda_\mathrm{inj}=$'+f'{ORIENTATIONTRUTHDICT[orientation]:.1f}'+'$\!^\circ$'
            '$\lambda_\mathrm{fit}=$'+f'{str(sdf["values"].iloc[0])}'+'$^\circ$'
        )
        props = dict(boxstyle='square', facecolor='white', alpha=0.95, pad=0.15,
                     linewidth=0)
        ax.text(0.03,0.03,txt,
                transform=ax.transAxes, bbox=props,
                ha='left',va='bottom', color='crimson', zorder=1)

    ax.set_ylabel('RV [m/s]')
    ax.set_ylim([-220,270])

    #
    # NEXT: S-value
    #
    ax = axs[1]
    df = pd.read_csv(rvpath)
    ax.errorbar(
        scale_x(df.bjd), df.svalue, df.svalue_err,
        marker='o', elinewidth=0.5, capsize=4, lw=0, mew=0.5, color='k',
        markersize=3, zorder=5
    )
    ax.set_ylabel('S-value')
    ax.set_ylim([0.64,0.76])
    ax.set_yticks([0.65,0.7,0.75])

    #
    # NEXT: the photometry 
    #
    ax = axs[2]

    if showmodelbands or showmodel:
        pklpath = '/Users/luke/.betty/run_Kepler_1627_QuadMulticolorTransit.pkl'
        if os.path.exists(pklpath):
            d = pickle.load(open(pklpath, 'rb'))
            m = d['model']
            trace = d['trace']
            map_estimate = d['map_estimate']
        else:
            raise NotImplementedError(f'did not find {pklpath}')

    from rudolf.paths import PHOTDIR

    bandpasses = 'g,r,i,z'.split(',')
    colors = 'blue,green,orange,red'.split(',')

    shift = 0
    delta = 0.0055
    for bp, c in zip(bandpasses,colors):
        lcpath = glob(
            os.path.join(PHOTDIR, 'MUSCAT3', f'*muscat3_{bp}_*csv')
        )[0]
        df = pd.read_csv(lcpath)

        _time, _flux = nparr(df.BJD_TDB), nparr(df.Flux)

        bintime = 600
        bd = time_bin_magseries(_time, _flux, binsize=bintime, minbinelems=2)
        _bintime, _binflux = bd['binnedtimes'], bd['binnedmags']

        ax.scatter(scale_x(_time),
                   _flux - shift,
                   c='darkgray', zorder=3, s=5, rasterized=False,
                   linewidths=0, alpha=0.1)

        ax.scatter(scale_x(_bintime),
                   _binflux - shift,
                   c=c, zorder=4, s=14, rasterized=False,
                   linewidths=0)

        if showmodel or showmodelbands:
            name = f"muscat3_{bp}"
            mod = map_estimate[f"{name}_mu_transit"]
            ax.plot(scale_x(_time), mod - shift, color=c, zorder=41, lw=0.5)

            mod_tr = np.array(trace.posterior[f"{name}_mu_transit"])

            # begin w/ (4, 2000, 345), ncores X nchains X time
            mod_tr = mod_tr.reshape(
                mod_tr.shape[0]*mod_tr.shape[1], mod_tr.shape[2]
            )

            if showmodelbands:
                # now it's ncores.nchains X time
                shadecolor = c
                #ax.fill_between(scale_x(_time),
                #                np.quantile(mod_tr,0.16,axis=0)-shift,
                #                np.quantile(mod_tr,0.84,axis=0)-shift, alpha=0.5,
                #                color=shadecolor, lw=0, label='1$\sigma$',zorder=40)
                ax.fill_between(scale_x(_time), np.quantile(mod_tr,0.02,axis=0)-shift,
                                np.quantile(mod_tr,0.98,axis=0)-shift,
                                alpha=0.4,
                                color=shadecolor, lw=0, label='2$\sigma$',
                                zorder=40)


        props = dict(boxstyle='square', facecolor='white', alpha=0.7, pad=0.15,
                     linewidth=0)
        txt = f'{bp}'
        ax.text(0.815, 1e-3 + np.nanmedian(_flux) - shift, txt,
                ha='left', va='top', bbox=props, zorder=6, color=c,
                fontsize='x-small')

        shift += delta

    ax.set_xlabel(f'JD - {int(tmid)}')
    ax.set_ylabel(f'Relative flux')
    ax.set_xlim([0.79,1.06])
    ax.set_ylim([0.978,1.009])

    for _t in [t_ing, t_egr]:
        for ix, ax in enumerate(axs):
            ylim = ax.get_ylim()
            ax.vlines(
                scale_x(_t), min(ylim), max(ylim), ls='--', lw=0.5, colors='k',
                alpha=0.5, zorder=-2
            )
            ax.set_ylim(ylim)
            if ix == 0:
                props = dict(boxstyle='square', facecolor='white', alpha=0.95,
                             pad=0.15, linewidth=0)
                txt = 'Expected\nIngress' if _t < tmid else 'Expected\nEgress'
                ax.text(scale_x(_t), 200, txt,
                        bbox=props,
                        ha='center',va='top', color='gray', zorder=1,
                        fontsize='xx-small')


    fig.tight_layout(h_pad=0.2, w_pad=0.2)

    s = ''
    if showmodel:
        s += "_showmodel"
    if showmodelbands:
        s += "_showmodelbands"

    outpath = os.path.join(outdir, f'rm_RV_and_phot{s}.png')
    savefig(fig, outpath, dpi=400)
    plt.close('all')


def plot_rvactivitypanel(outdir):

    # get data
    lines = ['Ca K', 'Ca H', 'Hα']
    globs = ['Kep*bj*order07*', 'Kep*bj*order07*', 'Kep*ij*order00*']
    deltawav = 5
    xlims = [
        [3933.66-deltawav, 3933.66+deltawav], # Ca K
        [3968.47-deltawav, 3968.47+deltawav], # Ca H
        [6562.8-deltawav, 6562.8+deltawav], # Halpa
    ]

    rvpath = os.path.join(DATADIR, 'spec', '20210809_rvs_template_V1298TAU.csv')
    rvdf = pd.read_csv(rvpath)

    # make plot
    plt.close('all')
    set_style()

    #fig, ax = plt.subplots(figsize=(4,3))
    fig = plt.figure(figsize=(5*1.3,3*1.3))
    axd = fig.subplot_mosaic(
        """
        012
        345
        """,
        gridspec_kw={
            #"width_ratios": [1, 1, 1, 1]
            "height_ratios": [1, 3]
        },
    )

    from scipy.ndimage import gaussian_filter1d

    for ix, (l,g,xlim) in enumerate(zip(lines, globs, xlims)):

        csvpaths = glob(os.path.join(DATADIR, 'rvactivity', g))
        assert len(csvpaths) > 0

        _df = pd.read_csv(csvpaths[0])
        axd[str(ix)].plot(
            _df.wav, _df.model_flx, c='k', zorder=3, lw=0.2
        )

        colors = plt.cm.viridis(np.linspace(0,1,len(csvpaths)))

        times, diff_flxs, wavs = [], [], []

        for c_ix, csvpath in enumerate(csvpaths):
            expindex = int(os.path.basename(csvpath).split("_")[1].split(".")[1])
            bjd = float(rvdf.loc[rvdf.expindex==expindex, 'bjd'])
            df = pd.read_csv(csvpath)
            times.append(bjd)

            if ix == 2:
                # mask cosmic
                sel = df.diff_flx > 0.15
                df.loc[sel, 'diff_flx'] = 0

            # smoothed
            fn = lambda x: gaussian_filter1d(x, sigma=6)
            diff_flxs.append(fn(nparr(df.diff_flx)))

            wavs.append(nparr(df.wav))

        lc = multiline(
            wavs, diff_flxs, 24*(nparr(times)-np.min(times)), cmap='Spectral',
            ax=axd[str(ix+3)], lw=1
        )

        axd[str(ix)].set_title(l)
        axd[str(ix)].set_xticklabels([])

        axd[str(ix)].set_xlim(xlim)
        axd[str(ix+3)].set_xlim(xlim)
        if ix in [0,1]:
            axd[str(ix+3)].set_ylim([-0.3, 0.3])
        elif ix == 2:
            axd[str(ix+3)].set_ylim([-0.1, 0.1])

        if ix == 2:
            axd[str(ix+3)].set_yticks([-0.1,0,0.1])

    axins1 = inset_axes(axd['5'], width="40%", height="5%", loc='lower right',
                        borderpad=1.5)
    cb = fig.colorbar(lc, cax=axins1, orientation="horizontal")
    cb.ax.tick_params(labelsize='xx-small')
    cb.ax.set_title('Time [hours]', fontsize='xx-small')
    # axd['B'].text(0.725,0.955, '$t$ [days]',
    #         transform=axd['B'].transAxes,
    #         ha='right',va='top', color='k', fontsize='xx-small')
    cb.ax.tick_params(size=0, which='both') # remove the ticks
    axins1.xaxis.set_ticks_position("bottom")

    fig.text(-0.01,0.5, 'Relative flux', va='center',
             rotation=90)
    fig.text(0.5,-0.01, 'Wavelength [$\AA$]', va='center', ha='center', rotation=0)

    fig.tight_layout()

    # set naming options
    s = ''

    bn = inspect.stack()[0][3].split("_")[1]
    outpath = os.path.join(outdir, f'{bn}{s}.png')
    savefig(fig, outpath, dpi=400)


def plot_rvchecks(outdir):

    # get data
    rvpath = os.path.join(DATADIR, 'spec', '20210809_rvs_template_V1298TAU.csv')
    df = pd.read_csv(rvpath)

    # make plot
    plt.close('all')
    set_style()

    fig, ax = plt.subplots(figsize=(4,3))

    ax.errorbar(
        df.svalue, df.rv, df.e_rv,
        marker='o', elinewidth=0.5, capsize=4, lw=0, mew=0.5, color='k',
        markersize=3, zorder=5
    )

    ax.set_xlabel('S-value')
    ax.set_ylabel('RV [m/s]')

    # set naming options
    s = '_rv_vs_svalue'

    bn = inspect.stack()[0][3].split("_")[1]
    outpath = os.path.join(outdir, f'{bn}{s}.png')
    savefig(fig, outpath, dpi=400)


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """
    usage:
        cmap = mpl.cm.viridis
        new_cmap = truncate_colormap(cmap, 0., 0.7)
    """
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.
        format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap


def multiline(xs, ys, c, ax=None, **kwargs):
    """Plot lines with different colorings
    Copy-pasted from
    https://stackoverflow.com/questions/38208700/matplotlib-plot-lines-with-colors-through-colormap

    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection

    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

    Returns
    -------
    lc : LineCollection instance.
    """

    # find axes
    ax = plt.gca() if ax is None else ax

    # create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)

    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    lc.set_array(np.asarray(c))

    # add lines to axes and rescale 
    #    Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(lc)
    ax.autoscale()
    return lc


def plot_phasedlc_quartiles(
    outdir, whichtype='slope', mask=None, from_trace=False,
    ylimd=None, binsize_minutes=20, map_estimate=None, fullxlim=False, BINMS=3,
    do_hacky_reprerror=True, N_samples=2000
):
    """
    Args:

        whichtype (str): "slope", "ttv", "p2p", or "sd" for
        "{whichtype}_quartiles".  This is the quantity that is being binned by.

        data (OrderedDict): data['tess'] = (time, flux, flux_err, t_exp)

        soln (az.data.inference_data.InferenceData): can MAP solution from
        PyMC3. can also be the posterior's trace itself (m.trace.posterior).
        If the posterior is passed, bands showing the 2-sigma uncertainty
        interval will be drawn.

        outpath (str): where to save the output.

        from_trace: True is using m.trace.posterior

    Optional:

        map_estimate: if passed, uses this as the "best fit" line. Otherwise,
        the nanmean is used (nanmedian was also considered).

    """
    from betty.plotting import (
        doublemedian, doublemean, doublepctile, get_ylimguess
    )

    # get the run_RotGPtransit results...
    ##########################################
    # BEGIN-COPYPASTE FROM run_RotGPtransit.PY
    from betty.paths import BETTYDIR
    from betty.modelfitter import ModelFitter

    # get data
    modelid, starid = 'RotGPtransit', 'Kepler_1627'
    datasets = OrderedDict()
    if starid == 'Kepler_1627':
        time, flux, flux_err, qual, texp = get_kep1627_kepler_lightcurve()
    else:
        raise NotImplementedError

    # NOTE: we have an abundance of data. so... drop all non-zero quality
    # flags.
    sel = (qual == 0)

    datasets['keplerllc'] = [time[sel], flux[sel], flux_err[sel], texp]

    priorpath = os.path.join(DATADIR, 'priors', f'{starid}_{modelid}_priors.py')
    assert os.path.exists(priorpath)
    priormod = SourceFileLoader('prior', priorpath).load_module()
    priordict = priormod.priordict

    pklpath = os.path.join(BETTYDIR, f'run_{starid}_{modelid}.pkl')
    PLOTDIR = outdir

    m = ModelFitter(modelid, datasets, priordict, plotdir=PLOTDIR,
                    pklpath=pklpath, overwrite=0, N_samples=N_samples,
                    N_cores=os.cpu_count())

    # END-COPYPASTE FROM run_RotGPtransit.PY
    ##########################################

    data = datasets
    soln = m.trace.posterior
    outpath = os.path.join(outdir, f'{starid}_{modelid}_phasedlc_{whichtype}_quartiles.png')

    if not fullxlim:
        scale_x = lambda x: x*24
    else:
        scale_x = lambda x: x

    assert len(data.keys()) == 1
    name = list(data.keys())[0]
    x,y,yerr,texp = data[name]

    if mask is None:
        mask = np.ones(len(x), dtype=bool)

    ####################
    # begin get TTV data

    DO_METHOD1 = 0
    DO_METHOD2 = 1

    if DO_METHOD1:
        datadir = os.path.join(DATADIR, 'ttv', 'Masuda', 'K05245.01/')
        tnum, tc, tc_err, tc_lin = np.loadtxt(datadir+"tc.dat").T
        ttv = tc - tc_lin
        slopes = []
        for _tnum in tnum:
            _t, _f, _ferr, _fmodel = np.loadtxt(datadir+"transit_%04d.dat"%int(_tnum)).T
            coeffs = np.polyfit(_t, _f, deg=1)
            slopes.append(coeffs[0])
        slopes = np.array(slopes)
        ttv_mean, ttv_std = np.mean(ttv), np.std(ttv)
        ttv_cut = 0.02 # doesn't mean match
        sel = (np.abs(ttv)<ttv_cut)

        print(f'N_transits originally: {len(ttv)}')
        print(f'N_transits after |TTV| < 0.02 days: {len(ttv[sel])}')

        ttv = ttv[sel]
        slopes = slopes[sel]
        tc = tc[sel]
        tc_err = tc_err[sel]

    elif DO_METHOD2:

        datadir = os.path.join(DATADIR, 'ttv', 'Masuda_20211009')
        _df = pd.read_csv(os.path.join(datadir, 'coeffs_w_variability.csv'))

        slopes = nparr(_df.dfdt) # ppt/day already
        ttv = nparr(_df.ttv)
        tc_err = nparr(_df.tc_err)
        tc = nparr(_df.tc)
        sd = nparr(_df.localSD)
        p2p = nparr(_df.localP2P)

        ttv_cut = 0.02*24*60 # 4 outliers > 30 minutes out
        sel = (np.abs(ttv)<ttv_cut)

        print(f'N_transits originally: {len(ttv)}')
        print(f'N_transits after |TTV| < 0.02 days: {len(ttv[sel])}')

        slopes = slopes[sel]
        ttv = ttv[sel]
        tc_err = tc_err[sel]
        tc = tc[sel]
        sd = sd[sel]
        p2p = p2p[sel]

    # get quartiles...
    quartiles = [0,25,50,75,100]
    paramdict = {
        'slope': [slopes, 'ppt/day', "\mathrm{d}f/\mathrm{d}t"],
        'ttv': [ttv, 'minutes', "\mathrm{TTV}"],
        'p2p': [p2p, 'ppt', "\mathrm{P2P}"],
        'sd': [sd, 'ppt', "\mathrm{STDEV}"]
    }

    allowed_types = list(paramdict.keys())
    assert whichtype in nparr(allowed_types)

    param = paramdict[whichtype][0] # e.g., "slopes"
    unit = paramdict[whichtype][1]
    paramstr = paramdict[whichtype][2]

    param_percentiles = [np.percentile(param, q) for q in quartiles ]
    print(whichtype)
    for q,s in zip(quartiles, param_percentiles):
        print(f'{q}%: {s:.4f} {unit}')

    ##########################################
    # make the plot

    plt.close('all')
    set_style()
    #fig = plt.figure(figsize=(0.66*10,0.66*12)) #standard
    fig = plt.figure(figsize=(0.66*9,0.66*6.5))
    axd = fig.subplot_mosaic(
        """
        01
        45
        23
        67
        """,
        gridspec_kw={
            "height_ratios": [1,2,1,2]
        }
    )

    if from_trace==True:
        _t0 = np.nanmean(soln["t0"])
        _per = np.nanmean(soln["period"])

        if len(soln["gp_pred"].shape)==3:
            # (4, 500, 46055), ncores X nchains X time
            medfunc = doublemean
            pctfunc = doublepctile
        elif len(soln["gp_pred"].shape)==2:
            medfunc = lambda x: np.mean(x, axis=0)
            pctfunc = lambda x: np.percentile(x, [2.5,97.5], axis=0)
        else:
            raise NotImplementedError
        gp_mod = (
            medfunc(soln["gp_pred"]) +
            medfunc(soln["mean"])
        )
        lc_mod = (
            medfunc(np.sum(soln["light_curves"], axis=-1))
        )
        lc_mod_band = (
            pctfunc(np.sum(soln["light_curves"], axis=-1))
        )

        _yerr = (
            np.sqrt(yerr[mask] ** 2 +
                    np.exp(2 * medfunc(soln["log_jitter"])))
        )

    if (from_trace == False) or (map_estimate is not None):
        if map_estimate is not None:
            # If map_estimate is given, over-ride the mean/median estimate above,
            # take the MAP.
            print('WRN! Overriding mean/median estimate with MAP.')
            soln = deepcopy(map_estimate)
        _t0 = soln["t0"]
        _per = soln["period"]
        gp_mod = soln["gp_pred"] + soln["mean"]
        lc_mod = soln["light_curves"][:, 0]
        import IPython; IPython.embed()
        _yerr = (
            np.sqrt(yerr[mask] ** 2 + np.exp(2 * soln["log_jitter"]))
        )

    if len(x) > int(2e4):
        # see https://github.com/matplotlib/matplotlib/issues/5907
        mpl.rcParams['agg.path.chunksize'] = 10000

    #
    # begin the plot!
    #

    for q_ix in range(0,4):

        #
        # Construct mask by slope quartile
        #
        x_fold = (x - _t0 + 0.5 * _per) % _per - 0.5 * _per

        param_lower = param_percentiles[q_ix]
        param_upper = param_percentiles[q_ix+1]
        sel = (
            (param >= param_lower) &
            (param <= param_upper)
        )
        txt = (
            '$'+paramstr+' \in$'+
            f'[{param_lower:.1f}, {param_upper:.1f}] {unit}'
        )

        these_tc = tc[sel]

        print(txt)

        mask = np.zeros(len(x), dtype=bool)
        WINDOW = 10/24 # days
        for _tc in these_tc:
            tmin, tmax = _tc-WINDOW,_tc+WINDOW
            mask |= ( (x > tmin) & (x < tmax) )

        #
        # Define upper axis
        #
        ax = axd[str(q_ix)]
        #ax.set_title(txt)
        y0 = (y[mask]-gp_mod[mask]) - np.nanmedian(y[mask]-gp_mod[mask])
        ax.errorbar(scale_x(x_fold[mask]), 1e3*(y0),
                    yerr=1e3*_yerr[mask], color="darkgray", label="data",
                    fmt='.', elinewidth=0.2, capsize=0, markersize=1,
                    rasterized=True, zorder=-1)

        binsize_days = (binsize_minutes / (60*24))
        orb_bd = phase_bin_magseries(
            x_fold[mask], y0, binsize=binsize_days, minbinelems=3
        )
        ax.scatter(
            scale_x(orb_bd['binnedphases']), 1e3*(orb_bd['binnedmags']), color='k',
            s=BINMS,
            alpha=1, zorder=1002#, linewidths=0.2, edgecolors='white'
        )


        #For plotting
        lc_modx = x_fold[mask]
        lc_mody = lc_mod[mask][np.argsort(lc_modx)]
        if from_trace==True:
            lc_mod_lo = lc_mod_band[0][mask][np.argsort(lc_modx)]
            lc_mod_hi = lc_mod_band[1][mask][np.argsort(lc_modx)]
        lc_modx = np.sort(lc_modx)

        ax.plot(scale_x(lc_modx), 1e3*lc_mody, color="C4", label="transit model",
                lw=1, zorder=1001, alpha=1)

        if from_trace==True:
            art = ax.fill_between(
                scale_x(lc_modx), 1e3*lc_mod_lo, 1e3*lc_mod_hi, color="C4",
                alpha=0.5, zorder=1000
            )
            art.set_edgecolor("none")

        ax.set_xticklabels([])

        #
        # residual axis
        #
        ax = axd[str(q_ix+4)]
        #ax.errorbar(24*x_fold[mask], 1e3*(y[mask] - gp_mod[mask] - lc_mod[mask]), yerr=1e3*_yerr[mask],
        #            color="darkgray", fmt='.', elinewidth=0.2, capsize=0,
        #            markersize=1, rasterized=True)

        binsize_days = (binsize_minutes / (60*24))
        y1 = (
            y[mask]-gp_mod[mask]-lc_mod[mask] -
            np.nanmedian(y[mask]-gp_mod[mask]-lc_mod[mask])
        )
        orb_bd = phase_bin_magseries(
            x_fold[mask], y1, binsize=binsize_days, minbinelems=3
        )
        ax.scatter(
            scale_x(orb_bd['binnedphases']), 1e3*(orb_bd['binnedmags']), color='k',
            s=BINMS, alpha=1, zorder=1002#, linewidths=0.2, edgecolors='white'
        )
        ax.axhline(0, color="C4", lw=1, ls='-', zorder=1000)

        props = dict(boxstyle='square', facecolor='white', alpha=0.5,
                     pad=0.15, linewidth=0)
        ax.text(0.03,0.07,txt,
                transform=ax.transAxes,
                ha='left',va='bottom', color='k', fontsize='xx-small', bbox=props)

        if from_trace==True:
            sigma = 30
            print(f'WRN! Smoothing plotted by by sigma={sigma}')
            _g =  lambda a: gaussian_filter(a, sigma=sigma)
            art = ax.fill_between(
                scale_x(lc_modx), 1e3*_g(lc_mod_hi-lc_mody), 1e3*_g(lc_mod_lo-lc_mody),
                color="C4", alpha=0.5, zorder=1000
            )
            art.set_edgecolor("none")

        #ax.set_xlabel("Hours from mid-transit")
        if fullxlim:
            ax.set_xlabel("Days from mid-transit")

    fig.text(-0.01,0.5, 'Relative flux [ppt]', va='center',
             rotation=90)
    fig.text(0.5,-0.01, 'Hours from mid-transit', va='center', ha='center',
             rotation=0)

    for k,a in axd.items():
        if not fullxlim:
            a.set_xlim(-0.4*24,0.4*24)
        else:
            a.set_xlim(-_per/2,_per/2)
        if isinstance(ylimd, dict):
            a.set_ylim(ylimd[k])
        else:
            # sensible default guesses
            _y = 1e3*(y[mask]-gp_mod[mask])
            axd['A'].set_ylim(get_ylimguess(_y))
            _y = 1e3*(y[mask] - gp_mod[mask] - lc_mod[mask])
            axd['B'].set_ylim(get_ylimguess(_y))

        format_ax(a)

    # NOTE: alt approach: override it as the stddev of the residuals. This is
    # dangerous, b/c if the errors are totally wrong, you might not know.
    if do_hacky_reprerror:
        sel = np.abs(orb_bd['binnedphases']*24)>3 # at least 3 hours from mid-transit
        binned_err = 1e3*np.nanstd((orb_bd['binnedmags'][sel]))
        print(f'WRN! Overriding binned unc as the residuals. Binned_err = {binned_err:.4f} ppt')
        #print(f'{_e:.2f}, {errorfactor*_e:.2f}')

    for k in [str(x) for x in range(0,4)]:
        _x,_y = 0.8*max(axd[k].get_xlim()), 0.7*min(axd[k].get_ylim())
        axd[k].errorbar(
            _x, _y, yerr=binned_err,
            fmt='none', ecolor='black', alpha=1, elinewidth=0.5, capsize=2,
            markeredgewidth=0.5
        )

    for k in [str(x) for x in range(4,8)]:
        _x,_y = 0.8*max(axd[k].get_xlim()), 0.6*min(axd[k].get_ylim())
        axd[k].errorbar(
            _x, _y, yerr=binned_err,
            fmt='none', ecolor='black', alpha=1, elinewidth=0.5, capsize=2,
            markeredgewidth=0.5
        )

    for k in [str(x) for x in range(0,8)]:
        if k in ['0','4','2','6']:
            continue
        else:
            axd[k].set_yticklabels([])
    for k in [str(x) for x in range(0,8)]:
        if k in ['6','7']:
            continue
        else:
            axd[k].set_xticklabels([])

    fig.tight_layout(h_pad=0.2, w_pad=0.2)

    savefig(fig, outpath, dpi=350)
    plt.close('all')


def plot_galex(outdir,
               clusters=['δ Lyr cluster', 'IC 2602', 'Pleiades'],
               extinctionmethod='gaia2018', show100pc=0,
               cleanhrcut=1, overplotkep1627=0, xval='bpmrp0'):
    """
    NOTE: we assume plot_hr has been run, to get the base data for the clusters.

    xval = 'bpmrp0' or 'jmk'
    """

    assert xval in ['bpmrp0', 'jmk']
    if xval == 'bpmrp0':
        get_xval = (
            lambda _df: np.array(
                _df['phot_bp_mean_mag_corr'] - _df['phot_rp_mean_mag_corr']
            )
        )
    elif xval == 'jmk':
        get_xval = lambda _df: np.array(
                _df['j_mag'] - _df['k_mag']
        )
    get_yval = lambda _df: np.array(
            _df['nuv_mag'] - _df['j_mag']
    )

    # make plot and get data
    plt.close('all')
    set_style()

    fig, ax = plt.subplots(figsize=(4,3))

    s = 2
    if overplotkep1627:
        # made by plot_hr
        outpath = os.path.join(
            RESULTSDIR, 'tables', 'deltalyr_kc19_cleansubset_withreddening.csv'
        )
        df = pd.read_csv(outpath)
        if cleanhrcut:
            df = df[get_clean_gaia_photometric_sources(df)]
        sel = (df.source_id == 2103737241426734336)
        sdf = df[sel]

        ra,dec = np.array([float(sdf.ra)]), np.array([float(sdf.dec)])
        idstring = 'kepler1627'
        starid = np.array(['2103737241426734336'])
        galex_df = get_galex_data(ra, dec, starid, idstring, verbose=True)
        twomass_df = get_2mass_data(ra, dec, starid, idstring, verbose=True)
        pdf = pd.concat((galex_df,twomass_df), axis=1)
        _df = pd.concat((sdf.reset_index(), pdf.reset_index()), axis=1)

        ax.plot(
            get_xval(_df), get_yval(_df),
            alpha=1, mew=0.5,
            zorder=9001, label='Kepler 1627',
            markerfacecolor='yellow', markersize=11, marker='*',
            color='black', lw=0
        )

    if 'Pleiades' in clusters:
        outpath = os.path.join(
            RESULTSDIR, 'tables',
            f'Pleiades_withreddening_{extinctionmethod}.csv'
        )
        _df = pd.read_csv(outpath)
        if cleanhrcut:
            _df = _df[get_clean_gaia_photometric_sources(_df)]

        idstring = 'Pleiades_CG18'
        starid = np.array(_df.source_id).astype(str)
        ra = np.array(_df.ra)
        dec = np.array(_df.dec)
        galex_df = get_galex_data(ra, dec, starid, idstring, verbose=True)
        twomass_df = get_2mass_data(ra, dec, starid, idstring, verbose=True)
        pdf = pd.concat((galex_df,twomass_df), axis=1)
        __df = pd.concat((_df.reset_index(), pdf.reset_index()), axis=1)

        ax.scatter(
            get_xval(__df), get_yval(__df), c='deepskyblue', alpha=1, zorder=1,
            s=s, rasterized=False, label='Pleiades', marker='o',
            edgecolors='k', linewidths=0.1
        )

    if 'IC 2602' in clusters:
        outpath = os.path.join(
            RESULTSDIR, 'tables', f'IC_2602_withreddening_{extinctionmethod}.csv'
        )
        _df = pd.read_csv(outpath)
        if cleanhrcut:
            _df = _df[get_clean_gaia_photometric_sources(_df)]

        idstring = 'IC2602_CG18'
        starid = np.array(_df.source_id).astype(str)
        ra = np.array(_df.ra)
        dec = np.array(_df.dec)
        galex_df = get_galex_data(ra, dec, starid, idstring, verbose=True)
        twomass_df = get_2mass_data(ra, dec, starid, idstring, verbose=True)
        pdf = pd.concat((galex_df,twomass_df), axis=1)
        __df = pd.concat((_df.reset_index(), pdf.reset_index()), axis=1)

        ax.scatter(
            get_xval(__df), get_yval(__df), c='orange', alpha=1, zorder=2,
            s=s, rasterized=False, label='IC 2602', marker='o',
            edgecolors='k', linewidths=0.1
        )

    if 'δ Lyr cluster' in clusters:
        outpath = os.path.join(
            RESULTSDIR, 'tables', 'deltalyr_kc19_cleansubset_withreddening.csv'
        )
        df = pd.read_csv(outpath)
        if cleanhrcut:
            df = df[get_clean_gaia_photometric_sources(df)]

        idstring = 'delLyrClusterKinematic_KC19'
        starid = np.array(df.source_id).astype(str)
        ra = np.array(df.ra)
        dec = np.array(df.dec)
        galex_df = get_galex_data(ra, dec, starid, idstring, verbose=True)
        twomass_df = get_2mass_data(ra, dec, starid, idstring, verbose=True)
        pdf = pd.concat((galex_df,twomass_df), axis=1)
        __df = pd.concat((df.reset_index(), pdf.reset_index()), axis=1)

        ax.scatter(
            get_xval(__df), get_yval(__df), c='k', alpha=1, zorder=10,
            s=s, rasterized=False, label='δ Lyr cluster', marker='o',
            edgecolors='k', linewidths=0.1
        )

    if xval == 'bpmrp0':
        ax.set_xlabel('$(G_{\mathrm{BP}}-G_{\mathrm{RP}})_0$ [mag]')
    elif xval == 'jmk':
        ax.set_xlabel('J-K [mag]')
    ax.set_ylabel('NUV-J [mag]')
    ax.set_ylim(ax.get_ylim()[::-1])

    if len(clusters) > 1:
        ax.legend(fontsize='xx-small', loc='upper right', handletextpad=0.1)

    #
    # append SpTypes (ignoring reddening)
    #
    if xval == 'bpmrp0':
        from rudolf.priors import AVG_EBpmRp
        tax = ax.twiny()
        tax.set_xlabel('Spectral Type')

        xlim = ax.get_xlim()
        getter = (
            get_SpType_BpmRp_correspondence
        )
        sptypes, xtickvals = getter(
            ['A0V','F0V','G0V','K2V','K5V','M0V','M2V','M4V']
        )
        print(sptypes)
        print(xtickvals)

        xvals = np.linspace(min(xlim), max(xlim), 100)
        tax.plot(xvals, np.ones_like(xvals), c='k', lw=0) # hidden, but fixes axis.
        tax.set_xlim(xlim)
        ax.set_xlim(xlim)

        tax.set_xticks(xtickvals+AVG_EBpmRp)
        tax.set_xticklabels(sptypes, fontsize='x-small')

        tax.xaxis.set_ticks_position('top')
        tax.tick_params(axis='x', which='minor', top=False)
        tax.get_yaxis().set_tick_params(which='both', direction='in')

    # set naming options
    s = f'{xval}'
    if show100pc:
        s += f'_show100pc'
    if len(clusters) > 1:
        s += '_'+'_'.join(clusters).replace(' ','_')
    if cleanhrcut:
        s += f'_cleanhrcut'
    if extinctionmethod:
        s += f'_{extinctionmethod}'
    if overplotkep1627:
        s += '_overplotkep1627'

    outpath = os.path.join(outdir, f'galexcolor_{s}.png')

    savefig(fig, outpath, dpi=500)
