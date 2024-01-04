"""
plot_TEMPLATE

Gaia (CMDs, RUWEs)
    plot_ruwe_vs_apparentmag
    plot_skychart
    plot_XYZvtang
    plot_hr
    plot_kerr21_XY
    plot_kepclusters_skychart

Gaia + TESS + Kepler:
    plot_rotationperiod_vs_color

GALEX:
    plot_galex

Kepler phot:
    plot_keplerlc
        _plot_zoom_light_curve
    plot_koikeplerlc
    plot_flare_checker
        _get_detrended_flare_data
    plot_ttv
    plot_ttv_vs_local_slope
    plot_rotation_period_windowslider
    plot_flare_pair_time_distribution
    plot_phasedlc_quartiles

Spec:
    plot_lithium
    plot_halpha
    plot_koiyouthlines
    plot_rvactivitypanel

RM:
    plot_simulated_RM
    plot_RM
    plot_RM_and_phot
    plot_youthlines
    plot_rv_checks

Cep-Her:
    plot_CepHer_weights
    plot_CepHer_quicklook_tests
    plot_CepHerExtended_quicklook_tests
    plot_CepHer_XYZvtang_sky

General for re-use:
    multiline: iterate through a colormap with many lines.
    truncate_colormap: extract a subset of a colormap
    plot_full_kinematics: corner plot of ra,dec,plx,PM,rv
"""
import os, corner, pickle, inspect
from copy import deepcopy
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
from astropy.modeling import models, fitting

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

from earhart.physicalpositions import (
    calc_vl_vb_physical, append_physicalpositions, get_vl_lsr_corr
)

from rudolf.paths import DATADIR, RESULTSDIR, TABLEDIR
from rudolf.helpers import (
    get_deltalyr_kc19_gaia_data, get_simulated_RM_data,
    get_keplerfieldfootprint_dict, get_deltalyr_kc19_comovers,
    get_deltalyr_kc19_cleansubset, get_manually_downloaded_kepler_lightcurve,
    get_set1_koi7368,
    get_gaia_catalog_of_nearby_stars, get_clustermembers_cg18_subset,
    get_mutau_members, get_ScoOB2_members,
    get_alphaPer_members,
    get_BPMG_members,
    supplement_gaia_stars_extinctions_corrected_photometry,
    get_clean_gaia_photometric_sources, get_galex_data, get_2mass_data,
    get_rsg5_kc19_gaia_data, get_ronan_cepher_augmented
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
                  shownakedeye=0, showcomovers=0, showkepstars=0,
                  showkepclusters=0, showdellyrcluster=1 ):

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

    if showdellyrcluster and not showtess:
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

    if showkepclusters:
        cluster_names = ['NGC6819', 'NGC6791', 'NGC6811', 'NGC6866']
        cras = [295.33, 290.22, 294.34, 300.983]
        cdecs =[40.18, 37.77, 46.378, 44.158]
        cplxs = [0.356, 0.192, 0.870, 0.686]
        ages_gyr = [1, 8, 1, 0.7]
        for _n, _ra, _dec in zip(cluster_names, cras, cdecs):
            ax.scatter(
                _ra, _dec, c='C0', alpha=0.8, zorder=4, s=20, rasterized=True,
                linewidths=0.2, marker='o', edgecolors='C0'
            )
            bbox = dict(facecolor='white', alpha=0.9, pad=0, edgecolor='white')
            deltay = 0.4
            ax.text(_ra, _dec+deltay, _n, ha='center', va='bottom',
                    fontsize=4, bbox=bbox, zorder=4)

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

    if not showtess and showdellyrcluster:
        leg = ax.legend(loc='lower left', handletextpad=0.1,
                        fontsize='x-small', framealpha=0.9)

    if not showdellyrcluster:
        ax.set_xlim([304,274])
        ax.set_ylim([26,49])

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
    if showkepclusters:
        s += '_showkepclusters'
    if not showdellyrcluster:
        s += '_nodellyr'


    bn = inspect.stack()[0][3].split("_")[1]
    outpath = os.path.join(outdir, f'{bn}{s}.png')
    savefig(f, outpath, dpi=400)


def plot_XYZvtang(outdir, show_1627=0, save_candcomovers=1, save_allphys=1,
                  show_comovers=0, show_sun=0, orientation=None, show_7368=0,
                  show_allknown=0, show_rsg5=0, show_set1=0):

    plt.close('all')
    set_style()

    # NOTE: assumes plot_XYZvtang has already been run
    df_sel = get_deltalyr_kc19_cleansubset()

    _, df_edr3, trgt_df = get_deltalyr_kc19_gaia_data()
    if show_7368 or show_set1:
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
    if show_7368 or show_set1:
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

        if (show_comovers and show_7368) or show_set1:
            axd[k].scatter(
                set1_df[set1_df.parallax_over_error>20][xv],
                set1_df[set1_df.parallax_over_error>20][yv], c='lime', alpha=1,
                zorder=8, s=2, edgecolors='none', rasterized=True, marker='.',
                label='(needs work) CH-2'
            )

        if show_allknown:
            namelist = ['Kepler-1627 A', 'KOI-7368', 'KOI-7913 A', 'Kepler-1643']
            markers = ['P', 'v', 'X', 's']
            # lime: CH-2 (KOI-7913, KOI-7368)
            # #ff6eff: RSG5 (Kepler-1643)
            # gray/black: del Lyr cluster (Kepler-1627)
            mfcs = ['white', 'lime', 'lime', '#ff6eff']

            # drops KOI-7913 B
            koi_df_dict = {k:v for k,v in koi_df_dict.items() if k in namelist}

            for mfc, marker, (name,_df) in zip(mfcs, markers, koi_df_dict.items()):
                axd[k].plot(
                    _df[xv], _df[yv], alpha=1, mew=0.5,
                    zorder=42, label=name, markerfacecolor=mfc,
                    markersize=12, marker=marker, color='black', lw=0
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
                df_rsg5_edr3[xv], df_rsg5_edr3[yv], c='#ff6eff', alpha=1, zorder=6,
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
    if show_set1:
        s += f"_showset1"

    bn = inspect.stack()[0][3].split("_")[1]
    outpath = os.path.join(outdir, f'{bn}{s}.png')
    savefig(fig, outpath, dpi=400)


def plot_koikeplerlc(outdir, xlim=[200,300]):
    """
    Kepler-1643, KOI-7368, KOI-7913, Kepler-1627.
    Each in a row.
    """

    EPHEMDICT = {
        'KOI_7368': {'t0': 2454970.06-2454833, 'per': 6.842939, 'tdur':2.54/24, 'n_tdurs':3.},
        'KOI_7913': {'t0': 2454987.513-2454833, 'per': 24.2783801, 'tdur':4.564/24, 'n_tdurs':2.5},
        'Kepler_1627': {'t0': 120.790531, 'per': 7.20280608, 'tdur':2.841/24, 'n_tdurs':3.5},
        'Kepler_1643': {'t0': 2454967.381-2454833, 'per': 5.34264143, 'tdur':2.401/24, 'n_tdurs':3.5},
    }

    starids = [
        'Kepler_1643', 'KOI_7368', 'KOI_7913', 'Kepler_1627'
    ]

    #
    # make plot
    #
    plt.close('all')
    set_style()

    fig, axs = plt.subplots(figsize=(6,2.7), nrows=4, ncols=1)

    # get initial offset
    x0s = []
    times, fluxs = [], []
    for ix, starid in enumerate(starids):

        if starid == 'Kepler_1627':
            lctype = 'longcadence'
        elif starid in ['KOI_7368', 'KOI_7913', 'Kepler_1643']:
            lctype = starid
        time, flux, flux_err, qual, texp = (
            get_manually_downloaded_kepler_lightcurve(
                lctype=lctype, norm_zero=1
            )
        )

        # NOTE: we have an abundance of data -> drop all non-zero quality flags.
        sel = (qual == 0)
        time, flux, flux_err, texp = time[sel], flux[sel], flux_err[sel], texp

        x0s.append(np.nanmin(time))
        times.append(time)
        fluxs.append(flux)

    x0 = np.nanmin(x0s)
    print(f'Got x0={x0}')

    # do the plotting
    for ix, starid in enumerate(starids):

        ax = axs[ix]

        x, y = times[ix], fluxs[ix]

        ax.scatter(x-x0, 1e3*(y), c="k", s=0.5, rasterized=True, linewidths=0,
                   zorder=42)
        ax.set_xlim(xlim)

        # Star/Planet name
        txt = starid.replace('_','-')
        props = dict(boxstyle='square', facecolor='white', alpha=0.95, pad=0.15,
                     linewidth=0)
        ax.text(0.98, 0.04, txt, transform=ax.transAxes, ha='right',va='bottom',
                color='k', zorder=43, fontsize='small', bbox=props)


        # Ephemeris
        t0 = EPHEMDICT[starid]['t0'] - x0
        period = EPHEMDICT[starid]['per']
        epochs = np.arange(-200,400,1)
        tra_times = t0 + period*epochs

        ax.set_ylim([-90,90])

        ymin, ymax = ax.get_ylim()
        ax.vlines(
            tra_times, ymin, ymax, colors='darkgray', alpha=1,
            linestyles=':', zorder=-2, linewidths=0.6
        )
        ax.set_ylim((ymin, ymax))

    for ax in axs[:-1]:
        ax.set_xticklabels([])

    fig.text(-0.01,0.5, 'Relative flux [ppt]', va='center', rotation=90)
    fig.text(0.5,-0.01, 'Days from start', ha='center')

    fig.tight_layout(h_pad=0.0)

    # set naming options
    s = f'_xlim{str(xlim[0]).zfill(4)}_{str(xlim[1]).zfill(4)}'

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
        time, flux, flux_err, qual, texp = get_manually_downloaded_kepler_lightcurve()
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
        ax.scatter(x[mask][g]-x0, 1e3*(y[mask][g]-1-gp_mod[g]), c="k", s=0.5,
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
            get_manually_downloaded_kepler_lightcurve(lctype='shortcadence')
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




def plot_rotation_period_windowslider(outdir, starid):

    from timmy.rotationperiod import measure_rotation_period_and_unc

    modelid = 'gptransit'
    # get data
    datasets = OrderedDict()
    if starid in ['KOI_7368', 'KOI_7913', 'Kepler_1643']:
        time, flux, flux_err, qual, texp = (
            get_manually_downloaded_kepler_lightcurve(lctype=starid+'_byquarter')
        )
    else:
        time, flux, flux_err, qual, texp = (
            get_manually_downloaded_kepler_lightcurve(lctype='longcadence_byquarter')
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
    overplotkep1627=0, overplotkoi7368=0, getstellarparams=0,
    show_allknown=0, overplotkep1643=0, overplotkoi7913=0,
    overplotkoi7913b=0, darkcolors=0, tinylims=0
):
    """
    clusters: ['$\delta$ Lyr cluster', 'IC 2602', 'Pleiades']
    """

    set_style()
    if tinylims:
        set_style('science')

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
        # note: standard...
        # f = plt.figure(figsize=(3,4.5))
        f = plt.figure(figsize=(4,4))
        if tinylims:
            f = plt.figure(figsize=(1.3,1.1))
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
        s = 9
    if tinylims:
        s = 1

    if 'δ Lyr cluster' in clusters:
        # mixed rasterizing along layers b/c we keep the loading times nice
        l0 = '$\delta$ Lyr candidates'
        ax.scatter(
            get_xval(df), get_yval(df), c='k', alpha=1, zorder=3,
            s=0.8*s, rasterized=False, linewidths=0.1, label=l0, marker='o',
            edgecolors='k'
        )

    if 'del-Lyr-clean' in clusters:
        outpath = os.path.join(
            RESULTSDIR, 'tables', f'del-Lyr_withreddening_{extinctionmethod}.csv'
        )
        if not os.path.exists(outpath):
            MANUALDIR = '/Users/luke/Dropbox/proj/ldb/data/Cep-Her'
            csvpath = os.path.join(MANUALDIR, 'delLyr.csv')
            _df = pd.read_csv(csvpath)
            _df = supplement_gaia_stars_extinctions_corrected_photometry(
                _df, extinctionmethod=extinctionmethod,
                savpath=os.path.join(RESULTSDIR,'tables','del-Lyr_stilism.csv')
            )
            _df.to_csv(outpath, index=False)
        _df = pd.read_csv(outpath)
        #if cleanhrcut:
        #    _df = _df[get_clean_gaia_photometric_sources(_df)]
        if reddening_corr:
            print('del-lyr (hard cuts)')
            print(_df['reddening[mag][stilism]'].describe())

        l0 = '$\delta$ Lyr candidates'
        ax.scatter(
            get_xval(_df), get_yval(_df), c='white', alpha=1, zorder=10,
            s=s, rasterized=False, label='$\delta$ Lyr', marker='o',
            edgecolors='k', linewidths=0.5
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

    if 'α Per' in clusters:
        # already done (trust the dr3 de-reddening)
        #outpath = os.path.join(
        #    RESULTSDIR, 'tables', f'alpha-Per_withreddening_{extinctionmethod}.csv'
        #)
        #if not os.path.exists(outpath):
        #    import IPython; IPython.embed()
        #    _df = supplement_gaia_stars_extinctions_corrected_photometry(
        #        _df, extinctionmethod=extinctionmethod,
        #        savpath=os.path.join(RESULTSDIR,'tables','alpha-Per_stilism.csv')
        #    )
        #    _df.to_csv(outpath, index=False)
        #_df = pd.read_csv(outpath)
        _df = get_alphaPer_members()
        if cleanhrcut:
            _df = _df[get_clean_gaia_photometric_sources(_df)]
        if reddening_corr:
            print('alpha-Per')
            print(_df['reddening[mag][stilism]'].describe())

        ax.scatter(
            get_xval(_df), get_yval(_df), c='red', alpha=1, zorder=10,
            s=s, rasterized=False, label='α Per', marker='o',
            edgecolors='k', linewidths=0.1
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
            get_xval(_df), get_yval(_df), c='lime', alpha=1, zorder=9000,
            s=1.5*s, rasterized=False, label='CH-2 candidates', marker='D',
            edgecolors='k', linewidths=0.2
        )

    if show_allknown or overplotkoi7913 or overplotkep1643 or overplotkoi7913b:
        _, _, _, koi_df_dict = get_deltalyr_kc19_gaia_data(return_all_targets=1)

        namelist = ['Kepler-1627 A', 'KOI-7368', 'KOI-7913 A', 'KOI-7913 B', 'Kepler-1643']
        markers = ['P', 'v', 'X', 'X', 's']
        # lime: CH-2 (KOI-7913, KOI-7368)
        # #ff6eff: RSG5 (Kepler-1643)
        # gray/black: del Lyr cluster (Kepler-1627)
        if not darkcolors:
            mfcs = ['white', 'lime', 'lime', 'lime', '#ff6eff']
        else:
            mfcs = ['white', 'lime', 'lime', 'lime', "#BDD7EC"]

        for mfc, marker, (name,_kdf) in zip(mfcs, markers, koi_df_dict.items()):

            cachepath = os.path.join(RESULTSDIR,'tables',f'{name}_stilism.csv')
            if not os.path.exists(cachepath):
                _kdf = supplement_gaia_stars_extinctions_corrected_photometry(
                    _kdf, extinctionmethod=extinctionmethod, savpath=cachepath
                )
                _kdf.to_csv(cachepath, index=False)
            _kdf = pd.read_csv(cachepath)

            if show_allknown:
                edgecolor = 'black' if not darkcolors else 'white'
                ax.plot(
                    get_xval(_kdf), get_yval(_kdf),
                    alpha=1, mew=0.5, zorder=9001, label=name, markerfacecolor=mfc,
                    markersize=11, marker=marker, color=edgecolor, lw=0
                )
            if (
                (overplotkoi7368 and name == 'KOI-7368')
                or
                (overplotkoi7913 and 'KOI-7913 A' in name)
                or
                (overplotkoi7913b and 'KOI-7913 B' in name)
                or
                (overplotkep1643 and name == 'Kepler-1643')
            ):
                edgecolor = 'black' if not darkcolors else 'white'
                ax.plot(
                    get_xval(_kdf), get_yval(_kdf),
                    alpha=1, mew=0.5, zorder=9001, label=name, markerfacecolor=mfc,
                    markersize=10, marker=marker, color=edgecolor, lw=0
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

    if 'RSG5' in clusters:
        # crappy selection based on KC19
        raise NotImplementedError('deprecated')

    if 'RSG-5' in clusters:
        # selection based on Kerr clustering and XYZ/vl/vb cuts
        outpath = os.path.join(
            RESULTSDIR, 'tables', f'RSG-5_auto_withreddening_{extinctionmethod}.csv'
        )
        if not os.path.exists(outpath):
            readpath = os.path.join(RESULTSDIR, 'CepHer_XYZvtang_sky',
                                    'RSG-5_auto_XYZ_vl_vb_cut.csv')
            df_rsg5_edr3 = pd.read_csv(readpath)

            _gdf = given_source_ids_get_gaia_data(
                nparr(df_rsg5_edr3['source_id']).astype(np.int64),
                'RSG-5_auto_rudolf', n_max=10000, overwrite=False,
                enforce_all_sourceids_viable=True, savstr='',
                whichcolumns='*', gaia_datarelease='gaiaedr3'
            )
            assert len(df_rsg5_edr3) == len(_gdf)
            _df = supplement_gaia_stars_extinctions_corrected_photometry(
                _gdf, extinctionmethod=extinctionmethod,
                savpath=os.path.join(RESULTSDIR,'tables','RSG-5_auto_stilism.csv')
            )
            _df.to_csv(outpath, index=False)
        _df = pd.read_csv(outpath)
        if cleanhrcut:
            _df = _df[get_clean_gaia_photometric_sources(_df)]
        if reddening_corr:
            print('RSG5')
            print(_df['reddening[mag][stilism]'].describe())

        #c='#ffa873', # orange
        # magenta, else light blue
        c = '#ff6eff' if not darkcolors else "#BDD7EC"
        edgecolors = 'k' if not darkcolors else 'white'
        ax.scatter(
            get_xval(_df), get_yval(_df),
            c=c,
            alpha=1, zorder=10,
            s=1.6*s, rasterized=False, label='RSG-5 candidates', marker='o',
            edgecolors=edgecolors, linewidths=0.2
        )


    if 'RSG-5-clean' in clusters:

        outpath = os.path.join(
            RESULTSDIR, 'tables', f'RSG-5-clean_withreddening_{extinctionmethod}.csv'
        )
        if not os.path.exists(outpath):
            MANUALDIR = '/Users/luke/Dropbox/proj/ldb/data/Cep-Her'
            csvpath = os.path.join(MANUALDIR, 'rsg5.csv')
            _df = pd.read_csv(csvpath)
            _df = supplement_gaia_stars_extinctions_corrected_photometry(
                _df, extinctionmethod=extinctionmethod,
                savpath=os.path.join(RESULTSDIR,'tables','rsg5-clean_stilism.csv')
            )
            _df.to_csv(outpath, index=False)
        _df = pd.read_csv(outpath)
        #if cleanhrcut:
        #    _df = _df[get_clean_gaia_photometric_sources(_df)]
        if reddening_corr:
            print('rsg-5 (hard cuts)')
            print(_df['reddening[mag][stilism]'].describe())

        #c='#ffa873', # orange
        # magenta, else light blue
        c = '#ff6eff' if not darkcolors else "#BDD7EC"
        edgecolors = 'k' if not darkcolors else 'white'
        ax.scatter(
            get_xval(_df), get_yval(_df),
            c=c,
            alpha=1, zorder=10,
            s=1.6*s, rasterized=False, label='RSG-5', marker='o',
            edgecolors=edgecolors, linewidths=0.2
        )

    if 'Cep Foregnd' in clusters:

        outpath = os.path.join(
            RESULTSDIR, 'tables', f'CepFgnd-clean_withreddening_{extinctionmethod}.csv'
        )
        if not os.path.exists(outpath):
            MANUALDIR = '/Users/luke/Dropbox/proj/ldb/data/Cep-Her'
            csvpath = os.path.join(MANUALDIR, 'ch01_cep.csv')
            _df = pd.read_csv(csvpath)
            _df = supplement_gaia_stars_extinctions_corrected_photometry(
                _df, extinctionmethod=extinctionmethod,
                savpath=os.path.join(RESULTSDIR,'tables','ch01_cep-clean_stilism.csv')
            )
            _df.to_csv(outpath, index=False)
        _df = pd.read_csv(outpath)
        #if cleanhrcut:
        #    _df = _df[get_clean_gaia_photometric_sources(_df)]
        if reddening_corr:
            print('ch01/cep clean (hard cuts)')
            print(_df['reddening[mag][stilism]'].describe())

        #c='#ffa873', # orange
        # magenta, else light blue
        c = 'cyan'
        edgecolors = 'k' if not darkcolors else 'white'
        ax.scatter(
            get_xval(_df), get_yval(_df),
            c=c,
            alpha=1, zorder=10,
            s=1.6*s, rasterized=False, label='Cep Foreground', marker='o',
            edgecolors=edgecolors, linewidths=0.2
        )



    if 'CH-2' in clusters:
        # selection based on Kerr clustering and XYZ/vl/vb cuts
        outpath = os.path.join(
            RESULTSDIR, 'tables', f'CH-2_auto_withreddening_{extinctionmethod}.csv'
        )
        if not os.path.exists(outpath):
            readpath = os.path.join(RESULTSDIR, 'CepHer_XYZvtang_sky',
                                    'CH-2_auto_XYZ_vl_vb_cut.csv')
            df_edr3 = pd.read_csv(readpath)

            _gdf = given_source_ids_get_gaia_data(
                nparr(df_edr3['source_id']).astype(np.int64),
                'CH-2_auto_rudolf',
                n_max=10000, overwrite=False,
                enforce_all_sourceids_viable=True, savstr='', whichcolumns='*',
                gaia_datarelease='gaiaedr3'
            )
            assert len(df_edr3) == len(_gdf)
            _df = supplement_gaia_stars_extinctions_corrected_photometry(
                _gdf, extinctionmethod=extinctionmethod,
                savpath=os.path.join(RESULTSDIR,'tables','CH-2_auto_stilism.csv')
            )
            _df.to_csv(outpath, index=False)
        _df = pd.read_csv(outpath)
        if cleanhrcut:
            _df = _df[get_clean_gaia_photometric_sources(_df)]
        if reddening_corr:
            print('CH-2')
            print(_df['reddening[mag][stilism]'].describe())

        ax.scatter(
            get_xval(_df), get_yval(_df), c='lime', alpha=1, zorder=9000,
            s=1.8*s, rasterized=False, label='CH-2 candidates', marker='D',
            edgecolors='k', linewidths=0.2
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
            get_xval(_df), get_yval(_df), #c='deepskyblue',
            c='lightskyblue',
            alpha=1, zorder=1,
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
        if smalllims:
            vmin, vmax = 10, 1000
        elif tinylims:
            vmin, vmax = 200, 5000
        else:
            vmin, vmax = 25, 2500

        norm = ImageNormalize(vmin=vmin, vmax=vmax,
                              stretch=LogStretch())

        cmap = "Greys" if not darkcolors else "Greys_r"
        density = ax.scatter_density(_x[s], _y[s], cmap=cmap, norm=norm)

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
                        print("overplotkoi7368")
                        sel = (mstar > 0.83) & (mstar < 0.93)
                    elif overplotkep1643:
                        print("overplotkep1643")
                        sel = (mstar > 0.80) & (mstar < 0.90)
                    elif overplotkoi7913:
                        print("overplotkoi7913")
                        sel = (mstar > 0.70) & (mstar < 0.80)
                    elif overplotkoi7913b:
                        print("overplotkoi7913b")
                        sel = (mstar > 0.65) & (mstar < 0.75)
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
                    elif overplotkep1643:
                        sel = (
                            (np.abs(iso_df.logAge - la) < 0.01) &
                            (iso_df.Mass > 0.80) &
                            (iso_df.Mass < 0.90)
                        )
                    elif overplotkoi7913:
                        sel = (
                            (np.abs(iso_df.logAge - la) < 0.01) &
                            (iso_df.Mass > 0.70) &
                            (iso_df.Mass < 0.80)
                        )
                    elif overplotkoi7913b:
                        sel = (
                            (np.abs(iso_df.logAge - la) < 0.01) &
                            (iso_df.Mass > 0.65) &
                            (iso_df.Mass < 0.75)
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
    if tinylims:
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # 6 8 10
        # 1 2 3

    ylim = ax.get_ylim()
    ax.set_ylim((max(ylim),min(ylim)))

    if len(clusters) > 1:
        ax.legend(fontsize='xx-small', loc='upper right', handletextpad=0.1,
                  borderaxespad=2.0, borderpad=0.8)

    if show100pc and 'phot_bp_mean_mag' in color0:
        ax.set_xlim([-1,4.5])
        ax.set_ylim((16, -3))
    elif show100pc and 'phot_g_mean_mag' in color0:
        ax.set_xlim([-0.2,2.0])
        ax.set_ylim((16, -3))

    if (smalllims and 'phot_bp_mean_mag' in color0) or (tinylims and 'phot_bp_mean_mag' in color0):
        ax.set_xlim([0.85,3.45])
        ax.set_ylim([11.5,5.0])
    elif smalllims and 'phot_bp_mean_mag' not in color0:
        raise NotImplementedError

    format_ax(ax)
    ax.tick_params(axis='x', which='both', top=False)
    if tinylims:
        ax.set_yticks([10,8,6])


    #
    # append SpTypes (ignoring reddening)
    #
    from rudolf.priors import AVG_EBpmRp
    tax = ax.twiny()
    #tax.set_xlabel('Spectral Type')

    xlim = ax.get_xlim()
    getter = (
        get_SpType_BpmRp_correspondence
        if 'phot_bp_mean_mag' in color0 else
        get_SpType_GmRp_correspondence
    )
    if not smalllims and not tinylims:
        sptypes, xtickvals = getter(
            ['A0V','F0V','G0V','K2V','K5V','M0V','M2V','M4V']
        )
    elif tinylims:
        sptypes, xtickvals = getter(
            ['K2V','M1V','M4V']
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

    if tinylims:
        tax.set_xticklabels([])

    if darkcolors:
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(axis='both', colors='white')

        tax.spines['bottom'].set_color('white')
        tax.spines['top'].set_color('white')
        tax.spines['left'].set_color('white')
        tax.spines['right'].set_color('white')
        tax.xaxis.label.set_color('white')
        tax.tick_params(axis='both', colors='white')

        f.patch.set_alpha(0)
        tax.patch.set_alpha(0)
        ax.patch.set_alpha(0)

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
    if len(clusters) >= 1:
        c0s += '_'+'_'.join(clusters).replace(' ','_')
    if cleanhrcut:
        c0s += f'_cleanhrcut'
    if extinctionmethod:
        c0s += f'_{extinctionmethod}'
    if smalllims:
        c0s += '_smalllims'
    if tinylims:
        c0s += '_tinylims'
    if overplotkep1627:
        c0s += '_overplotkep1627'
    if overplotkoi7368:
        c0s += '_overplotkoi7368'
    if overplotkep1643:
        c0s += '_overplotkep1643'
    if overplotkoi7913:
        c0s += '_overplotkoi7913'
    if overplotkoi7913b:
        c0s += '_overplotkoi7913b'
    if show_allknown:
        c0s += '_allknownkois'
    if darkcolors:
        c0s += "_darkcolors"

    outpath = os.path.join(outdir, f'hr{s}{c0s}.png')

    savefig(f, outpath, dpi=600)


def plot_rotationperiod_vs_color(outdir, runid, yscale='linear', cleaning=None,
                                 emph_binaries=False, refcluster_only=False,
                                 talk_aspect=0, xval_absmag=0,
                                 kinematic_selection=0,
                                 overplotkep1627=0, show_allknown=0,
                                 darkcolors=0, show_douglas=0):
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
        f, ax = plt.subplots(figsize=(4,4))

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(0.5)

    colordict = {
        'deltaLyrCluster': 'k',
        'CH-2': 'lime',
        'RSG-5': '#ff6eff' if not darkcolors else "#BDD7EC" # magenta/lightblue
        #'RSG-5': '#ffa873' # orange
    }
    sizedict = {
        'deltaLyrCluster': 8,
        'CH-2': 24,
        'RSG-5': 24
    }

    classes = ['pleiades', 'praesepe', f'{runid}']
    gray = 'gray' if not darkcolors else 'lightgray'
    colors = [gray, gray, colordict[runid]]
    zorders = [-3, -4, -1]
    markers = ['X', '+', 'o']
    praesepelw = 0.4 if not darkcolors else 0.4
    lws = [0., praesepelw, 0.35]
    mews= [0., 0.5, 2]
    _s = 3 if runid != 'VelaOB2' else 1.2
    pleiadessize = 15 if not darkcolors else 12
    ss = [pleiadessize, 12, sizedict[runid]]
    if runid == 'deltaLyrCluster':
        l = '$\delta$ Lyr candidates'
    elif runid in ['CH-2','RSG-5']:
        l = f'{runid} candidates'
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

        elif runid == 'deltaLyrCluster':
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

        elif runid in ['CH-2','RSG-5']:
            csvpath = os.path.join(TABLEDIR, 'RSG5_CH2_Prot_cleaned.csv')
            _df = pd.read_csv(csvpath)
            csvpath = os.path.join(DATADIR, 'Cep-Her',
                                   '20220311_Kerr_CepHer_Extended_Candidates_v0-result.csv')
            rdf = pd.read_csv(csvpath)
            _df['dr3_source_id'] = _df.dr3_source_id.astype(str)
            rdf['source_id'] = rdf.source_id.astype(str)
            _df = _df.merge(rdf, how='left', left_on='dr3_source_id',
                            right_on='source_id')

            # match Jason's keys against what I call these clusters
            if runid == 'CH-2':
                sel = (_df.cluster == 'CH-2')
            elif runid == 'RSG-5':
                sel = (_df.cluster == 'RSG-5')

            df = _df[sel]

            if runid == 'CH-2':
                # drop KOI-7913 A
                df = df[df.dr3_source_id != '2106235301785454208' ]
                # drop KOI-7913 A
                df = df[df.dr3_source_id != '2106235301785453824' ]
                # drop KOI-7368
                df = df[df.dr3_source_id != '2128840912955018368' ]

            elif runid == 'RSG-5':
                df = df[df.dr3_source_id != '2082142734285082368' ]

        if f'{runid}' not in _cls:
            key = '(BP-RP)0' if not xval_absmag else 'MGmag'
            xval = df[key]
            ykey = 'Prot'

        elif runid in ['CH-2', 'RSG-5']:
            xkey = '(BP-RP)0'
            xval = df[xkey]
            ykey = 'Prot_Adopted'
            cstr = '_corr'
            get_xval = lambda __df: np.array(
                __df['phot_bp_mean_mag'+cstr] - __df['phot_rp_mean_mag'+cstr]
            )

        else:
            ykey = 'period'
            if not xval_absmag:
                get_xval = lambda __df: np.array(
                    __df['phot_bp_mean_mag'+cstr] - __df['phot_rp_mean_mag'+cstr]
                )
                xval = get_xval(df)
            else:
                get_xval = lambda __df: np.array(
                    __df['phot_g_mean_mag'] + 5*np.log10(__df['parallax']/1e3) + 5
                )
                xval = get_xval(df)

        if refcluster_only and f'{runid}' in _cls:
            pass

        else:
            CUTOFF = 0.6
            if runid in ['CH-2','RSG-5'] and f'{runid}' in _cls:
                sel = (
                    (xval > CUTOFF) & (df.phot_g_mean_mag < 16)
                )
            else:
                sel = (xval > CUTOFF)
            edgecolors = 'k' if not darkcolors else 'white'
            ax.scatter(
                xval[sel],
                df[sel][ykey],
                c=_col, alpha=1, zorder=z, s=s, edgecolors=edgecolors,
                marker=m, linewidths=_lw, label=f"{l.replace('_',' ')}"
            )

        if runid in ['CH-2','RSG-5'] and f'{runid}' in _cls:
            # get a few quick statistics
            print(42*'-')
            print(f'For the cluster {runid}')
            if runid == 'CH-2':
                print('OMITTING 3 stars (KOI-7913, KOI-7368)')
            elif runid == 'RSG-5':
                print('OMITTING 1 star (Kepler-1643)')
            print(f'Started with {len(df[sel])} in {CUTOFF} < BP-RP0, and G < 16')
            print(f'and got {len(df[sel][~pd.isnull(df[sel][ykey])])} rotation period detections.')
            print(f'N={len(df[sel])} Detections and non-detections are as follows')
            print(df[sel][[
                'dr3_source_id', 'weight', '(BP-RP)0',
                'Prot_Adopted', 'Prot_TESS', 'Prot_ZTF']].sort_values(by='(BP-RP)0')
            )
            print(10*'#')
            print('Stars with Prot>10 days:')
            print(df[(df.Prot_Adopted > 10) & (df.phot_g_mean_mag < 16)][[
                'dr3_source_id', 'weight', '(BP-RP)0',
                'Prot_Adopted', 'Prot_TESS', 'Prot_ZTF']].sort_values(by='(BP-RP)0')
            )
            print(42*'-')

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

    if show_douglas:
        douglaspath = os.path.join(
            DATADIR, 'rotation', 'Douglas_PrivComm_zams.csv'
        )
        ddf = pd.read_csv(douglaspath)

        sel = (
            (
                (ddf.Cluster == "IC_2602")
                |
                (ddf.Cluster == "IC_2391")
            )
            &
            (ddf.Q1 == 0)
            &
            (ddf.CG_MemProb >= 0.3)
        )

        sddf = ddf[sel]

        EBmV = 0.07 # assumed 0.03 to 0.07 from Randich+2018; A_V=0.217 in Bouma+2020
        A_V = 0.217
        from earhart.extinction import AV_to_EBpmRp

        E_BpmRp = AV_to_EBpmRp(A_V)

        _xval = sddf.GAIAEDR3_BP - sddf.GAIAEDR3_RP - E_BpmRp
        _yval = sddf["Prot1"]

        ax.scatter(
            _xval,
            _yval,
            c='k', alpha=1, zorder=0, s=10, edgecolors='k',
            marker='o', linewidths=0.5, label=f"IC2602 & IC2391"
        )


    if show_allknown:
        _, _, _, koi_df_dict = get_deltalyr_kc19_gaia_data(return_all_targets=1)

        namelist = ['Kepler-1627 A', 'KOI-7368', 'KOI-7913 A', 'KOI-7913 B', 'Kepler-1643']
        markers = ['P', 'v', 'X', 'X', 's']
        # lime: CH-2 (KOI-7913, KOI-7368)
        # #ff6eff: RSG5 (Kepler-1643)
        # gray/black: del Lyr cluster (Kepler-1627)
        rsg5mfc = '#ff6eff' if not darkcolors else "#BDD7EC"
        mfcs = ['white', 'lime', 'lime', 'lime', rsg5mfc]

        from rudolf.starinfo import starinfodict as sd
        for mfc, marker, (name,_kdf) in zip(mfcs, markers, koi_df_dict.items()):

            cachepath = os.path.join(RESULTSDIR,'tables',f'{name}_stilism.csv')
            if not os.path.exists(cachepath):
                _kdf = supplement_gaia_stars_extinctions_corrected_photometry(
                    _kdf, extinctionmethod=extinctionmethod, savpath=cachepath
                )
                _kdf.to_csv(cachepath, index=False)
            _kdf = pd.read_csv(cachepath)

            Prot = sd[name]['Prot']

            if runid == 'CH-2' and name not in [
                'KOI-7368', 'KOI-7913 A', 'KOI-7913 B'
            ]:
                continue
            if runid == 'RSG-5' and name not in [
                'Kepler-1643'
            ]:
                continue
            if runid == 'deltaLyrCluster' and name not in [
                'Kepler-1627 A'
            ]:
                continue
            edgecolor = 'k' if not darkcolors else 'white'
            ax.plot(
                get_xval(_kdf), Prot,
                alpha=1, mew=0.5, zorder=-2, label=name, markerfacecolor=mfc,
                markersize=12, marker=marker, color=edgecolor, lw=0,
            )


    ax.set_ylabel('Rotation Period [days]', fontsize='medium')

    if not xval_absmag:
        ax.set_xlabel('($G_{\mathrm{BP}}-G_{\mathrm{RP}}$)$_0$ [mag]',
                      fontsize='medium')
        if runid in ['deltaLyrCluster', 'RSG-5', 'CH-2'] :
            ax.set_xlim((0.2, 2.7))
        else:
            ax.set_xlim((0.2, 3.6))
    else:
        ax.set_xlabel('Absolute $\mathrm{M}_{G}$ [mag]', fontsize='medium')
        ax.set_xlim((1.5, 10))

    if yscale == 'linear':
        ax.set_ylim((0,20))
    elif yscale == 'log':
        ax.set_ylim((0.05,13))
    else:
        raise NotImplementedError
    ax.set_yscale(yscale)

    format_ax(ax)

    ax.set_yticks([0, 5, 10, 15, 20])

    #
    # twiny for the SpTypes
    #
    tax = ax.twiny()
    #tax.set_xlabel('Spectral Type')

    xlim = ax.get_xlim()
    if runid in ['deltaLyrCluster', 'RSG-5', 'CH-2'] :
        splist = ['F0V','F5V','G0V','K0V','K3V','K5V','K7V','M0V','M1V','M2V','M3V']
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
    leg = ax.legend(loc=loc, handletextpad=0.3, fontsize='xx-small',
                    framealpha=1.0, borderaxespad=2.0, borderpad=0.8)

    if darkcolors:
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(axis='both', colors='white', which='minor')
        ax.tick_params(axis='both', colors='white', which='major')

        tax.spines['bottom'].set_color('white')
        tax.spines['top'].set_color('white')
        tax.spines['left'].set_color('white')
        tax.spines['right'].set_color('white')
        tax.xaxis.label.set_color('white')
        tax.tick_params(axis='both', colors='white')

        f.patch.set_alpha(0)
        tax.patch.set_alpha(0)
        ax.patch.set_alpha(0)

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
    if darkcolors:
        outstr += '_darkcolors'
    outstr += f'_{yscale}'
    outstr += f'_{cleaning}'
    if overplotkep1627:
        outstr += '_overplotkep1627'
    if refcluster_only:
        outstr += '_refclusteronly'
    if show_allknown:
        outstr += '_allknown'
    if show_douglas:
        outstr += '_douglas'
    outpath = os.path.join(outdir, f'{runid}_rotation{outstr}.png')
    savefig(f, outpath, dpi=600)



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
    lines = ['ca k', 'ca h', 'hα']
    globs = ['kep*bj*order07*', 'kep*bj*order07*', 'kep*ij*order00*']
    deltawav = 5
    xlims = [
        [3933.66-deltawav, 3933.66+deltawav], # ca k
        [3968.47-deltawav, 3968.47+deltawav], # ca h
        [6562.8-deltawav, 6562.8+deltawav], # halpa
    ]

    rvpath = os.path.join(datadir, 'spec', '20210809_rvs_template_v1298tau.csv')
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

        csvpaths = glob(os.path.join(datadir, 'rvactivity', g))
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
            wavs, diff_flxs, 24*(nparr(times)-np.min(times)), cmap='spectral',
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
    cb.ax.set_title('time [hours]', fontsize='xx-small')
    # axd['b'].text(0.725,0.955, '$t$ [days]',
    #         transform=axd['b'].transaxes,
    #         ha='right',va='top', color='k', fontsize='xx-small')
    cb.ax.tick_params(size=0, which='both') # remove the ticks
    axins1.xaxis.set_ticks_position("bottom")

    fig.text(-0.01,0.5, 'relative flux', va='center',
             rotation=90)
    fig.text(0.5,-0.01, 'wavelength [$\aa$]', va='center', ha='center', rotation=0)

    fig.tight_layout()

    # set naming options
    s = ''

    bn = inspect.stack()[0][3].split("_")[1]
    outpath = os.path.join(outdir, f'{bn}{s}.png')
    savefig(fig, outpath, dpi=400)


def plot_youthlines(outdir):

    # get data
    lines = ['Ca K', 'Ca H', 'Hα', 'Li-I']
    globs = ['Kep*bj*order07*', 'Kep*bj*order07*', 'Kep*ij*order00*', 'Kep*ij*order01*']
    deltawav = 5
    xlims = [
        [3933.66-deltawav, 3933.66+deltawav], # ca k
        [3968.47-deltawav, 3968.47+deltawav], # ca h
        [6562.8-deltawav, 6562.8+deltawav], # halpa
        [6707.8-deltawav, 6707.8+deltawav], # li6708
    ]

    # make plot
    plt.close('all')
    set_style()

    #fig, ax = plt.subplots(figsize=(4,3))
    fig = plt.figure(figsize=(4,3))
    axd = fig.subplot_mosaic(
        """
        01
        23
        """,
        gridspec_kw={
            #"width_ratios": [1, 1, 1, 1]
            #"height_ratios": [1, 3]
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

        axd[str(ix)].set_title(l)
        axd[str(ix)].set_xlim(xlim)

    fig.text(-0.01,0.5, 'Relative flux', va='center',
             rotation=90)
    fig.text(0.5,-0.01, r'Wavelength [$\AA$]', va='center', ha='center', rotation=0)

    fig.tight_layout()

    # set naming options
    s = ''

    bn = inspect.stack()[0][3].split("_")[1]
    outpath = os.path.join(outdir, f'{bn}{s}.png')
    savefig(fig, outpath, dpi=400)


def get_flx_wav_given_2d_and_target(flx_2d, wav_2d, target_wav):
    _preorder = np.argmin(np.abs(wav_2d - target_wav), axis=1)
    viable_orders = np.argwhere(
        (_preorder != wav_2d.shape[1]-1) & (_preorder != 0)
    )
    order = int(
        viable_orders[np.argmin(
            np.abs(_preorder[viable_orders] - wav_2d.shape[1]/2)
        )]
    )

    flx, wav = flx_2d[order, :], wav_2d[order, :]
    return flx, wav


def plot_koiyouthlines(outdir):

    lines = ['Hα', 'Li-I']
    starnames = ['Kepler-1627', 'KOI-7368', 'Kepler-1643', 'KOI-7913_A', 'KOI-7913_B']
    deltawav = 4
    xlims = [
        [6707.8-deltawav, 6707.8+deltawav], # li6708
        [6562.8-deltawav, 6562.8+deltawav], # halpha
    ]

    # make plot
    plt.close('all')
    set_style()

    fig, axs = plt.subplots(nrows=2, ncols=len(starnames), figsize=(7,3.5))

    from scipy.ndimage import gaussian_filter1d

    for ix, starname in enumerate(starnames):

        fitspaths = glob(os.path.join(
            DATADIR, 'koispectra', starname, 'ij*fits')
        )
        assert len(fitspaths) == 1
        spectrum_path = fitspaths[0]

        from cdips_followup.spectools import read_tres, read_hires

        if starname == 'KOI-7368':
            flx_2d, wav_2d = read_tres(spectrum_path)
            instrument = 'TRES'
        else:
            flx_2d, wav_2d = read_hires(
                spectrum_path, is_registered=0, return_err=0
            )
            instrument = 'HIRES'

        # upper: Li
        ax0 = axs[0,ix]
        # lower: Hα
        ax1 = axs[1,ix]

        norm = lambda x: x/np.nanmedian(x)
        fn = lambda x: gaussian_filter1d(x, sigma=2)

        flx, wav = get_flx_wav_given_2d_and_target(flx_2d, wav_2d, 6707.8)
        sel = ((xlims[0][0]-2) < wav) & (wav < xlims[0][1]+2)
        ax0.plot(
            wav[sel], fn(norm(flx[sel])), c='k', zorder=3, lw=0.2
        )

        flx, wav = get_flx_wav_given_2d_and_target(flx_2d, wav_2d, 6562.8)
        sel = (xlims[1][0]-2 < wav) & (wav < xlims[1][1]+2)
        ax1.plot(
            wav[sel], fn(norm(flx[sel])), c='k', zorder=3, lw=0.2
        )

        titlestr = starname.replace("_"," ")
        if starname == 'Kepler-1627':
            titlestr += ' A'
        ax0.set_title(titlestr, fontsize='small')
        ax0.set_xlim(xlims[0])
        ax1.set_xlim(xlims[1])

        props = dict(boxstyle='square', facecolor='white', alpha=0.8, pad=0.15,
                     linewidth=0)
        txt = 'Li-I'
        yval = 0.1 if 'KOI-7913' in starname else 0.7
        delta = 0.05 if starname not in ['Kepler-1627', 'KOI-7368'] else 0
        ax0.text(0.5+delta, yval, txt, transform=ax0.transAxes,
                 ha='right',va='bottom', color='k',
                 fontsize='x-small', bbox=props)
        txt = 'Hα'
        yval = 0.1 if 'KOI-7913' not in starname else 0.7
        ax1.text(0.9, yval, txt, transform=ax1.transAxes,
                 ha='right',va='bottom', color='k',
                 fontsize='x-small', bbox=props)

        from matplotlib.ticker import (
            MultipleLocator, FormatStrFormatter, AutoMinorLocator
        )
        if starname in ['Kepler-1627', 'KOI-7368']:
            ax0.yaxis.set_major_locator(MultipleLocator(0.2))
        else:
            ax0.yaxis.set_major_locator(MultipleLocator(0.1))
        ax0.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        if starname != 'KOI-7913_B':
            ax1.yaxis.set_major_locator(MultipleLocator(0.2))
        else:
            ax1.yaxis.set_major_locator(MultipleLocator(0.3))
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        #ax.xaxis.set_minor_locator(MultipleLocator(10))

        ax0.tick_params(axis='both', which='major', labelsize='x-small')
        ax1.tick_params(axis='both', which='major', labelsize='x-small')

    fig.text(-0.01,0.5, 'Relative flux', va='center',
             rotation=90)
    fig.text(0.5,-0.01, r'Wavelength [$\AA$]', va='center', ha='center',
             rotation=0)

    fig.tight_layout(w_pad=0.2, h_pad=0.5)

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
        time, flux, flux_err, qual, texp = get_manually_downloaded_kepler_lightcurve()
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


def plot_full_kinematics(df, outstr, outdir, galacticframe=0):
    """
    df: dataframe containing Gaia-key stars
    outstr: used as the head-string in the file name
    """

    if galacticframe:
        if not ('l' in df and 'b' in df):
            c = SkyCoord(ra=nparr(df.ra)*u.deg, dec=nparr(df.dec)*u.deg)
            df['l'] = c.galactic.l.value
            df['b'] = c.galactic.b.value

    plt.close('all')

    rvkey = (
        'dr2_radial_velocity' if 'dr2_radial_velocity' in df else 'radial_velocity'
    )

    if galacticframe:
        xkey, ykey = 'l', 'b'
        xl, yl = r'$l$ [deg]', r'$b$ [deg]'
    else:
        xkey, ykey = 'ra', 'dec'
        xl, yl = r'$\alpha$ [deg]', r'$\delta$ [deg]'

    params = [xkey, ykey, 'parallax', 'pmra', 'pmdec', rvkey]
    # whether to limit axis by 5/95th percetile
    qlimd = {
        xkey: 0, ykey: 0, 'parallax': 0, 'pmra': 0, 'pmdec': 0, rvkey: 0
    }
    # whether to limit axis by 99th percentile
    nnlimd = {
        xkey: 0, ykey: 0, 'parallax': 0, 'pmra': 0, 'pmdec': 0, rvkey: 0
    }
    ldict = {
        xkey: xl, ykey: yl,
        'parallax': r'$\pi$ [mas]', 'pmra': r"$\mu_{{\alpha'}}$ [mas/yr]",
        'pmdec':  r'$\mu_{{\delta}}$ [mas/yr]', rvkey: 'RV [km/s]'
    }

    nparams = len(params)
    f, axs = plt.subplots(figsize=(6,6), nrows=nparams-1, ncols=nparams-1)

    for i in range(nparams):
        for j in range(nparams):
            print(i,j)
            if j == nparams-1 or i == nparams-1:
                continue
            if j>i:
                axs[i,j].set_axis_off()
                continue

            xv = params[j]
            yv = params[i+1]
            print(i,j,xv,yv)

            alpha = 0.9
            axs[i,j].scatter(
                df[xv], df[yv], c='k', alpha=alpha, zorder=4, s=5,
                rasterized=True, label=outstr, marker='.', linewidths=0
            )

            # set the axis limits as needed
            if qlimd[xv]:
                xlim = (np.nanpercentile(nbhd_df[xv], 5),
                        np.nanpercentile(nbhd_df[xv], 95))
                axs[i,j].set_xlim(xlim)
            if qlimd[yv]:
                ylim = (np.nanpercentile(nbhd_df[yv], 5),
                        np.nanpercentile(nbhd_df[yv], 95))
                axs[i,j].set_ylim(ylim)
            if nnlimd[xv]:
                xlim = (np.nanpercentile(nbhd_df[xv], 1),
                        np.nanpercentile(nbhd_df[xv], 99))
                axs[i,j].set_xlim(xlim)
            if nnlimd[yv]:
                ylim = (np.nanpercentile(nbhd_df[yv], 1),
                        np.nanpercentile(nbhd_df[yv], 99))
                axs[i,j].set_ylim(ylim)


            # fix labels
            if j == 0 :
                axs[i,j].set_ylabel(ldict[yv], fontsize='small')
                if not i == nparams - 2:
                    # hide xtick labels
                    labels = [item.get_text() for item in axs[i,j].get_xticklabels()]
                    empty_string_labels = ['']*len(labels)
                    axs[i,j].set_xticklabels(empty_string_labels)

            if i == nparams - 2:
                axs[i,j].set_xlabel(ldict[xv], fontsize='small')
                if not j == 0:
                    # hide ytick labels
                    labels = [item.get_text() for item in axs[i,j].get_yticklabels()]
                    empty_string_labels = ['']*len(labels)
                    axs[i,j].set_yticklabels(empty_string_labels)

            if (not (j == 0)) and (not (i == nparams - 2)):
                # hide ytick labels
                labels = [item.get_text() for item in axs[i,j].get_yticklabels()]
                empty_string_labels = ['']*len(labels)
                axs[i,j].set_yticklabels(empty_string_labels)
                # hide xtick labels
                labels = [item.get_text() for item in axs[i,j].get_xticklabels()]
                empty_string_labels = ['']*len(labels)
                axs[i,j].set_xticklabels(empty_string_labels)

    f.tight_layout(h_pad=0.05, w_pad=0.05)

    axs[2,2].legend(loc='best', handletextpad=0.1, fontsize='medium', framealpha=0.7)
    leg = axs[2,2].legend(bbox_to_anchor=(0.8,0.8), loc="upper right",
                          handletextpad=0.1, fontsize='medium',
                          bbox_transform=f.transFigure)

    for ax in axs.flatten():
        format_ax(ax)

    s = '_'+outstr
    if galacticframe:
        s += f'_galactic'
    else:
        s += f'_icrs'

    outpath = os.path.join(outdir, f'full_kinematics{s}.png')

    savefig(f, outpath)


def plot_halpha(outdir, reference='TucHor'):

    # get data
    if reference != 'TucHor':
        raise NotImplementedError

    fitspath = os.path.join(
        DATADIR, 'cluster', 'Kraus_2014_AJ_147_146_TucHor_table2.fits'
    )
    hdul = fits.open(fitspath)
    df = Table(hdul[1].data).to_pandas()
    hdul.close()

    fitspath = os.path.join(
        DATADIR, 'cluster', 'Fang_2018_MNRAS_476_908_Pleiades_378.fits'
    )
    hdul = fits.open(fitspath)
    pdf = Table(hdul[1].data).to_pandas()
    hdul.close()

    #In [10]: Counter([t[0] for t in df['SpT']])
    #Out[10]: Counter({'M': 156, 'K': 42, 'F': 1, 'G': 6})
    sel = (df['SpT'].str.contains("M")) | (df['SpT'].str.contains("K"))
    sdf = df[sel]

    SpT_val = []
    for v in sdf.SpT:
        if v.startswith('M'):
            SpT_val.append(float(v[1:]))
        elif v.startswith('K'):
            SpT_val.append(float(v[1:])-10)
        else:
            raise NotImplementedError

    sdf['SpT_val'] = SpT_val

    # make plot
    plt.close('all')
    set_style()

    fig, ax = plt.subplots(figsize=(4,3))

    ax.scatter(
        sdf[sdf.Mm == 'Y'].SpT_val,
        sdf[sdf.Mm == 'Y'].EWHa,
        zorder=7,
        label='TucHor (Kraus+14)',
        c='k', marker='o', edgecolors='k', linewidths=0.3, s=6
    )
    # ax.scatter(
    #     sdf[sdf.Mm == 'N'].SpT_val,
    #     sdf[sdf.Mm == 'N'].EWHa,
    #     zorder=6,
    #     label='Field (Kraus+14)',
    #     c='white', marker='o', edgecolors='k', linewidths=0.3, s=4
    # )

    from cdips.utils.mamajek import get_interp_SpType_from_teff

    #for teff in np.arange(3600, 5300, 100):
    #    print(get_interp_SpType_from_teff(teff))

    sel_pleaides = (pdf.multi == 0)
    spdf = pdf[sel_pleaides]
    pl_teff = spdf.Teff
    pl_sptypes = [get_interp_SpType_from_teff(T, verbose=False) for T in pl_teff]

    pl_SpT_val = []
    for v in pl_sptypes:
        if v.startswith('M'):
            pl_SpT_val.append(float(v[1:-1]))
        elif v.startswith('K'):
            pl_SpT_val.append(float(v[1:-1])-10)
        else:
            pl_SpT_val.append(np.nan)
    spdf['SpT_val'] = pl_SpT_val

    np.random.seed(42)
    eps = np.random.normal(loc=0,scale=0.05,size=len(spdf))
    ax.scatter(
        spdf.SpT_val+eps,
        spdf.EW_Ha,
        zorder=6,
        label='Pleiades (Fang+18)',
        c='white', marker='o', edgecolors='k', linewidths=0.3, s=4
    )

    from rudolf.starinfo import starinfodict as sd
    namelist = ['KOI-7913 A', 'KOI-7913 B', 'Kepler-1643']
    markers = ['X', 'X', 's']
    mfcs = ['lime', 'lime', '#ff6eff']

    for n,m,mfc in zip(namelist, markers, mfcs):

        SpT_val = sd[n]['SpT'][1]
        Ha_EW = sd[n]['Halpha_EW']/(1e3)
        yerr = np.array(
            [sd[n]['Halpha_EW_merr'], sd[n]['Halpha_EW_perr']]
        ).reshape((2,1))/1e3

        ax.errorbar(
            SpT_val,
            Ha_EW,
            yerr=yerr,
            marker=m,
            c=mfc,
            label=n,
            elinewidth=1, capsize=0, lw=0, mew=0.5, markersize=10,
            mec='k',
            zorder=5
        )

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, 0.8*box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1), handletextpad=0.1,
              fontsize='xx-small', framealpha=1, ncol=3)

    ax.set_ylabel(r'H$\mathrm{\alpha}$ EW [$\mathrm{\AA}$]')
    ax.set_xlabel('Spectral Type')

    ax.set_xticks([-8,-6,-4,-2,0,2,4,6])
    ax.set_xticklabels(['K2', 'K4', 'K6', 'K8', 'M0', 'M2', 'M4', 'M6'])
    ax.set_xlim([-9.5, 3])

    ax.tick_params(axis='both', which='major', labelsize='small')

    ax.set_ylim((2,-6))


    # set naming options
    s = ''

    bn = inspect.stack()[0][3].split("_")[1]
    outpath = os.path.join(outdir, f'{bn}{s}.png')
    savefig(fig, outpath, dpi=400)


def plot_lithium(outdir, reference='Randich18', style='science', lgb_cepher=0,
                 kepms=10):

    set_style(style)

    from timmy.lithium import (
        get_Randich18_lithium, get_Berger18_lithium, get_Randich01_lithium,
        get_Kraus14_Mentuch08_TucHor,
        get_Pleiades_Bouvier17_Jones96_Soderblom93
    )

    if reference == 'Randich18':
        rdf = get_Randich18_lithium()
    elif 'Randich01' in reference:
        rdf = get_Randich01_lithium()
    if 'Pleiades' in reference:
        pdf = get_Pleiades_Bouvier17_Jones96_Soderblom93()
    bdf = get_Berger18_lithium()
    if lgb_cepher:
        import pandas as pd
        chdf = pd.read_csv(
            '/Users/luke/Dropbox/proj/CepHer_spectra_traceback/data/'
            '20231128_CepHer_RVs_LiEWs_Prots_x_GDR2.csv'
        )

    if reference == 'Randich18':
        selclusters = [
            # 'IC4665', # LDB 23.2 Myr
            #'NGC2547', # LDB 37.7 Myr
            'IC2602', # LDB 43.7 Myr
            'IC2391', # LDB 51.3 Myr
        ]
        selrdf = np.zeros(len(rdf)).astype(bool)
        for c in selclusters:
            selrdf |= rdf.Cluster.str.contains(c)

        srdf = rdf[selrdf]
        srdf_lim = srdf[srdf.f_EWLi==3]
        srdf_val = srdf[srdf.f_EWLi==0]

    elif reference == 'Randich01':
        srdf = rdf
        srdf_lim = srdf[srdf.f_EWLi=='<=']
        srdf_val = srdf[srdf.f_EWLi!='<=']

    elif reference == 'Pleiades':
        srdf = pdf
        srdf_lim = srdf[srdf.f_EWLi==1]
        srdf_val = srdf[srdf.f_EWLi==0]

    elif reference == 'Randich01_TucHorK14M08':
        tdf = get_Kraus14_Mentuch08_TucHor()
        tdf_lim = tdf[tdf.f_EWLi==1]
        tdf_val = tdf[tdf.f_EWLi==0]

        cols = 'Teff,e_Teff,EWLi,e_EWLi'.split(',')

        # merge IC2602 and Tuc-Hor dataframes
        _df_val = pd.concat((srdf_val[cols], tdf_val[cols]))
        _df_lim = pd.concat((srdf_lim[cols], tdf_lim[cols]))

        srdf_lim = deepcopy(_df_lim)
        srdf_val = deepcopy(_df_val)

    elif reference == 'Pleiades_Randich01_TucHorK14M08':

        # IC2602
        srdf = rdf
        srdf_lim = srdf[srdf.f_EWLi=='<=']
        srdf_val = srdf[srdf.f_EWLi!='<=']

        # TucHor
        tdf = get_Kraus14_Mentuch08_TucHor()
        tdf_lim = tdf[tdf.f_EWLi==1]
        tdf_val = tdf[tdf.f_EWLi==0]

        cols = 'Teff,e_Teff,EWLi,e_EWLi'.split(',')

        # merge IC2602 and Tuc-Hor dataframes
        import pandas as pd
        _df_val = pd.concat((srdf_val[cols], tdf_val[cols]))
        _df_lim = pd.concat((srdf_lim[cols], tdf_lim[cols]))

        srdf_lim = deepcopy(_df_lim)
        srdf_val = deepcopy(_df_val)

        # Pleiades
        spdf = deepcopy(pdf)
        spdf_lim = spdf[spdf.f_EWLi==1]
        spdf_val = spdf[spdf.f_EWLi==0]


    # young dictionary
    yd = {
        'val_teff_young': nparr(srdf_val.Teff),
        'val_teff_err_young': nparr(srdf_val.e_Teff),
        'val_li_ew_young': nparr(srdf_val.EWLi),
        'val_li_ew_err_young': nparr(srdf_val.e_EWLi),
        'lim_teff_young': nparr(srdf_lim.Teff),
        'lim_teff_err_young': nparr(srdf_lim.e_Teff),
        'lim_li_ew_young': nparr(srdf_lim.EWLi),
        'lim_li_ew_err_young': nparr(srdf_lim.e_EWLi),
    }
    if reference == 'Pleiades_Randich01_TucHorK14M08':
        pd = {
            'val_teff_young': nparr(spdf_val.Teff),
            'val_teff_err_young': nparr(spdf_val.e_Teff),
            'val_li_ew_young': nparr(spdf_val.EWLi),
            'val_li_ew_err_young': nparr(spdf_val.e_EWLi),
            'lim_teff_young': nparr(spdf_lim.Teff),
            'lim_teff_err_young': nparr(spdf_lim.e_Teff),
            'lim_li_ew_young': nparr(spdf_lim.EWLi),
            'lim_li_ew_err_young': nparr(spdf_lim.e_EWLi),
        }

    # field dictionary
    # SNR > 3
    field_det = ( (bdf.EW_Li_ / bdf.e_EW_Li_) > 3 )
    bdf_val = bdf[field_det]
    bdf_lim = bdf[~field_det]

    fd = {
        'val_teff_field': nparr(bdf_val.Teff),
        'val_li_ew_field': nparr(bdf_val.EW_Li_),
        'val_li_ew_err_field': nparr(bdf_val.e_EW_Li_),
        'lim_teff_field': nparr(bdf_lim.Teff),
        'lim_li_ew_field': nparr(bdf_lim.EW_Li_),
        'lim_li_ew_err_field': nparr(bdf_lim.e_EW_Li_),
    }

    d = {**yd, **fd}

    ##########
    # make tha plot 
    ##########

    plt.close('all')

    if not lgb_cepher:
        f, ax = plt.subplots(figsize=(4,3))
    else:
        f, ax = plt.subplots(figsize=(4,4))

    classes = ['young', 'field']
    colors = ['k', 'gray']
    zorders = [2, 1]
    markers = ['o', '.']
    ss = [2.5, 5]
    #NGC$\,$2547 & IC$\,$2602
    label = '40-50 Myr (IC2602 R01, TucHor K14)' if reference != 'Pleiades' else '112 Myr (Pleiades S93, J96, B16)'
    labels = [label, 'Kepler Field']
    alpha = 1

    # plot vals
    for _cls, _col, z, m, l, s in zip(classes, colors, zorders, markers,
                                      labels, ss):

        if reference == 'Pleiades_Randich01_TucHorK14M08':
            if _cls == 'young':
                ax.scatter(
                    d[f'val_teff_{_cls}'], d[f'val_li_ew_{_cls}'],
                    c=_col, alpha=alpha,
                    zorder=z, s=s,
                    rasterized=False, label=l, marker=m, linewidths=0.3
                )

                ax.scatter(
                    pd[f'val_teff_{_cls}'], pd[f'val_li_ew_{_cls}'],
                    c='white', alpha=alpha,
                    zorder=z-1, s=s,
                    rasterized=False, label='112 Myr (Pleiades S93, J96, B16)', marker=m, linewidths=0.3,
                    edgecolors='k'
                )

                ax.scatter(
                    d[f'lim_teff_{_cls}'], d[f'lim_li_ew_{_cls}'], c=_col,
                    alpha=alpha,
                    zorder=z+1, s=5*s, rasterized=False, linewidths=0, marker='v'
                )

                if lgb_cepher:
                    _ms, _cs, _clrs = (
                        ['s','o'], ['rsg5','delLyr'],
                        ['darkviolet','cyan']
                    )
                    for _m, _c, _clr in zip(_ms, _cs, _clrs):
                        s0 = (chdf.subcluster == _c)
                        ax.errorbar(
                            chdf[s0][f'Teff_Curtis20'], chdf[s0]['Fitted_Li_EW_mA'],
                            yerr=np.array([chdf[s0]['Fitted_Li_EW_mA_merr'],
                                           chdf[s0]['Fitted_Li_EW_mA_perr']]),
                            c=_clr, alpha=alpha, zorder=z, markersize=s,
                            elinewidth=0.35, capsize=0, mew=0.5, rasterized=False,
                            label=_c, marker=_m, lw=0
                        )
            else:
                ax.scatter(
                    d[f'val_teff_{_cls}'], d[f'val_li_ew_{_cls}'], c=_col,
                    alpha=alpha,
                    zorder=z, s=s, rasterized=False, linewidths=0, label=l, marker=m
                )


        else:
            if _cls == 'young':
                ax.errorbar(
                    d[f'val_teff_{_cls}'], d[f'val_li_ew_{_cls}'],
                    yerr=d[f'val_li_ew_err_{_cls}'], c=_col, alpha=alpha,
                    zorder=z, markersize=s, elinewidth=0.35, capsize=0, mew=0.5,
                    rasterized=False, label=l, marker=m, lw=0
                )

                ax.scatter(
                    d[f'lim_teff_{_cls}'], d[f'lim_li_ew_{_cls}'], c=_col,
                    alpha=alpha,
                    zorder=z+1, s=5*s, rasterized=False, linewidths=0, marker='v'
                )

            else:
                ax.scatter(
                    d[f'val_teff_{_cls}'], d[f'val_li_ew_{_cls}'], c=_col,
                    alpha=alpha,
                    zorder=z, s=s, rasterized=False, linewidths=0, label=l, marker=m
                )


    from rudolf.starinfo import starinfodict as sd

    # lime: CH-2 (KOI-7913, KOI-7368)
    # #ff6eff: RSG5 (Kepler-1643)
    # gray/black: del Lyr cluster (Kepler-1627)
    namelist = ['Kepler-1627 A', 'KOI-7368', 'KOI-7913 A', 'KOI-7913 B', 'Kepler-1643']
    markers = ['P', 'v', 'X', 'X', 's']
    mfcs = ['white', 'lime', 'lime', 'lime', '#ff6eff']

    for n,m,mfc in zip(namelist, markers, mfcs):

        teff = sd[n]['Teff']
        Li_EW = sd[n]['Li_EW']
        yerr = np.array([sd[n]['Li_EW_merr'],
                         sd[n]['Li_EW_perr']]).reshape((2,1))

        ax.errorbar(
            teff,
            Li_EW,
            yerr=yerr,
            marker=m,
            c=mfc,
            label=n,
            elinewidth=1, capsize=0, lw=0, mew=0.5, markersize=kepms,
            mec='k',
            zorder=5
        )

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, 0.8*box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1), handletextpad=0.1,
              fontsize='xx-small', framealpha=1, ncol=4)

    ax.set_ylabel('Li$_{6708}$ EW [m$\mathrm{\AA}$]')
    ax.set_xlabel('Effective Temperature [K]')

    #ax.set_xlim((4900, 6600))
    ax.set_ylim((-20,420))
    ax.set_xlim((3200, 6700))

    format_ax(ax)
    s = f'_{reference}'
    if lgb_cepher:
        s += f'_LGB-CepHer'
    outpath = os.path.join(outdir, f'lithium{s}.png')
    savefig(f, outpath)


def plot_CepHer_weights(outdir):
    """
    how are the weights distributed?  log-normal?  what parameters?
    """

    df, __df, _ = get_ronan_cepher_augmented()

    for _df, flag in zip([df, __df], ['RonanLGBFlag', 'RonanFlag']):

        plt.close('all')
        set_style()
        fig, ax = plt.subplots(figsize=(4,3))

        bins = np.logspace(-5,0,21)

        hists = ax.hist(_df.strengths, bins=bins, cumulative=False, color='k',
                        fill=False, histtype='step', linewidth=0.5)

        log10_x = np.log10(nparr(_df.strengths))

        # midpoints of logspaced values
        midway = 10**((np.log10(bins[0:-1])+np.log10(bins[1:]))/2)
        vals = hists[0]

        # fit a log-normal gaussian using a non-linear fitter
        g_init = models.Gaussian1D(
            amplitude=max(vals), mean=np.nanmedian(log10_x), stddev=1
        )
        fit = fitting.LevMarLSQFitter()
        g_fit = fit(g_init, np.log10(midway), vals)

        x_new = np.logspace(-5,0,1000)
        ax.plot(x_new, g_fit(np.log10(x_new)), zorder=1, c='C0')

        txt = f"α ~ logN({g_fit.mean.value:.4f}, {g_fit.stddev.value:.4f})"
        ax.set_title(txt)

        ax.set_xlabel('strengths [D]')
        ax.set_ylabel('count')
        ax.set_xscale('log')
        ax.set_yscale('linear')
        outpath = os.path.join(outdir, f'hist_weight_{flag}.png')
        savefig(fig, outpath)




def plot_CepHer_quicklook_tests(outdir):

    # get data
    csvpath = os.path.join(DATADIR, 'Cep-Her',
                           '20220311_Kerr_SPYGLASS205_Members_All.csv')
    df = pd.read_csv(csvpath)

    df = df.rename({'RA':'ra', 'Dec':'dec', 'RV':'dr2_radial_velocity'}, axis='columns')

    df['bp-rp'] = df['bp'] - df['rp']
    df['M_G'] = df['g'] + 5*np.log10(df['parallax']/1e3) + 5

    c = SkyCoord(ra=nparr(df.ra)*u.deg, dec=nparr(df.dec)*u.deg)
    df['l'] = c.galactic.l.value
    df['b'] = c.galactic.b.value

    # get: v_l, v_b
    from earhart.physicalpositions import (
        calc_vl_vb_physical,
        append_physicalpositions
    )
    v_l_cosb_km_per_sec, v_b_km_per_sec = calc_vl_vb_physical(
        nparr(df.ra), nparr(df.dec), nparr(df.pmra), nparr(df.pmdec),
        nparr(df.parallax)
    )

    df["v_l*"] = v_l_cosb_km_per_sec
    df['v_b'] = v_b_km_per_sec

    reference_df = pd.DataFrame(df.mean()).T
    append_physicalpositions(df, reference_df)

    xytuples = [
        ('ra', 'dec', 'linear', 'linear'),
        ('l', 'b', 'linear', 'linear'),
        ('pmra', 'pmdec', 'linear', 'linear'),
        ('bp-rp', 'g', 'linear', 'linear'),
        ('bp-rp', 'M_G', 'linear', 'linear'),
        ('parallax', 'age', 'linear', 'linear'),
        ('v_l*', 'v_b', 'linear', 'linear'),
        ('x_pc', 'y_pc', 'linear', 'linear'),
        ('x_pc', 'z_pc', 'linear', 'linear'),
        ('y_pc', 'z_pc', 'linear', 'linear'),
    ]

    for xy in xytuples:
        xkey, ykey = xy[0], xy[1]
        xscale, yscale = xy[2], xy[3]
        invert_y = True if ykey == 'M_G' else False
        invert_x = True if xkey == 'l' else False

        plt.close('all')
        set_style()
        fig, ax = plt.subplots(figsize=(4,3))
        ax.scatter(df[xkey], df[ykey], c='k', s=2, zorder=1)
        ax.set_xlabel(xkey)
        ax.set_ylabel(ykey)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        if invert_y:
            ax.set_ylim(ax.get_ylim()[::-1])
        if invert_x:
            ax.set_xlim(ax.get_xlim()[::-1])
        s = ''
        if xscale == 'log':
            s += '_logx'
        if yscale == 'log':
            s += '_logy'
        outpath = os.path.join(outdir, f'{xkey}_vs_{ykey}{s}.png')
        savefig(fig, outpath)


def _given_df_get_auxiliary_quantities(df):

    # calculate auxiliary quantities
    df['bp-rp'] = df['phot_bp_mean_mag'] - df['phot_rp_mean_mag']
    df['M_G'] = df['phot_g_mean_mag'] + 5*np.log10(df['parallax']/1e3) + 5

    c = SkyCoord(ra=nparr(df.ra)*u.deg, dec=nparr(df.dec)*u.deg)
    df['l'] = c.galactic.l.value
    df['b'] = c.galactic.b.value

    # get: v_l, v_b
    v_l_cosb_km_per_sec, v_b_km_per_sec = calc_vl_vb_physical(
        nparr(df.ra), nparr(df.dec), nparr(df.pmra), nparr(df.pmdec),
        nparr(df.parallax)
    )

    df["v_l*"] = v_l_cosb_km_per_sec
    df['v_b'] = v_b_km_per_sec

    return df


def plot_CepHerExtended_quicklook_tests(outdir):

    # get data
    csvpath = os.path.join(DATADIR, 'Cep-Her',
                           '20220311_Kerr_CepHer_Extended_Candidates.csv')
    df = pd.read_csv(csvpath)
    df = df[(df['photometric flag'].astype(bool)) & (df['astrometric flag'].astype(bool))]

    csvpath1 = os.path.join(DATADIR, 'Cep-Her',
                           '20220311_Kerr_SPYGLASS205_Members_All.csv')
    df1 = pd.read_csv(csvpath1)

    koi_dict = { # strengths
        'KOI-7368': '2128840912955018368', # 0.093
        'KOI-7913 A': '2106235301785454208', # 0.04
        'KOI-7913 B': '2106235301785453824', # 0.04
        'Kepler-1643': '2082142734285082368', # 0.24
        'Kepler-1627 A': '2103737241426734336', # 0.30
    }

    _mdf = df[np.in1d(df.source_id.astype(str),
                      np.array(list(koi_dict.values())))]

    print(_mdf[['source_id','strengths']])

    # verify that everything in the "Extended Candidates" list includes the
    # objects from the "base core members" list.
    mdf = df.merge(df1, left_on='source_id', right_on='GEDR3', how='inner')
    assert len(mdf) == len(df1)

    df = _given_df_get_auxiliary_quantities(df)
    _mdf = _given_df_get_auxiliary_quantities(_mdf)

    reference_df = pd.DataFrame(df.mean()).T
    df = append_physicalpositions(df, reference_df)
    _mdf = append_physicalpositions(_mdf, reference_df)

    xytuples = [
        ('ra', 'dec', 'linear', 'linear'),
        ('l', 'b', 'linear', 'linear'),
        ('pmra', 'pmdec', 'linear', 'linear'),
        ('bp-rp', 'phot_g_mean_mag', 'linear', 'linear'),
        ('bp-rp', 'M_G', 'linear', 'linear'),
        ('v_l*', 'v_b', 'linear', 'linear'),
        ('x_pc', 'y_pc', 'linear', 'linear'),
        ('x_pc', 'z_pc', 'linear', 'linear'),
        ('y_pc', 'z_pc', 'linear', 'linear'),
    ]

    strengths = [2e-2, 4e-2, 1e-1]
    sizes = [1.5, 2, 1]

    for strength_cut, size in zip(strengths, sizes):

        sdf = df[df.strengths > strength_cut]
        print(f'Strength cut: > {strength_cut}: {len(sdf)} objects')
        sstr = f'strengthgt{strength_cut:.2f}'

        csvpath = os.path.join(outdir, f'weight_{sstr}.csv')
        sdf.to_csv(csvpath, index=False)

        for xy in xytuples:

            xkey, ykey = xy[0], xy[1]
            xscale, yscale = xy[2], xy[3]
            invert_y = True if ykey in ['M_G','phot_g_mean_mag'] else False
            invert_x = True if xkey in ['l', 'ra'] else False

            # x vs y, colored by strength!

            # Add a colorbar.
            color = np.log10(sdf['strengths'])

            # https://matplotlib.org/stable/tutorials/colors/colormaps.html
            cmap = mpl.cm.get_cmap('plasma')
            # # Only show points for which color is defined (e.g., not all stars have
            # # ages reported in the table).
            # sel = ~pd.isnull(color)

            plt.close('all')
            set_style()
            fig, ax = plt.subplots(figsize=(4,3))
            _p = ax.scatter(sdf[xkey], sdf[ykey], c=color, s=size, zorder=2,
                            linewidths=0, marker='.', cmap=cmap)
            ax.set_xlabel(xkey)
            ax.set_ylabel(ykey)
            ax.set_xscale(xscale)
            ax.set_yscale(yscale)
            if invert_y:
                ax.set_ylim(ax.get_ylim()[::-1])
            if invert_x:
                ax.set_xlim(ax.get_xlim()[::-1])

            namelist = ['Kepler-1627 A', 'KOI-7368', 'KOI-7913 A',
                        'KOI-7913 B', 'Kepler-1643']
            markers = ['P', 'v', 'X', 'X', 's']
            # lime: CH-2 (KOI-7913, KOI-7368)
            # #ff6eff: RSG5 (Kepler-1643)
            # gray/black: del Lyr cluster (Kepler-1627)
            mfcs = ['white', 'lime', 'lime', 'lime', '#ff6eff']
            for name,marker,mfc in zip(namelist, markers, mfcs):
                source_id = koi_dict[name]
                sel = _mdf.source_id.astype(str) == source_id
                ax.plot(
                    _mdf[sel][xkey], _mdf[sel][ykey],
                    alpha=1, mew=0.5, zorder=1, label=name,
                    markerfacecolor=mfc, markersize=4, marker=marker,
                    color='black', lw=0
                )

            # For the colorbar, inset it into the main plot to keep the aspect
            # ratio.
            axins1 = inset_axes(ax, width="3%", height="20%",
                                loc='lower right', borderpad=0.7)

            cb = fig.colorbar(_p, cax=axins1, orientation="vertical",
                              extend="neither")
            cb.ax.tick_params(labelsize='x-small')
            cb.ax.yaxis.set_ticks_position('left')
            cb.ax.yaxis.set_label_position('left')
            cb.set_label('$\log_{10}$ weight', fontsize='x-small')

            ax.legend(loc='lower left', handletextpad=0.1, fontsize='xx-small',
                      framealpha=0.7)

            showkepler = True if xkey in ['l','ra'] and ykey in ['b','dec'] else False
            if showkepler:
                kep_d = get_keplerfieldfootprint_dict()
                for mod in np.sort(list(kep_d.keys())):
                    for op in np.sort(list(kep_d[mod].keys())):
                        this = kep_d[mod][op]
                        ra, dec = nparr(this['corners_ra']), nparr(this['corners_dec'])
                        if xkey == 'ra' and ykey == 'dec':
                            ax.fill(ra, dec, c='lightgray', alpha=0.95, lw=0,
                                    rasterized=True, zorder=-1)
                        elif xkey == 'l' and ykey == 'b':
                            c = SkyCoord(ra=nparr(ra)*u.deg, dec=nparr(dec)*u.deg)
                            glon = c.galactic.l.value
                            glat = c.galactic.b.value
                            ax.fill(glon, glat, c='lightgray', alpha=0.95, lw=0,
                                    rasterized=True, zorder=-1)


            s = ''
            if xscale == 'log':
                s += '_logx'
            if yscale == 'log':
                s += '_logy'
            outpath = os.path.join(outdir, f'weight_{sstr}_{xkey}_vs_{ykey}{s}.png')
            savefig(fig, outpath)


def plot_CepHer_XYZvtang_sky(outdir, showgroups=0, lsrcorr=0):
    """
    Figure 1 from Cep-Her paper
    """

    from rudolf.helpers import get_ronan_cepher_augmented
    df, __df, _mdf = get_ronan_cepher_augmented()
    koi_dict = { # strengths
        'KOI-7368': '2128840912955018368', # 0.093
        'KOI-7913 A': '2106235301785454208', # 0.04
        'KOI-7913 B': '2106235301785453824', # 0.04
        'Kepler-1643': '2082142734285082368', # 0.24
        'Kepler-1627 A': '2103737241426734336', # 0.30
    }

    # set up the axis dictionary.
    plt.close('all')
    set_style()
    # (8.5x11), -1inch all sides = (6.5 x 9), -1.5 inch bottom for caption.
    fig = plt.figure(figsize=(1.1*6.5, 1.1*6.8))
    axd = fig.subplot_mosaic(
        """
        AAAA
        AAAA
        BBCC
        BBDD
        EEFF
        EEFF
        """,
        gridspec_kw={
            "height_ratios": [1,1,1.8,1.8,1.4,1.4],
        }
    )

    df['v_l'] = df['v_l*']/np.cos(np.deg2rad(df['b']))

    vlstr = (
        "$v_{l*}$ [km$\,$s$^{-1}$]" if not lsrcorr else
        "$v_{l*} - v_{\mathrm{LSR}}$ [km$\,$s$^{-1}$]"
    )
    xytuples = [
        ('l', 'b', 'linear', 'linear', "A",
         ["$l$ [deg]", "$b$ [deg]"], [(102, 38), (-6,26)]),
        ('x_pc', 'y_pc', 'linear', 'linear', "B",
         ["$X$ [pc]", "$Y$ [pc]"], [(-105, 285), (135, 415)]),# (-22, 178)]),
        ('x_pc', 'z_pc', 'linear', 'linear', "C",
         ["$X$ [pc]", "$Z$ [pc]"], None),
        ('y_pc', 'z_pc', 'linear', 'linear', "D",
         ["$Y$ [pc]", "$Z$ [pc]"], None),
        ('v_l*', 'v_b', 'linear', 'linear', "E",
         [vlstr, "$v_{b}$ [km$\,$s$^{-1}$]"],
         [(-15,15), (-11,1)]),
        ('l', 'v_l*', 'linear', 'linear', "F",
         ["$l$ [deg]", vlstr],
         [(102,38), (-15,15)]),
    ]

    # save the augmented dataframe
    strength_cut = 0.00
    sdf = df[df.strengths > strength_cut]
    __sdf = __df[__df.strengths > strength_cut]
    print(f'Strength cut: > {strength_cut}: {len(__sdf)} objects')
    sstr = f'strengthgt{strength_cut:.2f}'
    csvpath = os.path.join(outdir, f'weight_{sstr}.csv')
    SELCOLS = ['source_id','l','b','x_pc','y_pc','z_pc',
               'v_l*','v_b','bp-rp','M_G','strengths']
    __sdf[SELCOLS].to_csv(csvpath, index=False)

    for xy in xytuples:

        xkey, ykey = xy[0], xy[1]
        xscale, yscale = xy[2], xy[3]
        invert_y = True if ykey in ['M_G','phot_g_mean_mag'] else False
        invert_x = True if xkey in ['l', 'ra'] else False
        axkey = xy[4]
        xlabel, ylabel = xy[5][0], xy[5][1]

        xylim = xy[6]
        if xylim is not None:
            xlim = xylim[0]
            ylim = xylim[1]

        if xkey == 'x_pc':
            x0 = 8122
        else:
            x0 = 0

        # assign axis
        ax = axd[axkey]

        if xkey == 'l' and ykey == 'v_l*':
            lons = np.arange(0,360,1)
            _, v_l_cosb_kms_upper, v_l_cosb_kms_lower = get_vl_lsr_corr(lons, get_errs=1)
            ax.fill_between(
                lons, v_l_cosb_kms_lower, v_l_cosb_kms_upper,
                alpha=0.1, color='black', lw=0.5, zorder=-2
            )

        # iterate over "gold" and "maybe" candidates
        for strength_cut, c, size in zip(
            [0.02, 0.10], ['darkgray','black'], [1.0,1.8]
        ):
            # STRENGTH_CUT: require > 0.02
            sdf = df[df.strengths > strength_cut]

            dx, dy = 0, 0
            if lsrcorr and (xkey == 'v_l*'):
                dx,_,_ = get_vl_lsr_corr(np.array(sdf['l']))
            if lsrcorr and (ykey == 'v_l*'):
                dy,_,_ = get_vl_lsr_corr(np.array(sdf['l']))

            print(f'Strength cut: > {strength_cut}: {len(sdf)} objects')
            sstr = f'strengthgt{strength_cut:.2f}'
            csvpath = os.path.join(outdir, f'weight_{sstr}.csv')

            sdf[SELCOLS].to_csv(csvpath, index=False)

            do_colorbar = 0

            if do_colorbar:
                # Add a colorbar.
                color = np.log10(sdf['strengths'])
                # https://matplotlib.org/stable/tutorials/colors/colormaps.html
                cmap = mpl.cm.get_cmap('plasma')
                _p = ax.scatter(sdf[xkey]+x0-dx, sdf[ykey]-dy, c=color, s=size,
                                zorder=2, linewidths=0, marker='.', cmap=cmap,
                                rasterized=True)
            else:
                _p = ax.scatter(sdf[xkey]+x0-dx, sdf[ykey]-dy, c=c, s=size, zorder=2,
                                linewidths=0, marker='.', rasterized=True)

        if showgroups:
            # Manually selected groups in glue.
            #groupnames = ['CH2_XYZ_vl_vb_cut.csv', 'RSG5_XYZ_vl_vb_cut.csv']
            # NOTE: this plot should have the automatically selected groups!
            groupnames = ['CH-2_auto_XYZ_vl_vb_cut.csv', 'RSG-5_auto_XYZ_vl_vb_cut.csv']
            groupcolors = ['lime', '#ff6eff']
            sizes = [3, 2]
            mews = [0.3, 0.1]
            ix = 0
            for groupname, color, size, mew in zip(
                groupnames, groupcolors, sizes, mews
            ):
                grouppath = os.path.join(outdir, groupname)
                _df = pd.read_csv(grouppath)
                _df['v_l'] = _df['v_l*']/np.cos(np.deg2rad(_df['b']))

                _dx, _dy = 0, 0
                if lsrcorr and (xkey == 'v_l*'):
                    _dx,_,_ = get_vl_lsr_corr(np.array(_df['l']))
                if lsrcorr and (ykey == 'v_l*'):
                    _dy,_,_ = get_vl_lsr_corr(np.array(_df['l']))

                ax.plot(
                    _df[xkey]+x0-_dx, _df[ykey]-_dy,
                    alpha=1, mew=mew, zorder=10+ix, markerfacecolor=color,
                    markersize=size, marker='o', color='black', lw=0
                )
                ix -= 1

        if xkey == 'x_pc' and ykey == 'y_pc':
            # overplot the ring at d=330 pc.
            def given_r_get_xy(R, num_samples=1000):
                theta = np.linspace(0, 2*np.pi, num_samples)
                x, y = R * np.cos(theta), R * np.sin(theta)
                return x,y

            for radius in [250, 300, 350, 400]:
                x,y = given_r_get_xy(radius)
                sel = (y>0) & (x>-120)
                x,y = x[sel], y[sel]
                ax.plot(x, y, alpha=0.2, color='black', lw=0.5, zorder=-2)

        # Show our objects!
        namelist = ['Kepler-1627 A', 'KOI-7368', 'KOI-7913 A',
                    'KOI-7913 B', 'Kepler-1643']
        markers = ['P', 'v', 'X', 'X', 's']
        # lime: CH-2 (KOI-7913, KOI-7368)
        # #ff6eff: RSG5 (Kepler-1643)
        # gray/black: del Lyr cluster (Kepler-1627)
        mfcs = ['white', 'lime', 'lime', 'lime', '#ff6eff']
        for name,marker,mfc in zip(namelist, markers, mfcs):
            source_id = koi_dict[name]
            sel = _mdf.source_id.astype(str) == source_id
            _mdf['v_l'] = _mdf['v_l*']/np.cos(np.deg2rad(_mdf['b']))

            _dx, _dy = 0, 0
            if lsrcorr and (xkey == 'v_l*'):
                _dx,_,_ = get_vl_lsr_corr(np.array(_mdf[sel]['l']))
            if lsrcorr and (ykey == 'v_l*'):
                _dy,_,_ = get_vl_lsr_corr(np.array(_mdf[sel]['l']))

            ax.plot(
                _mdf[sel][xkey]+x0-_dx, _mdf[sel][ykey]-_dy,
                alpha=1, mew=0.5, zorder=10, label=name,
                markerfacecolor=mfc, markersize=7, marker=marker,
                color='black', lw=0
            )

        ax.set_xlabel(xlabel, labelpad=1)
        ax.set_ylabel(ylabel, labelpad=1)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        if invert_y:
            ax.set_ylim(ax.get_ylim()[::-1])
        if invert_x:
            ax.set_xlim(ax.get_xlim()[::-1])
        if xylim is not None:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        if xkey == 'x_pc':
            ax.set_xticks([-50, 0, 50, 100, 150, 200, 250])
            from matplotlib.ticker import (
                MultipleLocator, FormatStrFormatter, AutoMinorLocator
            )
            ax.xaxis.set_major_locator(MultipleLocator(50))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax.xaxis.set_minor_locator(MultipleLocator(10))

        #ax.set_xticklabels(fontsize='x-small')
        #ax.set_yticklabels(fontsize='x-small')
        ax.tick_params(axis='both', which='major', labelsize='small')

        if do_colorbar:
            # For the colorbar, inset it into the main plot to keep the aspect
            # ratio.
            axins1 = inset_axes(ax, width="3%", height="20%",
                                loc='lower right', borderpad=0.7)

            cb = fig.colorbar(_p, cax=axins1, orientation="vertical",
                              extend="neither")
            cb.ax.tick_params(labelsize='x-small')
            cb.ax.yaxis.set_ticks_position('left')
            cb.ax.yaxis.set_label_position('left')
            cb.set_label('$\log_{10}$ weight', fontsize='x-small')

        if xkey == 'x_pc' and ykey == 'y_pc':
            ax.legend(loc='lower left', handletextpad=0.1, fontsize='x-small',
                      framealpha=0.7)

        showkepler = True if xkey in ['l','ra'] and ykey in ['b','dec'] else False
        if showkepler:
            kep_d = get_keplerfieldfootprint_dict()
            for mod in np.sort(list(kep_d.keys())):
                for op in np.sort(list(kep_d[mod].keys())):
                    this = kep_d[mod][op]
                    ra, dec = nparr(this['corners_ra']), nparr(this['corners_dec'])
                    if xkey == 'ra' and ykey == 'dec':
                        ax.fill(ra, dec, c='lightgray', alpha=0.95, lw=0,
                                rasterized=True, zorder=-1)
                    elif xkey == 'l' and ykey == 'b':
                        c = SkyCoord(ra=nparr(ra)*u.deg, dec=nparr(dec)*u.deg)
                        glon = c.galactic.l.value
                        glat = c.galactic.b.value
                        ax.fill(glon, glat, c='lightgray', alpha=0.95, lw=0,
                                rasterized=True, zorder=-1)

    s = ''
    if showgroups:
        s += '_showgroups'
    if lsrcorr:
        s += '_lsrcorr'
    outpath = os.path.join(outdir, f'CepHer_XYZvtang_sky{s}.png')

    fig.tight_layout(h_pad=0., w_pad=0.2)
    #f.tight_layout(h_pad=0.2, w_pad=0.2)

    savefig(fig, outpath)


def plot_kerr21_XY(outdir, tablenum=1, colorkey=None, show_CepHer=0):
    """
    Make a top-down plot like Figure 7 of Kerr+2021
    (https://ui.adsabs.harvard.edu/abs/2021ApJ...917...23K/abstract).

    Optionally color by stellar age, similar to Fig 24 of McBride+2021.
    (https://ui.adsabs.harvard.edu/abs/2021AJ....162..282M/abstract).

    Args:

        outdir (str): path to output directory

        tablenum (int): 1 or 2.

        colorkey (str or None): column string from Kerr+2021 table to color
        points by: for instance "Age", "bp-rp", "plx", or "P" (latter means
        "Probability of star being <50 Myr old).
    """

    set_style()

    assert isinstance(outdir, str)
    assert isinstance(colorkey, str) or (colorkey is None)

    #
    # get data, calculate galactic X,Y,Z positions (assuming well-measured
    # parallaxes).
    #
    if tablenum == 1:
        kerrpath = os.path.join(DATADIR, 'gaia', "Kerr_2021_Table1.txt")
    elif tablenum == 2:
        kerrpath = os.path.join(DATADIR, 'gaia',
                                "Kerr_2021_table2_apjac0251t2_mrt.txt")

    df = Table.read(kerrpath, format='cds').to_pandas()

    from sunnyhills.physicalpositions import calculate_XYZ_given_RADECPLX
    x,y,z = calculate_XYZ_given_RADECPLX(df.RAdeg, df.DEdeg, df.plx)

    if show_CepHer:
        from rudolf.helpers import get_ronan_cepher_augmented
        _df, __df, _mdf = get_ronan_cepher_augmented()

    #
    # make the plot
    #
    fig, ax = plt.subplots(figsize=(4,4))

    x_sun, y_sun = -8122, 0
    ax.scatter(
        x_sun, y_sun, c='black', alpha=1, zorder=1, s=20, rasterized=True,
        linewidths=1, marker='x'
    )

    if colorkey is None:
        # By default, just show all the stars as the same color.  The
        # "rasterized=True" kwarg here is good if you save the plots as pdfs,
        # to not need to save the positions of too many points.
        ax.scatter(
            x, y, c='black', alpha=1, zorder=2, s=2, rasterized=True,
            linewidths=0, marker='.'
        )

        if show_CepHer:
            sel = _df.strengths > 0.15
            ax.scatter(
                _df[sel].x_pc, _df[sel].y_pc, c='black', alpha=1, zorder=2,
                s=2, rasterized=True, linewidths=0, marker='.'
            )


    else:
        # Add a colorbar.
        color = df[colorkey]

        # Only show points for which color is defined (e.g., not all stars have
        # ages reported in the table).
        sel = ~pd.isnull(color)

        # Define the colormap.  See
        # https://matplotlib.org/stable/tutorials/colors/colormaps.html, there
        # are better choices, but this one is OK for comparing against
        # McBride+21.
        cmap = mpl.cm.get_cmap('rainbow')

        _p = ax.scatter(
            x[sel], y[sel], c=color[sel], alpha=1, zorder=3, s=2,
            rasterized=True, linewidths=0, marker='.', cmap=cmap
        )

        if (tablenum == 2 and colorkey == 'Weight'):
            for ix, lo in enumerate(np.arange(0,1,0.1)):
                hi = lo + 0.1
                sel = (
                    (~pd.isnull(color))
                    &
                    (color > lo)
                    &
                    (color <= hi)
                )
                ax.scatter(
                    x[sel], y[sel], c=color[sel], alpha=1, s=2,
                    rasterized=True, linewidths=0, marker='.', cmap=cmap,
                    zorder=10+ix
                )

        # For the colorbar, inset it into the main plot to keep the square
        # aspect ratio.
        axins1 = inset_axes(ax, width="3%", height="20%", loc='lower right',
                            borderpad=0.7)

        cb = fig.colorbar(_p, cax=axins1, orientation="vertical",
                          extend="neither")
        cb.ax.tick_params(labelsize='x-small')
        cb.ax.yaxis.set_ticks_position('left')
        cb.ax.yaxis.set_label_position('left')

        KEYLABELDICT = {
            'Age': 'Age [years]',
            'P': 'P$_{\mathrm{<50\ Myr}}$',
            'plx': 'Parallax [mas]',
        }
        if colorkey in KEYLABELDICT:
            cb.set_label(KEYLABELDICT[colorkey], fontsize='x-small')
        else:
            cb.set_label(colorkey, fontsize='x-small')

    ax.set_xlabel("X [pc]")
    ax.set_ylabel("Y [pc]")

    s = ''
    if colorkey:
        s += f"_{colorkey}"
    if show_CepHer:
        s += f"_ShowCepHer"

    outpath = os.path.join(outdir, f'kerr21t{tablenum}_XY{s}.png')
    fig.savefig(outpath, bbox_inches='tight', dpi=400)
    print(f"Made {outpath}")


def _get_theia520():
    csvpath = os.path.join(DATADIR, "gaia", "string_table1.csv")
    df = pd.read_csv(csvpath)
    df = df[df.group_id==520] # Theia-520 / UBC-1
    c = SkyCoord(ra=nparr(df.ra)*u.deg, dec=nparr(df.dec)*u.deg)
    df['l'] = c.galactic.l.value
    df['b'] = c.galactic.b.value
    return df


def _get_melange2():
    #csvpath = os.path.join(DATADIR, "gaia",
    #                       "Barber_2022_melange2_webplotdigitize.csv")
    csvpath = os.path.join(DATADIR, "gaia",
                           "Barber_2022_MELANGE-2.csv")
    df = pd.read_csv(csvpath)
    df = df[df['Voff(km/s)'] < 2]
    c = SkyCoord(ra=nparr(df.RA)*u.deg, dec=nparr(df.DEC)*u.deg)
    df['l'] = c.galactic.l.value
    df['b'] = c.galactic.b.value
    return df


def plot_kepclusters_skychart(outdir, showkepler=1, showkepclusters=1,
                              clusters=None, showplanets=0, darkcolors=False,
                              hideaxes=0, showET=0, showPLATO=0,
                              showcdipsages=0, style='science', factor=1,
                              cepher_alpha=1, figx=19/2, figy=7/2):
    """
    clusters: any of ['Theia-520', 'Melange-2', 'Cep-Her', 'δ Lyr', 'RSG-5', 'CH-2']
    """

    set_style(style)

    #
    # collect data
    #

    cluster_planet_dict = {
        'Theia-520': ['Kepler-52', 'Kepler-968'],
        'Melange-2': ['KOI-3876', 'Kepler-970'],
        'Cep-Her': ['Kepler-1627', 'Kepler-1643', 'KOI-7368', 'KOI-7913'],
        'δ Lyr': ['Kepler-1627'],
        'δ Lyr keck': ['Kepler-1627'],
        'RSG-5': ['Kepler-1643'],
        'RSG-5 keck': ['Kepler-1643'],
        'CH-2': ['KOI-7368', 'KOI-7913'],
        'ch03': [],
        'ch01': [],
        'ch06': [],
        'ch04': []
    }
    position_dict = {
        'Kepler-52': [80.47826, 18.08719],
        'Kepler-968': [80.34362, 18.81791],
        'KOI-3876': [70.72417, 11.07782],
        'Kepler-970': [72.46181, 12.58684],
        'Kepler-1627': [71.5646, 16.77961],
        'Kepler-1643': [79.86511, 7.481905],
        'KOI-7368': [80.62987, 12.99258],
        'KOI-7913': [75.79313, 16.30537]
    }
    cluster_color_dict = {
        'Theia-520': '#5c167f',
        'Melange-2': '#a4327e',
        'Cep-Her': '#ffa873' if not darkcolors else "#BDD7EC",
        'δ Lyr': 'white',
        'δ Lyr keck': 'white',
        'RSG-5': '#ff6eff',
        'RSG-5 keck': '#ff6eff',
        'CH-2': 'lime',
        'ch01': 'cyan', # Cep-Foreground
        'ch03': 'C5', # lyr-foreground
        'ch04': 'C3', # Hercules,
        'ch06': 'C4', # Hercules-diffuse,
        #'Theia-520': 'royalblue',
        #'Melange-2': 'goldenrod',
        #'Cep-Her': None,
        #'δ Lyr': 'white',
        #'RSG-5': '#ff6eff',
        #'CH-2': 'lime'
    }

    namelist = ['Kepler-52', 'Kepler-968', 'Kepler-1627', 'KOI-7368',
                'KOI-7913', 'Kepler-1643', 'Kepler-970', 'KOI-3876']
    ages = [3e8, 3e8, 3.8e7, 3.8e7, 3.8e7, 3.8e7, 1.1e8, 1.1e8]
    markers = ['o','d','P', 'v', 'X', 's', '^', 'p']
    sizes = [80, 80, 120, 120, 120, 120, 95, 95]

    clusterdict = {}
    if clusters is not None:
        for cluster in clusters:
            if cluster == 'Theia-520':
                df = _get_theia520()
            if cluster == 'Melange-2':
                df = _get_melange2()
            if cluster == 'δ Lyr':
                df = get_deltalyr_kc19_cleansubset()
            if cluster == 'RSG-5':
                csvpath = os.path.join(RESULTSDIR, 'CepHer_XYZvtang_sky',
                                       'RSG-5_auto_XYZ_vl_vb_cut.csv')
                df = pd.read_csv(csvpath)
            if cluster == 'CH-2':
                csvpath = os.path.join(RESULTSDIR, 'CepHer_XYZvtang_sky',
                                       'CH-2_auto_XYZ_vl_vb_cut.csv')
                df = pd.read_csv(csvpath)
            if cluster == 'Cep-Her':
                df, _, _ = get_ronan_cepher_augmented()
                chdf = deepcopy(df)
            if cluster in [
                'ch01', 'ch03', 'ch04', 'ch06', 'δ Lyr keck', 'RSG-5 keck'
            ]:
                DATADIR = '/Users/luke/Dropbox/proj/ldb/data/Cep-Her'
                namedict = {
                    'δ Lyr keck': 'delLyr.csv',
                    'RSG-5 keck': 'rsg5.csv',
                    'ch01': 'ch01_cep.csv',
                    'ch03': 'ch03_lyrfgd.csv',
                    'ch04': 'ch04_her.csv',
                    'ch06': 'ch06_herdiffuse.csv'
                }
                csvpath = os.path.join(DATADIR, namedict[cluster])
                df = pd.read_csv(csvpath)
            clusterdict[cluster] = df

    if showcdipsages:
        from cdips.utils.catalogs import get_cdips_catalog
        df = get_cdips_catalog(ver=0.6)
        ra, dec = nparr(df['ra']), nparr(df['dec'])
        c = SkyCoord(ra=nparr(ra)*u.deg, dec=nparr(dec)*u.deg)
        df['l'] = c.galactic.l.value
        df['b'] = c.galactic.b.value

        sel = (
            (df.parallax > 1)
            &
            (df.parallax / df.parallax_error > 10)
            &
            (df.mean_age < 9)
            &
            (~pd.isnull(df.cluster))
            &
            (df.phot_g_mean_mag < 17)
        )
        sdf = df[sel]

    #
    # make the plot
    #

    plt.close('all')
    #f, ax = plt.subplots(figsize=(15/2,7/2)) # for keynote
    f, ax = plt.subplots(figsize=(factor*figx,factor*figy))

    xkey, ykey = 'l', 'b'
    get_yval = lambda _df: np.array(_df[ykey])
    get_xval = lambda _df: np.array(_df[xkey])

    if clusters is not None:
        for cluster in clusters:
            df = clusterdict[cluster]
            color = cluster_color_dict[cluster]
            if cluster != 'Cep-Her':
                edgecolors = 'k'
                ax.scatter(
                    get_xval(df), get_yval(df), c=color, alpha=1,
                    s=5, rasterized=False, label=cluster, marker='o',
                    edgecolors=edgecolors, linewidths=0.1, zorder=5
                )
            else:
                #sdf = df[df.strengths > 0.02]
                #print(f'Strength cut: > 0.02: {len(sdf)} objects')
                #_p = ax.scatter(get_xval(sdf), get_yval(sdf), c='darkgray',
                #                s=1.5, linewidths=0, marker='.',
                #                rasterized=True)
                _sdf = df[df.strengths > 0.10]
                print(f'Strength cut: > 0.10: {len(_sdf)} objects')
                s = 5 if not darkcolors else 3
                edgecolors = 'k' if not darkcolors else 'white'
                linewidths = 0.1 if not darkcolors else 0.08
                ax.scatter(
                    get_xval(_sdf), get_yval(_sdf), c=color, alpha=cepher_alpha,
                    s=s, rasterized=False, label=cluster, marker='o',
                    edgecolors=edgecolors, linewidths=linewidths, zorder=5
                )

    if showcdipsages:

        # # METHOD-1:
        cmap = mpl.cm.get_cmap('magma_r', 6)
        bounds = np.arange(7.0,9.0,0.01)
        norm = mpl.colors.LogNorm(vmin=1e7, vmax=1e9)

        agebins = np.array([
            (5,7+1/3), (7+1/3,7+2/3), (7+2/3,8), (8,8+1/3), (8+1/3,8+2/3),
            (8+2/3,9)
        ])

        _N = 0
        for ix, agebin in enumerate(agebins[::-1]):
            sel = (
                (sdf.mean_age > agebin[0]) & (sdf.mean_age < agebin[1])
                &
                ((sdf.l  > 65 ) | (sdf.b < 20))
            )
            _p = ax.scatter(
                get_xval(sdf[sel]), get_yval(sdf[sel]),
                c=10**sdf[sel].mean_age, alpha=1, zorder=2+ix, s=1+0.3*ix,
                edgecolors='k',
                marker='o', cmap=cmap, linewidths=0.05, norm=norm,
                rasterized=True
            )

            glon_c = 76.5
            glat_c = 13.5
            halfwidth = 500**0.5 / 2 # 22.36 per side, or 11.18 half-side

            sel1 = (
                (sdf[sel].b > glat_c - halfwidth)
                &
                (sdf[sel].b < glat_c + halfwidth)
                &
                (sdf[sel].l > glon_c - halfwidth)
                &
                (sdf[sel].l < glon_c + halfwidth)
            )

            N = len(sdf[sel & sel1])
            print(f"{agebin[0]:.2f} to {agebin[1]:.2f}: {N}")
            _N += N

        print(f"b {glat_c - halfwidth} to {glat_c + halfwidth}")
        print(f"l {glon_c - halfwidth} to {glon_c + halfwidth}")
        print(f"Total: {_N}")

        #_p = ax.scatter(
        #    get_xval(sdf), get_yval(sdf),
        #    c=10**sdf.mean_age, alpha=1, zorder=2, s=5, edgecolors='k',
        #    marker='o', cmap=cmap, linewidths=0.2, norm=norm
        #)

        # draw the colored points
        axins1 = inset_axes(ax, width="25%", height="3%", loc='upper right',
                            borderpad=1.2)
        cb = f.colorbar(_p, cax=axins1, orientation="horizontal",
                        extend="min", norm=norm)

        cb.set_ticks([1e7,1e8,1e9])
        #cb.set_ticklabels(['$10^7$','$10^8$','$10^9$'])

        cb.ax.tick_params(labelsize='small')
        cb.ax.tick_params(size=0, which='both') # remove the ticks
        cb.ax.yaxis.set_ticks_position('left')
        cb.ax.yaxis.set_label_position('left')
        cb.set_label("Age [years]", fontsize='small')

        # del Lyr cluster
        txtclusternames = {
            #name, glon, glat, age?
            'δ Lyr': [67, 16.5, 40],
            'RSG-5': [83, 7.2, 40],
            'UBC-1': [87.8, 20.7, 350], # Theia 350
            'Teutsch-35': [69.6, 8, 170], # Theia 457
            'LDN-988e': [78.5, 3.2, "10"],
            'Theia-397': [87.5, 10.5, 150],
            #"NGC-7039": [0, 88, "<10"],
            #"FSR-0261": [0, 87, "<10"]
        }
        for k,v in txtclusternames.items():
            bbox = dict(facecolor='white', alpha=0.97, pad=0.1, edgecolor='white')
            x,y = v[0],v[1]
            txt = k + " (" + str(v[2]) + " Myr)"
            ax.text(x, y, txt, ha='center',
                    va='bottom', fontsize=6, bbox=bbox, zorder=99)


    if showplanets:
        for cluster in ['δ Lyr', 'RSG-5', 'CH-2']:
            pl_list = cluster_planet_dict[cluster]
            color = cluster_color_dict[cluster]
            for pl in pl_list:
                l,b = position_dict[pl]
                ind = np.argwhere(np.array(namelist)==pl)
                marker = markers[int(ind)]
                edgecolor = 'white' if darkcolors else 'k'
                ax.plot(
                    l, b,
                    alpha=1, mew=0.5, zorder=10, label=pl,
                    markerfacecolor=color, markersize=8, marker=marker,
                    color=edgecolor, lw=0
                )

    if showkepler:
        kep_d = get_keplerfieldfootprint_dict()
        for mod in np.sort(list(kep_d.keys())):
            for op in np.sort(list(kep_d[mod].keys())):
                this = kep_d[mod][op]
                ra, dec = nparr(this['corners_ra']), nparr(this['corners_dec'])
                c = SkyCoord(ra=nparr(ra)*u.deg, dec=nparr(dec)*u.deg)
                glon = c.galactic.l.value
                glat = c.galactic.b.value
                c = 'dimgray' if darkcolors else 'lightgray'
                alpha = 0.95 if not showET else 0.4
                ax.fill(glon, glat, c=c, alpha=alpha, lw=0,
                        rasterized=True, zorder=-1)

    if showET:
        glon_c = 76.5
        glat_c = 13.5
        halfwidth = 500**0.5 / 2 # 22.36 per side, or 11.18 half-side
        eps = 0.05
        modules = [
            [
                # top-right
                (glon_c+halfwidth, glat_c+halfwidth),
                (glon_c+halfwidth, glat_c+eps),
                (glon_c+eps, glat_c+eps),
                (glon_c+eps, glat_c+halfwidth),
                (glon_c+halfwidth, glat_c+halfwidth)
            ],
            [
                # bottom-right
                (glon_c+halfwidth, glat_c-eps),
                (glon_c+halfwidth, glat_c-halfwidth),
                (glon_c+eps, glat_c-halfwidth),
                (glon_c+eps, glat_c-eps),
                (glon_c+halfwidth, glat_c-eps)
            ],
            [
                # top-left
                (glon_c-eps, glat_c+halfwidth),
                (glon_c-eps, glat_c+eps),
                (glon_c-halfwidth, glat_c+eps),
                (glon_c-halfwidth, glat_c+halfwidth),
                (glon_c-eps, glat_c+halfwidth)
            ],
            [
                # bottom-left
                (glon_c-eps, glat_c-eps),
                (glon_c-eps, glat_c-halfwidth),
                (glon_c-halfwidth, glat_c-halfwidth),
                (glon_c-halfwidth, glat_c-eps),
                (glon_c-eps, glat_c-eps)
            ],
        ]

        glons_glats = [
            (glon_c+halfwidth, glat_c+halfwidth),
            (glon_c+halfwidth, glat_c-halfwidth),
            (glon_c-halfwidth, glat_c-halfwidth),
            (glon_c-halfwidth, glat_c+halfwidth),
            (glon_c+halfwidth, glat_c+halfwidth)
        ]
        glon = [x[0] for x in glons_glats]
        glat = [x[1] for x in glons_glats]
        c = 'dimgray' if darkcolors else 'lightgray'
        #ax.fill(glon, glat, c=c, alpha=0.15, lw=0,
        #        rasterized=True, zorder=-2)
        #ax.fill(glon, glat, facecolor='none', edgecolor='k', alpha=0.3,
        #        linewidth=1, ls='-', zorder=1)

        for module in modules:
            glon = [x[0] for x in module]
            glat = [x[1] for x in module]
            ax.fill(glon, glat, c=c, alpha=0.3, lw=0,
                    rasterized=True, zorder=-2)

        bbox = dict(facecolor='white', alpha=0.97, pad=0.1, edgecolor='white')
        ax.text(glon_c+halfwidth-1, glat_c+halfwidth-1, "Earth 2.0", ha='left',
                va='top', fontsize=9, bbox=bbox, zorder=4)
        #x0,y0 = 74.1, 20.5
        x0,y0 = 75, 17
        ax.text(x0, y0, "Kepler", ha='center',
                va='center', fontsize=9, bbox=bbox, zorder=4)


    if showPLATO:
        glon_c = 70
        glat_c = 30

        csvpath = '/Users/luke/Dropbox/proj/Earth_2pt0/PLATO_fov.csv'
        pldf = pd.read_csv(csvpath)
        ax.fill(
            glon_c + pldf['dlon'], glat_c + pldf['dlat'], c='lightgray',
            alpha=0.3, lw=0, rasterized=False, zorder=-5, hatch='/'
        )

        csvpath = '/Users/luke/Dropbox/proj/Earth_2pt0/PLATO_inner12_fov.csv'
        pldf = pd.read_csv(csvpath)
        ax.fill(
            glon_c + pldf['dlon'], glat_c + pldf['dlat'], c='lightgray',
            alpha=0.3, lw=0, rasterized=False, zorder=-5, hatch='x'
        )
        ax.fill(
            glon_c + pldf['dlat'], glat_c + pldf['dlon'], c='lightgray',
            alpha=0.3, lw=0, rasterized=False, zorder=-5, hatch='x'
        )

        csvpath = '/Users/luke/Dropbox/proj/Earth_2pt0/PLATO_innermost.csv'
        pldf = pd.read_csv(csvpath)
        ax.fill(
            glon_c + pldf['dlon'], glat_c + pldf['dlat'], c='lightgray',
            alpha=0.5, lw=0, rasterized=False, zorder=-5, hatch='....'
        )


    if showkepclusters:
        cluster_names = ['NGC6819', 'NGC6791', 'NGC6811', 'NGC6866']
        cras = [295.33, 290.22, 294.34, 300.983]
        cdecs =[40.18, 37.77, 46.378, 44.158]
        cplxs = [0.356, 0.192, 0.870, 0.686]
        ages_gyr = [1, 8, 1, 0.7]
        for _n, _ra, _dec in zip(cluster_names, cras, cdecs):
            c = SkyCoord(ra=_ra*u.deg, dec=_dec*u.deg)
            _l, _b = c.galactic.l.value, c.galactic.b.value
            ax.scatter(
                _l, _b, c='k', alpha=0.8, zorder=4, s=20, rasterized=True,
                linewidths=0.7, marker='x', edgecolors='k'
            )
            bbox = dict(facecolor='white', alpha=0.9, pad=0, edgecolor='white')
            deltay = 0.4
            if clusters == None:
                ax.text(_l, _b+deltay, _n, ha='center', va='bottom',
                        fontsize=4, bbox=bbox, zorder=4)

    #leg = ax.legend(loc='lower left', handletextpad=0.1,
    #                fontsize='x-small', framealpha=0.9)

    #ax.set_xlim([90, 58])
    #ax.set_ylim([0, 23])
    ax.set_xlim([102, 38])
    ax.set_ylim([-5, 25])
    if showET:
        ax.set_xlim([92, 58])
        ax.set_ylim([-4, 27])
    #if showPLATO:
    #    ax.set_xlim([120, 40])
    #    ax.set_ylim([-4, 50])


    if darkcolors:
        f.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        ax.set_axis_off()

    if not darkcolors:
        ax.set_xlabel(r'Galactic longitude, $l$ [deg]', fontsize='large')
        ax.set_ylabel(r'Galactic latitude, $b$ [deg]', fontsize='large')

    if hideaxes:
        f.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        ax.set_axis_off()
        ax.set_xlabel('')
        ax.set_ylabel('')

    s = ''
    if clusters is not None:
        s += "_".join([f"{s}" for s in clusters])
    if showcdipsages:
        s += '_showcdipsages'
    if showET:
        s += '_showET'
    if showPLATO:
        s += '_showPLATO'
    if showkepler:
        s += '_showkepler'
    if showkepclusters:
        s += '_showkepclusters'
    if showplanets:
        s += '_showplanets'
    if darkcolors:
        s += '_darkcolors'
    if hideaxes:
        s += '_hideaxes'

    bn = 'kepclusters_skychart'
    outpath = os.path.join(outdir, f'{bn}{s}.png')
    savefig(f, outpath, dpi=400)


def plot_prisinzano22_XY(outdir, colorkey=None, show_CepHer=0, show_randommw=0,
                         show_realrandommw=0,
                         noaxis=0, hide_prisinzano=0):
    """
    Args:

        outdir (str): path to output directory

        colorkey (str or None): column string from Prisinzano+2022 table...
    """

    set_style('clean')

    assert isinstance(outdir, str)
    assert isinstance(colorkey, str) or (colorkey is None)

    #
    # get data, calculate galactic X,Y,Z positions (assuming well-measured
    # parallaxes).
    #
    fitspath = os.path.join(DATADIR, 'gaia', "Prisizano_2022_t4.fits")

    hl = fits.open(fitspath)
    df = Table(hl[1].data).to_pandas()
    sdf = df[df.Plx / df.e_Plx > 15]

    from rudolf.physical_positions import (
        calculate_XYZ_given_LBPLX, calculate_XYZ_given_RADECPLX
    )
    x,y,z = calculate_XYZ_given_LBPLX(sdf.GLON, sdf.GLAT, sdf.Plx)

    kerrpath = os.path.join(DATADIR, 'gaia', "Kerr_2021_Table1.txt")
    df = Table.read(kerrpath, format='cds').to_pandas()
    kx,ky,kz = calculate_XYZ_given_RADECPLX(df.RAdeg, df.DEdeg, df.plx)

    rx = np.random.uniform(-500, +500, size=int(2e4))
    ry = np.random.uniform(-500, +500, size=int(2e4))
    sel = np.sqrt(rx**2 + ry**2) > 333
    rx, ry = rx[sel], ry[sel]
    if show_randommw:
        rx = np.random.uniform(-500, +500, size=int(2e5))
        ry = np.random.uniform(-500, +500, size=int(2e5))

    if show_CepHer:
        from rudolf.helpers import get_ronan_cepher_augmented
        _df, __df, _mdf = get_ronan_cepher_augmented()

    if show_realrandommw:
        from cdips.utils.gaiaqueries import given_votable_get_df
        votablepath = os.path.join(
            DATADIR, 'gaia',
            #"example_500parsec_random_draw-result_plxgt2_plxovererrorgt5_gmag_gt17.vot.gz"
            "example_500parsec_random_draw-result_plxgt1.8_plxovererrorgt10_gmag_gt17.vot.gz"
            #"example_500parsec_random_draw-result.vot.gz"
        )
        rdf = given_votable_get_df(votablepath, assert_equal='source_id')
        rx,ry,rz = calculate_XYZ_given_RADECPLX(rdf.ra, rdf.dec, rdf.parallax)
        sel = np.sqrt((rx+8122)**2 + ry**2) < 500
        rx, ry = rx[sel], ry[sel]

    #
    # make the plot
    #
    fig, ax = plt.subplots(figsize=(4,4))

    x_sun, y_sun = -8122, 0
    ax.scatter(
        x_sun, y_sun, c='black', alpha=1, zorder=1, s=20, rasterized=True,
        linewidths=1, marker='x'
    )

    if colorkey is None:
        # By default, just show all the stars as the same color.  The
        # "rasterized=True" kwarg here is good if you save the plots as pdfs,
        # to not need to save the positions of too many points.

        if not hide_prisinzano:
            sel = np.sqrt((x+8122)**2 + y**2) < 500
            ax.scatter(
                x[sel], y[sel], c='black', alpha=1, zorder=2, s=1.5, rasterized=True,
                linewidths=0, marker='.'
            )
            sel = np.sqrt((kx+8122)**2 + ky**2) < 500
            ax.scatter(
                kx[sel], ky[sel], c='black', alpha=1, zorder=2, s=1.5, rasterized=True,
                linewidths=0, marker='.'
            )

        if show_randommw:
            sel = np.sqrt(rx**2 + ry**2) < 500
            ax.scatter(
                -8122+rx[sel], ry[sel], c='black', alpha=1, zorder=2, s=1.5, rasterized=True,
                linewidths=0, marker='.'
            )

        if show_realrandommw:
            #sel = np.sqrt(rx**2 + ry**2) < 500
            ax.scatter(
                rx, ry, c='black', alpha=1, zorder=2, s=1.5, rasterized=True,
                linewidths=0, marker='.'
            )

        if show_CepHer:
            sel = _df.strengths > 0.15
            ax.scatter(
                _df[sel].x_pc, _df[sel].y_pc, c='black', alpha=1, zorder=2,
                s=2, rasterized=True, linewidths=0, marker='.'
            )


    else:
        raise NotImplementedError

    if noaxis:
        ax.set_axis_off()

    if not noaxis:
        ax.set_xlabel("X [pc]")
        ax.set_ylabel("Y [pc]")

    ax.set_xlim([-500-8122, 500-8122])
    ax.set_ylim([-500, 500])

    s = ''
    if colorkey:
        s += f"_{colorkey}"
    if show_CepHer:
        s += f"_ShowCepHer"
    if show_randommw:
        s += f"_showrandomMW"
    if hide_prisinzano:
        s += f"_hidekerrprisinzano"
    if noaxis:
        s += f"_noaxis"
    if show_realrandommw:
        s += f"_showrealrandomMW"

    outpath = os.path.join(outdir, f'prisinzano22_XY{s}.png')
    fig.savefig(outpath, bbox_inches='tight', dpi=400)
    print(f"Made {outpath}")
