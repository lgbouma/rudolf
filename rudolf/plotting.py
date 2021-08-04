"""
plot_TEMPLATE

plot_ruwe_vs_apparentmag
plot_simulated_RM
plot_skychart
plot_XYZvtang
plot_keplerlc
    _plot_zoom_light_curve
plot_flare_checker
plot_ttv
plot_ttv_vs_local_slope
plot_rotation_period_windowslider
plot_flare_pair_time_distribution
plot_hr
plot_rotationperiod_vs_color
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

from scipy.optimize import curve_fit

from astropy import units as u, constants as const
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.table import Table

import matplotlib.patheffects as pe
from matplotlib.ticker import MaxNLocator

from aesthetic.plot import savefig, format_ax, set_style

from astrobase.services.identifiers import gaiadr2_to_tic
from astrobase.lcmath import phase_magseries
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
    get_gaia_catalog_of_nearby_stars, get_clustermembers_cg18_subset,
    get_mutau_members, get_ScoOB2_members,
    supplement_gaia_stars_extinctions_corrected_photometry,
    get_clean_gaia_photometric_sources
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

    df_dr2, df_edr3, trgt_df = get_deltalyr_kc19_gaia_data()

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
        zorder=4, s=5, rasterized=True, linewidths=0, label='Theia 73 (KC19)', marker='.'
    )
    ax.plot(
        get_xval(trgt_df), get_yval(trgt_df), alpha=1, mew=0.5,
        zorder=8, label='Kepler 1627', markerfacecolor='yellow',
        markersize=20, marker='*', color='black', lw=0
    )

    leg = ax.legend(loc='upper left', handletextpad=0.1, fontsize='x-small',
                    framealpha=0.9)

    ax.set_xlabel('G [mag]', fontsize='large')
    ax.set_ylabel('EDR3 RUWE', fontsize='large')
    ax.set_yscale('log')

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
                  shownakedeye=0, showcomovers=0):

    set_style()

    df_dr2, df_edr3, trgt_df = get_deltalyr_kc19_gaia_data()
    if showcomovers:
        df_edr3 = get_deltalyr_kc19_comovers()

    plt.close('all')
    f, ax = plt.subplots(figsize=(4,3), constrained_layout=True)

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
            zorder=4, s=5, rasterized=True, linewidths=0,
            label='$\delta$ Lyr Comovers', marker='.'
        )
        ax.plot(
            get_xval(trgt_df), get_yval(trgt_df), alpha=1, mew=0.5,
            zorder=8, label='Kepler 1627', markerfacecolor='yellow',
            markersize=10, marker='*', color='black', lw=0
        )

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
        bounds = np.arange(-0.5, 4.5, 1)
        ticks = (np.arange(-1,4)+1)
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
            alpha=1, zorder=42, s=60, linewidths=0.2,
            marker='*', norm=norm, edgecolors='k'
        )

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
                        extend='max', ticks=ticks)
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
        ax.set_ylim([ymin,ymax])


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
            sel_names = ['Vega', 'Albireo', 'Deneb', '12Del2Lyr']
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
        leg = ax.legend(loc='upper left', handletextpad=0.1,
                        fontsize='x-small', framealpha=0.9)

    ax.set_xlabel(r'$\alpha$ [deg]', fontsize='large')
    ax.set_ylabel(r'$\delta$ [deg]', fontsize='large')

    s = ''
    if narrowlims:
        s += '_narrowlims'
    if showkepler:
        s += '_showkepler'
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
                  show_comovers=0, show_sun=0):

    plt.close('all')
    set_style()

    # NOTE: assumes plot_XYZvtang has already been run
    df_sel = get_deltalyr_kc19_cleansubset()

    _, df_edr3, trgt_df = get_deltalyr_kc19_gaia_data()
    # set "dr2_radial_velocity" according to Andrew Howard HIRES recon
    # spectrum. agrees with -16.9km/s+/-0.5km/s TRES.
    trgt_df.dr2_radial_velocity = -16.7

    from earhart.physicalpositions import append_physicalpositions
    df_edr3 = append_physicalpositions(df_edr3, trgt_df)
    df_sel = append_physicalpositions(df_sel, trgt_df)
    trgt_df = append_physicalpositions(trgt_df, trgt_df)

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

    factor=1.2
    fig = plt.figure(figsize=(factor*6,factor*4))
    axd = fig.subplot_mosaic(
        """
        AB
        CD
        """#,
        #gridspec_kw={
        #    "width_ratios": [1, 1, 1, 1]
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
            rasterized=True, marker='.'
        )
        if show_comovers:
            axd[k].scatter(
                scmdf[xv], scmdf[yv], c='k', alpha=1, zorder=8, s=2,
                edgecolors='none', rasterized=True, marker='.'
            )

        if show_1627:
            axd[k].plot(
                trgt_df[xv], trgt_df[yv], alpha=1, mew=0.5,
                zorder=42, label='Kepler 1627', markerfacecolor='yellow',
                markersize=7, marker='*', color='black', lw=0
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
            axd['B'].arrow(0.73, 0.07, delta_x, 0, length_includes_head=True,
                           head_width=1e-2, head_length=1e-2,
                           transform=axd['B'].transAxes)
            axd['B'].text(0.73+delta_x/2, 0.085, 'Galactic center',
                          va='bottom', ha='center',
                          transform=axd['B'].transAxes, fontsize='xx-small')

        elif k == 'C':
            delta_x = 0.1
            axd['C'].arrow(0.73, 0.07, delta_x, 0,
                         length_includes_head=True, head_width=1e-2,
                         head_length=1e-2,
                         transform=axd['C'].transAxes)
            axd['C'].text(0.73+delta_x/2, 0.085, 'Galactic rotation', va='bottom',
                        ha='center', transform=axd['C'].transAxes, fontsize='xx-small')


    # quiver option..

    #factor=2
    #x0,y0 = -7980, -220
    #axd['A'].quiver(
    #    x0, y0, factor*vdiff_median.d_x.value,
    #    factor*vdiff_median.d_y.value, angles='xy',
    #    scale_units='xy', scale=1, color='C0',
    #    width=6e-3, linewidths=4, headwidth=8, zorder=9
    #)
    ## NOTE the galactic motion is dominant!!!!
    # axd['A'].quiver(
    #     x0, y0, factor*c_median.v_x.value,
    #     factor*c_median.v_y.value, angles='xy',
    #     scale_units='xy', scale=1, color='gray',
    #     width=6e-3, linewidths=4, headwidth=10, zorder=9
    # )

    #x0,y0 = -8160, -50
    #axd['B'].quiver(
    #    x0, y0, factor*vdiff_median.d_x.value,
    #    factor*vdiff_median.d_z.value, angles='xy',
    #    scale_units='xy', scale=1, color='C0',
    #    width=6e-3, linewidths=4, headwidth=8, zorder=9
    #)

    #x0,y0 = -600, -50
    #axd['C'].quiver(
    #    x0, y0, factor*vdiff_median.d_y.value,
    #    factor*vdiff_median.d_z.value, angles='xy',
    #    scale_units='xy', scale=1, color='C0',
    #    width=6e-3, linewidths=4, headwidth=8, zorder=9
    #)

    #axd['C'].update({'xlabel': 'Y [pc]', 'ylabel': 'Z [pc]'})

    for _,ax in axd.items():
        format_ax(ax)

    fig.tight_layout()

    s = ''
    if show_1627:
        s += "_show1627"
    if show_comovers:
        s += "_showcomovers"
    if show_sun:
        s += "_showsun"

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
    # BEGIN-COPYPASTE FROM RUN_GPTRANSIT.PY
    from betty.paths import BETTYDIR
    from betty.modelfitter import ModelFitter

    # get data
    modelid, starid = 'gptransit', 'Kepler_1627'
    datasets = OrderedDict()
    if starid == 'Kepler_1627':
        time, flux, flux_err, qual, texp = get_kep1627_kepler_lightcurve()
    else:
        raise NotImplementedError

    # NOTE: we have an abundance of data. so... drop all non-zero quality
    # flags.
    sel = (qual == 0)

    datasets['keplerllc'] = [time[sel], flux[sel], flux_err[sel], texp]

    priorpath = os.path.join(DATADIR, 'priors', f'{starid}_priors.py')
    assert os.path.exists(priorpath)
    priormod = SourceFileLoader('prior', priorpath).load_module()
    priordict = priormod.priordict

    pklpath = os.path.join(BETTYDIR, f'run_{starid}_{modelid}.pkl')
    PLOTDIR = outdir

    m = ModelFitter(modelid, datasets, priordict, plotdir=PLOTDIR,
                    pklpath=pklpath, overwrite=0, N_samples=N_samples,
                    N_cores=os.cpu_count())

    # END-COPYPASTE FROM RUN_GPTRANSIT.PY
    ##########################################

    # make plot
    plt.close('all')
    set_style()

    fig = plt.figure(figsize=(5,3))
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
            flat_flux, trend_flux = flatten_starspots(
                time, flux, flux_err, P_rotation
            )
        elif method == 'itergp':
            from betty.mapfitroutines import flatten_starspots
            flat_flux, trend_flux = flatten_starspots(
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

    # get data
    datadir = os.path.join(DATADIR, 'ttv', 'Bouma', 'K05245.01/')
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

    z1 = np.poly1d(
        np.polyfit(slopes[np.abs(ttv)<ttv_cut],
                   ttv[np.abs(ttv)<ttv_cut], deg=1)
    )

    slope_guess = -5e-2
    intercept_guess = 0
    sel = np.abs(ttv)<ttv_cut
    # ignore the error bars... because they are probably wrong.
    p_opt, p_cov = curve_fit(
        _linear_model, slopes[sel], ttv[sel],
        p0=(slope_guess, intercept_guess)
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

    # 1000 will give ppt per day...
    # for context, the light curve changes by ~50 ppt per half rotation period
    # (which is ~1.3 days).
    XMULT = 1e3

    sel = np.abs(ttv)<ttv_cut
    slopes,ttv,tc_err = slopes[sel],ttv[sel],tc_err[sel]

    ax.errorbar(
        XMULT*slopes, ttv*24*60, tc_err*24*60,
        marker='o', elinewidth=0.5, capsize=4, lw=0, mew=0.5, color='k',
        markersize=3, zorder=5
    )

    t0 = np.linspace(-0.2, 0.2, 100)

    #ax.plot(24*t0, z1(t0)*24*60, '-', label='linear fit', color='C1', alpha=0.6)

    ax.plot(XMULT*t0, _linear_model(t0, lsfit_slope, lsfit_int)*24*60, '-',
            label='Best fit', color='C2', alpha=1, lw=1)

    ax.fill_between(XMULT*t0,
                    _linear_model(t0, lsfit_slope-lsfit_slope_err, lsfit_int)*24*60,
                    _linear_model(t0, lsfit_slope+lsfit_slope_err, lsfit_int)*24*60,
                    alpha=0.6,
                    color='C2', lw=0, label='1$\sigma$',zorder=-1)

    ax.fill_between(XMULT*t0,
                    _linear_model(t0, lsfit_slope-3*lsfit_slope_err, lsfit_int)*24*60,
                    _linear_model(t0, lsfit_slope+3*lsfit_slope_err, lsfit_int)*24*60,
                    alpha=0.2,
                    color='C2', lw=0, label='3$\sigma$',zorder=-2)

    print(f'Slope: {lsfit_slope:.4f} +/- {lsfit_slope_err:.4f} ppt/day')
    print(f'implies {abs(lsfit_slope/lsfit_slope_err):.2f}σ different from zero.')
    print(f'Intercept: {lsfit_int*24*60:.5f} +/- {lsfit_int_err*24*60:.5f} minutes')


    ax.set_ylim([-0.03*24*60,0.03*24*60])
    ax.set_xlim([-0.15*XMULT,0.15*XMULT])

    ax.legend(fontsize='x-small')

    # set naming options
    s = ''

    outpath = os.path.join(outdir, f'ttv_vs_local_slope{s}.png')
    savefig(fig, outpath, dpi=400)


def plot_rotation_period_windowslider(outdir):

    from timmy.rotationperiod import measure_rotation_period_and_unc

    # get data
    modelid, starid = 'gptransit', 'Kepler_1627'
    datasets = OrderedDict()
    time, flux, flux_err, qual, texp = (
        get_kep1627_kepler_lightcurve(lctype='longcadence_byquarter')
    )

    N_quarters = len(time)

    periods, period_uncs = [], []
    for i in range(N_quarters):
        print(i)

        t,f = time[i],flux[i]

        plotpath = os.path.join(outdir, f'kep1627_rotationperiodslider_quarter_ix{i}.png')
        p, p_unc = measure_rotation_period_and_unc(t, f, 1, 10,
                                                   period_fit_cut=0.5, nterms=1,
                                                   samples_per_peak=50,
                                                   plotpath=plotpath)

        periods.append(p)
        period_uncs.append(p_unc)

    outdf = pd.DataFrame(
        {'period': periods, 'period_unc': period_uncs}
    )
    outpath = os.path.join(outdir, 'rotation_period_windowslider_QUARTERS.csv')
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
    cleanhrcut=1, extinctionmethod='gaia2018', smalllims=0, overplotkep1627=0
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
        if isochrone == 'mist':
            # see /doc/20210226_isochrones_theory.txt
            from timmy.read_mist_model import ISOCMD
            isocmdpath = os.path.join(DATADIR, 'isochrones',
                                      'MIST_iso_6039374449e9d.iso.cmd')
            # relevant params: star_mass log_g log_L log_Teff Gaia_RP_DR2Rev
            # Gaia_BP_DR2Rev Gaia_G_DR2Rev
            isocmd = ISOCMD(isocmdpath)
            assert len(isocmd.isocmds) > 1

        elif isochrone == 'parsec':
            #
            # see /doc/20210226_isochrones_theory.txt
            #
            # v4
            isopath = os.path.join(DATADIR, 'isochrones',
                                   'output360587063784.dat')
            nored_iso_df = pd.read_csv(isopath, delim_whitespace=True, comment='#')

            # v5
            isopath = os.path.join(DATADIR, 'isochrones',
                                   'output813364732851.dat')
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

    s = 2
    if smalllims:
        s = 4

    # mixed rasterizing along layers b/c we keep the loading times nice
    l0 = '$\delta$ Lyr cluster'
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
            print('Pleaides')
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

        get_xval_no_corr = lambda _df: np.array(_df[color0] - _df['phot_rp_mean_mag'])
        get_yval_no_corr = lambda _df: np.array(_df['phot_g_mean_mag'] + 5*np.log10(_df['parallax']/1e3) + 5)

        _x = get_xval_no_corr(df_bkgd)
        _y = get_yval_no_corr(df_bkgd)
        s = np.isfinite(_x) & np.isfinite(_y)
        density = ax.scatter_density(_x[s], _y[s], cmap='Greys')

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

        from earhart.priors import AVG_AG, AVG_EBpmRp

        if isochrone == 'mist':

            ages = [100, 178, 316]
            N_ages = len(ages)
            colors = plt.cm.cool(np.linspace(0,1,N_ages))[::-1]

            for i, (a, c) in enumerate(zip(ages, colors)):
                mstar = isocmd.isocmds[i]['star_mass']
                sel = (mstar < 7)

                corr = 7.85
                _yval = (
                    isocmd.isocmds[i]['Gaia_G_DR2Rev'][sel] +
                    5*np.log10(np.nanmedian(core_df['parallax']/1e3)) + 5
                    + AVG_AG
                    + corr
                )

                if color0 == 'phot_bp_mean_mag':
                    _c0 = 'Gaia_BP_DR2Rev'
                elif color0 == 'phot_g_mean_mag':
                    _c0 = 'Gaia_G_DR2Rev'
                else:
                    raise NotImplementedError

                _xval = (
                    isocmd.isocmds[i][_c0][sel]-isocmd.isocmds[i]['Gaia_RP_DR2Rev'][sel]
                    + AVG_EBpmRp
                )

                ax.plot(
                    _xval,
                    _yval,
                    c=c, alpha=1., zorder=7, label=f'{a} Myr', lw=0.5
                )

        elif isochrone == 'parsec':

            ages = [100, 178, 316]
            logages = [8, 8.25, 8.5]
            N_ages = len(ages)
            #colors = plt.cm.cividis(np.linspace(0,1,N_ages))
            colors = plt.cm.spring(np.linspace(0,1,N_ages))
            colors = ['red','gold','lime']

            for i, (a, la, c) in enumerate(zip(ages, logages, colors)):

                sel = (
                    (np.abs(iso_df.logAge - la) < 0.01) &
                    (iso_df.Mass < 7)
                )

                corr = 7.80
                #corr = 7.65
                #corr = 7.60
                _yval = (
                    iso_df[sel]['Gmag'] +
                    5*np.log10(np.nanmedian(core_df['parallax']/1e3)) + 5
                    + AVG_AG
                    + corr
                )
                sel2 = (_yval < 15) # numerical issue
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
                    c=c, ls='-', alpha=1., zorder=7, label=f'{a} Myr', lw=0.5
                )
                ax.plot(
                    _xval, _yval,
                    c=c, ls='--', alpha=1., zorder=6, lw=0.5
                )

                nored_y = (
                    nored_iso_df[sel]['Gmag'] +
                    5*np.log10(np.nanmedian(core_df['parallax']/1e3)) + 5
                    + AVG_AG
                    + corr
                )
                nored_y = nored_y[sel2] # same jank numerical issue
                nored_x = (
                    nored_iso_df[sel][sel2][_c0] - nored_iso_df[sel][sel2]['G_RPmag']
                )

                diff_x = -(nored_x - _xval)
                diff_y = -(nored_y - _yval)

                # SED-dependent reddening check, usually off.
                print(42*'*')
                print(f'Median Bp-Rp difference: {np.median(diff_x):.4f}')
                print(42*'*')
                if i == 0:
                    sep = 2
                    # # NOTE: to show EVERYTHING
                    # ax.quiver(
                    #     nored_x[::sep], nored_y[::sep], diff_x[::sep], diff_y[::sep], angles='xy',
                    #     scale_units='xy', scale=1, color='magenta',
                    #     width=1e-3, linewidths=1, headwidth=5, zorder=9
                    # )

                    ax.quiver(
                        2.6, 3.5, np.nanmedian(diff_x[::sep]),
                        np.nanmedian(diff_y[::sep]), angles='xy',
                        scale_units='xy', scale=1, color='black',
                        width=3e-3, linewidths=2, headwidth=5, zorder=9
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
        ax.set_xlim([1,3.6])
        ax.set_ylim([12.5,5.5])
    elif smalllims and 'phot_bp_mean_mag' not in color0:
        raise NotImplementedError

    format_ax(ax)
    ax.tick_params(axis='x', which='both', top=False)


    #
    # append SpTypes (ignoring reddening)
    #
    from rudolf.priors import AVG_EBpmRp, AVG_EGmRp
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

    outpath = os.path.join(outdir, f'hr{s}{c0s}.png')

    savefig(f, outpath, dpi=400)


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
        l = '$\delta$ Lyr cluster'
    else:
        raise NotImplementedError
    labels = ['Pleaides', 'Praesepe', l]

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
                marker='o', linewidths=_lw, label="Astrometric binary"
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



