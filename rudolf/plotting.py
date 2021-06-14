"""
plot_TEMPLATE

plot_ruwe_vs_apparentmag
plot_simulated_RM
plot_skychart
plot_XYZvtang
plot_keplerlc
    _plot_zoom_light_curve
plot_flare_checker
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
    given_source_ids_get_gaia_data
)
from cdips.utils.tapqueries import given_source_ids_get_tic8_data
from cdips.utils.plotutils import rainbow_text
from cdips.utils.mamajek import get_interp_BpmRp_from_Teff

from rudolf.paths import DATADIR, RESULTSDIR
from rudolf.helpers import (
    get_gaia_cluster_data, get_simulated_RM_data,
    get_keplerfieldfootprint_dict, get_comovers,
    get_kep1627_kepler_lightcurve
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

    df_dr2, df_edr3, trgt_df = get_gaia_cluster_data()

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

    df_dr2, df_edr3, trgt_df = get_gaia_cluster_data()
    if showcomovers:
        df_edr3 = get_comovers()

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
            alpha=0.9, zorder=40, s=2, rasterized=True, linewidths=0,
            marker='.', norm=norm,
        )

        trgt_id = "2103737241426734336" # Kepler 1627
        trgt_mdf = mdf[mdf.source_id.astype(str) == trgt_id]

        cax = ax.scatter(
            trgt_mdf.ra, trgt_mdf.dec, c=trgt_mdf.n_tess_sector, cmap=cmap,
            alpha=1, zorder=42, s=60, linewidths=0.2,
            marker='*', norm=norm, edgecolors='k'
        )

        cb = f.colorbar(cax, extend='max', ticks=ticks)
        cb.ax.minorticks_off()

        cb.set_label("TESS Sectors", rotation=270, labelpad=10)

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


def plot_XYZvtang(outdir, show_1627=0, save_candcomovers=1, show_comovers=0,
                  show_sun=0):

    plt.close('all')
    set_style()

    df_dr2, df_edr3, trgt_df = get_gaia_cluster_data()
    # set "dr2_radial_velocity" according to Andrew Howard HIRES recon
    # spectrum. agrees with -16.9km/s+/-0.5km/s TRES.
    trgt_df.dr2_radial_velocity = -16.7

    from earhart.physicalpositions import append_physicalpositions
    df_edr3 = append_physicalpositions(df_edr3, trgt_df)
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

    # use plx S/N>20 to get good XYZ.
    sdf = df_edr3[df_edr3.parallax_over_error > 20]
    scmdf = cm_df_edr3[cm_df_edr3.parallax_over_error > 20]

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

    fig = plt.figure(figsize=(4,3))
    axd = fig.subplot_mosaic(
        """
        A
        B
        """,
        gridspec_kw={
            "height_ratios": [3, 1.5]
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
                          t0, wrap=False, sort=True)
    x_fold = _pd['phase']
    y_fold = _pd['mags']

    axd['A'].set_xlabel('')
    axd['A'].set_xticklabels([])

    #axd['B'].vlines(
    #    x_fold, 0, 1, ls='-', lw=0.5, colors='k'
    #)
    axd['B'].scatter(
        x_fold, y_fold, c='k', s=2, rasterized=True, linewidths=0,
        zorder=1

    )

    #axd['B'].set_ylim([0,1])
    axd['B'].set_yscale('log')
    axd['B'].set_ylabel('Ampl.')
    axd['B'].set_xlabel('Phase')
    axd['B'].set_ylim([1e-3,1e-1])
    #axd['B'].set_yticklabels([])

    axd['A'].set_xlim([-0.1,1.1])
    axd['B'].set_xlim([-0.1,1.1])

    fig.tight_layout()
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

