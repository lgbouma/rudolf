"""
plot_ruwe_vs_apparentmag
plot_simulated_RM
plot_skychart
plot_XYZvtang
"""
import os, corner, pickle, inspect
from glob import glob
from datetime import datetime
import numpy as np, matplotlib.pyplot as plt, pandas as pd, pymc3 as pm
from numpy import array as nparr
from collections import Counter

import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs

from astropy import units as u, constants as const
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.table import Table

import matplotlib.patheffects as pe
from matplotlib.ticker import MaxNLocator

from aesthetic.plot import savefig, format_ax, set_style

from astrobase.services.identifiers import gaiadr2_to_tic
from cdips.utils.gaiaqueries import (
    given_source_ids_get_gaia_data
)
from cdips.utils.tapqueries import given_source_ids_get_tic8_data
from cdips.utils.plotutils import rainbow_text
from cdips.utils.mamajek import get_interp_BpmRp_from_Teff

from rudolf.paths import DATADIR, RESULTSDIR
from rudolf.helpers import (
    get_gaia_cluster_data, get_simulated_RM_data, get_keplerfield_dict
)


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
        markersize=10, marker='*', color='black', lw=0
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


def plot_skychart(outdir, narrowlims=0, showkepler=0, showtess=0, shownakedeye=0):

    set_style()

    df_dr2, df_edr3, trgt_df = get_gaia_cluster_data()

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
            label='$\delta$ Lyr Cluster', marker='.'
        )
        ax.plot(
            get_xval(trgt_df), get_yval(trgt_df), alpha=1, mew=0.5,
            zorder=8, label='Kepler 1627', markerfacecolor='yellow',
            markersize=10, marker='*', color='black', lw=0
        )

    if showkepler:
        kep_d = get_keplerfield_dict()
        for mod in np.sort(list(kep_d.keys())):
            for op in np.sort(list(kep_d[mod].keys())):
                this = kep_d[mod][op]
                ra, dec = nparr(this['corners_ra']), nparr(this['corners_dec'])
                ax.fill(ra, dec, c='lightgray', alpha=0.95, lw=0,
                        rasterized=True, zorder=-1)

    if showtess:

        outcsv = os.path.join(outdir, 'showtess_stephenson1_cache.csv')

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
            stars_names = stars[pd.notnull(stars['proper'])]
            sel_names = ['Vega', 'Albireo', 'Deneb']
            stars_names = stars_names[
                stars_names.proper.str.contains('|'.join(sel_names))
            ]
            stars_names = stars_names[
                ~stars_names.proper.str.contains('Denebola')
            ]
            deltay=0.6
            for index, row in stars_names.iterrows():
                ax.text(row['ra'], row['dec']+deltay, row['proper'], ha='center',
                        va='bottom', fontsize='xx-small', bbox=bbox)



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

    bn = inspect.stack()[0][3].split("_")[1]
    outpath = os.path.join(outdir, f'{bn}{s}.png')
    savefig(f, outpath, dpi=400)


def plot_XYZvtang(outdir, show1627):

    plt.close('all')
    set_style()

    df_dr2, df_edr3, trgt_df = get_gaia_cluster_data()
    # set "dr2_radial_velocity" according to Andrew Howard HIRES recon
    # spectrum. agrees with -16.9km/s+/-0.5km/s TRES.
    trgt_df['dr2_radial_velocity'] = -16.7

    from earhart.physicalpositions import append_physicalpositions
    df_edr3 = append_physicalpositions(df_edr3, trgt_df)
    trgt_df = append_physicalpositions(trgt_df, med_df)

    # use plx S/N>20 to get good XYZ.
    sdf = df_edr3[df_edr3.parallax_over_error > 20]

    plt.close('all')

    fig = plt.figure(figsize=(8,3), constrained_layout=True)
    axd = fig.subplot_mosaic(
        """
        ABBDD
        ACCDD
        """,
        gridspec_kw={
            "width_ratios": [1, 1, 1, 1, 1]
        }
    )

    xydict = [
        "A":('x_pc', 'y_pc'),
        "B":('x_pc', 'z_pc'),
        "C":('y_pc', 'z_pc'),
        "D":('delta_pmra_prime_km_s', 'delta_pmdec_prime_km_s')
    ]
    ldict = {
        'x_pc':'X [pc]',
        'y_pc':'Y [pc]',
        'z_pc':'Z [pc]',
        'delta_pmra_prime_km_s': r"$\Delta \mu_{{\alpha'}}^{*}$ [km$\,$s$^{-1}$]",
        'delta_pmdec_prime_km_s': r"$\Delta \mu_{\delta}^{*}$ [km$\,$s$^{-1}$]"
    }

    for k,v in xydict.items():

        xv, yv = v[0], v[1]

        axd[k].scatter(
            sdf[xv], sdf[yv], c='k', alpha=1, zorder=7, s=2, edgecolors='none',
            rasterized=True, marker='.'
        )

        if show1627:
            axd[k].plot(
                trgt_df[xv], trgt_df[yv], alpha=1, mew=0.5,
                zorder=8, label='Kepler 1627', markerfacecolor='yellow',
                markersize=7, marker='*', color='black', lw=0
            )


        axd[k].update({'xlabel': ldict[xv], 'ylabel': ldict[yv]})

        # add orientation arrows
        if k == 'A':
            delta_x = 0.1
            axd['A'].arrow(0.73, 0.07, delta_x, 0, length_includes_head=True,
                           head_width=1e-2, head_length=1e-2,
                           transform=axd['A'].transAxes)
            axd['A'].text(0.73+delta_x/2, 0.085, 'Galactic center',
                          va='bottom', ha='center',
                          transform=axd['A'].transAxes, fontsize='xx-small')
            axd['A'].arrow(0.07, 0.73, 0, delta_x, length_includes_head=True,
                           head_width=1e-2, head_length=1e-2,
                           transform=axd['A'].transAxes)
            axd['A'].text(0.085, 0.73+delta_x/2, 'Galactic rotation',
                          va='center', ha='left', transform=axd['A'].transAxes,
                          fontsize='xx-small', rotation=90)

        elif k == 'B':
            delta_x = 0.1
            axd['B'].arrow(0.73, 0.07, delta_x, 0, length_includes_head=True,
                           head_width=1e-2, head_length=1e-2,
                           transform=axs[1].transAxes)
            axd['B'].text(0.73+delta_x/2, 0.085, 'Galactic center',
                          va='bottom', ha='center', transform=axs[1].transAxes,
                          fontsize='xx-small')

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


    s = ''
    if show1627:
        s += "show1627"

    bn = inspect.stack()[0][3].split("_")[1]
    outpath = os.path.join(outdir, f'{bn}{s}.png')
    savefig(f, outpath, dpi=400)
