import numpy as np, pandas as pd, matplotlib.pyplot as plt
import os, pickle
from numpy import array as nparr

from rudolf.paths import DATADIR, RESULTSDIR

from rudolf.helpers import (
    get_deltalyr_kc19_gaia_data, get_simulated_RM_data,
    get_keplerfieldfootprint_dict, get_deltalyr_kc19_comovers,
    get_deltalyr_kc19_cleansubset, get_manually_downloaded_kepler_lightcurve,
    get_gaia_catalog_of_nearby_stars, get_clustermembers_cg18_subset,
    get_mutau_members, get_ScoOB2_members,
    supplement_gaia_stars_extinctions_corrected_photometry,
    get_clean_gaia_photometric_sources
)
from cdips.utils.gaiaqueries import given_source_ids_get_gaia_data

from aesthetic.plot import savefig, format_ax, set_style

from scipy.ndimage.filters import uniform_filter
from scipy.interpolate import UnivariateSpline, interp1d

from astropy.modeling import models, fitting

extinctionmethod = 'gaia2018'

#
# getters made by plot_hr
#
def _get_deltalyr():
    deltalyrpath = os.path.join(
        RESULTSDIR, 'tables', 'deltalyr_kc19_cleansubset_withreddening.csv'
    )
    df = pd.read_csv(deltalyrpath)
    df = df[get_clean_gaia_photometric_sources(df)]
    return df

def _get_ucl():
    outpath = os.path.join(
        RESULTSDIR, 'tables', f'UCL_withreddening_{extinctionmethod}.csv'
    )
    _df = pd.read_csv(outpath)
    _df = _df[get_clean_gaia_photometric_sources(_df)]
    ruwe_df =  given_source_ids_get_gaia_data(
        nparr(_df.source_id).astype(np.int64), 'UCL_rudolf_dr2_ruwe',
        n_max=10000, overwrite=False, gaia_datarelease='gaiadr2',
        getdr2ruwe=True
    )
    mdf = _df.merge(ruwe_df, how='left', on='source_id')
    return mdf

def _get_ic2602():
    outpath = os.path.join(
        RESULTSDIR, 'tables', f'IC_2602_withreddening_{extinctionmethod}.csv'
    )
    _df = pd.read_csv(outpath)
    _df = _df[get_clean_gaia_photometric_sources(_df)]
    ruwe_df =  given_source_ids_get_gaia_data(
        nparr(_df.source_id).astype(np.int64), 'ic2602_rudolf_dr2_ruwe',
        n_max=10000, overwrite=False, gaia_datarelease='gaiadr2',
        getdr2ruwe=True
    )
    mdf = _df.merge(ruwe_df, how='left', on='source_id')
    return mdf

def _get_pleiades():
    outpath = os.path.join(
        RESULTSDIR, 'tables', f'Pleiades_withreddening_{extinctionmethod}.csv'
    )
    _df = pd.read_csv(outpath)
    _df = _df[get_clean_gaia_photometric_sources(_df)]
    ruwe_df =  given_source_ids_get_gaia_data(
        nparr(_df.source_id).astype(np.int64), 'pleiades_rudolf_dr2_ruwe',
        n_max=10000, overwrite=False, gaia_datarelease='gaiadr2',
        getdr2ruwe=True
    )
    mdf = _df.merge(ruwe_df, how='left', on='source_id')
    return mdf


def collect_isochrone_data():

    # hardcode options
    reddening_corr = 1
    extinctionmethod = 'gaia2018'
    color0 = 'phot_bp_mean_mag'

    # NOTE: need to assume hard-coded ages for the reference clusters
    age_dict = {
        'UCL': 16e6, # Preibisch & Mamajek 2008, Table 11, UCL
        'IC 2602': 38e6,
        'Pleiades': 112e6 # Dahn 2015
    }

    #
    # begin by defining "getters"
    #
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

    # save the HR diagram information, and the resulting binned values etc, as
    # entries in a dict.  structure is `hr_dict[cluster_name][parameter]`.
    hr_dict = {}
    clusters = ['δ Lyr cluster', 'IC 2602', 'Pleiades', 'UCL']
    getclusterfn = [_get_deltalyr, _get_ic2602, _get_pleiades, _get_ucl]

    for c,fn in zip(clusters, getclusterfn):

        print(c+'...')
        _df = fn()
        hr_dict[c] = {}

        xval = get_xval(_df)
        yval = get_yval(_df)

        outpath = os.path.join(RESULTSDIR, 'empirical_isochrone_age',
                               f"{c.replace(' ','_')}_forglue.csv")
        _df['BpmRp0'] = xval
        _df['MG0'] = yval
        selcols = ['source_id','BpmRp0','MG0']
        _df[selcols].to_csv(outpath, index=False)
        print(f'Wrote {outpath}...')

        # step 1: cut on RUWE<1.3
        assert 'ruwe' in _df

        RUWE_CUTOFF = 1.3
        sel = _df.ruwe < RUWE_CUTOFF
        print(f'RUWE<{RUWE_CUTOFF}: {len(_df[sel])}/{len(_df)}')

        outpath = os.path.join(RESULTSDIR, 'empirical_isochrone_age',
                               f"{c.replace(' ','_')}_gmag_vs_ruwe.png")
        f,ax = plt.subplots(figsize=(4,3))
        ax.scatter(_df.phot_g_mean_mag, _df.ruwe, s=1, c='k')
        ax.axhline(RUWE_CUTOFF, color="#aaaaaa", lw=0.5, zorder=-1)
        ax.update({'xlabel': 'G [mag]', 'ylabel': 'RUWE'})
        savefig(f, outpath, dpi=400)

        # step 2: cut on RV_error < 80th pctile
        rvkey = (
            'dr2_radial_velocity' if 'dr2_radial_velocity' in _df else
            'radial_velocity'
        )

        cutval = np.nanpercentile(_df[rvkey+'_error'], 80)
        sel &= (
            (_df[rvkey+'_error'] < cutval)
            |
            (pd.isnull(_df[rvkey+'_error']))
        )
        print(f'...& RV_error<80th pct ({cutval:.2f} km/s): {len(_df[sel])}/{len(_df)}')

        outpath = os.path.join(RESULTSDIR, 'empirical_isochrone_age',
                               f"{c.replace(' ','_')}_gmag_vs_rverror.png")
        f,ax = plt.subplots(figsize=(4,3))
        ax.scatter(_df.phot_g_mean_mag, _df[rvkey+'_error'], s=1, c='k')
        ax.axhline(cutval, color="#aaaaaa", lw=0.5, zorder=-1)
        ax.update({'xlabel': 'G [mag]', 'ylabel': 'DR2 RV error [km/s]'})
        savefig(f, outpath, dpi=400)

        # step 3: unresolved binaries manual.  these CSV files were manually
        # made by lasso'ing in glue.

        inpath = os.path.join(RESULTSDIR, 'empirical_isochrone_age',
                               f"{c.replace(' ','_')}_manual_CMD_lasso.csv")
        lasso_df = pd.read_csv(inpath)

        sel &= (
            _df.source_id.isin(lasso_df.source_id)
        )
        print(f'...& in manual photometric CMD lasso: {len(_df[sel])}/{len(_df)}')

        outpath = os.path.join(RESULTSDIR, 'empirical_isochrone_age',
                               f"{c.replace(' ','_')}_CMD.png")
        f,ax = plt.subplots(figsize=(6,4))
        ax.scatter(xval, yval, s=1, c='k', zorder=1)
        ax.scatter(xval[sel], yval[sel], s=1, c='C1', zorder=2)
        ax.update({'xlabel': '(Bp-Rp)0 [mag]', 'ylabel': '(MG)0'})
        savefig(f, outpath, dpi=400)

        # step 4: moving box average and stdevn in 0.10 mag bins, as suggested
        # by Gagne+2020
        xs, ys = xval[sel], yval[sel]

        dMag = 0.1
        bins = np.arange(0, 3+dMag, dMag)
        ymean,ystdev,N_in_bin,binmids = [],[],[],[]
        for bin_start in bins[:-1]:
            bin_end = bin_start + dMag

            in_bin = (bin_start < xs) & (bin_end > xs)

            N = len(xs[in_bin])
            N_in_bin.append(N)
            binmids.append(bin_start+dMag/2)

            if N >= 2:
                ymean.append(np.average(ys[in_bin]))
                ystdev.append(np.std(ys[in_bin]))
            else:
                ymean.append(np.nan)
                ystdev.append(np.nan)

        N_in_bin = nparr(N_in_bin)
        ymean = nparr(ymean)
        ystdev = nparr(ystdev)
        binmids = nparr(binmids)

        _s = np.isfinite(ymean)

        # cubic spline
        fn_BpmRp_to_AbsG = UnivariateSpline(binmids[_s], ymean[_s],
                                            k=3, s=1e-2, ext=1)

        # this is the (Bp-Rp)0 component
        x_interp = np.linspace(0.8, 2.9, 1001) #NOTE: for clean plots
        x_eval = np.arange(0.85, 2.85+0.1, dMag) # NOTE: to match in probability calc

        y_interp = fn_BpmRp_to_AbsG(x_interp)
        y_eval = fn_BpmRp_to_AbsG(x_eval)

        outpath = os.path.join(RESULTSDIR, 'empirical_isochrone_age',
                               f"{c.replace(' ','_')}_CMD_binning.png")
        f,ax = plt.subplots(figsize=(6,4))
        ax.scatter(xs, ys, s=1, c='C0', zorder=1)
        ax.errorbar(binmids, ymean, ystdev, marker='o', elinewidth=0.5,
                    capsize=4, lw=0, mew=0.5, color='k', markersize=3,
                    zorder=5)
        ax.plot(x_interp, y_interp, lw=1, c='C1', zorder=42)
        ax.update({'xlabel': '(Bp-Rp)0 [mag]', 'ylabel': '(MG)0'})
        savefig(f, outpath, dpi=400)

        hr_dict[c]['(Bp-Rp)0_fullsample'] = xval
        hr_dict[c]['(MG)0_fullsample'] = yval
        hr_dict[c]['(Bp-Rp)0'] = xval[sel]
        hr_dict[c]['(MG)0'] = yval[sel]
        hr_dict[c]['binmids'] = binmids
        hr_dict[c]['ymean'] = ymean
        hr_dict[c]['ystdev'] = ystdev
        hr_dict[c]['ymean_eval'] = ymean[(binmids >= 0.80) & (binmids <= 2.85)]
        hr_dict[c]['ystdev_eval'] = ystdev[(binmids >= 0.80) & (binmids <= 2.85)]
        hr_dict[c]['N_in_bin'] = N_in_bin
        hr_dict[c]['x_interp'] = x_interp
        hr_dict[c]['y_interp'] = y_interp
        hr_dict[c]['x_eval'] = x_eval
        hr_dict[c]['y_eval'] = y_eval
        hr_dict[c]['fn_BpmRp_to_AbsG'] = fn_BpmRp_to_AbsG

    # now make the grid!
    alpha = np.linspace(0,1,1000)

    # step1: UCL to IC2602:
    # isochrones (N_yeval, N_age_spaced_grid)
    I_step1 = (
        (1-alpha[None,:])*hr_dict['UCL']['y_eval'][:,None]
        + (alpha[None,:])*hr_dict['IC 2602']['y_eval'][:,None]
    )
    # step2: IC2602 to Pleiades:
    I_step2 = (
        (1-alpha[None,:])*hr_dict['IC 2602']['y_eval'][:,None]
        + (alpha[None,:])*hr_dict['Pleiades']['y_eval'][:,None]
    )

    # isochrone grid: (N_yeval, 2*N_age_spaced_grid)  (~50 X 2000)
    I_grid = np.hstack((I_step1, I_step2))

    # make evaluation plot
    outpath = os.path.join(RESULTSDIR, 'empirical_isochrone_age',
                           f"all_clusters_CMD_binning_interp.png")
    set_style()
    f,ax = plt.subplots(figsize=(4,4))

    colors = 'k,orange,deepskyblue,C1'.split(',')

    for ix, c, color in zip(range(len(clusters)), clusters, colors):

        #label = c+' spline'
        label = c
        ax.scatter(hr_dict[c]['(Bp-Rp)0'], hr_dict[c]['(MG)0'], s=0.25,
                   c=color, zorder=ix)
        #ax.errorbar(hr_dict[c]['binmids'], hr_dict[c]['ymean'],
        #            hr_dict[c]['ystdev'], marker='o', elinewidth=0.5,
        #            capsize=4, lw=0, mew=0.5, color=f'k', markersize=3,
        #            zorder=5)
        ax.plot(hr_dict[c]['x_interp'], hr_dict[c]['y_interp'], lw=0.5,
                c=color, zorder=42, label=label)

    for i in range(0,I_grid.shape[1],200):
        if i == 0:
            ax.plot(x_eval, I_grid[:,i], color='gray', lw=0.25,
                    label='Interpolated splines')
        else:
            ax.plot(x_eval, I_grid[:,i], color='gray', lw=0.25)

    ax.update({'xlabel': '$(G_{\mathrm{BP}}-G_{\mathrm{RP}})_0$ [mag]',
               'ylabel': '$M_{\mathrm{G},0}$ [mag]'})
    ax.legend(loc='best', fontsize='x-small')
    ax.set_ylim([12.5,3.5])
    ax.set_xlim([0.5,3.5])
    savefig(f, outpath, dpi=400)

    # calculate implied ages

    log10_A_step1 = (
        (1-alpha) * np.log10(age_dict['UCL']) +
        (alpha)*np.log10(age_dict['IC 2602'])
    )
    log10_A_step2 = (
        (1-alpha)* np.log10(age_dict['IC 2602']) +
        (alpha)*np.log10(age_dict['Pleiades'])
    )
    log10_A_grid = np.hstack((log10_A_step1, log10_A_step2))

    # calculate the age probabilities assuming a gaussian likelihood
    JITTER = 0.30

    for c in ['δ Lyr cluster', 'IC 2602']:

        ln_P_grid = -0.5 * np.nansum(
            (
                (hr_dict[c]['ymean_eval'][:,None] - I_grid) /
                (hr_dict[c]['ystdev_eval'][:,None] + JITTER )
            )**2
        ,axis=0)

        hr_dict[c]['ln_P_grid'] = ln_P_grid
        hr_dict[c]['P_grid'] = np.exp(ln_P_grid)
        hr_dict[c]['log10_A_grid'] = log10_A_grid

        # fit gaussian to get mean and std-devn
        g_init = models.Gaussian1D(
            amplitude=0.2, mean=log10_A_grid[np.argmax(ln_P_grid)], stddev=0.2
        )

        fit_g = fitting.LevMarLSQFitter()
        g_fit = fit_g(g_init, log10_A_grid, np.exp(ln_P_grid))

        txt = (
            f'{c}\n'+'$\log_{\mathrm{10}}$t [yr]: '+f'{g_fit.mean.value:.2f} +/- {g_fit.stddev.value:.2f}'
            + '\n' +
            f't [Myr]: {10**(g_fit.mean.value)/1e6:.1f} '+
            f'+{(10**(g_fit.mean.value+g_fit.stddev.value) - 10**(g_fit.mean.value))/1e6:.1f} '
            f'-{(10**(g_fit.mean.value) - 10**(g_fit.mean.value-g_fit.stddev.value))/1e6:.1f}'
        )
        print(txt)

        # check
        plt.close('all')
        outpath = os.path.join(RESULTSDIR, 'empirical_isochrone_age',
                               f"{c.replace(' ','_')}_age_posterior.png")
        f,ax = plt.subplots(figsize=(4,3))
        ax.plot(log10_A_grid, np.exp(ln_P_grid), lw=1, c='k')
        ax.text(0.97,0.97,txt, transform=ax.transAxes, ha='right',va='top', color='k', fontsize='xx-small')
        ax.update({'xlabel': '$\log_{10}t$ [yr]', 'ylabel': 'Relative probability'})
        savefig(f, outpath, dpi=400)

    cachepath = os.path.join(RESULTSDIR, 'empirical_isochrone_age',
                           f"isochrone_data_cache.pkl")
    with open(cachepath, "wb") as f:
        pickle.dump(c, f)

    print(f'Wrote {cachepath}')

if __name__ == "__main__":
    collect_isochrone_data()
