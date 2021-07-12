"""
Data getters:
    δ Lyr Cluster:
        get_deltalyr_kc19_gaia_data
        get_deltalyr_kc19_comovers
        get_deltalyr_kc19_cleansubset
    Kepler 1627:
        get_kep1627_kepler_lightcurve
        get_keplerfieldfootprint_dict
        get_flare_df

    Other stellar and cluster datasets:
        get_gaia_catalog_of_nearby_stars
        get_clustermembers_cg18_subset

    Supplement a set of Gaia stars with extinctions and corrected photometry:
        supplement_gaia_stars_extinctions_corrected_photometry

Proposal/RM-related:
    get_simulated_RM_data

One-offs to get the Stephenson-1 information:
    get_candidate_stephenson1_member_list
    supplement_sourcelist_with_gaiainfo
"""
import os, collections, pickle
import numpy as np, pandas as pd
from glob import glob
from copy import deepcopy

from numpy import array as nparr

from astropy.io import fits
from astropy import units as u
from astropy.table import Table
from astroquery.vizier import Vizier
from astroquery.xmatch import XMatch
from astropy.coordinates import SkyCoord

import cdips.utils.lcutils as lcu
import cdips.lcproc.detrend as dtr
import cdips.lcproc.mask_orbit_edges as moe

from cdips.utils.catalogs import (
    get_cdips_catalog, get_tic_star_information
)
from cdips.utils.gaiaqueries import (
    query_neighborhood, given_source_ids_get_gaia_data,
    given_dr2_sourceids_get_edr3_xmatch
)

from rudolf.paths import DATADIR, RESULTSDIR

def get_flare_df():

    from rudolf.plotting import _get_detrended_flare_data

    flaredir = os.path.join(RESULTSDIR, 'flares')

    # read data
    method = 'itergp'
    cachepath = os.path.join(flaredir, f'flare_checker_cache_{method}.pkl')
    c = _get_detrended_flare_data(cachepath, method)
    flpath = os.path.join(flaredir, f'fldict_{method}.csv')
    df = pd.read_csv(flpath)

    FL_AMP_CUTOFF = 5e-3
    sel = df.ampl_rec > FL_AMP_CUTOFF
    sdf = df[sel]

    return sdf


def get_kep1627_kepler_lightcurve(lctype='longcadence'):
    """
    Collect and stitch available Kepler quarters, after median-normalizing in
    each quater.
    """

    assert lctype in ['longcadence', 'shortcadence', 'longcadence_byquarter']

    if lctype in ['longcadence', 'longcadence_byquarter']:
        lcfiles = glob(os.path.join(DATADIR, 'phot', 'kplr*_llc.fits'))
    elif lctype == 'shortcadence':
        lcfiles = glob(os.path.join(DATADIR, 'phot', 'full_MAST_sc', 'MAST_*',
                                    'Kepler', 'kplr006184894*', 'kplr*_slc.fits'))
    else:
        raise NotImplementedError('could do short cadence here too')
    assert len(lcfiles) > 1

    timelist,f_list,ferr_list,qual_list,texp_list = [],[],[],[],[]

    for lcfile in lcfiles:

        hdul = fits.open(lcfile)
        d = hdul[1].data

        yval = 'PDCSAP_FLUX'
        time = d['TIME']
        _f, _f_err = d[yval], d[yval+'_ERR']
        flux = _f/np.nanmedian(_f)
        flux_err = _f_err/np.nanmedian(_f)
        qual = d['SAP_QUALITY']

        sel = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err)

        texp = np.nanmedian(np.diff(time[sel]))

        timelist.append(time[sel])
        f_list.append(flux[sel])
        ferr_list.append(flux_err[sel])
        qual_list.append(qual[sel])
        texp_list.append(texp)

    if lctype == 'longcadence' or 'shortcadence':
        time = np.hstack(timelist)
        flux = np.hstack(f_list)
        flux_err = np.hstack(ferr_list)
        qual = np.hstack(qual_list)
        texp = np.nanmedian(texp_list)
    elif lctype == 'longcadence_byquarter':
        return (
            timelist,
            f_list,
            ferr_list,
            qual_list,
            texp_list
        )

    # require ascending time
    s = np.argsort(time)
    flux = flux[s]
    flux_err = flux_err[s]
    qual = qual[s]
    time = time[s]
    assert np.all(np.diff(time) > 0)

    return (
        time.astype(np.float64),
        flux.astype(np.float64),
        flux_err.astype(np.float64),
        qual,
        texp
    )


def get_candidate_stephenson1_member_list():

    outpath = os.path.join(DATADIR, 'gaia', 'stephenson1_kc19.csv')

    if not os.path.exists(outpath):
        # Kounkel & Covey 2019 Stephenson1 candidate member list.
        csvpath = os.path.join(DATADIR, 'gaia', 'string_table1.csv')
        df = pd.read_csv(csvpath)

        sdf = df[np.array(df.group_id).astype(int) == 73]

        sdf['source_id'].to_csv(outpath, index=False)

    return pd.read_csv(outpath)


def supplement_sourcelist_with_gaiainfo(df):

    groupname = 'stephenson1'

    dr2_source_ids = np.array(df.source_id).astype(np.int64)

    dr2_x_edr3_df = given_dr2_sourceids_get_edr3_xmatch(
        dr2_source_ids, groupname, overwrite=True,
        enforce_all_sourceids_viable=True
    )

    # Take the closest (proper motion and epoch-corrected) angular distance as
    # THE single match.
    get_edr3_xm = lambda _df: (
        _df.sort_values(by='angular_distance').
        drop_duplicates(subset='dr2_source_id', keep='first')
    )
    s_edr3 = get_edr3_xm(dr2_x_edr3_df)

    edr3_source_ids = np.array(s_edr3.dr3_source_id).astype(np.int64)

    # get gaia dr2 data
    df_dr2 = given_source_ids_get_gaia_data(dr2_source_ids, groupname, n_max=10000,
                                            overwrite=True,
                                            enforce_all_sourceids_viable=True,
                                            savstr='',
                                            gaia_datarelease='gaiadr2')

    df_edr3 = given_source_ids_get_gaia_data(edr3_source_ids, groupname,
                                             n_max=10000, overwrite=True,
                                             enforce_all_sourceids_viable=True,
                                             savstr='',
                                             gaia_datarelease='gaiaedr3')

    outpath_dr2 = os.path.join(DATADIR, 'gaia', 'stephenson1_kc19_dr2.csv')
    outpath_edr3 = os.path.join(DATADIR, 'gaia', 'stephenson1_kc19_edr3.csv')
    outpath_dr2xedr3 = os.path.join(DATADIR, 'gaia', 'stephenson1_kc19_dr2xedr3.csv')

    df_dr2.to_csv(outpath_dr2, index=False)
    df_edr3.to_csv(outpath_edr3, index=False)
    dr2_x_edr3_df.to_csv(outpath_dr2xedr3, index=False)


def get_deltalyr_kc19_gaia_data():
    """
    Get all Kounkel & Covey 2019 "Stephenson 1" members.
    """

    outpath_dr2 = os.path.join(DATADIR, 'gaia', 'stephenson1_kc19_dr2.csv')
    outpath_edr3 = os.path.join(DATADIR, 'gaia', 'stephenson1_kc19_edr3.csv')

    if not os.path.exists(outpath_dr2):

        df = get_candidate_stephenson1_member_list()

        supplement_sourcelist_with_gaiainfo(df)

    df_dr2 = pd.read_csv(outpath_dr2)
    df_edr3 = pd.read_csv(outpath_edr3)

    trgt_id = "2103737241426734336" # Kepler 1627
    trgt_df = df_edr3[df_edr3.source_id.astype(str) == trgt_id]

    return df_dr2, df_edr3, trgt_df


def get_deltalyr_kc19_comovers():
    """
    Get the kinematic neighbors of Kepler 1627.  The contents of this file are
    EDR3 properties.

    made by plot_XYZvtang.py

        sel = (
            (df_edr3.delta_pmdec_prime_km_s > -5)
            &
            (df_edr3.delta_pmdec_prime_km_s < 2)
            &
            (df_edr3.delta_pmra_prime_km_s > -4)
            &
            (df_edr3.delta_pmra_prime_km_s < 2)
        )

    """
    csvpath = os.path.join(RESULTSDIR, 'tables',
                           'stephenson1_edr3_XYZvtang_candcomovers.csv')

    return pd.read_csv(csvpath)


def get_deltalyr_kc19_cleansubset():
    """
    To make this subset, I took the KC19 members (EDR3-crossedmatched).
    Then, I ran them through plot_XYZvtang, which created
        "stephenson1_edr3_XYZvtang_allphysical.csv"
    Then, I opened it up in glue.  I made a selection lasso in kinematic
    velocity space, in XZ, and XY position space.
    """
    csvpath = os.path.join(RESULTSDIR,
                           'glue_stephenson1_edr3_XYZvtang_allphysical',
                           'set0_select_kinematic_YX_ZX.csv')

    return pd.read_csv(csvpath)


def get_gaia_catalog_of_nearby_stars():
    fitspath = os.path.join(
        DATADIR, 'nearby_stars', 'GaiaCollaboration_2021_GCNS.fits'
    )
    hl = fits.open(fitspath)
    d = hl[1].data
    df = Table(d).to_pandas()

    COLDICT = {
        'GaiaEDR3': 'edr3_source_id',
        'RA_ICRS': 'ra',
        'DE_ICRS': 'dec',
        'Plx': 'parallax',
        'pmRA': 'pmra',
        'pmDE': 'pmdec',
        'Gmag': 'phot_g_mean_mag',
        'BPmag': 'phot_bp_mean_mag',
        'RPmag': 'phot_rp_mean_mag'
    }
    df = df.rename(columns=COLDICT)

    return df


def get_clustermembers_cg18_subset(clustername):
    """
    e.g., IC_2602, or "Melotte_22" for Pleaides.
    """

    csvpath = '/Users/luke/local/cdips/catalogs/cdips_targets_v0.6_nomagcut_gaiasources.csv'
    df = pd.read_csv(csvpath, sep=',')

    sel0 = (
        df.cluster.str.contains(clustername)
        &
        df.reference_id.str.contains('CantatGaudin2018a')
    )
    sdf = df[sel0]

    sel = []
    for ix, r in sdf.iterrows():
        c = np.array(r['cluster'].split(','))
        ref = np.array(r['reference_id'].split(','))
        this = np.any(
            np.in1d(
                'CantatGaudin2018a',
                ref[np.argwhere( c == clustername ).flatten()]
            )
        )
        sel.append(this)
    sel = np.array(sel).astype(bool)

    csdf = sdf[sel]

    return csdf


ORIENTATIONTRUTHDICT = {
    'prograde': 0,
    'retrograde': -150,
    'polar': 85
}

def get_simulated_RM_data(orientation, makeplot=1):

    #
    # https://github.com/gummiks/rmfit. Hirano+11,+12 implementation by
    # Gudmundur Stefansson.
    #
    from rmfit import RMHirano

    assert orientation in ['prograde', 'retrograde', 'polar']
    lam = ORIENTATIONTRUTHDICT[orientation]

    t_cadence = 20/(24*60) # 15 minutes, in days

    T0 = 2454953.790531
    P = 7.20281
    aRs = 12.03
    i = 86.138
    vsini = 20
    rprs = 0.0433
    e = 0.
    w = 90.
    # lam = 0
    u = [0.515, 0.23]

    beta = 4
    sigma = vsini / 1.31 # assume sigma is vsini/1.31 (see Hirano et al. 2010)

    times = np.arange(-2.5/24+T0,2.5/24+t_cadence+T0,t_cadence)

    R = RMHirano(lam,vsini,P,T0,aRs,i,rprs,e,w,u,beta,sigma,supersample_factor=7,exp_time=t_cadence,limb_dark='quadratic')
    rm = R.evaluate(times)

    return times, rm


def get_keplerfieldfootprint_dict():

    kep = pd.read_csv(
        os.path.join(DATADIR, 'skychart', 'kepler_field_footprint.csv')
    )

    # we want the corner points, not the mid-points
    is_mipoint = ((kep['row']==535) & (kep['column']==550))
    kep = kep[~is_mipoint]

    kep_coord = SkyCoord(
        np.array(kep['ra'])*u.deg, np.array(kep['dec'])*u.deg, frame='icrs'
    )
    kep_elon = kep_coord.barycentrictrueecliptic.lon.value
    kep_elat = kep_coord.barycentrictrueecliptic.lat.value
    kep['elon'] = kep_elon
    kep['elat'] = kep_elat

    kep_d = {}
    for module in np.unique(kep['module']):
        kep_d[module] = {}
        for output in np.unique(kep['output']):
            kep_d[module][output] = {}
            sel = (kep['module']==module) & (kep['output']==output)

            _ra = list(kep[sel]['ra'])
            _dec = list(kep[sel]['dec'])
            _elon = list(kep[sel]['elon'])
            _elat = list(kep[sel]['elat'])

            _ra = [_ra[0], _ra[1], _ra[3], _ra[2] ]
            _dec =  [_dec[0], _dec[1], _dec[3], _dec[2] ]
            _elon = [_elon[0], _elon[1], _elon[3], _elon[2] ]
            _elat = [_elat[0], _elat[1], _elat[3], _elat[2] ]

            _ra.append(_ra[0])
            _dec.append(_dec[0])
            _elon.append(_elon[0])
            _elat.append(_elat[0])

            kep_d[module][output]['corners_ra'] = _ra
            kep_d[module][output]['corners_dec'] = _dec
            kep_d[module][output]['corners_elon'] = _elon
            kep_d[module][output]['corners_elat'] = _elat

    return kep_d


def supplement_gaia_stars_extinctions_corrected_photometry(df):

    from cdips.utils.gaiaqueries import parallax_to_distance_highsn
    from rudolf.extinction import (
        retrieve_stilism_reddening, append_corrected_gaia_phot_Gagne2020
    )

    df['distance'] = parallax_to_distance_highsn(df['parallax'])
    df = retrieve_stilism_reddening(df, verbose=False)
    df = append_corrected_gaia_phot_Gagne2020(df)

    return df
