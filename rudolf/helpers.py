"""
Data getters:
    δ Lyr Cluster:
        get_deltalyr_kc19_gaia_data
        get_deltalyr_kc19_comovers
        get_deltalyr_kc19_cleansubset
        get_autorotation_dataframe

    Kepler 1627:
        get_manually_downloaded_kepler_lightcurve
        get_kep1627_muscat_lightcuve
        get_keplerfieldfootprint_dict
        get_flare_df
        get_becc_limits

    Get cluster datasets (useful for HR diagrams!):
        get_gaia_catalog_of_nearby_stars
        get_clustermembers_cg18_subset
        get_mutau_members
        get_ScoOB2_members
        get_BPMG_members
        get_gaia_catalog_of_nearby_stars

    Supplement a set of Gaia stars with extinctions and corrected photometry:
        supplement_gaia_stars_extinctions_corrected_photometry

    Supplement columns:
        append_phot_binary_column
        append_phot_membershipexclude_column

    Clean Gaia sources based on photometry:
        get_clean_gaia_photometric_sources

    All-sky photometric surveys (GALEX/2MASS)
        get_galex_data
        get_2mass_data

Proposal/RM-related:
    get_simulated_RM_data

One-offs to get the Stephenson-1 and RSG-5 information:
    get_candidate_stephenson1_member_list
    get_candidate_rsg5_member_list
    supplement_sourcelist_with_gaiainfo
"""

#############
## LOGGING ##
#############

import logging
from astrobase import log_sub, log_fmt, log_date_fmt

DEBUG = False
if DEBUG:
    level = logging.DEBUG
else:
    level = logging.INFO
LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=level,
    style=log_sub,
    format=log_fmt,
    datefmt=log_date_fmt,
)

LOGDEBUG = LOGGER.debug
LOGINFO = LOGGER.info
LOGWARNING = LOGGER.warning
LOGERROR = LOGGER.error
LOGEXCEPTION = LOGGER.exception

#############
## IMPORTS ##
#############

import os, collections, pickle
import numpy as np, pandas as pd
from glob import glob
from copy import deepcopy
from datetime import datetime
from collections import OrderedDict

from numpy import array as nparr

from astropy.io import fits
from astropy import units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord

from rudolf.paths import DATADIR, RESULTSDIR, PHOTDIR, LOCALDIR

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


def get_manually_downloaded_kepler_lightcurve(lctype='longcadence', norm_zero=0):
    """
    Collect and stitch available Kepler quarters, after median-normalizing in
    each quater.
    """

    assert lctype in [
        'longcadence', 'shortcadence', 'longcadence_byquarter', 'KOI_7368',
        'KOI_7368_byquarter', 'KOI_7913', 'Kepler_1643', 'KOI_7913_byquarter',
        'Kepler_1643_byquarter'
    ]

    if lctype in ['longcadence', 'longcadence_byquarter']:
        # Kepler-1627
        lcfiles = glob(os.path.join(DATADIR, 'phot', 'kplr*_llc.fits'))
    elif lctype == 'shortcadence':
        lcfiles = glob(os.path.join(DATADIR, 'phot', 'full_MAST_sc', 'MAST_*',
                                    'Kepler', 'kplr006184894*', 'kplr*_slc.fits'))
    elif lctype in [
        'KOI_7368', 'KOI_7913', 'Kepler_1643',
        'KOI_7368_byquarter', 'KOI_7913_byquarter', 'Kepler_1643_byquarter'
    ]:
        starid = lctype
        if starid.endswith('byquarter'):
            starid = "_".join(starid.split("_")[:-1])
        lcfiles = glob(os.path.join(DATADIR, starid, 'phot', 'MAST*', 'Kepler',
                                    'kplr*_lc_*', 'kplr*_llc.fits'))
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
        if norm_zero:
            flux = (_f/np.nanmedian(_f) - 1) # normalize around zero for GP regression
        else:
            flux = (_f/np.nanmedian(_f))
        flux_err = _f_err/np.nanmedian(_f)
        qual = d['SAP_QUALITY']

        sel = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err)

        texp = np.nanmedian(np.diff(time[sel]))

        timelist.append(time[sel])
        f_list.append(flux[sel])
        ferr_list.append(flux_err[sel])
        qual_list.append(qual[sel])
        texp_list.append(texp)

    if (
        lctype in [
            'longcadence', 'shortcadence', 'KOI_7368', 'Kepler_1643', 'KOI_7913'
        ]
    ):
        time = np.hstack(timelist)
        flux = np.hstack(f_list)
        flux_err = np.hstack(ferr_list)
        qual = np.hstack(qual_list)
        texp = np.nanmedian(texp_list)
    elif lctype in [
        'longcadence_byquarter', 'KOI_7368_byquarter', 'Kepler_1643_byquarter',
        'KOI_7913_byquarter'
    ]:
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


def get_kep1627_muscat_lightcuve():
    """
    Collect MuSCAT data. Return an ordered dict with keys 'muscat_b' and values
    time, flux, flux_err.
    """

    datasets = OrderedDict()

    bandpasses = 'g,r,i,z'.split(',')

    # nb. the files from the team contain airmass, dx, dy, FWHM, and peak ADU
    # too. not needed in our approach
    for bp in bandpasses:

        lcpath = glob(
            os.path.join(PHOTDIR, 'MUSCAT3', f'*muscat3_{bp}_*csv')
        )[0]
        df = pd.read_csv(lcpath)

        # converting b/c theano only understands float64
        _time, _flux, _fluxerr = (
            nparr(df.BJD_TDB).astype(np.float64),
            nparr(df.Flux).astype(np.float64),
            nparr(df.Err).astype(np.float64)
        )

        _texp = np.nanmedian(np.diff(_time))

        key = f'muscat3_{bp}'

        datasets[key] = [_time, _flux, _fluxerr, _texp]

    return datasets


def get_candidate_stephenson1_member_list():

    outpath = os.path.join(DATADIR, 'gaia', 'stephenson1_kc19.csv')

    if not os.path.exists(outpath):
        # Kounkel & Covey 2019 Stephenson1 candidate member list.
        csvpath = os.path.join(DATADIR, 'gaia', 'string_table1.csv')
        df = pd.read_csv(csvpath)

        sdf = df[np.array(df.group_id).astype(int) == 73]

        sdf['source_id'].to_csv(outpath, index=False)

    return pd.read_csv(outpath)


def get_candidate_rsg5_member_list():

    outpath = os.path.join(DATADIR, 'gaia', 'rsg5_kc19.csv')

    if not os.path.exists(outpath):
        # Kounkel & Covey 2019 Stephenson1 candidate member list.
        csvpath = os.path.join(DATADIR, 'gaia', 'string_table1.csv')
        df = pd.read_csv(csvpath)

        sdf = df[np.array(df.group_id).astype(int) == 96]

        sdf['source_id'].to_csv(outpath, index=False)

    return pd.read_csv(outpath)


def supplement_sourcelist_with_gaiainfo(df, groupname='stephenson1'):

    from cdips.utils.gaiaqueries import given_dr2_sourceids_get_edr3_xmatch

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
    from cdips.utils.gaiaqueries import given_source_ids_get_gaia_data

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

    outpath_dr2 = os.path.join(DATADIR, 'gaia', f'{groupname}_kc19_dr2.csv')
    outpath_edr3 = os.path.join(DATADIR, 'gaia', f'{groupname}_kc19_edr3.csv')
    outpath_dr2xedr3 = os.path.join(DATADIR, 'gaia', f'{groupname}_kc19_dr2xedr3.csv')

    df_dr2.to_csv(outpath_dr2, index=False)
    df_edr3.to_csv(outpath_edr3, index=False)
    dr2_x_edr3_df.to_csv(outpath_dr2xedr3, index=False)


def get_deltalyr_kc19_gaia_data(return_all_targets=0):
    """
    Get all Kounkel & Covey 2019 "Stephenson 1" members.
    """

    from cdips.utils.gaiaqueries import given_source_ids_get_gaia_data

    outpath_dr2 = os.path.join(DATADIR, 'gaia', 'stephenson1_kc19_dr2.csv')
    outpath_edr3 = os.path.join(DATADIR, 'gaia', 'stephenson1_kc19_edr3.csv')

    if not os.path.exists(outpath_dr2):
        df = get_candidate_stephenson1_member_list()
        supplement_sourcelist_with_gaiainfo(df, groupname='stephenson1')

    df_dr2 = pd.read_csv(outpath_dr2)
    df_edr3 = pd.read_csv(outpath_edr3)

    trgt_id = "2103737241426734336" # Kepler 1627
    trgt_df = df_edr3[df_edr3.source_id.astype(str) == trgt_id]

    if not return_all_targets:
        return df_dr2, df_edr3, trgt_df

    trgt_id_dict = {
        'KOI-7368': "2128840912955018368",
        'KOI-7913': "2106235301785454208",
        'Kepler-1643': "2082142734285082368" # aka. KOI-6186
    }

    koi_df_dict = {}
    for k,trgt_id in trgt_id_dict.items():

        _df = df_edr3[df_edr3.source_id.astype(str) == trgt_id]

        if len(_df) == 0:
            print(f'Running single-object EDR3 search for {trgt_id}...')
            _df = given_source_ids_get_gaia_data(
                np.array([trgt_id]).astype(np.int64), str(trgt_id), n_max=2,
                overwrite=False, enforce_all_sourceids_viable=True, savstr='',
                gaia_datarelease='gaiaedr3')

        assert len(_df) > 0

        koi_df_dict[k] = _df

    return df_dr2, df_edr3, trgt_df, koi_df_dict


def get_rsg5_kc19_gaia_data():
    """
    Get all Kounkel & Covey 2019 "RSG_5" (Theia 96) members.
    """

    outpath_dr2 = os.path.join(DATADIR, 'gaia', 'rsg5_kc19_dr2.csv')
    outpath_edr3 = os.path.join(DATADIR, 'gaia', 'rsg5_kc19_edr3.csv')

    if not os.path.exists(outpath_dr2):
        df = get_candidate_rsg5_member_list()
        supplement_sourcelist_with_gaiainfo(df, groupname='rsg5')

    df_dr2 = pd.read_csv(outpath_dr2)
    df_edr3 = pd.read_csv(outpath_edr3)

    return df_dr2, df_edr3



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
    raise NotImplementedError('this is deprecated. use get_deltalyr_kc19_cleansubset.')
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


def get_set1_koi7368():
    """
    As for get_deltalyr_kc19_cleansubset, but for the KOI 7368 neighbors.
    """
    csvpath = os.path.join(RESULTSDIR,
                           'glue_stephenson1_edr3_XYZvtang_allphysical',
                           'set1_select_kinematic_YX_ZX.csv')

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

    if 'l' not in csdf:
        _c = SkyCoord(ra=nparr(csdf.ra)*u.deg, dec=nparr(csdf.dec)*u.deg)
        csdf['l'] = _c.galactic.l.value
        csdf['b'] = _c.galactic.b.value

    return csdf


def get_BPMG_members():
    """
    BPMG = beta pic moving group

    This retrieves BPMG members from Ujjwal+2020.  I considered also requiring
    Gagne+18 matches, but this yielded only 25 stars.
    """

    csvpath = '/Users/luke/local/cdips/catalogs/cdips_targets_v0.6_nomagcut_gaiasources.csv'
    df = pd.read_csv(csvpath, sep=',')

    sel0 = (
        df.cluster.str.contains('BPMG')
        &
        df.reference_id.str.contains('Ujjwal2020')
    )
    sdf = df[sel0]

    if 'l' not in sdf:
        _c = SkyCoord(ra=nparr(sdf.ra)*u.deg, dec=nparr(sdf.dec)*u.deg)
        sdf['l'] = _c.galactic.l.value
        sdf['b'] = _c.galactic.b.value

    return sdf


def get_mutau_members():

    tablepath = os.path.join(DATADIR, 'cluster',
                             'Gagne_2020_mutau_table12_apjabb77et12_mrt.txt')
    df = Table.read(tablepath, format='ascii.cds').to_pandas()

    sel = (
        (df['Memb-Type'] == 'IM')
        |
        (df['Memb-Type'] == 'CM')
        #|
        #(df['Memb-Type'] == 'LM')
    )

    sdf = df[sel]

    #Gagne's tables are pretty annoying
    RAh, RAm, RAs = nparr(sdf['Hour']), nparr(sdf['Minute']), nparr(sdf['Second'])

    RA_hms =  [str(rah).zfill(2)+'h'+
               str(ram).zfill(2)+'m'+
               str(ras).zfill(2)+'s'
               for rah,ram,ras in zip(RAh, RAm, RAs)]

    DEd, DEm, DEs = nparr(sdf['Degree']),nparr(sdf['Arcminute']),nparr(sdf['Arcsecond'])
    DEsign = nparr(sdf['Sign'])
    DEsign[DEsign != '-'] = '+'

    DE_dms = [str(desgn)+
              str(ded).zfill(2)+'d'+
              str(dem).zfill(2)+'m'+
              str(des).zfill(2)+'s'
              for desgn,ded,dem,des in zip(DEsign, DEd, DEm, DEs)]

    coords = SkyCoord(ra=RA_hms, dec=DE_dms, frame='icrs')

    RA = coords.ra.value
    dec = coords.dec.value

    sdf['ra'] = RA
    sdf['dec'] = dec

    _c = SkyCoord(ra=nparr(sdf.ra)*u.deg, dec=nparr(sdf.dec)*u.deg)
    sdf['l'] = _c.galactic.l.value
    sdf['b'] = _c.galactic.b.value

    # columns
    COLDICT = {
        'plx': 'parallax',
        'pmRA': 'pmra',
        'pmDE': 'pmdec',
        'Gmag': 'phot_g_mean_mag',
        'BPmag': 'phot_bp_mean_mag',
        'RPmag': 'phot_rp_mean_mag'
    }
    sdf = sdf.rename(columns=COLDICT)

    return sdf


def get_ScoOB2_members():

    from cdips.catalogbuild.vizier_xmatch_utils import get_vizier_table_as_dataframe
    from cdips.utils.gaiaqueries import given_source_ids_get_gaia_data

    vizier_search_str = "Damiani J/A+A/623/A112"
    whichcataloglist = "J/A+A/623/A112"

    srccolumns = 'DR2Name|RAJ2000|DEJ2000|GLON|GLAT|Plx|pmGLON|pmGLAT|Sel|Pop'
    dstcolumns = 'source_id|ra|dec|l|b|parallax|pm_l|pl_b|Sel|Pop'

    # ScoOB2_PMS
    df0 = get_vizier_table_as_dataframe(
        vizier_search_str, srccolumns, dstcolumns,
        whichcataloglist=whichcataloglist, table_num=0
    )
    # ScoOB2_UMS
    df1 = get_vizier_table_as_dataframe(
        vizier_search_str, srccolumns, dstcolumns,
        whichcataloglist=whichcataloglist, table_num=1
    )

    gdf0 = given_source_ids_get_gaia_data(
        np.array(df0.source_id).astype(np.int64), 'ScoOB2_PMS_Damiani19',
        n_max=12000, overwrite=False
    )
    gdf1 = given_source_ids_get_gaia_data(
        np.array(df1.source_id).astype(np.int64), 'ScoOB2_UMS_Damiani19',
        n_max=12000, overwrite=False
    )
    selcols = ['source_id', 'Sel', 'Pop']
    mdf0 = gdf0.merge(df0[selcols], on='source_id', how='inner')
    mdf1 = gdf1.merge(df1[selcols], on='source_id', how='inner')

    assert len(mdf0) == 10839
    assert len(mdf1) == 3598

    df = pd.concat((mdf0, mdf1)).reset_index()

    # proper-motion and v_tang selected...
    # require population to be UCL-1, since:
    # Counter({'': 1401, 'D2b': 3510, 'IC2602': 260, 'D1': 2058, 'LCC-1': 84,
    #         'UCL-2': 51, 'D2a': 750, 'UCL-3': 47, 'Lup III': 69, 'UCL-1':
    #         551, 'USC-D2': 1210, 'USC-f': 347, 'USC-n': 501})
    sel = (df.Sel == 'pv')
    sel &= (df.Pop == 'UCL-1')

    sdf = df[sel]

    return sdf


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


def supplement_gaia_stars_extinctions_corrected_photometry(
    df, extinctionmethod='gaia2018', savpath=None
):
    """
    Using distances calculated as 1/parallax, retrieve the STILISM reddening
    coefficients for a set of Gaia stars, and append corrected photometry
    columns to the working DataFrame using those coefficients.
    """

    from cdips.utils.gaiaqueries import parallax_to_distance_highsn
    from rudolf.extinction import (
        retrieve_stilism_reddening,
        append_corrected_gaia_phot_Gagne2020,
        append_corrected_gaia_phot_Gaia2018
    )

    # cache the slow step
    if not os.path.exists(savpath):
        df['distance'] = parallax_to_distance_highsn(df['parallax'])
        df = retrieve_stilism_reddening(df, verbose=False, outpath=savpath)
    else:
        df = pd.read_csv(savpath)

    if extinctionmethod == 'gaia2018':
        df = append_corrected_gaia_phot_Gaia2018(df)
    if extinctionmethod == 'gagne2020':
        df = append_corrected_gaia_phot_Gagne2020(df)

    return df


def get_clean_gaia_photometric_sources(df):
    """
    Given a dataframe of Gaia DR2 columns, calculate the "cleaning selection"
    described on page 3 / Appendix B of GaiaCollab+2018 HR diagram paper.  The
    point of this selection is to remove sources that are expected to have
    photometric or astrometric problems.  Binaries are left in.

    Returns: boolean np.ndarray of sources that meet the selection cut.
    """
    chisq = nparr(df.astrometric_chi2_al)
    nuprime = nparr(df.astrometric_n_good_obs_al)
    Gmag = nparr(df.phot_g_mean_mag)

    sel0 = np.sqrt(chisq/(nuprime-5)) < 1.2*np.maximum(
        1,
        np.exp(-0.2*(Gmag-19.5))
    )

    sel1 = (df.parallax_over_error > 5)

    sel2 = (
        (df.phot_g_mean_flux_over_error > 50)
        &
        (df.phot_rp_mean_flux_over_error > 20)
        &
        (df.phot_bp_mean_flux_over_error > 20)
    )

    sel3 = (
        (
            df.phot_bp_rp_excess_factor < 1.3 +
            0.06*(df.phot_bp_mean_mag - df.phot_rp_mean_mag)**2
        )
        &
        (
            df.phot_bp_rp_excess_factor > 1.0 +
            0.015*(df.phot_bp_mean_mag - df.phot_rp_mean_mag)**2
        )
    )

    sel4 = df.visibility_periods_used > 8

    sel = sel0 & sel1 & sel2 & sel3 & sel4

    return sel


def get_autorotation_dataframe(runid='deltaLyrCluster', verbose=1,
                               cleaning='defaultcleaning'):
    """
    runid = 'deltaLyrCluster', for example

    Cleaning options:
        'defaultcleaning' P<15d, LSP>0.1, Nequal==0, Nclose<=1.
        'harderlsp' P<15d, LSP>0.15, Nequal==0, Nclose<=1.
        'nocleaning': P<99d.
        'defaultcleaning_cutProtColor': add Prot-color plane cut to
            defaultcleaning.
    """

    assert isinstance(cleaning, str)

    rotdir = os.path.join(DATADIR, 'rotation')

    rotpath = os.path.join(rotdir, f'{runid}_rotation_periods.csv')
    # make the deltaLyrCluster auto-rotation dict... every version this needs
    # to be deleted and remade
    if runid=='deltaLyrCluster' and not os.path.exists(rotpath):

        # NOTE: this is the file provided by Jason Curtis, with the
        # rotation periods he measured for KC19 cluster members.
        df = pd.read_csv(
            os.path.join(rotdir, 'Theia73-Prot_Results-Prelim-v2.csv')
        )

        #
        # get all the gaia information for the cluster...
        #
        df_dr2, df_edr3, trgt_df = get_deltalyr_kc19_gaia_data()

        # get mapping between DR2 and EDR3 source_ids
        path_dr2xedr3 = os.path.join(DATADIR, 'gaia', 'stephenson1_kc19_dr2xedr3.csv')
        csvpath_dr2xedr3 = os.path.join(DATADIR, 'gaia', 'stephenson1_kc19_dr2xedr3.csv')
        dr2_x_edr3_df = pd.read_csv(csvpath_dr2xedr3)
        get_edr3_xm = lambda _df: (
            _df.sort_values(by='angular_distance').
            drop_duplicates(subset='dr2_source_id', keep='first')
        )
        s_edr3 = get_edr3_xm(dr2_x_edr3_df)

        # merge everything into one table: first get a dataframe with
        # both DR2 and EDR3 source_ids. then, merge this against
        # rotation periods.
        mdf0 = s_edr3.merge(df_edr3, how='left',
                            left_on='dr3_source_id',
                            right_on='source_id')
        assert len(mdf0) == len(df_dr2)

        mdf = mdf0.merge(df, left_on='dr2_source_id',
                         right_on='Gaia_DR2_Source', how='left')

        #
        # get T mag using Stassun+2019 Eq1
        #
        Tmag_pred = (
            mdf['phot_g_mean_mag']
            - 0.00522555 * (mdf['phot_bp_mean_mag'] - mdf['phot_rp_mean_mag'])**3
            + 0.0891337 * (mdf['phot_bp_mean_mag'] - mdf['phot_rp_mean_mag'])**2
            - 0.633923 * (mdf['phot_bp_mean_mag'] - mdf['phot_rp_mean_mag'])
            + 0.0324473
        )

        mdf['Tmag_pred'] = Tmag_pred

        # good. now, all this dataframe needs to be useable is the
        # CROWDING metrics.

        # Count how many stars inside the aperture are brighter.
        from astroquery.mast import Catalogs
        nequal,nclose,nfaint = [],[],[]
        ix = 0
        for ra,dec,TESSMAG in zip(
            nparr(mdf.ra), nparr(mdf.dec), nparr(mdf.Tmag_pred)
        ):
            LOGINFO(f'{datetime.utcnow().isoformat()}: {ix}/{len(mdf)}...')
            APSIZE = 1 # assumes radius 1 pixel, fine for crowding
            radius = APSIZE*21.0*u.arcsec
            nbhr_stars = Catalogs.query_region(
                "{} {}".format(float(ra), float(dec)),
                catalog="TIC",
                radius=radius
            )
            nequal.append(
                len(nbhr_stars[nbhr_stars['Tmag'] < TESSMAG])
            )
            nclose.append(
                len(nbhr_stars[nbhr_stars['Tmag'] < (TESSMAG+1.25)])
            )
            nfaint.append(
                len(nbhr_stars[nbhr_stars['Tmag'] < (TESSMAG+2.5)])
            )
            ix += 1

        mdf['nequal'] = nparr(nequal)
        mdf['nclose'] = nparr(nclose)
        mdf['nfaint'] = nparr(nfaint)

        mdf.to_csv(rotpath, index=False)

    df = pd.read_csv(rotpath)
    if runid == 'deltaLyrCluster':
        df = append_phot_binary_column(df)
        df = append_phot_membershipexclude_column(df)
        # rename from Jason's column names
        COLDICT = {
            'Prot_Adopted': 'period',
            'Power_LS_Auto': 'lspval'
        }
        df = df.rename(columns=COLDICT)

    if cleaning in ['defaultcleaning', 'periodogram_match', 'curtiscleaning',
                    'match234_alias','harderlsp', 'defaultcleaning_cutProtColor']:
        # automatic selection criteria for viable rotation periods
        NEQUAL_CUTOFF = 0 # could also do 1
        NCLOSE_CUTOFF = 1
        LSP_CUTOFF = 0.10 # standard
        if cleaning == 'harderlsp':
            LSP_CUTOFF = 0.20

        sel = (df.period < 100)

        if cleaning in ['defaultcleaning', 'harderlsp']:
            sel &= (
                (df.nequal <= NEQUAL_CUTOFF)
                &
                (df.nclose <= NCLOSE_CUTOFF)
                &
                (df.period < 15)
                &
                (df.lspval > LSP_CUTOFF)
            )

        elif cleaning in ['curtiscleaning']:
            # Jason already applied LS Prot < 11 days, and LS Power > 0.2 for
            # TESS... with visual validation values replacing LS auto values
            # ... and Kepler periods replacing TESS periods.
            # So just apply the binarity check.
            sel &= (
                (df.nequal <= NEQUAL_CUTOFF)
                &
                (df.nclose <= NCLOSE_CUTOFF)
                &
                (df.period < 15)
                &
                (df.lspval > 0.20)
            )

        elif cleaning == 'defaultcleaning_cutProtColor':
            assert 0
            #FIXME
            from earhart.priors import AVG_EBpmRp
            BpmRp0 = (
                df['phot_bp_mean_mag'] - df['phot_rp_mean_mag'] - AVG_EBpmRp
            )
            Prot_boundary = PleaidesQuadProtModel(BpmRp0)
            sel &= (
                df.period < Prot_boundary
            )

    elif cleaning in ['nocleaning']:
        NEQUAL_CUTOFF = 99999
        NCLOSE_CUTOFF = 99999
        LSP_CUTOFF = 0
        sel = (
            (df.period < 99)
        )
    else:
        raise ValueError(f'Got cleaning == {cleaning}, not recognized.')

    if cleaning in ['defaultcleaning', 'nocleaning', 'harderlsp',
                    'defaultcleaning_cutProtColor', 'curtiscleaning']:
        pass

    elif cleaning == 'periodogram_match':
        raise NotImplementedError
        sel_periodogram_match = (
            (0.9 < (df.spdmperiod/df.period))
            &
            (1.1 > (df.spdmperiod/df.period))
        )
        sel &= sel_periodogram_match

    elif cleaning == 'match234_alias':
        raise NotImplementedError
        sel_match = (
            (0.9 < (df.spdmperiod/df.period))
            &
            (1.1 > (df.spdmperiod/df.period))
        )
        sel_spdm2x = (
            (1.9 < (df.spdmperiod/df.period))
            &
            (2.1 > (df.spdmperiod/df.period))
        )
        sel_spdm3x = (
            (2.9 < (df.spdmperiod/df.period))
            &
            (3.1 > (df.spdmperiod/df.period))
        )
        sel_spdm4x = (
            (3.9 < (df.spdmperiod/df.period))
            &
            (4.1 > (df.spdmperiod/df.period))
        )
        sel &= (
            sel_match
            |
            sel_spdm2x
            |
            sel_spdm3x
            |
            sel_spdm4x
        )

    else:
        raise ValueError(f'Got cleaning == {cleaning}, not recognized.')

    ref_sel = (
        (df.nequal <= NEQUAL_CUTOFF)
        &
        (df.nclose <= NCLOSE_CUTOFF)
    )

    if verbose:
        LOGINFO(f'Getting autorotation dataframe for {runid}...')
        LOGINFO(f'Starting with {len(df[ref_sel])} entries that meet NEQUAL and NCLOSE criteria...')
        LOGINFO(f'Got {len(df[sel])} entries with P<15d, LSP>{LSP_CUTOFF}, nequal<={NEQUAL_CUTOFF}, nclose<={NCLOSE_CUTOFF}')
        if cleaning == 'periodogram_match':
            LOGINFO(f'...AND required LS and SPDM periods to agree.')
        elif cleaning == 'match234_alias':
            LOGINFO(f'...AND required LS and SPDM periods to agree (up to 1x,2x,3x,4x harmonic).')
        LOGINFO(10*'.')

    return df[sel], df


def append_phot_binary_column(df, DIFFERENCE_CUTOFF=0.3):

    from scipy.interpolate import interp1d

    csvpath = os.path.join(DATADIR, 'gaia',
                           'deltaLyrCluster_AbsG_BpmRp_empirical_locus_webplotdigitized.csv')
    ldf = pd.read_csv(csvpath)

    fn_BpmRp_to_AbsG = interp1d(ldf.BpmRp, ldf.AbsG, kind='quadratic',
                                bounds_error=False, fill_value=np.nan)

    get_yval = (
        lambda _df: np.array(
            _df['phot_g_mean_mag'] + 5*np.log10(_df['parallax']/1e3) + 5
        )
    )
    get_xval = (
        lambda _df: np.array(
            _df['phot_bp_mean_mag'] - _df['phot_rp_mean_mag']
        )
    )

    sel_photbin = (
        get_yval(df) <
        ( fn_BpmRp_to_AbsG(get_xval(df)) - DIFFERENCE_CUTOFF)
    )

    df['is_phot_binary'] = sel_photbin

    maketemp = 1
    if maketemp:
        import matplotlib.pyplot as plt
        plt.close('all')
        fig,ax=plt.subplots(figsize=(4,3))
        sel = df.is_phot_binary
        ax.scatter(
            get_xval(df), get_yval(df), c='k', zorder=1,s=2
        )
        ax.scatter(
            get_xval(df[sel]), get_yval(df[sel]), c='r', zorder=2,s=2
        )
        ax.set_ylim(ax.get_ylim()[::-1])
        fig.savefig('temp.png')
        plt.close('all')

    return df


def append_phot_membershipexclude_column(df):

    from scipy.interpolate import interp1d

    csvpath = os.path.join(DATADIR, 'gaia',
                           'deltaLyrCluster_AbsG_BpmRp_empirical_lowerbound_webplotdigitized.csv')
    ldf = pd.read_csv(csvpath)

    fn_BpmRp_to_AbsG = interp1d(ldf.BpmRp, ldf.AbsG, kind='quadratic',
                                bounds_error=False, fill_value=np.nan)

    get_yval = (
        lambda _df: np.array(
            _df['phot_g_mean_mag'] + 5*np.log10(_df['parallax']/1e3) + 5
        )
    )
    get_xval = (
        lambda _df: np.array(
            _df['phot_bp_mean_mag'] - _df['phot_rp_mean_mag']
        )
    )

    #signflip because magnitudes
    sel_phot_nonmember = (
        get_yval(df) > fn_BpmRp_to_AbsG(get_xval(df))
    )

    df['is_phot_nonmember'] = sel_phot_nonmember

    return df


def get_becc_limits(datasets, soln):

    ecc = soln["ecc"]
    b = soln["b"]

    fn = lambda x: np.nanpercentile(x, 95)
    LOGINFO(f'e < {fn(ecc):.3f} at 2-sigma (95th)')
    LOGINFO(f'b < {fn(b):.3f} at 2-sigma (95th)')

    fn = lambda x: np.nanpercentile(x, 99.7)
    LOGINFO(f'e < {fn(ecc):.3f} at 3-sigma (99.7th)')
    LOGINFO(f'b < {fn(b):.3f} at 3-sigma (99.7th)')


def get_galex_data(ra, dec, starids, idstring, verbose=False):
    """
    Args:
        ra, dec (np.ndarray): arrays of GALEX positional crossmatches to run.
        Assumed to be in degrees, `ra` from 0 to 360.

        starids (np.ndarray): array of strings for star identifiers.

        idstring (str): used to cache the query.

    Returns:
        DataFrame containing starids, nuv_mag, err, fuv_mag, err.
    """

    import warnings
    from astroquery.exceptions import NoResultsWarning
    from astroquery.mast import Catalogs

    cachedir = os.path.join(LOCALDIR, 'galex')
    if not os.path.exists(cachedir):
        os.mkdir(cachedir)

    assert isinstance(ra, np.ndarray)
    assert isinstance(dec, np.ndarray)
    assert isinstance(starids, np.ndarray)
    assert len(ra) == len(dec) == len(starids)
    assert np.max(ra) < 360
    assert np.min(ra) > 0
    assert isinstance(idstring, str)

    outpath = os.path.join(cachedir, idstring+'_galexcache.csv')
    if os.path.exists(outpath):
        LOGINFO(f'Found {outpath}, loading it from cache.')
        df = pd.read_csv(outpath)
        return df

    N = len(ra)

    if verbose:
        LOGINFO(f'Searching GALEX for {N} sources...')

    # Suppress the noresultswarning from the catalogs package
    warnings.filterwarnings("ignore", category=NoResultsWarning)

    # MAST returns: ['distance_arcmin', 'objID', 'survey', 'ra', 'dec', 'band',
    # 'fov_radius', 'IAUName', 'nuv_exptime', 'fuv_exptime', 'fuv_mag',
    # 'fuv_magerr', 'nuv_mag', 'nuv_magerr', 'fuv_flux', 'fuv_fluxerr',
    # 'nuv_flux', 'nuv_fluxerr', 'e_bv', 'ra_cent', 'dec_cent', 'xCenter',
    # 'yCenter', 'SSAPFitsFile', 'nuv_flux_auto', 'nuv_flux_aper_7',
    # 'nuv_artifact', 'fuv_flux_auto', 'fuv_flux_aper_7', 'fuv_artifact']

    outdf = pd.DataFrame({
        'starid': starids,
        'galex_distance_arcmin': np.nan*np.ones(N),
        'galex_survey': np.nan*np.ones(N),
        'nuv_mag': np.nan*np.ones(N),
        'nuv_magerr': np.nan*np.ones(N),
        'fuv_mag': np.nan*np.ones(N),
        'fuv_magerr': np.nan*np.ones(N)
    })

    coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')

    kvdict = {
        'galex_distance_arcmin': 'distance_arcmin',
        'galex_survey': 'survey',
        'nuv_mag': 'nuv_mag',
        'nuv_magerr': 'nuv_magerr',
        'fuv_mag': 'fuv_mag',
        'fuv_magerr': 'fuv_magerr'
    }

    for ix, starid, ra, dec in zip(
        range(N), starids, coord.ra.value, coord.dec.value
    ):

        if verbose:
            LOGINFO(f'GALEX query {ix}/{N}...')

        querystr = str(ra) + " " + str(dec)
        radius = (5*u.arcsec).to(u.deg).value
        try:
            t_galex = Catalogs.query_object(
                querystr, catalog="Galex", radius=radius, TIMEOUT=60
            )

            # use the most precise NUV magnitude result. at times, GII rather than
            # AIS or MIS surveys.
            if len(t_galex[t_galex['nuv_magerr'] > 0]):
                galex_df = t_galex.to_pandas()
                sgalex_df = galex_df[galex_df.nuv_magerr > 0]
                sel_row = sgalex_df.iloc[np.argmin(sgalex_df.nuv_magerr)]
                # assign to the output array.
                for k,v in kvdict.items():
                    outdf.loc[outdf.starid == starid, k] = sel_row[v]
        except Exception as e:
            LOGERROR(f'ERROR! FAILED GALEX query {ix}/{N}: {e}')
            pass

    outdf.to_csv(outpath, index=False)
    LOGINFO(f'Wrote {outpath}')

    return outdf


def get_2mass_data(ra, dec, starids, idstring, verbose=False):
    """
    Args:
        ra, dec (np.ndarray): arrays of 2MASS positional crossmatches to run.
        Assumed to be in degrees, `ra` from 0 to 360.

        starids (np.ndarray): array of strings for star identifiers.

        idstring (str): used to cache the query.

    Returns:
        DataFrame containing starids, JHK mags and errs.
    """

    import warnings
    from astroquery.exceptions import NoResultsWarning
    from astroquery.irsa import Irsa

    cachedir = os.path.join(LOCALDIR, 'twomass')
    if not os.path.exists(cachedir):
        os.mkdir(cachedir)

    assert isinstance(ra, np.ndarray)
    assert isinstance(dec, np.ndarray)
    assert isinstance(starids, np.ndarray)
    assert len(ra) == len(dec) == len(starids)
    assert np.max(ra) < 360
    assert np.min(ra) > 0
    assert isinstance(idstring, str)

    outpath = os.path.join(cachedir, idstring+'_2masscache.csv')
    if os.path.exists(outpath):
        LOGINFO(f'Found {outpath}, loading it from cache.')
        df = pd.read_csv(outpath)
        return df

    N = len(ra)

    if verbose:
        LOGINFO(f'Searching 2MASS for {N} sources...')

    # Suppress the noresultswarning from astroquery.
    warnings.filterwarnings("ignore", category=NoResultsWarning)

    #FIXME FIXME
    # https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-dd?catalog=fp_psc
    # MAST returns: ['ra', 'dec', 'clon', 'clat', 'err_maj', 'err_min',
    # 'err_ang', 'designation', 'j_m', 'j_cmsig', 'j_msigcom', 'j_snr', 'h_m',
    # 'h_cmsig', 'h_msigcom', 'h_snr', 'k_m', 'k_cmsig', 'k_msigcom',
    # 'k_snr', 'ph_qual', 'rd_flg', 'bl_flg', 'cc_flg', 'ndet',
    # 'gal_contam', 'mp_flg', 'hemis', 'xdate', 'scan', 'glon', 'glat',
    # 'a', 'dist_opt', 'phi_opt', 'b_m_opt', 'vr_m_opt', 'nopt_mchs',
    # 'ext_key', 'dist', 'angle', 'j_h', 'h_k', 'j_k']

    outdf = pd.DataFrame({
        'starid': starids,
        'twomass_distance': np.nan*np.ones(N),
        'j_mag': np.nan*np.ones(N),
        'j_magerr': np.nan*np.ones(N),
        'h_mag': np.nan*np.ones(N),
        'h_magerr': np.nan*np.ones(N),
        'k_mag': np.nan*np.ones(N),
        'k_magerr': np.nan*np.ones(N),
        'ph_qual': np.nan*np.ones(N),
    })

    coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')

    kvdict = {
        'twomass_distance': 'dist',
        'j_mag': 'j_m',
        'j_magerr': 'j_cmsig',
        'h_mag': 'h_m',
        'h_magerr': 'h_cmsig',
        'k_mag': 'k_m',
        'k_magerr': 'k_cmsig',
        'ph_qual': 'ph_qual',
    }

    for ix, starid, ra, dec in zip(
        range(N), starids, coord.ra.value, coord.dec.value
    ):

        if verbose:
            LOGINFO(f'2MASS query {ix}/{N}...')

        try:
            querystr = str(ra) + " " + str(dec)
            t_tmass = Irsa.query_region(
                querystr, catalog='fp_psc', radius='0d0m5s'
            )

            # use the brightest 2mass source within 5 arcseconds.  (binarity checks
            # will be independent!)
            if len(t_tmass[t_tmass['j_m']>0]) > 0:
                tmass_df = t_tmass.to_pandas()
                stmass_df = tmass_df[tmass_df.j_m > 0]
                sel_row = stmass_df.iloc[np.argmin(stmass_df.j_m)]
                # assign to the output array.
                for k,v in kvdict.items():
                    outdf.loc[outdf.starid == starid, k] = sel_row[v]
        except Exception as e:
            LOGERROR(f'ERROR! FAILED 2MASS query {ix}/{N}: {e}')
            pass

    outdf.to_csv(outpath, index=False)
    LOGINFO(f'Wrote {outpath}')

    return outdf
