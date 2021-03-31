"""
Data getters:
    get_gaia_cluster_data
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


def get_gaia_cluster_data():

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
