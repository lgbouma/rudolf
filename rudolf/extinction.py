"""
Tools for working out the extinction.

General-purpose:

    given_EBmV_and_BpmRp_get_A_X

    retrieve_stilism_reddening

    append_corrected_gaia_phot_Gaia2018:
            given STILISM reddening, append *_corr photometry columns.

    append_corrected_gaia_phot_Gagne2020;
            as above, but with the Gagne+20 corrections, instead of Gaia+18
"""

import numpy as np, pandas as pd
import requests
from io import StringIO
import sys, os
from datetime import datetime

from scipy.interpolate import interp1d

from rudolf.paths import DATADIR

def given_EBmV_and_BpmRp_get_A_X(EBmV, BpmRp, bandpass='G'):
    """
    Assuming GaiaCollaboration_2018_table1 coefficients, convert an E(B-V)
    value to a A_G, A_BP, or A_RP value.

    bandpass: 'G','BP', or 'RP'
    """
    assert bandpass in ['G','BP','RP']

    corr_path = os.path.join(DATADIR, 'extinction',
                             'GaiaCollaboration_2018_table1.csv')
    cdf = pd.read_csv(corr_path, sep=',')

    A_0 = 3.1 * E_BmV

    c1 = float(cdf.loc[cdf.bandpass==f'k{bp}', 'c1'])
    c2 = float(cdf.loc[cdf.bandpass==f'k{bp}', 'c2'])
    c3 = float(cdf.loc[cdf.bandpass==f'k{bp}', 'c3'])
    c4 = float(cdf.loc[cdf.bandpass==f'k{bp}', 'c4'])
    c5 = float(cdf.loc[cdf.bandpass==f'k{bp}', 'c5'])
    c6 = float(cdf.loc[cdf.bandpass==f'k{bp}', 'c6'])
    c7 = float(cdf.loc[cdf.bandpass==f'k{bp}', 'c7'])

    # nb Eq 1 of the paper has BpmRp0 ... which presents a bit of
    # a self-consistency issue
    k_X = (
        c1
        + c2*(BpmRp)
        + c3*(BpmRp)**2
        + c4*(BpmRp)**3
        + c5*A_0
        + c6*A_0**2
        + c7*A_0*BpmRp
    )

    A_X = k_X * A_0

    return A_X


def append_corrected_gaia_phot_Gaia2018(df):
    """
    Using the coefficients calculated by GaiaCollaboration+2018 Table
    1, and the STILISM reddening values, calculate corrected Gaia
    photometric magnitudes.  Assumes you have acquired STILISM
    reddening estimates per retrieve_stilism_reddening below.

    Args:
        df (DataFrame): contains Gaia photometric magnitudes, and STILISM
        reddening columns.

    Returns:
        Same DataFrame, with 'phot_g_mean_mag_corr', 'phot_rp_mean_mag_corr',
        'phot_bp_mean_mag_corr' columns.
    """

    corr_path = os.path.join(DATADIR, 'extinction',
                             'GaiaCollaboration_2018_table1.csv')
    cdf = pd.read_csv(corr_path, sep=',')

    bandpasses = ['G','BP','RP']

    E_BmV = df['reddening[mag][stilism]']
    A_0 = 3.1 * E_BmV

    for bp in bandpasses:

        c1 = float(cdf.loc[cdf.bandpass==f'k{bp}', 'c1'])
        c2 = float(cdf.loc[cdf.bandpass==f'k{bp}', 'c2'])
        c3 = float(cdf.loc[cdf.bandpass==f'k{bp}', 'c3'])
        c4 = float(cdf.loc[cdf.bandpass==f'k{bp}', 'c4'])
        c5 = float(cdf.loc[cdf.bandpass==f'k{bp}', 'c5'])
        c6 = float(cdf.loc[cdf.bandpass==f'k{bp}', 'c6'])
        c7 = float(cdf.loc[cdf.bandpass==f'k{bp}', 'c7'])

        BpmRp = df.phot_bp_mean_mag - df.phot_rp_mean_mag

        # nb Eq 1 of the paper has BpmRp0 ... which presents a bit of
        # a self-consistency issue
        k_X = (
            c1
            + c2*(BpmRp)
            + c3*(BpmRp)**2
            + c4*(BpmRp)**3
            + c5*A_0
            + c6*A_0**2
            + c7*A_0*BpmRp
        )

        A_X = k_X * A_0

        # each "corrected magnitude" is whatever the observed
        # magnitude was, MINUS the extinction (because it needs to
        # lower the magnitude, making the star brighter)
        df[f'phot_{bp.lower()}_mean_mag_corr'] = (
            df[f'phot_{bp.lower()}_mean_mag'] - A_X
        )

    return df


def append_corrected_gaia_phot_Gagne2020(df):
    """
    Using the coefficients calculated by Gagne+20 Table 8, and the STILISM
    reddening values, calculate corrected Gaia photometric magnitudes.  Assumes
    you have acquired STILISM reddening estimates per
    retrieve_stilism_reddening below.

    Args:
        df (DataFrame): contains Gaia photometric magnitudes, and STILISM
        reddening columns.

    Returns:
        Same DataFrame, with 'phot_g_mean_mag_corr', 'phot_rp_mean_mag_corr',
        'phot_bp_mean_mag_corr' columns.
    """
    corr_path = os.path.join(DATADIR, 'extinction',
                             'Gagne_2020_apjabb77et8_ascii.csv')
    cdf = pd.read_csv(corr_path, comment='#', sep=',')

    # define the interpolation functions
    fn_GmRp_to_R_G = interp1d(
        cdf['G-G_RP_uncorrected'], cdf['R(G)'], kind='quadratic',
        bounds_error=False, fill_value=np.nan
    )
    fn_GmRp_to_R_G_RP = interp1d(
        cdf['G-G_RP_uncorrected'], cdf['R(G_RP)'], kind='quadratic',
        bounds_error=False, fill_value=np.nan
    )
    fn_GmRp_to_R_G_BP = interp1d(
        cdf['G-G_RP_uncorrected'], cdf['R(G_BP)'], kind='quadratic',
        bounds_error=False, fill_value=np.nan
    )

    GmRp = df['phot_g_mean_mag'] - df['phot_rp_mean_mag']

    R_G = fn_GmRp_to_R_G(GmRp)
    R_G_RP = fn_GmRp_to_R_G_RP(GmRp)
    R_G_BP = fn_GmRp_to_R_G_BP(GmRp)

    assert 'reddening[mag][stilism]' in df
    E_BmV = df['reddening[mag][stilism]']

    G_corr = df['phot_g_mean_mag'] - E_BmV * R_G
    G_RP_corr = df['phot_rp_mean_mag'] - E_BmV * R_G_RP
    G_BP_corr = df['phot_bp_mean_mag'] - E_BmV * R_G_BP

    df['phot_g_mean_mag_corr'] = G_corr
    df['phot_rp_mean_mag_corr'] = G_RP_corr
    df['phot_bp_mean_mag_corr'] = G_BP_corr

    return df


def retrieve_stilism_reddening(df, verbose=True, outpath=None):
    """
    Note: this is slow. 1 second per item. so, 10 minutes for 600 queries.
    --------------------
    (Quoting the website)
    Retrieve tridimensional maps of the local InterStellar Matter (ISM) based on
    measurements of starlight absorption by dust (reddening effects) or gaseous
    species (absorption lines or bands). See Lallement et al, A&A, 561, A91 (2014),
    Capitanio et al, A&A, 606, A65 (2017), Lallement et al, submitted (2018). The
    current map is based on the inversion of reddening estimates towards 71,000
    target stars.

    Institute : Observatoire de Paris
    Version : 4.1
    Creation date : 2018-03-19T13:45:30.782928
    Grid unit :
        x ∈ [-1997.5, 1997.5] with step of 5 parsec
        y ∈ [-1997.5, 1997.5] with step of 5 parsec
        z ∈ [-297.5, 297.5] with step of 5 parsec
        Sun position : (0,0,0)
        Values unit : magnitude/parsec
    --------------------
    Args:

    df:
        pandas DataFrame with columns:  l, b, distance (deg, deg and pc)

    Returns:

        DataFrame with new columns:  "distance[pc][stilism]",
        "reddening[mag][stilism]", "distance_uncertainty[pc][stilism]",
        "reddening_uncertainty_min[mag][stilism]",
        "reddening_uncertainty_max[mag][stilism]"

    Where "reddening" means "E(B-V)".
    """

    URL = "http://stilism.obspm.fr/reddening?frame=galactic&vlong={}&ulong=deg&vlat={}&ulat=deg&distance={}"

    df.loc[:, "distance[pc][stilism]"] = np.nan
    df.loc[:, "reddening[mag][stilism]"] = np.nan
    df.loc[:, "distance_uncertainty[pc][stilism]"] = np.nan
    df.loc[:, "reddening_uncertainty_min[mag][stilism]"] = np.nan
    df.loc[:, "reddening_uncertainty_max[mag][stilism]"] = np.nan

    print('Beginning STILISM webqueries...')
    for index, row in df.iterrows():
        print(f'{datetime.utcnow().isoformat()}: {index}/{len(df)}...')

        if verbose:
            print("l:", row["l"], "deg, b:", row["b"], "deg, distance:",
                  row["distance"], "pc")

        res = requests.get(
            URL.format(row["l"], row["b"], row["distance"]), allow_redirects=True
        )
        if res.ok:
            file = StringIO(res.content.decode("utf-8"))
            dfstilism = pd.read_csv(file)
            if verbose:
                print(dfstilism)
            df.loc[index, "distance[pc][stilism]"] = (
                dfstilism["distance[pc]"][0]
            )
            df.loc[index, "reddening[mag][stilism]"] = (
                dfstilism["reddening[mag]"][0]
            )
            df.loc[index, "distance_uncertainty[pc][stilism]"] = (
                dfstilism["distance_uncertainty[pc]"][0]
            )
            df.loc[index, "reddening_uncertainty_min[mag][stilism]"] = (
                dfstilism["reddening_uncertainty_min[mag]"][0]
            )
            df.loc[index, "reddening_uncertainty_max[mag][stilism]"] = (
                dfstilism["reddening_uncertainty_max[mag]"][0]
            )

    if isinstance(outpath, str):
        df.to_csv(outpath, index=False)

    return df
