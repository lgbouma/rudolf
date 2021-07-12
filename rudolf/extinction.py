"""
Tools for working out the extinction.

General-purpose:
    retrieve_stilism_reddening
    get_corrected_gaia_phot_Gagne2020
"""

import numpy as np, pandas as pd
import requests
from io import StringIO
import sys, os
from datetime import datetime

from scipy.interpolate import interp1d

from rudolf.paths import DATADIR

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


def retrieve_stilism_reddening(df, verbose=True):
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

    return df
