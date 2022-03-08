"""
# pull a big box around rsg5
# looking at kerr's plot and KC19, we're going to want plx=2-5mas
# spanning galactic latitudes of 0 to 25 deg
# glon of 40 to 100
"""

from astropy.io import fits
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord
from rudolf.paths import LOCALDIR, DATADIR, RESULTSDIR
import os
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from numpy import array as nparr

from rudolf.helpers import get_clean_gaia_photometric_sources
from earhart.physicalpositions import (
    append_physicalpositions, calc_vl_vb_physical
)
from rudolf.plotting import plot_full_kinematics

csvpath = os.path.join(
    DATADIR, 'Cep-Her', 'GaiaArchiveQuery_CepHer-result_cleaned.csv'
)

if not os.path.exists(csvpath):

    # From the Gaia archive, I ran:
    initial_sql_query = (
        "SELECT * "
        "FROM gaiaedr3.gaia_source "
        "WHERE b > 0 and b < 25 and l > 40 and l < 100 "
        "and parallax > 2 and parallax < 5 "
    )

    # This yielded 3,397,462 stars.  Not too bad!
    fitspath = os.path.join(LOCALDIR, 'cepher-result.fits')

    if not os.path.exists(fitspath):
        print('Need to run')
        print(initial_sql_query)
        assert 0

    df = Table.read(fitspath, format='fits').to_pandas()

    # probably want to do some basic quality impositions...
    sdf = df[get_clean_gaia_photometric_sources(df)]

    # calculate v_l and v_b
    v_l_cosb_km_per_sec, v_b_km_per_sec = calc_vl_vb_physical(
        nparr(sdf.ra), nparr(sdf.dec),
        nparr(sdf.pmra), nparr(sdf.pmdec),
        nparr(sdf.parallax)
    )

    sdf['v_l_cosb_kms'] = v_l_cosb_km_per_sec
    sdf['v_b_kms'] = v_b_km_per_sec

    # calculate XYZ, and also vtang relative to median RSG5 parameters.
    # (latter based on CG20 list)
    # https://ui.adsabs.harvard.edu/abs/2020A%26A...633A..99C/abstract
    fitspath = os.path.join(DATADIR, 'Cep-Her',
                            'CantatGaudin_2020_mirages_RSG5_only.fits')
    cg20_rsg5_df = Table.read(fitspath, format='fits').to_pandas()

    mapdict = {
        'RA_ICRS':'ra',
        'DE_ICRS':'dec',
        'GLON':'l',
        'GLAT':'b',
        'Plx':'parallax',
        'pmRA':'pmra',
        'pmDE':'pmdec',
        'RV':'radial_velocity'
    }
    cg20_rsg5_df = cg20_rsg5_df.rename(columns=mapdict)

    # "core" members
    scg20_rsg5_df = cg20_rsg5_df[cg20_rsg5_df.Proba > 0.7]

    getcols = ['ra', 'dec', 'parallax', 'pmra', 'pmdec', 'radial_velocity']
    median_rsg5_df = pd.DataFrame(scg20_rsg5_df[getcols].median()).T
    spdf = append_physicalpositions(sdf, median_rsg5_df)

    print(f'Ran initial sql query:')
    print(initial_sql_query)
    print(f'... yielded {len(df)} sources.')
    print(f'Imposing basic quality cuts:')
    print(f'... yielded {len(sdf)} sources.')
    print(f'... for which positions/vtang were computed, relative to median '
          'CG20 RSG5 members.')

    # FIXME: you're going to want to compute the HR diagram parameters too for
    # quicklook in glue
    spdf['M_G'] = np.array(
            spdf['phot_g_mean_mag'] + 5*np.log10(spdf['parallax']/1e3) + 5
    )

    spdf['BP-RP'] = np.array(
            spdf['phot_bp_mean_mag'] - spdf['phot_rp_mean_mag']
    )

    spdf.to_csv(csvpath, index=False)

spdf = pd.read_csv(csvpath)
import IPython; IPython.embed()
