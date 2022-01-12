"""
# pull a big box around rsg5
# looking at kerr's plot, we're going to want d=200-400pc
# spanning galactic latitudes of 0 to 20 deg  (maybe 2 to 20 deg)
# glon of 70 to 90 to get RSG-5 and 2:???
# glon down to 50 to get the stephenson-1/delta lyr complex
"""

from astropy.io import fits
from astropy.table import Table
from rudolf.paths import LOCALDIR, DATADIR, RESULTSDIR
import os
import numpy as np, pandas as pd, matplotlib.pyplot as plt

from rudolf.helpers import get_clean_gaia_photometric_sources
from earhart.physicalpositions import append_physicalpositions
from rudolf.plotting import plot_full_kinematics

csvpath = os.path.join(
    DATADIR, 'Cep-Her', 'GaiaArchiveQuery_rsg5-result_cleaned.csv'
)

if not os.path.exists(csvpath):

    # From the Gaia archive, I ran:
    initial_sql_query = (
        "SELECT * "
        "FROM gaiaedr3.gaia_source "
        "WHERE b > 0 and b < 20 and l > 70 and l < 90 "
        "and parallax > 2.5 and parallax < 5 "
    )
    fitspath = os.path.join(LOCALDIR, 'rsg5-result.fits')
    df = Table.read(fitspath, format='fits').to_pandas()

    # probably want to do some basic quality impositions...
    sdf = df[get_clean_gaia_photometric_sources(df)]

    # get whatever the median RSG5 parameters are. calculate XYZ and vtang relative
    # to them.  do this by cutting on Prob>0.5 RSG_5 members from
    # Cantat-Gaudin+2020 (mirages):
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

    outdir = os.path.join(RESULTSDIR, 'Cep-Her')
    plot_full_kinematics(cg20_rsg5_df, 'RSG5_CG20', outdir, galacticframe=1)

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
