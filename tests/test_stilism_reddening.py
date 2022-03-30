import numpy as np
from rudolf.helpers import get_deltalyr_kc19_cleansubset
from rudolf.extinction import (
    retrieve_stilism_reddening, append_corrected_gaia_phot_Gagne2020
)
from cdips.utils.gaiaqueries import (
    parallax_to_distance_highsn,
    given_source_ids_get_gaia_data
)

test_del_lyr = 0
test_single_star = 1

if test_del_lyr:
    df = get_deltalyr_kc19_cleansubset()
    df['distance'] = parallax_to_distance_highsn(df['parallax'])
    rdf = retrieve_stilism_reddening(df.sample(n=10), verbose=False)
    assert len(rdf) == 10
    adf = append_corrected_gaia_phot_Gagne2020(rdf)

if test_single_star:
    df = given_source_ids_get_gaia_data(
        np.array([np.int64(5288681857665822080)]).astype(np.int64),
        'justatest',
        gaia_datarelease='gaiadr2'
    )
    df['distance'] = parallax_to_distance_highsn(df['parallax'])
    rdf = retrieve_stilism_reddening(df, verbose=False)
    adf = append_corrected_gaia_phot_Gagne2020(rdf)

    EBmV = float(adf['reddening[mag][stilism]'])
    print(EBmV)

    bpmrp0 = float(adf['phot_bp_mean_mag'] - adf['phot_rp_mean_mag'])
    print(bpmrp0)

import IPython; IPython.embed()
