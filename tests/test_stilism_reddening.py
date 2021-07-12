from rudolf.helpers import get_deltalyr_kc19_cleansubset
from rudolf.extinction import (
    retrieve_stilism_reddening, append_corrected_gaia_phot_Gagne2020
)
from cdips.utils.gaiaqueries import parallax_to_distance_highsn

df = get_deltalyr_kc19_cleansubset()

df['distance'] = parallax_to_distance_highsn(df['parallax'])

rdf = retrieve_stilism_reddening(df.sample(n=10), verbose=False)
assert len(rdf) == 10

adf = append_corrected_gaia_phot_Gagne2020(rdf)

import IPython; IPython.embed()
