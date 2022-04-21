import numpy as np
from cdips.utils.gaiaqueries import given_dr3_sourceids_get_dr2_xmatch

dr3_source_ids = np.array(["2082142734285082368"]).astype(np.int64)

dr2_x_edr3_df = given_dr3_sourceids_get_dr2_xmatch(
    dr3_source_ids, 'test_0', overwrite=False
)

get_dr2_xm = lambda _df: (
	_df.sort_values(by='angular_distance').
	drop_duplicates(subset='dr3_source_id', keep='first')
)
s_dr2 = get_dr2_xm(dr2_x_edr3_df)

assert str(s_dr2.dr2_source_id.loc[0]) == "2082142734285082368"
