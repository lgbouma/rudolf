#!/usr/bin/env python3
"""
This example script reads Table 3 from Bouma+2022, also known as v0.6 of the
CDIPS target list.  After reading the table using python's pandas library, we
show what the table contains for some stars in the Pleiades.

Dependencies: python3, numpy, pandas
Author: Luke Bouma
Contact: <luke@astro.caltech.edu>
"""

import pandas as pd, numpy as np

# Read in the table of young, age-dated, and age-dateable stars using the
# pandas DataFrame type.  Note that astropy.table's Table.read function would
# work here as well -- the parser just needs to be able to distinguish
# comma-separated strings from the comma separation in the CSV table itself.
df = pd.read_csv("cdips_targets_v0.6_nomagcut_gaiasources_table.csv")

# Select only stars that are actually reported to be in a cluster, rather than
# field stars thought to be young.
sel = ~pd.isnull(df.cluster)
sdf = df[sel]

# How many unique cluster name strings are there? Note: if a given star is
# reported by multiple literature sources to be in a cluster, the "cluster",
# "age", "reference_id", and "reference_bibcode" columns are all comma
# separated strings.

# Assuming Python3 is being run, print using f-strings.
N_stars = len(sdf)
N_uniq = len(np.unique(sdf.cluster))
print(42*'.')
print(f"There are {N_stars} stars reported to be in clusters.")
print(f"These stars have {N_uniq} unique cluster name strings.")

# Search for stars containing the string "Pleiades".  Note that this selection
# is not complete!  Other possible strings to match on include "Melotte_22",
# "Melotte 22", and more.  The best way to resolve all possibilities if you
# have a particular cluster for which you want all possible candidates would be
# to use a function that finds all the unique strings (like np.unique).
clustername = 'Pleiades'
sel = sdf.cluster.str.contains(clustername)
_df = sdf[sel]
print(42*'.')
print(
    f"Searching for stars containing the string {clustername} "
    f"yields {len(_df)} stars, including the following."
)

selcols = 'dr2_source_id,cluster,age,reference_id,reference_bibcode'.split(',')
print(_df[selcols].sample(n=20))

print(42*'.')
print("Let's look at Gaia DR2 65112739296356608 closer...")

for c in selcols:
    source_id = np.int64(65112739296356608)
    print(f"{c}: {_df.loc[_df.dr2_source_id == source_id, c].iloc[0]}")

print('So each of the cluster, age, reference_id, and reference_bibcode '
      'columns is comma-separated to account for different reports from the '
      'literature.')
print(42*'.')

print('For a second manipulation example using pandas, see '
      'example_read_script_1.py')
