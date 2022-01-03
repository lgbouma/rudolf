#!/usr/bin/env python3
"""
This example script reads Table 3 from Bouma+2022, and then uses the pandas
boolean logic operators to select a set of interesting open clusters in the
northern celestial hemisphere studied by Kounkel & Covey 2019.

Author: Luke Bouma
Contact: <luke@astro.caltech.edu>
"""
import pandas as pd
import numpy as np

# Read in the table of young, age-dated, and age-dateable stars using the
# pandas DataFrame type. Select only stars that are actually reported to be in
# a cluster, rather than field stars thought to be young.
df = pd.read_csv("cdips_targets_v0.6_nomagcut_gaiasources_table.csv")
df = df[~pd.isnull(df.cluster)]

# In this example, we will select clusters from the union of "cluster_names"
# and "special_names".  As in the first example, this selection is not complete
# for any given cluster.  For the Pleiades for instance; other possible strings
# to match on include "Melotte_22", "Melotte 22", would yield more members.
# The best way to resolve all possibilities if you have a particular cluster
# for which you want all possible candidates would be to use a function that
# finds all the unique strings (like np.unique).

# NOTE: these names come from Kounkel & Covey 2019.  Other authors may have
# used different names to refer to the same clusters.
cluster_names = [
    "Stephenson_1", # logt 7.5, 100-500pc, aka. Theia 73
    "Alpha_per", # logt 7.8, 100-200pc, Theia 133
    "kc19group_506", # 350 Myr, 100pc, aka. Theia 506
    "kc19group_507", # 350 Myr, 100-200pc, aka. Theia 507.
    "AB_Dor", # ~120 Myr, 100-300pc from KC19.
    "Pleiades", # 120 Myr, 120 pc.
]
special_names = [
    "UBC_1", # 350 Myr, 300-400pc, aka. Theia 520
    "RSG_5", # logt 7.5, 300-400pc.
]

sel = np.zeros(len(df)).astype(bool)
for cluster_name in cluster_names:
    sel |= df.cluster.str.contains(cluster_name)

# Avoid "UBC_1" matching to clusters like "UBC_10", "UBC_11", "UBC_186",
# when performing the string matching.
for cluster_name in special_names:
    sel |= (
        # This selects singleton cases
        (df.cluster == cluster_name)
        |
        # This selects cases that contain the string "UBC_1,", i.e., they begin
        # with that name.
        df.cluster.str.contains(cluster_name+',')
        |
        # This selects cases that contain the string ",UBC_1,", i.e., UBC_1 is
        # somewhere in the middle of the string.
        df.cluster.str.contains(','+cluster_name+',')
        |
        # This selects cases that end with the string "UBC_1".  Note that
        # pandas matches the pattern as a regular expression, see
        # https://pandas.pydata.org/docs/reference/api/pandas.Series.str.contains.html
        df.cluster.str.contains(','+cluster_name+'$')
    )

np.random.seed(42)
sdf = df[sel]
selcols = 'dr2_source_id,cluster,mean_age,age,reference_id,reference_bibcode'.split(',')
N = 30

print(f'Selecting stars matching {cluster_names} and {special_names}...')
print(f'Yields {len(sdf)} stars')
print(f'A random sampling of N={N} them is as follows:')
print(sdf[selcols].sample(n=N))
