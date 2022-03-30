import pandas as pd, numpy as np
import os
from rudolf.paths import DATADIR

# get data
csvpath = os.path.join(DATADIR, 'Cep-Her',
                       '20220311_Kerr_CepHer_Extended_Candidates.csv')
df = pd.read_csv(csvpath)
df = df[(df['photometric flag'].astype(bool)) & (df['astrometric flag'].astype(bool))]

print(f'Start with {len(df)} candidate members.')
print(f'Start with {len(df[df.strengths>0.02])} cands w/ strength>0.02.')

# check 
csvpath = os.path.join(DATADIR, 'Cep-Her',
                       'KOI_crossmatches_weight_gt_0.02.txt')
kdf = pd.read_csv(csvpath)
kdf.kicid = kdf.kicid.astype(str)

csvpaths = [
    os.path.join(DATADIR, 'NasaExoplanetArchive',
                 'q1_q17_dr25_koi_2022.03.27_08.37.49.csv'),
    os.path.join(DATADIR, 'NasaExoplanetArchive',
                 'cumulative_2022.03.27_08.38.21.csv'),
]

for csvpath in csvpaths:
    print(42*'-')
    print(os.path.basename(csvpath))
    koi_df = pd.read_csv(csvpath, comment='#', sep=',')
    koi_df['kepid'] = koi_df['kepid'].astype(str)

    mdf = kdf.merge(koi_df, how='left', left_on='kicid', right_on='kepid')

    for disposition in ['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE']:
        print(42*'-')
        print(disposition)
        print(mdf[mdf.koi_disposition == disposition])
