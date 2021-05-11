import numpy as np, pandas as pd
import re, os
# https://ads.readthedocs.io/en/latest/#getting-started
import ads

from rudolf.paths import PAPERDIR

csvpath = '/Users/luke/Dropbox/proj/cdips/data/cluster_data/list_of_lists_keys_paths_assembled_v0.5_gaiasources.csv'
df = pd.read_csv(csvpath)

outcols = (
    'reference_id,bibcode,N_gaia,Nstars_with_age,N_Rplt16_orclose'
)
outcols = outcols.split(',')

sdf = df[outcols]

# Don't mention that the table actually includes HATS and HATN candidates.
sel = (sdf.reference_id != 'HATSandHATNcandidates20210505')
sdf = sdf[sel].reset_index()

#
# First get all the bibtex information from ADS.
# replace "ARTICLE{2014AJ...}" style with reference_ids
#
assert np.all(~pd.isnull(sdf.bibcode))
bibcodes = list(sdf.bibcode)
bibcodes = [b.replace("A%26A","A&A") for b in bibcodes]

print('beginning ads query')
r = ads.ExportQuery(bibcodes).execute()

MAPDICT = {
    'candYSO':r'$\texttt{Y*?}$',
    'YSO':r'$\texttt{Y*O}$',
    'PMS':r'$\texttt{pMS*}$',
    'TTS':r'$\texttt{TT*}$',
    'candTTS':r'$\texttt{TT?}$',
    'gt250':r'$d>250\,{\rm pc}$',
    'lt250':r'$d>250\,{\rm pc}$',
    'ums':'UMS',
    'pms':'PMS',
    'PscEri': 'Psc-Eri',
    'Pleiades': 'Pleiades',
    't1': 'Coma-Ber',
    't2': 'Neighbor Group',
    '':''
}

ref_ids, scnds = [], []

for _b, _id in zip(bibcodes, list(sdf.reference_id)):

    if _id.startswith("SIMBAD_"):
        scnd = MAPDICT[_id.split("SIMBAD_")[1]]
        _id = "SIMBAD"
    elif _id.startswith("GaiaCollaboration2018"):
        scnd = MAPDICT[_id.split("GaiaCollaboration2018")[1]]
        _id = "GaiaCollaboration2018"
    elif _id.startswith("Damiani2019"):
        scnd = MAPDICT[_id.split("Damiani2019")[1]]
        _id = "Damiani2019"
    elif _id.startswith("Zari2018"):
        scnd = MAPDICT[_id.split("Zari2018")[1]]
        _id = "Zari2018"
    elif _id.startswith("RoserSchilbach2020"):
        scnd = MAPDICT[_id.split("RoserSchilbach2020")[1]]
        _id = "RoserSchilbach2020"
    elif _id.startswith("Furnkranz2019"):
        scnd = MAPDICT[_id.split("Furnkranz2019")[1]]
        _id = "Furnkranz2019"
    else:
        scnd = ''

    trgt = "ARTICLE{"+_b
    dstn = "ARTICLE{"+_id
    r = re.sub(trgt, dstn, r)
    ref_ids.append(_id)
    scnds.append(scnd)

outbibtex = os.path.join(PAPERDIR, 'metadata_table_references.bib')

with open(outbibtex, 'w') as f:
    f.writelines(r)

print(f'Wrote {outbibtex}')

#
# Then make and write the latex table.
#
ref = (
    '\citet{'+pd.Series(ref_ids).astype(str)+'} '+pd.Series(scnds).astype(str)
)

outdf = pd.DataFrame({
    'Reference': ref,
    'N_{\mathrm{Gaia}}': sdf.N_gaia,
    'N_{\mathrm{Age}}': sdf.Nstars_with_age,
    'N_{G_\mathrm{RP}<16}': sdf.N_Rplt16_orclose
})
outpath = os.path.join(PAPERDIR, 'metadata_table_data.tex')
outdf.to_latex(outpath, index=False, escape=False)
print(f'Wrote {outpath}')

