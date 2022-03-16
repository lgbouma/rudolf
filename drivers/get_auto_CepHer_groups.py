import os
import pandas as pd, matplotlib.pyplot as plt
from rudolf.paths import DATADIR, RESULTSDIR

# get data, made via plot_CepHer_XYZvtang_sky
csvpath = os.path.join(RESULTSDIR, 'CepHer_XYZvtang_sky',
                       'weight_strengthgt0.00.csv')
df = pd.read_csv(csvpath)

print(f'N stars after applying photometric and astrometric quality cuts: {len(df)}')

x0 = 8122

# RSG-5:
# X: 45 to 75
# Y: 320 to 350
# Z: 40 to 70
# vl: 4 to 6 kms
# vb: -3 to -4 kms

sel = (
    (df['x_pc']+x0 > 45) &
    (df['x_pc']+x0 < 75) &
    (df['y_pc'] > 320) &
    (df['y_pc'] < 350) &
    (df['z_pc'] > 40) &
    (df['z_pc'] < 70) &
    (df['v_b'] > -4) &
    (df['v_b'] < -3) &
    (df['v_l*'] > 4) &
    (df['v_l*'] < 6)
)
sdf = df[sel]
print(f'N candidate RSG-5 members: {len(sdf)}')

csvpath = os.path.join(RESULTSDIR, 'CepHer_XYZvtang_sky',
                       'RSG-5_auto_XYZ_vl_vb_cut.csv')
sdf.to_csv(csvpath, index=False)
print(f'Wrote {csvpath}')

plt.close('all')
fig, ax = plt.subplots()
ax.scatter(sdf['bp-rp'], sdf['M_G'], s=1, c='k')
ax.update({'xlabel':'bp-rp', 'ylabel':'M_G', 'ylim':ax.get_ylim()[::-1]})
fig.savefig(
    os.path.join(RESULTSDIR, 'CepHer_XYZvtang_sky', 'RSG-5_auto_XYZ_vl_vb_cut_CAMD.png'),
    dpi=400
)

# CH-2:
# X: 20 to 80
# Y: 230 to 270
# Z: 70 to 105
# vb: -3.5 to -1.5 kms
# vl: 2 to 6 kms

sel = (
    (df['x_pc']+x0 > 20) &
    (df['x_pc']+x0 < 70) &
    (df['y_pc'] > 230) &
    (df['y_pc'] < 270) &
    (df['z_pc'] > 75) &
    (df['z_pc'] < 105) &
    (df['v_b'] > -3.5) &
    (df['v_b'] < -1.5) &
    (df['v_l*'] > 2) &
    (df['v_l*'] < 6)
)
sdf = df[sel]
print(f'N candidate CH-2 members: {len(sdf)}')

csvpath = os.path.join(RESULTSDIR, 'CepHer_XYZvtang_sky',
                       'CH-2_auto_XYZ_vl_vb_cut.csv')
sdf.to_csv(csvpath, index=False)
print(f'Wrote {csvpath}')

plt.close('all')
fig, ax = plt.subplots()
ax.scatter(sdf['bp-rp'], sdf['M_G'], s=1, c='k')
ax.update({'xlabel':'bp-rp', 'ylabel':'M_G', 'ylim':ax.get_ylim()[::-1]})
fig.savefig(
    os.path.join(RESULTSDIR, 'CepHer_XYZvtang_sky', 'CH-2_auto_XYZ_vl_vb_cut_CAMD.png'),
    dpi=400
)
