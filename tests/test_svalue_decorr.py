import os
import numpy as np, pandas as pd
from rudolf.paths import DATADIR, RESULTSDIR
from numpy import array as nparr
import matplotlib.pyplot as plt
from aesthetic.plot import savefig, format_ax, set_style

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

rvpath = os.path.join(DATADIR, 'spec', '20210809_rvs_template_V1298TAU.csv')
outdir = os.path.join(RESULTSDIR, 'RM', 'svalue_decorr_tests')
if not os.path.exists(outdir):
    os.mkdir(outdir)

df = pd.read_csv(rvpath)

rv_vec = nparr(df.rv) - np.mean(df.rv)
e_rv_vec = nparr(df.e_rv)
s_vec = nparr(df.svalue) - np.mean(df.svalue)

# exclude outlier
sel = ~(df.bjd == 2459433.975944)
s_rv_vec = rv_vec[sel]
s_e_rv_vec = e_rv_vec[sel]
s_s_vec = s_vec[sel]
s_time = df.bjd[sel]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(s_s_vec.reshape(-1,1), s_rv_vec)

# Make predictions
rv_pred = regr.predict(s_s_vec.reshape(-1,1))

# The coefficients
print('Coefficients: \n', regr.coef_)
# The coefficient of determination: 1 is perfect prediction

# Plot outputs
plt.errorbar(s_s_vec, s_rv_vec, s_e_rv_vec, color='black', elinewidth=0.5,
             capsize=4, lw=0, mew=0.5, markersize=3)
plt.plot(s_s_vec, rv_pred, color='blue', linewidth=3)
plt.xlabel('S-value')
plt.ylabel('RV')

plt.savefig(os.path.join(outdir, 'rv_vs_svalue_linearregression.png'))

# Calc resid, plot it
plt.close('all')
rv_resid = s_rv_vec - rv_pred

set_style()

fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(4,3), sharex=True)

axs[0].errorbar(s_time, s_rv_vec, s_e_rv_vec, zorder=2, c='k', elinewidth=0.5, capsize=4,
                lw=0, mew=0.5, markersize=3)
axs[0].scatter(df.bjd[~sel], rv_vec[~sel], zorder=2, c='r', s=1, marker='X')
axs[0].plot(s_time, rv_pred, zorder=1, c='C0')
axs[0].set_ylabel('RV')

axs[1].scatter(s_time, s_s_vec, c='k', s=1)
axs[1].scatter(df.bjd[~sel], s_vec[~sel], zorder=2, c='r', s=1, marker='X')
axs[1].set_ylabel('S-value')

axs[2].errorbar(s_time, rv_resid, s_e_rv_vec, c='k', elinewidth=0.5, capsize=4,
                lw=0, mew=0.5, markersize=3)
axs[2].set_ylabel('RV resid')
axs[2].set_xlabel('time')

fig.tight_layout()

plt.savefig(os.path.join(outdir, 'rv_vs_svalue_linearregression_resid.png'), dpi=300)
