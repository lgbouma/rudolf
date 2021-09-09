import os
import rudolf.plotting as rp
from rudolf.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'RM')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

models = ['quadratic','linear','trendonly','quadraticretrograde','quadraticprograde']
#models = ['quadraticretrograde', 'quadraticprograde']
#models = ['trendonly']

for model in models:
    rp.plot_RM(PLOTDIR, model=model)
    rp.plot_RM_and_phot(PLOTDIR, model=model)
