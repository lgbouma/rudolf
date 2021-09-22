import os
import rudolf.plotting as rp
from rudolf.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'RM')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

models = ['quadraticfreejitter', 'linearfreejitter', 'trendonlyfreejitter',
          'quadraticretrogradefreejitter','quadraticprogradefreejitter']
#models = ['trendonlyfreejitter',
#          'quadraticretrogradefreejitter','quadraticprogradefreejitter']


#models = ['quadraticfreejitter', 'quadratic']
#models = ['quadratic','linear','trendonly','quadraticretrograde','quadraticprograde']
#models = ['quadraticretrograde', 'quadraticprograde']
#models = ['trendonly']

for model in models:
    rp.plot_RM(PLOTDIR, model=model)
    rp.plot_RM_and_phot(PLOTDIR, model=model, showmodel=1)
    rp.plot_RM_and_phot(PLOTDIR, model=model, showmodelbands=1)
    rp.plot_RM_and_phot(PLOTDIR, model=model)
