import os
import rudolf.plotting as rp
from rudolf.paths import RESULTSDIR
import numpy as np

PLOTDIR = os.path.join(RESULTSDIR, 'phasedlc_quartiles')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

ylimd = {'0':[-3.5, 2.5],'1':[-3.5, 2.5],'2':[-3.5, 2.5],'3':[-3.5, 2.5],
         '4':[-0.19,0.19],'5':[-0.19,0.19],'6':[-0.19,0.19],'7':[-0.19,0.19],}
rp.plot_phasedlc_quartiles(PLOTDIR, whichtype='ttv', from_trace=True,
                           ylimd=ylimd, fullxlim=False,
                           do_hacky_reprerror=True)
rp.plot_phasedlc_quartiles(PLOTDIR, whichtype='slope', from_trace=True,
                           ylimd=ylimd, fullxlim=False,
                           do_hacky_reprerror=True)
