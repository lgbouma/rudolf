import os
import rudolf.plotting as rp
from rudolf.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'rotation')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

from earhart.priors import AVG_EBpmRp

runid = 'deltaLyrCluster'

#TODO: do a cut in rotation periods?
# c = 'defaultcleaning_cutProtColor'
# rp.plot_rotationperiod_vs_color(
#     PLOTDIR, runid, E_BpmRp, core_halo=1, yscale='linear', cleaning=c,
#     emph_binaries=0, talk_aspect=1, showPleiadesQuad=0
# )
# rp.plot_rotationperiod_vs_color(
#     PLOTDIR, runid, E_BpmRp, core_halo=0, yscale='linear', cleaning=c,
#     emph_binaries=1, talk_aspect=1, showPleiadesQuad=0
# )

# add kep1627
rp.plot_rotationperiod_vs_color(
    PLOTDIR, runid, yscale='linear', cleaning='defaultcleaning',
    emph_binaries=0, talk_aspect=1, kinematic_selection=1,
    overplotkep1627=1
)
assert 0

for c in ['defaultcleaning', 'harderlsp', 'nocleaning']:
    for k in [1,0]:
        for e in [0,1]:
            rp.plot_rotationperiod_vs_color(
                PLOTDIR, runid, yscale='linear', cleaning=c,
                emph_binaries=e, talk_aspect=1, kinematic_selection=k
            )

# pleiades and praesepe only
rp.plot_rotationperiod_vs_color(
    PLOTDIR, runid, yscale='linear',
    cleaning='defaultcleaning', emph_binaries=0, refcluster_only=True,
    talk_aspect=1
)

# TODO: what about the binaries?
