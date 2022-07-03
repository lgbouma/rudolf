import os
import rudolf.plotting as rp
from rudolf.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'rotation')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

from earhart.priors import AVG_EBpmRp

# eres-vii
rp.plot_rotationperiod_vs_color(
    PLOTDIR, 'RSG-5', yscale='linear', cleaning='defaultcleaning',
    emph_binaries=0, talk_aspect=1, kinematic_selection=1,
    overplotkep1627=0, show_allknown=1, darkcolors=1
)
assert 0

# add all KOIs
for runid in ['CH-2', 'RSG-5', 'deltaLyrCluster']:
    rp.plot_rotationperiod_vs_color(
        PLOTDIR, runid, yscale='linear', cleaning='defaultcleaning',
        emph_binaries=0, talk_aspect=1, kinematic_selection=1,
        overplotkep1627=0, show_allknown=1
    )

runid = 'deltaLyrCluster'
# pleiades and praesepe only
rp.plot_rotationperiod_vs_color(
    PLOTDIR, runid, yscale='linear',
    cleaning='defaultcleaning', emph_binaries=0, refcluster_only=True,
    talk_aspect=1
)
# add kep1627
rp.plot_rotationperiod_vs_color(
    PLOTDIR, runid, yscale='linear', cleaning='defaultcleaning',
    emph_binaries=0, talk_aspect=1, kinematic_selection=1,
    overplotkep1627=1
)
# add kep1627
rp.plot_rotationperiod_vs_color(
    PLOTDIR, runid, yscale='linear', cleaning='curtiscleaning',
    emph_binaries=0, talk_aspect=1, kinematic_selection=1,
    overplotkep1627=1
)

for c in ['defaultcleaning', 'curtiscleaning', 'harderlsp', 'nocleaning']:
    for k in [1,0]:
        for e in [0,1]:
            rp.plot_rotationperiod_vs_color(
                PLOTDIR, runid, yscale='linear', cleaning=c,
                emph_binaries=e, talk_aspect=1, kinematic_selection=k
            )
