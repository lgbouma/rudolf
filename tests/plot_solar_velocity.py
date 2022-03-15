import os
import numpy as np, matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy import units as u

from astropy.coordinates import (
    Galactocentric, CartesianRepresentation, CartesianDifferential
)
import astropy.coordinates as coord
_ = coord.galactocentric_frame_defaults.set('v4.0')

from astropy.coordinates import Galactic

# Schonrich+2010 solar velocity wrt local standard of rest
U = 11.10 * u.km/u.s
U_hi = 0.69 * u.km/u.s
U_lo = 0.75 * u.km/u.s
V = 12.24 * u.km/u.s
V_hi = 0.47 * u.km/u.s
V_lo = 0.47 * u.km/u.s
W = 7.25 * u.km/u.s
W_hi = 0.37 * u.km/u.s
W_lo = 0.36 * u.km/u.s

# # note "uvw" mean "x,y,z" in galactic coordinates
# c1 = Galactic(u=10*u.pc, v=200*u.pc, w=20*u.pc, U=U, V=V, W=W,
#               representation_type=CartesianRepresentation,
#               differential_type=CartesianDifferential)

# define initial frame spanning longitude...
lons = np.arange(0,360,1)
_lat = 42   # doesn't matter
_dist = 300 # doesn't matter

lats = _lat*np.ones_like(lons)
dists_pc = _dist*np.ones_like(lons)

# + + +
# - - -
# + + -
# - - +
# + - -
# - + +
# + - +
# - + -

for _U, _V, _W, label in zip(
    [U, U+U_hi, U-U_lo],# U+U_hi, U-U_lo],
    [V, V+V_hi, V-V_lo],# V+V_hi, V-V_lo],
    [W, W+W_hi, W-W_lo],# W-W_lo, W+W_hi],
    [0, 1, 2]#, 3, 4]
):
    print(42*'-')
    print(label)
    v_l_cosb_kms_list = []
    for ix, (lon, lat, dist_pc) in enumerate(zip(lons, lats, dists_pc)):
        print(f'{ix}/{len(lons)}')
        gal = Galactic(lon*u.deg, lat*u.deg, distance=dist_pc*u.pc)
        vel_to_add = CartesianDifferential(_U, _V, _W)
        newdata = gal.data.to_cartesian().with_differentials(vel_to_add)
        newgal = gal.realize_frame(newdata)
        pm_l_cosb_AU_per_yr = (newgal.pm_l_cosb.value*1e-3) * dist_pc * (1*u.AU/u.yr)
        v_l_cosb_kms = pm_l_cosb_AU_per_yr.to(u.km/u.second)
        v_l_cosb_kms_list.append(v_l_cosb_kms.value)

    v_l_cosb_kms = -np.array(v_l_cosb_kms_list)

    plt.plot(lons, v_l_cosb_kms, lw=1, label=label)

plt.legend(fontsize='x-small')
plt.xlabel('galactic lon')
plt.ylabel('v_l*')
plt.savefig('test_solar_velocity.png', dpi=400)
