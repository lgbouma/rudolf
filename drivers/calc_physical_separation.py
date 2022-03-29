from astropy.coordinates import SkyCoord
from astropy import units as u, constants as c
from astropy.table import Table
import pandas as pd, numpy as np
from cdips.utils.gaiaqueries import parallax_to_distance_highsn

t = Table.read('../data/gaia/KOI7913_EDR3_system.fits')
t['source_id'] = t['Source'].astype(str)

df = t.to_pandas()

a = '2106235301785454208' # primary
b = '2106235301785453824' # secondary

adf = df[df.source_id.str.contains(a)]
bdf = df[df.source_id.str.contains(b)]

dist_a = parallax_to_distance_highsn(float(adf.Plx))
dist_b = parallax_to_distance_highsn(float(bdf.Plx))
avg_dist = (dist_a + dist_b)/2

c_a = SkyCoord(ra=float(adf.RA_ICRS)*u.degree,
               dec=float(adf.DE_ICRS)*u.degree,
               distance=dist_a*u.pc, frame='icrs')
c_b = SkyCoord(ra=float(bdf.RA_ICRS)*u.degree,
               dec=float(bdf.DE_ICRS)*u.degree,
               distance=dist_b*u.pc, frame='icrs')

sep_3d = c_a.separation_3d(c_b)

print('3d separation')
print(sep_3d.to(u.pc))
print(sep_3d.to(u.AU))

sep = c_a.separation(c_b)

print('2d separation')
print(sep.to(u.arcsec))

# small, relative to uncertainty from distance...
abs_unc = np.sqrt(
    (float(adf.e_RA_ICRS)*u.mas)**2 +
    (float(adf.e_DE_ICRS)*u.mas)**2 +
    (float(bdf.e_RA_ICRS)*u.mas)**2 +
    (float(bdf.e_DE_ICRS)*u.mas)**2
)
print('+/-', abs_unc.to(u.arcsec))

print(f'2d sep in AU: {sep.to(u.arcsec).value*avg_dist:.2f} AU')

rel_unc = (dist_a - dist_b)/dist_a

print(f'2d sep unc in AU: {(rel_unc*sep).to(u.arcsec).value*avg_dist:.2f} AU')
import IPython; IPython.embed()
