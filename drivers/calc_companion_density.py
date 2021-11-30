from astropy import units as u, constants as c
import numpy as np

# from http://perso.ens-lyon.fr/isabelle.baraffe/BHAC15dir/BHAC15_tracks+structure

# at 32 Myr (logt 7.503435), 0.300 Msun has Rstar 0.481 
# at 35 Myr (logt 7.544197), 0.300 Msun has Rstar 0.465
# at 40 Myr (logt 7.605767), 0.300 Msun has Rstar 0.443
#
# at 32 Myr (logt 7.499998), 0.400 Msun has Rstar 0.555
# at 35 Myr (logt 7.544004), 0.400 Msun has Rstar 0.537
# at 40 Myr (logt 7.603379), 0.400 Msun has Rstar 0.514

ages = [32,35,40,
        32,35,40,
        32,35,40]
Ms = np.array(
    [0.3,0.3,0.3,
     0.4,0.4,0.4,
     0.33,0.33,0.33]
)*u.Msun
Rs = np.array(
    [0.481,0.465,0.443,
     0.555,0.537,0.514,
     (0.555*1/3+0.481*2/3), (0.537*1/3+0.465*2/3), (0.514*1/3+0.443*2/3)]
)*u.Rsun

for M, R, A in zip(Ms,Rs,ages):

    V = (4/3)*np.pi*R**3

    rho = M/V

    print(f'{A} Myr {M.value:.3f} Msun, {R.value:.3f} Rsun, {rho.cgs:.2f}')

print('default: 40 Myr, 0.33 Msun')
#FIXME FIXME TODO VERIFY WHETHER THIS RSTAR IS AT ALL COMPATIBLE WITH THE
#QUOTED TEFF... ALSO HOW EXACTLY DID KRAUS DO HIS ESTIMATE?
