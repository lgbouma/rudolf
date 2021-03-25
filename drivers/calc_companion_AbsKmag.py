import numpy as np

comp_kpmag = 13.69 # 11.19 primary, +2.5 dmag guess

plx = 3.0087/1e3

abs_kpmag = comp_kpmag + 5*np.log10(plx) + 5

print(f'{abs_kpmag:.2f}')
