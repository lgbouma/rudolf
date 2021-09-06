import numpy as np, pandas as pd, matplotlib.pyplot as plt
import os, pickle
from numpy import array as nparr

from astropy import units as u, constants as c

# Follow Owen+20 guesstimates
upper_delta_Mp = 0.4*u.Mearth
lower_delta_Mp = 0.1*u.Mearth

longest_delta_t = 50*u.Myr
shortest_delta_t = 20*u.Myr

biggest_Mdot = upper_delta_Mp/shortest_delta_t
smallest_Mdot = lower_delta_Mp/longest_delta_t

print(f'biggest_Mdot: {biggest_Mdot}')
print(f'smallest_Mdot: {smallest_Mdot}')
