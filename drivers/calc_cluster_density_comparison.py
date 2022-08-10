N_rsg5 = 173
vol_rsg5 = 30*30*40*1.5*2.5
n_rsg5 = N_rsg5/vol_rsg5

N_ch2 = 37
vol_ch2 = 50*40*30*2*4
n_ch2 = N_ch2/vol_ch2

# Roser & Schilbach 2020 report 1387 stars in PscEri.  Requiring apparent G<16.5 (roughly comparable to our limiting magnitude in RSG-5 and CH-2), gives 858 stars.  Approximate its shape as a 600 pc long cylinder, with radius ~30 pc, and tangential velocity radius 2.5km/s.
N_psceri = 858
vol_psceri = 600*3.141*(30**2)*3.141*2.5**2
n_psceri = N_psceri/vol_psceri

print(f"n_rsg5/n_ch2: {n_rsg5/n_ch2:.2f}")
print(f"n_rsg5/n_psceri: {n_rsg5/n_psceri:.2f}")
print(f"n_ch2/n_psceri: {n_ch2/n_psceri:.2f}")
