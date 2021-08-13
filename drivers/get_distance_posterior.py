from cdips.utils.gaiaqueries import parallax_to_distance_highsn

d,up,lo=parallax_to_distance_highsn(3.009,e_parallax_mas=0.032)
print(d,up,lo)
