# for delta Lyr cluster

AVG_EBmV = 0.032 # pm 0.006

from earhart.extinction import EBmV_to_AV, AV_to_EBpmRp

AVG_EBpmRp = AV_to_EBpmRp(EBmV_to_AV(AVG_EBmV))

AVG_AG = -99

AVG_EGmRp = -99
