
### - Modulo com funcoes de normalização diversas ----###

# Bibliotecas necessarias:
import numpy as np

def minmax(v,vmin,vmax):  # v pode ser um array ou semente um numero!
    ##############################
    vnorm = []
    if np.size(v) == 1:
        vnorm.append( (v - vmin)/(vmax - vmin) )
    else:
        for i in range( np.size(v) ):
            vnorm.append((v[i] - vmin)/(vmax - vmin))
    
    return np.array(vnorm)
