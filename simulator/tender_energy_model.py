import math
import numpy as np
from scipy.interpolate import interp1d

def tender_compute_energy(fpe_array_size: int, M, K, N, batch_size):
    energy = 0.0
    m_tiles = math.ceil(M / fpe_array_size)
    n_tiles = math.ceil(N / fpe_array_size)
    if fpe_array_size == 64:
        if M == 1:
            JOULE_K64 = 65.184e-9
            JOULE_K128 = 90.816e-9
            JOULE_K256 = 142.048e-9
            
            x = np.array([64, 128, 256], dtype=float)
            y = np.array([JOULE_K64, JOULE_K128, JOULE_K256], dtype=float)
        else:
            JOULE_K16 = 61.91e-9
            JOULE_K64 = 123.84e-9
            JOULE_K128 = 208.292e-9
            JOULE_K256 = 374.916e-9
        
            x = np.array([16, 64, 128, 256], dtype=float)
            y = np.array([JOULE_K16, JOULE_K64, JOULE_K128, JOULE_K256], dtype=float)
        f = interp1d(x, y, kind='linear', fill_value='extrapolate')
        energy_per_tile = f(K)
        
        # if K < 16:
        #     energy_per_tile = (K - 16) / (64 - 16) * (JOULE_K64 - JOULE_K16) + JOULE_K16
        # elif K < 64:
        #     energy_per_tile = (K - 16) / (64 - 16) * (JOULE_K64 - JOULE_K16) + JOULE_K16
        # elif K < 128:
        #     energy_per_tile = (K - 64) / (128 - 64) * (JOULE_K128 - JOULE_K64) + JOULE_K64
        # elif K < 256:
        #     energy_per_tile = (K - 128) / (256 - 128) * (JOULE_K256 - JOULE_K128) + JOULE_K128
        # else:
        #     energy_per_tile = (K - 256) / (256 - 128) * (JOULE_K256 - JOULE_K128) + JOULE_K256 
        
        energy = energy_per_tile * m_tiles * n_tiles * batch_size
        
    return energy
    
    