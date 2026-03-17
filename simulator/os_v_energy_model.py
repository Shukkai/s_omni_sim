import math
import numpy as np
from scipy.interpolate import interp1d


def os_v_compute_energy(array_m: int, array_n: int, M, K, N, batch_size, qbit):
    energy = 0.0
    k_eff = math.ceil(K / 4)
    m_tiles = math.ceil(M / array_m)
    n_tiles = math.ceil(N / (array_n * 32)) 
    if array_m == 32 and array_n == 4:
        if M == 1:
            JOULE_M16 = 75.978e-9
            JOULE_M64 = 271.2e-9
            JOULE_M128 = 514.3e-9
            JOULE_M256 = 1000.716e-9
            x = np.array([16, 64, 128, 256], dtype=float)
            y = np.array([JOULE_M16, JOULE_M64, JOULE_M128, JOULE_M256], dtype=float)
            f = interp1d(x, y, kind='linear', fill_value='extrapolate')
            energy_per_tile = f(k_eff)
            
            energy = energy_per_tile * n_tiles / array_m * qbit * batch_size
        else:
            JOULE_M16 = 61.596e-9
            JOULE_M64 = 193.132e-9
            JOULE_M128 = 364.14e-9
            JOULE_M256 = 693.744e-9
            x = np.array([16, 64, 128, 256], dtype=float)
            y = np.array([JOULE_M16, JOULE_M64, JOULE_M128, JOULE_M256], dtype=float)
            f = interp1d(x, y, kind='linear', fill_value='extrapolate')
            energy_per_tile = f(k_eff)
            
            energy = energy_per_tile * m_tiles * n_tiles * qbit * batch_size
        
    return energy
    
    