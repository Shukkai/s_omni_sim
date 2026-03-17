import math
import numpy as np
from scipy.interpolate import interp1d


def ws_compute_energy(array_m: int, array_n: int, M, K, N, batch_size, qbit):
    energy = 0.0
    k_eff = math.ceil(K / 4)
    k_tiles = math.ceil(k_eff / array_m)
    n_tiles = math.ceil(N / (array_n * 32)) 
    if array_m == 16 and array_n == 2:
        JOULE_M16 = 19.76e-9
        JOULE_M64 = 69.872e-9
        JOULE_M128 = 135.584e-9
        JOULE_M256 = 271.04e-9
        x = np.array([16, 64, 128, 256], dtype=float)
        y = np.array([JOULE_M16, JOULE_M64, JOULE_M128, JOULE_M256], dtype=float)
        f = interp1d(x, y, kind='linear', fill_value='extrapolate')
        energy_per_tile = f(M)

        # if M < 16:
        #     energy_per_tile = (M - 16) / (64 - 16) * (JOULE_M64 - JOULE_M16) + JOULE_M16
        # elif M < 64:
        #     energy_per_tile = (M - 16) / (64 - 16) * (JOULE_M64 - JOULE_M16) + JOULE_M16
        # elif M < 128:
        #     energy_per_tile = (M - 64) / (128 - 64) * (JOULE_M128 - JOULE_M64) + JOULE_M64
        # elif M < 256:
        #     energy_per_tile = (M - 128) / (256 - 128) * (JOULE_M256 - JOULE_M128) + JOULE_M128
        # else:
        #     energy_per_tile = (M - 256) / (256 - 128) * (JOULE_M256 - JOULE_M128) + JOULE_M256
            
        energy = energy_per_tile * k_tiles * n_tiles * qbit * batch_size
    elif array_m == 32 and array_n == 4:
        JOULE_M16 = 57.652e-9
        JOULE_M64 = 190.8e-9
        JOULE_M128 = 363.8e-9
        JOULE_M256 = 714.604e-9
        if M < 16:
            energy_per_tile = (M - 16) / (64 - 16) * (JOULE_M64 - JOULE_M16) + JOULE_M16
        elif M < 64:
            energy_per_tile = (M - 16) / (64 - 16) * (JOULE_M64 - JOULE_M16) + JOULE_M16
        elif M < 128:
            energy_per_tile = (M - 64) / (128 - 64) * (JOULE_M128 - JOULE_M64) + JOULE_M64
        elif M < 256:
            energy_per_tile = (M - 128) / (256 - 128) * (JOULE_M256 - JOULE_M128) + JOULE_M128
        else:
            energy_per_tile = (M - 256) / (256 - 128) * (JOULE_M256 - JOULE_M128) + JOULE_M256
            
        energy = energy_per_tile * k_tiles * n_tiles * qbit * batch_size
    return energy
    
    