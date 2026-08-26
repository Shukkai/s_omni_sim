import math
import numpy as np
from scipy.interpolate import interp1d

def omni_compute_energy(array_m: int, array_n: int, M, K, N, batch_size, qbit, mode):
    energy = 0.0
    if mode == "WS":
        k_eff = math.ceil(K / 4)
        k_tiles = math.ceil(k_eff / array_m)
        n_tiles = math.ceil(N / (array_n * 32)) 
        if array_m == 32 and array_n == 4:
            JOULE_M32 = 127.9e-9
            JOULE_M64 = 240.8e-9
            JOULE_M128 = 464.8e-9
            JOULE_M256 = 908.3e-9
            if M < 32:
                energy_per_tile = (M - 32) / (64 - 32) * (JOULE_M64 - JOULE_M32) + JOULE_M32
            elif M < 64:
                energy_per_tile = (M - 32) / (64 - 32) * (JOULE_M64 - JOULE_M32) + JOULE_M32
            elif M < 128:
                energy_per_tile = (M - 64) / (128 - 64) * (JOULE_M128 - JOULE_M64) + JOULE_M64
            elif M < 256:
                energy_per_tile = (M - 128) / (256 - 128) * (JOULE_M256 - JOULE_M128) + JOULE_M128
            else:
                energy_per_tile = (M - 256) / (256 - 128) * (JOULE_M256 - JOULE_M128) + JOULE_M256
                
            energy = energy_per_tile * k_tiles * n_tiles * qbit * batch_size
    elif mode == "OS":
        k_eff = math.ceil(K / 4)
        m_tiles = math.ceil(M / array_m)
        n_tiles = math.ceil(N / (array_n * 32)) 
        if array_m == 32 and array_n == 4:
            if M == 1:
                JOULE_K32 = 137.8e-9
                JOULE_K64 = 265.4e-9
                JOULE_K128 = 504.0e-9
                JOULE_K256 = 935.4e-9
            else:
                JOULE_K32 = 149.0e-9
                JOULE_K64 = 274.3e-9
                JOULE_K128 = 522.6e-9
                JOULE_K256 = 1006.6e-9
                
            x = np.array([32, 64, 128, 256], dtype=float)
            y = np.array([JOULE_K32, JOULE_K64, JOULE_K128, JOULE_K256], dtype=float)
            f = interp1d(x, y, kind='linear', fill_value='extrapolate')
            energy_per_tile = f(k_eff)
            # if K < 32:
            #     energy_per_tile = (K - 32) / (64 - 32) * (JOULE_K64 - JOULE_K32) + JOULE_K32
            # elif K < 64:
            #     energy_per_tile = (K - 32) / (64 - 32) * (JOULE_K64 - JOULE_K32) + JOULE_K32
            # elif K < 128:
            #     energy_per_tile = (K - 64) / (128 - 64) * (JOULE_K128 - JOULE_K64) + JOULE_K64
            # elif K < 256:
            #     energy_per_tile = (K - 128) / (256 - 128) * (JOULE_K256 - JOULE_K128) + JOULE_K128
            # else:
            #     energy_per_tile = (K - 256) / (256 - 128) * (JOULE_K256 - JOULE_K128) + JOULE_K256
                
            energy = energy_per_tile * m_tiles * n_tiles * qbit * batch_size
            if M == 1:
                energy = energy / array_m


    # interp1d returns np.float64; coerce so OperationMetrics.compute_energy is
    # always a plain float.  Numerically inert -- float(np.float64(x)) == x --
    # but it stops numpy scalars leaking into every downstream consumer, where
    # they serialize and compare differently from the other energy models.
    return float(energy)
    
    