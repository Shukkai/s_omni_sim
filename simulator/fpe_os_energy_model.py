import math



def fpe_os_compute_energy(fpe_array_size: int, M, K, N, batch_size):
    energy = 0.0
    m_tiles = math.ceil(M / fpe_array_size)
    n_tiles = math.ceil(N / fpe_array_size)
    if fpe_array_size == 64:
        if M == 1:
            JOULE_K16 = 47.9e-9
            JOULE_K64 = 108.5e-9
            JOULE_K128 = 190.4e-9
            JOULE_K256 = 356.2e-9
            if K < 16:
                energy_per_tile = (K - 16) / (64 - 16) * (JOULE_K64 - JOULE_K16) + JOULE_K16
            elif K < 64:
                energy_per_tile = (K - 16) / (64 - 16) * (JOULE_K64 - JOULE_K16) + JOULE_K16
            elif K < 128:
                energy_per_tile = (K - 64) / (128 - 64) * (JOULE_K128 - JOULE_K64) + JOULE_K64
            elif K < 256:
                energy_per_tile = (K - 128) / (256 - 128) * (JOULE_K256 - JOULE_K128) + JOULE_K128
            else:
                energy_per_tile = (K - 256) / (256 - 128) * (JOULE_K256 - JOULE_K128) + JOULE_K256
        else:
        
            JOULE_K16 = 134.088e-9
            JOULE_K64 = 334.718e-9
            JOULE_K128 = 597.536e-9
            JOULE_K256 = 1104.966e-9
            if K < 16:
                energy_per_tile = (K - 16) / (64 - 16) * (JOULE_K64 - JOULE_K16) + JOULE_K16
            elif K < 64:
                energy_per_tile = (K - 16) / (64 - 16) * (JOULE_K64 - JOULE_K16) + JOULE_K16
            elif K < 128:
                energy_per_tile = (K - 64) / (128 - 64) * (JOULE_K128 - JOULE_K64) + JOULE_K64
            elif K < 256:
                energy_per_tile = (K - 128) / (256 - 128) * (JOULE_K256 - JOULE_K128) + JOULE_K128
            else:
                energy_per_tile = (K - 256) / (256 - 128) * (JOULE_K256 - JOULE_K128) + JOULE_K256 
        
        energy = energy_per_tile * m_tiles * n_tiles * batch_size
        
    return energy
    
    