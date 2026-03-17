READ_C  = 7.431 # uA/MHz per 128 bits io
WRITE_C = 8.742 # uA/MHz per 128 bits io
VDD     = 0.75  # V

JOULE_PER_READ_128  = READ_C  * VDD * 1e-12
JOULE_PER_WRITE_128 = WRITE_C * VDD * 1e-12

def sram_energy(data_byte: int, is_write: bool = False) -> float:
    """
    Estimate sram access energy (Joules) for a given data size.
    
    Args:
        data_byte (int): Total data size in bytes to read or write.
        is_write (bool): True for write, False for read.

    Returns:
        float: Estimated total SRAM energy in Joules.
    """
    num_operation = data_byte // 16 + (1 if data_byte % 16 else 0)

    if is_write:
        energy = JOULE_PER_WRITE_128 * num_operation
    else:
        energy = JOULE_PER_READ_128 * num_operation

    return energy