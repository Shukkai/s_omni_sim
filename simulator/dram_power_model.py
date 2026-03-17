
READ_ROW  = 0.00000004508   # J per row
ACT       = 0.000000000872  # J per activation
READ_COLS = 0.0000000003935 # J per burst (read)
WRITE_COLS= 0.000000000345  # J per burst (write)

def dram_energy(data_byte: int, is_write: bool = False) -> float:
    """
    Estimate DRAM access energy (Joules) for a given data size.
    
    Args:
        data_byte (int): Total data size in bytes to read or write.
        is_write (bool): True for write, False for read.

    Returns:
        float: Estimated total DRAM energy in Joules.
    """
    # Each row is 1024 bytes
    num_rows = data_byte // 1024
    remain_cols = data_byte % 1024

    # Each burst = 8 bytes
    burst_length = remain_cols // 8 + (1 if remain_cols % 8 else 0)

    if is_write:
        energy = WRITE_COLS * burst_length
    else:
        energy = READ_COLS * burst_length

    # Add row activation + per-row overhead
    total_energy = READ_ROW * num_rows + ACT + energy

    return total_energy


if __name__ == "__main__":
    read_energy  = dram_energy(745, is_write=False)
    write_energy = dram_energy(745, is_write=True)
    print(f"Read  4KB -> {read_energy:.3e} J")
    print(f"Write 4KB -> {write_energy:.3e} J")