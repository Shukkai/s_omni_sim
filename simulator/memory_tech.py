"""
DRAM technology presets -- bandwidth *and* access granularity together.

The simulator carries `dram_bandwidth_gbps` and `dram_burst_bytes` as two
independent knobs, which lets a study set a combination no real part has.  They
are not independent in hardware: a memory technology fixes both at once, and
`study2.md` section 10 showed the burst is the term that decides which KV
pruning axis is allowed to work at all.  Sweeping bandwidth while holding the
burst at a DDR5 value therefore answers a question about no real system.

Each preset below is **one channel / one stack at nominal peak**, with the
derivation written out so it can be checked rather than trusted:

    DDR5-6400      6400 MT/s x 64-bit bus            = 51.2 GB/s
                   BL16 on a 32-bit subchannel       = 64 B
    DDR5-6400 x2   two channels                      = 102.4 GB/s
    LPDDR5X-8533   8533 MT/s x 64-bit                = 68.3 GB/s
                   BL32 on a 16-bit channel          = 64 B
    GDDR6-16       16 Gbps x 32-bit device           = 64.0 GB/s
                   BL16 on a 32-bit channel          = 64 B
    HBM2E          3.2 Gbps x 1024-bit stack         = 409.6 GB/s
                   BL4 on a 64-bit pseudo-channel    = 32 B
    HBM3           6.4 Gbps x 1024-bit stack         = 819.2 GB/s
                   BL8 on a 32-bit pseudo-channel    = 32 B

**The 32 B burst is the interesting column, not the bandwidth.**  HBM halves the
access granularity, so a 64 B KV entry becomes *two* bursts instead of one --
which means a channel mask retaining 64 of 128 channels is burst-aligned on HBM
and is not on DDR5.  The memory technology, not the pruning algorithm, decides
whether the saving is collectable.

**What these presets are not.**  Nominal peak, so no refresh, no bank conflicts,
no read/write turnaround and no controller efficiency factor -- real sustained
bandwidth is typically 70-85% of these.  The simulator's roofline has no latency
or queueing term either (see `ram_sim_plan.md`), so a preset changes *how many
bytes are charged and how fast they stream*, nothing more.  Energy per bit also
differs sharply across these technologies (HBM is roughly 3-5 pJ/bit against
DDR5's 8-15) and is deliberately **not** wired in here, because the energy model
carries its own constants and changing them would move every published energy
number for reasons unrelated to this file.
"""

import dataclasses
from typing import Dict


@dataclasses.dataclass(frozen=True)
class MemoryTechnology:
    """One memory part at nominal peak, per channel or per stack."""
    name: str
    bandwidth_gbps: float
    burst_bytes: int
    derivation: str


MEMORY_TECHNOLOGIES: Dict[str, MemoryTechnology] = {
    'DDR5-6400': MemoryTechnology(
        'DDR5-6400', 51.2, 64,
        '6400 MT/s x 64-bit = 51.2 GB/s; BL16 x 32-bit subchannel = 64 B'),
    'DDR5-6400-x2': MemoryTechnology(
        'DDR5-6400-x2', 102.4, 64,
        'two DDR5-6400 channels'),
    'LPDDR5X-8533': MemoryTechnology(
        'LPDDR5X-8533', 68.3, 64,
        '8533 MT/s x 64-bit = 68.3 GB/s; BL32 x 16-bit channel = 64 B'),
    'GDDR6-16': MemoryTechnology(
        'GDDR6-16', 64.0, 64,
        '16 Gbps x 32-bit device = 64 GB/s; BL16 x 32-bit = 64 B'),
    'HBM2E': MemoryTechnology(
        'HBM2E', 409.6, 32,
        '3.2 Gbps x 1024-bit stack = 409.6 GB/s; BL4 x 64-bit pseudo-ch = 32 B'),
    'HBM3': MemoryTechnology(
        'HBM3', 819.2, 32,
        '6.4 Gbps x 1024-bit stack = 819.2 GB/s; BL8 x 32-bit pseudo-ch = 32 B'),
}

#: The preset the simulator's own defaults already describe.  Stated as a
#: constant so a study can assert it rather than assume it.
DEFAULT_TECHNOLOGY = 'DDR5-6400'


def memory_technology(name: str) -> MemoryTechnology:
    """Look up a preset, with the valid names in the error."""
    try:
        return MEMORY_TECHNOLOGIES[name]
    except KeyError:
        raise KeyError(
            f"unknown memory technology {name!r}; "
            f"known: {', '.join(sorted(MEMORY_TECHNOLOGIES))}") from None


def with_memory_technology(hw, name: str):
    """Return a copy of `hw` configured for one memory technology.

    Sets bandwidth and burst together, which is the whole point -- see the
    module docstring.  `hw` is not mutated, so a sweep can hold one base config
    and derive from it.
    """
    tech = memory_technology(name)
    return dataclasses.replace(
        hw,
        dram_bandwidth_gbps=tech.bandwidth_gbps,
        dram_burst_bytes=tech.burst_bytes,
    )
