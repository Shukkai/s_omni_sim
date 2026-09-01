"""
On-chip buffer presets -- the SRAM the RTL actually builds.

`sram_capacity_kb` models on-chip memory as **one pool** that any operand may
draw from, and `sram_port_model = "ported"` (§19) splits its *bandwidth* three
ways while leaving its *capacity* undivided.  The RTL does neither.  The
Omni-LUT system block diagram builds four physically separate SRAMs with fixed
sizes, fixed word widths, and no way to trade capacity between them:

    input buffer    1024 x 2048 b  x1  =   256 KB    word 256 B
    scale buffer    1024 x 2048 b  x1  =   256 KB    word 256 B
    weight buffer   1024 x 2048 b  x8  = 2,048 KB    word 256 B
    output buffer   1024 x 4096 b  x1  =   512 KB    word 512 B
                                         ---------
                                           3,072 KB = 3.00 MB

**The word widths are the array geometry, and that is the check that matters.**

    input word   2048 b / act_bits 16      = 128 elements = array_m x MU
    output word  4096 b / accum_bits 32    = 128 elements = array_n x NUM_RAC

So one input-buffer word is exactly one cycle of activation operand -- 256 B,
or 128 GB/s at 500 MHz -- and one output word is exactly one column tile of
accumulators.  §19 measured the activation port at 255.7 B/cycle against a
predicted 256 and concluded that 128 GB/s had always been an *activation-port*
number rather than an aggregate.  The RTL says so outright: the buffer is built
one cycle wide.  `preflight()` in `analysis/memory/buffers_run.py` asserts both
identities rather than trusting this comment.

**What the partition changes that the pool could not express.**

  * **Input and output are separate memories at different widths.**  §19 put
    A-reads and result-writes on one "unified" port at one bandwidth; they are
    two ports, and the output port is 2x wider (512 B/cycle).
  * **The scale buffer has no counterpart in the model at all.**  It is as large
    as the input buffer and carries a first-class operand -- the RTL gives it
    its own load command (`CMD_scale_size`, `CMD_scale_base_addr`) and its own
    DRAM type code (`2'd1`).  `hw.model_scale_traffic` adds the term.
  * **Capacity cannot be traded.**  A batch that needs more activation space
    cannot borrow from the 2 MB of weight buffer, which is what §8's
    "batch buys capacity" result implicitly assumed.

**Two consequences that fall straight out of the numbers**, both asserted in
the sweep rather than argued here:

  * The input buffer holds `256 KB / (d_model x act_bits/8)` = **32 activation
    rows** for a 4096-wide model -- exactly `array_m`.  The machine *is*
    `sram_m_tile = 32`, so §20's nomination of a 512-row block describes a
    buffer that was never built.
  * One 32K-context Key cache at 4 bits is `32768 x 128 x 4 b` = **2,048 KB**,
    which is the weight buffer exactly.  K alone fills it at 32K and K+V needs
    twice that, so on-chip KV residency (§11) tops out near 16K on this part.

**What these presets are not.**  Capacity and word width only.  There is no
port count per buffer (a 2048-bit word is assumed to be one access), no bank
conflict model within the 8 weight banks, and no LSU/DMA latency -- the RTL's
own 492 -> 927 ns measurement says the load path is *not* hidden behind compute,
which is evidence for `overlap_model = "serial"` (§17) but is one micro-benchmark
and not enough to build a latency term from.
"""

import dataclasses
from typing import Dict


@dataclasses.dataclass(frozen=True)
class BufferSpec:
    """One on-chip SRAM: depth x width, possibly banked."""
    depth: int
    width_bits: int
    banks: int = 1

    @property
    def bytes(self) -> int:
        """Total capacity across banks."""
        return self.depth * self.width_bits // 8 * self.banks

    @property
    def kb(self) -> float:
        return self.bytes / 1024

    @property
    def word_bytes(self) -> int:
        """Bytes moved per access -- one word, which is one cycle's operand."""
        return self.width_bits // 8


@dataclasses.dataclass(frozen=True)
class BufferConfig:
    """The four on-chip memories of the Omni-LUT system, together."""
    name: str
    input: BufferSpec
    scale: BufferSpec
    weight: BufferSpec
    output: BufferSpec
    derivation: str

    @property
    def total_bytes(self) -> int:
        return (self.input.bytes + self.scale.bytes
                + self.weight.bytes + self.output.bytes)


BUFFER_CONFIGS: Dict[str, BufferConfig] = {
    # The system block diagram, transcribed.  Depth 1024 is corroborated by the
    # DRAM address map in the same note, whose address field is `[9:0]` -- ten
    # bits, one word per address, 1024 words.
    'OMNI_LUT_RTL': BufferConfig(
        'OMNI_LUT_RTL',
        input=BufferSpec(1024, 2048, 1),
        scale=BufferSpec(1024, 2048, 1),
        weight=BufferSpec(1024, 2048, 8),
        output=BufferSpec(1024, 4096, 1),
        derivation=('input/scale 1024x2048b, weight 1024x2048b x8 banks, '
                    'output 1024x4096b; 3.00 MB total'),
    ),
}

#: The RTL the simulator is meant to describe.
DEFAULT_BUFFER_CONFIG = 'OMNI_LUT_RTL'


def buffer_config(name: str) -> BufferConfig:
    """Look up a preset, with the valid names in the error."""
    try:
        return BUFFER_CONFIGS[name]
    except KeyError:
        raise KeyError(
            f"unknown buffer config {name!r}; "
            f"known: {', '.join(sorted(BUFFER_CONFIGS))}") from None


def with_buffer_config(hw, name: str, enforce: bool = True,
                       scale_traffic: bool = True):
    """Return a copy of `hw` wired to one buffer configuration.

    Sets every per-buffer capacity and width together, because they are one
    part -- the same argument `with_memory_technology` makes for bandwidth and
    burst.  `hw` is not mutated.

    Args:
        enforce:       set `sram_buffer_model = "partitioned"`, which is what
                       makes the capacities bind and the ports separate.
                       False loads the geometry without acting on it, which is
                       useful for reporting what a config *would* demand.
        scale_traffic: also switch on `model_scale_traffic`.  Separate because
                       the scale operand is a *traffic* term that moves DRAM
                       bytes, while everything else here is capacity and width.
    """
    cfg = buffer_config(name)
    return dataclasses.replace(
        hw,
        sram_buffer_model="partitioned" if enforce else "pool",
        model_scale_traffic=bool(scale_traffic),
        input_buffer_bytes=cfg.input.bytes,
        input_buffer_word_bytes=cfg.input.word_bytes,
        scale_buffer_bytes=cfg.scale.bytes,
        scale_buffer_word_bytes=cfg.scale.word_bytes,
        weight_buffer_bytes=cfg.weight.bytes,
        weight_buffer_word_bytes=cfg.weight.word_bytes,
        weight_buffer_banks=cfg.weight.banks,
        output_buffer_bytes=cfg.output.bytes,
        output_buffer_word_bytes=cfg.output.word_bytes,
    )
