"""
Hardware Simulator for LLM Inference

Supports multiple compute dataflow modes (LUT_OS, LUT_WS, FPE_OS, OMNI,
TENDER, VPU) with optional FlashAttention-style tiled attention computation.

Key features:
  - Prefill & decode phase simulation for transformer models
  - Per-operation cycle, memory, energy, and utilization tracking
  - FlashAttention tiled attention (configurable block size)
  - Peak SRAM footprint estimation

Usage:
    python simulator.py
"""

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from enum import Enum

import dram_power_model
import sram_power_model
import ws_energy_model
import fpe_os_energy_model
import omni_energy_model
import tender_energy_model
import os_energy_model
import os_v_energy_model
import vpu_energy_model


# ============================================================================
# Enums
# ============================================================================

class Phase(Enum):
    PREFILL = "prefill"
    DECODE = "decode"


class OperationType(Enum):
    Q_PROJ = "q_proj"
    K_PROJ = "k_proj"
    V_PROJ = "v_proj"
    QK_MATMUL = "qk_matmul"
    ATTN_V_MATMUL = "attn_v_matmul"
    FLASH_ATTN = "flash_attn"      # Fused QK + softmax + AttnV
    O_PROJ = "o_proj"
    GATE = "gate"                  # MoE router/gating network
    FC1 = "fc1"
    FC2 = "fc2"


class NonGEMMOpType(Enum):
    """Non-GEMM operations in LLM inference pipeline."""
    RMSNORM_PRE_ATTN = "rmsnorm_pre_attn"      # RMSNorm before attention
    RMSNORM_PRE_FFN = "rmsnorm_pre_ffn"        # RMSNorm before FFN
    RMSNORM_FINAL = "rmsnorm_final"            # Final RMSNorm (last layer only)
    ROPE = "rope"                              # Rotary Position Embedding on Q & K
    ATTN_SCALE = "attn_scale"                  # Q scaling by 1/sqrt(d_k)
    SOFTMAX = "softmax"                        # Attention softmax
    RESIDUAL_ATTN = "residual_attn"            # Residual add after attention
    RESIDUAL_FFN = "residual_ffn"              # Residual add after FFN
    SILU = "silu"                              # SiLU activation in FFN
    GELU = "gelu"                              # GELU activation (alternative)
    GATED_MUL = "gated_mul"                    # Element-wise mul for gated FFN (SwiGLU)
    EMBEDDING = "embedding"                     # Token embedding lookup
    LM_HEAD_SOFTMAX = "lm_head_softmax"        # Final softmax for next token prediction


class ComputeMode(Enum):
    AA = "AA"   # Activation × Activation
    AW = "AW"   # Activation × Weight


# ============================================================================
# Configuration Dataclasses
# ============================================================================

@dataclass
class ModelConfig:
    """Transformer model architecture parameters."""
    num_layers: int
    num_heads: int
    num_kv_heads: int   # For GQA: < num_heads; for MHA: same as num_heads
    d_model: int
    d_ffn: int          # Per-expert FFN hidden dimension
    head_dim: int       # d_model // num_heads

    # MoE parameters (dense model: both = 1)
    num_experts: int = 1            # Total number of experts
    num_active_experts: int = 1     # Top-K experts selected per token

    @property
    def d_kv(self) -> int:
        """KV projection output dimension (= d_model for MHA, smaller for GQA/MQA)."""
        return self.num_kv_heads * self.head_dim

    @property
    def is_moe(self) -> bool:
        return self.num_experts > 1


@dataclass
class WorkloadConfig:
    """Inference workload parameters."""
    batch_size: int
    input_tokens: int       # Prefill sequence length
    output_tokens: int      # Number of tokens to generate
    flash_block_size: int = 0   # 0 = standard attention; >0 = FlashAttention tile size (Br = Bc)


@dataclass
class HardwareConfig:
    """Hardware accelerator configuration."""
    # --- Array dimensions ---
    array_m: int
    array_n: int
    replication: int = 1    # LUT spatial replication factor
    mu: int = 4             # LUT multi-use factor (elements per cycle)
    RAC: int = 32           # Number of RAC units
    FPE_array_size: int = 64

    # --- VPU for non-GEMM operations ---
    vpu_width: int = 128    # VPU SIMD width (elements per cycle)

    # --- Clock & bandwidth ---
    freq_mhz: int = 500
    dram_bandwidth_gbps: float = 51.2   # DRAM bandwidth in GB/s

    # On-chip SRAM read+write bandwidth in GB/s.  0 = unlimited, the default,
    # which is how every result predating this field was produced: capacity was
    # enforced (`sram_capacity_kb`) but throughput never was, so an operation
    # could move unbounded bytes per cycle to and from SRAM for free.
    #
    # The array geometry implies roughly `MU x array_n x NUM_RAC x kv_bits`
    # = 4 x 4 x 32 x 4 b = 256 B/cycle, or 128 GB/s at 500 MHz -- but that is
    # *one operand port*, while `sram_read` here is a lump of A-reads, B-reads
    # and C accumulator traffic.  Charging the lump against one port's number
    # over-charges, sometimes by a lot, which is exactly why this ships inert
    # and the sweep that sets it reports what it does to every headline number
    # (`analysis/memory/bandwidth_run.py`).
    sram_bandwidth_gbps: float = 0.0

    # How `sram_bandwidth_gbps` is charged against the on-chip memories.
    #
    #   "lumped"  one number for everything: (sram_read + sram_write) / bw.
    #             Every result predating this field was produced this way, so
    #             it is the default.
    #   "ported"  three independent memories, each moving bytes concurrently,
    #             and the operation waits for the slowest:
    #             max(activation, weight, accumulator).
    #
    # **"ported" is the structure OMNI_LUT.pdf Fig. 4 actually draws.**  The
    # system diagram shows a *Unified Buffer*, a separate *Weight Buffer*, and
    # a separate *Accumulator* block hanging off the PE array -- three
    # memories, not one.  "lumped" pushes all three through a single port,
    # which is why SS16(c) had to park prefill: it charged the accumulator's
    # partial-sum recirculation against the activation port's bandwidth.
    #
    # Under "ported", `sram_bandwidth_gbps` is *per port* rather than
    # aggregate.  That is the reading the geometry supports: the array
    # consumes `array_m x MU x act_bits/8` = 256 B/cycle of activations, or
    # 128 GB/s at 500 MHz, and 128 GB/s was always an activation-port number.
    sram_port_model: str = "lumped"

    # Accumulator bandwidth in GB/s, used only under `sram_port_model =
    # "ported"`.  0 = unlimited, the default.
    #
    # Unlimited is the *physical* default here, not merely the inert one: the
    # accumulator in Fig. 4 is a datapath block wired directly to the PE
    # array's partial-sum outputs, not a client of the unified buffer, so its
    # traffic does not cross an SRAM port at all.  Set a finite number only to
    # ask what it would cost if it did.
    accum_bandwidth_gbps: float = 0.0

    # How much an operation's memory time can hide behind another's compute.
    #
    #   "serial"     sum(max(compute, memory)) over operations -- no overlap at
    #                all.  Every published latency number was produced this way,
    #                so it is the default.
    #   "pipelined"  max(sum(compute), sum(memory)) over a phase -- the opposite
    #                extreme, where a deep enough software pipeline hides one
    #                entirely behind the other.
    #
    # Neither is the truth; they **bracket** it.  Real hardware double-buffers,
    # so op i+1's operands prefetch during op i's compute, and a long chain of
    # such ops tends toward the pipelined bound -- but only with enough SRAM to
    # hold two operand sets, which `sram_capacity_kb` would have to allow.
    #
    # The gap is not small.  It is widest where a phase alternates between
    # memory-bound and compute-bound operations, which is exactly what decode
    # does (AW projections are memory-bound, `attn_v` is compute-bound), so
    # "serial" overstates decode TPOT by up to 1.75x -- more than most of the KV
    # techniques in this repo were measured to *save*.
    overlap_model: str = "serial"

    # How `LUT_OS_V` counts output-stationary rounds.
    #
    #   "tiled"   rounds = ceil(M/array_m) * n_tiles, except at M == 1 which
    #             has its own branch.  This is the original model and every
    #             published number was produced with it, so it is the default.
    #   "packed"  rounds = ceil(M * n_tiles / array_m), the accumulator-budget
    #             form.
    #
    # **The two disagree by `array_m / M` for 2 <= M < array_m, and the
    # original is the one that is wrong there.**  The array holds
    # `array_m x (array_n x NUM_RAC)` accumulators, so a round can retire
    # `array_m` accumulator tiles wherever they come from -- different output
    # rows, different column tiles, or a mix.  "tiled" rounds `M` up to a whole
    # 32-row tile *before* multiplying by `n_tiles`, so it never packs column
    # tiles across rows unless `M >= array_m`, and charges a full 32-row pass
    # for M=2.
    #
    # The `M == 1` branch already does the right thing, and that is the tell:
    # `ceil(1 x n_tiles / array_m)` *is* `ceil(n_tiles / array_m)`, so the
    # special case is not special -- it is the only place the general formula
    # was written down.  "packed" restores it for every M.
    #
    # Measured effect: decode `q_proj` cycles go 32.96x from batch 1 to batch 2
    # for a 2x workload under "tiled", then sit flat from batch 2 to batch 32 --
    # the model charges the same cycles for 2 sequences as for 32.  That
    # discontinuity is what makes decode look compute-bound from batch 2
    # (`analysis/memory/regime_run.py`), so it gates the regime map.
    #
    # **Scope: `LUT_OS_V` only.**  `LUT_OS` carries the same accumulator
    # argument but not the same evidence -- OMNI_LUT.pdf SS IV-C/IV-D documents
    # the row-broadcast for the OS-V M=1 case specifically, and decode AW ops
    # resolve to `LUT_OS_V`, which is where the defect was measured.  Widening
    # it to `LUT_OS` would move prefill numbers on an argument from first
    # principles alone.
    os_rounds_model: str = "tiled"

    # --- On-chip memory ---
    # 0 = unlimited: peak_sram_bytes is reported but never enforced, which is
    # how every result predating this field was produced.  Set a real capacity
    # to make an operation whose working set does not fit pay for the spill.
    sram_capacity_kb: int = 0

    # How many batch elements are resident at once during *attention*.
    #
    #   "sequential"  one (batch, head) instance at a time -- the original
    #                 model, and the default, so existing results stand.
    #   "concurrent"  all batch elements co-resident, heads still sequential.
    #
    # Projections and FFN are unaffected by this switch: they are already
    # issued as a single GEMM with M = batch x seq_len, so batch is a
    # dimension there and their footprint already scales with it.  Attention
    # is issued once per (batch, head) with batch folded into `batch_size`,
    # which the footprint ignored -- so batch was a dimension in one half of
    # the model and a loop in the other.  "concurrent" makes the two agree,
    # which is what makes "how much batch fits" a well-posed question.
    sram_batch_model: str = "sequential"

    # --- DRAM access granularity ---
    # Minimum useful transfer size, in bytes.  A DRAM controller moves whole
    # bursts, so an access that touches `run` contiguous bytes actually costs
    # `ceil(run / burst) * burst`.  0 = disabled: bytes are charged exactly as
    # requested, which is how every result predating this field was produced.
    #
    # This is inert for contiguous traffic (a 16 KB weight block rounded to a
    # 32 B burst is still 16 KB) and only bites when the access pattern is
    # short-run -- a page-selective KV read, say -- which is exactly the case a
    # flat bandwidth number makes look ~100x cheaper than it is.
    dram_burst_bytes: int = 0

    # --- Prefill row tiling ---
    # How many rows of A (the activation matrix) the schedule holds on chip at
    # once.  0 = all `M` of them, which is what `_calculate_peak_sram` has
    # always assumed and how every result predating this field was produced.
    #
    # That assumption is the repo's one outright *bug*.  A prefill GEMM has
    # `M` = the whole sequence, so the claimed working set is O(seq x d_model)
    # -- 59 MB at 2K context and 2.1 GB at 32K -- which no plausible SRAM
    # fits.  Prefill therefore overflows at every capacity and its spill charge
    # is a meaningless constant (SS7).  Real hardware tiles the sequence.
    #
    # **This is a capacity bug and only a capacity bug**, which SS19 is what
    # settled: the SRAM *traffic* terms were suspected of the same defect, but
    # the activation term measures 255.7 B/cycle against an array that consumes
    # 256, so it is right and tiling cannot change it.
    #
    # What tiling costs, and where:
    #
    #   - `LUT_WS` pays.  Weight-stationary holds B across the whole `M`
    #     stream, so blocking the rows re-loads the weight tile once per block
    #     and re-pays the array's fill/drain once per block.  Both are charged.
    #   - The output-stationary and FPE modes pay **nothing**, because they
    #     were already tiled: their traffic terms re-read B once per
    #     `array_m` (or `FPE_array_size`) row tile, which only makes sense if
    #     one row tile of A is resident at a time.  For those modes the
    #     footprint was simply inconsistent with the loop nest the traffic
    #     model already described, and this field makes the two agree.
    #
    # A row block is a real loop nest -- produce a block of activations, run it
    # through every column and K tile, retire it -- not a re-tiling of the
    # kind `_apply_sram_capacity` declines to model.
    sram_m_tile: int = 0

    # --- On-chip KV buffer ---
    # Bytes of KV cache held on chip *between decode steps*, in KB.
    #
    # The KV cache is append-only: entries 1..n-1 are bit-identical at step t
    # and step t+1.  With no buffer the whole cache is re-read from DRAM every
    # token, which is what this model did unconditionally and what every decode
    # number predating this field assumes.  0 keeps that behaviour.
    #
    # This is a carve-out of `sram_capacity_kb`, not additional memory; keeping
    # the two consistent is the caller's job, and `capacity_run.py` reports the
    # working set a given configuration needs.
    kv_sram_kb: int = 0

    # --- Attention score staging ---
    # On-chip buffer for the attention score matrix, in KB.  0 = the scores are
    # written to DRAM by QK and read back by Attn.V unconditionally, which is
    # what every number predating this field assumes.
    #
    # The test is per (batch, head) instance and at ROW granularity: the scores
    # stay on chip iff one query row's score vector fits, `kv_len * act_bits/8`.
    # A whole-matrix test would be false at every plausible size (a 32K score
    # matrix is 2 GB per instance) and so would be inert exactly where the
    # traffic is.  A row block that fits maps onto a real loop nest -- produce
    # rows, softmax, consume in Attn.V -- and no score byte reaches DRAM.
    #
    # A carve-out of `sram_capacity_kb`, not additional memory.
    score_sram_kb: int = 0

    # Charge prefill attention for reading K/V from DRAM.  False reproduces the
    # original model, which gates the KV read on `is_decode` and so charges
    # prefill nothing -- it assumes K/V are resident while assuming the much
    # larger score matrix is not.  Only meaningful together with
    # `score_sram_kb`; see `_score_staged`.
    prefill_kv_dram_read: bool = False

    # --- Bit widths ---
    act_bits: int = 16
    accumulate_bits: int = 32
    weight_bits: int = 4
    kv_cache_bits: int = 16

    # --- Dataflow modes ---
    AW_mode: str = "LUT_WS"    # VPU, FPE_OS, LUT_OS, LUT_OS_V, LUT_WS, OMNI, TENDER
    AA_mode: str = "VPU"

    # --- Optimizations ---
    tender_m1_opt: bool = False   # M=1 drain optimization for TENDER (like FPE_OS)

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name)
                for f in self.__dataclass_fields__.values()}


# ============================================================================
# Metrics
# ============================================================================

@dataclass
class OperationMetrics:
    """Metrics collected for a single operation execution."""
    shape: Tuple[int, int, int]     # (M, K, N) for A(M×K) @ B(K×N)

    # Performance
    cycles: int = 0
    flops: int = 0
    utilization: float = 0.0
    throughput: float = 0.0

    # Memory access (bytes).  dram_read / dram_write are *logical* bytes: what
    # the operation asked for.  The _eff pair is what the DRAM actually moved
    # once each access is rounded up to a burst.  They are equal unless
    # hw.dram_burst_bytes is set, and both are kept so a report can show the
    # waste rather than silently changing what "dram_read" has always meant.
    dram_read: int = 0
    dram_write: int = 0
    dram_read_eff: int = 0
    dram_write_eff: int = 0
    sram_read: int = 0
    sram_write: int = 0

    # The same bytes, split by which on-chip memory moves them (Fig. 4:
    # Unified Buffer / Weight Buffer / Accumulator).  These partition
    # `sram_read` and `sram_write` exactly --
    #   sram_read  == sram_read_a + sram_read_b + sram_acc_read
    #   sram_write == sram_write_out + sram_acc_write
    # -- so the lumps keep meaning what they always meant and no existing
    # consumer moves.  All zero means "no split recorded" (non-GEMM/VPU ops),
    # which `_sram_ports` reads as pure activation traffic.
    sram_read_a: int = 0        # activation reads, Unified Buffer
    sram_read_b: int = 0        # weight / KV reads, Weight Buffer
    sram_acc_read: int = 0      # partial-sum reads, Accumulator
    sram_acc_write: int = 0     # partial-sum writes, Accumulator
    sram_write_out: int = 0     # final result writes, Unified Buffer

    # Energy (Joules)
    compute_energy: float = 0.0
    dram_read_energy: float = 0.0
    dram_write_energy: float = 0.0
    sram_read_energy: float = 0.0
    sram_write_energy: float = 0.0

    # SRAM capacity requirement (bytes)
    peak_sram_bytes: int = 0

    # KV bytes held on chip between decode steps, and so NOT re-read from DRAM
    # (only meaningful when hw.kv_sram_kb > 0)
    kv_resident_bytes: int = 0

    # Capacity enforcement (only meaningful when hw.sram_capacity_kb > 0)
    sram_overflow: bool = False      # working set exceeded capacity
    sram_refetch_bytes: int = 0      # extra DRAM reads charged by the spill policy
                                     # (already included in dram_read)

    # --- Derived properties ---
    @property
    def dram_energy(self) -> float:
        return self.dram_read_energy + self.dram_write_energy

    @property
    def sram_energy(self) -> float:
        return self.sram_read_energy + self.sram_write_energy

    @property
    def total_energy(self) -> float:
        return self.compute_energy + self.dram_energy + self.sram_energy


def _aggregate_metrics(metrics_list: List[OperationMetrics]) -> OperationMetrics:
    """Aggregate a list of OperationMetrics into a single summary."""
    total = OperationMetrics(shape=(0, 0, 0))
    weighted_util = 0.0
    max_peak_sram = 0

    for m in metrics_list:
        total.cycles += m.cycles
        total.flops += m.flops
        total.dram_read += m.dram_read
        total.dram_write += m.dram_write
        total.dram_read_eff += m.dram_read_eff
        total.dram_write_eff += m.dram_write_eff
        total.sram_read += m.sram_read
        total.sram_write += m.sram_write
        total.sram_read_a += m.sram_read_a
        total.sram_read_b += m.sram_read_b
        total.sram_acc_read += m.sram_acc_read
        total.sram_acc_write += m.sram_acc_write
        total.sram_write_out += m.sram_write_out
        total.compute_energy += m.compute_energy
        total.dram_read_energy += m.dram_read_energy
        total.dram_write_energy += m.dram_write_energy
        total.sram_read_energy += m.sram_read_energy
        total.sram_write_energy += m.sram_write_energy
        weighted_util += m.utilization * m.cycles
        max_peak_sram = max(max_peak_sram, m.peak_sram_bytes)
        total.sram_overflow |= m.sram_overflow
        total.sram_refetch_bytes += m.sram_refetch_bytes
        total.kv_resident_bytes += m.kv_resident_bytes

    total.utilization = weighted_util / total.cycles if total.cycles > 0 else 0.0
    total.peak_sram_bytes = max_peak_sram
    return total


@dataclass
class NonGEMMMetrics:
    """Metrics for a single non-GEMM operation."""
    op_type: NonGEMMOpType
    num_elements: int = 0       # Total elements processed
    
    # Performance
    cycles: int = 0             # VPU cycles
    
    # Element-wise operation counts (for energy calculation)
    # These count the number of primitive operations performed
    ops_add: int = 0
    ops_sub: int = 0
    ops_mul: int = 0
    ops_div: int = 0
    ops_exp: int = 0
    ops_rsqrt: int = 0
    ops_sqrt: int = 0
    ops_recip: int = 0
    ops_sin: int = 0
    ops_cos: int = 0
    ops_sigmoid: int = 0
    ops_reduce_sum: int = 0
    ops_reduce_max: int = 0
    ops_square: int = 0
    ops_neg: int = 0
    ops_max: int = 0            # Element-wise max (e.g., ReLU)
    
    # Memory (bytes) - for activation reads/writes
    sram_read: int = 0
    sram_write: int = 0
    
    # Energy (will be calculated using VPU energy model)
    compute_energy: float = 0.0
    sram_read_energy: float = 0.0
    sram_write_energy: float = 0.0
    
    @property
    def total_energy(self) -> float:
        return self.compute_energy + self.sram_read_energy + self.sram_write_energy
    
    @property 
    def sram_energy(self) -> float:
        return self.sram_read_energy + self.sram_write_energy
    
    def calculate_energy(self, vpu_config: 'vpu_energy_model.VPUEnergyConfig') -> float:
        """Calculate compute energy using VPU energy model."""
        energy = 0.0
        energy += self.ops_add * vpu_config.vec_add
        energy += self.ops_sub * vpu_config.vec_sub
        energy += self.ops_mul * vpu_config.vec_mul
        energy += self.ops_div * vpu_config.vec_div
        energy += self.ops_exp * vpu_config.vec_exp
        energy += self.ops_rsqrt * vpu_config.vec_rsqrt
        energy += self.ops_sqrt * vpu_config.vec_sqrt
        energy += self.ops_recip * vpu_config.vec_recip
        energy += self.ops_sin * vpu_config.vec_sin
        energy += self.ops_cos * vpu_config.vec_cos
        energy += self.ops_sigmoid * vpu_config.vec_sigmoid
        energy += self.ops_reduce_sum * vpu_config.vec_reduce_sum
        energy += self.ops_reduce_max * vpu_config.vec_reduce_max
        energy += self.ops_square * vpu_config.vec_square
        energy += self.ops_neg * vpu_config.vec_neg
        energy += self.ops_max * vpu_config.vec_max
        self.compute_energy = energy
        return energy


def _aggregate_non_gemm_metrics(metrics_list: List[NonGEMMMetrics]) -> dict:
    """Aggregate non-GEMM metrics by operation type."""
    totals = defaultdict(lambda: {
        'num_elements': 0, 'cycles': 0,
        'ops_add': 0, 'ops_sub': 0, 'ops_mul': 0, 'ops_div': 0,
        'ops_exp': 0, 'ops_rsqrt': 0, 'ops_sqrt': 0, 'ops_recip': 0,
        'ops_sin': 0, 'ops_cos': 0, 'ops_sigmoid': 0,
        'ops_reduce_sum': 0, 'ops_reduce_max': 0, 'ops_square': 0,
        'ops_neg': 0, 'ops_max': 0,
        'sram_read': 0, 'sram_write': 0,
        'compute_energy': 0.0, 'sram_read_energy': 0.0, 'sram_write_energy': 0.0,
        'count': 0
    })
    
    for m in metrics_list:
        t = totals[m.op_type]
        t['num_elements'] += m.num_elements
        t['cycles'] += m.cycles
        t['ops_add'] += m.ops_add
        t['ops_sub'] += m.ops_sub
        t['ops_mul'] += m.ops_mul
        t['ops_div'] += m.ops_div
        t['ops_exp'] += m.ops_exp
        t['ops_rsqrt'] += m.ops_rsqrt
        t['ops_sqrt'] += m.ops_sqrt
        t['ops_recip'] += m.ops_recip
        t['ops_sin'] += m.ops_sin
        t['ops_cos'] += m.ops_cos
        t['ops_sigmoid'] += m.ops_sigmoid
        t['ops_reduce_sum'] += m.ops_reduce_sum
        t['ops_reduce_max'] += m.ops_reduce_max
        t['ops_square'] += m.ops_square
        t['ops_neg'] += m.ops_neg
        t['ops_max'] += m.ops_max
        t['sram_read'] += m.sram_read
        t['sram_write'] += m.sram_write
        t['compute_energy'] += m.compute_energy
        t['sram_read_energy'] += m.sram_read_energy
        t['sram_write_energy'] += m.sram_write_energy
        t['count'] += 1
    
    return totals


@dataclass
class PhaseMetrics:
    """Metrics for a full phase (Prefill or Decode)."""
    phase: Phase
    aa_ops: Dict[OperationType, List[OperationMetrics]] = field(
        default_factory=lambda: defaultdict(list)
    )
    aw_ops: Dict[OperationType, List[OperationMetrics]] = field(
        default_factory=lambda: defaultdict(list)
    )
    non_gemm_ops: List[NonGEMMMetrics] = field(default_factory=list)

    def add_operation(self, op_type: OperationType, mode: ComputeMode,
                      metrics: OperationMetrics):
        target = self.aa_ops if mode == ComputeMode.AA else self.aw_ops
        target[op_type].append(metrics)
    
    def add_non_gemm_operation(self, metrics: NonGEMMMetrics):
        self.non_gemm_ops.append(metrics)

    # ---- Aggregation helpers ------------------------------------------------

    def _all_metrics(self) -> List[OperationMetrics]:
        out: List[OperationMetrics] = []
        for d in (self.aa_ops, self.aw_ops):
            for lst in d.values():
                out.extend(lst)
        return out

    def get_total_metrics(self) -> OperationMetrics:
        return _aggregate_metrics(self._all_metrics())

    def get_aa_total(self) -> OperationMetrics:
        return _aggregate_metrics([m for lst in self.aa_ops.values() for m in lst])

    def get_aw_total(self) -> OperationMetrics:
        return _aggregate_metrics([m for lst in self.aw_ops.values() for m in lst])

    def get_operation_total(self, op_type: OperationType,
                            mode: ComputeMode) -> OperationMetrics:
        d = self.aa_ops if mode == ComputeMode.AA else self.aw_ops
        return _aggregate_metrics(d.get(op_type, []))

    @property
    def peak_sram_bytes(self) -> int:
        """Maximum SRAM capacity required by any single operation."""
        return max((m.peak_sram_bytes for m in self._all_metrics()), default=0)


@dataclass
class SimulationResults:
    """Complete simulation results across prefill + decode."""
    prefill: PhaseMetrics
    decode: PhaseMetrics

    def get_total_metrics(self) -> OperationMetrics:
        return _aggregate_metrics([
            self.prefill.get_total_metrics(),
            self.decode.get_total_metrics(),
        ])

    @property
    def peak_sram_bytes(self) -> int:
        return max(self.prefill.peak_sram_bytes, self.decode.peak_sram_bytes)

    # ---- KV cache statistics ------------------------------------------------

    def get_kv_cache_writes(self) -> dict:
        """Return the number of K and V values written to KV cache.

        KV cache writes come from K_PROJ and V_PROJ DRAM writes.
        The shape of each projection is (M, d_model, d_kv), producing
        M × d_kv output values per execution.

        Returns a dict with:
          - k_values / v_values:  total number of scalar values written
          - k_bytes  / v_bytes:   total bytes written to DRAM
          - total_values / total_bytes: combined K + V
          - prefill_k_values / prefill_v_values / decode_k_values / decode_v_values
        """
        result = {}
        for phase_name, phase in [('prefill', self.prefill), ('decode', self.decode)]:
            for proj, op_type in [('k', OperationType.K_PROJ), ('v', OperationType.V_PROJ)]:
                ops = phase.aw_ops.get(op_type, [])
                values = sum(m.shape[0] * m.shape[2] for m in ops)   # M × N
                dram_bytes = sum(m.dram_write for m in ops)
                result[f'{phase_name}_{proj}_values'] = values
                result[f'{phase_name}_{proj}_bytes'] = dram_bytes

        result['k_values'] = result['prefill_k_values'] + result['decode_k_values']
        result['v_values'] = result['prefill_v_values'] + result['decode_v_values']
        result['k_bytes'] = result['prefill_k_bytes'] + result['decode_k_bytes']
        result['v_bytes'] = result['prefill_v_bytes'] + result['decode_v_bytes']
        result['total_values'] = result['k_values'] + result['v_values']
        result['total_bytes'] = result['k_bytes'] + result['v_bytes']
        return result

    # ---- Serialization ------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "prefill": self._phase_to_dict(self.prefill),
            "decode": self._phase_to_dict(self.decode),
            "total": self._metrics_to_dict(self.get_total_metrics()),
            "peak_sram_bytes": self.peak_sram_bytes,
            "kv_cache_writes": self.get_kv_cache_writes(),
        }

    def _phase_to_dict(self, phase: PhaseMetrics) -> dict:
        def _ops_dict(ops, mode):
            out = {}
            for op_type, op_list in ops.items():
                out[op_type.value] = {
                    "executions": [self._metrics_to_dict(m) for m in op_list],
                    "total": self._metrics_to_dict(
                        phase.get_operation_total(op_type, mode)),
                    "count": len(op_list),
                }
            return out

        return {
            "aa_operations": _ops_dict(phase.aa_ops, ComputeMode.AA),
            "aw_operations": _ops_dict(phase.aw_ops, ComputeMode.AW),
            "aa_total": self._metrics_to_dict(phase.get_aa_total()),
            "aw_total": self._metrics_to_dict(phase.get_aw_total()),
            "phase_total": self._metrics_to_dict(phase.get_total_metrics()),
            "peak_sram_bytes": phase.peak_sram_bytes,
        }

    @staticmethod
    def _metrics_to_dict(m: OperationMetrics) -> dict:
        return {
            "shape": m.shape,
            "cycles": m.cycles,
            "flops": m.flops,
            "utilization": m.utilization,
            "throughput": m.throughput,
            "memory": {
                "dram_read": m.dram_read,
                "dram_write": m.dram_write,
                "dram_read_eff": m.dram_read_eff,
                "dram_write_eff": m.dram_write_eff,
                "sram_read": m.sram_read,
                "sram_write": m.sram_write,
            },
            "energy": {
                "compute": m.compute_energy,
                "dram_read": m.dram_read_energy,
                "dram_write": m.dram_write_energy,
                "sram_read": m.sram_read_energy,
                "sram_write": m.sram_write_energy,
                "dram_total": m.dram_energy,
                "sram_total": m.sram_energy,
                "total": m.total_energy,
            },
            "peak_sram_bytes": m.peak_sram_bytes,
            "sram_overflow": m.sram_overflow,
            "sram_refetch_bytes": m.sram_refetch_bytes,
            "kv_resident_bytes": m.kv_resident_bytes,
        }


# ============================================================================
# Simulator
# ============================================================================

class Simulator:
    """Cycle-level hardware simulator for LLM inference."""

    # Architecture constants
    MU = 4          # Multi-use factor (elements per LUT lookup)
    NUM_RAC = 32    # Number of RAC units
    LANES_EQUIV = 4096  # Equivalent compute lanes for utilization calculation

    def __init__(self, hw: HardwareConfig):
        self.hw = hw

    # ---- Public API ---------------------------------------------------------

    def simulate(self, model: ModelConfig, workload: WorkloadConfig) -> SimulationResults:
        """Run a full inference simulation (prefill + decode)."""
        prefill_metrics = PhaseMetrics(phase=Phase.PREFILL)
        decode_metrics = PhaseMetrics(phase=Phase.DECODE)

        self._simulate_prefill(model, workload, prefill_metrics)
        self._simulate_decode(model, workload, decode_metrics)

        return SimulationResults(prefill=prefill_metrics, decode=decode_metrics)

    # ---- Phase simulation ---------------------------------------------------

    def _simulate_prefill(self, model: ModelConfig, workload: WorkloadConfig,
                          metrics: PhaseMetrics):
        """Simulate the prefill phase (process all input tokens at once)."""
        seq_len = workload.input_tokens
        batch = workload.batch_size
        
        # 1. Token embedding lookup (once at the start)
        embedding = self._simulate_embedding_lookup(
            num_tokens=batch * seq_len, 
            d_model=model.d_model
        )
        metrics.add_non_gemm_operation(embedding)
        
        # 2. Transformer layers (GEMM + non-GEMM)
        self._simulate_transformer_step(
            metrics, model, workload,
            proj_m=batch * seq_len,
            attn_q_len=seq_len,
            kv_len=seq_len,
            is_decode=False,
        )
        
        # 3. Final RMSNorm (after last layer)
        final_rmsnorm = self._simulate_rmsnorm(
            num_tokens=batch * seq_len,
            d_model=model.d_model,
            op_type=NonGEMMOpType.RMSNORM_FINAL
        )
        metrics.add_non_gemm_operation(final_rmsnorm)
        
        # 4. LM head softmax (next token prediction)
        # Note: LM head projection is a GEMM (d_model → vocab_size)
        # We only model the softmax here
        lm_softmax = self._simulate_lm_head_softmax(batch_size=batch)
        metrics.add_non_gemm_operation(lm_softmax)

    def _simulate_decode(self, model: ModelConfig, workload: WorkloadConfig,
                         metrics: PhaseMetrics):
        """Simulate the decode phase (generate one token at a time)."""
        batch = workload.batch_size
        
        for token_idx in range(1, workload.output_tokens):
            kv_len = workload.input_tokens + token_idx
            
            # 1. Token embedding lookup (for the new token)
            embedding = self._simulate_embedding_lookup(
                num_tokens=batch,
                d_model=model.d_model
            )
            metrics.add_non_gemm_operation(embedding)
            
            # 2. Transformer layers
            self._simulate_transformer_step(
                metrics, model, workload,
                proj_m=batch,
                attn_q_len=1,
                kv_len=kv_len,
                is_decode=True,
                token_idx=token_idx,
            )
            
            # 3. Final RMSNorm
            final_rmsnorm = self._simulate_rmsnorm(
                num_tokens=batch,
                d_model=model.d_model,
                op_type=NonGEMMOpType.RMSNORM_FINAL
            )
            metrics.add_non_gemm_operation(final_rmsnorm)
            
            # 4. LM head softmax
            lm_softmax = self._simulate_lm_head_softmax(batch_size=batch)
            metrics.add_non_gemm_operation(lm_softmax)

    # ---- Layer simulation ---------------------------------------------------

    def _simulate_transformer_step(
        self, metrics: PhaseMetrics,
        model: ModelConfig, workload: WorkloadConfig,
        proj_m: int, attn_q_len: int, kv_len: int,
        is_decode: bool, token_idx: int = -1,
    ):
        """Simulate one representative transformer layer, then replicate ×num_layers.

        This covers: Q/K/V projections → Attention → O projection → FFN.

        Args:
            proj_m:     M dimension for projection matmuls (batch × seq_tokens).
            attn_q_len: Query sequence length (seq_len for prefill, 1 for decode).
            kv_len:     Key/value sequence length (grows during decode).
            is_decode:  Whether we are in the decode phase.
            token_idx:  Current decode token index (-1 for prefill).
        """
        d = model.d_model
        d_kv = model.d_kv
        d_ffn = model.d_ffn
        batch = workload.batch_size
        num_heads = model.num_heads
        num_kv_heads = model.num_kv_heads
        head_dim = model.head_dim
        seq_len = workload.input_tokens

        # Collect (op_type, compute_mode, metrics) for one layer
        layer_ops: List[Tuple[OperationType, ComputeMode, OperationMetrics]] = []

        # 1a. Q projection (AW) — output dim = d_model
        q_m = self._simulate_matmul(
            OperationType.Q_PROJ, ComputeMode.AW, shape=(proj_m, d, d),
            batch_size=1, is_decode=is_decode,
            seq_len=seq_len, kv_len=kv_len,
        )
        layer_ops.append((OperationType.Q_PROJ, ComputeMode.AW, q_m))

        # 1b. K / V projections (AW) — output dim = d_kv (= d_model for MHA)
        for op_type in (OperationType.K_PROJ, OperationType.V_PROJ):
            m = self._simulate_matmul(
                op_type, ComputeMode.AW, shape=(proj_m, d, d_kv),
                batch_size=1, is_decode=is_decode,
                seq_len=seq_len, kv_len=kv_len,
            )
            layer_ops.append((op_type, ComputeMode.AW, m))

        # 2. Attention (Activation × Activation)
        if workload.flash_block_size > 0:
            # --- FlashAttention fused path ---
            flash_m = self._simulate_flash_attention(
                model, workload,
                attn_q_len=attn_q_len, kv_len=kv_len,
                batch_size=batch * num_heads,
                kv_batch_size=batch * num_kv_heads,
                is_decode=is_decode, sram_batch=batch,
            )
            layer_ops.append((OperationType.FLASH_ATTN, ComputeMode.AA, flash_m))
        else:
            # --- Standard attention: separate QK and Attn·V ---
            qk_m = self._simulate_matmul(
                OperationType.QK_MATMUL, ComputeMode.AA,
                shape=(attn_q_len, head_dim, kv_len),
                batch_size=batch * num_heads, is_decode=is_decode,
                seq_len=seq_len, kv_len=kv_len,
                kv_batch_size=batch * num_kv_heads, sram_batch=batch,
            )
            layer_ops.append((OperationType.QK_MATMUL, ComputeMode.AA, qk_m))

            av_m = self._simulate_matmul(
                OperationType.ATTN_V_MATMUL, ComputeMode.AA,
                shape=(attn_q_len, kv_len, head_dim),
                batch_size=batch * num_heads, is_decode=is_decode,
                seq_len=seq_len, kv_len=kv_len,
                kv_batch_size=batch * num_kv_heads, sram_batch=batch,
            )
            layer_ops.append((OperationType.ATTN_V_MATMUL, ComputeMode.AA, av_m))

        # 3. O projection (AW)
        o_m = self._simulate_matmul(
            OperationType.O_PROJ, ComputeMode.AW, shape=(proj_m, d, d),
            batch_size=1, is_decode=is_decode,
            seq_len=seq_len, kv_len=kv_len,
        )
        layer_ops.append((OperationType.O_PROJ, ComputeMode.AW, o_m))

        # 4. FFN (with MoE support)
        num_active = model.num_active_experts

        if model.is_moe:
            # 4a. Gate / router: (proj_m, d_model) @ (d_model, num_experts)
            gate_m = self._simulate_matmul(
                OperationType.GATE, ComputeMode.AW,
                shape=(proj_m, d, model.num_experts),
                batch_size=1, is_decode=is_decode,
                seq_len=seq_len, kv_len=kv_len,
            )
            layer_ops.append((OperationType.GATE, ComputeMode.AW, gate_m))

        # 4b. FC1 — each active expert has its own weights (d_model → d_ffn)
        # `d_ffn_1` / `d_ffn_2` are the hidden units each matrix actually
        # touches.  Both are `model.d_ffn` unless `_ffn_active_neurons` is
        # overridden, so the dense model is unchanged.  They can differ: a mask
        # derived from FC1's own output cannot skip FC1.
        d_ffn_1 = self._ffn_active_neurons(model, OperationType.FC1, is_decode)
        d_ffn_2 = self._ffn_active_neurons(model, OperationType.FC2, is_decode)
        fc1_m = self._simulate_matmul(
            OperationType.FC1, ComputeMode.AW, shape=(proj_m, d, d_ffn_1),
            batch_size=1, is_decode=is_decode,
            seq_len=seq_len, kv_len=kv_len,
        )
        for _ in range(num_active):
            layer_ops.append((OperationType.FC1, ComputeMode.AW, fc1_m))

        # 4c. FC2 — each active expert has its own weights (d_ffn → d_model)
        fc2_m = self._simulate_matmul(
            OperationType.FC2, ComputeMode.AW, shape=(proj_m, d_ffn_2, d),
            batch_size=1, is_decode=is_decode,
            seq_len=seq_len, kv_len=kv_len,
        )
        for _ in range(num_active):
            layer_ops.append((OperationType.FC2, ComputeMode.AW, fc2_m))

        # Replicate across all layers
        for op_type, compute_mode, op_metrics in layer_ops:
            for _ in range(model.num_layers):
                metrics.add_operation(op_type, compute_mode, op_metrics)

        # ========== Non-GEMM Operations (per layer, replicated) ==========
        # These are simulated once and replicated across all layers
        
        # Number of tokens being processed in this step
        num_tokens = proj_m  # batch * seq_len for prefill, batch * 1 for decode
        
        # Collect non-GEMM ops for one layer
        layer_non_gemm_ops: List[NonGEMMMetrics] = []
        
        # 1. RMSNorm before attention
        rmsnorm_pre_attn = self._simulate_rmsnorm(
            num_tokens=num_tokens, d_model=d,
            op_type=NonGEMMOpType.RMSNORM_PRE_ATTN
        )
        layer_non_gemm_ops.append(rmsnorm_pre_attn)
        
        # 2. RoPE on Q and K (applied to Q and K projections)
        # RoPE is applied to d_model elements (all heads)
        rope_q = self._simulate_rope(num_tokens=num_tokens, d_model=d)
        rope_k = self._simulate_rope(num_tokens=num_tokens, d_model=d_kv)
        layer_non_gemm_ops.append(rope_q)
        layer_non_gemm_ops.append(rope_k)
        
        # 3. Attention scaling (Q * 1/sqrt(d_k))
        attn_scale = self._simulate_attn_scale(num_tokens=num_tokens, d_model=d)
        layer_non_gemm_ops.append(attn_scale)
        
        # 4. Softmax (after QK matmul, before Attn·V)
        # Shape: [batch * num_heads, q_len, kv_len]
        softmax = self._simulate_softmax(
            batch_heads=batch * num_heads,
            q_len=attn_q_len,
            kv_len=kv_len
        )
        layer_non_gemm_ops.append(softmax)
        
        # 5. Residual add after attention (x + attention_output)
        residual_attn = self._simulate_residual_add(
            num_tokens=num_tokens, d_model=d,
            op_type=NonGEMMOpType.RESIDUAL_ATTN
        )
        layer_non_gemm_ops.append(residual_attn)
        
        # 6. RMSNorm before FFN
        rmsnorm_pre_ffn = self._simulate_rmsnorm(
            num_tokens=num_tokens, d_model=d,
            op_type=NonGEMMOpType.RMSNORM_PRE_FFN
        )
        layer_non_gemm_ops.append(rmsnorm_pre_ffn)
        
        # 7. SiLU activation (after gate projection in SwiGLU)
        # Applied to d_ffn elements
        # Both element-wise stages run over what FC1 actually produced, not
        # over the nominal hidden dimension: units that were never computed
        # have nothing to activate or gate.
        silu = self._simulate_silu(num_tokens=num_tokens, d_ffn=d_ffn_1)
        layer_non_gemm_ops.append(silu)
        
        # 8. Gated multiplication (gate * up in SwiGLU)
        gated_mul = self._simulate_gated_mul(num_tokens=num_tokens,
                                             d_ffn=d_ffn_1)
        layer_non_gemm_ops.append(gated_mul)
        
        # 9. Residual add after FFN (x + ffn_output)
        residual_ffn = self._simulate_residual_add(
            num_tokens=num_tokens, d_model=d,
            op_type=NonGEMMOpType.RESIDUAL_FFN
        )
        layer_non_gemm_ops.append(residual_ffn)
        
        # Replicate non-GEMM ops across all layers
        for non_gemm_op in layer_non_gemm_ops:
            for _ in range(model.num_layers):
                metrics.add_non_gemm_operation(non_gemm_op)

    # ---- Single matmul simulation -------------------------------------------

    def _simulate_matmul(
        self, op_type: OperationType, compute_mode: ComputeMode,
        shape: Tuple[int, int, int],
        batch_size: int = 1, is_decode: bool = False,
        seq_len: int = 0, kv_len: int = 0,
        kv_batch_size: int = 0, sram_batch: int = 1,
    ) -> OperationMetrics:
        """Simulate a single matrix multiplication A(M×K) @ B(K×N) = C(M×N).

        Args:
            kv_batch_size: Effective batch size for the B operand (K/V) DRAM
                access.  For GQA, this is batch × num_kv_heads while batch_size
                is batch × num_heads.  0 means same as batch_size (MHA).
            sram_batch: The workload batch alone, heads excluded -- how many
                attention instances must be co-resident under
                `hw.sram_batch_model == "concurrent"`.  Only meaningful for
                AA ops; AW ops already carry batch in M.

        Returns fully-populated OperationMetrics.
        """
        M, K, N = shape
        hw = self.hw

        # Resolve dataflow mode (handles OMNI → LUT_WS / LUT_OS_V)
        base_mode, resolved_mode = self._resolve_dataflow_mode(compute_mode, is_decode)
        qbit = hw.weight_bits if compute_mode == ComputeMode.AW else hw.kv_cache_bits

        metrics = OperationMetrics(shape=shape)
        metrics.flops = 2 * M * K * N * batch_size

        # 1. Cycles
        metrics.cycles = self._calculate_cycles(M, K, N, qbit, compute_mode,
                                                resolved_mode, batch_size)

        # 2. Memory access
        mem = self._calculate_memory_access(
            M, K, N, compute_mode, op_type, resolved_mode, batch_size,
            is_decode=is_decode, seq_len=seq_len, kv_len=kv_len,
            kv_batch_size=kv_batch_size,
        )
        metrics.dram_read = mem["dram_read"]
        metrics.dram_write = mem["dram_write"]
        metrics.dram_read_eff = mem["dram_read_eff"]
        metrics.dram_write_eff = mem["dram_write_eff"]
        metrics.kv_resident_bytes = mem["kv_resident_bytes"]
        metrics.sram_read = mem["sram_read"]
        metrics.sram_write = mem["sram_write"]
        metrics.sram_read_a = mem["sram_read_a"]
        metrics.sram_read_b = mem["sram_read_b"]
        metrics.sram_acc_read = mem["sram_acc_read"]
        metrics.sram_acc_write = mem["sram_acc_write"]
        metrics.sram_write_out = mem["sram_write_out"]

        # 3. Peak SRAM footprint, and the spill it forces if capacity is finite
        metrics.peak_sram_bytes = self._calculate_peak_sram(
            M, K, N, compute_mode, resolved_mode, batch_size,
            sram_batch=sram_batch,
        )
        metrics.sram_overflow, metrics.sram_refetch_bytes = (
            self._apply_sram_capacity(
                M, K, N, compute_mode, resolved_mode, batch_size,
                metrics.peak_sram_bytes,
            )
        )
        # Re-fetch traffic is real DRAM traffic: fold it in before energy and
        # roofline see the operation, not after.  A spilled operand is re-read
        # as a contiguous block, so it costs the same logically and effectively.
        metrics.dram_read += metrics.sram_refetch_bytes
        metrics.dram_read_eff += self._dram_effective_bytes(
            metrics.sram_refetch_bytes, metrics.sram_refetch_bytes)
        mem["dram_read"] = metrics.dram_read
        mem["dram_read_eff"] = metrics.dram_read_eff

        # 4. Utilization
        metrics.utilization = (
            (metrics.flops / 2) / (metrics.cycles * self.LANES_EQUIV)
            if metrics.cycles > 0 else 0.0
        )

        # 5. Energy
        energy = self._calculate_memory_energy(mem)
        metrics.dram_read_energy = energy["dram_read"]
        metrics.dram_write_energy = energy["dram_write"]
        metrics.sram_read_energy = energy["sram_read"]
        metrics.sram_write_energy = energy["sram_write"]
        metrics.compute_energy = self._calculate_compute_energy(
            M, K, N, batch_size, qbit, base_mode, resolved_mode, is_decode,
        )

        return metrics

    # ---- FlashAttention simulation ------------------------------------------

    def _simulate_flash_attention(
        self, model: ModelConfig, workload: WorkloadConfig,
        attn_q_len: int, kv_len: int,
        batch_size: int, kv_batch_size: int,
        is_decode: bool, sram_batch: int = 1,
    ) -> OperationMetrics:
        """Simulate FlashAttention-style fused tiled attention.

        Instead of materializing the full (seq × seq) attention matrix,
        FlashAttention tiles the computation:

            For each Q block (Br rows):
                For each K/V block (Bc rows):
                    S_block = Q_block @ K_block^T       (Br × Bc)
                    Update online softmax statistics
                    O_block += softmax(S_block) @ V_block

        This dramatically reduces peak SRAM from O(seq²) to O(Br × Bc).

        Args:
            attn_q_len: Query sequence length (full seq for prefill, 1 for decode).
            kv_len:     Key/value sequence length.
            batch_size: batch_size × num_heads (each head is independent).
            is_decode:  Whether this is the decode phase.
        """
        hw = self.hw
        head_dim = model.head_dim
        B = workload.flash_block_size

        Br = min(B, attn_q_len)   # Query block size
        Bc = min(B, kv_len)       # Key/value block size
        num_q_blocks = math.ceil(attn_q_len / Br)
        num_kv_blocks = math.ceil(kv_len / Bc)

        # Resolve dataflow for the AA matmuls inside attention
        base_mode, resolved_mode = self._resolve_dataflow_mode(ComputeMode.AA, is_decode)
        qbit = hw.kv_cache_bits

        # --- FLOPs (identical to standard attention) ---
        total_flops = 2 * 2 * attn_q_len * head_dim * kv_len * batch_size

        # --- Cycles: per-tile QK + SV matmuls ---
        qk_tile_cycles = self._calculate_cycles(
            Br, head_dim, Bc, qbit, ComputeMode.AA, resolved_mode, batch_size=1)
        sv_tile_cycles = self._calculate_cycles(
            Br, Bc, head_dim, qbit, ComputeMode.AA, resolved_mode, batch_size=1)
        # softmax_overhead = Br  # Online softmax is lightweight
        cycles_per_tile = qk_tile_cycles + sv_tile_cycles
        total_cycles = num_q_blocks * num_kv_blocks * cycles_per_tile * batch_size

        # --- Memory access ---
        act_bits = hw.act_bits
        kv_bits = hw.kv_cache_bits
        accum_bits = hw.accumulate_bits

        Q_total_bits = batch_size * attn_q_len * head_dim * act_bits
        K_total_bits = batch_size * kv_len * head_dim * kv_bits
        V_total_bits = batch_size * kv_len * head_dim * kv_bits
        O_total_bits = batch_size * attn_q_len * head_dim * accum_bits

        # SRAM traffic: Q read once per KV-block pass; K, V read once per Q-block pass
        sram_read_bits = (Q_total_bits * num_kv_blocks
                          + K_total_bits * num_q_blocks
                          + V_total_bits * num_q_blocks)
        sram_write_bits = O_total_bits

        # DRAM: decode reads previous KV cache from DRAM; prefill keeps Q/K/V in SRAM
        # KV DRAM reads scale with num_kv_heads (GQA), not num_heads
        eff_kv_batch = kv_batch_size if kv_batch_size > 0 else batch_size
        dram_read_bits = 0
        kv_resident_bytes = 0
        if is_decode and kv_len > 1:
            kv_prev = kv_len - 1
            dram_read_bits = eff_kv_batch * kv_prev * head_dim * kv_bits * 2  # K + V
            # One operation reads both K and V, so it gets the whole buffer.
            kv_resident_bytes = self._kv_resident_bytes(
                dram_read_bits // 8, share=1.0)
            dram_read_bits -= kv_resident_bytes * 8
        elif hw.prefill_kv_dram_read and kv_len > 0:
            # Mirrors the standard path's prefill K/V read.  Flash re-reads the
            # whole K/V once per Q block -- that is the real cost of its tiling,
            # and the reason a larger block is cheaper.  Without this, giving
            # only the standard path a prefill read would make flash look
            # artificially cheap and invert the convergence check.
            dram_read_bits = (eff_kv_batch * kv_len * head_dim * kv_bits * 2
                              * num_q_blocks)
            kv_resident_bytes = self._kv_resident_bytes(
                dram_read_bits // 8, share=1.0)
            dram_read_bits -= kv_resident_bytes * 8
        dram_write_bits = 0  # Attention scores never written to DRAM

        # --- Peak SRAM footprint ---
        # Simultaneously in SRAM: Q_block + K_block + V_block + S_block + O_block + stats
        peak_sram_bits = (
            Br * head_dim * act_bits          # Q block
            + Bc * head_dim * kv_bits         # K block
            + Bc * head_dim * kv_bits         # V block
            + Br * Bc * act_bits              # S block (attention scores)
            + Br * head_dim * accum_bits      # O block (output accumulator)
            + Br * 2 * 32                     # Online softmax running max + sum
        )
        # Same batch-residency rule as _calculate_peak_sram: one (batch, head)
        # instance by default, all batch elements co-resident under
        # "concurrent".  Flash is an AA path, so the switch applies here too.
        if hw.sram_batch_model == "concurrent":
            peak_sram_bits *= max(1, sram_batch)

        # --- Build metrics ---
        metrics = OperationMetrics(
            shape=(attn_q_len, head_dim, kv_len),
            cycles=total_cycles,
            flops=total_flops,
            peak_sram_bytes=peak_sram_bits // 8,
            dram_read=dram_read_bits // 8,
            dram_write=dram_write_bits // 8,
            sram_read=sram_read_bits // 8,
            sram_write=sram_write_bits // 8,
            # Q is the activation operand; K and V are the B operand, and in
            # flash they are re-streamed once per Q block.  Online softmax
            # keeps the running accumulator in the array, so -- like every
            # other output-stationary path here -- there is no accumulator
            # recirculation to charge.
            sram_read_a=Q_total_bits * num_kv_blocks // 8,
            sram_read_b=(K_total_bits + V_total_bits) * num_q_blocks // 8,
            sram_write_out=sram_write_bits // 8,
        )

        # Capacity check.  FlashAttention is already tiled to a fixed block, so
        # there is no resident operand to spill -- the fix is a smaller Br/Bc,
        # which is a re-tiling and out of scope for policy v1.  Flag it so an
        # infeasible block size is visible, and charge nothing.
        cap = hw.sram_capacity_kb * 1024
        metrics.sram_overflow = cap > 0 and metrics.peak_sram_bytes > cap

        # Utilization
        metrics.utilization = (
            (metrics.flops / 2) / (metrics.cycles * self.LANES_EQUIV)
            if metrics.cycles > 0 else 0.0
        )

        metrics.kv_resident_bytes = kv_resident_bytes

        # Flash reads one contiguous K/V block per (batch, head), so its runs
        # are long and the burst term is inert -- but compute it the same way
        # rather than assuming, so a future short-run flash variant is covered.
        metrics.dram_read_eff = self._dram_effective_bytes(
            metrics.dram_read, metrics.dram_read)
        metrics.dram_write_eff = self._dram_effective_bytes(
            metrics.dram_write, metrics.dram_write)

        # Energy: memory
        mem_access = {
            "dram_read": metrics.dram_read, "dram_write": metrics.dram_write,
            "dram_read_eff": metrics.dram_read_eff,
            "dram_write_eff": metrics.dram_write_eff,
            "sram_read": metrics.sram_read, "sram_write": metrics.sram_write,
        }
        energy = self._calculate_memory_energy(mem_access)
        metrics.dram_read_energy = energy["dram_read"]
        metrics.dram_write_energy = energy["dram_write"]
        metrics.sram_read_energy = energy["sram_read"]
        metrics.sram_write_energy = energy["sram_write"]

        # Energy: compute (sum of tile-level QK + SV energies)
        qk_tile_energy = self._calculate_compute_energy(
            Br, head_dim, Bc, batch_size=1, qbit=qbit,
            base_mode=base_mode, resolved_mode=resolved_mode, is_decode=is_decode,
        )
        sv_tile_energy = self._calculate_compute_energy(
            Br, Bc, head_dim, batch_size=1, qbit=qbit,
            base_mode=base_mode, resolved_mode=resolved_mode, is_decode=is_decode,
        )
        metrics.compute_energy = (
            (qk_tile_energy + sv_tile_energy) * num_q_blocks * num_kv_blocks * batch_size
        )

        return metrics

    # ---- Mode resolution ----------------------------------------------------

    def _resolve_dataflow_mode(self, compute_mode: ComputeMode,
                               is_decode: bool) -> Tuple[str, str]:
        """Resolve the effective dataflow mode, handling OMNI dispatch.

        Returns:
            (base_mode, resolved_mode)
            - base_mode:     Original config mode (needed for energy model selection).
            - resolved_mode: Actual dataflow used for cycles & memory calculation.
        """
        hw = self.hw
        base_mode = hw.AW_mode if compute_mode == ComputeMode.AW else hw.AA_mode

        if hw.AW_mode == "OMNI":
            resolved_mode = "LUT_OS_V" if is_decode else "LUT_WS"
        else:
            resolved_mode = base_mode

        return base_mode, resolved_mode

    # ---- Cycle calculation --------------------------------------------------

    def _calculate_cycles(self, M: int, K: int, N: int, qbit: int,
                          compute_mode: ComputeMode, mode: str,
                          batch_size: int) -> int:
        """Calculate operation cycles for A(M×K) @ B(K×N).

        Cycle counts depend on the dataflow mode and array geometry.
        """
        hw = self.hw
        array_m = hw.array_m
        array_n = hw.array_n
        fpe_size = hw.FPE_array_size
        replication = hw.replication

        LUT_GEN_CYCLES = 3
        OUTPUT_CYCLES = 2
        INPUT_CYCLES = 1

        if mode in ("LUT_OS", "LUT_OS_V"):
            k_eff = math.ceil(K / self.MU)
            m_tiles = math.ceil(M / array_m)
            n_tiles = math.ceil(N / (array_n * self.NUM_RAC))

            if M == 1:
                per_round = LUT_GEN_CYCLES + k_eff + 1 + array_n + OUTPUT_CYCLES
            else:
                per_round = LUT_GEN_CYCLES + k_eff + array_m + array_n + OUTPUT_CYCLES

            if mode == "LUT_OS_V" and M == 1:
                rounds = math.ceil(n_tiles / array_m / replication)
            elif mode == "LUT_OS_V" and hw.os_rounds_model == "packed":
                # Accumulator-budget form: a round retires `array_m` tiles
                # wherever they come from.  Reduces to the M == 1 branch above
                # at M = 1, which is the whole argument for it.
                rounds = math.ceil(M * n_tiles / array_m / replication)
            else:
                rounds = math.ceil(m_tiles * n_tiles / replication)

            return batch_size * per_round * rounds * qbit

        elif mode == "LUT_WS":
            k_eff = math.ceil(K / self.MU)
            k_tiles = math.ceil(k_eff / array_m)
            n_tiles = math.ceil(N / (array_n * self.NUM_RAC))

            # Row blocking re-pays fill/drain once per block but streams the
            # same `M` activations in total, so the exact form is
            # `M + m_blocks * overhead` rather than `m_blocks * (mt +
            # overhead)` -- the latter would charge for the padding rows of a
            # final short block.  `m_blocks == 1` gives back `M + overhead`
            # exactly, which is why the untiled default is bit-identical.
            overhead = array_n + array_m + INPUT_CYCLES + OUTPUT_CYCLES
            m_blocks = self._row_blocks(M)
            rounds = math.ceil(n_tiles * k_tiles / replication)
            return batch_size * rounds * (M + m_blocks * overhead) * qbit

        elif mode == "FPE_OS":
            m_tiles = math.ceil(M / fpe_size)
            n_tiles = math.ceil(N / fpe_size)

            if M == 1:
                per_tile = K + 1 + fpe_size + INPUT_CYCLES + OUTPUT_CYCLES
            else:
                per_tile = K + fpe_size + fpe_size + INPUT_CYCLES + OUTPUT_CYCLES

            return batch_size * m_tiles * n_tiles * per_tile

        elif mode == "TENDER":
            m_tiles = math.ceil(M / fpe_size)
            n_tiles = math.ceil(N / fpe_size)

            if M == 1 and hw.tender_m1_opt:
                per_tile = (K + 1 + fpe_size
                            + INPUT_CYCLES + OUTPUT_CYCLES + 16)  # +16 rescale
            else:
                per_tile = (K + fpe_size + fpe_size
                            + INPUT_CYCLES + OUTPUT_CYCLES + 16)  # +16 rescale
            return batch_size * m_tiles * n_tiles * per_tile

        elif mode == "FPE_WS":
            k_tiles = math.ceil(K / fpe_size)
            n_tiles = math.ceil(N / fpe_size)

            per_tile = M + fpe_size + fpe_size + INPUT_CYCLES + OUTPUT_CYCLES
            return batch_size * k_tiles * n_tiles * per_tile

        elif mode == "VPU":
            return (M * K * N * batch_size) // 64

        return 0

    # ---- Memory access calculation ------------------------------------------

    def _dram_effective_bytes(self, logical: int, run_bytes: int,
                              cap_bytes: int = 0) -> int:
        """Round a DRAM access up to whole bursts.

        `logical` is the total bytes wanted; `run_bytes` is how many of them
        are *contiguous* per access.  Each run costs
        `ceil(run / burst) * burst`, so the total scales by that ratio.

        The distinction is the whole point: 1 MB read as one contiguous block
        and 1 MB read as 27,000 scattered 38-byte snippets are the same
        `logical` and wildly different costs.  With `dram_burst_bytes = 0`,
        or a run at least as long as a burst and burst-aligned, this returns
        `logical` unchanged.
        """
        burst = self.hw.dram_burst_bytes
        if burst <= 0 or run_bytes <= 0:
            return logical
        moved_per_run = math.ceil(run_bytes / burst) * burst
        charged = logical * moved_per_run // run_bytes
        # A scattered read can never cost more than giving up and reading the
        # whole contiguous region the scattered set sits inside.  Without this
        # a fine-grained mask prices above a dense read, which no controller
        # would ever actually do.  `cap_bytes = 0` means "no covering region
        # known", and every base-model access is already whole-entry runs, so
        # this clamp is inert unless a sub-entry mask is in play.
        if cap_bytes > 0:
            return min(charged, cap_bytes)
        return charged

    def _kv_resident_bytes(self, kv_bytes: int, share: float = 0.5) -> int:
        """How much of this op's KV working set stays on chip between steps.

        The cache is append-only, so anything held in the KV buffer at step
        `t` is still valid at `t+1` and costs no DRAM read.  Whatever does not
        fit is streamed from DRAM every token, as before.

        `share` splits the buffer between the two decode attention ops: QK
        reads K, Attn·V reads V, so each gets half.  FlashAttention reads both
        in one operation and passes `share=1.0`.

        **Steady-state model.** The one-time cost of filling the buffer after
        prefill is not charged, which slightly favours residency: over 256
        output tokens the fill is under 0.4% of the KV traffic, and it is a
        constant rather than a per-token term.  Charging it would need
        `_calculate_memory_access` to know the decode step index, which it
        deliberately does not.
        """
        buf = self.hw.kv_sram_kb * 1024
        if buf <= 0:
            return 0
        return min(kv_bytes, int(buf * share))

    def _score_staged(self, score_width: int) -> bool:
        """Do the attention scores stay on chip between QK and Attn.V?

        `score_width` is the score-vector width for ONE (batch, head) instance
        -- `kv_len` -- deliberately *not* multiplied by `batch_size`.  The
        write site multiplies by `batch_size = batch x num_heads`; if that
        multiplied count were tested here, the buffer requirement would scale
        with head count and staging would never fire.  Heads are processed as
        separate instances, so one instance's row is what has to fit.

        Both the QK write and the Attn.V read-back consult this one function,
        so the two sides of the score cannot disagree about where it lives.
        """
        buf = self.hw.score_sram_kb * 1024
        if buf <= 0 or score_width <= 0:
            return False
        return score_width * self.hw.act_bits // 8 <= buf

    def _kv_dram_run_entries(self, kv_prev: int) -> int:
        """Contiguous KV entries per DRAM access.

        The cache is laid out per (batch, head), so a dense or compacted cache
        reads one head's whole block contiguously -- the default here.  A
        page-selective reader (Quest / TidalDecode / NSA) overrides this with
        its page size, which is what makes the burst term bite.
        """
        return kv_prev

    def _kv_dram_run_bytes(self, run_entries: int, head_dim: int,
                           kv_bits: int) -> int:
        """Contiguous bytes moved per KV DRAM access.

        Default: whole entries, so one run is `run_entries` complete
        `head_dim x kv_bits` rows and the run is always an exact multiple of
        an entry.  That is what makes token-granular selection free on this
        hardware -- a 4-bit entry is 64 B, exactly one burst.

        An *unstructured channel* mask breaks that assumption, because it
        fragments the entry itself: the run drops below one entry and burst
        granularity starts to bite.  Overriding this is the only way to
        express that, which is why it is a separate hook from
        `_kv_dram_run_entries`.
        """
        return run_entries * head_dim * kv_bits // 8

    def _sram_ports(self, m) -> Tuple[int, int, int]:
        """Split one operation's SRAM bytes into (activation, weight, accum).

        The activation port carries A-reads *and* the final result writes:
        both live in the Unified Buffer, and one layer's result is the next
        layer's A.  The weight port carries B-reads (weights, or KV in an AA
        op).  The accumulator carries partial-sum recirculation only.

        An operation that recorded no split at all -- every non-GEMM VPU op,
        and `mode == "VPU"`, none of which have a tiled loop nest to split --
        is pure activation traffic, so its whole lump goes to the activation
        port.  That keeps "ported" from silently pricing those at zero.
        """
        a = m.sram_read_a
        b = m.sram_read_b
        acc = m.sram_acc_read + m.sram_acc_write
        out = m.sram_write_out
        if a == 0 and b == 0 and acc == 0 and out == 0:
            return m.sram_read + m.sram_write, 0, 0
        return a + out, b, acc

    def _sram_time(self, m) -> float:
        """Seconds one operation spends moving bytes to and from SRAM.

        0.0 when `sram_bandwidth_gbps` is 0 (unlimited), which makes every
        roofline below reduce exactly to `max(compute, dram)` -- the two-term
        form every published result was produced with.

        Under `sram_port_model = "ported"` the three memories of Fig. 4 move
        their bytes concurrently and the operation waits for the slowest, so
        the lumped sum becomes a max over ports.  See `sram_port_model`.
        """
        hw = self.hw
        bw = hw.sram_bandwidth_gbps * 1e9
        if hw.sram_port_model != "ported":
            if bw <= 0:
                return 0.0
            return (m.sram_read + m.sram_write) / bw

        act, wgt, acc = self._sram_ports(m)
        acc_bw = hw.accum_bandwidth_gbps * 1e9
        t_act = act / bw if bw > 0 else 0.0
        t_wgt = wgt / bw if bw > 0 else 0.0
        t_acc = acc / acc_bw if acc_bw > 0 else 0.0
        return max(t_act, t_wgt, t_acc)

    def _roofline_time_over(self, metrics, freq: float, dram_bw: float) -> float:
        """Roofline time for a group of operations, honouring `overlap_model`.

        `"serial"` sums each operation's own `max(...)`; `"pipelined"` sums each
        resource across the group and takes one `max` at the end, which is the
        limit a deep software pipeline approaches.  Every roofline site routes
        through here so the two models cannot disagree between call sites.
        """
        if self.hw.overlap_model != "pipelined":
            # Accumulated explicitly, NOT with sum(): CPython 3.12+ gives sum()
            # compensated (Neumaier) summation over floats, which is more
            # accurate and therefore differs from every previously published
            # number in the last ULP.  The gate exists to make a real regression
            # visible, and last-ULP noise across 22k values is exactly what
            # would hide one.
            total = 0.0
            for m in metrics:
                total += self._op_roofline_time(m, freq, dram_bw)
            return total
        compute = dram = sram = 0.0
        for m in metrics:
            compute += m.cycles / freq if freq > 0 else 0.0
            dram += ((m.dram_read_eff + m.dram_write_eff) / dram_bw
                     if dram_bw > 0 else 0.0)
            sram += self._sram_time(m)
        return max(compute, dram, sram)

    def _op_roofline_time(self, m, freq: float, dram_bw: float) -> float:
        """Roofline time for one operation: max(compute, DRAM, SRAM).

        Every roofline site calls this rather than inlining the max, because a
        term that reaches four of five sites is worse than no term at all --
        the numbers stay plausible and stop being consistent.
        """
        ct = m.cycles / freq if freq > 0 else 0.0
        mt = ((m.dram_read_eff + m.dram_write_eff) / dram_bw
              if dram_bw > 0 else 0.0)
        return max(ct, mt, self._sram_time(m))

    def _kv_covering_bytes(self, logical_bytes: int, head_dim: int,
                           kv_bits: int) -> int:
        """Bytes of the contiguous region the scattered KV set sits inside.

        This is the clamp on `_dram_effective_bytes`: a gathering reader can
        always fall back to reading the whole covering region in one stream,
        so it never pays more than that.

        0 = "no covering region known", which disables the clamp, and is the
        default *deliberately*.  For a dense or compacted read the covering
        region is the read itself, so a clamp would be a no-op; for a
        page-selective read it is the entire cache, far above anything the
        burst term can charge.  Only a sub-entry mask -- where the charge can
        exceed a dense read of the same region -- needs it, so only that
        subclass supplies one, and no existing result can move.
        """
        return 0

    # ---- FFN activation sparsity hooks --------------------------------------
    #
    # Three hooks, all default-identical, mirroring the KV mask hooks above.
    # The base model returns "dense, contiguous, no clamp" from all three, so
    # nothing moves unless a subclass overrides them --
    # `analysis/act_sparsity/act_sparsity.py` is the one that does.
    #
    # Why the FFN and why hooks rather than a `HardwareConfig` field: activation
    # sparsity is a property of the *model and its inputs*, not of the machine.
    # A gated FFN drives most of its hidden units to near-zero for any given
    # token, and skipping unit `j` skips column `j` of FC1 and row `j` of FC2 --
    # weights, not activations, are what stops being fetched.  That aims at
    # SS3's finding that decode idles ~85% waiting on *weights*.

    def _ffn_active_neurons(self, model, op_type: 'OperationType',
                            is_decode: bool) -> int:
        """How many FFN hidden units this operation actually touches.

        Defaults to all of them, which is every result predating this hook.

        Per operation rather than per layer, because *where the mask comes
        from* decides which matrices can use it.  Thresholding the FFN input
        (TEAL) or predicting from it (Deja Vu) masks FC1 and FC2 alike;
        thresholding FC1's own output (CATS) cannot skip the work that
        produced it, so FC1 stays dense and only FC2 is sparse.
        """
        return model.d_ffn

    def _aw_weight_run_bytes(self, op_type: 'OperationType',
                             logical_bytes: int) -> int:
        """Contiguous run length of an AW operation's weight read, in bytes.

        Defaults to the whole read -- one contiguous block, which is what the
        model has always assumed and what a dense weight matrix actually is.

        This is the FFN's version of the question SS15 asked about KV: a mask
        only collects its byte saving if the retained bytes are contiguous
        enough to fill a burst.  The answer differs, and the reason it differs
        is the point -- see `analysis/act_sparsity/`.
        """
        return logical_bytes

    def _aw_weight_covering_bytes(self, op_type: 'OperationType',
                                  logical_bytes: int) -> int:
        """Bytes of the contiguous region a scattered weight read sits inside.

        0 = no covering region known, which disables the clamp in
        `_dram_effective_bytes` and is the default.  A reader that gathers
        scattered weights can always give up and stream the dense matrix
        instead, so it never pays more than that; only a sub-burst mask can
        reach the clamp.
        """
        return 0

    def _calculate_memory_access(
        self, M: int, K: int, N: int,
        compute_mode: ComputeMode, op_type: OperationType,
        mode: str, batch_size: int,
        is_decode: bool = False, seq_len: int = 0, kv_len: int = 0,
        kv_batch_size: int = 0,
    ) -> dict:
        """Calculate DRAM and SRAM read/write bytes for one operation.

        Args:
            kv_batch_size: Effective batch for KV DRAM access (GQA).  0 = batch_size.
        """
        hw = self.hw
        act_bits = hw.act_bits
        accum_bits = hw.accumulate_bits
        weight_bits = hw.weight_bits
        kv_bits = hw.kv_cache_bits

        qbit = weight_bits if compute_mode == ComputeMode.AW else kv_bits

        # Operand sizes (bits)
        A_bits = batch_size * M * K * act_bits
        if compute_mode == ComputeMode.AA:
            B_bits = batch_size * K * N * kv_bits
        else:
            B_bits = batch_size * K * N * weight_bits
        C_accum_bits = batch_size * M * N * accum_bits

        # DRAM writes: K/V projections write KV cache to DRAM;
        #              QK matmul writes attention scores to DRAM (standard attention only)
        if op_type in (OperationType.K_PROJ, OperationType.V_PROJ):
            dram_write_bits = batch_size * M * N * kv_bits
        elif op_type == OperationType.QK_MATMUL:
            # Attention scores (M × N) spill to DRAM between QK and Attn·V --
            # unless a row's worth fits on chip, in which case they never leave.
            dram_write_bits = (0 if self._score_staged(N)
                               else batch_size * M * N * act_bits)
        else:
            dram_write_bits = 0

        # ---- SRAM traffic (mode-dependent) ----
        # Accumulated per *port* -- activation, weight, accumulator -- and the
        # lumped `sram_read` / `sram_write` are their sums, so both views come
        # out of one set of terms and cannot drift apart.  See
        # `hw.sram_port_model` for why the split exists.
        a_read_bits = 0        # activations, Unified Buffer
        b_read_bits = 0        # weights / KV, Weight Buffer
        acc_read_bits = 0      # partial sums in,  Accumulator
        acc_write_bits = 0     # partial sums out, Accumulator
        out_write_bits = 0     # final result, Unified Buffer

        if mode in ("LUT_OS", "LUT_OS_V"):
            m_tiles = math.ceil(M / hw.array_m)
            n_tiles = math.ceil(N / (hw.array_n * self.NUM_RAC))
            a_read_bits = A_bits * n_tiles * qbit
            b_read_bits = B_bits * m_tiles
            # Output-stationary: the partial sums never leave the array, which
            # is the entire point of the dataflow.  No accumulator traffic.
            out_write_bits = C_accum_bits

        elif mode == "LUT_WS":
            k_eff = math.ceil(K / self.MU)
            k_tiles = math.ceil(k_eff / hw.array_m)
            n_tiles = math.ceil(N / (hw.array_n * self.NUM_RAC))
            a_read_bits = A_bits * n_tiles * qbit
            # Weight-stationary keeps B resident across the whole `M` stream,
            # so blocking the rows re-loads it once per block.  That is the
            # honest cost of tiling and the reason the field is a knob rather
            # than a correction: bigger blocks trade capacity for weight
            # traffic.  The OS and FPE modes already charge B per row tile.
            b_read_bits = B_bits * self._row_blocks(M)
            # Weight-stationary walks K in `k_tiles` passes and the bit-planes
            # in `qbit` more, and every pass but the last recirculates a full
            # 32-bit partial-sum matrix through the accumulator.  At prefill
            # shapes that is `k_tiles * qbit` = 128 round trips, and it
            # dominates the lump ~4:1 over the activations -- which is exactly
            # what parked prefill in SS16(c).
            acc_read_bits = C_accum_bits * (k_tiles - 1) * qbit
            out_write_bits = C_accum_bits
            acc_write_bits = C_accum_bits * (k_tiles * qbit - 1)

        elif mode == "FPE_OS":
            m_tiles = math.ceil(M / hw.FPE_array_size)
            n_tiles = math.ceil(N / hw.FPE_array_size)
            B_bits_fpe = batch_size * K * N * 16   # FPE uses FP16 internally
            a_read_bits = A_bits * n_tiles
            b_read_bits = B_bits_fpe * m_tiles
            out_write_bits = C_accum_bits

        elif mode == "TENDER":
            m_tiles = math.ceil(M / hw.FPE_array_size)
            n_tiles = math.ceil(N / hw.FPE_array_size)
            a_read_bits = A_bits * n_tiles
            b_read_bits = B_bits * m_tiles
            out_write_bits = C_accum_bits

        sram_read_bits = a_read_bits + b_read_bits + acc_read_bits
        sram_write_bits = out_write_bits + acc_write_bits

        # ---- DRAM reads ----
        # Tracked as (bits, contiguous-run-bytes) so each component can be
        # rounded up to a burst independently: they have very different access
        # shapes, and lumping them together would average that away.
        # (bits, contiguous-run-bytes, covering-region-bytes); the third is 0
        # when no covering region applies, which disables the clamp.
        read_parts: List[Tuple[int, int, int]] = []
        kv_resident_bytes = 0
        # Effective KV batch for GQA (defaults to batch_size for MHA)
        eff_kv_batch = kv_batch_size if kv_batch_size > 0 else batch_size

        if (is_decode
                and op_type in (OperationType.QK_MATMUL, OperationType.ATTN_V_MATMUL)
                and kv_len > 0):
            # Decode attention: read cached KV from DRAM (scales with num_kv_heads)
            head_dim = K if op_type == OperationType.QK_MATMUL else N
            kv_prev = max(0, kv_len - 1)
            # One run is however many consecutive entries the reader takes in
            # one go, times the per-entry width.
            run_entries = min(kv_prev, self._kv_dram_run_entries(kv_prev))
            kv_bits_total = eff_kv_batch * kv_prev * head_dim * kv_bits
            # Whatever is held on chip between steps is not re-read.
            resident = self._kv_resident_bytes(kv_bits_total // 8)
            kv_resident_bytes = resident
            kv_logical = kv_bits_total // 8 - resident
            read_parts.append((kv_logical * 8,
                               self._kv_dram_run_bytes(run_entries, head_dim,
                                                       kv_bits),
                               self._kv_covering_bytes(kv_logical, head_dim,
                                                       kv_bits)))

        elif (self.hw.prefill_kv_dram_read
                and not is_decode
                and op_type in (OperationType.QK_MATMUL,
                                OperationType.ATTN_V_MATMUL)
                and kv_len > 0):
            # Prefill attention reads the K/V it just wrote back from DRAM.
            # The original model charged nothing here -- the decode branch above
            # is gated on `is_decode` and the AW branch below is an `elif`, so a
            # prefill AA op reached neither.  That made prefill assert K/V were
            # resident while asserting the (much larger) score matrix was not.
            head_dim = K if op_type == OperationType.QK_MATMUL else N
            run_entries = min(kv_len, self._kv_dram_run_entries(kv_len))
            kv_bits_total = eff_kv_batch * kv_len * head_dim * kv_bits
            resident = self._kv_resident_bytes(kv_bits_total // 8)
            kv_resident_bytes = resident
            kv_logical = kv_bits_total // 8 - resident
            read_parts.append((kv_logical * 8,
                               self._kv_dram_run_bytes(run_entries, head_dim,
                                                       kv_bits),
                               self._kv_covering_bytes(kv_logical, head_dim,
                                                       kv_bits)))

        elif compute_mode == ComputeMode.AW:
            # AW operations: weights loaded from DRAM.  Dense and contiguous by
            # default; the hooks let an activation-sparsity study say how
            # fragmented a masked read really is.
            read_parts.append((
                B_bits,
                self._aw_weight_run_bytes(op_type, B_bits // 8),
                self._aw_weight_covering_bytes(op_type, B_bits // 8),
            ))

        # Attn·V reads attention scores (A operand) back from DRAM, contiguous.
        # `K` is kv_len here, so it is the same width the QK write tested.
        if op_type == OperationType.ATTN_V_MATMUL and not self._score_staged(K):
            score_bits = batch_size * M * K * act_bits
            read_parts.append((score_bits, score_bits // 8, 0))

        dram_read_bits = sum(bits for bits, _, _ in read_parts)
        dram_read_eff = sum(
            self._dram_effective_bytes(bits // 8, run, cap)
            for bits, run, cap in read_parts
        )
        # Writes are contiguous in every path modelled so far.
        dram_write_eff = self._dram_effective_bytes(
            dram_write_bits // 8, dram_write_bits // 8)

        # Convert bits → bytes
        return {
            "dram_read_eff":  dram_read_eff,
            "dram_write_eff": dram_write_eff,
            "kv_resident_bytes": kv_resident_bytes,
            "dram_read":  dram_read_bits  // 8,
            "dram_write": dram_write_bits // 8,
            "sram_read":  sram_read_bits  // 8,
            "sram_write": sram_write_bits // 8,
            "sram_read_a":    a_read_bits    // 8,
            "sram_read_b":    b_read_bits    // 8,
            "sram_acc_read":  acc_read_bits  // 8,
            "sram_acc_write": acc_write_bits // 8,
            "sram_write_out": out_write_bits // 8,
        }

    # ---- Peak SRAM calculation ----------------------------------------------

    def _calculate_peak_sram(
        self, M: int, K: int, N: int,
        compute_mode: ComputeMode, mode: str,
        batch_size: int, sram_batch: int = 1,
    ) -> int:
        """Estimate peak SRAM footprint (bytes) for one operation.

        This is the minimum SRAM capacity required to hold the working set
        of one tile/round of computation simultaneously:
          - Input A tile (or full A if held in SRAM)
          - Input B tile
          - Output C accumulator tile

        Batch residency differs by operand, and the difference is the point:

          - **Projections and FFN** (`ComputeMode.AW`) are issued as a single
            GEMM with `M = batch x seq_len`, so batch is already a dimension
            here: A and C scale with it, while the weight tile does not.  That
            is the correct picture -- weights are shared across the batch, and
            sharing them is the whole reason to batch.
          - **Attention** (`ComputeMode.AA`) is issued once per (batch, head)
            with batch folded into `batch_size`, which this function
            historically ignored -- so one instance was resident regardless of
            batch.  `sram_batch` (the true batch, heads excluded) restores it
            under `hw.sram_batch_model == "concurrent"`: batch elements are
            co-resident, heads still run back-to-back.

        `sram_batch` defaults to 1, and "sequential" ignores it, so the
        original per-instance behaviour is what you get unless asked otherwise.
        """
        hw = self.hw
        # Attention instances that must be resident together.  Only AA is
        # affected: AW already carries batch in M (see above), so scaling it
        # here as well would count the same batch twice.
        if compute_mode == ComputeMode.AA and hw.sram_batch_model == "concurrent":
            batch_resident = max(1, sram_batch)
        else:
            batch_resident = 1
        act_bits = hw.act_bits
        accum_bits = hw.accumulate_bits
        qbit = hw.weight_bits if compute_mode == ComputeMode.AW else hw.kv_cache_bits

        # Rows of A on chip at once.  `hw.sram_m_tile = 0` (the default) makes
        # this the whole `M`, which is the original assumption -- and the one
        # outright bug this model had, since a prefill GEMM's `M` is the whole
        # sequence.  C is produced a row block at a time and follows it.
        m_res = self._resident_rows(M)
        A_bytes = m_res * K * act_bits // 8

        if mode in ("LUT_OS", "LUT_OS_V"):
            col_tile = min(N, hw.array_n * self.NUM_RAC)
            B_tile = K * col_tile * qbit // 8
            C_tile = m_res * col_tile * accum_bits // 8
            per_instance = A_bytes + B_tile + C_tile

        elif mode == "LUT_WS":
            k_eff = math.ceil(K / self.MU)
            row_tile = min(k_eff, hw.array_m) * self.MU
            col_tile = min(N, hw.array_n * self.NUM_RAC)
            B_tile = row_tile * col_tile * qbit // 8
            C_tile = m_res * col_tile * accum_bits // 8
            per_instance = A_bytes + B_tile + C_tile

        elif mode == "FPE_OS":
            B_tile = K * min(N, hw.FPE_array_size) * 16 // 8   # FP16
            C_tile = min(m_res, hw.FPE_array_size) * min(N, hw.FPE_array_size) * accum_bits // 8
            per_instance = A_bytes + B_tile + C_tile

        elif mode == "TENDER":
            B_tile = K * min(N, hw.FPE_array_size) * qbit // 8
            C_tile = min(m_res, hw.FPE_array_size) * min(N, hw.FPE_array_size) * accum_bits // 8
            per_instance = A_bytes + B_tile + C_tile

        elif mode == "VPU":
            B_bytes = K * N * qbit // 8
            C_bytes = m_res * N * accum_bits // 8
            per_instance = A_bytes + B_bytes + C_bytes

        else:
            per_instance = A_bytes  # Fallback: at least A

        return per_instance * batch_resident

    def _row_blocks(self, M: int) -> int:
        """How many row blocks of A the schedule walks (1 = untiled).

        `hw.sram_m_tile` is the number of A rows held on chip at once; 0 means
        all of them, which is the default and reproduces every published
        number.  See `HardwareConfig.sram_m_tile`.
        """
        mt = self.hw.sram_m_tile
        if mt <= 0 or mt >= M:
            return 1
        return math.ceil(M / mt)

    def _resident_rows(self, M: int) -> int:
        """How many rows of A are on chip at once."""
        mt = self.hw.sram_m_tile
        return M if mt <= 0 else min(M, mt)

    def _column_tiles(self, N: int, mode: str) -> int:
        """Number of column tiles the operation loops over.

        This is the loop across which A is reused, so it is also the number of
        times A must be re-read if it cannot stay resident.
        """
        hw = self.hw
        if mode in ("LUT_OS", "LUT_OS_V", "LUT_WS"):
            return math.ceil(N / (hw.array_n * self.NUM_RAC))
        if mode in ("FPE_OS", "TENDER"):
            return math.ceil(N / hw.FPE_array_size)
        return 1   # VPU processes the whole operand

    def _apply_sram_capacity(
        self, M: int, K: int, N: int,
        compute_mode: ComputeMode, mode: str,
        batch_size: int, peak_bytes: int,
    ) -> Tuple[bool, int]:
        """Spill policy v1.  Returns (overflow, extra DRAM read bytes).

        `_calculate_peak_sram` models the working set as A + B_tile + C_tile:
        B and C are already charged per tile, so A is the one operand the
        footprint assumes stays resident for the whole operation.  When the
        working set does not fit, A is the thing that has to go, and it is
        re-read from DRAM once per column tile instead of once per operation.

        Deliberately *not* modelled: re-tiling.  Shrinking the tile to fit
        would change the cycle count as well as the traffic, and the cycle
        model is out of scope here.  Two consequences worth stating plainly:

          - If a single B_tile already exceeds capacity (LUT_OS_V at long
            context, where the KV tile grows with kv_len), spilling A does not
            actually rescue the operation and this returns a charge that is a
            lower bound.  `sram_overflow` is still set, so the shortfall is
            visible rather than silent.
          - A single-column-tile operation overflows at zero charge for the
            same reason.

        The charge is the cost *under the modelled loop nest* (A resident, B
        streamed per tile).  A scheduler free to re-tile M would hold A-tiles
        and re-read the smaller operand instead, paying less -- so on a large
        A this reads high, and the flag, not the byte count, is the load-
        bearing output.  `sram_overflow` is the honest signal ("this
        configuration does not fit"); `sram_refetch_bytes` is a first-order
        cost, not a full spill model.
        """
        cap = self.hw.sram_capacity_kb * 1024
        if cap <= 0 or peak_bytes <= cap:
            return False, 0

        A_bytes = batch_size * M * K * self.hw.act_bits // 8
        n_tiles = self._column_tiles(N, mode)
        return True, A_bytes * (n_tiles - 1)

    # ---- Energy calculation -------------------------------------------------

    def _calculate_compute_energy(
        self, M: int, K: int, N: int,
        batch_size: int, qbit: int,
        base_mode: str, resolved_mode: str,
        is_decode: bool,
    ) -> float:
        """Dispatch to the appropriate compute energy model."""
        hw = self.hw

        if base_mode == "OMNI":
            omni_mode = "OS" if is_decode else "WS"
            return omni_energy_model.omni_compute_energy(
                array_m=hw.array_m, array_n=hw.array_n,
                M=M, K=K, N=N, batch_size=batch_size,
                qbit=qbit, mode=omni_mode,
            )

        elif resolved_mode == "LUT_WS":
            return ws_energy_model.ws_compute_energy(
                array_m=hw.array_m, array_n=hw.array_n,
                M=M, K=K, N=N, batch_size=batch_size, qbit=qbit,
            )

        elif resolved_mode == "FPE_OS":
            return fpe_os_energy_model.fpe_os_compute_energy(
                fpe_array_size=hw.FPE_array_size,
                M=M, K=K, N=N, batch_size=batch_size,
            )

        elif resolved_mode == "TENDER":
            return tender_energy_model.tender_compute_energy(
                fpe_array_size=hw.FPE_array_size,
                M=M, K=K, N=N, batch_size=batch_size,
            )

        elif resolved_mode == "LUT_OS":
            return os_energy_model.os_compute_energy(
                array_m=hw.array_m, array_n=hw.array_n,
                M=M, K=K, N=N, batch_size=batch_size, qbit=qbit,
            )

        elif resolved_mode == "LUT_OS_V":
            return os_v_energy_model.os_v_compute_energy(
                array_m=hw.array_m, array_n=hw.array_n,
                M=M, K=K, N=N, batch_size=batch_size, qbit=qbit,
            )

        return 0.0

    def _calculate_memory_energy(self, mem_access: dict) -> dict:
        """Calculate DRAM and SRAM energy from byte-level access counts.

        DRAM energy is charged on the *effective* bytes: moving a burst costs
        its energy whether or not every byte in it was wanted.  Note that
        `dram_power_model.dram_energy` already models 1024 B rows and 8 B
        bursts internally, so this is the coarser, access-pattern-level
        rounding layered on top of that -- the two describe different things
        (which bytes get fetched vs. what fetching them costs), and the
        effective count is the right input to the latter.
        """
        dram_read = mem_access.get("dram_read_eff", mem_access["dram_read"])
        dram_write = mem_access.get("dram_write_eff", mem_access["dram_write"])
        return {
            "dram_read":  dram_power_model.dram_energy(dram_read,  is_write=False),
            "dram_write": dram_power_model.dram_energy(dram_write, is_write=True),
            "sram_read":  sram_power_model.sram_energy(mem_access["sram_read"],  is_write=False),
            "sram_write": sram_power_model.sram_energy(mem_access["sram_write"], is_write=True),
        }

    # ---- Non-GEMM Operation Simulation --------------------------------------

    def _simulate_rmsnorm(self, num_tokens: int, d_model: int, 
                          op_type: NonGEMMOpType) -> NonGEMMMetrics:
        """Simulate RMSNorm operation.
        
        RMSNorm(x) = x * rsqrt(mean(x^2) + eps) * gamma
        
        Per token (d_model elements):
          - square: d_model ops
          - reduce_sum: d_model ops (for mean)
          - rsqrt: 1 op (per token, amortized as d_model for VPU)
          - mul: 2*d_model ops (scale by rsqrt, then by gamma)
        """
        hw = self.hw
        act_bits = hw.act_bits
        num_elements = num_tokens * d_model
        
        # VPU cycles: process d_model elements per token
        # For each token: square -> reduce -> rsqrt -> mul -> mul
        # Assume pipelined: dominant is element-wise ops
        vpu_ops_per_token = d_model * 4  # square + reduce + mul + mul (rsqrt amortized)
        total_vpu_ops = num_tokens * vpu_ops_per_token
        cycles = math.ceil(total_vpu_ops / hw.vpu_width)
        
        # Memory: read input, write output (same size)
        sram_read_bytes = num_elements * act_bits // 8
        sram_write_bytes = num_elements * act_bits // 8
        
        metrics = NonGEMMMetrics(
            op_type=op_type,
            num_elements=num_elements,
            cycles=cycles,
            ops_square=num_elements,
            ops_reduce_sum=num_elements,
            ops_rsqrt=num_tokens,  # One rsqrt per token
            ops_mul=num_elements * 2,  # Scale and gamma
            sram_read=sram_read_bytes,
            sram_write=sram_write_bytes,
        )
        
        # Calculate SRAM energy
        metrics.sram_read_energy = sram_power_model.sram_energy(sram_read_bytes, is_write=False)
        metrics.sram_write_energy = sram_power_model.sram_energy(sram_write_bytes, is_write=True)
        
        return metrics

    def _simulate_rope(self, num_tokens: int, d_model: int) -> NonGEMMMetrics:
        """Simulate Rotary Position Embedding (RoPE).
        
        Applied to Q and K projections. For each token:
          x_rot = x * cos(θ) + rotate(x) * sin(θ)
        
        Per element (applied to all d_model elements):
          - sin: 1 (typically precomputed, but we count lookup)
          - cos: 1
          - mul: 2 (x*cos, rotate(x)*sin)
          - add: 1
        """
        hw = self.hw
        act_bits = hw.act_bits
        num_elements = num_tokens * d_model
        
        # VPU cycles for RoPE
        vpu_ops_per_element = 5  # sin + cos + 2*mul + add
        total_vpu_ops = num_elements * vpu_ops_per_element
        cycles = math.ceil(total_vpu_ops / hw.vpu_width)
        
        sram_read_bytes = num_elements * act_bits // 8
        sram_write_bytes = num_elements * act_bits // 8
        
        metrics = NonGEMMMetrics(
            op_type=NonGEMMOpType.ROPE,
            num_elements=num_elements,
            cycles=cycles,
            ops_sin=num_elements,
            ops_cos=num_elements,
            ops_mul=num_elements * 2,
            ops_add=num_elements,
            sram_read=sram_read_bytes,
            sram_write=sram_write_bytes,
        )
        
        metrics.sram_read_energy = sram_power_model.sram_energy(sram_read_bytes, is_write=False)
        metrics.sram_write_energy = sram_power_model.sram_energy(sram_write_bytes, is_write=True)
        
        return metrics

    def _simulate_attn_scale(self, num_tokens: int, d_model: int) -> NonGEMMMetrics:
        """Simulate attention scaling: Q * (1 / sqrt(d_k)).
        
        Single multiplication per element.
        """
        hw = self.hw
        act_bits = hw.act_bits
        num_elements = num_tokens * d_model
        
        cycles = math.ceil(num_elements / hw.vpu_width)
        
        sram_read_bytes = num_elements * act_bits // 8
        sram_write_bytes = num_elements * act_bits // 8
        
        metrics = NonGEMMMetrics(
            op_type=NonGEMMOpType.ATTN_SCALE,
            num_elements=num_elements,
            cycles=cycles,
            ops_mul=num_elements,
            sram_read=sram_read_bytes,
            sram_write=sram_write_bytes,
        )
        
        metrics.sram_read_energy = sram_power_model.sram_energy(sram_read_bytes, is_write=False)
        metrics.sram_write_energy = sram_power_model.sram_energy(sram_write_bytes, is_write=True)
        
        return metrics

    def _simulate_softmax(self, batch_heads: int, q_len: int, kv_len: int) -> NonGEMMMetrics:
        """Simulate softmax over attention scores.
        
        Shape: [batch * num_heads, q_len, kv_len]
        
        Per row (kv_len elements):
          - reduce_max: kv_len ops
          - sub: kv_len ops (x - max)
          - exp: kv_len ops
          - reduce_sum: kv_len ops
          - div: kv_len ops
        """
        hw = self.hw
        act_bits = hw.act_bits
        
        num_rows = batch_heads * q_len
        num_elements = num_rows * kv_len
        
        # VPU ops: 5 passes over the data
        vpu_ops_per_row = kv_len * 5
        total_vpu_ops = num_rows * vpu_ops_per_row
        cycles = math.ceil(total_vpu_ops / hw.vpu_width)
        
        sram_read_bytes = num_elements * act_bits // 8
        sram_write_bytes = num_elements * act_bits // 8
        
        metrics = NonGEMMMetrics(
            op_type=NonGEMMOpType.SOFTMAX,
            num_elements=num_elements,
            cycles=cycles,
            ops_reduce_max=num_elements,
            ops_sub=num_elements,
            ops_exp=num_elements,
            ops_reduce_sum=num_elements,
            ops_div=num_elements,
            sram_read=sram_read_bytes,
            sram_write=sram_write_bytes,
        )
        
        metrics.sram_read_energy = sram_power_model.sram_energy(sram_read_bytes, is_write=False)
        metrics.sram_write_energy = sram_power_model.sram_energy(sram_write_bytes, is_write=True)
        
        return metrics

    def _simulate_residual_add(self, num_tokens: int, d_model: int,
                                op_type: NonGEMMOpType) -> NonGEMMMetrics:
        """Simulate residual addition: x + sublayer_output."""
        hw = self.hw
        act_bits = hw.act_bits
        num_elements = num_tokens * d_model
        
        cycles = math.ceil(num_elements / hw.vpu_width)
        
        # Read two inputs, write one output
        sram_read_bytes = num_elements * act_bits // 8 * 2
        sram_write_bytes = num_elements * act_bits // 8
        
        metrics = NonGEMMMetrics(
            op_type=op_type,
            num_elements=num_elements,
            cycles=cycles,
            ops_add=num_elements,
            sram_read=sram_read_bytes,
            sram_write=sram_write_bytes,
        )
        
        metrics.sram_read_energy = sram_power_model.sram_energy(sram_read_bytes, is_write=False)
        metrics.sram_write_energy = sram_power_model.sram_energy(sram_write_bytes, is_write=True)
        
        return metrics

    def _simulate_silu(self, num_tokens: int, d_ffn: int) -> NonGEMMMetrics:
        """Simulate SiLU (Swish) activation: x * sigmoid(x).
        
        Implementation: x / (1 + exp(-x))
          - neg: 1 op
          - exp: 1 op
          - add: 1 op (1 + exp)
          - div: 1 op (or recip + mul)
          - mul: 1 op (x * sigmoid)
        
        Or with dedicated sigmoid:
          - sigmoid: 1 op
          - mul: 1 op
        """
        hw = self.hw
        act_bits = hw.act_bits
        num_elements = num_tokens * d_ffn
        
        # Using composed implementation (neg + exp + add + recip + mul)
        vpu_ops_per_element = 5
        total_vpu_ops = num_elements * vpu_ops_per_element
        cycles = math.ceil(total_vpu_ops / hw.vpu_width)
        
        sram_read_bytes = num_elements * act_bits // 8
        sram_write_bytes = num_elements * act_bits // 8
        
        metrics = NonGEMMMetrics(
            op_type=NonGEMMOpType.SILU,
            num_elements=num_elements,
            cycles=cycles,
            ops_neg=num_elements,
            ops_exp=num_elements,
            ops_add=num_elements,
            ops_recip=num_elements,
            ops_mul=num_elements,
            sram_read=sram_read_bytes,
            sram_write=sram_write_bytes,
        )
        
        metrics.sram_read_energy = sram_power_model.sram_energy(sram_read_bytes, is_write=False)
        metrics.sram_write_energy = sram_power_model.sram_energy(sram_write_bytes, is_write=True)
        
        return metrics

    def _simulate_gated_mul(self, num_tokens: int, d_ffn: int) -> NonGEMMMetrics:
        """Simulate gated multiplication: gate * up_projection.
        
        Used in SwiGLU-style FFN: output = SiLU(gate) * up
        """
        hw = self.hw
        act_bits = hw.act_bits
        num_elements = num_tokens * d_ffn
        
        cycles = math.ceil(num_elements / hw.vpu_width)
        
        # Read two inputs (gate, up), write one output
        sram_read_bytes = num_elements * act_bits // 8 * 2
        sram_write_bytes = num_elements * act_bits // 8
        
        metrics = NonGEMMMetrics(
            op_type=NonGEMMOpType.GATED_MUL,
            num_elements=num_elements,
            cycles=cycles,
            ops_mul=num_elements,
            sram_read=sram_read_bytes,
            sram_write=sram_write_bytes,
        )
        
        metrics.sram_read_energy = sram_power_model.sram_energy(sram_read_bytes, is_write=False)
        metrics.sram_write_energy = sram_power_model.sram_energy(sram_write_bytes, is_write=True)
        
        return metrics

    def _simulate_embedding_lookup(self, num_tokens: int, d_model: int) -> NonGEMMMetrics:
        """Simulate token embedding lookup.
        
        This is primarily a memory operation (DRAM read).
        """
        hw = self.hw
        act_bits = hw.act_bits
        num_elements = num_tokens * d_model
        
        # Embedding lookup is memory-bound, minimal compute
        cycles = math.ceil(num_elements / hw.vpu_width)  # Just copying
        
        sram_read_bytes = 0  # Embeddings come from DRAM
        sram_write_bytes = num_elements * act_bits // 8
        
        metrics = NonGEMMMetrics(
            op_type=NonGEMMOpType.EMBEDDING,
            num_elements=num_elements,
            cycles=cycles,
            sram_read=sram_read_bytes,
            sram_write=sram_write_bytes,
        )
        
        metrics.sram_write_energy = sram_power_model.sram_energy(sram_write_bytes, is_write=True)
        
        return metrics

    def _simulate_lm_head_softmax(self, batch_size: int, vocab_size: int = 32000) -> NonGEMMMetrics:
        """Simulate final softmax over vocabulary for next token prediction.
        
        Shape: [batch_size, vocab_size]
        Typically vocab_size ~ 32000 for LLaMA
        """
        hw = self.hw
        act_bits = hw.act_bits
        
        num_elements = batch_size * vocab_size
        
        # Same as attention softmax but over vocab dimension
        vpu_ops_per_row = vocab_size * 5  # max, sub, exp, sum, div
        total_vpu_ops = batch_size * vpu_ops_per_row
        cycles = math.ceil(total_vpu_ops / hw.vpu_width)
        
        sram_read_bytes = num_elements * act_bits // 8
        sram_write_bytes = num_elements * act_bits // 8
        
        metrics = NonGEMMMetrics(
            op_type=NonGEMMOpType.LM_HEAD_SOFTMAX,
            num_elements=num_elements,
            cycles=cycles,
            ops_reduce_max=num_elements,
            ops_sub=num_elements,
            ops_exp=num_elements,
            ops_reduce_sum=num_elements,
            ops_div=num_elements,
            sram_read=sram_read_bytes,
            sram_write=sram_write_bytes,
        )
        
        metrics.sram_read_energy = sram_power_model.sram_energy(sram_read_bytes, is_write=False)
        metrics.sram_write_energy = sram_power_model.sram_energy(sram_write_bytes, is_write=True)
        
        return metrics

    # ---- Results I/O --------------------------------------------------------

    def save_results(self, results: SimulationResults, output_path: str):
        """Save simulation results to a human-readable text file."""
        txt_path = (output_path.replace(".json", ".txt")
                    if output_path.endswith(".json") else output_path + ".txt")

        total = results.get_total_metrics()

        with open(txt_path, 'w') as f:
            self._write_header(f)
            self._write_hw_config(f)
            self._write_overall_summary(f, total, results.peak_sram_bytes)
            self._write_kv_cache_summary(f, results)
            self._write_phase_summary(f, "PREFILL PHASE", results.prefill,
                                      results.prefill.get_total_metrics())
            self._write_phase_summary(f, "DECODE PHASE", results.decode,
                                      results.decode.get_total_metrics())

        print(f"Results saved to: {txt_path}")

    # ---- Formatting helpers -------------------------------------------------

    def _write_header(self, f):
        f.write("=" * 80 + "\n")
        f.write("SIMULATION RESULTS\n")
        f.write("=" * 80 + "\n\n")

    def _write_hw_config(self, f):
        f.write("Hardware Configuration:\n")
        f.write("-" * 80 + "\n")
        for key, value in self.hw.to_dict().items():
            f.write(f"  {key:20s}: {value}\n")
        f.write("\n")

    def _write_overall_summary(self, f, total: OperationMetrics, peak_sram: int):
        f.write("Overall Summary:\n")
        f.write("-" * 80 + "\n")
        self._write_metrics_block(f, total)
        f.write(f"  Peak SRAM:           {peak_sram:,} bytes ({peak_sram / 1024:.2f} KB)\n")
        cap_kb = self.hw.sram_capacity_kb
        if cap_kb > 0:
            verdict = "OVERFLOWS" if total.sram_overflow else "fits"
            f.write(f"  SRAM capacity:       {cap_kb:,} KB -> {verdict}\n")
            if total.sram_refetch_bytes:
                f.write(f"  Spill re-fetch:      {total.sram_refetch_bytes:,} bytes "
                        f"(included in DRAM read)\n")
        f.write("\n")

    def _write_kv_cache_summary(self, f, results: SimulationResults):
        kv = results.get_kv_cache_writes()
        f.write("KV Cache Writes:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Prefill K values:    {kv['prefill_k_values']:>18,}   "
                f"({kv['prefill_k_bytes'] / (1024**3):.2f} GB)\n")
        f.write(f"  Prefill V values:    {kv['prefill_v_values']:>18,}   "
                f"({kv['prefill_v_bytes'] / (1024**3):.2f} GB)\n")
        f.write(f"  Decode  K values:    {kv['decode_k_values']:>18,}   "
                f"({kv['decode_k_bytes'] / (1024**3):.2f} GB)\n")
        f.write(f"  Decode  V values:    {kv['decode_v_values']:>18,}   "
                f"({kv['decode_v_bytes'] / (1024**3):.2f} GB)\n")
        f.write(f"  Total K values:      {kv['k_values']:>18,}   "
                f"({kv['k_bytes'] / (1024**3):.2f} GB)\n")
        f.write(f"  Total V values:      {kv['v_values']:>18,}   "
                f"({kv['v_bytes'] / (1024**3):.2f} GB)\n")
        f.write(f"  Total KV values:     {kv['total_values']:>18,}   "
                f"({kv['total_bytes'] / (1024**3):.2f} GB)\n")
        f.write("\n")

    def _write_phase_summary(self, f, title: str, phase: PhaseMetrics,
                             phase_total: OperationMetrics):
        f.write("=" * 80 + "\n")
        f.write(f"{title}\n")
        f.write("=" * 80 + "\n\n")

        # Phase totals
        f.write("Phase Totals:\n")
        f.write("-" * 80 + "\n")
        self._write_metrics_block(f, phase_total)
        f.write(f"  Peak SRAM:           {phase.peak_sram_bytes:,} bytes "
                f"({phase.peak_sram_bytes / 1024:.2f} KB)\n")
        f.write("\n")

        # AA operations
        aa_total = phase.get_aa_total()
        f.write("AA Operations Summary:\n")
        f.write("-" * 80 + "\n")
        self._write_metrics_block(f, aa_total)
        f.write("\n")
        self._write_op_details(f, phase.aa_ops, phase, ComputeMode.AA)

        # AW operations
        aw_total = phase.get_aw_total()
        f.write("AW Operations Summary:\n")
        f.write("-" * 80 + "\n")
        self._write_metrics_block(f, aw_total)
        f.write("\n")
        self._write_op_details(f, phase.aw_ops, phase, ComputeMode.AW)

    def _write_metrics_block(self, f, m: OperationMetrics):
        """Write a standard metrics block (reused for overall / phase / op totals)."""
        f.write(f"  Total Cycles:        {m.cycles:,}\n")
        f.write(f"  Total FLOPs:         {m.flops:,}\n")
        f.write(f"  Avg Utilization:     {m.utilization:.2%}\n")
        f.write(f"  DRAM Read:           {m.dram_read:,} bytes ({m.dram_read / (1024**3):.2f} GB)\n")
        if self.hw.dram_burst_bytes > 0:
            moved = m.dram_read_eff + m.dram_write_eff
            asked = m.dram_read + m.dram_write
            waste = moved / asked if asked else 1.0
            f.write(f"  DRAM moved (burst):  {moved:,} bytes "
                    f"({moved / (1024**3):.2f} GB, {waste:.3f}x requested)\n")
        f.write(f"  DRAM Write:          {m.dram_write:,} bytes ({m.dram_write / (1024**3):.2f} GB)\n")
        f.write(f"  SRAM Read:           {m.sram_read:,} bytes ({m.sram_read / (1024**3):.2f} GB)\n")
        f.write(f"  SRAM Write:          {m.sram_write:,} bytes ({m.sram_write / (1024**3):.2f} GB)\n")
        f.write(f"  Compute Energy:      {m.compute_energy:.2f} J\n")
        f.write(f"  DRAM Read Energy:    {m.dram_read_energy:.2f} J\n")
        f.write(f"  DRAM Write Energy:   {m.dram_write_energy:.2f} J\n")
        f.write(f"  DRAM Energy:         {m.dram_energy:.2f} J\n")
        f.write(f"  SRAM Read Energy:    {m.sram_read_energy:.2f} J\n")
        f.write(f"  SRAM Write Energy:   {m.sram_write_energy:.2f} J\n")
        f.write(f"  SRAM Energy:         {m.sram_energy:.2f} J\n")
        f.write(f"  Total Energy:        {m.total_energy:.2f} J\n")

    def _write_op_details(self, f,
                          ops_dict: Dict[OperationType, List[OperationMetrics]],
                          phase: PhaseMetrics, mode: ComputeMode):
        """Write per-operation-type detail lines."""
        for op_type, op_list in ops_dict.items():
            op_total = phase.get_operation_total(op_type, mode)
            first_shape = op_list[0].shape if op_list else None
            f.write(f"  {op_type.value:20s}:\n")
            f.write(f"    First exec shape:   {first_shape}\n")
            f.write(f"    Executions:        {len(op_list):6d}\n")
            f.write(f"    Cycles:            {op_total.cycles:12,}\n")
            f.write(f"    FLOPs:             {op_total.flops:12,}\n")
            f.write(f"    Avg Utilization:   {op_total.utilization:6.2%}\n")
            f.write(f"    DRAM Read:         {op_total.dram_read:12,} bytes "
                    f"({op_total.dram_read / (1024**2):8.2f} MB)\n")
            f.write(f"    DRAM Write:        {op_total.dram_write:12,} bytes "
                    f"({op_total.dram_write / (1024**2):8.2f} MB)\n")
            f.write(f"    SRAM Read:         {op_total.sram_read:12,} bytes "
                    f"({op_total.sram_read / (1024**2):8.2f} MB)\n")
            f.write(f"    SRAM Write:        {op_total.sram_write:12,} bytes "
                    f"({op_total.sram_write / (1024**2):8.2f} MB)\n")
            f.write(f"    Compute Energy:    {op_total.compute_energy:12.2f} J\n")
            f.write(f"    DRAM Read Energy:  {op_total.dram_read_energy:12.2f} J\n")
            f.write(f"    DRAM Write Energy: {op_total.dram_write_energy:12.2f} J\n")
            f.write(f"    DRAM Energy:       {op_total.dram_energy:12.2f} J\n")
            f.write(f"    SRAM Read Energy:  {op_total.sram_read_energy:12.2f} J\n")
            f.write(f"    SRAM Write Energy: {op_total.sram_write_energy:12.2f} J\n")
            f.write(f"    SRAM Energy:       {op_total.sram_energy:12.2f} J\n")
            f.write(f"    Total Energy:      {op_total.total_energy:12.2f} J\n")
            f.write(f"    Peak SRAM:         {op_total.peak_sram_bytes:12,} bytes "
                    f"({op_total.peak_sram_bytes / 1024:8.2f} KB)\n")
        f.write("\n")

    # ---- Roofline Model Analysis --------------------------------------------

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        """Format time in seconds to a human-readable string."""
        if seconds == 0:
            return "0.00 ns"
        elif abs(seconds) < 1e-6:
            return f"{seconds * 1e9:.2f} ns"
        elif abs(seconds) < 1e-3:
            return f"{seconds * 1e6:.2f} us"
        elif abs(seconds) < 1:
            return f"{seconds * 1e3:.4f} ms"
        else:
            return f"{seconds:.4f} s"

    def _roofline_analyze_ops(self, ops_dict, mode_label, freq, dram_bw):
        """Compute roofline metrics per operation type.

        For each operation type, iterates through every individual execution,
        computes compute-time (cycles / freq) vs memory-time (DRAM bytes / BW),
        and classifies each as memory-bound or compute-bound.

        Returns a list of summary dicts, one per operation type.
        """
        results = []
        for op_type, op_list in ops_dict.items():
            if not op_list:
                continue

            total_ct = 0.0   # total compute time (s)
            total_mt = 0.0   # total memory time (s)
            total_rt = 0.0   # total roofline time (s)
            total_flops = 0
            total_dram = 0
            mem_bound_n = 0

            for m in op_list:
                ct = m.cycles / freq
                dram_bytes = m.dram_read_eff + m.dram_write_eff
                mt = dram_bytes / dram_bw if dram_bw > 0 else 0.0
                rt = self._op_roofline_time(m, freq, dram_bw)

                total_ct += ct
                total_mt += mt
                total_rt += rt
                total_flops += m.flops
                total_dram += dram_bytes
                if mt > ct:
                    mem_bound_n += 1

            ai = total_flops / total_dram if total_dram > 0 else float('inf')
            bound = "Memory" if mem_bound_n > len(op_list) // 2 else "Compute"

            results.append({
                'op_type': op_type.value,
                'mode': mode_label,
                'shape': op_list[0].shape,
                'count': len(op_list),
                'compute_time': total_ct,
                'memory_time': total_mt,
                'roofline_time': total_rt,
                'ai': ai,
                'bound': bound,
            })
        return results

    def _write_roofline_phase(self, f, title, phase, freq, dram_bw):
        """Write roofline table for one phase.  Returns total roofline time (s)."""
        f.write("=" * 120 + "\n")
        f.write(f"{title}\n")
        f.write("=" * 120 + "\n\n")

        all_ops = []
        all_ops.extend(self._roofline_analyze_ops(phase.aw_ops, "AW", freq, dram_bw))
        all_ops.extend(self._roofline_analyze_ops(phase.aa_ops, "AA", freq, dram_bw))

        # Table header
        hdr = (f"  {'Operation':<18} {'Mode':<5} {'Shape':<24} {'Count':>7} "
               f"{'Comp Time':>14} {'Mem Time':>14} {'Roofline':>14} "
               f"{'AI (F/B)':>10}  {'Bound':<8}")
        f.write(hdr + "\n")
        f.write("  " + "-" * (len(hdr) - 2) + "\n")

        total_rt = 0.0
        total_ct = 0.0
        total_mt = 0.0

        for d in all_ops:
            shape_str = f"({d['shape'][0]}, {d['shape'][1]}, {d['shape'][2]})"
            ai_str = f"{d['ai']:.1f}" if d['ai'] != float('inf') else "inf"
            f.write(
                f"  {d['op_type']:<18} {d['mode']:<5} {shape_str:<24} {d['count']:>7} "
                f"{self._fmt_time(d['compute_time']):>14} "
                f"{self._fmt_time(d['memory_time']):>14} "
                f"{self._fmt_time(d['roofline_time']):>14} "
                f"{ai_str:>10}  {d['bound']:<8}\n")
            total_rt += d['roofline_time']
            total_ct += d['compute_time']
            total_mt += d['memory_time']

        f.write("  " + "-" * (len(hdr) - 2) + "\n")
        f.write(f"  {'TOTAL':<18} {'':<5} {'':<24} {'':>7} "
                f"{self._fmt_time(total_ct):>14} "
                f"{self._fmt_time(total_mt):>14} "
                f"{self._fmt_time(total_rt):>14}\n")

        return total_rt

    def compute_roofline_latency(self, results: SimulationResults,
                                  workload: WorkloadConfig,
                                  dram_bandwidth_gbps: float | None = None,
                                  ) -> Tuple[float, float]:
        """Compute TTFT and TPOT using the roofline model.

        For each individual GEMM execution, the roofline time is
        ``max(compute_time, memory_time)`` where:
          - compute_time = cycles / freq
          - memory_time  = (dram_read + dram_write) / dram_bandwidth

        Args:
            dram_bandwidth_gbps: Override DRAM bandwidth (GB/s).
                Defaults to ``self.hw.dram_bandwidth_gbps``.

        Returns:
            (ttft, tpot) in seconds.
        """
        if dram_bandwidth_gbps is None:
            dram_bandwidth_gbps = self.hw.dram_bandwidth_gbps
        dram_bw = dram_bandwidth_gbps * 1e9
        freq = self.hw.freq_mhz * 1e6

        def _phase_roofline_time(phase: PhaseMetrics) -> float:
            # Pipelining is a property of the whole phase, not of one operation,
            # so the group handed over here is every GEMM in it.
            return self._roofline_time_over(
                (m for ops_dict in (phase.aa_ops, phase.aw_ops)
                 for op_list in ops_dict.values() for m in op_list),
                freq, dram_bw)

        ttft = _phase_roofline_time(results.prefill)
        decode_total = _phase_roofline_time(results.decode)
        num_steps = max(1, workload.output_tokens - 1)
        tpot = decode_total / num_steps

        return ttft, tpot

    def compute_roofline_latency_breakdown(self, results: SimulationResults,
                                            workload: WorkloadConfig,
                                            dram_bandwidth_gbps: float | None = None,
                                            ) -> dict:
        """Compute TTFT and TPOT broken down by AA-GEMM, AW-GEMM, and non-GEMM.

        Returns:
            dict with keys: ttft_aa, ttft_aw, ttft_non_gemm, ttft_total,
                            tpot_aa, tpot_aw, tpot_non_gemm, tpot_total
            All values in seconds.
        """
        if dram_bandwidth_gbps is None:
            dram_bandwidth_gbps = self.hw.dram_bandwidth_gbps
        dram_bw = dram_bandwidth_gbps * 1e9
        freq = self.hw.freq_mhz * 1e6

        def _phase_breakdown(phase: PhaseMetrics) -> dict:
            aa_time = 0.0
            aw_time = 0.0
            non_gemm_time = 0.0

            # AA ops
            for op_list in phase.aa_ops.values():
                for m in op_list:
                    aa_time += self._op_roofline_time(m, freq, dram_bw)

            # AW ops
            for op_list in phase.aw_ops.values():
                for m in op_list:
                    aw_time += self._op_roofline_time(m, freq, dram_bw)

            # Under "pipelined" a per-category split is not measurable -- the
            # whole point is that one category's memory hides behind another's
            # compute, so the time does not belong to either.  Both categories
            # are therefore scaled by the phase's pipelined/serial ratio, which
            # keeps the parts summing to the total `compute_roofline_latency`
            # reports.  Treat the split as an **attribution, not a measurement**;
            # the totals are the trustworthy output.
            if self.hw.overlap_model == "pipelined":
                serial_gemm = aa_time + aw_time
                if serial_gemm > 0:
                    scale = self._roofline_time_over(
                        (m for d in (phase.aa_ops, phase.aw_ops)
                         for lst in d.values() for m in lst),
                        freq, dram_bw) / serial_gemm
                    aa_time *= scale
                    aw_time *= scale

            # Non-GEMM ops (VPU) — cycle-bound only, no DRAM access
            for m in phase.non_gemm_ops:
                non_gemm_time += m.cycles / freq

            return {
                'aa': aa_time,
                'aw': aw_time,
                'non_gemm': non_gemm_time,
                'total': aa_time + aw_time + non_gemm_time,
            }

        prefill = _phase_breakdown(results.prefill)
        decode = _phase_breakdown(results.decode)
        num_steps = max(1, workload.output_tokens - 1)

        return {
            'ttft_aa': prefill['aa'],
            'ttft_aw': prefill['aw'],
            'ttft_non_gemm': prefill['non_gemm'],
            'ttft_total': prefill['total'],
            'tpot_aa': decode['aa'] / num_steps,
            'tpot_aw': decode['aw'] / num_steps,
            'tpot_non_gemm': decode['non_gemm'] / num_steps,
            'tpot_total': decode['total'] / num_steps,
        }

    def save_roofline_report(self, results: SimulationResults,
                              model: ModelConfig, workload: WorkloadConfig,
                              output_path: str,
                              dram_bandwidth_gbps: float | None = None):
        """Generate a roofline-model analysis report with TTFT / TPOT metrics.

        This is written to *output_path* and is completely independent of the
        standard simulation report produced by ``save_results``.

        Args:
            dram_bandwidth_gbps: Override DRAM bandwidth (GB/s).
                Defaults to ``self.hw.dram_bandwidth_gbps``.
        """
        if dram_bandwidth_gbps is None:
            dram_bandwidth_gbps = self.hw.dram_bandwidth_gbps
        dram_bw = dram_bandwidth_gbps * 1e9   # bytes/s
        freq = self.hw.freq_mhz * 1e6         # Hz

        with open(output_path, 'w') as f:
            # ---- Header ----
            f.write("=" * 120 + "\n")
            f.write("ROOFLINE MODEL ANALYSIS REPORT\n")
            f.write("=" * 120 + "\n\n")

            f.write("Configuration:\n")
            f.write("-" * 120 + "\n")
            f.write(f"  DRAM Bandwidth:    {dram_bandwidth_gbps} GB/s\n")
            f.write(f"  Clock Frequency:   {self.hw.freq_mhz} MHz\n")
            f.write(f"  AW Dataflow:       {self.hw.AW_mode}\n")
            f.write(f"  AA Dataflow:       {self.hw.AA_mode}\n")
            f.write(f"  Model Layers:      {model.num_layers}\n")
            f.write(f"  Input Tokens:      {workload.input_tokens}\n")
            f.write(f"  Output Tokens:     {workload.output_tokens}\n")
            if workload.flash_block_size > 0:
                f.write(f"  FlashAttention:    block_size={workload.flash_block_size}\n")
            f.write("\n")

            # ---- Prefill ----
            ttft = self._write_roofline_phase(
                f, "PREFILL PHASE (TTFT)", results.prefill, freq, dram_bw)
            f.write(f"\n  >>> TTFT (Time-To-First-Token): {self._fmt_time(ttft)} <<<\n\n")

            # ---- Decode ----
            decode_total = self._write_roofline_phase(
                f, "DECODE PHASE (TPOT)", results.decode, freq, dram_bw)
            num_steps = max(1, workload.output_tokens - 1)
            tpot = decode_total / num_steps
            f.write(f"\n  Total Decode Time:    {self._fmt_time(decode_total)}\n")
            f.write(f"  Decode Steps:         {num_steps}\n")
            f.write(f"  >>> TPOT (Time-Per-Output-Token): {self._fmt_time(tpot)} <<<\n\n")

            # ---- Summary ----
            total_time = ttft + decode_total
            f.write("=" * 120 + "\n")
            f.write("SUMMARY\n")
            f.write("=" * 120 + "\n")
            f.write(f"  TTFT:              {self._fmt_time(ttft)}\n")
            f.write(f"  TPOT:              {self._fmt_time(tpot)}\n")
            f.write(f"  Total Inference:   {self._fmt_time(total_time)}\n")
            total_tokens = workload.output_tokens
            throughput = total_tokens / total_time if total_time > 0 else 0
            f.write(f"  Throughput:        {throughput:.1f} tokens/s\n")

        print(f"Roofline report saved to: {output_path}")

    # ---- Full Pipeline Report (GEMM + Non-GEMM) -----------------------------

    def save_full_pipeline_report(self, results: SimulationResults,
                                   model: ModelConfig, workload: WorkloadConfig,
                                   output_path: str,
                                   vpu_config: 'vpu_energy_model.VPUEnergyConfig | None' = None,
                                   dram_bandwidth_gbps: float | None = None):
        """Generate a full pipeline report including GEMM and non-GEMM operations.
        
        This report includes:
          - GEMM operations (existing roofline analysis)
          - Non-GEMM operations (VPU-based: RMSNorm, Softmax, SiLU, RoPE, etc.)
          - Full pipeline latency (TTFT, TPOT) considering all operations
          - Energy breakdown with VPU energy requirements
        
        Args:
            vpu_config: VPU energy configuration. If None, uses default (placeholder values).
            dram_bandwidth_gbps: Override DRAM bandwidth (GB/s).
        """
        if dram_bandwidth_gbps is None:
            dram_bandwidth_gbps = self.hw.dram_bandwidth_gbps
        if vpu_config is None:
            vpu_config = vpu_energy_model.DEFAULT_VPU_ENERGY
        
        # Calculate VPU energy for all non-GEMM operations
        for m in results.prefill.non_gemm_ops:
            m.calculate_energy(vpu_config)
        for m in results.decode.non_gemm_ops:
            m.calculate_energy(vpu_config)
            
        dram_bw = dram_bandwidth_gbps * 1e9   # bytes/s
        freq = self.hw.freq_mhz * 1e6         # Hz
        vpu_width = self.hw.vpu_width

        with open(output_path, 'w') as f:
            # ---- Header ----
            f.write("=" * 120 + "\n")
            f.write("FULL PIPELINE ANALYSIS REPORT (GEMM + Non-GEMM Operations)\n")
            f.write("=" * 120 + "\n\n")

            # ---- Configuration ----
            f.write("Configuration:\n")
            f.write("-" * 120 + "\n")
            f.write(f"  DRAM Bandwidth:    {dram_bandwidth_gbps} GB/s\n")
            f.write(f"  Clock Frequency:   {self.hw.freq_mhz} MHz\n")
            f.write(f"  VPU Width:         {vpu_width} lanes\n")
            f.write(f"  AW Dataflow:       {self.hw.AW_mode}\n")
            f.write(f"  AA Dataflow:       {self.hw.AA_mode}\n")
            f.write(f"  Model Layers:      {model.num_layers}\n")
            f.write(f"  d_model:           {model.d_model}\n")
            f.write(f"  d_ffn:             {model.d_ffn}\n")
            f.write(f"  Input Tokens:      {workload.input_tokens}\n")
            f.write(f"  Output Tokens:     {workload.output_tokens}\n")
            if workload.flash_block_size > 0:
                f.write(f"  FlashAttention:    block_size={workload.flash_block_size}\n")
            f.write("\n")

            # ---- Prefill Phase ----
            f.write("=" * 120 + "\n")
            f.write("PREFILL PHASE\n")
            f.write("=" * 120 + "\n\n")
            
            # GEMM operations
            gemm_prefill_time = self._write_roofline_phase(
                f, "GEMM Operations (Prefill)", results.prefill, freq, dram_bw)
            
            # Non-GEMM operations
            non_gemm_prefill_time, non_gemm_prefill_summary = self._write_non_gemm_phase(
                f, "Non-GEMM Operations (Prefill)", results.prefill, freq, vpu_config)
            
            total_prefill_time = gemm_prefill_time + non_gemm_prefill_time
            f.write(f"\n  GEMM Time:       {self._fmt_time(gemm_prefill_time)}\n")
            f.write(f"  Non-GEMM Time:   {self._fmt_time(non_gemm_prefill_time)}\n")
            f.write(f"  >>> TTFT (Total): {self._fmt_time(total_prefill_time)} <<<\n\n")

            # ---- Decode Phase ----
            f.write("=" * 120 + "\n")
            f.write("DECODE PHASE\n")
            f.write("=" * 120 + "\n\n")
            
            # GEMM operations
            gemm_decode_time = self._write_roofline_phase(
                f, "GEMM Operations (Decode)", results.decode, freq, dram_bw)
            
            # Non-GEMM operations
            non_gemm_decode_time, non_gemm_decode_summary = self._write_non_gemm_phase(
                f, "Non-GEMM Operations (Decode)", results.decode, freq, vpu_config)
            
            total_decode_time = gemm_decode_time + non_gemm_decode_time
            num_steps = max(1, workload.output_tokens - 1)
            gemm_tpot = gemm_decode_time / num_steps
            non_gemm_tpot = non_gemm_decode_time / num_steps
            total_tpot = total_decode_time / num_steps
            
            f.write(f"\n  GEMM Time:       {self._fmt_time(gemm_decode_time)} (TPOT: {self._fmt_time(gemm_tpot)})\n")
            f.write(f"  Non-GEMM Time:   {self._fmt_time(non_gemm_decode_time)} (TPOT: {self._fmt_time(non_gemm_tpot)})\n")
            f.write(f"  >>> TPOT (Total): {self._fmt_time(total_tpot)} <<<\n\n")

            # ---- Energy Breakdown ----
            f.write("=" * 120 + "\n")
            f.write("ENERGY BREAKDOWN\n")
            f.write("=" * 120 + "\n\n")
            
            # GEMM energy
            gemm_total = results.get_total_metrics()
            f.write("GEMM Energy:\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Compute Energy:      {gemm_total.compute_energy:.4f} J\n")
            f.write(f"  DRAM Read Energy:    {gemm_total.dram_read_energy:.4f} J\n")
            f.write(f"  DRAM Write Energy:   {gemm_total.dram_write_energy:.4f} J\n")
            f.write(f"  SRAM Energy:         {gemm_total.sram_energy:.4f} J\n")
            f.write(f"  Total GEMM Energy:   {gemm_total.total_energy:.4f} J\n\n")
            
            # Non-GEMM energy summary
            f.write("Non-GEMM Energy (VPU):\n")
            f.write("-" * 80 + "\n")
            self._write_non_gemm_energy_summary(f, results, vpu_config)
            
            # ---- VPU Energy Requirements ----
            f.write("\n")
            f.write("=" * 120 + "\n")
            f.write("VPU ENERGY CHARACTERIZATION REQUIREMENTS\n")
            f.write("=" * 120 + "\n\n")
            self._write_vpu_energy_requirements(f, results)
            
            # ---- Summary ----
            f.write("\n")
            f.write("=" * 120 + "\n")
            f.write("FULL PIPELINE SUMMARY\n")
            f.write("=" * 120 + "\n\n")
            
            total_inference_time = total_prefill_time + total_decode_time
            throughput = workload.output_tokens / total_inference_time if total_inference_time > 0 else 0
            
            f.write(f"  TTFT (Time-To-First-Token):     {self._fmt_time(total_prefill_time)}\n")
            f.write(f"    - GEMM:                       {self._fmt_time(gemm_prefill_time)}\n")
            f.write(f"    - Non-GEMM:                   {self._fmt_time(non_gemm_prefill_time)}\n")
            f.write(f"\n")
            f.write(f"  TPOT (Time-Per-Output-Token):   {self._fmt_time(total_tpot)}\n")
            f.write(f"    - GEMM:                       {self._fmt_time(gemm_tpot)}\n")
            f.write(f"    - Non-GEMM:                   {self._fmt_time(non_gemm_tpot)}\n")
            f.write(f"\n")
            f.write(f"  Total Inference Time:           {self._fmt_time(total_inference_time)}\n")
            f.write(f"  Throughput:                     {throughput:.1f} tokens/s\n")
            f.write(f"\n")
            f.write(f"  Total GEMM Energy:              {gemm_total.total_energy:.4f} J\n")
            # Calculate total non-GEMM energy
            all_non_gemm = results.prefill.non_gemm_ops + results.decode.non_gemm_ops
            total_non_gemm_energy = sum(m.total_energy for m in all_non_gemm)
            f.write(f"  Total Non-GEMM Energy:          {total_non_gemm_energy:.4f} J\n")
            total_energy = gemm_total.total_energy + total_non_gemm_energy
            f.write(f"  Total Energy:                   {total_energy:.4f} J\n")

        print(f"Full pipeline report saved to: {output_path}")
        
        return {
            'ttft_gemm': gemm_prefill_time,
            'ttft_non_gemm': non_gemm_prefill_time,
            'ttft_total': total_prefill_time,
            'tpot_gemm': gemm_tpot,
            'tpot_non_gemm': non_gemm_tpot,
            'tpot_total': total_tpot,
            'total_time': total_inference_time,
            'throughput': throughput,
            'gemm_energy': gemm_total.total_energy,
            'non_gemm_energy': total_non_gemm_energy,
            'total_energy': total_energy,
        }

    def _write_non_gemm_phase(self, f, title: str, phase: PhaseMetrics, 
                               freq: float, vpu_config) -> tuple:
        """Write non-GEMM operation summary for a phase.
        
        Returns (total_time_seconds, summary_dict).
        """
        f.write("\n")
        f.write("-" * 80 + "\n")
        f.write(f"{title}\n")
        f.write("-" * 80 + "\n")
        
        # Aggregate by operation type
        summary = _aggregate_non_gemm_metrics(phase.non_gemm_ops)
        
        total_cycles = 0
        total_sram_bytes = 0
        
        # Table header
        f.write(f"\n  {'Operation':<25} {'Count':>8} {'Elements':>14} {'Cycles':>14} {'Time':>14}\n")
        f.write("  " + "-" * 85 + "\n")
        
        for op_type, data in summary.items():
            cycles = data['cycles']
            time_s = cycles / freq
            total_cycles += cycles
            total_sram_bytes += data['sram_read'] + data['sram_write']
            
            f.write(f"  {op_type.value:<25} {data['count']:>8} {data['num_elements']:>14,} "
                   f"{cycles:>14,} {self._fmt_time(time_s):>14}\n")
        
        total_time = total_cycles / freq
        f.write("  " + "-" * 85 + "\n")
        f.write(f"  {'TOTAL':<25} {'':<8} {'':<14} {total_cycles:>14,} {self._fmt_time(total_time):>14}\n")
        
        return total_time, summary

    def _write_non_gemm_energy_summary(self, f, results: SimulationResults, 
                                        vpu_config) -> None:
        """Write non-GEMM energy summary."""
        # Combine prefill and decode non-GEMM ops
        all_non_gemm = results.prefill.non_gemm_ops + results.decode.non_gemm_ops
        summary = _aggregate_non_gemm_metrics(all_non_gemm)
        
        # Calculate energy for each operation type
        total_sram_energy = 0.0
        
        total_vpu_energy = 0.0
        
        f.write(f"\n  {'Operation':<25} {'SRAM Energy (J)':>15} {'VPU Energy (J)':>15} {'Total (J)':>15}\n")
        f.write("  " + "-" * 75 + "\n")
        
        for op_type, data in summary.items():
            sram_energy = data['sram_read_energy'] + data['sram_write_energy']
            vpu_energy = data['compute_energy']
            total_sram_energy += sram_energy
            total_vpu_energy += vpu_energy
            f.write(f"  {op_type.value:<25} {sram_energy:>15.6f} {vpu_energy:>15.6f} {sram_energy + vpu_energy:>15.6f}\n")
        
        total_non_gemm_energy = total_sram_energy + total_vpu_energy
        f.write("  " + "-" * 75 + "\n")
        f.write(f"  {'TOTAL':<25} {total_sram_energy:>15.6f} {total_vpu_energy:>15.6f} {total_non_gemm_energy:>15.6f}\n")

    def _write_vpu_energy_requirements(self, f, results: SimulationResults) -> None:
        """Write the VPU energy characterization requirements."""
        # Combine all non-GEMM ops
        all_non_gemm = results.prefill.non_gemm_ops + results.decode.non_gemm_ops
        
        # Sum up all primitive operation counts
        totals = {
            'ops_add': 0, 'ops_sub': 0, 'ops_mul': 0, 'ops_div': 0,
            'ops_exp': 0, 'ops_rsqrt': 0, 'ops_sqrt': 0, 'ops_recip': 0,
            'ops_sin': 0, 'ops_cos': 0, 'ops_sigmoid': 0,
            'ops_reduce_sum': 0, 'ops_reduce_max': 0, 'ops_square': 0,
            'ops_neg': 0, 'ops_max': 0
        }
        
        for m in all_non_gemm:
            totals['ops_add'] += m.ops_add
            totals['ops_sub'] += m.ops_sub
            totals['ops_mul'] += m.ops_mul
            totals['ops_div'] += m.ops_div
            totals['ops_exp'] += m.ops_exp
            totals['ops_rsqrt'] += m.ops_rsqrt
            totals['ops_sqrt'] += m.ops_sqrt
            totals['ops_recip'] += m.ops_recip
            totals['ops_sin'] += m.ops_sin
            totals['ops_cos'] += m.ops_cos
            totals['ops_sigmoid'] += m.ops_sigmoid
            totals['ops_reduce_sum'] += m.ops_reduce_sum
            totals['ops_reduce_max'] += m.ops_reduce_max
            totals['ops_square'] += m.ops_square
            totals['ops_neg'] += m.ops_neg
            totals['ops_max'] += m.ops_max
        
        f.write("To calculate VPU energy, provide energy per operation (in Joules) for:\n\n")
        
        f.write(f"  {'Primitive Operation':<25} {'Total Count':>18}  Energy/Op (J)    Formula\n")
        f.write("  " + "-" * 90 + "\n")
        
        ops_info = [
            ('vec_add', totals['ops_add'], 'Vector addition (a + b)'),
            ('vec_sub', totals['ops_sub'], 'Vector subtraction (a - b)'),
            ('vec_mul', totals['ops_mul'], 'Vector multiplication (a * b)'),
            ('vec_div', totals['ops_div'], 'Vector division (a / b)'),
            ('vec_exp', totals['ops_exp'], 'Exponential (exp(x))'),
            ('vec_rsqrt', totals['ops_rsqrt'], 'Reciprocal sqrt (1/sqrt(x))'),
            ('vec_sqrt', totals['ops_sqrt'], 'Square root (sqrt(x))'),
            ('vec_recip', totals['ops_recip'], 'Reciprocal (1/x)'),
            ('vec_sin', totals['ops_sin'], 'Sine (sin(x)) [RoPE]'),
            ('vec_cos', totals['ops_cos'], 'Cosine (cos(x)) [RoPE]'),
            ('vec_sigmoid', totals['ops_sigmoid'], 'Sigmoid (1/(1+exp(-x)))'),
            ('vec_reduce_sum', totals['ops_reduce_sum'], 'Sum reduction'),
            ('vec_reduce_max', totals['ops_reduce_max'], 'Max reduction'),
            ('vec_square', totals['ops_square'], 'Square (x^2)'),
            ('vec_neg', totals['ops_neg'], 'Negate (-x)'),
            ('vec_max', totals['ops_max'], 'Element-wise max'),
        ]
        
        for name, count, desc in ops_info:
            if count > 0:
                f.write(f"  {name:<25} {count:>18,}  [E_{name}]       Total = count × E\n")
        
        f.write("\n")
        f.write("  Total VPU Energy = Σ (count_i × energy_per_op_i)\n\n")
        f.write("  Fill in energy values in: vpu_energy_model.py -> VPUEnergyConfig\n")


# ============================================================================
# Main
# ============================================================================

def main():
    # ---- Model config ----
    from model_configs import get_model_config
    model = get_model_config("OPT-6.7B")  

    # ---- Workload config ----
    workload = WorkloadConfig(
        batch_size=1,
        input_tokens=2048,
        output_tokens=512,
        flash_block_size=256,     # 0 = standard attention; try 256 for FlashAttention
    )

    # ---- Hardware configs (uncomment one) ----

    # FIGLUT + FPE 16*2
    # hw = HardwareConfig(
    #     array_m=16, array_n=2, replication=4, FPE_array_size=64,
    #     act_bits=16, accumulate_bits=32, weight_bits=4, kv_cache_bits=16,
    #     AW_mode="LUT_WS", AA_mode="FPE_OS",
    # )

    # FIGLUT + FPE
    # hw = HardwareConfig(
    #     array_m=32, array_n=4, replication=1, FPE_array_size=64,
    #     act_bits=16, accumulate_bits=32, weight_bits=4, kv_cache_bits=16,
    #     AW_mode="LUT_WS", AA_mode="FPE_OS",
    # )

    # Pure FPE
    # hw = HardwareConfig(
    #     array_m=64, array_n=64, FPE_array_size=64,
    #     act_bits=16, accumulate_bits=32, weight_bits=4, kv_cache_bits=16,
    #     AW_mode="FPE_OS", AA_mode="FPE_OS",
    # )

    # OMNI
    # hw = HardwareConfig(
    #     array_m=32, array_n=4,
    #     act_bits=16, accumulate_bits=32, weight_bits=4, kv_cache_bits=4,
    #     AW_mode="OMNI", AA_mode="OMNI",
    # )

    # TENDER
    hw = HardwareConfig(
        array_m=64, array_n=64, FPE_array_size=64,
        act_bits=8, accumulate_bits=32, weight_bits=8, kv_cache_bits=8,
        AW_mode="TENDER", AA_mode="TENDER", tender_m1_opt=True
    )
    
    # hw = HardwareConfig(
    #     array_m=32, array_n=4,
    #     act_bits=16, accumulate_bits=32, weight_bits=4, kv_cache_bits=4,
    #     AW_mode="LUT_OS_V", AA_mode="LUT_OS_V",
    # )

    # ---- Run simulation ----
    simulator = Simulator(hw)
    results = simulator.simulate(model, workload)
    simulator.save_results(results, "simulation_results")
    simulator.save_roofline_report(results, model, workload, "roofline_report.txt")
    
    # Generate full pipeline report (GEMM + Non-GEMM)
    pipeline_stats = simulator.save_full_pipeline_report(
        results, model, workload, "full_pipeline_report.txt"
    )

    # ---- Quick summary ----
    total = results.get_total_metrics()
    ttft, tpot = simulator.compute_roofline_latency(results, workload)
    print(f"\n{'='*60}")
    print(f"GEMM Operations:")
    print(f"  Total Energy:    {total.total_energy:.4f} J")
    print(f"  Total Cycles:    {total.cycles:,}")
    print(f"  Utilization:     {total.utilization:.2%}")
    print(f"  Peak SRAM:       {results.peak_sram_bytes:,} bytes "
          f"({results.peak_sram_bytes / 1024:.1f} KB)")
    print(f"\nFull Pipeline (GEMM + Non-GEMM):")
    print(f"  TTFT (total):    {Simulator._fmt_time(pipeline_stats['ttft_total'])}")
    print(f"    - GEMM:        {Simulator._fmt_time(pipeline_stats['ttft_gemm'])}")
    print(f"    - Non-GEMM:    {Simulator._fmt_time(pipeline_stats['ttft_non_gemm'])}")
    print(f"  TPOT (total):    {Simulator._fmt_time(pipeline_stats['tpot_total'])}")
    print(f"    - GEMM:        {Simulator._fmt_time(pipeline_stats['tpot_gemm'])}")
    print(f"    - Non-GEMM:    {Simulator._fmt_time(pipeline_stats['tpot_non_gemm'])}")
    print(f"  Throughput:      {pipeline_stats['throughput']:.1f} tokens/s")
    if workload.flash_block_size > 0:
        print(f"\n  FlashAttention:  block_size={workload.flash_block_size}")
    print(f"{'='*60}")
    print("\nSimulation completed successfully!")
    print("See 'full_pipeline_report.txt' for detailed analysis.")


if __name__ == "__main__":
    main()
