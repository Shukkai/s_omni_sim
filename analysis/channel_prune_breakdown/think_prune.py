"""
ThinK-style KV *channel* pruning on top of the stock simulator.

`simulator/` is untouched, exactly as with `cycle_units.py` and `kv_budget.py`:
this is a subclass that shortens the `head_dim` axis of the two decode
attention GEMMs.

The distinction from `kv_budget.py` matters, and is the whole point of this
study.  Eviction shortens `kv_len`; ThinK shortens `head_dim`.  Those two axes
land in completely different places in `LUT_OS_V`:

    qk_matmul     = (q_len, head_dim, kv_len)   -> head_dim is K, the REDUCTION dim
    attn_v_matmul = (q_len, kv_len, head_dim)   -> head_dim is N, the OUTPUT dim

and the OS-V cost model is

    k_eff     = ceil(K / MU)
    n_tiles   = ceil(N / (array_n * NUM_RAC))       = ceil(N / 128)
    rounds    = ceil(n_tiles / array_m)             = ceil(n_tiles / 32)
    per_round = 3 (LGU) + k_eff + 1 + array_n (fill/drain) + 2 (accum)

`per_round` has no N term at all, and for any `d_v_ret` in 1..128 both
`n_tiles` and `rounds` are 1.  So pruning Value channels cannot change the
`attn_v` cycle count -- it can only idle array columns.  Pruning Key channels
does shrink `k_eff`, but `qk_matmul` is a small share of decode.

Both phases are modelled, because they use different dataflows and the null
arrives in each for a different reason:

    prefill (LUT_WS)   n_tiles = ceil(N / 128), and the round cost is
                       M + array_n + array_m + 1 + 2 -- again no N term, but
                       here the retained channels map to columns of a single
                       128-wide tile, so pruning to 77 idles 37.5% of them.
    decode  (LUT_OS_V) as above: 128 of 4096 lanes were live to begin with.

Scope, stated explicitly so the numbers are not over-read:

  * Prefill pruning is HYPOTHETICAL and off unless `prune_prefill` is set.
    ThinK's channel selection is query-driven and observed after the prompt is
    processed, so real ThinK cannot act during prefill -- the same decode-only
    convention `kv_budget.py` uses.  It is modelled anyway to show that the
    cycle null is a property of the LUT dataflows rather than of the phase.
  * `o_proj` stays at `d_model`.  ThinK-V zeroes pruned Value channels rather
    than physically renumbering the head output, so the projection's reduction
    dimension is unchanged.  This is the conservative choice: it does not
    credit ThinK with a saving on a stage it does not touch.
  * Channel *selection* cost is excluded, matching the compaction study.
"""

import math
import os
import sys
from typing import Tuple

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'cycle_breakdown'))

from cycle_units import UnitAwareSimulator                      # noqa: E402
from simulator import OperationType, PhaseMetrics, Simulator    # noqa: E402


class ThinKSimulator(UnitAwareSimulator):
    """`UnitAwareSimulator` with ThinK channel pruning applied during decode.

    Args:
        d_k_ret: retained Key channels per head.  0 or None = dense.
        d_v_ret: retained Value channels per head.  0 or None = dense.
        prune_prefill: also prune the prefill attention GEMMs.  Hypothetical
            for ThinK (selection is query-driven and only available after the
            prompt); see the module docstring.

    Rewriting the operation *shape* is enough to drive every downstream
    number, because cycles, DRAM traffic, SRAM traffic, peak footprint and
    utilization are all derived from it inside `Simulator._simulate_matmul`.
    In particular `_calculate_memory_access` reads the cached KV as
    ``eff_kv_batch * kv_prev * head_dim * kv_bits`` with
    ``head_dim = K if QK_MATMUL else N``, so the DRAM saving falls out of the
    same one-line rewrite as the cycle result.
    """

    def __init__(self, hw, d_k_ret: int = 0, d_v_ret: int = 0,
                 prune_prefill: bool = False, **kwargs):
        super().__init__(hw, **kwargs)
        self.d_k_ret = d_k_ret or 0
        self.d_v_ret = d_v_ret or 0
        self.prune_prefill = prune_prefill

    def _prune_shape(self, op_type, shape, is_decode) -> Tuple[int, int, int]:
        if not is_decode and not self.prune_prefill:
            return shape
        M, K, N = shape
        if op_type == OperationType.QK_MATMUL and self.d_k_ret:
            return (M, min(K, self.d_k_ret), N)
        if op_type == OperationType.ATTN_V_MATMUL and self.d_v_ret:
            return (M, K, min(N, self.d_v_ret))
        return shape

    def _simulate_matmul(self, op_type, compute_mode, shape,
                         batch_size: int = 1, is_decode: bool = False,
                         **kwargs):
        return super()._simulate_matmul(
            op_type, compute_mode,
            self._prune_shape(op_type, shape, is_decode),
            batch_size=batch_size, is_decode=is_decode, **kwargs)

    def _simulate_flash_attention(self, *args, **kwargs):
        if self.d_k_ret or self.d_v_ret:
            raise NotImplementedError(
                "ThinK pruning is modelled on the standard (non-fused) "
                "attention path only, where qk_matmul and attn_v_matmul are "
                "separate stages.  Run with flash_block_size=0.")
        return super()._simulate_flash_attention(*args, **kwargs)


# ============================================================================
# Array occupancy  --  the cost ThinK-V actually pays
# ============================================================================

def array_lanes(hw, mode: str = 'LUT_OS_V') -> int:
    """Output channels one round of `mode` can drive at once.

    The two phases differ, and that is why the same pruning ratio costs
    different amounts of array in each:

      LUT_OS_V (decode)  a round spans the whole array -- `array_m` rows, each
          fed by its own weight FIFO (Omni-LUT Fig. 8c), each covering
          `array_n * NUM_RAC` columns.  32 * 4 * 32 = 4096, the constant the
          simulator calls `LANES_EQUIV`.
      LUT_WS (prefill)   rows carry the reduction dimension instead, so the
          output width of a round is one `array_n * NUM_RAC` = 128 tile.
    """
    if mode == 'LUT_WS':
        return hw.array_n * Simulator.NUM_RAC
    return hw.array_m * hw.array_n * Simulator.NUM_RAC


def occupancy(hw, d_ret: int, mode: str = 'LUT_OS_V') -> dict:
    """Lane occupancy of one `attn_v` round retaining `d_ret` channels.

    Two nested levels of waste, and the rounding is the smaller one:

      * within-round granularity -- a RAC group covers `array_n` columns, so
        77 retained channels physically occupy 80 lanes, whatever the mode;
      * round occupancy -- against `array_lanes(mode)`.  In prefill that is a
        128-wide tile, so 77 channels leave 37.5% of it idle.  In decode it is
        the full 4096, of which `head_dim` = 128 was the most ever used.

    Returns lane counts plus both ratios, and `useful_frac`, the product,
    which is what `OperationMetrics.utilization` independently reports for the
    decode path.
    """
    lanes = array_lanes(hw, mode)
    occupied = math.ceil(d_ret / hw.array_n) * hw.array_n
    return {
        'd_ret': d_ret,
        'mode': mode,
        'lanes': lanes,
        'occupied_lanes': occupied,
        'occupied_frac': occupied / lanes,
        'in_round_frac': d_ret / occupied if occupied else 0.0,
        'useful_frac': d_ret / lanes,
    }


def stage_utilization(phase: PhaseMetrics, op_type: OperationType) -> float:
    """Reported utilization of one attention stage.

    `Simulator._simulate_matmul` sets ``utilization = (flops/2) /
    (cycles * LANES_EQUIV)``, i.e. useful MACs over issued lane-cycles.  Every
    execution of a given stage in a step is identical (the same layer metrics
    replicated `num_layers` times), so the first record is representative.
    """
    ops = phase.aa_ops.get(op_type) or []
    return ops[0].utilization if ops else 0.0


# ============================================================================
# Analytic cross-checks
# ============================================================================

def osv_per_round(hw, K: int, M: int = 1) -> int:
    """`LUT_OS_V` cycles per round -- the formula, with no N anywhere in it."""
    k_eff = math.ceil(K / Simulator.MU)
    if M == 1:
        return 3 + k_eff + 1 + hw.array_n + 2
    return 3 + k_eff + hw.array_m + hw.array_n + 2


def ws_per_round(hw, M: int) -> int:
    """`LUT_WS` cycles per round -- the prefill formula, also free of N."""
    return M + hw.array_n + hw.array_m + 1 + 2


def osv_rounds(hw, N: int, M: int = 1) -> int:
    """`LUT_OS_V` round count.  Equals 1 for every N <= array_n * NUM_RAC."""
    n_tiles = math.ceil(N / (hw.array_n * Simulator.NUM_RAC))
    if M == 1:
        return math.ceil(n_tiles / hw.array_m / hw.replication)
    return math.ceil(n_tiles / hw.replication)


def kv_cache_dram_bytes(model, hw, batch: int, kv_len: int,
                        d_k_ret: int, d_v_ret: int) -> dict:
    """KV bytes re-read from DRAM for one decode step, all layers.

    Mirrors the `is_decode` branch of `Simulator._calculate_memory_access`:
    the current token's KV is still in registers, so only `kv_len - 1` entries
    come back, scaled by `num_kv_heads` under GQA.  Used to cross-check the
    stage-level DRAM figures, which also carry the attention-score spill.
    """
    eff_kv_batch = batch * model.num_kv_heads
    kv_prev = max(0, kv_len - 1)
    per_layer = eff_kv_batch * kv_prev * hw.kv_cache_bits / 8
    k_bytes = per_layer * d_k_ret * model.num_layers
    v_bytes = per_layer * d_v_ret * model.num_layers
    return {'k_bytes': k_bytes, 'v_bytes': v_bytes,
            'total_bytes': k_bytes + v_bytes}


def retained_channels(head_dim: int, prune_ratio: float) -> int:
    """Channels kept at ThinK pruning ratio `prune_ratio` (their lambda).

    ThinK reports the *pruned* fraction, so lambda = 0.4 keeps 60% of the
    channels.  Everything in this study is labelled by retained channels,
    which is the quantity the hardware actually sees.
    """
    return max(1, int(round(head_dim * (1.0 - prune_ratio))))
