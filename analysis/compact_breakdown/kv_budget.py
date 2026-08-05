"""
KV-cache budget (eviction + compaction) on top of the stock simulator.

`simulator/` is untouched, exactly as with `cycle_units.py`: this is a subclass
that clamps the decode-time `kv_len` to a budget, which is what a *compacted*
KV cache looks like to the datapath -- H2O keeps the cache physically dense by
refilling evicted slots with newly-added KV, so a budget-k cache is simply a
length-k cache.

Masked (non-compacted) eviction is deliberately NOT modelled here: it is a
different cost structure (dead groups still occupy tiles) and belongs in a
separate mask-granularity study.
"""

import math
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'cycle_breakdown'))

from cycle_units import UnitAwareSimulator          # noqa: E402
from simulator import PhaseMetrics                  # noqa: E402


# Units that are paid per round regardless of how many operands the round
# actually carries: LGU table generation, systolic fill/drain, operand issue
# and the final accumulator write-back.  These do not shrink with the budget.
FIXED_UNITS = ('lgu', 'pe_array_fill_drain', 'input_load', 'accumulator')
USEFUL_UNITS = ('pe_array_compute', 'fpe_array_compute')


class KVBudgetSimulator(UnitAwareSimulator):
    """`UnitAwareSimulator` with a compacted KV cache of at most `kv_budget`.

    Args:
        kv_budget: maximum retained KV entries during decode.  0 or None = no
            eviction (dense baseline).  Prefill is unaffected -- eviction only
            constrains what decode reads back.
    """

    def __init__(self, hw, kv_budget: int = 0, **kwargs):
        super().__init__(hw, **kwargs)
        self.kv_budget = kv_budget or 0

    def _simulate_transformer_step(self, metrics: PhaseMetrics, model, workload,
                                   proj_m, attn_q_len, kv_len, is_decode,
                                   token_idx=-1):
        if is_decode and self.kv_budget > 0:
            kv_len = min(kv_len, self.kv_budget)
        return super()._simulate_transformer_step(
            metrics, model, workload, proj_m=proj_m, attn_q_len=attn_q_len,
            kv_len=kv_len, is_decode=is_decode, token_idx=token_idx)


def attention_unit_cycles(phase: PhaseMetrics) -> dict:
    """Sum per-unit cycles over the attention (AA) ops of one phase.

    The phase-level unit breakdown mixes in the weight-stationary projection
    and FFN stages, which have their own fill/drain.  For the fixed-overhead
    knee we want the attention ops alone, since those are the only ones whose
    operand count shrinks with the KV budget.
    """
    units = {}
    for op_list in phase.aa_ops.values():
        for m in op_list:
            for unit, c in getattr(m, 'unit_cycles', {}).items():
                units[unit] = units.get(unit, 0) + c

    fixed = sum(units.get(u, 0) for u in FIXED_UNITS)
    useful = sum(units.get(u, 0) for u in USEFUL_UNITS)
    total = sum(units.values())
    return {
        'units': units,
        'fixed': fixed,
        'useful': useful,
        'total': total,
        'fixed_share': fixed / total if total else 0.0,
    }


def compaction_payback(kv_read_per_token: float, budget_frac: float,
                       prefill_kv_writeback: float = 0.0) -> dict:
    """Analytical payback for compacting a KV cache down to `budget_frac`.

    Cost is one-time: stream the cache once and write back the survivors.
    Benefit is per-token: decode re-reads the whole cache every step, so the
    evicted fraction is saved on every subsequent token.

        payback_tokens = (1 + b) / (1 - b)

    If eviction is decided during (or at the end of) prefill, the survivors are
    the only KV ever written to DRAM.  There is then no separate gather at all
    and the prefill writeback shrinks too, so the net cost is negative -- this
    is the `fused_*` result.
    """
    gather = kv_read_per_token * (1.0 + budget_frac)
    saving_per_token = kv_read_per_token * (1.0 - budget_frac)

    out = {
        'budget_frac': budget_frac,
        'gather_bytes': gather,
        'saving_bytes_per_token': saving_per_token,
        'payback_tokens': (gather / saving_per_token
                           if saving_per_token > 0 else float('inf')),
        # Fused into the prefill writeback path: no gather, and the evicted
        # KV is never written out in the first place.
        'fused_gather_bytes': 0.0,
        'fused_prefill_saving_bytes': prefill_kv_writeback * (1.0 - budget_frac),
        'fused_payback_tokens': 0.0,
    }
    return out


def budget_tokens(context: int, frac: float) -> int:
    """Retained-entry count for a budget expressed as a fraction of context."""
    return max(1, int(math.ceil(frac * context)))
