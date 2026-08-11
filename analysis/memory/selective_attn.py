"""
Select-without-evict KV attention (Quest / TidalDecode / NSA) on the stock
simulator.

`simulator/` is untouched, the same add-on pattern as `cycle_units.py`,
`kv_budget.py` and `think_prune.py`: a subclass that rewrites what decode
attention reads.

**Why this needed Stage 2 first.**  `kv_budget.py` deferred masked / selective
reading to "a separate mask-granularity study" because on a flat bandwidth
model it is *indistinguishable* from compacted eviction.  Both read `k` entries
instead of `n`, so both divide DRAM bytes by `n/k` and both shrink the GEMM the
same way.  The two differ in exactly two places, and neither was modelled:

  1. **Run length.**  Eviction compacts, so its `k` entries are one contiguous
     block.  Selection does not compact -- the full cache stays in DRAM and the
     reader gathers `k/page` scattered pages -- so its runs are one page long.
     That only costs anything once accesses are charged by burst, which is
     `dram_burst_bytes` (Stage 2) and `_kv_dram_run_entries`, overridden here.
  2. **Selection cost.**  Choosing which pages to read means touching metadata
     for *every* page, every token.  Quest reads a per-page min/max vector pair;
     that traffic scales with the full context, not with `k`, so it is exactly
     the term that stops selection from being free at small `k`.

So the honest claim this file exists to test is not "selection is fast" -- the
flat model already said that, wrongly -- but "how much of selection's paper
saving survives burst granularity and its own metadata reads".

**Scope, stated so the numbers are not over-read.**

  * Decode only, matching `kv_budget.py`: selection is query-driven, so it has
    nothing to act on until there is a query.
  * Page *scoring* arithmetic is not modelled, only the metadata DRAM reads it
    forces.  Scoring is a few VPU ops per page against a DRAM read per page, so
    the read is the term that matters; ignoring the arithmetic understates
    selection's cost slightly, which is the conservative direction here.
  * The summary array is assumed contiguous per head -- the favourable
    assumption for Quest, since a scattered summary layout would cost more.
  * Approximation quality is out of scope entirely.  This says what selection
    *costs*, never whether the selected pages were the right ones.
"""

import math
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'cycle_breakdown'))

from cycle_units import UnitAwareSimulator                      # noqa: E402
from simulator import ComputeMode, OperationType, PhaseMetrics  # noqa: E402


class SelectiveAttnSimulator(UnitAwareSimulator):
    """Reads `select_pages` of `ceil(kv_len / page_size)` KV pages per token.

    Args:
        page_size: KV entries per page.  Quest and NSA use 16-32; TidalDecode
            selects individual tokens, which is `page_size = 1`.
        select_pages: fixed number of pages to read.  0 = use `select_frac`.
        select_frac: fraction of pages to read, if `select_pages` is 0.
            0 = read everything (the dense baseline).
        summary_vectors: `head_dim`-wide vectors read per page per token to
            score it.  Quest reads a min and a max, so 2.  0 models selection
            as free, which is what the flat model implicitly assumed.
        compacted: if True, pretend the selected pages are contiguous.  This is
            not a real design -- it is the knob that isolates how much of the
            result is burst granularity rather than entry count.
    """

    def __init__(self, hw, page_size: int = 16, select_pages: int = 0,
                 select_frac: float = 0.0, summary_vectors: int = 0,
                 compacted: bool = False, **kwargs):
        super().__init__(hw, **kwargs)
        self.page_size = max(1, page_size)
        self.select_pages = max(0, select_pages)
        self.select_frac = select_frac
        self.summary_vectors = max(0, summary_vectors)
        self.compacted = compacted
        # Set per decode step so _calculate_memory_access can price selection
        # against the *full* context after kv_len has been clamped.
        self._full_kv_len = 0

    # ---- Selection arithmetic ----------------------------------------------

    def pages_for(self, kv_len: int) -> int:
        """Total pages covering a context of `kv_len` entries."""
        return math.ceil(kv_len / self.page_size) if kv_len > 0 else 0

    def selected_pages(self, kv_len: int) -> int:
        """Pages actually read.  0 selection parameters = read everything."""
        n_pages = self.pages_for(kv_len)
        if self.select_pages > 0:
            return min(n_pages, self.select_pages)
        if self.select_frac > 0:
            return min(n_pages, max(1, math.ceil(self.select_frac * n_pages)))
        return n_pages

    def selected_entries(self, kv_len: int) -> int:
        """KV entries the attention GEMMs actually see."""
        return min(kv_len, self.selected_pages(kv_len) * self.page_size)

    # ---- Hooks ------------------------------------------------------------

    def _kv_dram_run_entries(self, kv_prev: int) -> int:
        """One page per contiguous run -- unless the pages are compacted.

        This is the override that makes selection cost more than eviction for
        the same entry count, and it does nothing at all unless
        `hw.dram_burst_bytes` is set.
        """
        if self.compacted:
            return kv_prev
        return self.page_size

    def _simulate_transformer_step(self, metrics: PhaseMetrics, model, workload,
                                   proj_m, attn_q_len, kv_len, is_decode,
                                   token_idx=-1):
        if is_decode:
            self._full_kv_len = kv_len
            kv_len = self.selected_entries(kv_len)
        else:
            self._full_kv_len = 0
        return super()._simulate_transformer_step(
            metrics, model, workload, proj_m=proj_m, attn_q_len=attn_q_len,
            kv_len=kv_len, is_decode=is_decode, token_idx=token_idx)

    def _calculate_memory_access(self, M, K, N, compute_mode, op_type, mode,
                                 batch_size, is_decode=False, seq_len=0,
                                 kv_len=0, kv_batch_size=0) -> dict:
        mem = super()._calculate_memory_access(
            M, K, N, compute_mode, op_type, mode, batch_size,
            is_decode=is_decode, seq_len=seq_len, kv_len=kv_len,
            kv_batch_size=kv_batch_size)

        # Selection metadata: scored once per token per page over the FULL
        # context, charged to QK since that is where selection happens.  Note
        # this scales with the context, not with what was selected -- which is
        # the whole reason aggressive selection stops paying off.
        if (is_decode and self.summary_vectors > 0
                and op_type == OperationType.QK_MATMUL
                and self._full_kv_len > 0):
            head_dim = K
            eff_kv_batch = kv_batch_size if kv_batch_size > 0 else batch_size
            n_pages = self.pages_for(max(0, self._full_kv_len - 1))
            summary_bits = (eff_kv_batch * n_pages * self.summary_vectors
                            * head_dim * self.hw.kv_cache_bits)
            summary_bytes = summary_bits // 8
            # Per head the summary array is a separate compact block, so its
            # run is the whole thing -- the favourable assumption for Quest.
            run = (n_pages * self.summary_vectors * head_dim
                   * self.hw.kv_cache_bits // 8)
            mem["dram_read"] += summary_bytes
            mem["dram_read_eff"] += self._dram_effective_bytes(summary_bytes, run)

        return mem

    # ---- Reporting helper --------------------------------------------------

    def selection_summary(self, kv_len: int) -> dict:
        n_pages = self.pages_for(kv_len)
        k_pages = self.selected_pages(kv_len)
        return {
            'kv_len': kv_len,
            'page_size': self.page_size,
            'n_pages': n_pages,
            'selected_pages': k_pages,
            'selected_entries': self.selected_entries(kv_len),
            'read_frac': k_pages / n_pages if n_pages else 1.0,
        }
