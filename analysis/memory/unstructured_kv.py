"""
Unstructured KV pruning -- what an irregular mask costs that a compacted one
does not.

**The gap this fills.**  Every KV technique measured so far assumes the
retained set is *compacted into one contiguous run*: `kv_budget.py` compacts
explicitly, `think_prune.py` rewrites `head_dim` to `d_ret` and lets the
simulator read a narrower but still solid entry, and `selective_attn.py`
gathers whole pages.  Real pruning masks are not like that.  A per-head channel
mask keeps a different 64 channels in every head; a per-head token mask keeps a
different 3% of tokens.  The retained *count* is the same and the retained
*addresses* are scattered, and on a burst-addressed DRAM those are not the same
cost.

**What decides the answer is the layout, not the mask.**  A KV entry at 4-bit
is `head_dim 128 x 4/8` = 64 B, exactly one burst (`selective_report.md` A/B).
So:

  * **token-major** (the layout the simulator assumes -- one token's channels
    contiguous, tokens sequential): a *token* mask cuts on entry boundaries, so
    every run is a whole number of bursts and scattering is free.  A *channel*
    mask cuts *inside* the entry, so a run is `group x 4/8` bytes and anything
    below the full 128 channels is sub-burst.
  * **channel-major** (transposed -- one channel's whole history contiguous):
    exactly inverted.  A channel mask now selects whole contiguous rows and is
    free; a token mask cuts inside the row and is sub-burst.

Which means the two axes want **opposite layouts**, and any layout choice makes
the other axis unstructured.  That is the result this file exists to measure,
and it is a property of this hardware's 64 B entry rather than of any
particular pruning paper.

**Head-wise is the exception** and is modelled by reducing `num_kv_heads`: each
head is its own address region in both layouts, so dropping heads is contiguous
by construction and no burst term can touch it.

**The clamp.**  A scattered read is never charged more than reading the whole
region it sits inside -- see `Simulator._dram_effective_bytes(cap_bytes=...)`.
Without it a fine mask prices *above* a dense read, which no controller would
do.  With it, fine-grained pruning degrades to exactly dense, which is the
honest statement of the failure: **the traffic saving does not go negative, it
goes to zero.**

**Scope.**
  * Decode only, matching `kv_budget.py` and `selective_attn.py`.
  * Accuracy is entirely out of scope.  This says what a mask *costs*, never
    whether it kept the right elements.  Unstructured masks exist precisely
    because they are more accurate at equal budget; the point here is to price
    that, not to dispute it.
  * Mask *storage* is charged (`mask_mode`); mask *computation* is not.
  * Cycles come from the shape rewrite, exactly as in `think_prune.py`, so the
    channel-axis cycle null carries over unchanged.
"""

import math
import os
import sys
from typing import Tuple

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'cycle_breakdown'))

from cycle_units import UnitAwareSimulator                      # noqa: E402
from simulator import OperationType, PhaseMetrics               # noqa: E402

TOKEN_MAJOR = 'token_major'
CHANNEL_MAJOR = 'channel_major'


class UnstructuredKVSimulator(UnitAwareSimulator):
    """Decode KV reads under an irregular retention mask.

    Args:
        layout: `token_major` (a token's channels contiguous, the layout the
            base simulator assumes) or `channel_major` (a channel's history
            contiguous).
        keep_tokens: fraction of `kv_len` retained.  1.0 = dense.
        keep_channels: fraction of `head_dim` retained.  1.0 = dense.
        token_group: consecutive retained tokens per run.  1 = fully
            unstructured (each retained token isolated); large = structured /
            compacted.  Only matters when `keep_tokens < 1`.
        channel_group: consecutive retained channels per run.  1 = fully
            unstructured; `head_dim` = the whole entry, i.e. compacted.  Only
            matters when `keep_channels < 1`.
        mask_mode: `none` charges no mask storage.  `static` charges one
            per-head token bitmap plus one per-head channel bitmap, re-read
            every decode step.  `per_token` charges a full `kv_len x head_dim`
            bitmap -- the cost of a mask that varies per token *and* per
            channel.
    """

    def __init__(self, hw, layout: str = TOKEN_MAJOR,
                 keep_tokens: float = 1.0, keep_channels: float = 1.0,
                 token_group: int = 1, channel_group: int = 1,
                 mask_mode: str = 'none', **kwargs):
        super().__init__(hw, **kwargs)
        if layout not in (TOKEN_MAJOR, CHANNEL_MAJOR):
            raise ValueError(f"unknown layout {layout!r}")
        if mask_mode not in ('none', 'static', 'per_token'):
            raise ValueError(f"unknown mask_mode {mask_mode!r}")
        self.layout = layout
        self.keep_tokens = keep_tokens
        self.keep_channels = keep_channels
        self.token_group = max(1, token_group)
        self.channel_group = max(1, channel_group)
        self.mask_mode = mask_mode
        self._full_kv_len = 0

    # ---- Retention arithmetic ---------------------------------------------

    @property
    def tokens_pruned(self) -> bool:
        return self.keep_tokens < 1.0

    @property
    def channels_pruned(self) -> bool:
        return self.keep_channels < 1.0

    def retained_tokens(self, kv_len: int) -> int:
        if not self.tokens_pruned or kv_len <= 0:
            return kv_len
        return max(1, math.ceil(self.keep_tokens * kv_len))

    def retained_channels(self, head_dim: int) -> int:
        if not self.channels_pruned or head_dim <= 0:
            return head_dim
        return max(1, math.ceil(self.keep_channels * head_dim))

    def _full_head_dim(self, narrowed: int) -> int:
        """Recover the unpruned `head_dim` from the narrowed QK `K`."""
        if not self.channels_pruned:
            return narrowed
        return math.ceil(narrowed / self.keep_channels)

    # ---- Hooks -------------------------------------------------------------

    def _kv_dram_run_entries(self, kv_prev: int) -> int:
        """Consecutive *entries* per run, in token-major order.

        A token mask breaks entry-to-entry contiguity, so the run collapses to
        `token_group`.  With no token mask the whole head block is one run.
        """
        if self.layout == TOKEN_MAJOR and self.tokens_pruned:
            return self.token_group
        return kv_prev

    def _kv_dram_run_bytes(self, run_entries: int, head_dim: int,
                           kv_bits: int) -> int:
        """Contiguous bytes per run, which is where the layout decides.

        `head_dim` arrives already narrowed to the retained channel count,
        because the shape rewrite below runs first.
        """
        if self.layout == TOKEN_MAJOR:
            if not self.channels_pruned:
                # Entries are whole, so runs are entry-aligned and therefore
                # burst-aligned at 4-bit.  Token scattering is free.
                return run_entries * head_dim * kv_bits // 8
            # The mask cuts inside the entry: a run is one channel group, and
            # token contiguity is irrelevant because the fragment bounds it.
            return max(1, self.channel_group * kv_bits // 8)
        # Channel-major: a run is a stretch of one channel's history.  Whole
        # rows when tokens are dense, `token_group` values when they are not.
        span = self.token_group if self.tokens_pruned else run_entries
        return max(1, span * kv_bits // 8)

    def _kv_covering_bytes(self, logical_bytes: int, head_dim: int,
                           kv_bits: int) -> int:
        """The dense region the retained elements are scattered across.

        `logical_bytes` counts only what survived the mask, so the covering
        region is that divided back out by whichever fraction was applied
        *along the fragmenting axis* -- the axis whose mask cuts inside a
        contiguous run in this layout.  The other axis is entry- or
        row-aligned, so it shrinks the region as well as the data and does not
        belong in the ratio.
        """
        frac = (self.keep_channels if self.layout == TOKEN_MAJOR
                else self.keep_tokens)
        if frac >= 1.0:
            return 0
        return int(logical_bytes / frac)

    # ---- Shape rewrite (drives cycles, SRAM and logical DRAM) --------------

    def _prune_shape(self, op_type, shape, is_decode) -> Tuple[int, int, int]:
        if not is_decode or not self.channels_pruned:
            return shape
        M, K, N = shape
        if op_type == OperationType.QK_MATMUL:
            return (M, self.retained_channels(K), N)
        if op_type == OperationType.ATTN_V_MATMUL:
            return (M, K, self.retained_channels(N))
        return shape

    def _simulate_matmul(self, op_type, compute_mode, shape,
                         batch_size: int = 1, is_decode: bool = False,
                         **kwargs):
        return super()._simulate_matmul(
            op_type, compute_mode,
            self._prune_shape(op_type, shape, is_decode),
            batch_size=batch_size, is_decode=is_decode, **kwargs)

    def _simulate_transformer_step(self, metrics: PhaseMetrics, model, workload,
                                   proj_m, attn_q_len, kv_len, is_decode,
                                   token_idx=-1):
        if is_decode:
            self._full_kv_len = kv_len
            kv_len = self.retained_tokens(kv_len)
        else:
            self._full_kv_len = 0
        return super()._simulate_transformer_step(
            metrics, model, workload, proj_m=proj_m, attn_q_len=attn_q_len,
            kv_len=kv_len, is_decode=is_decode, token_idx=token_idx)

    # ---- Mask storage ------------------------------------------------------

    def _calculate_memory_access(self, M, K, N, compute_mode, op_type, mode,
                                 batch_size, is_decode=False, seq_len=0,
                                 kv_len=0, kv_batch_size=0) -> dict:
        mem = super()._calculate_memory_access(
            M, K, N, compute_mode, op_type, mode, batch_size,
            is_decode=is_decode, seq_len=seq_len, kv_len=kv_len,
            kv_batch_size=kv_batch_size)

        if (is_decode and self.mask_mode != 'none'
                and op_type == OperationType.QK_MATMUL
                and self._full_kv_len > 0):
            eff_kv_batch = kv_batch_size if kv_batch_size > 0 else batch_size
            full_tokens = max(0, self._full_kv_len - 1)
            full_dim = self._full_head_dim(K)
            if self.mask_mode == 'static':
                # One bit per token plus one bit per channel, per head: the
                # mask says *which* elements survive, and is re-read each step.
                mask_bits = eff_kv_batch * (full_tokens + full_dim)
            else:
                # A mask that varies per (token, channel): one bit per element
                # of the *full* cache.  At 4-bit KV that is a quarter of the
                # dense cache -- charged whether or not the element survived.
                mask_bits = eff_kv_batch * full_tokens * full_dim
            mask_bytes = mask_bits // 8
            mem["dram_read"] += mask_bytes
            # The bitmap is a compact contiguous block per head.
            mem["dram_read_eff"] += self._dram_effective_bytes(
                mask_bytes, mask_bytes)
        return mem

    # ---- Flash ------------------------------------------------------------

    def _simulate_flash_attention(self, *args, **kwargs):
        if self.tokens_pruned or self.channels_pruned:
            raise NotImplementedError(
                "Unstructured KV pruning is modelled on the standard "
                "(non-fused) attention path only.  Run with "
                "flash_block_size=0.")
        return super()._simulate_flash_attention(*args, **kwargs)
