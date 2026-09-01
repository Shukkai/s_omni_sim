"""
FFN activation sparsity on top of the stock simulator.

`simulator/` is untouched beyond three default-identical hooks, exactly as with
`unstructured_kv.py`: this is a subclass that says how many FFN hidden units an
operation actually touches, and how contiguous the retained weight bytes are.

**What is being modelled.**  A gated FFN drives most of its hidden units to
near-zero for any given token, and a threshold or a predictor turns that into a
usable mask (TEAL, CATS, Deja Vu).  Skipping unit `j` skips **column `j` of FC1
and row `j` of FC2** -- so what stops being fetched is *weights*, which is what
makes this different from every KV technique in `study.md`.  §3 found decode
idles ~85% of the time waiting on weights; this is the only lever in the repo
aimed at that idle.

**Three axes, because three separate things decide whether it pays.**

1. `density` -- the fraction of hidden units active per token.

2. `mask_source` -- where the mask comes from, which decides *which* matrices
   can use it:
     * `"input"`   thresholding or predicting from the FFN input (TEAL, Deja
                   Vu).  FC1 and FC2 are both sparse.
     * `"output"`  thresholding FC1's own output (CATS).  You cannot skip the
                   work that produced the thing you threshold, so FC1 stays
                   dense and only FC2 is sparse.

3. `share_mask` -- whether the `M` tokens in one GEMM share a mask.  **This is
   the axis that decides the whole technique**, and it is not a knob a designer
   turns; it is a fact about the workload:
     * `True`   every token in the operation uses the same mask.  An upper
                bound, real only at `M = 1`.
     * `False`  each token has its own, so the weight columns that must be
                fetched are the **union** over the operation's tokens:
                `1 - (1 - density)^M`.  At `M = 1` this is exactly `density`
                and the two models coincide; by `M = 32` at 10% density it is
                96.6%, and by prefill's `M` it is indistinguishable from dense.

   The two bracket the truth the way `hw.overlap_model` does, and `False` is
   the honest default.

**Layout, and why it is not §15's answer.**  §15 found KV channel pruning
collects nothing unstructured, because a retained channel is 0.5 B in
token-major order and a DDR5 burst is 64 B.  The same question here has the
opposite answer, and `weight_layout` is what makes it askable:
    * `"neuron_major"` -- unit `j`'s weights are contiguous, so one unit is
      `d_model * weight_bits/8` = 2048 B for LLaMA-3-8B: **32 whole bursts, at
      group 1**.
    * `"model_major"` -- unit `j` is strided one element per row, so a run is
      `weight_bits/8` = 0.5 B and the mask collects nothing.
The difference is that a weight layout is chosen **offline, by the compiler**,
while a KV layout is dictated online by an append-only cache.  §15's obligation
was unmeetable; this one is a build-time decision.

**Selection cost is excluded**, as it is for every technique in §4-§15.  A
Deja Vu predictor is a real matmul and a CATS threshold is a real VPU pass;
neither is charged here.
"""

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, '..', '..'))
for p in ('simulator', 'analysis/cycle_breakdown'):
    sys.path.insert(0, os.path.join(_root, *p.split('/')))

from cycle_units import UnitAwareSimulator          # noqa: E402
from simulator import OperationType                 # noqa: E402

SPARSE_OPS = (OperationType.FC1, OperationType.FC2)


class ActSparsitySimulator(UnitAwareSimulator):
    """`UnitAwareSimulator` with a masked FFN hidden dimension.

    Args:
        density:       fraction of FFN hidden units active per token.
                       1.0 = dense, and reproduces the stock simulator exactly.
        mask_source:   "input" (FC1 and FC2 both sparse) or "output" (FC1
                       dense, FC2 sparse).  See the module docstring.
        share_mask:    True = all `M` tokens of a GEMM share one mask (upper
                       bound); False = per-token masks, so the fetched weight
                       set is the union over the operation's tokens.
        weight_layout: "neuron_major" (a unit's weights are contiguous) or
                       "model_major" (strided one element per row).
        neuron_group:  how many *consecutive* hidden units share a mask
                       decision.  1 = fully unstructured.
    `proj_m` and `d_model` are not constructor arguments: they differ between
    prefill and decode within a single run, so they are read off each
    transformer step instead.
    """

    def __init__(self, hw, density: float = 1.0, mask_source: str = "input",
                 share_mask: bool = False, weight_layout: str = "neuron_major",
                 neuron_group: int = 1, **kwargs):
        super().__init__(hw, **kwargs)
        if not 0.0 < density <= 1.0:
            raise ValueError(f"density must be in (0, 1], got {density}")
        if mask_source not in ("input", "output"):
            raise ValueError(f"unknown mask_source {mask_source!r}")
        if weight_layout not in ("neuron_major", "model_major"):
            raise ValueError(f"unknown weight_layout {weight_layout!r}")
        self.density = density
        self.mask_source = mask_source
        self.share_mask = share_mask
        self.weight_layout = weight_layout
        self.neuron_group = max(1, neuron_group)
        # Filled in per transformer step; the defaults only matter if a hook
        # is somehow reached before the first step, which it is not.
        self.proj_m = 1
        self.d_model = 4096

    def _simulate_transformer_step(self, metrics, model, workload, proj_m,
                                   attn_q_len, kv_len, is_decode,
                                   token_idx=-1):
        """Capture the shape facts the hooks need, then defer to the base.

        `proj_m` is `batch x seq_len` in prefill and `batch` in decode, and the
        union term depends on it -- so it has to come from the step rather than
        from the constructor, or one run could not cover both phases.
        """
        self.proj_m = max(1, proj_m)
        self.d_model = model.d_model
        return super()._simulate_transformer_step(
            metrics, model, workload, proj_m=proj_m, attn_q_len=attn_q_len,
            kv_len=kv_len, is_decode=is_decode, token_idx=token_idx)

    # ---- The density that actually applies to one operation ----------------

    def _op_density(self, op_type) -> float:
        """Fraction of hidden units this operation must touch.

        1.0 means "dense", and every hook below returns its inert value for it.
        """
        if op_type not in SPARSE_OPS or self.density >= 1.0:
            return 1.0
        if op_type == OperationType.FC1 and self.mask_source == "output":
            # CATS thresholds FC1's own output; FC1 has to produce it.
            return 1.0
        if self.share_mask:
            return self.density
        # Per-token masks: the operation fetches the *union* of its tokens'
        # columns.  `1 - (1-d)^M`, which is `d` at M = 1 and saturates fast.
        return 1.0 - (1.0 - self.density) ** self.proj_m

    # ---- Hook 1: how many hidden units the operation touches ---------------

    def _ffn_active_neurons(self, model, op_type, is_decode: bool) -> int:
        d = self._op_density(op_type)
        if d >= 1.0:
            return model.d_ffn
        return max(1, int(round(model.d_ffn * d)))

    # ---- Hooks 2 and 3: how fragmented the masked weight read is -----------

    def _neuron_slice_bytes(self) -> float:
        """Contiguous bytes belonging to one hidden unit, given the layout."""
        wb = self.hw.weight_bits / 8.0
        if self.weight_layout == "neuron_major":
            # The unit's whole d_model-long weight vector is contiguous.
            return self.d_model * wb
        # Strided: one element per row of the other dimension.
        return wb

    def _aw_weight_run_bytes(self, op_type, logical_bytes: int) -> int:
        if self._op_density(op_type) >= 1.0:
            return logical_bytes
        run = self.neuron_group * self._neuron_slice_bytes()
        # A run can never exceed the read it is part of.
        return max(1, min(logical_bytes, int(run)))

    def _aw_weight_covering_bytes(self, op_type, logical_bytes: int) -> int:
        d = self._op_density(op_type)
        if d >= 1.0:
            return 0
        # The dense matrix the scattered set sits inside: a gathering reader
        # can always give up and stream that instead.
        return int(logical_bytes / d)
