"""
Packing independent decode attention instances into the idle OS-V rows.

`simulator/` is untouched, the same add-on pattern as `cycle_units.py`,
`kv_budget.py`, `think_prune.py` and `selective_attn.py`.

**The opportunity.**  Decode `attn_v` is issued as `(M=1, K=kv_len, N=head_dim)`
with `head_dim = 128`, so

    n_tiles = ceil(128 / (array_n * NUM_RAC)) = ceil(128 / 128) = 1
    rounds  = ceil(n_tiles / array_m)         = ceil(1 / 32)     = 1

One of 32 PE rows does work, **at any context length**, and cycles scale
exactly linearly with `batch x num_heads`.  Measured occupancy is 3.12% = 1/32.
`qk_matmul` is different: `N = kv_len`, so `n_tiles = ceil(kv_len/128)` and the
rows fill up naturally once `kv_len >= array_m * array_n * NUM_RAC = 4096`.

**What the hardware would have to do.**  Verified against OMNI_LUT.pdf IV-C and
IV-D.  The LUT is generated from the *activation* operand -- the query for
QK^T, the attention scores for Attn.V -- not from the KV cache.  So P packed
instances need P distinct LUTs and P ungated LGUs.  The array has 32 LGUs (one
per row); OS-V gates 31 of them *precisely because* M=1, and broadcasts the
survivor to all 32 rows.  Packing is the generalization of that broadcast: P
LGUs, each driving `array_m / P` rows.

What packed instances *share* is the K/V bit-plane stream -- the "weight"
operand -- and sharing the weight operand across rows is what output-stationary
already does.  The paper describes no packing of heads or batch elements and
never treats batch as an architectural knob, so this is an extension, not a
documented feature.

**Scope, stated so the numbers are not over-read.**

  * Decode only, standard (non-fused) attention.  `_simulate_flash_attention`
    raises rather than guess: it calls `_calculate_cycles` with `batch_size=1`
    and multiplies outside, so dividing the batch inside would apply the
    `rounds` inflation without its compensating term -- a P-fold pessimisation
    dressed up as a result.
  * **The energy model already assumes packing.**  `os_v_energy_model.py:23`
    charges `energy_per_tile * n_tiles / array_m * qbit * batch_size`, and
    `omni_energy_model.py` divides the M==1 OS energy by `array_m`.  Energy is
    amortized over all 32 rows while the cycle model charges a full round for
    one row.  The two halves of the model disagree, and packing is what makes
    them agree.  Any "packing is energy-neutral" reading is an artefact of that
    `/array_m`, not a physical finding.
  * The LGU ungating power, the weight-FIFO bandwidth, the P live LUTs and the
    P-way broadcast tree are all **uncharged**.  `pack_run.py` section E
    computes what they would cost rather than waving at them.
  * Approximation quality is not at issue: packing is exact, it reorders work
    without changing any arithmetic.
"""

import math
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'cycle_breakdown'))

from cycle_units import (                                        # noqa: E402
    LUT_GEN_CYCLES, OUTPUT_CYCLES, UnitAwareSimulator, cycle_units,
)
from simulator import ComputeMode, OperationType, Simulator      # noqa: E402


def packed_osv_cycles(hw, M: int, K: int, N: int, qbit: int,
                      batch_size: int, pack: int,
                      mu: int = Simulator.MU,
                      num_rac: int = Simulator.NUM_RAC) -> int:
    """Cycles for one packed OS-V operation.

    Deliberately an *independent* expression of the total rather than a call
    into `cycle_units`, so the sync assertion in `pack_run.py` compares two
    derivations instead of restating one.
    """
    k_eff = math.ceil(K / mu)
    n_tiles = math.ceil(N / (hw.array_n * num_rac))
    rows_per_inst = max(1, hw.array_m // pack)
    # Each packed instance still contributes exactly one activation row, so the
    # M=1 fill term is unchanged -- it is per instance, not per array.
    per_round = LUT_GEN_CYCLES + k_eff + 1 + hw.array_n + OUTPUT_CYCLES
    rounds = math.ceil(n_tiles / rows_per_inst / hw.replication)
    return math.ceil(batch_size / pack) * per_round * rounds * qbit


def row_slot_waste(hw, N: int, pack: int = 1,
                   num_rac: int = Simulator.NUM_RAC) -> float:
    """Fraction of row-slots left idle by the `rounds` ceiling.

    Unpacked, `rounds = ceil(n_tiles / array_m)` rounds *up* to whole 32-row
    passes, so the final pass can leave up to `array_m - 1` rows idle.  Packing
    subdivides the array into `pack` groups of `array_m/pack` rows, which is a
    finer quantum, so less of the tail is wasted.
    """
    n_tiles = math.ceil(N / (hw.array_n * num_rac))
    rows_per_inst = max(1, hw.array_m // pack)
    issued = math.ceil(n_tiles / rows_per_inst) * rows_per_inst
    return (issued - n_tiles) / issued if issued else 0.0


def max_useful_pack(hw, N: int, K: int = 128, qbit: int = 4,
                    batch_size: int = 32,
                    num_rac: int = Simulator.NUM_RAC) -> int:
    """Largest P that still strictly reduces cycles for an OS-V op.

    Determined by search over `packed_osv_cycles` rather than from a closed
    form, because the closed form is wrong in a way that matters.  Two distinct
    effects are in play:

      * **N < array_m tiles** -- rows genuinely sit idle, and packing fills
        them.  `attn_v` lives here permanently (`n_tiles = 1`).
      * **N not a multiple of array_m tiles** -- every row is busy in the body,
        but the *tail* round is partly idle.  Packing shrinks the quantum and
        recovers it, worth up to `array_m / n_tiles` just past a tile boundary
        and decaying as `1/n_tiles`.

    The second effect is why `qk_matmul` is not the no-op a closed form
    predicts: decode `kv_len` is `context + token_idx`, so `n_tiles` is almost
    never an exact multiple of `array_m`.  Packing is neutral only when it is.
    """
    best = packed_osv_cycles(hw, 1, K, N, qbit, batch_size, 1,
                             num_rac=num_rac)
    useful = 1
    p = 1
    while p * 2 <= hw.array_m:
        p *= 2
        if hw.array_m % p:
            continue
        c = packed_osv_cycles(hw, 1, K, N, qbit, batch_size, p,
                              num_rac=num_rac)
        if c < best:
            best, useful = c, p
    return useful


class PackedOSVSimulator(UnitAwareSimulator):
    """`UnitAwareSimulator` that packs `pack` decode attention instances.

    Args:
        pack: instances retired per OS-V pass.  Must divide `array_m`.
            1 = today's behaviour, and the class is then provably inert
            because `_calculate_cycles` delegates straight to `super()`.
        gqa_share: the packed instances are query heads of one GQA group, so
            they share the K/V tile.  Only meaningful up to the group size
            (`num_heads // num_kv_heads`); beyond that the tiles are distinct
            whatever this says, and `peak_sram` reflects it.
        pack_qk: also pack `qk_matmul`.  On by default, but it is a no-op at
            `kv_len >= 4096` by construction -- see `max_useful_pack`.
    """

    def __init__(self, hw, pack: int = 1, gqa_share: bool = False,
                 gqa_group: int = 1, pack_qk: bool = True, **kwargs):
        super().__init__(hw, **kwargs)
        if pack < 1:
            raise ValueError("pack must be >= 1")
        if hw.array_m % pack != 0:
            raise ValueError(
                f"pack={pack} must divide array_m={hw.array_m}: the packed "
                f"instances each own array_m/pack rows, and a partial row "
                f"group is not a thing the array can express.")
        self.pack = pack
        self.gqa_share = gqa_share
        self.gqa_group = max(1, gqa_group)
        self.pack_qk = pack_qk
        self._op = None    # (op_type, is_decode) for the cycle/SRAM hooks

    # ---- Context plumbing ---------------------------------------------------

    def _simulate_matmul(self, op_type, compute_mode, shape, **kwargs):
        # `_calculate_cycles` and `_calculate_peak_sram` receive only shapes,
        # so stash which operation is in flight -- the same trick
        # `SelectiveAttnSimulator` uses for `_full_kv_len`.
        self._op = (op_type, kwargs.get('is_decode', False))
        try:
            return super()._simulate_matmul(op_type, compute_mode, shape,
                                            **kwargs)
        finally:
            self._op = None

    def _simulate_flash_attention(self, *args, **kwargs):
        if self.pack > 1:
            raise NotImplementedError(
                "OS-V packing is modelled on the standard (non-fused) "
                "attention path only.  _simulate_flash_attention calls "
                "_calculate_cycles with batch_size=1 and applies the real "
                "batch outside, so the packed batch term cannot be applied "
                "there without double-counting.  Run with flash_block_size=0.")
        return super()._simulate_flash_attention(*args, **kwargs)

    def _pack_for(self, M: int, mode: str, compute_mode) -> int:
        """Resolve the pack factor for the operation currently in flight."""
        op_type, is_decode = self._op or (None, False)
        if (self.pack == 1 or not is_decode or M != 1
                or mode != "LUT_OS_V"
                or compute_mode != ComputeMode.AA
                or op_type not in (OperationType.QK_MATMUL,
                                   OperationType.ATTN_V_MATMUL)):
            return 1
        if op_type == OperationType.QK_MATMUL and not self.pack_qk:
            return 1
        return self.pack

    # ---- Cycles -------------------------------------------------------------

    def _calculate_cycles(self, M, K, N, qbit, compute_mode, mode, batch_size):
        p = self._pack_for(M, mode, compute_mode)
        if p == 1:
            # Delegate untouched: at the disabled default this class is inert
            # including the unit split, which is what makes preflight #1 exact
            # rather than approximate.
            return super()._calculate_cycles(M, K, N, qbit, compute_mode,
                                             mode, batch_size)
        self._pending.append(
            cycle_units(self.hw, M, K, N, qbit, mode, batch_size,
                        mu=self.MU, num_rac=self.NUM_RAC, pack=p))
        return packed_osv_cycles(self.hw, M, K, N, qbit, batch_size, p,
                                 mu=self.MU, num_rac=self.NUM_RAC)

    # ---- Co-residency: what the packing costs -------------------------------

    def _calculate_peak_sram(self, M, K, N, compute_mode, mode, batch_size,
                             sram_batch: int = 1) -> int:
        """P instances in flight means P working sets resident at once.

        The parent models batch elements as sequential by default
        (`hw.sram_batch_model == "sequential"`), so without this override the
        packing would look free.  It is not: the K/V tile is the dominant term
        at long context, and only a GQA group genuinely shares one.
        """
        base = super()._calculate_peak_sram(M, K, N, compute_mode, mode,
                                            batch_size, sram_batch=sram_batch)
        p = self._pack_for(M, mode, compute_mode)
        if p == 1:
            return base

        hw = self.hw
        qbit = hw.kv_cache_bits
        col_tile = min(N, hw.array_n * self.NUM_RAC)
        A_bytes = M * K * hw.act_bits // 8
        B_tile = K * col_tile * qbit // 8
        C_tile = M * col_tile * hw.accumulate_bits // 8

        # A and C are per instance always.  The parent may already be holding
        # several batch elements under "concurrent"; take the larger rather
        # than multiplying, so cross-batch packing is not counted twice.
        parent_resident = max(1, base // max(1, A_bytes + B_tile + C_tile))
        inst = max(parent_resident, p)

        # B is shared only within a GQA group -- those query heads read the
        # same K/V tensor.  Past the group size the tiles are distinct.
        groups = math.ceil(p / self.gqa_group) if self.gqa_share else p
        return inst * (A_bytes + C_tile) + groups * B_tile

    # ---- Reporting helper ---------------------------------------------------

    def pack_summary(self, kv_len: int, head_dim: int = 128) -> dict:
        return {
            'pack': self.pack,
            'gqa_share': self.gqa_share,
            'max_useful_attn_v': max_useful_pack(self.hw, head_dim),
            'max_useful_qk': max_useful_pack(self.hw, kv_len),
        }
