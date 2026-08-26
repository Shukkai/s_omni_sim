"""
GNN stage 2: a cycle model for aggregation, built entirely out of the existing
one.

`simulator/simulator.py` is untouched -- the same add-on pattern as
`cycle_units.py`, `unstructured_kv.py` and `array_pack.py`, and for the same
reason: `analysis/regression/baseline.py check` has to stay green by
construction, not by inspection.

**The identity this file is built on.**  Aggregation written as *pull* -- for
each destination node `v`, `h[v] = sum over u in N(v) of a_vu * x[u]` -- has
shape::

    M = 1            one destination row at a time
    K = deg(v)       one term per incoming edge
    N = F            the feature width

which is **exactly the shape of decode `attn_v`**, with `deg` in `kv_len`'s slot
and `F` in `head_dim`'s.  Not "analogous to": the same three integers into the
same function.  `gnn_run.py` pre-flight 9 proves it by running a real decode
`attn_v` operation through `_simulate_matmul` and asserting the aggregation
cycle count matches bit for bit.  Everything decode `attn_v` is known to suffer
from therefore transfers, and three `study.md` results arrive already measured:

  * **section 4(b), the fixed-overhead knee.**  `per_round = 3 (LGU) +
    ceil(K/MU) + 1 + array_n + 2`, so the constant is 10 cycles and the useful
    work is `ceil(deg/4)`.  A degree-1 node spends 10 of 11 cycles on overhead.
    KV eviction reaches that regime only at a 0.4% budget; a citation graph
    *starts* there.
  * **section 5, the N-null.**  `rounds = ceil(n_tiles / array_m)` with
    `n_tiles = ceil(N / (array_n * NUM_RAC))`, so N is free in cycles until
    `N > array_m * array_n * NUM_RAC = 4096`.  Note this is *wider* than
    section 5's `N <= 128` statement, which was about `n_tiles` alone; the
    `/array_m` in `rounds` extends the null by another 32x.  Every feature
    width any GNN uses is inside it.
  * **section 14, the packing floor.**  `M = 1` lights one of 32 PE rows.
    Aggregation inherits the 3.12% occupancy, at every degree.

**The round-count defect, and why push is reported twice.**
`Simulator._calculate_cycles` charges OS-V rounds as::

    M == 1 :  rounds = ceil(n_tiles / array_m)              # packs, correct
    else   :  rounds = ceil(ceil(M / array_m) * n_tiles)    # does not pack

The `else` branch rounds `M` up to a whole 32-row tile *before* multiplying by
`n_tiles`, so for `M` in 2..31 it issues `n_tiles` full passes where the
accumulator budget allows `ceil(M * n_tiles / array_m)`.  The overcharge is
`ceil(M/array_m) * n_tiles / ceil(M * n_tiles / array_m)` -- 2x at F=256
(`n_tiles = 2`) for any `M <= 16`, and up to `array_m / M` as `n_tiles` grows.

`HardwareConfig.os_rounds_model` (stage 11) is the fix, shipped **inert**:
`"tiled"` is the default and reproduces every published number, `"packed"` is
the accumulator-budget form.  So this file does not reimplement the correction
-- `push_cycles()` returns the `"tiled"` number and the `"packed"` number, both
straight out of `_calculate_cycles`, by holding a second `Simulator` whose
`hw` differs in that one field.  Pre-flight 13 asserts they bracket exactly the
ratio above.

**Pull is unaffected either way** (`M = 1` takes the branch both models agree
on, which is stage 11's own argument for `"packed"`), so every pull number here
is model-independent.  **Push is not**: it is issued at `M = deg(u)`, and most
nodes in most graphs have degree 2..31.  No push-vs-pull comparison in
`gnn_run.py` is reported without both columns.

**What is modelled and what is not.**

  * Cycles are the array's.  Index decode, the CSR row-pointer walk, the
    scatter-add into the destination accumulator and any pipeline bubble
    between two nodes of different degree are **not** charged.  All of them are
    per-node costs, so they inflate the fixed 10 rather than the useful
    `ceil(deg/4)` -- i.e. leaving them out is generous to the LUT, which is the
    direction that matters given the conclusion.
  * `qbit` for an AA operation is `hw.kv_cache_bits`, and here it means the
    precision of the **adjacency coefficient** `a_vu` -- the operand the LUT
    bit-plane loop iterates over.  `study.md` section 13 makes it the only axis
    that is a linear multiplier on cycles *and* on bytes, so it is swept rather
    than fixed.
  * The VPU null uses `hw.vpu_width` lanes.  The simulator's own `"VPU"`
    dataflow uses a hardcoded 64 (`_calculate_cycles`), which is half the
    declared `vpu_width = 128`.  Both are reported; the 128-lane number is the
    headline because it is what the hardware config declares, and it is the
    harder null for the LUT to beat.
  * Aggregation over a whole graph is the sum of per-node costs with **no
    overlap between nodes** and no batching of same-degree nodes into one pass.
    That matches how the simulator prices a sequence of operations
    (`overlap_model = "serial"`).

**Stage 3: P-way packing.**  `M = 1` is also an *opportunity*, and it is the
one `analysis/array_packing/array_pack.py` already exploits for decode
`attn_v`: consecutive destination nodes are independent instances, so `P` of
them can share one OS-V pass, each owning `array_m / P` rows.  This file adds
no cycle arithmetic for that either -- `packed_pull_cycles` calls
`array_pack.packed_osv_cycles`, and pre-flight 15 asserts a real decode
`attn_v` through `array_pack.PackedOSVSimulator` returns the identical count.
Two things the packing model here has to say that the attention one did not:

  * **The recovery is `P / ceil(P * n_tiles / array_m)`** per node, so it
    saturates at `array_m / n_tiles` and packing past `P* = array_m / n_tiles`
    buys nothing.  Measured, not assumed: section I.
  * **The schedule matters more than P does.**  A packed pass charges
    `ceil(K/MU)` once, so `P` nodes of different degree in one pass all pay the
    *maximum* degree in the pass.  `schedule` selects between three: `"ideal"`
    groups nodes of exactly equal degree and charges no partial pass (a lower
    bound, and the one that is bit-identical to stage 2 at `P = 1`);
    `"exact"` is the same grouping with whole passes, which fragments badly
    because a power-law degree distribution has thousands of nearly-empty
    buckets; `"sorted"` sorts by degree and fills passes greedily, paying the
    group maximum.  Section K measures all three and `"sorted"` is the
    realistic one.
"""

import math
import os
import sys
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'cycle_breakdown'))

sys.path.insert(0, os.path.join(_here, '..', 'array_packing'))

from array_pack import (                                         # noqa: E402
    max_useful_pack, packed_osv_cycles,
)
from cycle_units import UnitAwareSimulator                       # noqa: E402
from simulator import (                                          # noqa: E402
    ComputeMode, OperationMetrics, Simulator,
)

PULL = 'pull'
PUSH = 'push'
VPU = 'vpu'
DATAFLOWS = (PULL, PUSH, VPU)

# `per_round` for the M=1 OS-V branch is LUT_GEN(3) + k_eff + 1 + array_n +
# OUTPUT(2).  Everything except k_eff is the fixed overhead.
FIXED_PER_ROUND = 3 + 1 + 2      # + array_n, added where array_n is known

IDEAL = 'ideal'      # equal-degree groups, partial passes not charged
SORTED = 'sorted'    # degree-sorted greedy fill, group pays its max degree
EXACT = 'exact'      # equal-degree groups, whole passes only
SCHEDULES = (IDEAL, SORTED, EXACT)


def live_row_bytes_per_cycle(hw, pack: int = 1,
                             mu: int = Simulator.MU,
                             num_rac: int = Simulator.NUM_RAC) -> int:
    """KV-SRAM read bytes per cycle with `pack` OS-V rows live at once.

    The same expression `pack_run.py` section E uses, kept here rather than
    imported because it is arithmetic over `HardwareConfig` fields, not a
    simulator call -- and because the number it produces for a GNN is **4x**
    `study.md` section 16(d)'s, which was computed at `kv_cache_bits = 4`
    while aggregation runs at 16.  See `gnn_run.py` section K.
    """
    return mu * hw.array_n * num_rac * hw.kv_cache_bits // 8 * pack


@dataclass
class AggregateCost:
    """One `A_hat @ X` aggregation, cycles and bytes.

    `cycles` is what the simulator's own arithmetic charges.  `cycles_fixed`
    and `cycles_useful` split it the way `study.md` section 4(b) does, so the
    knee is visible rather than asserted.  `cycles_corrected` differs from
    `cycles` only for push, where it applies the `rounds` fix described in the
    module docstring.
    """
    dataflow: str
    feat_dim: int
    qbit: int
    cycles: float
    cycles_corrected: float
    cycles_fixed: float          # the part that does not shrink with degree
    cycles_useful: float         # the ceil(deg/MU) part
    nodes: float
    edges: float
    flops: int
    dram_read_eff: int
    dram_write_eff: int
    distribution_aware: bool

    @property
    def fixed_share(self) -> float:
        return self.cycles_fixed / self.cycles if self.cycles else 0.0

    @property
    def cycles_per_edge(self) -> float:
        return self.cycles / self.edges if self.edges else 0.0


class GNNSimulator(UnitAwareSimulator):
    """`Simulator` plus a cycle model for sparse aggregation.

    Adds no operation type and overrides no parent method: every cycle number
    below comes out of the inherited `_calculate_cycles`, called with the shape
    the dataflow implies.  That is the whole design -- if the aggregation cycle
    model were written out longhand here it would be an assertion about the
    hardware; calling the existing function makes it a consequence of the
    hardware model already published.

    Args:
        vpu_width: lanes for the VPU null.  `None` takes `hw.vpu_width`.
    """

    def __init__(self, hw, vpu_width: Optional[int] = None, **kwargs):
        super().__init__(hw, **kwargs)
        self.vpu_width = vpu_width if vpu_width else hw.vpu_width
        # The same hardware with stage 11's corrected round count, used only to
        # produce push's second column.  Building it here rather than writing
        # the formula out means the correction is the simulator's, not this
        # file's, and it tracks any future change to `"packed"`.
        self._packed = Simulator(replace(hw, os_rounds_model="packed"))

    # ---- Per-node cycle models --------------------------------------------

    def pull_cycles(self, deg: int, feat_dim: int, qbit: int) -> int:
        """Cycles to gather one destination node's `deg` neighbours.

        `(M=1, K=deg, N=F)` straight into the inherited `_calculate_cycles` on
        the `LUT_OS_V` path -- the same call decode `attn_v` makes.
        """
        return self._calculate_cycles(1, deg, feat_dim, qbit, ComputeMode.AA,
                                      "LUT_OS_V", 1)

    def pull_fixed_useful(self, deg: int, feat_dim: int,
                          qbit: int) -> Tuple[int, int]:
        """Split `pull_cycles` into (fixed, useful).

        Derived from the array geometry rather than restated, and pre-flight 10
        asserts the two halves sum to `pull_cycles`.
        """
        hw = self.hw
        n_tiles = math.ceil(feat_dim / (hw.array_n * self.NUM_RAC))
        rounds = math.ceil(n_tiles / hw.array_m / hw.replication)
        fixed = (FIXED_PER_ROUND + hw.array_n) * rounds * qbit
        useful = math.ceil(deg / self.MU) * rounds * qbit
        return fixed, useful

    def push_cycles(self, deg: int, feat_dim: int,
                    qbit: int) -> Tuple[int, int]:
        """Cycles to push one source node's row to its `deg` destinations.

        `(M=deg, K=1, N=F)`: a rank-1 outer product per source node.  Returns
        `(os_rounds_model="tiled", os_rounds_model="packed")` -- the default
        and the corrected round count, both from `_calculate_cycles`, never
        rewritten here.  They are equal only when `n_tiles == 1` or `deg` is a
        multiple of `array_m`.
        """
        args = (deg, 1, feat_dim, qbit, ComputeMode.AA, "LUT_OS_V", 1)
        return (self._calculate_cycles(*args),
                self._packed._calculate_cycles(*args))

    def vpu_cycles(self, edges: float, feat_dim: int) -> float:
        """The null: `E * F / vpu_width` multiply-accumulates, no LUT.

        One lane does one MAC per cycle, so a gather-accumulate over `E` edges
        of `F`-wide features is `E * F / lanes` cycles.  No `qbit` term -- a
        VPU lane is priced per element, not per bit-plane, which is precisely
        the asymmetry the LUT has to overcome.
        """
        return edges * feat_dim / self.vpu_width

    def vpu_cycles_simulator_mode(self, edges: float, feat_dim: int) -> float:
        """The same null through the simulator's own `"VPU"` dataflow.

        `_calculate_cycles` hardcodes `(M*K*N*batch)//64` for `mode == "VPU"`,
        which is half `hw.vpu_width`.  Reported alongside so the choice of
        denominator is visible rather than buried.
        """
        return edges * feat_dim / 64

    # ---- Whole-graph aggregation ------------------------------------------

    def _degree_buckets(self, g, distribution_aware: bool
                        ) -> List[Tuple[int, float]]:
        """`[(degree, node_count)]`, either the real distribution or one bucket.

        The mean-degree path uses `round(avg_degree)` because the cycle model
        takes an integer `K`; the rounding is reported by `gnn_run.py` section
        G rather than hidden, and it is exactly the approximation the
        side-by-side columns exist to measure.
        """
        if distribution_aware:
            return g.degree_distribution()
        return [(max(1, round(g.avg_degree)), float(g.num_nodes))]

    def aggregate_cost(self, g, feat_dim: int, qbit: int, dataflow: str,
                       distribution_aware: bool = True,
                       gather=None) -> AggregateCost:
        """Cost of one `A_hat @ X` over graph `g` at width `feat_dim`.

        Cycles are summed over the degree distribution: pull charges one
        `(1, deg(v), F)` operation per *destination*, push one `(deg(u), 1, F)`
        per *source*, and the VPU charges `E * F / lanes` in one go.  Since the
        distribution is the same object either way (the graph's degree sequence
        is its own transpose in edge count if not in identity), push and pull
        see the same buckets.

        `gather` is an optional `GatherCost` from `gnn_run.py`.  It is
        duck-typed rather than imported so this module does not depend on the
        report script; when supplied, its DRAM terms are attached so the result
        can go through `Simulator._op_roofline_time` unchanged.
        """
        if dataflow not in DATAFLOWS:
            raise ValueError(f"unknown dataflow {dataflow!r}")
        buckets = self._degree_buckets(g, distribution_aware)
        nodes = sum(c for _, c in buckets)
        edges = sum(d * c for d, c in buckets)

        cycles = corrected = fixed = useful = 0.0
        if dataflow == PULL:
            for d, c in buckets:
                cycles += self.pull_cycles(d, feat_dim, qbit) * c
                f, u = self.pull_fixed_useful(d, feat_dim, qbit)
                fixed += f * c
                useful += u * c
            corrected = cycles
        elif dataflow == PUSH:
            for d, c in buckets:
                m, corr = self.push_cycles(d, feat_dim, qbit)
                cycles += m * c
                corrected += corr * c
            # Push's `per_round` has no degree term at all (K=1); the degree
            # lives in `rounds`.  So "fixed" is the array_m + array_n fill,
            # charged once per round.
            fixed = cycles * (self.hw.array_m + self.hw.array_n) / (
                3 + 1 + self.hw.array_m + self.hw.array_n + 2)
            useful = cycles - fixed
        else:
            cycles = corrected = self.vpu_cycles(edges, feat_dim)
            useful = cycles

        dram_r = dram_w = 0
        if gather is not None:
            dram_r = gather.charged + gather.structure
            dram_w = gather.writeback

        return AggregateCost(
            dataflow=dataflow, feat_dim=feat_dim, qbit=qbit,
            cycles=cycles, cycles_corrected=corrected,
            cycles_fixed=fixed, cycles_useful=useful,
            nodes=nodes, edges=edges,
            flops=int(2 * edges * feat_dim),
            dram_read_eff=dram_r, dram_write_eff=dram_w,
            distribution_aware=distribution_aware,
        )

    def aggregate_metrics(self, cost: AggregateCost,
                          use_corrected: bool = False) -> OperationMetrics:
        """Wrap an `AggregateCost` as an `OperationMetrics`.

        Only so the *existing* `_op_roofline_time` can price it -- the roofline
        is then the same `max(compute, DRAM, SRAM)` every other number in this
        repo is produced with, rather than a second latency model that could
        drift from it.  `sram_read`/`sram_write` are left at 0: the simulator's
        SRAM-bandwidth term is opt-in and inert at `sram_bandwidth_gbps = 0`,
        and there is no defensible SRAM traffic figure for a gather whose
        on-chip behaviour this model does not describe.
        """
        cyc = cost.cycles_corrected if use_corrected else cost.cycles
        m = OperationMetrics(shape=(int(cost.nodes), int(cost.edges /
                                                         max(cost.nodes, 1)),
                                    cost.feat_dim))
        m.cycles = int(round(cyc))
        m.flops = cost.flops
        m.dram_read = m.dram_read_eff = cost.dram_read_eff
        m.dram_write = m.dram_write_eff = cost.dram_write_eff
        m.utilization = ((m.flops / 2) / (m.cycles * self.LANES_EQUIV)
                         if m.cycles > 0 else 0.0)
        return m

    def aggregate_roofline_ms(self, cost: AggregateCost,
                              use_corrected: bool = False) -> float:
        freq = self.hw.freq_mhz * 1e6
        bw = self.hw.dram_bandwidth_gbps * 1e9
        return self._op_roofline_time(
            self.aggregate_metrics(cost, use_corrected), freq, bw) * 1e3

    # ---- The crossover ----------------------------------------------------

    def crossover_degree(self, feat_dim: int, qbit: int,
                         max_degree: int = 100_000) -> Optional[int]:
        """Smallest degree at which pull/LUT beats the VPU on cycles, or None.

        Searched rather than solved, because the closed form (`(10 + ceil(d/MU)) * qbit < d * F / lanes`) has a `ceil` in it and the answer
        near the knee is off by one either way.  `None` means the LUT never
        wins at that width and precision, which -- see `gnn_run.py` section G
        -- is the common case, not the exception.
        """
        for d in range(1, max_degree + 1):
            if self.pull_cycles(d, feat_dim, qbit) < self.vpu_cycles(1, feat_dim) * d:
                return d
        return None
