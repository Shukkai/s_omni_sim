"""
GNN on Omni-LUT, stage 2: Combine simulated, Aggregate simulated.

A GNN layer is `H' = sigma( A_hat @ (H @ W) )` and it splits into two halves
with opposite characters:

  * **Combine** `H @ W` -- a dense GEMM whose `W` is shared across every node.
    Shape `(num_nodes, F_in) x (F_in, F_out)`, structurally identical to a
    transformer FFN.  `Simulator._simulate_matmul` is shape-driven, so this
    needs **no simulator change at all**: it runs through the existing AW /
    `LUT_WS` path exactly as `fc1` does.  Everything in section B is a real
    simulator measurement.

  * **Aggregate** `A_hat @ .` -- a sparse gather-accumulate over the edge list.
    The simulator has no sparse operand: it would charge `N^2` for a dense
    `A_hat`, which is orders of magnitude off (section D measures how far).  So
    aggregation is priced here by the closed-form `gather_cost()` below, which
    is stage 1's deliverable and stage 2's specification.

**Why the gather formula is the interesting half.**  It is the same shape as
`study.md` section 15's KV burst analysis, with the **feature width playing
head_dim's role** against the DRAM burst.  A neighbour's feature row is
`F * act_bits / 8` bytes read at an arbitrary address, so a row shorter than a
burst wastes the remainder -- and Kipf & Welling's `hidden_dim = 16` at FP16 is
a 32 B row against DDR5's 64 B burst, i.e. exactly half wasted, on the most
cited GNN configuration there is.

**The clamp, and why it is not the same clamp as section 15's.**  A gather can
never cost more than giving up and streaming the whole feature matrix, so the
saving goes to zero rather than negative -- same one-sided semantics as
`_kv_covering_bytes`.  But there it is a rarely-fired safety net, and here it is
the common case: `E > N` on every graph that exists, so **pull/gather never
moves fewer feature bytes than push/stream when the accumulators fit**.  What
makes gathering right is capacity, not traffic.  Streaming needs all `N * F_out`
accumulators resident and re-reads `X` once per block that does not fit; the
gather is destination-stationary and needs one row, whatever the graph size.
Section C2 measures where that crosses over, and the answer is one inequality:
`stream_passes < avg_degree * burst_waste`.

**Stage 2 (sections F-H).**  Aggregation now has a *cycle* model as well as a
DRAM model, and it needed no new arithmetic at all, because pull aggregation --
`h[v] = sum a_vu x[u]` over `v`'s neighbours -- is issued as `(M=1, K=deg(v),
N=F)`, which is **the shape of decode `attn_v`** with `deg` in `kv_len`'s slot.
Pre-flight 9 proves that by running a real decode `attn_v` operation and
asserting the aggregation cycle count matches it bit for bit.  `gnn_sim.py`
holds the model and the caveats; the load-bearing one is that push aggregation
(`M = deg(u)`) lands on a `rounds` branch of `_calculate_cycles` that does not
pack `M < array_m`, so section G reports push twice -- as charged and as
corrected -- and never compares it to pull on the charged number alone.

Degrees come from `GraphConfig.degree_distribution()`, a **synthesised**
power-law fit whose only tie to the dataset is that its mean reproduces the
published `avg_degree`.  Section G reports mean-degree and distribution-aware
totals side by side so the size of that modelling choice is visible.

Usage:
    python gnn_run.py
    python gnn_run.py --csv gnn.csv --report gnn_report.md
"""

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, '..', '..'))
for p in ('simulator', 'analysis'):
    sys.path.insert(0, os.path.join(_root, *p.split('/')))

from simulator import (                                              # noqa: E402
    HardwareConfig, Simulator, ComputeMode, OperationType,
)
from graph_configs import (                                          # noqa: E402
    GraphConfig, get_graph_config, list_graphs,
)
from memory_tech import memory_technology                            # noqa: E402
from report import Report                                            # noqa: E402
sys.path.insert(0, os.path.join(_root, 'analysis', 'array_packing'))
from array_pack import PackedOSVSimulator                            # noqa: E402
from gnn_sim import (                                               # noqa: E402
    GNNSimulator, PULL, PUSH, VPU,
    IDEAL, SORTED, EXACT, SCHEDULES, live_row_bytes_per_cycle,
)

ACT_BITS = 16
WEIGHT_BITS = 4
ACCUM_BITS = 32
INDEX_BYTES = 4        # int32 CSR column indices
EDGE_WEIGHT_BITS = 16  # GCN's 1/sqrt(d_u d_v), one per non-zero
TECHS = ['DDR5-6400', 'HBM3']
SMALL = ['Cora', 'CiteSeer', 'PubMed']          # dense A_hat is simulable
CAPACITY_KB = [0, 1024, 8192]                   # 0 = unlimited
QBITS = [16, 8, 4, 2]                           # adjacency-coefficient bits
KNEE_DEGREES = [1, 2, 3, 4, 8, 16, 32, 64, 128, 492, 4096]


def base_hw(burst_bytes=0, dram_bw=51.2, sram_capacity_kb=0):
    """The array Omni-LUT is measured with everywhere else in this repo."""
    return HardwareConfig(
        array_m=32, array_n=4, FPE_array_size=64,
        act_bits=ACT_BITS, accumulate_bits=ACCUM_BITS,
        weight_bits=WEIGHT_BITS, kv_cache_bits=ACT_BITS,
        AW_mode="OMNI", AA_mode="OMNI",
        dram_bandwidth_gbps=dram_bw, dram_burst_bytes=burst_bytes,
        sram_capacity_kb=sram_capacity_kb,
    )


# ============================================================================
# The gather cost formula -- stage 1's deliverable
# ============================================================================

@dataclass
class GatherCost:
    """Closed-form DRAM cost of one `A_hat @ X` aggregation.

    Every field is bytes unless named otherwise.  `charged` is the number a
    cost model should use; the rest are there so the number can be checked
    rather than trusted.
    """
    row_bytes: int          # one neighbour's feature row
    burst_bytes: int
    logical: int            # E * row -- what the gather asks for
    effective: int          # E * ceil(row/burst)*burst -- what DRAM moves
    stream: int             # the dense-streaming alternative, passes included
    stream_passes: int      # how many times streaming re-reads X
    charged: int            # min(effective, stream) -- the cheaper dataflow
    stream_wins: bool       # True when push/stream beat pull/gather
    dataflow: str           # 'stream' or 'gather', whichever charged is
    structure: int          # CSR indices + edge weights (contiguous)
    writeback: int          # N * F * act/8 output
    total: int              # charged + structure + writeback
    burst_waste: float      # effective / logical
    flops: int              # 2 * E * F
    accum_resident: int     # N * F * accum/8 -- what streaming needs on chip
    gather_resident: int    # 1 * F * accum/8 -- what the gather needs on chip


def gather_cost(g: GraphConfig, feat_dim: int, act_bits: int = ACT_BITS,
                burst_bytes: int = 0, accum_bits: int = ACCUM_BITS,
                sram_capacity_kb: int = 0,
                index_bytes: int = INDEX_BYTES,
                edge_weight_bits: int = EDGE_WEIGHT_BITS) -> GatherCost:
    """DRAM cost of aggregating `feat_dim`-wide features over `g`'s edges.

    The model, in three terms.

    **1. The gather.**  Each of the `E` non-zeros in `A_hat` reads one
    neighbour's feature row from an arbitrary address::

        row       = feat_dim * act_bits / 8
        logical   = E * row
        effective = E * ceil(row / burst) * burst

    This is `Simulator._dram_effective_bytes(logical, row)` with the row as the
    contiguous run, and pre-flight 4 asserts the two agree.  Rows shorter than
    a burst pay the whole burst -- the `hidden_dim = 16` case.

    **2. The clamp, which turns out to be a dataflow choice.**  A gather can
    never cost more than giving up and streaming the whole feature matrix once
    per destination block, pushing each row to its neighbours as it goes::

        stream  = passes * N * row      (contiguous, so no burst waste)
        charged = min(effective, stream)

    Same one-sided semantics as `_kv_covering_bytes` in the KV path -- the
    saving from sparsity goes to zero, never negative.  But unlike the KV case
    the clamp is not a rarely-fired safety net, because `E > N` on every graph
    that exists: **pull/gather never moves fewer feature bytes than push/stream
    when the accumulators fit.**  What makes gathering the right choice is
    capacity, not traffic.

    Streaming accumulates into all `N` destinations at once, so it needs
    `N * feat_dim * accum_bits/8` on chip and re-reads `X` once per destination
    block that does not fit.  The gather is destination-stationary: it finishes
    node `v` before moving on and needs one row of accumulators regardless of
    graph size.  So the comparison reduces to one inequality::

        stream wins  <=>  stream_passes < avg_degree * burst_waste

    and `sram_capacity_kb = 0` (unlimited, `passes = 1`) is streaming's best
    case, where it always wins.  Section C sweeps real capacities, which is
    where the crossover actually lives.

    The streaming term charges a full `N * row` per pass, i.e. it assumes no
    locality in the partition.  A METIS-style partition would read less; that
    only makes streaming better, so this is conservative about when gathering
    is the right answer.

    **3. The structure.**  CSR column indices (`E`) plus row pointers (`N+1`)
    plus one GCN normalisation coefficient per non-zero.  All contiguous, so no
    burst penalty -- but not free, and at small `feat_dim` not small either: a
    4 B index against a 32 B feature row is 12.5%, and against a 4-bit-quantised
    8 B row it is 50%.

    Not modelled: TLB and page behaviour, row-buffer hits from any accidental
    locality in the edge order, and any coalescing a real controller might do
    across two neighbours that happen to be adjacent.  All three make the real
    cost *lower* than `effective`, so this is an upper bound on the gather and
    the clamp is what keeps it a sane one.
    """
    row = max(1, feat_dim * act_bits // 8)
    logical = g.num_edges * row
    moved = (math.ceil(row / burst_bytes) * burst_bytes
             if burst_bytes > 0 else row)
    effective = g.num_edges * moved

    accum_resident = g.num_nodes * feat_dim * accum_bits // 8
    gather_res = feat_dim * accum_bits // 8
    cap = sram_capacity_kb * 1024
    passes = 1 if cap <= 0 else max(1, math.ceil(accum_resident / cap))
    stream = passes * g.num_nodes * row

    charged = min(effective, stream)
    structure = (g.num_edges * index_bytes
                 + (g.num_nodes + 1) * index_bytes
                 + g.num_edges * edge_weight_bits // 8)
    writeback = g.num_nodes * row

    return GatherCost(
        row_bytes=row, burst_bytes=burst_bytes,
        logical=logical, effective=effective,
        stream=stream, stream_passes=passes,
        charged=charged, stream_wins=stream < effective,
        dataflow='stream' if stream < effective else 'gather',
        structure=structure, writeback=writeback,
        total=charged + structure + writeback,
        burst_waste=effective / logical if logical else 1.0,
        flops=2 * g.num_edges * feat_dim,
        accum_resident=accum_resident, gather_resident=gather_res,
    )


# ============================================================================
# Combine -- straight through the existing simulator
# ============================================================================

def combine_metrics(sim: Simulator, num_nodes: int, f_in: int, f_out: int):
    """One `H @ W` GEMM through the unmodified simulator.

    `OperationType.FC1` is used deliberately: in `_calculate_memory_access` it
    selects the plain AW path -- weights read from DRAM as one contiguous
    block, no KV branch, no score spill -- which is exactly what a shared
    weight matrix is.  Nodes take `M`, the same slot `batch x seq_len` takes in
    a transformer, because a node is to a GNN layer what a token is to an FFN.
    """
    return sim._simulate_matmul(
        OperationType.FC1, ComputeMode.AW, (num_nodes, f_in, f_out),
        batch_size=1, is_decode=False,
    )


def dense_adjacency_metrics(sim: Simulator, num_nodes: int, feat_dim: int):
    """What the simulator would charge for `A_hat @ X` with `A_hat` DENSE.

    This is the honest measurement of what is missing: an AA matmul of shape
    `(N, N, F)`.  It is not a proposal, it is the null against which the sparse
    formula is scored in section D.
    """
    return sim._simulate_matmul(
        OperationType.ATTN_V_MATMUL, ComputeMode.AA, (num_nodes, num_nodes,
                                                      feat_dim),
        batch_size=1, is_decode=False,
    )


def roofline_ms(sim: Simulator, m) -> float:
    freq = sim.hw.freq_mhz * 1e6
    bw = sim.hw.dram_bandwidth_gbps * 1e9
    return sim._op_roofline_time(m, freq, bw) * 1e3


def fmt_bytes(x):
    for unit, div in (('TB', 1e12), ('GB', 1e9), ('MB', 1e6), ('KB', 1e3)):
        if abs(x) >= div:
            return f"{x/div:.2f} {unit}"
    return f"{x:.0f} B"


# ============================================================================
# Pre-flight
# ============================================================================

def preflight():
    """Seven checks. Each one is a way stage 1 could be silently wrong."""
    print("Pre-flight")
    sim = Simulator(base_hw())

    # 1. Combine really is the existing AW path: FLOPs are the GEMM's.
    g = get_graph_config('Cora')
    f_in, f_out = g.layer_shapes()[0]
    m = combine_metrics(sim, g.num_nodes, f_in, f_out)
    want = 2 * g.num_nodes * f_in * f_out
    assert m.flops == want, (m.flops, want)
    print(f"  1. Combine FLOPs = 2*N*F_in*F_out = {want:,} ok")

    # 2. Combine's DRAM read is the weight matrix and nothing else -- the whole
    #    claim that W is shared across nodes.  Node count must not appear.
    w_bytes = f_in * f_out * WEIGHT_BITS // 8
    assert m.dram_read == w_bytes, (m.dram_read, w_bytes)
    m2 = combine_metrics(sim, g.num_nodes * 10, f_in, f_out)
    assert m2.dram_read == m.dram_read, (m2.dram_read, m.dram_read)
    print(f"  2. Combine DRAM read = |W| = {w_bytes:,} B, 10x nodes: unchanged ok")

    # 3. Combine cycles are AFFINE in node count: a constant per-node marginal
    #    cost plus a fixed array fill.  Not exactly proportional -- the fill is
    #    real -- so the check is that the second difference vanishes, which is
    #    what "nodes are just more rows of the same GEMM" actually means.
    c1, c2, c3 = (combine_metrics(sim, k * g.num_nodes, f_in, f_out).cycles
                  for k in (1, 2, 3))
    assert c2 - c1 == c3 - c2, (c1, c2, c3)
    per_node = (c2 - c1) / g.num_nodes
    fill = c1 - per_node * g.num_nodes
    assert m2.cycles == round(per_node * g.num_nodes * 10 + fill), m2.cycles
    print(f"  3. Combine cycles affine in N: {per_node:.0f} cyc/node "
          f"+ {fill:.0f} fill, exact at 10x ok")

    # 4. The gather formula agrees with the simulator's own burst rounding.
    for tech_name in TECHS:
        t = memory_technology(tech_name)
        s = Simulator(base_hw(burst_bytes=t.burst_bytes))
        for feat in (16, 32, 128, 256):
            gc = gather_cost(g, feat, burst_bytes=t.burst_bytes)
            ref = s._dram_effective_bytes(gc.logical, gc.row_bytes)
            assert gc.effective == ref, (tech_name, feat, gc.effective, ref)
    print("  4. gather_cost effective == Simulator._dram_effective_bytes ok")

    # 5. burst_bytes = 0 is exactly inert -- effective == logical.
    gc0 = gather_cost(g, 16, burst_bytes=0)
    assert gc0.effective == gc0.logical and gc0.burst_waste == 1.0
    print("  5. burst_bytes=0 inert: effective == logical ok")

    # 6. The clamp is one-sided, and the dataflow crossover is exactly
    #    `passes < deg * waste`.  Checked at every capacity, not just the
    #    unlimited one, because the unlimited case is degenerate: E > N on
    #    every real graph, so streaming always wins when passes == 1.
    for name in list_graphs():
        gg = get_graph_config(name)
        for feat in (16, 256):
            for cap in CAPACITY_KB:
                gc = gather_cost(gg, feat, burst_bytes=64,
                                 sram_capacity_kb=cap)
                assert gc.charged <= gc.effective and gc.charged <= gc.stream
                assert gc.stream_wins == (
                    gc.stream_passes < gg.avg_degree * gc.burst_waste), (
                    name, feat, cap, gc.stream_passes, gg.avg_degree)
    for name in list_graphs():
        assert gather_cost(get_graph_config(name), 256,
                           burst_bytes=64).stream_wins, name
    print("  6. crossover is exactly passes < deg*waste; at unlimited "
          "capacity streaming wins on every graph ok")

    # 7. Capacity drives streaming's passes and nothing else -- in particular
    #    the gather's own residency is one row regardless of graph size, which
    #    is the entire reason gathering is ever the right answer.
    gc_u = gather_cost(get_graph_config('Reddit'), 256, burst_bytes=64,
                       sram_capacity_kb=0)
    gc_c = gather_cost(get_graph_config('Reddit'), 256, burst_bytes=64,
                       sram_capacity_kb=1024)
    assert gc_u.stream_passes == 1 and gc_c.stream_passes > 1
    assert gc_u.gather_resident == gc_c.gather_resident == 256 * 4
    assert gc_u.effective == gc_c.effective, "gather must be capacity-free"
    assert gc_c.stream > gc_u.stream
    print(f"  7. capacity: Reddit streaming needs {gc_c.stream_passes:,} passes "
          f"at 1 MB; gather needs {gc_u.gather_resident} B either way ok")

    # ---- stage 2 ----------------------------------------------------------
    gs = GNNSimulator(base_hw())

    # 8. The VPU null is exactly `E * F / vpu_width` -- no LUT term, no qbit
    #    term.  If this drifts, every ratio in section G is meaningless,
    #    because the VPU is the denominator of all of them.
    for feat in (16, 128, 256, 602):
        for e in (1, 10556, 114615892):
            assert gs.vpu_cycles(e, feat) == e * feat / gs.hw.vpu_width
    assert gs.vpu_width == 128 and gs.vpu_cycles(2, 64) == 1.0
    print(f"  8. VPU null == E*F/{gs.vpu_width} exactly ok")

    # 9. THE shape identity, proved rather than asserted: a real decode
    #    `attn_v` operation with (kv_len, head_dim) = (deg, F) must produce the
    #    identical cycle count to a pull aggregation of a degree-`deg` node at
    #    feature width `F`.  Same function, same three integers.  Everything in
    #    sections F-H rests on this one check.
    for deg in (1, 4, 17, 128, 492, 4096):
        for feat in (16, 64, 128, 256, 512):
            av = gs._simulate_matmul(
                OperationType.ATTN_V_MATMUL, ComputeMode.AA,
                (1, deg, feat), batch_size=1, is_decode=True,
                seq_len=1, kv_len=deg)
            pull = gs.pull_cycles(deg, feat, gs.hw.kv_cache_bits)
            assert av.cycles == pull, (deg, feat, av.cycles, pull)
    print("  9. pull aggregation == decode attn_v cycles, bit for bit, "
          "over 30 (deg, F) pairs ok")

    # 10. The fixed/useful split really is a split: it must reconstruct the
    #     cycle count, not merely resemble it.  This is what lets section F
    #     quote a fixed-overhead share.
    for deg in KNEE_DEGREES:
        for feat in (16, 256, 4096):
            for q in QBITS:
                f, u = gs.pull_fixed_useful(deg, feat, q)
                assert f + u == gs.pull_cycles(deg, feat, q), (deg, feat, q)
    print("  10. pull cycles == fixed + useful, exactly, over 132 cases ok")

    # 11. The distribution-aware sum reduces to the mean-degree one on a
    #     REGULAR graph.  Without this the two columns in section G could
    #     differ for a reason other than the distribution.
    reg = GraphConfig('regular-8', num_nodes=10_000, num_edges=80_000,
                      feat_dim=64, num_classes=4, hidden_dim=64,
                      min_degree=8, max_degree=8)
    assert reg.degree_distribution() == [(8, 10_000.0)]
    for flow in (PULL, PUSH, VPU):
        a = gs.aggregate_cost(reg, 64, 16, flow, distribution_aware=True)
        b = gs.aggregate_cost(reg, 64, 16, flow, distribution_aware=False)
        assert a.cycles == b.cycles, (flow, a.cycles, b.cycles)
    print("  11. regular graph: distribution-aware == mean-degree, all three "
          "dataflows ok")

    # 12. The synthesised distribution conserves both marginals it claims to.
    #     Node count is exact by construction; the edge count is what says the
    #     gamma fit converged, and it is the number a distribution-aware total
    #     is only comparable to a mean-degree one because of.
    for name in list_graphs():
        gg = get_graph_config(name)
        dist = gg.degree_distribution()
        assert abs(sum(c for _, c in dist) - gg.num_nodes) < 1e-6, name
        assert abs(gg.degree_edge_error()) < 1e-6, (name,
                                                    gg.degree_edge_error())
        assert 1.0 < gg.degree_exponent < 3.5, (name, gg.degree_exponent)
    print("  12. degree distributions conserve nodes and edges to 1e-6; "
          "fitted gamma in (1.0, 3.5) on all six ok")

    # 13. Push's two numbers are both simulator outputs -- `os_rounds_model`
    #     "tiled" and "packed" (stage 11) -- so the defect is quoted, not
    #     paraphrased, and they must differ by exactly the rounds ratio.
    #     `deg = 1` is excluded because a degree-1 source node takes the M==1
    #     branch, which both models agree on.
    worst = (0, 0, 1.0)
    for deg in range(2, 65):
        for feat in (256, 4096):
            model, corr = gs.push_cycles(deg, feat, 16)
            ref = gs._calculate_cycles(deg, 1, feat, 16, ComputeMode.AA,
                                       "LUT_OS_V", 1)
            assert model == ref, (deg, feat, model, ref)
            n_tiles = math.ceil(feat / (gs.hw.array_n * gs.NUM_RAC))
            ratio = (math.ceil(deg / gs.hw.array_m) * n_tiles
                     / math.ceil(deg * n_tiles / gs.hw.array_m))
            assert abs(model / corr - ratio) < 1e-9, (deg, feat)
            if ratio > worst[2]:
                worst = (deg, feat, ratio)
    print(f"  13. push model == _calculate_cycles; overcharge is exactly "
          f"ceil(M/32)*n_tiles/ceil(M*n_tiles/32), worst {worst[2]:.0f}x at "
          f"M={worst[0]}, F={worst[1]} ok")

    # 14. The searched crossover degree agrees with the closed form
    #     `d* = 10 * qbit / (F/vpu_width - qbit/MU)` up to the
    #     staircase `ceil(deg/MU)` introduces -- and both agree on WHEN there
    #     is no crossover, which is the finding section G leads with.
    for feat in (16, 128, 256, 512, 1024, 2048, 4096):
        for q in QBITS:
            slope = feat / gs.vpu_width - q / gs.MU
            found = gs.crossover_degree(feat, q, 200_000)
            if slope <= 0:
                assert found is None, (feat, q, found)
                continue
            closed = math.ceil((3 + 1 + 2 + gs.hw.array_n) * q / slope)
            assert found is not None and closed <= found <= closed + gs.MU, (
                feat, q, found, closed)
    print("  14. crossover: none exactly when F <= vpu_width*qbit/MU = 32*qbit; "
          "otherwise within MU of the closed form ok")

    # 15. The packed cycle count IS `PackedOSVSimulator`'s, on a real decode
    #     `attn_v`.  Stage 3 must not acquire a second packing model; this is
    #     the assertion that it did not.  216 shapes.
    n15 = 0
    for pk in (1, 2, 4, 8, 16, 32):
        ps = PackedOSVSimulator(base_hw(), pack=pk)
        for kv_len in (128, 1000, 2048, 4096):
            for head_dim in (64, 128, 256):
                for b in (8, 32, 256):
                    ps._op = (OperationType.ATTN_V_MATMUL, True)
                    got = ps._calculate_cycles(
                        1, kv_len, head_dim, ps.hw.kv_cache_bits,
                        ComputeMode.AA, "LUT_OS_V", b)
                    ps._op = None
                    mine = math.ceil(b / pk) * gs.packed_pass_cycles(
                        kv_len, head_dim, ps.hw.kv_cache_bits, pk)
                    assert got == mine, (pk, kv_len, head_dim, b, got, mine)
                    n15 += 1
    print(f"  15. packed_pass_cycles == PackedOSVSimulator attn_v, "
          f"{n15} shapes ok")

    # 16. `pack = 1` under the IDEAL schedule reproduces stage 2's pull
    #     numbers exactly.  Relative, not absolute: the two sum the same
    #     terms in the same order but through `sum()` vs `+=`, which differ
    #     in the last bit at ogbn-products' 9e8 cycles.
    for name in list_graphs():
        g = get_graph_config(name)
        for feat in (128, 256):
            st2 = gs.aggregate_cost(g, feat, ACT_BITS, PULL)
            st3 = gs.packed_aggregate_cost(g, feat, ACT_BITS, 1, IDEAL)
            assert math.isclose(st2.cycles, st3.cycles, rel_tol=1e-12), (
                name, feat, st2.cycles, st3.cycles)
            assert math.isclose(st3.baseline_cycles, st2.cycles,
                                rel_tol=1e-12), (name, feat)
    print("  16. pack=1 + ideal schedule == stage 2's pull, all graphs ok")

    # 17. Packing recovery saturates exactly at `array_m / n_tiles` -- the
    #     claim section I leads with, asserted rather than eyeballed.
    for feat in (64, 128, 256, 512, 1024, 2048, 4096):
        nt = math.ceil(feat / (gs.hw.array_n * gs.NUM_RAC))
        best, bestc = 1, None
        for pk in (1, 2, 4, 8, 16, 32):
            c = gs.packed_pass_cycles(32, feat, ACT_BITS, pk) / pk
            if bestc is None or c < bestc - 1e-9:
                bestc, best = c, pk
        assert best == max(1, gs.hw.array_m // nt), (feat, nt, best)
    print("  17. measured P* == array_m/n_tiles at every width ok")
    print()


# ============================================================================
# Sections
# ============================================================================

def sweep(report_path):
    rows = []
    rpt = Report(
        report_path,
        "GNN on Omni-LUT, stage 2",
        "Both halves of the layer simulated: Combine as an AW GEMM, Aggregate "
        "as decode `attn_v` with a graph in it",
        source='analysis/gnn/gnn_run.py',
        setup=[
            f"Array 32x4, MU=4, RAC=32, VPU {HardwareConfig.vpu_width} lanes, "
            f"500 MHz. act={ACT_BITS} b, weights={WEIGHT_BITS} b, "
            f"accum={ACCUM_BITS} b, AW/AA = OMNI.",
            f"CSR indices {INDEX_BYTES} B, GCN edge coefficients "
            f"{EDGE_WEIGHT_BITS} b (= the AA `qbit`, swept in G2).",
            "Graph statistics from `simulator/graph_configs.py`. Sections A-E "
            "(stage 1) use `avg_degree`; F-H (stage 2) sum over that module's "
            "**synthesised** power-law degree distribution, and G1 reports both "
            "so the difference is visible.",
            "Sections F-H add no code to `simulator/simulator.py`: the "
            "aggregation cycle model in `analysis/gnn/gnn_sim.py` is calls into "
            "the existing `_calculate_cycles`, and `baseline.py check` is "
            "unchanged by construction.",
        ],
    )

    # ---- A. the workloads --------------------------------------------------
    rpt.section(
        "A. The graph workloads",
        "Six standard benchmarks, spanning four orders of magnitude in node "
        "count and two in average degree. `Combine FLOPs` and `Aggregate "
        "FLOPs` are summed over all layers, combine-first ordering.")
    a_rows = []
    for name in list_graphs():
        g = get_graph_config(name)
        c_flops = sum(2 * g.num_nodes * fi * fo for fi, fo in g.layer_shapes())
        a_flops = sum(2 * g.num_edges * fo for _, fo in g.layer_shapes())
        a_rows.append([
            g.name, f"{g.num_nodes:,}", f"{g.num_edges:,}",
            f"{g.avg_degree:.1f}", f"{g.density:.1e}",
            f"{g.feat_dim}", f"{g.hidden_dim}",
            f"{c_flops/1e9:.2f}", f"{a_flops/1e9:.3f}",
            f"{a_flops/c_flops:.3f}",
        ])
        rows.append(dict(section='A', graph=g.name, nodes=g.num_nodes,
                         edges=g.num_edges, avg_degree=g.avg_degree,
                         density=g.density, feat_dim=g.feat_dim,
                         hidden_dim=g.hidden_dim, combine_gflops=c_flops/1e9,
                         aggregate_gflops=a_flops/1e9))
    rpt.table(
        ['graph', 'nodes', 'edges', 'deg', 'density', 'F_in', 'H',
         'combine GF', 'aggr GF', 'aggr/comb'],
        a_rows, aligns='lrrrrrrrrr')
    rpt.note(
        "Aggregation is a small fraction of the arithmetic almost everywhere "
        "-- 0.1% of layer FLOPs on CiteSeer, 9% on ogbn-arxiv, and only on "
        "Reddit's degree-492 graph does it reach 47%. That is exactly why it "
        "is the interesting half: it does little work and, as section E "
        "shows, moves nearly all the bytes.")

    rpt.section(
        "A2. The degree distributions, which are a model and not a dataset",
        "Sections F-H sum per-node costs over a degree distribution, and "
        "`avg_degree` alone cannot supply one. `GraphConfig.degree_distribution()` "
        "**synthesises** a discrete power law `p(d) ~ d^-gamma` on `[1, "
        "max_degree]` and solves the single free parameter `gamma` so the mean "
        "reproduces the published `avg_degree` exactly. `max_degree` is the "
        "commonly reported maximum for each dataset; **gamma is derived here, "
        "not taken from any paper**, and no bucket below is a dataset fact.")
    a2_rows = []
    for name in list_graphs():
        g = get_graph_config(name)
        dist = dict(g.degree_distribution())
        a2_rows.append([
            g.name, f"{g.avg_degree:.1f}", f"{g.max_degree:,}",
            f"{g.degree_exponent:.3f}",
            f"{dist[1]/g.num_nodes*100:.1f}%",
            f"{sum(c for d, c in dist.items() if d <= 4)/g.num_nodes*100:.1f}%",
            f"{g.degree_head_share(0.1)*100:.1f}%",
            f"{g.degree_edge_error():+.1e}",
        ])
        rows.append(dict(section='A2', graph=g.name, avg_degree=g.avg_degree,
                         max_degree=g.max_degree,
                         degree_exponent=g.degree_exponent,
                         deg1_share=dist[1]/g.num_nodes,
                         head10_edge_share=g.degree_head_share(0.1),
                         degree_edge_error=g.degree_edge_error()))
    rpt.table(
        ['graph', 'mean deg', 'max deg', 'fitted gamma', 'deg 1',
         'deg <= 4', 'edges in top 10% of nodes', 'edge error'],
        a2_rows, aligns='lrrrrrrr')
    rpt.note(
        "The fitted exponents land in 1.27-2.10. The citation graphs' 1.87-2.10 "
        "sits at the low edge of the 2.0-3.0 usually reported for such "
        "networks, and **Reddit's 1.27 should not be believed as a power law "
        "at all** -- Reddit is dense and its real minimum degree is far above "
        "1, so forcing `min_degree = 1` puts 24.7% of its nodes at degree 1 "
        "where the real graph has none. That error is in the VPU's favour and "
        "against the LUT, i.e. it makes the LUT look worse on the one graph "
        "where the LUT comes closest to winning. Worth knowing before quoting "
        "Reddit's 2.17x in G1 to three digits.")
    rpt.note(
        "`edge error` is the relative error in `num_edges` implied by the "
        "distribution, and it is ~1e-9 because the counts are kept "
        "**fractional**. Rounding them to integers is the obvious move and it "
        "is wrong: the tail degrees each have an expected count well below 1 "
        "while carrying most of the edges, so integer rounding silently loses "
        "up to **34%** of `num_edges` (measured on ogbn-arxiv while building "
        "this). Fractional counts are the only way the distribution-aware total "
        "in G1 is comparable to the mean-degree one at all.")

    # ---- B. Combine, simulated --------------------------------------------
    rpt.section(
        "B. Combine, measured through the unmodified simulator",
        "`_simulate_matmul(FC1, AW, (N, F_in, F_out))`. No simulator change "
        "was needed for any number in this table -- the GEMM core is "
        "shape-driven, and a node occupies the `M` slot a token occupies in an "
        "FFN. DRAM here is the weight matrix only; `bound` is the per-op "
        "roofline verdict at DDR5-6400.")
    sim = Simulator(base_hw())
    freq = sim.hw.freq_mhz * 1e6
    bw = sim.hw.dram_bandwidth_gbps * 1e9
    b_rows = []
    for name in list_graphs():
        g = get_graph_config(name)
        for li, (fi, fo) in enumerate(g.layer_shapes()):
            m = combine_metrics(sim, g.num_nodes, fi, fo)
            ct = m.cycles / freq
            mt = (m.dram_read_eff + m.dram_write_eff) / bw
            b_rows.append([
                g.name, f"L{li}", f"{g.num_nodes:,}x{fi}x{fo}",
                f"{m.cycles:,}", f"{m.flops/1e9:.3f}",
                f"{m.utilization*100:.1f}%",
                fmt_bytes(m.dram_read), fmt_bytes(m.peak_sram_bytes),
                f"{roofline_ms(sim, m):.3f}",
                'compute' if ct >= mt else 'memory',
            ])
            rows.append(dict(section='B', graph=g.name, layer=li, M=g.num_nodes,
                             K=fi, N=fo, cycles=m.cycles, gflops=m.flops/1e9,
                             utilization=m.utilization, dram_read=m.dram_read,
                             peak_sram=m.peak_sram_bytes,
                             roofline_ms=roofline_ms(sim, m)))
    rpt.table(
        ['graph', 'layer', 'M x K x N', 'cycles', 'GFLOP', 'util',
         'DRAM (W)', 'peak SRAM', 'ms', 'bound'],
        b_rows, aligns='llrrrrrrrl')
    rpt.note(
        "Combine is compute-bound everywhere, and utilisation tracks `F_out` "
        "against the array's `array_n * RAC = 128` output lanes almost "
        "exactly: `F_out = 256` gives 94-100%, the Planetoid `hidden_dim = 16` "
        "gives 11.5-12.3% against a 12.5% ceiling, and the 3-to-7-wide "
        "classifier heads give under 1%. Narrow GNN layers underuse this array "
        "for the same reason narrow projections do in the transformer studies "
        "-- and unlike the burst effect in C1, no memory technology helps.")
    rpt.note(
        "`DRAM (W)` is weights only. The AW path charges no activation DRAM at "
        "all -- it assumes `H` is SRAM-resident, which is what the transformer "
        "studies assume for an FFN. `peak SRAM` shows what that assumption "
        "costs: 1.74 GB for ogbn-products, against an on-chip budget of "
        "megabytes. Section E carries the correction rather than hiding it; "
        "this is the same untiled-activation limitation `study.md` section 7 "
        "and the bandwidth study already flag, arriving on a workload where "
        "`M` is millions of nodes instead of thousands of tokens.")

    # ---- C. the gather -----------------------------------------------------
    rpt.section(
        "C1. Aggregate: the burst penalty on a neighbour gather",
        "`gather_cost()` at the hidden width each graph actually uses. "
        "`logical` is what the gather asks for; `effective` is what DRAM moves "
        "once each neighbour's feature row is rounded up to a burst, because "
        "the row sits at an address the edge list picked.")
    c_rows = []
    for tech_name in TECHS:
        t = memory_technology(tech_name)
        for name in list_graphs():
            g = get_graph_config(name)
            gc = gather_cost(g, g.hidden_dim, burst_bytes=t.burst_bytes)
            c_rows.append([
                t.name, g.name, f"{g.hidden_dim}", f"{gc.row_bytes} B",
                f"{t.burst_bytes} B", f"{gc.burst_waste:.2f}x",
                fmt_bytes(gc.logical), fmt_bytes(gc.effective),
                fmt_bytes(gc.structure),
                f"{gc.structure/(gc.effective+gc.structure)*100:.0f}%",
            ])
            rows.append(dict(section='C1', graph=g.name, tech=t.name,
                             burst_bytes=t.burst_bytes, feat_dim=g.hidden_dim,
                             row_bytes=gc.row_bytes, logical=gc.logical,
                             effective=gc.effective, structure=gc.structure,
                             burst_waste=gc.burst_waste))
    rpt.table(
        ['tech', 'graph', 'F', 'row', 'burst', 'waste', 'logical',
         'effective', 'structure', 'struct %'],
        c_rows, aligns='llrrrrrrrr')
    rpt.note(
        "The 16-wide Planetoid hidden layer is a 32 B row against DDR5's 64 B "
        "burst: every neighbour read wastes exactly half a burst, 2.00x, on "
        "the most cited GNN configuration there is. HBM3's 32 B burst erases "
        "it exactly. This is HBM's *granularity* deciding the answer, not its "
        "bandwidth -- the same mechanism `study.md` section 15 found for 4-bit "
        "KV entries, with the feature width playing head_dim's role.")
    rpt.note(
        "The CSR indices and GCN coefficients are 10% of aggregation traffic "
        "at F=16 on DDR5, 18-19% on HBM3 (the same bytes against a feature "
        "term the smaller burst halved), and 1% at F=256. They are contiguous, "
        "so no burst penalty -- but at 4-bit quantised features an 8 B row "
        "against a 4 B index makes the graph structure a third of the traffic, "
        "which is the term a feature-quantisation study would hit first.")

    # ---- C2. the dataflow crossover ---------------------------------------
    rpt.section(
        "C2. Gather or stream: the crossover is capacity, not sparsity",
        "The gather is never charged more than streaming the whole feature "
        "matrix and pushing each row to its neighbours. That clamp is not a "
        "rare safety net here: `E > N` on every graph, so **push/stream always "
        "moves fewer feature bytes than pull/gather when the accumulators "
        "fit**. What makes gathering right is capacity. Streaming needs all "
        "`N x F` accumulators resident and re-reads `X` once per block that "
        "does not fit; the gather needs one row, always. DDR5-6400.")
    c2_rows = []
    for name in list_graphs():
        g = get_graph_config(name)
        F = g.hidden_dim
        base = gather_cost(g, F, burst_bytes=64)
        cells = []
        for cap in CAPACITY_KB:
            gc = gather_cost(g, F, burst_bytes=64, sram_capacity_kb=cap)
            cells += [f"{gc.stream_passes:,}", gc.dataflow]
            rows.append(dict(section='C2', graph=g.name, feat_dim=F,
                             sram_capacity_kb=cap,
                             stream_passes=gc.stream_passes,
                             stream=gc.stream, effective=gc.effective,
                             charged=gc.charged, dataflow=gc.dataflow,
                             accum_resident=gc.accum_resident,
                             gather_resident=gc.gather_resident))
        c2_rows.append([
            g.name, f"{F}", f"{g.avg_degree:.1f}",
            f"{g.avg_degree * base.burst_waste:.1f}",
            fmt_bytes(base.accum_resident), f"{base.gather_resident} B",
            fmt_bytes(base.effective), *cells,
        ])
    rpt.table(
        ['graph', 'F', 'deg', 'deg x waste', 'stream needs', 'gather needs',
         'gather bytes',
         'passes (inf)', 'wins', 'passes (1 MB)', 'wins',
         'passes (8 MB)', 'wins'],
        c2_rows, aligns='lrrrrrrrlrlrl')
    rpt.note(
        "The crossover is one inequality: **streaming wins iff "
        "`passes < avg_degree x burst_waste`.** Reddit's degree of 492 buys so "
        "much slack that streaming still wins at 8 MB despite needing 30 "
        "passes over a 238 MB accumulator. ogbn-products, at degree 50 and a "
        "2.4 GB accumulator, crosses over: it is the one graph here where a "
        "real chip must gather.")
    rpt.note(
        "This is the classic push-vs-pull dataflow choice, and the useful "
        "result is that on this hardware it is decided by SRAM capacity "
        "against average degree, with the DRAM burst as the multiplier on "
        "degree. A narrow hidden layer does not just waste bursts, it shifts "
        "the crossover -- doubling the waste doubles the degree slack that "
        "keeps streaming ahead.")

    # ---- D. sparse vs dense ------------------------------------------------
    rpt.section(
        "D. Sparse config vs the dense null",
        "What the simulator would charge today for `A_hat @ X` with `A_hat` "
        "treated as a dense `(N, N, F)` AA matmul, against the sparse formula. "
        "This is the measurement of what is missing, not a proposal.")
    d_rows = []
    for name in list_graphs():
        g = get_graph_config(name)
        F = g.hidden_dim
        gc = gather_cost(g, F, burst_bytes=64, sram_capacity_kb=8192)
        dense_flops = 2 * g.num_nodes * g.num_nodes * F
        dense_bytes = g.num_nodes * g.num_nodes * ACT_BITS // 8
        note = ''
        if name in SMALL:
            m = dense_adjacency_metrics(sim, g.num_nodes, F)
            dense_cycles = f"{m.cycles:,}"
            dense_ms = f"{roofline_ms(sim, m):.1f}"
            assert m.flops == dense_flops, (m.flops, dense_flops)
        else:
            dense_cycles = 'not run'
            dense_ms = '--'
            note = 'A_hat alone exceeds addressable memory'
        d_rows.append([
            g.name, f"{gc.flops/1e9:.4g}", f"{dense_flops/1e9:.4g}",
            f"{dense_flops/gc.flops:,.0f}x",
            fmt_bytes(gc.total), fmt_bytes(dense_bytes),
            f"{dense_bytes/gc.total:,.0f}x", dense_cycles, dense_ms, note,
        ])
        rows.append(dict(section='D', graph=g.name,
                         sparse_gflops=gc.flops/1e9,
                         dense_gflops=dense_flops/1e9,
                         flop_ratio=dense_flops/gc.flops,
                         sparse_bytes=gc.total, dense_bytes=dense_bytes,
                         byte_ratio=dense_bytes/gc.total))
    rpt.table(
        ['graph', 'sparse GF', 'dense GF', 'FLOP ratio', 'sparse B',
         'dense A_hat B', 'byte ratio', 'dense cycles', 'dense ms', ''],
        d_rows, aligns='lrrrrrrrrl')
    rpt.note(
        "The ratio is `N / avg_degree`, i.e. `1 / density` -- 695x on Cora, "
        "48,479x on ogbn-products. A dense `A_hat` is not a conservative "
        "approximation of a sparse one, it is a different problem. Section C's "
        "formula is what stage 2 has to put behind an operation type; running "
        "the dense shape and calling it aggregation would be meaningless.")
    rpt.note(
        "The dense adjacency matrix for ogbn-products would be 12 exabytes at "
        "FP16, so those three rows are arithmetic, not simulator output. The "
        "three that were run agree with the closed form to the FLOP "
        "(asserted).")

    # ---- E. the balance ----------------------------------------------------
    rpt.section(
        "E. Where the layer's time and bytes actually go",
        "Combine from section B (simulated), Aggregate from section C "
        "(formula), summed over layers at DDR5-6400 with an 8 MB SRAM -- the "
        "cheaper dataflow per layer, from C2. `aggr byte share` is the "
        "question the whole exercise is about.")
    e_rows = []
    for name in list_graphs():
        g = get_graph_config(name)
        c_bytes = c_cycles = c_act = 0
        a_bytes = a_flops = 0
        for fi, fo in g.layer_shapes():
            m = combine_metrics(sim, g.num_nodes, fi, fo)
            c_bytes += m.dram_read_eff + m.dram_write_eff
            c_cycles += m.cycles
            # What the AW path does NOT charge: reading H and writing H@W.
            # Contiguous, so no burst term; an upper bound assuming nothing is
            # kept on chip between layers, as the peak-SRAM column says it
            # cannot be.
            c_act += (g.num_nodes * fi * ACT_BITS // 8
                      + g.num_nodes * fo * ACT_BITS // 8)
            gc = gather_cost(g, fo, burst_bytes=64, sram_capacity_kb=8192)
            a_bytes += gc.total
            a_flops += gc.flops
        c_ms = c_cycles / freq * 1e3
        a_ms = a_bytes / bw * 1e3     # aggregation is memory-bound by assumption
        e_rows.append([
            g.name, fmt_bytes(c_bytes), fmt_bytes(c_act), fmt_bytes(a_bytes),
            f"{a_bytes/(a_bytes+c_bytes)*100:.1f}%",
            f"{a_bytes/(a_bytes+c_bytes+c_act)*100:.1f}%",
            f"{c_ms:.3f}", f"{a_ms:.3f}",
            f"{a_ms/(a_ms+c_ms)*100:.1f}%",
            f"{a_flops/a_bytes:.2f}",
        ])
        rows.append(dict(section='E', graph=g.name, combine_bytes=c_bytes,
                         combine_activation_bytes=c_act,
                         aggregate_bytes=a_bytes, combine_ms=c_ms,
                         aggregate_ms=a_ms,
                         aggregate_intensity=a_flops/a_bytes))
    rpt.table(
        ['graph', 'combine B (W)', 'combine B (act)', 'aggr B',
         'aggr share (W only)', 'aggr share (+act)', 'combine ms',
         'aggr ms', 'aggr time share', 'aggr FLOP/B'],
        e_rows, aligns='lrrrrrrrrr')
    rpt.note(
        "**The two share columns are the honest bracket.** `combine B (W)` is "
        "what the simulator charges -- weights only. `combine B (act)` is the "
        "feature matrix the AW path assumes is SRAM-resident and section B's "
        "peak-SRAM column shows is not, added here as an upper bound. **The "
        "correction changes the answer on the small graphs and not on the "
        "large ones.** Cora, CiteSeer and PubMed have 500-to-3703-wide raw "
        "features over a few thousand nodes, so the activation term dominates "
        "and aggregation falls from 93-100% to 1.7-11.4%. On the three large "
        "graphs it stays at 86-96%. The uncorrected column is not usable on "
        "its own, and stage 6 of the memory plan (`m_tile_rows`) is what would "
        "settle the middle properly.")
    balance = (2 * Simulator.LANES_EQUIV * freq) / bw
    rpt.note(
        "Aggregation's arithmetic intensity is 0.9-1.2 FLOP/byte on five of "
        "the six graphs -- Reddit reaches 13.4 only because degree 492 lets "
        "streaming amortise one feature read over 492 edges. This array's "
        f"compute-to-DDR5 balance point is `2 x {Simulator.LANES_EQUIV} lanes "
        f"x {freq/1e6:.0f} MHz / {bw/1e9:.1f} GB/s` = **{balance:.0f} "
        "FLOP/byte**, so aggregation sits roughly two orders of magnitude "
        "inside the memory-bound region. No LUT arrangement changes that: the "
        "LUT amortises over a *reused* operand, and aggregation's operand is "
        "the graph, touched once.")
    rpt.note(
        "The `aggr ms` column charges aggregation at pure DRAM bandwidth with "
        "no compute term, which is the right bound given the intensity above "
        "but is not a simulated latency, and it is not overlapped with "
        "combine (`study.md` section 17's `overlap_model` would bracket that). "
        "**Section H replaces it with a simulated one, and reverses it**: with "
        "a cycle model, aggregation is compute-bound on every graph.")

    # ---- F. the shape identity --------------------------------------------
    gs = GNNSimulator(base_hw(burst_bytes=64, sram_capacity_kb=8192))
    rpt.section(
        "F. Aggregation is decode `attn_v` with a graph in it",
        "Pull aggregation -- `h[v] = sum a_vu x[u]` over v's neighbours -- is "
        "issued as `(M=1, K=deg(v), N=F)`. That is the shape decode `attn_v` "
        "is issued as, with `deg` in `kv_len`'s slot and `F` in `head_dim`'s. "
        "Pre-flight 9 proves it: a real decode `attn_v` operation at "
        "`(kv_len, head_dim) = (deg, F)` returns the identical cycle count over "
        "30 pairs. So `study.md` section 4(b)'s fixed-overhead knee is not "
        "analogous to aggregation's, it *is* aggregation's. Per destination "
        "node, F=256, 16-bit adjacency coefficients.")
    f_rows = []
    for deg in KNEE_DEGREES:
        cyc = gs.pull_cycles(deg, 256, ACT_BITS)
        fx, us = gs.pull_fixed_useful(deg, 256, ACT_BITS)
        vpu = gs.vpu_cycles(deg, 256)
        # The N-null, checked rather than claimed: identical at every width the
        # array can hold in one round.
        for feat in (16, 128, 256, 4096):
            assert gs.pull_cycles(deg, feat, ACT_BITS) == cyc, (deg, feat)
        f_rows.append([
            f"{deg}", f"{cyc:,}", f"{fx:,}", f"{us:,}", f"{fx/cyc*100:.1f}%",
            f"{vpu:,.1f}", f"{cyc/vpu:.1f}x",
        ])
        rows.append(dict(section='F', degree=deg, feat_dim=256, qbit=ACT_BITS,
                         pull_cycles=cyc, fixed=fx, useful=us,
                         vpu_cycles=vpu, lut_over_vpu=cyc/vpu))
    rpt.table(
        ['deg(v)', 'pull cycles', 'fixed', 'useful', 'fixed %', 'VPU cycles',
         'LUT / VPU'],
        f_rows, aligns='rrrrrrr')
    rpt.note(
        "**The knee that section 4(b) needed a 0.4% KV budget to reach is "
        "where a citation graph starts.** `per_round = 3 (LGU) + ceil(deg/4) + "
        "1 + array_n + 2`, so 10 cycles are fixed and `ceil(deg/4)` is useful. "
        "At Cora's mean degree of 3.9 that is 1 useful cycle against 10 -- "
        "90.9% overhead, against the 23.5% that was the most extreme point in "
        "the KV study. Reddit's degree 492 is the only graph here that gets the "
        "fixed share down to single figures (7.5%).")
    rpt.note(
        "**The N-null is wider than section 5 states, and it does not help.** "
        "Section 5 found `attn_v` flat in `head_dim` because `n_tiles = "
        "ceil(N/128) = 1` for `N <= 128`. The `rounds = ceil(n_tiles/array_m)` "
        "term extends that by another 32x: pull cycles are *identical* for "
        "every `F <= array_m x array_n x NUM_RAC = 4096` (asserted in this "
        "section for F = 16, 128, 256 and 4096). Every feature width any GNN "
        "uses is inside the null. The VPU's cost is linear in F throughout, so "
        "a wider feature is exactly where the LUT gains -- see G3.")
    rpt.note(
        "`M = 1` also means 1 of 32 PE rows does work at every degree, the "
        "3.12% occupancy of section 14. `array_pack.py`'s P-way packing applies "
        "unchanged and would recover up to 32x here too, since consecutive "
        "destination nodes are exactly the independent instances it packs. "
        "That is stage 3's first move and it is not modelled below.")

    # ---- G1. the three dataflows ------------------------------------------
    rpt.section(
        "G1. Pull, push and the VPU null, at the configured 16-bit coefficient",
        "Cycles for one whole-graph aggregation at each graph's hidden width. "
        "`pull (mean)` uses `round(avg_degree)` for every node; `pull (dist)` "
        "sums over the synthesised power-law degree distribution. **`push "
        "charged` is what the default `os_rounds_model = 'tiled'` bills and "
        "it is too high**: push is issued as `M = deg(u)`, and that model's "
        "OS-V `rounds` multiplies `ceil(M/32)` by `n_tiles` instead of packing, "
        "overcharging by up to 16x for `M` in 2..31 -- most nodes in most "
        "graphs. `push corrected` is the same simulator with stage 11's "
        "`os_rounds_model = 'packed'`. Both are shown; the charged column is "
        "not a number to quote alone.")
    g_rows = []
    for name in list_graphs():
        g = get_graph_config(name)
        F = g.hidden_dim
        pm = gs.aggregate_cost(g, F, ACT_BITS, PULL, distribution_aware=False)
        pd = gs.aggregate_cost(g, F, ACT_BITS, PULL, distribution_aware=True)
        pu = gs.aggregate_cost(g, F, ACT_BITS, PUSH, distribution_aware=True)
        vp = gs.aggregate_cost(g, F, ACT_BITS, VPU, distribution_aware=True)
        g_rows.append([
            g.name, f"{F}", f"{g.avg_degree:.1f}",
            f"{pm.cycles:.3e}", f"{pd.cycles:.3e}",
            f"{pd.cycles/pm.cycles:.3f}x", f"{pd.fixed_share*100:.1f}%",
            f"{pu.cycles:.3e}", f"{pu.cycles_corrected:.3e}",
            f"{vp.cycles:.3e}", f"{pd.cycles/vp.cycles:.1f}x",
        ])
        rows.append(dict(section='G1', graph=g.name, feat_dim=F,
                         qbit=ACT_BITS, pull_mean=pm.cycles,
                         pull_dist=pd.cycles, pull_fixed_share=pd.fixed_share,
                         push_charged=pu.cycles,
                         push_corrected=pu.cycles_corrected,
                         vpu_cycles=vp.cycles,
                         lut_over_vpu=pd.cycles/vp.cycles))
    rpt.table(
        ['graph', 'F', 'deg', 'pull (mean)', 'pull (dist)', 'dist/mean',
         'pull fixed %', 'push charged', 'push corrected', 'VPU',
         'pull / VPU'],
        g_rows, aligns='lrrrrrrrrrr')
    rpt.note(
        "**The LUT loses to the VPU on every graph at the configured 16-bit "
        "coefficient -- by 2.2x on Reddit and 529x on CiteSeer.** Not a knee, "
        "a slope: pull costs `(10 + ceil(deg/4)) x qbit` and the VPU "
        "costs `deg x F / 128`, so the per-degree slopes are `qbit/4` and "
        "`F/128`. At `qbit = 16` and `F = 256` that is 4 against 2, and no "
        "degree can rescue a losing slope. G3 solves that condition.")
    rpt.note(
        "**The degree distribution changes the total by 0.2-6.5%, and the "
        "premise that it would change it a lot is wrong.** Both cost models are "
        "*affine* in degree, so summing over the distribution and evaluating at "
        "the mean agree up to the `ceil(deg/4)` staircase -- Jensen has almost "
        "nothing to bite on here. What the distribution does change is the "
        "*spread*: on Cora 58.8% of nodes have degree 1 and spend 10 of 11 "
        "cycles on overhead, while the top 10% of nodes own 60.7% of the edges. "
        "The distribution is still the right thing to model -- but the mean "
        "turned out to be an adequate summary, and that is a measurement, not "
        "an assumption. It would not survive a cost model with a `deg^2` term.")
    rpt.note(
        "**Push loses to pull on every graph, on both the charged and the "
        "corrected number** -- 2.0-10.0x charged, 2.0-9.8x corrected. The "
        "correction matters most on ogbn-arxiv (5.11x -> 3.86x) and not at all "
        "on the Planetoid graphs, where `F = 16` gives `n_tiles = 1` and the "
        "two formulas coincide identically. The reason push loses is not the "
        "defect: with `K = 1` its `per_round` is `3 + 1 + array_m + array_n + "
        "2 = 42` cycles for a rank-1 update, i.e. it pays the full 32-row "
        "systolic fill to broadcast one source row. Push is the wrong shape for "
        "an output-stationary array and the defect only makes it look worse "
        "than it is.")

    # ---- G2. the bit-width sweep ------------------------------------------
    rpt.section(
        "G2. Adjacency-coefficient precision is the design lever",
        "`qbit` for an AA operation is `hw.kv_cache_bits`; here it is the "
        "precision of the GCN coefficient `a_vu`, the operand the LUT's "
        "bit-plane loop iterates over. Section 13 makes it the only axis that "
        "multiplies cycles *and* bytes. Ratio of pull-LUT cycles to VPU "
        "cycles; below 1.00x the LUT wins.")
    g2_rows = []
    for name in list_graphs():
        g = get_graph_config(name)
        F = g.hidden_dim
        cells = []
        for q in QBITS:
            pd = gs.aggregate_cost(g, F, q, PULL, distribution_aware=True)
            vp = gs.aggregate_cost(g, F, q, VPU, distribution_aware=True)
            cells.append(f"{pd.cycles/vp.cycles:.2f}x")
            rows.append(dict(section='G2', graph=g.name, feat_dim=F, qbit=q,
                             pull_dist=pd.cycles, vpu_cycles=vp.cycles,
                             lut_over_vpu=pd.cycles/vp.cycles))
        g2_rows.append([g.name, f"{F}", f"{g.avg_degree:.1f}"] + cells)
    rpt.table(['graph', 'F', 'deg'] + [f"{q} b" for q in QBITS],
              g2_rows, aligns='lrrrrrr')
    rpt.note(
        "**The stage-2 hypothesis was a 4-bit statement, and at 4 bits it "
        "holds almost exactly.** It predicted the VPU winning ~90x on Cora, "
        "~2x on ogbn-arxiv, roughly parity on ogbn-products and the LUT winning "
        "~1.9x on Reddit. Measured at `qbit = 4`: **95.0x, 2.04x, 0.92x and "
        "0.54x (LUT wins 1.84x)** -- every one inside 10%. **At the configured "
        "`qbit = 16` it is false on all four**, because cycles are exactly "
        "linear in `qbit` while the VPU has no `qbit` term at all: the LUT "
        "prices per bit-plane, the VPU per element. Quadrupling coefficient "
        "precision quadruples aggregation and leaves the null untouched.")
    rpt.note(
        "16-bit is what `base_hw()` configures and what every other number in "
        "this report uses, so the 16-bit column is the one describing this "
        "machine. Whether 4-bit GCN coefficients are accurate enough is an "
        "accuracy question this repo does not answer -- but `a_vu = "
        "1/sqrt(d_u d_v)` is a *deterministic function of the two degrees*, "
        "not a learned value, so it is quantisable without retraining anything, "
        "and at 4 bits it is the difference between the LUT losing 2.2x and "
        "winning 1.8x on Reddit.")

    # ---- G3. where the LUT can win at all ----------------------------------
    rpt.section(
        "G3. The window, solved rather than swept",
        "Smallest degree at which pull-LUT beats the VPU, searched over degree "
        "(`gnn_sim.crossover_degree`). `none` means no degree wins, at any "
        "size. Pre-flight 14 checks every entry against the closed form "
        "`d* = 10 x qbit / (F/vpu_width - qbit/MU)`.")
    g3_rows = []
    for feat in (16, 64, 128, 256, 512, 1024, 2048, 4096, 8192):
        cells = []
        for q in QBITS:
            d = gs.crossover_degree(feat, q, 200_000)
            cells.append('none' if d is None else f"{d}")
            rows.append(dict(section='G3', feat_dim=feat, qbit=q,
                             crossover_degree=(d if d else 0)))
        g3_rows.append([f"{feat}"] + cells)
    rpt.table(['F'] + [f"{q} b" for q in QBITS], g3_rows, aligns='rrrrr')
    rpt.note(
        "**The condition is on the feature width, not the degree.** LUT cycles "
        "grow with degree at `qbit/MU` per unit; VPU cycles at `F/vpu_width`. "
        "So a crossover exists at all iff `F > vpu_width x qbit / MU = "
        "32 x qbit`, and degree only decides where inside that regime you land. "
        "At 16-bit coefficients the LUT needs `F > 512` -- twice the widest "
        "hidden layer any of these six benchmarks uses. **Degree was the wrong "
        "variable to hypothesise about.**")
    rpt.note(
        "**And the window has a ceiling, for the same reason section 5's null "
        "has an edge.** Above `F = 4096` the N-null ends, `rounds` grows with "
        "`F`, and LUT and VPU cycles scale together -- the F terms cancel "
        "exactly. The 8192 row is identical to the 4096 row, which is that "
        "arithmetic showing up as a measurement. So the LUT beats a VPU at "
        "aggregation in exactly one band: **`32 x qbit < F <= 4096`** -- empty "
        "for `qbit >= 128`, and `512 < F <= 4096` at the configured 16 bits.")

    # ---- H. the first end-to-end layer -------------------------------------
    rpt.section(
        "H. End-to-end GNN layer: both halves simulated",
        "Combine from `_simulate_matmul` (section B) and Aggregate from "
        "`_calculate_cycles` via the pull shape, both priced through the same "
        "`Simulator._op_roofline_time` -- `max(compute, DRAM)` per operation, "
        "summed over layers, DDR5-6400 with an 8 MB SRAM. Aggregation's DRAM is "
        "stage 1's `gather_cost` with the cheaper dataflow from C2. `aggr "
        "bound` says which of the two terms won.")
    h_rows = []
    for name in list_graphs():
        g = get_graph_config(name)
        c_ms = a_ms = a_vpu_ms = a_cyc_ms = a_dram_ms = 0.0
        for fi, fo in g.layer_shapes():
            cm = combine_metrics(sim, g.num_nodes, fi, fo)
            c_ms += roofline_ms(sim, cm)
            gc = gather_cost(g, fo, burst_bytes=64, sram_capacity_kb=8192)
            ac = gs.aggregate_cost(g, fo, ACT_BITS, PULL, True, gather=gc)
            av = gs.aggregate_cost(g, fo, ACT_BITS, VPU, True, gather=gc)
            a_ms += gs.aggregate_roofline_ms(ac)
            a_vpu_ms += gs.aggregate_roofline_ms(av)
            a_cyc_ms += ac.cycles / freq * 1e3
            a_dram_ms += (ac.dram_read_eff + ac.dram_write_eff) / bw * 1e3
        total = c_ms + a_ms
        h_rows.append([
            g.name, f"{c_ms:.3f}", f"{a_cyc_ms:.3f}", f"{a_dram_ms:.3f}",
            f"{a_ms:.3f}", 'compute' if a_cyc_ms >= a_dram_ms else 'memory',
            f"{total:.3f}", f"{a_ms/total*100:.1f}%",
            f"{a_vpu_ms:.3f}", f"{(c_ms+a_vpu_ms):.3f}",
            f"{total/(c_ms+a_vpu_ms):.2f}x",
        ])
        rows.append(dict(section='H', graph=g.name, combine_ms=c_ms,
                         aggregate_cycles_ms=a_cyc_ms,
                         aggregate_dram_ms=a_dram_ms,
                         aggregate_ms=a_ms, layer_ms=total,
                         aggregate_vpu_ms=a_vpu_ms,
                         layer_ms_vpu=c_ms+a_vpu_ms))
    rpt.table(
        ['graph', 'combine ms', 'aggr compute ms', 'aggr DRAM ms',
         'aggr ms', 'aggr bound', 'layer ms', 'aggr share',
         'aggr ms (VPU)', 'layer ms (VPU)', 'VPU speedup'],
        h_rows, aligns='lrrrrlrrrrr')
    rpt.note(
        "**Stage 1 predicted aggregation would be memory-bound, and with a "
        "cycle model it is not: it is compute-bound on all six graphs.** "
        "Section E charged it at pure DRAM bandwidth because there was nothing "
        "else to charge it at, and section E's own arithmetic intensity of "
        "0.9-1.2 FLOP/byte against an 80 FLOP/byte balance point said that was "
        "right. It was right about the FLOPs and wrong about the cycles: the "
        "LUT does not spend its cycles on FLOPs. `aggr compute ms` is 2.4x the "
        "DRAM term on ogbn-products, its narrowest margin, and 251x on Cora; "
        "on the small graphs essentially all of it is the fixed 10 cycles per "
        "node.")
    rpt.note(
        "**Moving aggregation off the LUT and onto the VPU makes the whole "
        "layer 2.3-18.5x faster** at 16-bit coefficients. Aggregation is 75-99% "
        "of the layer with the LUT doing it, and 2-95% with the VPU doing it. "
        "This is stage 1's closing prediction confirmed with a different "
        "mechanism than it named: Omni-LUT is an excellent Combine engine and "
        "the wrong shape for Aggregate -- because of `M = 1` and the fixed 10, "
        "not because of bandwidth. **ogbn-products is the one graph where the "
        "VPU drives compute under DRAM** -- its 1.54 s is section E's gather "
        "traffic and nothing else, the memory-bound regime stage 1 described, "
        "and it is exactly the graph C2 found must gather. Reddit's 0.53 s is "
        "still compute (531 ms against 99 ms of DRAM): at degree 492 even a "
        "128-lane VPU has real arithmetic to do.")
    rpt.note(
        "Uncharged on the aggregation side, all of it in the LUT's favour: "
        "index decode, the CSR row-pointer walk, the scatter-add into the "
        "destination accumulator, and any bubble between two nodes of different "
        "degree. Those are per-node costs, so they add to the fixed 10 rather "
        "than to the useful `ceil(deg/4)`. Uncharged on the combine side, in "
        "the other direction: section E's `combine B (act)` activation traffic, "
        "which the AW path assumes is SRAM-resident. Both halves are therefore "
        "lower bounds, and section E's bracket says the aggregation share is "
        "the more reliable of the two on the three large graphs.")

    hw = gs.hw
    # ---- I. how far packing can go -----------------------------------------
    rpt.section(
        "I. P-way destination packing: the ceiling is the tile count",
        "Aggregation's `M = 1` was section G's problem -- one destination node "
        "cannot fill 32 array rows. It is also the opportunity "
        "`analysis/array_packing/` already exploits for decode `attn_v`: `P` "
        "destinations are independent instances, so they can share one OS-V "
        "pass, each owning `array_m / P` rows. Cycles come from "
        "`array_pack.packed_osv_cycles` -- the attention packing model, "
        "unmodified -- and pre-flight 15 asserts a real decode `attn_v` through "
        "`PackedOSVSimulator` returns the identical count over 216 shapes.")
    i_rows = []
    for F in [64, 128, 256, 512, 1024, 2048, 4096]:
        nt = math.ceil(F / (hw.array_n * gs.NUM_RAC))
        base = gs.packed_pass_cycles(32, F, ACT_BITS, 1)
        recs, best, bestc = [], 1, None
        for P in [1, 2, 4, 8, 16, 32]:
            c = gs.packed_pass_cycles(32, F, ACT_BITS, P) / P
            recs.append(f"{base / c:.2f}x")
            if bestc is None or c < bestc - 1e-9:
                bestc, best = c, P
        i_rows.append([str(F), str(nt), f"{hw.array_m / nt:.2f}x"] + recs
                      + [str(best)])
        rows.append(dict(section='I', feat_dim=F, n_tiles=nt,
                         pack_bound=hw.array_m / nt, pack_star=best))
    rpt.table(
        ['F', 'n_tiles', 'bound', 'P=1', 'P=2', 'P=4', 'P=8', 'P=16', 'P=32',
         'P*'],
        i_rows, aligns='rrrrrrrrrr')
    rpt.note(
        "**Recovery is `P / ceil(P x n_tiles / array_m)` and saturates at "
        "`array_m / n_tiles`, exactly.** A destination needs `n_tiles` rows, so "
        "once `array_m / P` has fallen to `n_tiles` the row groups are exactly "
        "sized and further packing would have to split a row it cannot split. "
        "The measured `P*` equals the bound at every width in the table, with "
        "no rounding slack -- 32x at `F <= 128`, halving per doubling of `F`, "
        "and **1.00x at `F = 4096`**, where packing buys nothing at all. "
        "Packing and the N-null end at the same width for the same reason: "
        "both are statements about `n_tiles` reaching `array_m`.")

    # ---- J. does packing move the band? ------------------------------------
    rpt.section(
        "J. The band, re-measured under packing",
        "G3 found the LUT beats a VPU only for `32 x qbit < F <= 4096`, which "
        "excluded every benchmark here. Packing divides the per-node LUT cost "
        "by up to `P` and should therefore divide the band's lower edge by `P`. "
        "Searched, not solved -- the `ceil` still makes the closed form off by "
        "one near the knee.")
    j_rows = []
    for P in [1, 2, 4, 8, 16, 32]:
        first = None
        for F in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
            d = gs.packed_crossover_degree(F, ACT_BITS, P, max_degree=5000)
            if d is not None:
                first = (F, d)
                break
        j_rows.append([str(P), f"{32 * ACT_BITS / P:.0f}",
                       str(first[0]) if first else 'never',
                       str(first[1]) if first else '--'])
        rows.append(dict(section='J', pack=P,
                         band_edge=32 * ACT_BITS / P,
                         first_F=first[0] if first else 0,
                         crossover_deg=first[1] if first else 0))
    rpt.table(['P', 'band edge 32xq/P', 'first winning F', 'at degree'],
              j_rows, aligns='rrrr')
    rpt.note(
        "**The band's lower edge moves exactly as `32 x qbit / P`, and the "
        "crossover degree does not move at all.** Every row crosses at degree "
        "43. That invariance is not a coincidence: packing divides LUT cycles "
        "per node by `P`, and the width that first qualifies also falls by `P`, "
        "which divides the VPU's `F / vpu_width` by the same factor. The two "
        "sides scale together and the degree at which they meet is preserved. "
        "**So packing widens the band without ever making a sparse graph "
        "cheaper to gather -- degree 43 is the entry fee at every `P`.**")

    # ---- K. the schedule ---------------------------------------------------
    rpt.section(
        "K. Scheduling the packs, and the verdict on real graphs",
        "A packed pass issues one `ceil(K/MU)` operand stream for all `P` "
        "nodes in it, so a pass costs the *maximum* degree it contains. Three "
        "schedules: `ideal` groups exactly-equal degrees and does not charge "
        "partial passes (a lower bound, bit-identical to stage 2 at `P = 1`); "
        "`exact` is the same grouping with whole passes; `sorted` sorts by "
        "degree and fills greedily. `P = 32`, `F = 256`, 16-bit.")
    k_rows = []
    for name in list_graphs():
        g = get_graph_config(name)
        cs = {sch: gs.packed_aggregate_cost(g, 256, ACT_BITS, 32, sch)
              for sch in SCHEDULES}
        b = cs[IDEAL].baseline_cycles
        k_rows.append([
            g.name, f"{b:.3e}",
            f"{b / cs[IDEAL].cycles:.2f}x", f"{b / cs[SORTED].cycles:.2f}x",
            f"{b / cs[EXACT].cycles:.2f}x",
            f"{cs[SORTED].degree_inflation:.2f}x"])
        rows.append(dict(section='K', graph=g.name, baseline_cycles=b,
                         ideal_recovery=b / cs[IDEAL].cycles,
                         sorted_recovery=b / cs[SORTED].cycles,
                         exact_recovery=b / cs[EXACT].cycles,
                         sorted_deg_inflation=cs[SORTED].degree_inflation))
    rpt.table(['graph', 'baseline cyc', 'ideal', 'sorted', 'exact',
               'deg inflation'], k_rows, aligns='lrrrrr')
    rpt.note(
        "**Sorting is within 4% of the unreachable bound; grouping by equal "
        "degree is catastrophic.** `sorted` lands at 15.40x-15.95x against "
        "`ideal`'s 16.00x, inflating the charged degree by only 1.01x-1.43x, "
        "because sorting bounds a pass's overcharge by the *bucket width* "
        "rather than by the spread of the distribution. `exact` reaches "
        "**0.05x on ogbn-arxiv -- twenty times slower than not packing at "
        "all** -- because a power-law tail holds thousands of buckets with "
        "fewer than `P` nodes and each still costs a whole pass. The intuition "
        "that the schedule is the hard part is wrong; only the naive schedule "
        "is.")

    k2_rows = []
    for name in list_graphs():
        g = get_graph_config(name)
        for fi, fo in g.layer_shapes():
            nt = math.ceil(fo / (hw.array_n * gs.NUM_RAC))
            P = max(1, hw.array_m // nt)
            while hw.array_m % P:
                P -= 1
            c1 = gs.packed_aggregate_cost(g, fo, ACT_BITS, 1, SORTED)
            cp = gs.packed_aggregate_cost(g, fo, ACT_BITS, P, SORTED)
            vpu = gs.vpu_cycles(g.num_edges, fo)
            k2_rows.append([
                g.name, f"{g.avg_degree:.1f}", str(fo), str(P),
                f"{c1.cycles:.3e}", f"{cp.cycles:.3e}", f"{vpu:.3e}",
                ('**%.2fx**' % (vpu / cp.cycles)) if cp.cycles < vpu
                else f"{vpu / cp.cycles:.2f}x"])
            rows.append(dict(section='K2', graph=g.name, feat_dim=fo, pack=P,
                             pull_p1_cycles=c1.cycles,
                             pull_packed_cycles=cp.cycles, vpu_cycles=vpu,
                             lut_speedup=vpu / cp.cycles))
    rpt.table(
        ['graph', 'avg deg', 'F_out', 'P*', 'pull P=1', 'pull P=P*', 'VPU',
         'LUT vs VPU'], k2_rows, aligns='lrrrrrrr')
    rpt.note(
        "**Packing reverses stage 2's verdict on the three large benchmarks "
        "and leaves the three small ones exactly where they were.** G3 found "
        "no benchmark inside the band. With `P*`-way packing and a sorted "
        "schedule, ogbn-arxiv's hidden layer reaches **1.94x** the VPU, "
        "ogbn-products **4.35x**, and Reddit **7.35x** -- and Reddit and "
        "ogbn-products stay ahead on their output layers too, at 2.34x and "
        "1.60x. Cora, CiteSeer and PubMed stay at 0.02x-0.10x: their widths "
        "are 3-16, so the VPU's `E x F / 128` is tiny while the LUT still pays "
        "its 10 fixed cycles per node no matter how few features it moves.")
    rpt.note(
        "**Both variables are load-bearing, which is why neither section alone "
        "found this.** ogbn-arxiv wins at degree 13.8 and `F = 256` but loses "
        "at degree 13.8 and `F = 40`; Reddit wins at `F = 41` on degree 492. "
        "Stage 2 concluded the condition was on width alone because it "
        "measured at `P = 1`, where the width threshold is 512 and nothing "
        "reaches it. **Packing is what makes degree matter again.**")
    rpt.note(
        f"**The cost is bandwidth, and it is the same bill section 16(d) "
        f"presented for attention.** `P` live OS-V rows read "
        f"`{live_row_bytes_per_cycle(hw, 1)} B/cycle` each from KV SRAM, so "
        f"`P* = 16` on a 256-wide layer needs "
        f"{live_row_bytes_per_cycle(hw, 16) * hw.freq_mhz * 1e6 / 1e12:.2f} "
        f"TB/s and `P* = 32` needs "
        f"{live_row_bytes_per_cycle(hw, 32) * hw.freq_mhz * 1e6 / 1e12:.2f} "
        "TB/s. That is **4x** `study.md` section 16(d)'s figure for the same "
        "`P`, for a reason specific to this workload: attention packs at "
        "`kv_cache_bits = 4` and aggregation runs at 16. The speedups above "
        "are a compute-side ceiling and the port is the thing that decides "
        "whether any of it is reachable.")

    rpt.summary([
        "**Combine needs nothing new.** Every number in section B came out of "
        "the unmodified simulator: `_simulate_matmul` is shape-driven, and a "
        "GNN's `H @ W` is an FFN with nodes in the token slot. The mapping "
        "holds.",
        "**Aggregate needed nothing new either, once written as a pull.** "
        "`h[v] = sum a_vu x[u]` is `(M=1, K=deg(v), N=F)` -- the shape decode "
        "`attn_v` is issued as. Pre-flight 9 runs a real `attn_v` operation at "
        "`(kv_len, head_dim) = (deg, F)` and gets the identical cycle count, so "
        "the identity is proved rather than asserted, and `study.md` sections "
        "4(b), 5 and 14 transfer intact. A *dense* `A_hat` is still hopeless -- "
        "`1/density`, 695x on Cora, 48,479x on ogbn-products (section D) -- "
        "which is why the sparse shape had to be issued per node.",
        "**The LUT loses to a plain VPU at aggregation on every graph, at the "
        "16-bit coefficient precision this machine is configured with** -- 2.2x "
        "on Reddit, 529x on CiteSeer (G1), making the whole layer 2.3-18.5x "
        "slower than putting aggregation on the VPU (H). The hypothesis that "
        "the crossover is a degree around 50 was **a 4-bit statement**: at "
        "`qbit = 4` it reproduces to within 10% on all four graphs it named, "
        "and at 16 bits it is false on all four. The real condition is on the "
        "*feature width*: a crossover exists at all iff `F > vpu_width x qbit / "
        "MU = 32 x qbit`, and the N-null ends at `F = 4096`, so the LUT beats a "
        "VPU in exactly one band, **`32 x qbit < F <= 4096`** (G3). Degree only "
        "decides where inside that band you land. **All of which is a "
        "`P = 1` statement, and stage 3 overturns half of it -- see the "
        "next bullet.**",
        "**Packing the array `P` ways moves the band's lower edge to "
        "`32 x qbit / P` and reverses the verdict on the three large "
        "benchmarks.** `M = 1` was the whole problem: one destination node "
        "cannot fill 32 array rows, so `array_m / n_tiles` of the array "
        "idles. Packing `P` independent destinations into one OS-V pass "
        "recovers exactly that factor and no more -- measured `P*` equals "
        "`array_m / n_tiles` at every width, and is **1.00x at `F = 4096`** "
        "(I). At `P*` with a degree-sorted schedule the LUT beats the VPU "
        "on **Reddit (7.35x), ogbn-products (4.35x) and ogbn-arxiv "
        "(1.94x)**, while Cora, CiteSeer and PubMed stay at 0.02x-0.10x "
        "because their 3-16-wide layers give the VPU almost nothing to do "
        "(K). **Packing is what makes degree matter again**: the crossover "
        "degree is 43 at every `P`, so a graph still has to be dense "
        "enough to clear it (J).",
        "**Sorting the packs is nearly free; grouping them by equal degree "
        "is a disaster.** A packed pass charges one `ceil(K/MU)` for all "
        "`P` nodes, so it costs its maximum degree -- which sounds like the "
        "hard part and is not. Degree-sorted greedy filling lands within "
        "4% of the unreachable equal-degree bound (15.40x-15.95x against "
        "16.00x). Insisting on exactly-equal-degree packs instead reaches "
        "**0.05x on ogbn-arxiv -- 20x slower than not packing** -- because "
        "a power-law tail has thousands of buckets holding fewer than `P` "
        "nodes and each still costs a whole pass (K).",
        "**Aggregation turns out to be compute-bound, not memory-bound** -- the "
        "opposite of stage 1's prediction, and by the LUT's own fixed overhead "
        "rather than by arithmetic. Section E's 0.9-1.2 FLOP/byte was right "
        "about the FLOPs; the LUT simply does not spend its cycles on FLOPs. "
        "Only ogbn-products flips back to memory-bound once the VPU does the "
        "aggregating (H).",
        "**The degree distribution mattered less than expected, and that is a "
        "measured result.** `graph_configs.py` now carries a synthesised "
        "power-law fit per graph (gamma solved so the mean reproduces the "
        "published `avg_degree`). Distribution-aware totals differ from "
        "mean-degree ones by **0.2-6.5%**, because both cost models are affine "
        "in degree and Jensen has nothing to bite on. The distribution still "
        "changes the picture per node -- 58.8% of Cora's nodes are degree 1 and "
        "burn 10 of 11 cycles on overhead -- just not the total.",
        "**Push aggregation is the one dataflow here that the OS-V round-count "
        "defect touches, so it is reported twice.** Push is issued at "
        "`M = deg(u)`, and the default `os_rounds_model = 'tiled'` overcharges "
        "by up to 16x for `M` in 2..31 -- most nodes in most graphs. G1 shows "
        "`tiled` and stage 11's `packed` side by side, both simulator outputs. "
        "Push loses to pull either way (2.0-9.8x corrected), for a reason "
        "neither model touches: with `K = 1` it pays the full 32-row systolic "
        "fill to broadcast one source row. **Pull is on the branch both models "
        "agree on**, so every headline number above is round-model independent.",
        "**The feature row is the new KV entry.** At Kipf & Welling's "
        "`hidden_dim = 16`, an FP16 feature row is 32 B against DDR5's 64 B "
        "burst: 2.00x waste on the most cited GNN configuration there is. "
        "HBM3's 32 B burst erases it exactly, reproducing `study.md` section "
        "15's finding on a different workload.",
        "**Gather-vs-stream is decided by capacity, not by sparsity.** `E > N` "
        "on every graph, so pulling `E` neighbour rows never moves fewer bytes "
        "than streaming `X` once and pushing -- when the accumulators fit. The "
        "crossover is exactly `stream_passes < avg_degree x burst_waste`, and at "
        "8 MB it splits the two mid-size graphs from the dense one: "
        "ogbn-arxiv (degree 14, 21 passes) and ogbn-products (degree 50, 299 "
        "passes) must gather, while Reddit's degree of 492 keeps streaming "
        "ahead through 29 passes.",
        f"**Aggregation is memory-bound by about two orders of magnitude** "
        f"(0.9-1.2 FLOP/byte against this array's {balance:.0f} FLOP/byte "
        "balance point) and has no reusable operand, so the LUT mechanism has "
        "nothing to amortise over -- true of the *bytes*, and section H shows "
        "the cycles reach the same verdict first. **Omni-LUT is an excellent "
        "Combine engine, and the wrong shape for Aggregate at `P = 1`** -- "
        "confirmed on both axes, and the qualifier is stage 3's: the shape "
        "problem is `M = 1`, it is fixable by packing, and what it costs "
        "is 8.2 TB/s of KV-SRAM port at `P* = 16` (K). Whether that port "
        "is buildable is the open question the compute-side result now "
        "rests on.",
    ])
    return rpt, rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default=os.path.join(_here, 'gnn.csv'))
    p.add_argument('--report', default=os.path.join(_here, 'gnn_report.md'))
    args = p.parse_args()

    preflight()
    rpt, rows = sweep(args.report)
    rpt.save()

    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(args.csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows -> {args.csv}")
    print(f"Wrote report      -> {args.report}")


if __name__ == '__main__':
    main()
