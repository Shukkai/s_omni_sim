"""
GNN on Omni-LUT, stage 1: Combine simulated, Aggregate priced.

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

ACT_BITS = 16
WEIGHT_BITS = 4
ACCUM_BITS = 32
INDEX_BYTES = 4        # int32 CSR column indices
EDGE_WEIGHT_BITS = 16  # GCN's 1/sqrt(d_u d_v), one per non-zero
TECHS = ['DDR5-6400', 'HBM3']
SMALL = ['Cora', 'CiteSeer', 'PubMed']          # dense A_hat is simulable
CAPACITY_KB = [0, 1024, 8192]                   # 0 = unlimited


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
    print()


# ============================================================================
# Sections
# ============================================================================

def sweep(report_path):
    rows = []
    rpt = Report(
        report_path,
        "GNN on Omni-LUT, stage 1",
        "Combine measured through the existing simulator; Aggregate priced by "
        "a closed-form gather model",
        source='analysis/gnn/gnn_run.py',
        setup=[
            f"Array 32x4, MU=4, RAC=32, 500 MHz. act={ACT_BITS} b, "
            f"weights={WEIGHT_BITS} b, accum={ACCUM_BITS} b, AW/AA = OMNI.",
            f"CSR indices {INDEX_BYTES} B, GCN edge coefficients "
            f"{EDGE_WEIGHT_BITS} b.",
            "Graph statistics from `simulator/graph_configs.py`.",
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
        "Stage 2 is what turns it into a simulated number.")

    rpt.summary([
        "**Combine needs nothing new.** Every number in section B came out of "
        "the unmodified simulator: `_simulate_matmul` is shape-driven, and a "
        "GNN's `H @ W` is an FFN with nodes in the token slot. The mapping "
        "holds.",
        "**Aggregate cannot use the simulator at all today.** A dense `A_hat` "
        "overcharges by `1/density` -- 695x on Cora, 48,479x on ogbn-products "
        "(section D). The closed-form `gather_cost()` in this file is the "
        "specification stage 2 implements.",
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
        "nothing to amortise over. The expected stage-2 result is that "
        "Omni-LUT is an excellent Combine engine and structurally the wrong "
        "shape for Aggregate.",
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
