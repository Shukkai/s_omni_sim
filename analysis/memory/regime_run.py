"""
The regime map: what decode is bound by, and which lever can still move it.

Every result in `study.md` is a technique measured at one or two operating
points. This asks the prior question -- **what is decode bound by, as a function
of (batch, context)** -- and then, for each regime, **what is the largest
speedup any lever could possibly deliver there.** A technique aimed at a
resource that is not the bottleneck cannot help no matter how well it works, and
most of this repo's negative results turn out to be exactly that.

**The one measurement that reframes it.** `study.md` section 3 says "decode is
DRAM-bound" and every later section inherits that. It is a **batch-1
statement**: compute/DRAM is 0.15 at batch 1 / 2K and above 1.0 from batch 2
upward. At batch the array is reported compute-bound at every context, which is
the opposite regime, and the techniques that work there are the opposite ones.

**BLOCKED, and this file is what found the blocker.** The batch >= 2 half of
the map is currently sitting on a cycle-model defect, so its numbers must not be
quoted as a hardware result. `_calculate_cycles` charges `LUT_OS_V` rounds as::

    M == 1 :  rounds = ceil(n_tiles / array_m)          # packs n_tiles 32-wide
    else   :  rounds = ceil(ceil(M / array_m) * n_tiles) # no packing at all

The `else` branch rounds `M` up to a whole 32-row tile *before* multiplying by
`n_tiles`, so it never packs accumulator tiles across rows unless `M >= 32`. The
accumulator budget actually allows `ceil(M * n_tiles / array_m)`, which
reproduces the `M == 1` special case exactly and is the general form of it. The
gap is `array_m / M`:

    M          1     2     4     8    16    32
    model      1    32    32    32    32    32
    allowed    1     2     4     8    16    32
    overcharge 1x  16x    8x    4x    2x    1x   (N = 4096, n_tiles = 32)

Measured consequence: decode `q_proj` cycles go **32.96x** from batch 1 to batch
2 for a 2x workload, then **1.00x flat** from batch 2 all the way to batch 32 --
the model charges the same cycles for 2 sequences as for 32. That discontinuity,
not the hardware, is what puts C/D above 1.0 at batch 2.

What it does and does not corrupt: comparisons *at fixed batch* cancel it, so
sections 4, 12 and 14's technique ratios stand. Cross-batch absolute times do
not, which includes section 17's C/D split and therefore the regime boundary
below. Fixing it is a cycle-path change that moves published numbers, so it is a
gated stage, not an edit.

**The four ceilings.** For each point this computes the speedup available if a
whole resource became free, by re-running the roofline over the same per-op
metrics with one term suppressed:

  * `overlap`   -- perfect double-buffering: `max(sum C, sum D)` instead of
                   `sum max(C, D)`. Ceiling on scheduling.
  * `kv_free`   -- all attention DRAM removed. **Ceiling on every KV technique
                   in this repo at once** -- eviction, selection, residency,
                   channel pruning. If this is 1.02x, no KV paper can help here.
  * `w_free`    -- all weight DRAM removed. Ceiling on weight quantisation
                   below W4, and on what batching already buys by amortising.
  * `packing`   -- `attn_v` cycles / 32 (section 14's exact measured recovery).
                   Ceiling on array packing.

These are **not compositional** and are not meant to be: each is an upper bound
on one family, computed so the families can be ranked at a point. The largest
one names what is worth researching there; a small one is a proof that a family
is dead in that regime regardless of how good the algorithm gets.

Nothing here changes the simulator. The interventions are arithmetic over the
per-op metrics a normal `simulate()` already returns.

Usage:
    python regime_run.py
    python regime_run.py --csv regime.csv --report regime_report.md
"""

import argparse
import csv
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, '..', '..'))
for p in ('simulator', 'analysis', 'analysis/memory'):
    sys.path.insert(0, os.path.join(_root, *p.split('/')))

from simulator import HardwareConfig, WorkloadConfig, Simulator      # noqa: E402
from model_configs import get_model_config                           # noqa: E402
from report import Report                                            # noqa: E402

MODEL = 'LLaMA-3-8B'
OUTPUT_TOKENS = 4
CONTEXTS = [2048, 4096, 8192, 16384, 32768]
BATCHES = [1, 2, 4, 8, 16, 32]
PACK_RECOVERY = 32          # section 14: attn_v recovers exactly 32x on cycles


def base_hw(**kw):
    return HardwareConfig(
        array_m=32, array_n=4, FPE_array_size=64,
        act_bits=16, accumulate_bits=32, weight_bits=4, kv_cache_bits=4,
        AW_mode="OMNI", AA_mode="OMNI", **kw)


def op_times(r, freq, bw):
    """Every decode GEMM as (name, compute_s, weight_dram_s, kv_dram_s).

    **The split is by read/write, not by operation.** Grouping AW ops as
    "weights" is wrong: `k_proj` and `v_proj` *write the KV cache*, so their
    `dram_write` is KV traffic that scales with batch, and pre-flight 5 catches
    it (weight DRAM otherwise drifts 0.04% between batch 1 and 32). So:

      * weight bytes = AW `dram_read` -- the weight matrices, and nothing else.
      * KV bytes     = AW `dram_write` (the KV writeback, `k_proj`/`v_proj`
                       only) + all AA traffic (cache reads and score spill).

    That matters for the ceilings: eviction decided during prefill removes the
    writeback too (`study.md` section 4(c)), so it belongs on the KV side of a
    bound on KV techniques.
    """
    out = []
    for op, lst in r.decode.aw_ops.items():
        for m in lst:
            out.append((op.value, m.cycles / freq,
                        m.dram_read_eff / bw, m.dram_write_eff / bw))
    for op, lst in r.decode.aa_ops.items():
        for m in lst:
            out.append((op.value, m.cycles / freq, 0.0,
                        (m.dram_read_eff + m.dram_write_eff) / bw))
    return out


def serial(ops):
    """sum max(compute, dram) -- the model every published number uses."""
    t = 0.0
    for _, c, wd, kd in ops:
        d = wd + kd
        t += c if c > d else d
    return t


def measure(batch, context):
    """One (batch, context) point: the split, and the four ceilings."""
    m = get_model_config(MODEL)
    w = WorkloadConfig(batch_size=batch, input_tokens=context,
                       output_tokens=OUTPUT_TOKENS, flash_block_size=0)
    sim = Simulator(base_hw())
    r = sim.simulate(m, w)
    freq = sim.hw.freq_mhz * 1e6
    bw = sim.hw.dram_bandwidth_gbps * 1e9
    steps = max(1, w.output_tokens - 1)
    ops = op_times(r, freq, bw)

    compute = sum(c for _, c, _, _ in ops)
    w_dram = sum(wd for _, _, wd, _ in ops)
    kv_dram = sum(kd for _, _, _, kd in ops)
    dram = w_dram + kv_dram
    attn_v_c = sum(c for n, c, _, _ in ops if n == 'attn_v_matmul')

    base = serial(ops)
    # Ceilings: same roofline, one resource suppressed.
    pipelined = max(compute, dram)
    kv_free = serial([(n, c, wd, 0.0) for n, c, wd, kd in ops])
    w_free = serial([(n, c, 0.0, kd) for n, c, wd, kd in ops])
    packed = serial([(n, c / PACK_RECOVERY if n == 'attn_v_matmul' else c,
                      wd, kd) for n, c, wd, kd in ops])

    return {
        'batch': batch, 'context': context,
        'compute_ms': compute / steps * 1e3,
        'dram_ms': dram / steps * 1e3,
        'cd_ratio': compute / dram,
        'regime': 'compute' if compute > dram else 'memory',
        'tpot_ms': base / steps * 1e3,
        'kv_dram_share': kv_dram / dram,
        'weight_dram_ms': w_dram / steps * 1e3,
        'attn_v_compute_share': attn_v_c / compute,
        'ceil_overlap': base / pipelined,
        'ceil_kv_free': base / kv_free,
        'ceil_w_free': base / w_free,
        'ceil_packing': base / packed,
    }


def best_lever(row):
    """Which family has the largest ceiling here, and by how much."""
    cands = {'packing': row['ceil_packing'], 'overlap': row['ceil_overlap'],
             'KV bytes': row['ceil_kv_free'], 'weight bytes': row['ceil_w_free']}
    name = max(cands, key=cands.get)
    return name, cands[name]


# ============================================================================
# Pre-flight
# ============================================================================

def preflight():
    """Six checks. Each is a way the ceilings could be silently wrong."""
    print("Pre-flight")
    m = get_model_config(MODEL)

    # 1. The hand-rolled serial roofline reproduces the simulator's own TPOT.
    #    If this drifts, every ceiling below is taken against the wrong base.
    for batch, ctx in ((1, 8192), (32, 32768)):
        w = WorkloadConfig(batch_size=batch, input_tokens=ctx,
                           output_tokens=OUTPUT_TOKENS, flash_block_size=0)
        sim = Simulator(base_hw())
        r = sim.simulate(m, w)
        freq, bw = sim.hw.freq_mhz * 1e6, sim.hw.dram_bandwidth_gbps * 1e9
        _, tpot = sim.compute_roofline_latency(r, w)
        mine = serial(op_times(r, freq, bw)) / max(1, w.output_tokens - 1)
        rel = abs(mine - tpot) / tpot
        assert rel < 1e-9, (batch, ctx, mine, tpot, rel)
    print("  1. hand-rolled serial roofline == compute_roofline_latency ok")

    # 2. The overlap ceiling reproduces hw.overlap_model='pipelined' exactly --
    #    proves `max(sum C, sum D)` is the same object section 17 measured.
    for batch, ctx in ((1, 32768), (8, 8192)):
        w = WorkloadConfig(batch_size=batch, input_tokens=ctx,
                           output_tokens=OUTPUT_TOKENS, flash_block_size=0)
        sp = Simulator(base_hw(overlap_model='pipelined'))
        rp = sp.simulate(m, w)
        _, tpot_p = sp.compute_roofline_latency(rp, w)
        row = measure(batch, ctx)
        got = row['tpot_ms'] / row['ceil_overlap']
        assert abs(got - tpot_p * 1e3) / (tpot_p * 1e3) < 1e-9, (got, tpot_p)
    print("  2. ceil_overlap == overlap_model='pipelined' ok")

    # 3. Every ceiling is >= 1: suppressing a resource can never slow a
    #    max()-based roofline down.
    for batch, ctx in ((1, 2048), (8, 8192), (32, 32768)):
        row = measure(batch, ctx)
        for k in ('ceil_overlap', 'ceil_kv_free', 'ceil_w_free',
                  'ceil_packing'):
            assert row[k] >= 1.0 - 1e-12, (batch, ctx, k, row[k])
    print("  3. all four ceilings >= 1.0 ok")

    # 4. Suppressing BOTH DRAM terms leaves exactly the compute time -- proves
    #    the kv/weight split is a partition with nothing falling between it.
    row = measure(8, 8192)
    w = WorkloadConfig(batch_size=8, input_tokens=8192,
                       output_tokens=OUTPUT_TOKENS, flash_block_size=0)
    sim = Simulator(base_hw())
    r = sim.simulate(m, w)
    freq, bw = sim.hw.freq_mhz * 1e6, sim.hw.dram_bandwidth_gbps * 1e9
    ops = op_times(r, freq, bw)
    none_dram = serial([(n, c, 0.0, 0.0) for n, c, wd, kd in ops])
    assert abs(none_dram - sum(c for _, c, _, _ in ops)) < 1e-12
    print("  4. kv/weight is a partition: both suppressed == compute only ok")

    # 5. Weight DRAM is constant in batch (study.md section 12's claim, which
    #    is the whole reason the regime moves with batch at all).
    w1 = measure(1, 32768)['weight_dram_ms']
    w32 = measure(32, 32768)['weight_dram_ms']
    assert abs(w1 - w32) / w1 < 1e-9, (w1, w32)
    print(f"  5. weight DRAM constant in batch: {w1:.2f} ms at b1 and b32 ok")

    # 6. The regime really does flip with batch -- the finding this file is
    #    built to establish, asserted rather than eyeballed off a table.
    assert measure(1, 2048)['regime'] == 'memory'
    assert measure(32, 2048)['regime'] == 'compute'
    print("  6. regime flips memory -> compute between batch 1 and 32 ok")
    print()


# ============================================================================
# Sections
# ============================================================================

def sweep(report_path):
    rows = []
    grid = {}
    for b in BATCHES:
        for c in CONTEXTS:
            grid[(b, c)] = measure(b, c)
            rows.append(dict(section='map', **grid[(b, c)]))

    rpt = Report(
        report_path,
        "The regime map",
        "What decode is bound by, and which lever can still move it",
        source='analysis/memory/regime_run.py',
        setup=[
            f"{MODEL}, Omni-LUT-KV4 (32x4, W4A16KV4, 500 MHz, DDR5-6400).",
            "Decode only. Serial roofline, the model every published number "
            "in `study.md` uses.",
        ],
    )

    # ---- A. the map --------------------------------------------------------
    rpt.section(
        "A. Compute / DRAM, over the operating grid",
        "Ratio of decode compute time to decode DRAM time. Below 1.0 the array "
        "is waiting on memory; above 1.0 memory is waiting on the array.")
    hdr = ['batch'] + [f"{c//1024}K" for c in CONTEXTS]
    a_rows = []
    for b in BATCHES:
        a_rows.append([str(b)] + [f"{grid[(b, c)]['cd_ratio']:.2f}"
                                  for c in CONTEXTS])
    rpt.table(hdr, a_rows, aligns='l')
    b_rows = []
    for b in BATCHES:
        b_rows.append([str(b)] + [grid[(b, c)]['regime'][:4] for c in CONTEXTS])
    rpt.table(hdr, b_rows, aligns='l', caption="Regime (mem / comp)")
    rpt.note(
        "**The batch-1 row is a result. The rest of this table is not, yet.** "
        "At batch 1 the array is memory-bound at every context up to 32K, where "
        "it lands almost exactly on the balance point -- and `study.md` section "
        "3's headline \"decode is DRAM-bound\" is exactly that row, a batch-1 "
        "statement that never says so.")
    rpt.note(
        "**Everything from batch 2 rightward is blocked on a cycle-model "
        "defect this sweep found.** `_calculate_cycles` packs `n_tiles` across "
        "the 32 array rows only when `M == 1`; at `M >= 2` it rounds `M` up to "
        "a whole 32-row tile before multiplying by `n_tiles`, so it charges "
        "`array_m / M` times what the accumulator budget allows -- **16x at "
        "batch 2**, 4x at batch 8, exact at batch 32. Decode `q_proj` cycles "
        "jump **32.96x** from batch 1 to batch 2 for a 2x workload, then sit "
        "**1.00x flat** from batch 2 to batch 32. That discontinuity is what "
        "puts C/D above 1.0 at batch 2, not the hardware. See the module "
        "docstring for the derivation.")
    rpt.note(
        "**Two different mechanisms push the same way.** Batch amortises the "
        "constant weight traffic over more work, and context grows attention "
        "compute quadratically while weight traffic stays flat. So the "
        "memory-bound corner is small, and it is precisely the corner a "
        "single-user latency demo sits in.")

    # ---- B. what the bound is made of --------------------------------------
    rpt.section(
        "B. What each bound is made of",
        "`KV DRAM` is the cache reads, the score spill and the `k_proj`/`v_proj` "
        "writeback; `weight DRAM` is the weight matrices alone. `attn_v cycles` is that one operation's share of all decode "
        "compute.")
    c_rows = []
    for b in BATCHES:
        for c in (2048, 32768):
            g = grid[(b, c)]
            c_rows.append([
                str(b), f"{c//1024}K", g['regime'],
                f"{g['weight_dram_ms']:.1f}",
                f"{g['dram_ms'] - g['weight_dram_ms']:.1f}",
                f"{g['kv_dram_share']*100:.1f}%",
                f"{g['attn_v_compute_share']*100:.1f}%",
            ])
            rows.append(dict(section='B', **g))
    rpt.table(
        ['batch', 'ctx', 'regime', 'weight DRAM ms', 'KV DRAM ms',
         'KV share of DRAM', 'attn_v share of compute'],
        c_rows, aligns='llr')
    rpt.note(
        "**In the memory-bound corner the memory is weights, not KV.** At "
        "batch 1 / 2K, attention is a few per cent of DRAM -- so a KV technique "
        "is attacking a few per cent of the bottleneck. That single line "
        "explains every negative result in `study.md` sections 5, 10, 11 and 13 "
        "at once: they were all measured there.")
    rpt.note(
        "**In the compute-bound region the compute is `attn_v`.** Its share of "
        "decode cycles rises with both axes, so the compute-bound regime has "
        "one dominant operation -- and section 14 already showed that operation "
        "runs at 3.12% occupancy.")

    # ---- C. the ceilings ---------------------------------------------------
    rpt.section(
        "C. What any lever could possibly buy, per regime",
        "Speedup if a whole resource became free. `KV bytes` is the ceiling on "
        "**every KV technique in this repo at once** -- eviction, selection, "
        "residency, channel pruning -- so a small number there is a proof that "
        "the family is dead in that regime, whatever the algorithm.")
    d_rows = []
    for b in BATCHES:
        for c in CONTEXTS:
            g = grid[(b, c)]
            name, val = best_lever(g)
            d_rows.append([
                str(b), f"{c//1024}K", g['regime'],
                f"{g['ceil_packing']:.2f}x", f"{g['ceil_overlap']:.2f}x",
                f"{g['ceil_kv_free']:.2f}x", f"{g['ceil_w_free']:.2f}x",
                f"{name} ({val:.2f}x)",
            ])
    rpt.table(
        ['batch', 'ctx', 'regime', 'packing', 'overlap', 'KV bytes',
         'weight bytes', 'largest lever'],
        d_rows, aligns='llrrrrrl')
    rpt.note(
        "**The batch-1 row is the solid result, and it is a strong one.** "
        "Removing *all* KV traffic -- every eviction, selection, residency and "
        "pruning scheme at once, working perfectly -- buys **1.01x at 2K and "
        "1.07x at 32K**. That is an upper bound on the entire KV literature at "
        "batch 1, and it holds regardless of algorithm. The lever there is "
        "weight bytes, which is the resource batching already attacks.")
    rpt.note(
        "**The batch >= 2 rows inherit the cycle defect and are directional "
        "only.** They say packing is the largest ceiling in the compute-bound "
        "region, which is probably right -- `attn_v` at 3.12% occupancy is real "
        "and independent of the defect -- but the magnitudes rest on decode AW "
        "cycles that are overcharged `array_m / M`. Re-run after the rounds fix "
        "before quoting any number in these rows.")
    rpt.note(
        "**These are ceilings, not achievable speedups, and they do not "
        "compose.** Each suppresses one resource entirely while holding the "
        "others; two applied together interact through the same `max()` the "
        "roofline is built on. They rank families at a point -- which is what "
        "'what is worth researching here' needs -- and nothing more.")

    # ---- D. the crossover --------------------------------------------------
    rpt.section(
        "D. Where the boundary sits",
        "Smallest batch at which each context is compute-bound.")
    e_rows = []
    for c in CONTEXTS:
        first = next((b for b in BATCHES if grid[(b, c)]['regime'] == 'compute'),
                     None)
        e_rows.append([
            f"{c//1024}K", str(first) if first else '> 32',
            f"{grid[(1, c)]['cd_ratio']:.2f}",
            f"{grid[(1, c)]['ceil_kv_free']:.2f}x",
            f"{grid[(32, c)]['ceil_packing']:.2f}x",
        ])
        rows.append(dict(section='D', context=c,
                         first_compute_bound_batch=first or 0))
    rpt.table(
        ['ctx', 'first compute-bound batch', 'C/D at b1',
         'KV ceiling at b1', 'packing ceiling at b32'],
        e_rows, aligns='lr')
    rpt.note(
        "**The memory-bound regime is batch 1 and nothing else.** Every context "
        "measured flips by batch 2. A design targeting throughput never enters "
        "it; a design targeting single-stream latency never leaves it. Those "
        "are two different accelerators and this document has been quoting "
        "numbers from both.")

    rpt.summary([
        "**Found a cycle-model defect that blocks the map's own headline.** "
        "`_calculate_cycles` packs `n_tiles` across the array's 32 rows only at "
        "`M == 1`; from `M >= 2` it charges `array_m / M` times what the "
        "accumulator budget allows -- 16x at batch 2, exact at batch 32. Decode "
        "`q_proj` goes 32.96x from batch 1 to 2 for a 2x workload, then 1.00x "
        "flat to batch 32. **Every batch >= 2 number here, and `study.md` "
        "section 17's C/D split, is waiting on the fix.**",
        "**Decode is memory-bound at batch 1** -- C/D 0.15 at 2K rising to 1.00 "
        "at 32K. `study.md` section 3's \"decode is DRAM-bound\" is that row "
        "and only that row, a batch-1 statement the document inherited without "
        "restating. This half is unaffected by the defect: batch 1 *is* the "
        "correct `M == 1` path.",
        "**In that corner the memory is weights, not KV** -- attention is a few "
        "per cent of decode DRAM at batch 1 / 2K. Every KV technique measured "
        "there was attacking a few per cent of the bottleneck, which is the "
        "single explanation for sections 5, 10, 11 and 13's negative results.",
        "**The hardware limitation worth attacking is array occupancy.** "
        "`attn_v` runs at 3.12% of 4096 lanes because `M = 1`, and packing is "
        "the largest ceiling at every compute-bound point measured. That "
        "diagnosis is independent of the defect; the magnitudes are not.",
        "**The KV-bytes ceiling bounds every KV paper at once.** Where it is "
        "small, no eviction, selection, residency or pruning scheme can help, "
        "however good the algorithm -- the resource is not the bottleneck.",
        "**Two accelerators, not one.** Single-stream latency lives at batch 1 "
        "and wants fewer weight bytes; throughput lives at batch >= 2 and wants "
        "array occupancy. Quoting a technique without its regime is quoting "
        "half a result -- and `study.md` currently quotes both regimes without "
        "ever naming the boundary.",
    ])
    return rpt, rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default=os.path.join(_here, 'regime.csv'))
    p.add_argument('--report', default=os.path.join(_here, 'regime_report.md'))
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
        wtr = csv.DictWriter(f, fieldnames=keys)
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {args.csv}")
    print(f"Wrote report      -> {args.report}")


if __name__ == '__main__':
    main()
