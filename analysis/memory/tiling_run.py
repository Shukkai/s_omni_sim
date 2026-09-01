"""
Prefill row tiling: the repo's one outright bug, and the frontier behind it.

`_calculate_peak_sram` held the whole activation matrix -- `A_bytes = M * K *
act_bits/8` with `M` the full prefill sequence -- so prefill's claimed working
set was O(seq x d_model): 59 MB at 2K context and **2.1 GB at 32K**.  No
plausible SRAM fits that, so prefill overflowed at every capacity, its spill
charge was a meaningless constant, and `study.md` §7 could only publish a
decode table.  Real hardware tiles the sequence; the model did not.

**It is a capacity bug and only a capacity bug.**  §16(c) had suspected the same
untiled A of also corrupting the SRAM *traffic* terms and parked prefill's
bandwidth numbers behind this fix.  §19 measured it instead: the activation term
is 255.7 B/cycle against an array that consumes 256, so it was right, and tiling
cannot change it.  This file inherits a smaller problem than it was handed.

`hw.sram_m_tile` (0 = untiled, the default) blocks the row loop.  What that
costs is not uniform across dataflows, and the asymmetry is the interesting
part:

  * `LUT_WS` **pays**.  Weight-stationary holds B across the whole `M` stream,
    so a row block re-loads the weight tile and re-pays the array's fill/drain
    once per block.
  * The output-stationary and FPE modes pay **nothing**.  Their traffic terms
    already re-read B once per `array_m` (or `FPE_array_size`) row tile, which
    only makes sense if one row tile of A is resident -- so for those modes the
    footprint was merely inconsistent with the loop nest the traffic model
    already described, and tiling makes the two agree at zero cost.

**One limitation, stated rather than modelled.**  A row block re-reads B from
*SRAM*, not from DRAM: the model has always charged an AW operation's weight
DRAM read exactly once and re-read B from SRAM per row tile, for every mode.
That convention predates this field (LUT_OS has re-read B per row tile since the
beginning) and this file keeps it rather than changing it for the tiled case
alone.  If the weight matrix does not fit on chip -- 29.4 MB for one LLaMA-3-8B
FFN projection -- a real machine re-reads it from DRAM per block too, and the
TTFT costs below are optimistic by that amount.  `sram_overflow` is what flags
the configurations where that bites.

Usage:
    python tiling_run.py
    python tiling_run.py --csv tiling.csv --report tiling_report.md
"""

import argparse
import csv
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, '..', '..'))
for p in ('simulator', 'analysis', 'analysis/cycle_breakdown',
          'analysis/memory'):
    sys.path.insert(0, os.path.join(_root, *p.split('/')))

from simulator import HardwareConfig, WorkloadConfig, Simulator      # noqa: E402
from model_configs import get_model_config                           # noqa: E402
from report import Report                                            # noqa: E402

MODEL = 'LLaMA-3-8B'
CONTEXT = 32768
OUTPUT_TOKENS = 4
KV_BITS = 4
SCORE_SRAM_KB = 128
TILES = [0, 8192, 2048, 512, 128, 32, 8, 1]
CAPACITIES_KB = [1024, 2048, 4096, 8192, 16384, 32768]
CONTEXTS = [2048, 8192, 32768]
KB = 1024
GB = 1e9


def base_hw(m_tile=0, capacity_kb=0, aw="OMNI"):
    return HardwareConfig(
        array_m=32, array_n=4, FPE_array_size=64,
        act_bits=16, accumulate_bits=32, weight_bits=4, kv_cache_bits=KV_BITS,
        AW_mode=aw, AA_mode="OMNI",
        sram_m_tile=m_tile, sram_capacity_kb=capacity_kb,
        score_sram_kb=SCORE_SRAM_KB,
    )


def measure(hw, context=CONTEXT, batch=1):
    sim = Simulator(hw)
    m = get_model_config(MODEL)
    w = WorkloadConfig(batch_size=batch, input_tokens=context,
                       output_tokens=OUTPUT_TOKENS, flash_block_size=0)
    r = sim.simulate(m, w)
    ttft, tpot = sim.compute_roofline_latency(r, w)
    pf = r.prefill.get_total_metrics()
    return {'prefill_peak': r.prefill.peak_sram_bytes,
            'decode_peak': r.decode.peak_sram_bytes,
            'prefill_cycles': pf.cycles,
            'weight_sram': pf.sram_read_b,
            'prefill_dram': pf.dram_read_eff + pf.dram_write_eff,
            'ttft_s': ttft, 'tpot_s': tpot,
            'overflow': pf.sram_overflow,
            'refetch': pf.sram_refetch_bytes}


def largest_tile_that_fits(capacity_kb, context=CONTEXT):
    """Biggest -- so cheapest -- row block whose prefill working set fits.

    Searched over powers of two rather than solved, because the footprint is a
    sum of terms only one of which scales with the block.
    """
    cap = capacity_kb * KB
    best = None
    mt = 1
    while mt <= context:
        if measure(base_hw(m_tile=mt), context)['prefill_peak'] <= cap:
            best = mt
        else:
            break
        mt *= 2
    return best


# ============================================================================
# Pre-flight
# ============================================================================

def preflight():
    # 1. Untiled is the default, and the default is the original model.
    assert HardwareConfig(array_m=32, array_n=4).sram_m_tile == 0

    untiled = measure(base_hw())

    # 2. A block at least as long as M is the same as no block at all -- the
    #    boundary case that makes the field safe.
    big = measure(base_hw(m_tile=CONTEXT * 2))
    for k in ('prefill_peak', 'prefill_cycles', 'weight_sram', 'ttft_s'):
        assert big[k] == untiled[k], f"m_tile >= M must be inert, {k} moved"

    # 3. The bug itself: untiled prefill claims a working set no SRAM holds.
    assert untiled['prefill_peak'] > 2 * GB, \
        f"expected the 2.1 GB untiled claim, got {untiled['prefill_peak']}"

    # 4. And tiling fixes it -- the whole point.
    small = measure(base_hw(m_tile=128))
    assert small['prefill_peak'] < 16 * KB * KB, \
        "a 128-row block should bring prefill under 16 MB"

    # 5. Footprint falls monotonically with the block; latency rises
    #    monotonically. A frontier, not a free lunch.
    prev_peak, prev_ttft = None, None
    for mt in (8192, 2048, 512, 128, 32, 8):
        d = measure(base_hw(m_tile=mt))
        if prev_peak is not None:
            assert d['prefill_peak'] < prev_peak, "footprint must fall"
            assert d['ttft_s'] > prev_ttft, "TTFT must rise"
        prev_peak, prev_ttft = d['prefill_peak'], d['ttft_s']

    # 6. LUT_WS weight traffic scales exactly with the block count -- the
    #    mechanism, asserted rather than inferred from the trend.
    for mt in (2048, 512, 128):
        d = measure(base_hw(m_tile=mt))
        assert d['weight_sram'] == untiled['weight_sram'] * (CONTEXT // mt), \
            f"weight SRAM traffic should scale by M/m_tile at {mt}"

    # 7. Output-stationary pays nothing but the footprint: its traffic model
    #    already re-read B per `array_m` row tile, so tiling only makes the
    #    footprint agree with the loop nest it already described.
    os_untiled = measure(base_hw(aw="LUT_OS"))
    os_tiled = measure(base_hw(m_tile=128, aw="LUT_OS"))
    assert os_tiled['prefill_cycles'] == os_untiled['prefill_cycles'], \
        "tiling must not change output-stationary cycles"
    assert os_tiled['weight_sram'] == os_untiled['weight_sram'], \
        "tiling must not change output-stationary weight traffic"
    assert os_tiled['prefill_peak'] < os_untiled['prefill_peak'], \
        "tiling must still shrink the output-stationary footprint"

    # 8. Decode is untouched: its GEMMs have M = 1 or M = batch, both far
    #    below any block worth setting.
    assert untiled['decode_peak'] == small['decode_peak'], \
        "decode footprint must not move with a prefill row block"
    assert untiled['tpot_s'] == small['tpot_s'], "decode TPOT must not move"

    print("pre-flight: 8 checks passed")


# ============================================================================
# Sweep
# ============================================================================

def sweep(report_path):
    rows = []
    preflight()

    untiled = measure(base_hw())

    rep = Report(
        report_path,
        "Prefill row tiling",
        subtitle="The capacity bug, and the frontier it was hiding",
        source="analysis/memory/tiling_run.py",
        setup=[f"Model: LLaMA-3-8B on Omni-LUT, 4-bit KV, context "
               f"{CONTEXT:,}, batch 1, standard attention, scores staged.",
               "Prefill resolves to `LUT_WS`; the row block is "
               "`hw.sram_m_tile`, in rows of the activation matrix."])

    rep.summary([
        f"**The untiled model claimed a "
        f"{untiled['prefill_peak'] / GB:.1f} GB prefill working set**, so "
        f"prefill overflowed at every capacity and §7 could only publish a "
        f"decode table. A 512-row block brings it to "
        f"{measure(base_hw(m_tile=512))['prefill_peak'] / (KB * KB):,.0f} MB.",
        "**Tiling is a frontier, not a fix with no cost.** Weight-stationary "
        "re-loads the weight tile and re-pays the array's fill/drain once per "
        "block, so the footprint falls and TTFT rises — monotonically in both "
        "directions, asserted in pre-flight.",
        "**The knee is wide.** 512 rows buys a **64x** smaller footprint for "
        "**1.075x** TTFT. Past 128 rows the curve turns hard: 32 rows costs "
        "2.22x and 8 rows costs 5.87x.",
        "**Output-stationary pays nothing but the footprint.** Its traffic "
        "model already re-read B once per `array_m` row tile, which only makes "
        "sense if one row tile of A is resident — so for those modes the "
        "footprint was simply inconsistent with the loop nest, and tiling "
        "costs zero cycles and zero bytes.",
        "**Decode does not move**, and cannot: its GEMMs have `M = 1` or "
        "`M = batch`, both far below any block worth setting.",
    ])

    # ---- A ------------------------------------------------------------------
    rep.section(
        "A. The frontier",
        "Prefill working set and what it costs, at 32K context.")
    trows = []
    for mt in TILES:
        d = measure(base_hw(m_tile=mt))
        rows.append({'section': 'A', 'm_tile': mt, 'context': CONTEXT, **d})
        peak = d['prefill_peak']
        peak_s = (f"{peak / GB:,.2f} GB" if peak >= GB
                  else f"{peak / (KB * KB):,.1f} MB" if peak >= KB * KB
                  else f"{peak / KB:,.0f} KB")
        trows.append([
            "untiled" if mt == 0 else f"{mt:,}", peak_s,
            f"{untiled['prefill_peak'] / peak:,.0f}x",
            f"{d['weight_sram'] / GB:,.1f} GB",
            f"{d['ttft_s']:,.1f} s", f"{d['ttft_s'] / untiled['ttft_s']:.3f}x",
        ])
    rep.table(["row block", "prefill peak", "smaller by", "weight SRAM",
               "TTFT", "vs untiled"], trows, aligns="lrrrrr")
    rep.note(
        "**512 rows is the operating point.** 64x less on-chip memory for 7.5% "
        "of TTFT, and the activation block is then 4 MB against a 33 MB "
        "working set — the B and C tiles, not A, are what is left. Below 128 "
        "rows the fill/drain the array re-pays per block starts to dominate "
        "the activation stream itself, which is what turns the curve.")

    # ---- B ------------------------------------------------------------------
    rep.section(
        "B. §7's prefill row, finally",
        "Largest row block whose prefill working set fits a given SRAM — "
        "largest because it is also the cheapest — and what it costs. This is "
        "the table §7 could not publish.")
    trows = []
    for cap in CAPACITIES_KB:
        mt = largest_tile_that_fits(cap)
        if mt is None:
            trows.append([f"{cap / KB:,.0f} MB", "—", "—", "—",
                          "does not fit at any block"])
            rows.append({'section': 'B', 'capacity_kb': cap, 'm_tile': None})
            continue
        d = measure(base_hw(m_tile=mt))
        rows.append({'section': 'B', 'capacity_kb': cap, 'm_tile': mt, **d})
        trows.append([
            f"{cap / KB:,.0f} MB", f"{mt:,} rows",
            f"{d['prefill_peak'] / (KB * KB):,.1f} MB",
            f"{d['ttft_s']:,.1f} s",
            f"{d['ttft_s'] / untiled['ttft_s']:.3f}x",
        ])
    rep.table(["SRAM", "largest block that fits", "prefill peak", "TTFT",
               "vs untiled"], trows, aligns="lrrrr")
    rep.note(
        "**Capacity and TTFT trade smoothly**, which is what the untiled model "
        "could never show: every row of this table was 'overflow, charge a "
        "constant' before. Note the decode floor from §7 — 924.5 KB, set by "
        "the FFN and projection tiles — sits underneath all of it, so a chip "
        "sized for decode alone needs a small block to run prefill at all.")

    # ---- C ------------------------------------------------------------------
    rep.section(
        "C. How the bug scaled with context",
        "The untiled footprint is O(seq x d_model), so it got worse exactly "
        "where prefill matters. A fixed 512-row block does not.")
    trows = []
    for ctx in CONTEXTS:
        u = measure(base_hw(), context=ctx)
        t = measure(base_hw(m_tile=512), context=ctx)
        rows.append({'section': 'C', 'context': ctx,
                     'untiled_peak': u['prefill_peak'],
                     'tiled_peak': t['prefill_peak'],
                     'ttft_untiled': u['ttft_s'], 'ttft_tiled': t['ttft_s']})
        trows.append([
            f"{ctx:,}",
            f"{u['prefill_peak'] / (KB * KB):,.0f} MB",
            f"{t['prefill_peak'] / (KB * KB):,.1f} MB",
            f"{u['prefill_peak'] / t['prefill_peak']:,.0f}x",
            f"{t['ttft_s'] / u['ttft_s']:.3f}x",
        ])
    rep.table(["context", "untiled peak", "512-row peak", "smaller by",
               "TTFT cost"], trows, aligns="lrrrr")
    rep.note(
        "**The untiled footprint is linear in context and the tiled one is "
        "nearly flat**, so the two diverge without limit — the bug was worst "
        "exactly where the long-context story this repo is about lives. The "
        "tiled column is not *quite* flat, and the reason is worth naming: it "
        "holds at 14.3 MB through 8K and then rises to 32.3 MB at 32K, "
        "because past ~16K the binding term stops being the activation block "
        "and becomes **attention's KV tile**, which grows with `kv_len` and "
        "which no row block touches. That is §7's decode result arriving in "
        "prefill. The TTFT cost of the fix stays near-constant across the "
        "sweep, because the extra fill/drain scales with the block count the "
        "same way the activation stream scales with the sequence.")

    rep.save()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=os.path.join(_here, 'tiling.csv'))
    ap.add_argument('--report', default=os.path.join(_here, 'tiling_report.md'))
    args = ap.parse_args()

    rows = sweep(args.report)
    keys = sorted({k for r in rows for k in r})
    with open(args.csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.csv} and {args.report}")


if __name__ == '__main__':
    main()
