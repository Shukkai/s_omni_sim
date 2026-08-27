"""DRAM latency, and the queue depth needed to hide it.

`dram_bandwidth_gbps` is a peak.  Reaching it needs enough reads in flight to
cover the round trip -- Little's law:

    reachable_bw = min(peak, outstanding x burst_bytes / latency)

Every roofline in this repo before `dram_latency_ns` / `dram_max_outstanding`
existed assumed the datasheet number, which is the assumption that a request
queue of unbounded depth is always full.  This study removes it.

**It changes the headline.**  DDR5-6400 at a 64 B burst and 90 ns needs **72
reads in flight** to sustain 51.2 GB/s.  At a 32-deep queue the machine reaches
22.8 GB/s -- 44% of the datasheet -- and the memory-bound region of the decode
grid grows from **11 of 30 cells to 21 of 30**.  At 16 deep it is **30 of 30**:
decode is memory-bound everywhere, which is what the literature says and what
this repo's ideal-DRAM model denied.

So the disagreement with the Omni-LUT paper had **two** independent causes, and
this is the second.  The first is GQA: the paper evaluates only MHA models, and
MHA quadruples KV traffic without touching attention compute.  Either one alone
moves decode toward memory-bound; together they account for the whole gap.

**Scope, deliberately narrow.**  This is a steady-state *throughput* clamp, not
a latency model.  It answers "what bandwidth can a queue of this depth sustain
on a streaming read".  It does not model a dependent access chain, a row miss,
refresh, or a gather whose requests are not independent.  For a scattered read
the effective depth is lower than configured, so **this term is optimistic** --
the honest direction for a term whose absence was the previous error.

Run:  python analysis/memory/latency_run.py
"""

import argparse
import csv
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, '..', '..'))
for _p in ('simulator', 'analysis'):
    sys.path.insert(0, os.path.join(_root, *_p.split('/')))

from simulator import (                                              # noqa: E402
    HardwareConfig, Simulator, WorkloadConfig,
)
from model_configs import get_model_config                           # noqa: E402
from report import Report                                            # noqa: E402

MODEL = 'LLaMA-3-8B'
OUTPUT_TOKENS = 4
BATCHES = [1, 2, 4, 8, 16, 32]
CONTEXTS = [2048, 4096, 8192, 16384, 32768]

PEAK_GBPS = 51.2
BURST = 64
LATENCIES_NS = [60.0, 90.0, 120.0]
DEPTHS = [8, 16, 32, 64, 72, 128]

#: (latency_ns, depth).  `ideal` is what every published number was produced
#: with; pre-flight 1 asserts it still reproduces them.
PROFILES = [
    ('ideal (no latency term)', 0.0, 0),
    ('90 ns, 128 deep', 90.0, 128),
    ('90 ns, 64 deep', 90.0, 64),
    ('90 ns, 32 deep', 90.0, 32),
    ('90 ns, 16 deep', 90.0, 16),
]


def base_hw(lat_ns=0.0, depth=0, **kw):
    return HardwareConfig(
        array_m=32, array_n=4, FPE_array_size=64,
        act_bits=16, accumulate_bits=32, weight_bits=4, kv_cache_bits=4,
        AW_mode="OMNI", AA_mode="OMNI", os_rounds_model='packed',
        dram_burst_bytes=BURST, dram_latency_ns=lat_ns,
        dram_max_outstanding=depth, **kw)


def legs(sim, batch, context):
    """`(compute_s, dram_s)` for one decode step, at the sim's effective BW."""
    m = get_model_config(MODEL)
    w = WorkloadConfig(batch_size=batch, input_tokens=context,
                       output_tokens=OUTPUT_TOKENS, flash_block_size=0)
    r = sim.simulate(m, w)
    bw = sim.effective_dram_bw_gbps() * 1e9
    freq = sim.hw.freq_mhz * 1e6
    steps = max(1, OUTPUT_TOKENS - 1)
    c = d = 0.0
    for group in (r.decode.aw_ops, r.decode.aa_ops):
        for _op, lst in group.items():
            for mt in lst:
                c += mt.cycles / freq
                d += (mt.dram_read_eff + mt.dram_write_eff) / bw
    return c / steps, d / steps


# ============================================================================
# Pre-flight
# ============================================================================

def preflight():
    print("Pre-flight")

    # 1. Both fields at 0 reproduce the published split exactly.
    s = Simulator(base_hw(0.0, 0))
    assert s.effective_dram_bw_gbps() == PEAK_GBPS
    c, d = legs(s, 1, 2048)
    assert abs(c * 1e3 - 7.67) < 0.05 and abs(d * 1e3 - 51.28) < 0.05, (c, d)
    print("  1. both fields at 0 reproduce 7.67 / 51.28 ms at b1/2K ok")

    # 2. **Setting only one field is a no-op.**  Neither alone determines a
    #    bandwidth, and a half-configured clamp silently changing results is
    #    the failure mode this asserts against.
    for hw in (base_hw(90.0, 0), base_hw(0.0, 32)):
        assert Simulator(hw).effective_dram_bw_gbps() == PEAK_GBPS
    print("  2. latency alone, or depth alone, is exactly inert ok")

    # 3. Little's law round-trips: a depth at or above the requirement gives
    #    back the peak exactly, one below does not.
    s = Simulator(base_hw(90.0, 1))
    need = s.required_outstanding()
    assert abs(need - PEAK_GBPS * 1e9 * 90e-9 / BURST) < 1e-9, need
    assert Simulator(base_hw(90.0, int(need) + 1)).effective_dram_bw_gbps() \
        == PEAK_GBPS
    assert Simulator(base_hw(90.0, int(need) - 1)).effective_dram_bw_gbps() \
        < PEAK_GBPS
    print(f"  3. required depth {need:.0f} is exact: +1 gives peak, "
          f"-1 does not ok")

    # 4. The clamp only ever lowers, and is monotone in depth.
    prev = 0.0
    for depth in DEPTHS:
        bw = Simulator(base_hw(90.0, depth)).effective_dram_bw_gbps()
        assert bw >= prev and bw <= PEAK_GBPS, (depth, bw, prev)
        prev = bw
    print("  4. monotone in depth, never above peak ok")

    # 5. Latency scales the requirement linearly -- the law, not a fit.
    a = Simulator(base_hw(60.0, 1)).required_outstanding()
    b = Simulator(base_hw(120.0, 1)).required_outstanding()
    assert abs(b - 2 * a) < 1e-9, (a, b)
    print("  5. doubling latency doubles the depth requirement ok")

    # 6. Compute is untouched -- this is a memory term only.
    c0, _ = legs(Simulator(base_hw(0.0, 0)), 8, 8192)
    c1, _ = legs(Simulator(base_hw(90.0, 16)), 8, 8192)
    assert abs(c0 - c1) < 1e-12, (c0, c1)
    print("  6. the clamp does not disturb the compute leg ok")
    print()


# ============================================================================
# Sections
# ============================================================================

def sweep(report_path):
    rows = []
    rpt = Report(
        report_path,
        "DRAM latency",
        "What the datasheet bandwidth costs to actually reach",
        source='analysis/memory/latency_run.py',
        setup=[
            f"{MODEL}, Omni-LUT-KV4 (32x4, W4A16KV4, 500 MHz).",
            f"DDR5-6400: {PEAK_GBPS} GB/s peak, {BURST} B burst.",
            "Decode only, serial roofline.",
        ],
    )

    # ---- A. the requirement ------------------------------------------------
    rpt.section(
        "A. How many reads must be in flight to reach the datasheet number",
        "Little's law: sustaining `B` bytes/s with a round trip of `L` needs "
        "`B x L / burst` requests outstanding. This is the requirement, not a "
        "measurement -- but it is the requirement every bandwidth figure in "
        "this repo silently assumed was met.")
    a_rows = []
    for lat in LATENCIES_NS:
        need = Simulator(base_hw(lat, 1)).required_outstanding()
        a_rows.append([f"{lat:.0f}", f"{need:.0f}",
                       f"{need * BURST / 1024:.1f} KB"])
        rows.append(dict(section='A', latency_ns=lat, required_depth=need))
    rpt.table(['latency ns', 'reads in flight', 'bytes in flight'],
              a_rows, aligns='rrr')
    rpt.note(
        f"**{PEAK_GBPS} GB/s at 90 ns needs 72 reads in flight** -- 4.5 KB of "
        "data in the air at all times. That is a real design constraint on the "
        "request queue, and it is the constraint an unbounded-bandwidth "
        "roofline assumes away.")

    # ---- B. what a real queue reaches --------------------------------------
    rpt.section(
        "B. What a finite queue actually reaches",
        "Sustainable bandwidth by queue depth at 90 ns, against the "
        f"{PEAK_GBPS} GB/s on the datasheet.")
    b_rows = []
    for depth in DEPTHS:
        bw = Simulator(base_hw(90.0, depth)).effective_dram_bw_gbps()
        b_rows.append([str(depth), f"{bw:.1f}", f"{bw / PEAK_GBPS * 100:.0f}%"])
        rows.append(dict(section='B', depth=depth, eff_bw_gbps=bw,
                         frac_of_peak=bw / PEAK_GBPS))
    rpt.table(['queue depth', 'reachable GB/s', 'of peak'],
              b_rows, aligns='rrr')
    rpt.note(
        "**A 32-deep queue reaches 44% of the datasheet number.** Bandwidth is "
        "not a property of the DRAM alone; it is a property of the DRAM and "
        "the requester together. Halving the queue halves the bandwidth, "
        "exactly, until the requirement is met -- and past it, more depth buys "
        "nothing.")

    # ---- C. the grid moves -------------------------------------------------
    rpt.section(
        "C. The regime grid, under each profile",
        "Decode compute / DRAM. Below 1.00 the array waits on memory. The "
        "first block is every published number in this repo.")
    for label, lat, depth in PROFILES:
        sim = Simulator(base_hw(lat, depth))
        bw = sim.effective_dram_bw_gbps()
        c_rows = []
        nmem = 0
        firsts = {}
        for b in BATCHES:
            cells = []
            for ctx in CONTEXTS:
                c, d = legs(sim, b, ctx)
                v = c / d
                if v < 1.0:
                    nmem += 1
                elif ctx not in firsts:
                    firsts[ctx] = b
                cells.append(f"{v:.2f}")
                rows.append(dict(section='C', profile=label, latency_ns=lat,
                                 depth=depth, batch=b, context=ctx,
                                 cd_ratio=v, eff_bw_gbps=bw))
            c_rows.append([str(b)] + cells)
        rpt.table(['batch'] + [f"{c//1024}K" for c in CONTEXTS],
                  c_rows, aligns='r' * (len(CONTEXTS) + 1))
        rpt.note(
            f"**{label}** -- effective bandwidth **{bw:.1f} GB/s**, "
            f"**{nmem} of 30 cells memory-bound**. First compute-bound batch: "
            + ', '.join(f"{c//1024}K = {firsts.get(c, 'never')}"
                        for c in CONTEXTS) + ".")

    rpt.note(
        "**The triangle is an artefact of assuming an infinitely deep request "
        "queue.** At 128 outstanding it is the published grid exactly. At 64 "
        "it has grown to 13 cells. At 32 -- a plausible depth for an edge "
        "accelerator -- it is **21 of 30**, and at 16 it is **the whole grid**: "
        "decode is memory-bound everywhere, at every batch and every context.")
    rpt.note(
        "**This is the second of two independent reasons this repo disagreed "
        "with the Omni-LUT paper about bound-ness, and the paper is right "
        "under either.** The first is GQA -- the paper evaluates only MHA "
        "models, which quadruple KV traffic without touching attention "
        "compute. Latency is the second, and it does not need MHA to bite: "
        "even on a GQA model a 16-deep queue makes every cell memory-bound.")

    rpt.summary([
        f"**Bandwidth is a property of the DRAM *and the requester*.** "
        f"{PEAK_GBPS} GB/s at a 64 B burst and 90 ns needs **72 reads in "
        "flight**; a 32-deep queue reaches **22.8 GB/s, 44% of the "
        "datasheet**. Every roofline in this repo predating this term assumed "
        "the requirement was met.",
        "**The memory-bound region grows from 11 of 30 cells to 21 of 30 at a "
        "32-deep queue, and to 30 of 30 at 16.** The compute-bound triangle "
        "survives only while the queue is deep enough to hide the round trip. "
        "It is not a property of the workload; it is a property of the "
        "assumption.",
        "**Together with GQA this closes the gap with the published "
        "literature.** Two independent causes, either sufficient: the paper "
        "measures MHA models, and no roofline here charged for latency. Under "
        "either correction decode is memory-bound where the paper says it is.",
        "**The term is deliberately optimistic and should be read that way.** "
        "It is a steady-state throughput clamp on a streaming read -- no "
        "dependent chains, no row misses, no refresh, and no allowance for a "
        "gather whose requests cannot all be in flight at once. A scattered "
        "KV read does worse than this, not better.",
    ])
    return rpt, rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default=os.path.join(_here, 'latency.csv'))
    p.add_argument('--report', default=os.path.join(_here,
                                                    'latency_report.md'))
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
