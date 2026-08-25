"""
How much does "no overlap" cost?

The roofline sums `max(compute, memory)` **per operation** and never lets one
operation's memory hide behind another's compute.  Real hardware double-buffers:
operation i+1's operands prefetch during operation i's compute, and a long chain
of such operations tends toward `max(sum(compute), sum(memory))` instead.

`hw.overlap_model` makes both available:

    "serial"     sum(max(c, m))   -- today, and every published number
    "pipelined"  max(sum c, sum m) -- the opposite extreme

**Neither is the truth; they bracket it.**  That is the honest framing and it is
why this ships with `"serial"` as the default rather than switching over: a
pipelined machine needs enough SRAM to hold two operand sets, which
`sram_capacity_kb` would have to allow, and the model has no way to check that a
given schedule is actually issuable.

**Why it matters more than it sounds.**  The gap is widest where a phase
alternates between memory-bound and compute-bound operations -- which is exactly
what decode does, since AW projections are memory-bound and `attn_v` is
compute-bound under a 4-bit KV cache.  At 32K context, batch 1, `"serial"`
overstates decode TPOT by **1.75x**, which is larger than most of the KV
techniques in `study.md` were measured to *save*.  Every latency number in that
document is an upper bound by some factor in this table.

Usage:
    python overlap_run.py
    python overlap_run.py --csv overlap.csv --report overlap_report.md
"""

import argparse
import csv
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, '..', '..'))
for p in ('simulator', 'analysis', 'analysis/cycle_breakdown',
          'analysis/compact_breakdown', 'analysis/memory'):
    sys.path.insert(0, os.path.join(_root, *p.split('/')))

from simulator import HardwareConfig, WorkloadConfig, Simulator      # noqa: E402
from model_configs import get_model_config                           # noqa: E402
from kv_budget import KVBudgetSimulator                              # noqa: E402
from report import Report                                            # noqa: E402

MODEL = 'LLaMA-3-8B'
OUTPUT_TOKENS = 4
CONTEXTS = [2048, 8192, 32768]
BATCHES = [1, 8, 32]


ROUNDS_MODEL = 'tiled'      # --rounds-model; 'tiled' is what section 17 published


def hw(overlap='serial', **kw):
    kw.setdefault('os_rounds_model', ROUNDS_MODEL)
    return HardwareConfig(
        array_m=32, array_n=4, FPE_array_size=64,
        act_bits=16, accumulate_bits=32, weight_bits=4, kv_cache_bits=4,
        AW_mode="OMNI", AA_mode="OMNI", overlap_model=overlap, **kw)


def measure(overlap, batch, context, sim_cls=Simulator, **kw):
    m = get_model_config(MODEL)
    w = WorkloadConfig(batch_size=batch, input_tokens=context,
                       output_tokens=OUTPUT_TOKENS, flash_block_size=0)
    sim = sim_cls(hw(overlap), **kw)
    r = sim.simulate(m, w)
    ttft, tpot = sim.compute_roofline_latency(r, w)
    # Resource totals, so the table can say *which* roof each phase lands on.
    freq = sim.hw.freq_mhz * 1e6
    bw = sim.hw.dram_bandwidth_gbps * 1e9
    c = d = 0.0
    for grp in (r.decode.aa_ops, r.decode.aw_ops):
        for lst in grp.values():
            for om in lst:
                c += om.cycles / freq
                d += (om.dram_read_eff + om.dram_write_eff) / bw
    steps = max(1, w.output_tokens - 1)
    return {'ttft_s': ttft, 'tpot_s': tpot,
            'decode_compute_s': c / steps, 'decode_dram_s': d / steps}


def sweep(report_path):
    rows = []
    preflight()

    rep = Report(
        report_path,
        "Compute/memory overlap",
        subtitle="What the no-overlap assumption costs every latency number",
        source="analysis/memory/overlap_run.py",
        setup=["Model: LLaMA-3-8B on Omni-LUT, 4-bit KV, standard attention.",
               "`serial` = `sum(max(compute, memory))`, the model every "
               "published number used. `pipelined` = `max(sum compute, sum "
               "memory)`. They bracket the truth; neither is it."])

    rep.summary([
        "**`serial` overstates decode TPOT by up to 1.75x** (32K, batch 1). "
        "That is larger than ThinK, residency, selection metadata and most "
        "eviction configurations were measured to *save* — the modelling "
        "assumption outweighs the techniques being modelled.",
        "**The gap tracks how mismatched the two roofs are.** It is largest "
        "where a phase alternates memory-bound AW projections with a "
        "compute-bound `attn_v`, and collapses toward 1.0x once one resource "
        "dominates — at batch 32 decode is compute-bound throughout and "
        "overlap buys only 1.05–1.12x.",
        "**Prefill barely moves** (1.00–1.07x): it is compute-bound almost "
        "everywhere, so there is little memory time to hide.",
        "**It does change a conclusion: eviction's batch-1 gains were inflated "
        "by the assumption.** `evict-1024` at 32K goes **2.460x to 1.452x** "
        "once overlap is allowed — 41% of the claimed speedup was the model, "
        "not the technique. At batch 32 the same config holds 15.957x to "
        "14.323x.",
        "**The reason is which roof each one attacks.** At batch 1 eviction's "
        "lever is DRAM traffic, and pipelining already hides DRAM under "
        "compute, so the two compete for the same win. At batch 32 eviction "
        "also cuts `attn_v` cycles — `kv_len` is its `K` — so it lowers the "
        "binding roof and keeps paying. **This sharpens §12 rather than "
        "contradicting it: batch 1 was an even worse place to measure than §12 "
        "said.**",
    ])

    # ---- A ------------------------------------------------------------------
    rep.section(
        "A. Decode — what the assumption costs",
        "Per-token roofline time. `compute` and `DRAM` are the phase totals the "
        "two models combine differently.")
    for batch in BATCHES:
        trows = []
        for ctx in CONTEXTS:
            s = measure('serial', batch, ctx)
            p = measure('pipelined', batch, ctx)
            rows.append({'section': 'A', 'batch': batch, 'context': ctx,
                         'serial_tpot': s['tpot_s'], 'pipelined_tpot': p['tpot_s'],
                         **{f'serial_{k}': v for k, v in s.items()}})
            roof = ('DRAM' if s['decode_dram_s'] > s['decode_compute_s']
                    else 'compute')
            trows.append([f"{ctx:,}", f"{s['decode_compute_s']*1e3:.1f} ms",
                          f"{s['decode_dram_s']*1e3:.1f} ms",
                          f"{s['tpot_s']*1e3:.2f} ms",
                          f"{p['tpot_s']*1e3:.2f} ms",
                          f"{s['tpot_s']/p['tpot_s']:.2f}x", roof])
        rep.table(["context", "compute", "DRAM", "serial TPOT",
                   "pipelined TPOT", "overstated by", "binding roof"], trows,
                  aligns="rrrrrrc", caption=f"batch {batch}")
    rep.note(
        "**The gap is a balance measure, and 2x is its hard ceiling.** "
        "`sum(max)` can exceed `max(sum)` by at most 2x, reached exactly when "
        "the two resources are equal. At 32K batch 1 they nearly are — **73.3 "
        "ms of compute against 73.4 ms of DRAM** — so the 1.75x measured there "
        "is close to the worst case the assumption can produce, not a midpoint. "
        "At batch 32 compute leads DRAM roughly 3:1, little is left to hide, "
        "and the same mechanism yields 1.12x.")
    rep.note(
        "**That near-tie at 32K batch 1 is itself the finding.** §3 explained "
        "decode's compute/roofline gap narrowing from 6.8x to 1.7x as attention "
        "compute growing to meet the memory wall; this shows it lands almost "
        "exactly on it. That is the single worst operating point for a "
        "no-overlap model, and it is the one `study.md` quotes most.")

    # ---- B ------------------------------------------------------------------
    rep.section("B. Prefill", "TTFT, same two models.")
    trows = []
    for ctx in CONTEXTS:
        s = measure('serial', 1, ctx)
        p = measure('pipelined', 1, ctx)
        rows.append({'section': 'B', 'context': ctx, 'serial_ttft': s['ttft_s'],
                     'pipelined_ttft': p['ttft_s']})
        trows.append([f"{ctx:,}", f"{s['ttft_s']:.2f} s", f"{p['ttft_s']:.2f} s",
                      f"{s['ttft_s']/p['ttft_s']:.2f}x"])
    rep.table(["context", "serial TTFT", "pipelined TTFT", "overstated by"],
              trows)
    rep.note(
        "Prefill is compute-bound almost everywhere — `LUT_WS` streams a long "
        "activation dimension against a resident weight tile — so there is "
        "little memory time available to hide and the two models nearly agree.")

    # ---- C ------------------------------------------------------------------
    rep.section(
        "C. Does it change any conclusion? — yes, at batch 1",
        "Decode TPOT speedup over dense at 32K, under each overlap model. This "
        "was written expecting the ratios to survive, since both numerator and "
        "denominator move. They do not, and the reason is worth more than the "
        "expectation was.")
    trows = []
    for label, budget in (('evict 4096', 4096), ('evict 1024', 1024),
                          ('evict 256', 256)):
        cells = [label]
        for batch in (1, 32):
            base_s = measure('serial', batch, 32768)['tpot_s']
            base_p = measure('pipelined', batch, 32768)['tpot_s']
            ms = measure('serial', batch, 32768, sim_cls=KVBudgetSimulator,
                         kv_budget=budget)['tpot_s']
            mp = measure('pipelined', batch, 32768, sim_cls=KVBudgetSimulator,
                         kv_budget=budget)['tpot_s']
            rows.append({'section': 'C', 'technique': label, 'batch': batch,
                         'serial_speedup': base_s / ms,
                         'pipelined_speedup': base_p / mp})
            cells += [f"{base_s/ms:.3f}x", f"{base_p/mp:.3f}x"]
        trows.append(cells)
    rep.table(["technique", "batch 1 serial", "batch 1 pipelined",
               "batch 32 serial", "batch 32 pipelined"], trows, aligns="lrrrr")
    rep.note(
        "**At batch 1 roughly 40% of eviction's measured speedup was the "
        "no-overlap assumption.** Once DRAM can hide under compute, cutting "
        "DRAM further buys nothing: the compute roof (73 ms at 32K) does not "
        "move, and eviction at batch 1 is a pure traffic technique. Note the "
        "three budgets converge — 1.391x / 1.452x / 1.468x — because they are "
        "all pressed against the same compute roof. **Under `serial` they still "
        "look separable (2.156x / 2.460x / 2.538x), which is the assumption "
        "manufacturing a distinction that a pipelined machine would not see.**")
    rep.note(
        "**At batch 32 they largely survive** (15.957x to 14.323x) because "
        "eviction there is not only a traffic technique: `kv_len` is `attn_v`'s "
        "reduction dimension, so cutting entries lowers the *compute* roof too. "
        "A technique that moves both roofs is robust to how they are combined; "
        "one that moves only the slack roof is not.")
    rep.note(
        "**This sharpens §12 instead of contradicting it.** §12 said "
        "entry-count techniques were measured in the wrong place because batch "
        "1 gives them only a tenth of decode traffic to attack. This adds a "
        "second, independent reason the batch-1 numbers overstate: even the "
        "traffic they *do* remove was being charged as if none of it could "
        "overlap. **Rankings by batch are unchanged; batch-1 magnitudes are "
        "upper bounds twice over.**")

    # ---- D ------------------------------------------------------------------
    rep.section("D. What neither model captures")
    rep.note(
        "**`pipelined` assumes buffering it never checks for.** Hiding op i+1's "
        "operands behind op i's compute needs two operand sets resident, and "
        "nothing here confirms `sram_capacity_kb` allows it. The bound is "
        "reachable in principle, not certified for any particular schedule.")
    rep.note(
        "**Dependencies are ignored.** Within a layer, `attn_v` cannot start "
        "before softmax, which cannot start before `qk` — so the reachable "
        "overlap is across *layers* and decode steps, not within a layer's "
        "attention chain. A real scheduler would also face a pipeline fill and "
        "drain at each phase boundary, which is uncharged here.",)
    rep.note(
        "**Non-GEMM work is outside both models.** `compute_roofline_latency` "
        "iterates GEMM operations only, so the VPU softmax — 27.9% of prefill "
        "cycles at 32K (§2) — is absent from both columns rather than "
        "overlapped in one of them.")
    rep.note(
        "**The truth is somewhere between, and closer to `pipelined` for long "
        "chains.** Decode issues 32 layers x ~9 operations back to back with no "
        "control flow between them, which is the regime software pipelining "
        "works best in. Reporting the pair is more honest than picking one.")

    rep.save()
    return rows


def preflight():
    """Five checks, each pinning a claim the report makes."""
    # 1. "serial" is exactly today's model.
    s = measure('serial', 1, 32768)
    ref_hw = hw('serial')
    assert ref_hw.overlap_model == 'serial', "serial must be the default path"
    assert HardwareConfig(array_m=32, array_n=4).overlap_model == 'serial', \
        "the shipped default must be serial"

    # 2. Pipelined is never slower -- max(sums) <= sum(maxes) always.
    for ctx in CONTEXTS:
        for b in BATCHES:
            a = measure('serial', b, ctx)
            p = measure('pipelined', b, ctx)
            assert p['tpot_s'] <= a['tpot_s'] + 1e-12, \
                f"pipelined slower at {ctx}/{b}"
            assert p['ttft_s'] <= a['ttft_s'] + 1e-12

    # 3. Pipelined decode equals max(compute, dram) exactly -- the definition.
    p = measure('pipelined', 1, 32768)
    want = max(p['decode_compute_s'], p['decode_dram_s'])
    assert abs(p['tpot_s'] - want) / want < 1e-9, \
        f"pipelined TPOT {p['tpot_s']} != max(compute, dram) {want}"

    # 4. The breakdown's parts still sum to the total under both models.
    m = get_model_config(MODEL)
    w = WorkloadConfig(batch_size=1, input_tokens=32768,
                       output_tokens=OUTPUT_TOKENS, flash_block_size=0)
    for mode in ('serial', 'pipelined'):
        sim = Simulator(hw(mode))
        r = sim.simulate(m, w)
        bd = sim.compute_roofline_latency_breakdown(r, w)
        ttft, _ = sim.compute_roofline_latency(r, w)
        gemm = bd['ttft_aa'] + bd['ttft_aw']
        assert abs(gemm - ttft) / ttft < 1e-9, \
            f"{mode}: breakdown AA+AW {gemm} != roofline TTFT {ttft}"

    # 5. An unknown string is treated as serial rather than silently pipelining.
    odd = measure('nonsense-value', 1, 32768)
    assert odd['tpot_s'] == s['tpot_s'], "unknown overlap_model must be serial"

    print("pre-flight: 5 checks passed")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rounds-model', default='tiled',
                   choices=('tiled', 'packed'),
                   help="'tiled' reproduces what study.md section 17 published; "
                        "'packed' applies stage 11's OS-V round-count fix, "
                        "which changes decode compute from batch 2 upward.")
    p.add_argument('--csv', default=os.path.join(_here, 'overlap.csv'))
    p.add_argument('--report',
                   default=os.path.join(_here, 'overlap_report.md'))
    args = p.parse_args()
    global ROUNDS_MODEL
    ROUNDS_MODEL = args.rounds_model

    rows = sweep(args.report)

    keys = sorted({k for r in rows for k in r})
    with open(args.csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows -> {args.csv}")
    print(f"Wrote report      -> {args.report}")


if __name__ == '__main__':
    main()
