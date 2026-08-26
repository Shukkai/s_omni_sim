"""
What actually fits on chip: SRAM capacity as a constraint, not a report.

Until now `peak_sram_bytes` was computed and printed but never enforced, so
every latency result in `study.md` implicitly assumed unlimited on-chip memory.
`HardwareConfig.sram_capacity_kb` now makes an operation whose working set does
not fit pay for the spill (policy v1, see `_apply_sram_capacity`).  This sweep
turns that into the claim the accelerator story wants: at a given SRAM size,
which configurations are actually runnable, and what overflow costs when they
are not.

Two structural facts shape what this script can sweep.  Both are properties of
`_calculate_peak_sram` that enforcement merely made visible:

1. **Prefill holds the whole activation matrix.**  `A_bytes = M*K*act_bits/8`
   with `M` = the full prefill length, so prefill's working set is
   O(seq x d_model): 59 MB at 2K context, 2.1 GB at 32K.  No plausible SRAM
   fits that, so prefill overflows everywhere and its row of any fits/does-not-
   fit table is a constant.  A real accelerator tiles prefill over the sequence;
   the model does not.  That is a peak-SRAM modelling gap, not a hardware
   finding, and it is why the tables below lead with decode.

2. **Batch used to be a loop in one half of the model and a dimension in the
   other.**  Projections are issued as one GEMM with `proj_m = batch * seq_len`,
   so batch scaled their footprint; attention is issued once per (batch, head)
   with batch folded into `batch_size`, which `_calculate_peak_sram` ignored, so
   attention's footprint did not move with batch at all.  `sram_batch_model`
   settles it: `"sequential"` is the old behaviour and stays the default, and
   `"concurrent"` makes batch elements co-resident in attention too (heads still
   run back-to-back), which is what the projection side already assumed.  Under
   `"concurrent"` batch is a real capacity axis and section F answers "how much
   batch fits".

Prefill remains unusable for capacity claims because of fact 1, so the sweep is
decode-centric: section D reports the decode spill charge, and the prefill charge
is omitted because policy v1 prices a residency assumption that is itself wrong.

Usage:
    python capacity_run.py
    python capacity_run.py --csv capacity.csv --report capacity_report.md

The CSV is the raw sweep; `capacity_report.md` is the readable one, and it is
tracked, because `.gitignore` excludes `*.csv` and `*.txt` and a result nobody
can open is not a result.
"""

import argparse
import csv
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, '..', '..'))
sys.path.insert(0, os.path.join(_root, 'simulator'))
sys.path.insert(0, os.path.join(_root, 'analysis'))
sys.path.insert(0, os.path.join(_root, 'analysis', 'cycle_breakdown'))
sys.path.insert(0, os.path.join(_root, 'analysis', 'compact_breakdown'))

from simulator import HardwareConfig, WorkloadConfig      # noqa: E402
from model_configs import get_model_config                # noqa: E402
from kv_budget import KVBudgetSimulator                   # noqa: E402
from report import Report                                # noqa: E402


MODEL = 'LLaMA-3-8B'
OUTPUT_TOKENS = 32


def omni4(sram_capacity_kb: int,
          batch_model: str = "sequential") -> HardwareConfig:
    """The study's main configuration: Omni-LUT with a 4-bit KV cache."""
    return HardwareConfig(
        array_m=32, array_n=4, FPE_array_size=64,
        act_bits=16, accumulate_bits=32, weight_bits=4, kv_cache_bits=4,
        AW_mode="OMNI", AA_mode="OMNI",
        sram_capacity_kb=sram_capacity_kb,
        sram_batch_model=batch_model,
    )


# Decode-scale capacities: the range a real on-chip buffer actually spans.
# 1024 and 2176 are the two thresholds the sweep resolves, so both are bracketed.
CAPACITIES_KB = [256, 512, 768, 1024, 1536, 2048, 2176, 4096]
CONTEXTS = [2048, 8192, 32768]
KV_BUDGETS = [0, 4096, 1024, 256]     # 0 = dense

# Batch sweep: capacities span the range where batch actually binds, and the
# decode length is cut to keep the O(batch x capacity x budget) sweep tractable
# -- peak SRAM is a per-step property, so fewer steps changes nothing about it.
BATCHES = [1, 2, 4, 8, 16, 32]
BATCH_CAPACITIES_KB = [1024, 2048, 4096, 8192, 16384, 32768]
SHORT_DECODE = 2


def run(capacity_kb: int, context: int, kv_budget: int, batch: int = 1,
        batch_model: str = "sequential", output_tokens: int = OUTPUT_TOKENS):
    model = get_model_config(MODEL)
    workload = WorkloadConfig(batch_size=batch, input_tokens=context,
                              output_tokens=output_tokens, flash_block_size=0)
    sim = KVBudgetSimulator(omni4(capacity_kb, batch_model),
                            kv_budget=kv_budget)
    results = sim.simulate(model, workload)
    pre = results.prefill.get_total_metrics()
    dec = results.decode.get_total_metrics()
    ttft, tpot = sim.compute_roofline_latency(results, workload)
    return {
        'capacity_kb': capacity_kb,
        'context': context,
        'kv_budget': kv_budget,
        'batch': batch,
        'batch_model': batch_model,
        'prefill_peak': results.prefill.peak_sram_bytes,
        'decode_peak': results.decode.peak_sram_bytes,
        'prefill_overflow': pre.sram_overflow,
        'decode_overflow': dec.sram_overflow,
        'decode_refetch': dec.sram_refetch_bytes,
        'prefill_refetch': pre.sram_refetch_bytes,
        'decode_dram_read': dec.dram_read,
        'ttft_s': ttft,
        'tpot_s': tpot,
    }


def _label(kvb: int) -> str:
    return 'dense' if kvb == 0 else str(kvb)


def sweep(report_path):
    rows = []
    rep = Report(
        report_path,
        "SRAM capacity",
        subtitle="What actually fits on chip, and how much batch it supports",
        source="analysis/memory/capacity_run.py",
        setup=["Model: LLaMA-3-8B on Omni-LUT, 4-bit KV, standard attention."])

    rep.summary([
        "**Decode's working set is floored at 924.5 KB** by the FFN/projection "
        "tiles. The KV tile only becomes the binding term past ~16K context — "
        "below that, capacity is not the constraint at all.",
        "**At 32K a KV budget buys a smaller chip**: 2,176 KB dense vs 1,024 KB "
        "for any budget ≤ 4096 — 2.1x less SRAM.",
        "**Batch and capacity trade one-for-one** under `sram_batch_model="
        "\"concurrent\"`. At 32K in 4 MB: dense fits batch 1, a 4096-entry budget "
        "fits 8, a 1024-entry budget fits 32.",
        "**Prefill is excluded and this is a model gap, not a result** — "
        "`_calculate_peak_sram` holds the entire activation matrix, 2.1 GB at 32K, "
        "so it overflows at every plausible capacity.",
    ])

    # ---- A ------------------------------------------------------------------
    rep.section(
        "A. Where the working set sits",
        "Unlimited capacity, so these are requirements rather than outcomes.")
    unlimited = {}
    trows = []
    for ctx in CONTEXTS:
        for kvb in KV_BUDGETS:
            r = run(0, ctx, kvb)
            unlimited[(ctx, kvb)] = r
            rows.append(r)
            trows.append([f"{ctx:,}", _label(kvb), f"{r['prefill_peak']:,}",
                          f"{r['decode_peak']:,}",
                          f"{r['decode_peak'] / 1024:.1f} KB"])
    rep.table(["context", "kv_budget", "prefill peak", "decode peak",
               "decode peak"], trows)
    rep.note(
        "Prefill is O(seq x d_model) — the whole activation matrix is modelled as "
        "resident — so it overflows at every capacity below and its spill charge "
        "is a meaningless constant. **A real accelerator tiles prefill over the "
        "sequence; the model does not.** Everything below is therefore decode.")

    # ---- B ------------------------------------------------------------------
    rep.section("B. What decode fits into", "`.` = fits, `X` = spills.")
    trows = []
    for ctx in CONTEXTS:
        for kvb in KV_BUDGETS:
            cells = [f"{ctx:,}", _label(kvb)]
            for cap in CAPACITIES_KB:
                r = run(cap, ctx, kvb)
                rows.append(r)
                cells.append('X' if r['decode_overflow'] else '.')
            trows.append(cells)
    rep.table(["context", "kv_budget"] + [f"{c:,} KB" for c in CAPACITIES_KB],
              trows, aligns="rr")

    # ---- C ------------------------------------------------------------------
    rep.section("C. Smallest capacity decode runs in")
    trows = []
    for ctx in CONTEXTS:
        for kvb in KV_BUDGETS:
            hit = 0
            for cap in CAPACITIES_KB:
                if not run(cap, ctx, kvb)['decode_overflow']:
                    hit = cap
                    break
            trows.append([f"{ctx:,}", _label(kvb),
                          f"{hit:,} KB" if hit else f"> {CAPACITIES_KB[-1]:,} KB"])
    rep.table(["context", "kv_budget", "min capacity"], trows)

    # ---- D ------------------------------------------------------------------
    rep.section(
        "D. What under-provisioning costs",
        "Context 32,768, dense.")
    ref = unlimited[(32768, 0)]
    trows = []
    for cap in CAPACITIES_KB:
        r = run(cap, 32768, 0)
        f_p = r['tpot_s'] / ref['tpot_s'] if ref['tpot_s'] else 0.0
        trows.append([f"{cap:,} KB", f"{r['decode_refetch']:,}",
                      f"{r['tpot_s']*1e3:.3f} ms", f"{f_p:.3f}x"])
    rep.table(["capacity", "decode re-fetch", "TPOT", "vs unlimited"], trows,
              aligns="l")
    rep.note(
        "Read the 1,024 KB row against section B: flagged as an overflow there, "
        "charged nothing here. That is spill policy v1's documented lower bound — "
        "past 1,024 KB the binding term is the KV tile itself, and a tile that "
        "does not fit needs re-tiling, which v1 refuses to model. **The charge is "
        "non-monotonic: the larger overflow prices lower.** `sram_overflow` is the "
        "trustworthy output; `sram_refetch_bytes` is a first-order cost.\n"
        "Prefill's charge is omitted for the same reason squared: it overflows at "
        "every capacity here, so v1 returns a constant ~770 GB regardless. Not "
        "usable.")

    # ---- E ------------------------------------------------------------------
    rep.section(
        "E. The two batch-residency models",
        "Context 32,768, dense. `_calculate_peak_sram` has no batch term, yet "
        "projections issue as one GEMM with `proj_m = batch x seq_len` — so batch "
        "was a dimension in one half of the model and a loop in the other.")
    trows = []
    for b in BATCHES:
        rs = run(0, 32768, 0, batch=b, batch_model="sequential",
                 output_tokens=SHORT_DECODE)
        rc = run(0, 32768, 0, batch=b, batch_model="concurrent",
                 output_tokens=SHORT_DECODE)
        rows.extend([rs, rc])
        trows.append([str(b), f"{rs['decode_peak']:,}", f"{rc['decode_peak']:,}",
                      f"{rc['decode_peak'] / 1024 / 1024:.2f} MB"])
    rep.table(["batch", "sequential", "concurrent", "concurrent"], trows)
    rep.note(
        "`\"sequential\"` is flat by construction — attention runs one "
        "(batch, head) instance at a time. `\"concurrent\"` is linear, which is "
        "what the projection side already assumed. It remains a **scheduling "
        "assumption, not a measurement**, and `\"sequential\"` stays the default.")

    # ---- F ------------------------------------------------------------------
    rep.section(
        "F. How much batch fits",
        f"`sram_batch_model=\"concurrent\"`. `-` = even batch 1 overflows; "
        f"{BATCHES[-1]} is the sweep ceiling, so those cells mean 'at least that'.")
    trows = []
    for ctx in CONTEXTS:
        for kvb in KV_BUDGETS:
            cells = [f"{ctx:,}", _label(kvb)]
            for cap in BATCH_CAPACITIES_KB:
                fits = 0
                for b in BATCHES:
                    r = run(cap, ctx, kvb, batch=b, batch_model="concurrent",
                            output_tokens=SHORT_DECODE)
                    if r['decode_overflow']:
                        break
                    fits = b
                cells.append(str(fits) if fits else '-')
            trows.append(cells)
    rep.table(["context", "kv_budget"] +
              [f"{c:,} KB" for c in BATCH_CAPACITIES_KB], trows, aligns="rr")
    rep.note(
        "Doubling capacity doubles the batch, and at fixed capacity a KV budget "
        "buys batch directly: **at 32K in 4 MB, dense fits batch 1, a 4096-entry "
        "budget fits 8, and a 1024-entry budget fits 32.** This is the claim the "
        "study wanted and could not previously make — but it rests on "
        "`\"concurrent\"`, a scheduling assumption.")

    rep.save()
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv', default=os.path.join(_here, 'capacity.csv'))
    p.add_argument('--report',
                   default=os.path.join(_here, 'capacity_report.md'))
    args = p.parse_args()

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
