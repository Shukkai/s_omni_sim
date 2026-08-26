"""
Every KV-reduction result in `study.md` was measured at batch 1.  This asks
whether that was the right place to measure.

Weight traffic during decode is *constant* in batch -- the weights are read once
and reused across the batch -- while KV traffic scales linearly with it.  So the
KV share of decode DRAM, which is the ceiling on what any KV technique can
possibly win, moves enormously with batch:

    batch 1,  8K context ->  10.1% of decode DRAM is KV
    batch 32, 32K context -> 93.5%

At batch 1 a KV technique is competing for a tenth of the traffic; at batch 32
it is competing for nearly all of it.  ThinK's 1.005x-1.052x grid in section 5
was not measuring a weak technique so much as a technique boxed in by an
arithmetic ceiling: pruning half of a 10% slice cannot yield more than ~2%.

This sweep re-runs eviction, ThinK channel pruning and select-without-evict at
batch 1 / 8 / 32 to separate "the technique does not help" from "the technique
was measured where nothing could help".

Everything is decode-side; prefill is unaffected by all three techniques (and
by the batch argument here, which only enters decode's KV term).

Usage:
    python batch_run.py
    python batch_run.py --csv batch_scaling.csv --report batch_scaling_report.md
"""

import argparse
import csv
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, '..', '..'))
for p in ('simulator', 'analysis', 'analysis/cycle_breakdown',
          'analysis/compact_breakdown', 'analysis/channel_prune_breakdown',
          'analysis/memory'):
    sys.path.insert(0, os.path.join(_root, *p.split('/')))

from simulator import HardwareConfig, WorkloadConfig, Simulator   # noqa: E402
from model_configs import get_model_config                        # noqa: E402
from kv_budget import KVBudgetSimulator                           # noqa: E402
from think_prune import ThinKSimulator                            # noqa: E402
from selective_attn import SelectiveAttnSimulator                 # noqa: E402
from report import Report                                         # noqa: E402

MODEL = 'LLaMA-3-8B'
BATCHES = [1, 8, 32]
CONTEXTS = [8192, 32768]
OUTPUT_TOKENS = 4     # TPOT is per-token; more steps change no ratio here
HEAD_DIM = 128
LAMBDA = 0.4          # ThinK pruning fraction -> 77 of 128 channels retained
D_RET = int(round(HEAD_DIM * (1 - LAMBDA)))


def hw():
    return HardwareConfig(
        array_m=32, array_n=4, FPE_array_size=64,
        act_bits=16, accumulate_bits=32, weight_bits=4, kv_cache_bits=4,
        AW_mode="OMNI", AA_mode="OMNI",
    )


def _measure(sim, batch, context):
    model = get_model_config(MODEL)
    w = WorkloadConfig(batch_size=batch, input_tokens=context,
                       output_tokens=OUTPUT_TOKENS, flash_block_size=0)
    r = sim.simulate(model, w)
    aw = r.decode.get_aw_total().dram_read
    aa = r.decode.get_aa_total().dram_read
    ttft, tpot = sim.compute_roofline_latency(r, w)
    return {'aw_dram': aw, 'aa_dram': aa, 'decode_dram': aw + aa,
            'kv_share': aa / (aw + aa) if (aw + aa) else 0.0,
            'tpot_s': tpot}


# Each technique as (label, factory).  All are decode-side add-ons over the
# same hardware, so they are directly comparable.
TECHNIQUES = [
    ('dense',            lambda: Simulator(hw())),
    ('evict 4096',       lambda: KVBudgetSimulator(hw(), kv_budget=4096)),
    ('evict 1024',       lambda: KVBudgetSimulator(hw(), kv_budget=1024)),
    (f'ThinK-K d={D_RET}', lambda: ThinKSimulator(hw(), d_k_ret=D_RET)),
    ('select 25%',       lambda: SelectiveAttnSimulator(
        hw(), page_size=16, select_frac=0.25, summary_vectors=2)),
    ('select 3%',        lambda: SelectiveAttnSimulator(
        hw(), page_size=16, select_frac=0.03, summary_vectors=2)),
]


def sweep(report_path):
    rows = []
    rep = Report(
        report_path,
        "KV reduction vs batch",
        subtitle="Were the KV studies measured in the right place?",
        source="analysis/memory/batch_run.py",
        setup=["Model: LLaMA-3-8B on Omni-LUT, 4-bit KV, standard attention."])

    rep.summary([
        "Decode **weight traffic is constant in batch** — read once, reused across "
        "it — while KV traffic scales linearly. So the KV share of decode DRAM, "
        "the hard ceiling on any KV technique, runs from **10.1% at batch 1 / 8K "
        "to 93.5% at batch 32 / 32K**.",
        "**Entry-count techniques were measured in the wrong place.** At 32K, "
        "`evict-1024` goes 2.460x to **15.957x** and `select-3%` goes 2.404x to "
        "**12.854x** from batch 1 to 32. Their batch-1 numbers understate them "
        "several-fold.",
        "**Channel pruning does not recover** — ThinK-K holds ~1.05x at every "
        "batch. That is a *different* ceiling and must not be conflated: the bytes "
        "it saves were never on the critical path.",
    ])

    # ---- A ------------------------------------------------------------------
    rep.section(
        "A. The ceiling itself",
        "'Max speedup' is what a technique removing **all** KV traffic could "
        "achieve on decode DRAM.")
    trows = []
    for ctx in CONTEXTS:
        for b in BATCHES:
            m = _measure(Simulator(hw()), b, ctx)
            ceiling = 1 / (1 - m['kv_share']) if m['kv_share'] < 1 else float('inf')
            rows.append({'section': 'A', 'context': ctx, 'batch': b,
                         'technique': 'dense', **m})
            trows.append([f"{ctx:,}", str(b), f"{m['aw_dram']:,}",
                          f"{m['aa_dram']:,}", f"{m['kv_share']:.1%}",
                          f"{ceiling:.2f}x"])
    rep.table(["context", "batch", "weights (AW)", "KV + attn (AA)", "KV share",
               "max speedup"], trows)

    # ---- B ------------------------------------------------------------------
    rep.section("B. Decode TPOT speedup vs dense, by batch")
    for ctx in CONTEXTS:
        base = {b: _measure(Simulator(hw()), b, ctx)['tpot_s'] for b in BATCHES}
        trows = []
        for label, factory in TECHNIQUES:
            if label == 'dense':
                continue
            cells = [label]
            for b in BATCHES:
                m = _measure(factory(), b, ctx)
                sp = base[b] / m['tpot_s'] if m['tpot_s'] else 0.0
                rows.append({'section': 'B', 'context': ctx, 'batch': b,
                             'technique': label, 'speedup': sp, **m})
                cells.append(f"{sp:.3f}x")
            trows.append(cells)
        rep.table(["technique"] + [f"batch {b}" for b in BATCHES], trows,
                  aligns="l", caption=f"context {ctx:,}")
    rep.note(
        "The same technique, the same hardware, the same context — only the batch "
        "changes. What looked marginal at batch 1 is the dominant effect at "
        "batch 32.")

    # ---- C ------------------------------------------------------------------
    rep.section(
        "C. Fraction of dense decode DRAM remaining",
        "Context 32,768. Separates cause from effect: how much traffic each "
        "technique actually removes.")
    base = {b: _measure(Simulator(hw()), b, 32768)['decode_dram'] for b in BATCHES}
    trows = []
    for label, factory in TECHNIQUES:
        if label == 'dense':
            continue
        cells = [label]
        for b in BATCHES:
            m = _measure(factory(), b, 32768)
            frac = m['decode_dram'] / base[b] if base[b] else 0.0
            rows.append({'section': 'C', 'context': 32768, 'batch': b,
                         'technique': label, 'dram_frac': frac, **m})
            cells.append(f"{frac:.3f}x")
        trows.append(cells)
    rep.table(["technique"] + [f"batch {b}" for b in BATCHES], trows, aligns="l")
    rep.note(
        "At batch 1 even aggressive selection barely moves the total, because the "
        "weights it cannot touch are most of it.")

    # ---- D ------------------------------------------------------------------
    rep.section("D. Two different stories, which must not be conflated")
    rep.note(
        "**Entry-count techniques (eviction, selection) were measured in the wrong "
        "place.** They cut `kv_len`, so their saving is pure DRAM traffic — and "
        "traffic is exactly what batch makes dominant. At 32K, `evict-1024` goes "
        "2.460x to 15.957x and `select-3%` goes 2.404x to 12.854x from batch 1 to "
        "32. Re-baseline them at the batch the accelerator is meant to serve.")
    rep.note(
        "**Channel pruning (ThinK) does not recover: 1.034x to 1.054x at 32K.** "
        "This is not the same ceiling. ThinK *does* cut bytes — section C shows "
        "decode DRAM falling to 0.825x at batch 32 — but latency barely moves, so "
        "those bytes were never on the critical path. `attn_v` is compute-bound "
        "under a 4-bit KV cache and the `LUT_OS_V` round cost has no N term, so "
        "pruning `head_dim` idles array columns instead of saving cycles. That is "
        "the cycle null from `study.md` section 5, a property of the dataflow "
        "rather than of the measurement point. More batch cannot fix it; a "
        "different dataflow or head packing might.")

    rep.save()
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv', default=os.path.join(_here, 'batch_scaling.csv'))
    p.add_argument('--report',
                   default=os.path.join(_here, 'batch_scaling_report.md'))
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
