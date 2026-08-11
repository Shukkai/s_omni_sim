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
import contextlib
import csv
import io
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, '..', '..'))
for p in ('simulator', 'analysis/cycle_breakdown', 'analysis/compact_breakdown',
          'analysis/channel_prune_breakdown', 'analysis/selective_attn'):
    sys.path.insert(0, os.path.join(_root, *p.split('/')))

from simulator import HardwareConfig, WorkloadConfig, Simulator   # noqa: E402
from model_configs import get_model_config                        # noqa: E402
from kv_budget import KVBudgetSimulator                           # noqa: E402
from think_prune import ThinKSimulator                            # noqa: E402
from selective_attn import SelectiveAttnSimulator                 # noqa: E402

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


def sweep():
    rows = []

    # ---- A. The ceiling itself ---------------------------------------------
    print("A. KV share of decode DRAM -- the ceiling on any KV technique\n")
    print(f"  {'context':>8} {'batch':>6} | {'weights (AW)':>17} "
          f"{'KV+attn (AA)':>17} {'KV share':>9} {'max speedup':>12}")
    for ctx in CONTEXTS:
        for b in BATCHES:
            m = _measure(Simulator(hw()), b, ctx)
            ceiling = 1 / (1 - m['kv_share']) if m['kv_share'] < 1 else float('inf')
            rows.append({'section': 'A', 'context': ctx, 'batch': b,
                         'technique': 'dense', **m})
            print(f"  {ctx:>8} {b:>6} | {m['aw_dram']:>17,} {m['aa_dram']:>17,} "
                  f"{m['kv_share']:>8.1%} {ceiling:>11.2f}x")
    print("\n  Weight traffic is constant in batch -- read once, reused across the\n"
          "  batch -- while KV scales linearly.  'max speedup' is what a technique\n"
          "  that removed *all* KV traffic could achieve on decode DRAM.\n")

    # ---- B. Each technique across batch ------------------------------------
    print("B. Decode TPOT speedup vs dense, by batch\n")
    for ctx in CONTEXTS:
        print(f"  context {ctx:,}")
        header = ''.join(f"{'batch ' + str(b):>13}" for b in BATCHES)
        print(f"  {'technique':>16}{header}")
        base = {b: _measure(Simulator(hw()), b, ctx)['tpot_s'] for b in BATCHES}
        for label, factory in TECHNIQUES:
            if label == 'dense':
                continue
            cells = ''
            for b in BATCHES:
                m = _measure(factory(), b, ctx)
                sp = base[b] / m['tpot_s'] if m['tpot_s'] else 0.0
                rows.append({'section': 'B', 'context': ctx, 'batch': b,
                             'technique': label, 'speedup': sp, **m})
                cells += f"{sp:>12.3f}x"
            print(f"  {label:>16}{cells}")
        print()
    print("  The same technique, the same hardware, the same context -- only the\n"
          "  batch changes.  What looked marginal at batch 1 is the dominant\n"
          "  effect at batch 32.\n")

    # ---- C. DRAM reduction, to separate cause from effect ------------------
    print("C. Decode DRAM vs dense (bytes), by batch  (context 32,768)\n")
    header = ''.join(f"{'batch ' + str(b):>13}" for b in BATCHES)
    print(f"  {'technique':>16}{header}")
    base = {b: _measure(Simulator(hw()), b, 32768)['decode_dram'] for b in BATCHES}
    for label, factory in TECHNIQUES:
        if label == 'dense':
            continue
        cells = ''
        for b in BATCHES:
            m = _measure(factory(), b, 32768)
            frac = m['decode_dram'] / base[b] if base[b] else 0.0
            rows.append({'section': 'C', 'context': 32768, 'batch': b,
                         'technique': label, 'dram_frac': frac, **m})
            cells += f"{frac:>12.3f}x"
        print(f"  {label:>16}{cells}")
    print("\n  Fraction of dense decode DRAM remaining.  At batch 1 even aggressive\n"
          "  selection barely moves the total, because the weights it cannot touch\n"
          "  are most of it.\n")

    # ---- D. Reading this ---------------------------------------------------
    print("D. Two different stories, and they must not be conflated\n")
    print("  ENTRY-COUNT techniques (eviction, selection) were measured in the\n"
          "  wrong place.  They cut kv_len, so their saving is pure DRAM traffic,\n"
          "  and traffic is exactly what batch makes dominant.  evict-1024 at 32K\n"
          "  goes 2.46x -> 15.96x from batch 1 to 32; select-3% goes 2.40x ->\n"
          "  12.85x.  Their batch-1 numbers understate them several-fold.\n")
    print("  CHANNEL pruning (ThinK) does NOT recover: 1.034x -> 1.054x at 32K.\n"
          "  This is not the same ceiling.  ThinK does cut bytes -- section C shows\n"
          "  decode DRAM falling to 0.825x at batch 32 -- but latency barely moves,\n"
          "  so the bytes it saves were not on the critical path.  attn_v is\n"
          "  compute-bound under a 4-bit KV cache, and the LUT_OS_V round cost has\n"
          "  no N term, so pruning head_dim idles array columns instead of saving\n"
          "  cycles.  That is the cycle null from section 5, and it is a property\n"
          "  of the dataflow, not of the measurement point.  More batch cannot fix\n"
          "  it; a different dataflow or head-packing might.\n")
    print("  So: re-baseline eviction and selection at the batch the accelerator is\n"
          "  actually meant to serve.  Leave section 5's ThinK conclusion standing --\n"
          "  it was right, for the reason it gave.\n")
    return rows


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for st in self._streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self._streams:
            st.flush()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv', default=os.path.join(_here, 'batch_scaling.csv'))
    p.add_argument('--report',
                   default=os.path.join(_here, 'batch_scaling_report.md'))
    args = p.parse_args()

    buf = io.StringIO()
    with contextlib.redirect_stdout(_Tee(sys.stdout, buf)):
        rows = sweep()

    keys = sorted({k for r in rows for k in r})
    with open(args.csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    with open(args.report, 'w') as f:
        f.write("# KV reduction vs batch: were we measuring in the right place?\n\n")
        f.write("Generated by `analysis/batch_scaling/batch_run.py`.\n"
                "Model: LLaMA-3-8B on Omni-LUT, 4-bit KV.  See the module\n"
                "docstring for why batch is the axis that matters here.\n\n")
        f.write("```\n")
        f.write(buf.getvalue())
        f.write("```\n")

    print(f"Wrote {len(rows)} rows -> {args.csv}")
    print(f"Wrote report      -> {args.report}")


if __name__ == '__main__':
    main()
