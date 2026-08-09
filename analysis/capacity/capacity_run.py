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

2. **Batch enters through M, not through a batch term.**  `_calculate_peak_sram`
   has no batch factor -- its docstring says batch elements are sequential --
   but projections are issued as one GEMM with `proj_m = batch * seq_len`, so
   batch scales the working set anyway, through the activation operand.  The two
   halves of the model disagree about whether batch is a loop or a dimension.
   Section E shows the consequence; resolving it is a modelling decision, not a
   sweep, so this script reports it rather than papering over it.

Consequently the only place the capacity question is both real and well-posed
is decode, where the working set is small enough to fit a plausible buffer and a
KV budget shrinks it directly.  Section D therefore reports the decode charge;
the prefill charge is printed but is not a usable number, because policy v1
prices a residency assumption (fact 1) that is itself wrong.

Usage:
    python capacity_run.py
    python capacity_run.py --csv capacity.csv
"""

import argparse
import csv
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, '..', '..'))
sys.path.insert(0, os.path.join(_root, 'simulator'))
sys.path.insert(0, os.path.join(_root, 'analysis', 'cycle_breakdown'))
sys.path.insert(0, os.path.join(_root, 'analysis', 'compact_breakdown'))

from simulator import HardwareConfig, WorkloadConfig      # noqa: E402
from model_configs import get_model_config                # noqa: E402
from kv_budget import KVBudgetSimulator                   # noqa: E402


MODEL = 'LLaMA-3-8B'
OUTPUT_TOKENS = 32


def omni4(sram_capacity_kb: int) -> HardwareConfig:
    """The study's main configuration: Omni-LUT with a 4-bit KV cache."""
    return HardwareConfig(
        array_m=32, array_n=4, FPE_array_size=64,
        act_bits=16, accumulate_bits=32, weight_bits=4, kv_cache_bits=4,
        AW_mode="OMNI", AA_mode="OMNI",
        sram_capacity_kb=sram_capacity_kb,
    )


# Decode-scale capacities: the range a real on-chip buffer actually spans.
# 1024 and 2176 are the two thresholds the sweep resolves, so both are bracketed.
CAPACITIES_KB = [256, 512, 768, 1024, 1536, 2048, 2176, 4096]
CONTEXTS = [2048, 8192, 32768]
KV_BUDGETS = [0, 4096, 1024, 256]     # 0 = dense


def run(capacity_kb: int, context: int, kv_budget: int, batch: int = 1):
    model = get_model_config(MODEL)
    workload = WorkloadConfig(batch_size=batch, input_tokens=context,
                              output_tokens=OUTPUT_TOKENS, flash_block_size=0)
    sim = KVBudgetSimulator(omni4(capacity_kb), kv_budget=kv_budget)
    results = sim.simulate(model, workload)
    pre = results.prefill.get_total_metrics()
    dec = results.decode.get_total_metrics()
    ttft, tpot = sim.compute_roofline_latency(results, workload)
    return {
        'capacity_kb': capacity_kb,
        'context': context,
        'kv_budget': kv_budget,
        'batch': batch,
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv', default=os.path.join(_here, 'capacity.csv'))
    args = p.parse_args()

    rows = []

    # ---- A. Where the working set sits -------------------------------------
    print("A. Peak working set, unlimited capacity\n")
    print(f"  {'context':>8} {'kv_budget':>10} {'prefill':>16} {'decode':>14} "
          f"{'decode':>10}")
    print(f"  {'':>8} {'':>10} {'':>16} {'(bytes)':>14} {'(KB)':>10}")
    unlimited = {}
    for ctx in CONTEXTS:
        for kvb in KV_BUDGETS:
            r = run(0, ctx, kvb)
            unlimited[(ctx, kvb)] = r
            rows.append(r)
            print(f"  {ctx:>8} {_label(kvb):>10} {r['prefill_peak']:>16,} "
                  f"{r['decode_peak']:>14,} {r['decode_peak'] / 1024:>10.1f}")
    print("\n  Prefill is O(seq x d_model) -- the whole activation matrix is\n"
          "  modelled as resident, so it overflows at every capacity below.\n")

    # ---- B. What decode fits into ------------------------------------------
    print("B. Decode: fits / overflows by SRAM capacity  (. = fits, X = spills)\n")
    header = ''.join(f"{c:>7}" for c in CAPACITIES_KB)
    print(f"  {'context':>8} {'kv_budget':>10}{header}   (KB)")
    for ctx in CONTEXTS:
        for kvb in KV_BUDGETS:
            cells = ''
            for cap in CAPACITIES_KB:
                r = run(cap, ctx, kvb)
                rows.append(r)
                cells += f"{'X' if r['decode_overflow'] else '.':>7}"
            print(f"  {ctx:>8} {_label(kvb):>10}{cells}")
    print()

    # ---- C. Smallest capacity decode runs in --------------------------------
    print("C. Smallest swept capacity with no decode overflow\n")
    print(f"  {'context':>8} {'kv_budget':>10} {'min capacity':>16}")
    for ctx in CONTEXTS:
        for kvb in KV_BUDGETS:
            hit = 0
            for cap in CAPACITIES_KB:
                if not run(cap, ctx, kvb)['decode_overflow']:
                    hit = cap
                    break
            shown = f"{hit:,} KB" if hit else f"> {CAPACITIES_KB[-1]:,} KB"
            print(f"  {ctx:>8} {_label(kvb):>10} {shown:>16}")
    print()

    # ---- D. What under-provisioning costs -----------------------------------
    print("D. Cost of running decode under-provisioned  (context 32768, dense)\n")
    print(f"  {'capacity':>10} {'decode refetch':>18} {'TPOT':>11} {'vs unlim':>9}")
    ref = unlimited[(32768, 0)]
    for cap in CAPACITIES_KB:
        r = run(cap, 32768, 0)
        f_p = r['tpot_s'] / ref['tpot_s'] if ref['tpot_s'] else 0.0
        print(f"  {cap:>7,} KB {r['decode_refetch']:>18,} "
              f"{r['tpot_s'] * 1e3:>9.3f} ms {f_p:>8.3f}x")
    print("\n  Read the 1,024 KB row against table B: it is flagged as an overflow\n"
          "  there but charged nothing here.  That is policy v1's documented lower\n"
          "  bound -- past 1,024 KB the binding term is the KV tile itself, and a\n"
          "  tile that does not fit needs re-tiling, which v1 refuses to model.  So\n"
          "  the charge is non-monotonic: the *larger* overflow prices lower.  The\n"
          "  flag is the trustworthy output; the bytes are a first-order cost.\n")
    print("  Prefill's charge is omitted for the same reason squared: it overflows\n"
          "  at every capacity here, so v1 prices the residency assumption of fact 1\n"
          "  and returns a constant ~770 GB regardless of capacity.  Not usable.\n")

    # ---- E. How batch actually enters the footprint -------------------------
    print("E. Batch enters through proj_m = batch x seq_len, despite\n"
          "   _calculate_peak_sram having no batch term\n"
          "   (context 8192, dense, capacity 1024 KB)\n")
    print(f"  {'batch':>6} {'prefill peak':>16} {'decode peak':>14} "
          f"{'prefill refetch':>18}")
    for b in (1, 2, 4, 8):
        r = run(1024, 8192, 0, batch=b)
        rows.append(r)
        print(f"  {b:>6} {r['prefill_peak']:>16,} {r['decode_peak']:>14,} "
              f"{r['prefill_refetch']:>18,}")
    print()

    with open(args.csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {args.csv}")


if __name__ == '__main__':
    main()
