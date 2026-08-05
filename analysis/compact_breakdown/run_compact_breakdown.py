"""
Compaction breakdown: what a compacted KV cache is worth on Omni-LUT.

Three results, in increasing order of how much is genuinely unknown:

  A. PAYBACK   (analytical) -- what a one-time compaction costs, in decode
     steps.  Settled by arithmetic; reported so the paper does not argue for
     something that is not in doubt.
  B. REGIME MAP -- ceiling speedup over batch x context.  Bounds every other
     claim: at batch 1 / 2K context, decode is weight-bound and no amount of
     KV eviction helps.
  C. KNEE -- the per-round cost that does NOT shrink with the budget (LGU
     table generation, systolic fill/drain, operand issue, accumulator).
     Negligible at full cache, dominant at the aggressive budgets that KV
     compression papers headline.  Architecture-specific and unmeasured.

Usage:
    python run_compact_breakdown.py
    python run_compact_breakdown.py --contexts 32768 --batches 1,8,32
"""

import argparse
import csv
import json
import os
import sys
from multiprocessing import Pool, cpu_count

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, '..', '..'))
sys.path.insert(0, os.path.join(_root, 'simulator'))
sys.path.insert(0, os.path.join(_root, 'analysis', 'cycle_breakdown'))
sys.path.insert(0, _here)

from simulator import Simulator, WorkloadConfig, HardwareConfig   # noqa: E402
from model_configs import get_model_config, list_models           # noqa: E402
from cycle_units import compute_stage_cycle_breakdown             # noqa: E402
from kv_budget import (                                           # noqa: E402
    KVBudgetSimulator, attention_unit_cycles, compaction_payback, budget_tokens,
)

ATTN_STAGES = {'qk_matmul', 'attn_v_matmul', 'softmax', 'flash_attn'}


# ---- Simulation -------------------------------------------------------------

def build_hw(args) -> HardwareConfig:
    return HardwareConfig(
        array_m=args.array_m, array_n=args.array_n,
        replication=args.replication, FPE_array_size=args.fpe_size,
        act_bits=args.act_bits, accumulate_bits=32,
        weight_bits=args.weight_bits, kv_cache_bits=args.kv_bits,
        AW_mode=args.aw_mode, AA_mode=args.aa_mode,
        freq_mhz=args.freq_mhz, dram_bandwidth_gbps=args.dram_bw,
    )


def run_point(task):
    """Simulate one (batch, context, budget) point."""
    args, batch, context, frac = task
    model = get_model_config(args.model)
    hw = build_hw(args)
    k = budget_tokens(context, frac)

    workload = WorkloadConfig(
        batch_size=batch, input_tokens=context,
        output_tokens=args.output_tokens, flash_block_size=0,
    )
    sim = KVBudgetSimulator(hw, kv_budget=0 if frac >= 1.0 else k,
                            model_bqu=False)
    results = sim.simulate(model, workload)
    bd = compute_stage_cycle_breakdown(sim, results, workload)

    dec = bd['decode_per_token']
    cycles = sum(r['cycles'] for r in dec.values())
    eff_time = sum(r['eff_time'] for r in dec.values())
    kv_dram = sum(r['dram_bytes'] for s, r in dec.items() if s in ATTN_STAGES)
    total_dram = sum(r['dram_bytes'] for r in dec.values())
    attn_cycles = sum(r['cycles'] for s, r in dec.items() if s in ATTN_STAGES)

    # Fixed-vs-useful split inside the attention ops (the knee).
    knee = attention_unit_cycles(results.decode)
    steps = bd['num_decode_steps']

    return {
        'batch': batch, 'context': context, 'budget_frac': frac,
        'budget_tokens': k if frac < 1.0 else context,
        'decode_cycles': cycles,
        'decode_attn_cycles': attn_cycles,
        'decode_eff_time': eff_time,
        'decode_kv_dram': kv_dram,
        'decode_total_dram': total_dram,
        'kv_dram_share': kv_dram / total_dram if total_dram else 0.0,
        'attn_fixed_cycles': knee['fixed'] / steps,
        'attn_useful_cycles': knee['useful'] / steps,
        'attn_fixed_share': knee['fixed_share'],
        'prefill_kv_writeback': results.get_kv_cache_writes()['prefill_k_bytes']
                                + results.get_kv_cache_writes()['prefill_v_bytes'],
    }


# ---- Reporting --------------------------------------------------------------

def fmt_bytes(b):
    for unit, div in (('GB', 1e9), ('MB', 1e6), ('KB', 1e3)):
        if b >= div:
            return f"{b/div:.2f} {unit}"
    return f"{b:.0f} B"


def report_payback(rows, args):
    """A. Analytical compaction payback."""
    dense = {(r['batch'], r['context']): r for r in rows if r['budget_frac'] >= 1.0}
    print("\n" + "=" * 78)
    print("  A. COMPACTION PAYBACK  (analytical)")
    print("=" * 78)
    print("\n  One-time cost: stream the cache once, write back survivors.")
    print("  Per-token benefit: decode re-reads the whole cache every step.\n")

    b0 = min(r['batch'] for r in rows)
    for ctx in sorted({r['context'] for r in rows}):
        d = dense.get((b0, ctx))
        if not d:
            continue
        print(f"  context {ctx}, batch {b0}:  KV read/token = "
              f"{fmt_bytes(d['decode_kv_dram'])}, "
              f"prefill KV writeback = {fmt_bytes(d['prefill_kv_writeback'])}")
        print(f"    {'budget':>8} {'gather':>10} {'saves/token':>12} "
              f"{'payback':>10}   fused-into-writeback")
        for frac in sorted({r['budget_frac'] for r in rows}):
            if frac >= 1.0:
                continue
            p = compaction_payback(d['decode_kv_dram'], frac,
                                   d['prefill_kv_writeback'])
            print(f"    {frac*100:>7.0f}% {fmt_bytes(p['gather_bytes']):>10} "
                  f"{fmt_bytes(p['saving_bytes_per_token']):>12} "
                  f"{p['payback_tokens']:>8.2f} tok   "
                  f"0 gather, -{fmt_bytes(p['fused_prefill_saving_bytes'])} prefill")
        print()


def report_regime(rows, args):
    """B. Ceiling speedup over batch x context."""
    print("=" * 78)
    print(f"  B. REGIME MAP  (speedup at {args.headline_budget*100:.0f}% budget)")
    print("=" * 78)
    print("\n  Decode roofline time per token; ceiling = perfect compaction.\n")

    contexts = sorted({r['context'] for r in rows})
    batches = sorted({r['batch'] for r in rows})
    idx = {(r['batch'], r['context'], r['budget_frac']): r for r in rows}

    print(f"    {'batch':>6} {'context':>8} {'KV % DRAM':>11} "
          f"{'dense':>11} {'budgeted':>11} {'speedup':>9}")
    for b in batches:
        for c in contexts:
            d = idx.get((b, c, 1.0))
            k = idx.get((b, c, args.headline_budget))
            if not d or not k:
                continue
            sp = d['decode_eff_time'] / k['decode_eff_time']
            print(f"    {b:>6} {c:>8} {d['kv_dram_share']*100:>10.1f}% "
                  f"{d['decode_eff_time']*1e3:>10.1f}ms "
                  f"{k['decode_eff_time']*1e3:>10.1f}ms {sp:>8.2f}x")
    print()


def report_knee(rows, args):
    """C. Fixed per-round overhead vs budget."""
    print("=" * 78)
    print("  C. FIXED-OVERHEAD KNEE")
    print("=" * 78)
    print("\n  Cycles paid per round no matter how few operands it carries:")
    print("  LGU table generation + systolic fill/drain + issue + accumulator.")
    print("  These do not shrink with the KV budget.\n")

    b0 = min(r['batch'] for r in rows)
    for ctx in sorted({r['context'] for r in rows}):
        sel = sorted((r for r in rows if r['context'] == ctx and r['batch'] == b0),
                     key=lambda r: -r['budget_frac'])
        if not sel:
            continue
        print(f"  context {ctx}, batch {b0}:")
        print(f"    {'budget':>8} {'tokens':>8} {'useful':>14} {'fixed':>12} "
              f"{'fixed share':>12} {'attn cyc/tok':>14}")
        for r in sel:
            print(f"    {r['budget_frac']*100:>7.1f}% {r['budget_tokens']:>8} "
                  f"{r['attn_useful_cycles']:>14,.0f} "
                  f"{r['attn_fixed_cycles']:>12,.0f} "
                  f"{r['attn_fixed_share']*100:>11.1f}% "
                  f"{r['decode_attn_cycles']:>14,.0f}")
        print()


# ---- CLI --------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Compaction breakdown for KV eviction on Omni-LUT.")
    p.add_argument('--model', default='LLaMA-3-8B',
                   help=f"available: {', '.join(list_models())}")
    p.add_argument('--contexts', default='2048,8192,32768')
    p.add_argument('--batches', default='1,8,32')
    p.add_argument('--budgets', default='1.0,0.5,0.2,0.1,0.05,0.02,0.01,0.004',
                   help="retained fraction of context")
    p.add_argument('--headline-budget', type=float, default=0.2)
    p.add_argument('--output-tokens', type=int, default=32,
                   help="decode steps to simulate (TPOT is averaged)")

    p.add_argument('--array-m', type=int, default=32)
    p.add_argument('--array-n', type=int, default=4)
    p.add_argument('--replication', type=int, default=1)
    p.add_argument('--fpe-size', type=int, default=64)
    p.add_argument('--act-bits', type=int, default=16)
    p.add_argument('--weight-bits', type=int, default=4)
    p.add_argument('--kv-bits', type=int, default=4)
    p.add_argument('--aw-mode', default='OMNI')
    p.add_argument('--aa-mode', default='OMNI')
    p.add_argument('--freq-mhz', type=int, default=500)
    p.add_argument('--dram-bw', type=float, default=51.2)
    p.add_argument('--out-prefix', default='compact_breakdown')
    return p.parse_args()


def main():
    args = parse_args()
    contexts = [int(x) for x in args.contexts.split(',') if x.strip()]
    batches = [int(x) for x in args.batches.split(',') if x.strip()]
    budgets = sorted({float(x) for x in args.budgets.split(',') if x.strip()}
                     | {1.0, args.headline_budget}, reverse=True)

    print(f"Model:    {args.model}")
    print(f"Hardware: AW={args.aw_mode} AA={args.aa_mode} "
          f"{args.array_m}x{args.array_n} "
          f"W{args.weight_bits}A{args.act_bits}KV{args.kv_bits} "
          f"{args.freq_mhz} MHz {args.dram_bw} GB/s")
    print(f"Sweep:    contexts={contexts} batches={batches} budgets={budgets}")

    tasks = [(args, b, c, f) for b in batches for c in contexts for f in budgets]
    print(f"\nRunning {len(tasks)} simulations ...")

    n_workers = max(1, min(len(tasks), cpu_count() - 1))
    with Pool(processes=n_workers) as pool:
        rows = list(pool.imap_unordered(run_point, tasks))

    # Sanity: the dense point must reproduce a stock Simulator run.
    ref_args, b0, c0 = args, min(batches), max(contexts)
    ref = Simulator(build_hw(ref_args)).simulate(
        get_model_config(args.model),
        WorkloadConfig(batch_size=b0, input_tokens=c0,
                       output_tokens=args.output_tokens, flash_block_size=0))
    dense = next(r for r in rows
                 if r['batch'] == b0 and r['context'] == c0 and r['budget_frac'] >= 1.0)
    steps = max(1, args.output_tokens - 1)
    assert abs(dense['decode_cycles']
               - ref.decode.get_total_metrics().cycles / steps
               - sum(m.cycles for m in ref.decode.non_gemm_ops) / steps) < 1, \
        "dense point does not match a stock Simulator run"
    print("  dense baseline matches stock Simulator ✓")

    report_payback(rows, args)
    report_regime(rows, args)
    report_knee(rows, args)

    rows.sort(key=lambda r: (r['batch'], r['context'], -r['budget_frac']))
    with open(f"{args.out_prefix}.json", 'w') as f:
        json.dump({'config': vars(args), 'rows': rows}, f, indent=2)
    with open(f"{args.out_prefix}.csv", 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {args.out_prefix}.json and {args.out_prefix}.csv")


if __name__ == "__main__":
    main()
