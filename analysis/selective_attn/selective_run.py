"""
Select-without-evict vs compacted eviction: what selection actually costs.

`kv_budget.py` deferred selective reading because on a flat bandwidth model it
is indistinguishable from compacted eviction.  Stage 2 added burst granularity,
so the two can now be told apart -- and this sweep asks how much of selection's
paper saving survives once you charge it for (a) gathering pages instead of a
contiguous block and (b) reading per-page metadata over the whole context.

Usage:
    python selective_run.py
    python selective_run.py --csv selective.csv --report selective_report.md
"""

import argparse
import contextlib
import csv
import io
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, '..', '..'))
sys.path.insert(0, os.path.join(_root, 'simulator'))
sys.path.insert(0, os.path.join(_root, 'analysis', 'cycle_breakdown'))
sys.path.insert(0, os.path.join(_root, 'analysis', 'compact_breakdown'))
sys.path.insert(0, _here)

from simulator import HardwareConfig, WorkloadConfig     # noqa: E402
from model_configs import get_model_config               # noqa: E402
from kv_budget import KVBudgetSimulator                  # noqa: E402
from selective_attn import SelectiveAttnSimulator        # noqa: E402

MODEL = 'LLaMA-3-8B'
CONTEXT = 32768
OUTPUT_TOKENS = 8
BURST = 64          # bytes; DDR-class burst


def hw(burst=0, kv_bits=4):
    return HardwareConfig(
        array_m=32, array_n=4, FPE_array_size=64,
        act_bits=16, accumulate_bits=32, weight_bits=4, kv_cache_bits=kv_bits,
        AW_mode="OMNI", AA_mode="OMNI", dram_burst_bytes=burst,
    )


def workload(batch=1):
    return WorkloadConfig(batch_size=batch, input_tokens=CONTEXT,
                          output_tokens=OUTPUT_TOKENS, flash_block_size=0)


def run_sel(burst=0, kv_bits=4, **kw):
    model, w = get_model_config(MODEL), workload()
    sim = SelectiveAttnSimulator(hw(burst, kv_bits), **kw)
    r = sim.simulate(model, w)
    d = r.decode.get_total_metrics()
    ttft, tpot = sim.compute_roofline_latency(r, w)
    return {'dram_read': d.dram_read, 'dram_read_eff': d.dram_read_eff,
            'tpot_s': tpot, 'sim': sim}


def run_evict(budget, burst=0, kv_bits=4):
    model, w = get_model_config(MODEL), workload()
    sim = KVBudgetSimulator(hw(burst, kv_bits), kv_budget=budget)
    r = sim.simulate(model, w)
    d = r.decode.get_total_metrics()
    ttft, tpot = sim.compute_roofline_latency(r, w)
    return {'dram_read': d.dram_read, 'dram_read_eff': d.dram_read_eff,
            'tpot_s': tpot}


def preflight():
    """Assertions that must hold before any number below is worth reading."""
    checks = 0

    # 1. No selection parameters => dense baseline, exactly.
    dense = run_sel()
    plain = run_evict(0)
    assert dense['dram_read'] == plain['dram_read'], "dense selective != dense"
    assert abs(dense['tpot_s'] - plain['tpot_s']) < 1e-15, "dense TPOT drift"
    checks += 1

    # 2. Page size = full context, one page selected => still dense.
    full = run_sel(page_size=CONTEXT + OUTPUT_TOKENS, select_pages=1)
    assert full['dram_read'] == dense['dram_read'], "full-page selective != dense"
    checks += 1

    # 3. Selecting every page => dense, whatever the page size.
    allsel = run_sel(page_size=16, select_frac=1.0)
    assert allsel['dram_read'] == dense['dram_read'], "select-all != dense"
    checks += 1

    # 4. With burst off, selection and compacted eviction agree exactly at the
    #    same retained entry count -- the reason this study was deferred.
    ps, k = 16, 64
    s = run_sel(page_size=ps, select_pages=k)
    e = run_evict(ps * k)
    assert s['dram_read'] == e['dram_read'], "flat model should not separate them"
    checks += 1

    # 5. Effective == logical whenever the burst term is off.
    assert dense['dram_read'] == dense['dram_read_eff'], "burst leaked at 0"
    checks += 1

    print(f"Pre-flight: {checks} assertions passed\n")
    return checks


def sweep():
    rows = []
    preflight()

    model_dense = run_sel()
    dense_bytes = model_dense['dram_read']
    dense_tpot = model_dense['tpot_s']
    print(f"Dense decode baseline: {dense_bytes:,} B DRAM read, "
          f"TPOT {dense_tpot * 1e3:.3f} ms\n")

    # ---- A. Selection vs compacted eviction, at equal entries --------------
    print("A. Same retained entries: selection vs compacted eviction\n"
          f"   (context {CONTEXT:,}, page 16, 4-bit KV, {BURST} B burst)\n")
    print(f"  {'pages':>6} {'entries':>8} | {'evict eff':>16} {'select eff':>16} "
          f"{'ratio':>7}")
    for k in (16, 64, 256, 1024):
        ps = 16
        e = run_evict(ps * k, burst=BURST)
        s = run_sel(burst=BURST, page_size=ps, select_pages=k)
        ratio = s['dram_read_eff'] / e['dram_read_eff']
        rows.append({'section': 'A', 'pages': k, 'entries': ps * k,
                     'evict_eff': e['dram_read_eff'],
                     'select_eff': s['dram_read_eff'], 'ratio': ratio})
        print(f"  {k:>6} {ps*k:>8} | {e['dram_read_eff']:>16,} "
              f"{s['dram_read_eff']:>16,} {ratio:>7.3f}x")
    print("\n  Identical.  A 4-bit KV entry is 128 x 4/8 = 64 B, so a page of 16 is\n"
          "  1024 B -- a whole number of 64 B bursts.  Gathering pages costs nothing\n"
          "  extra because every page boundary is already burst-aligned.\n")

    # ---- B. Where granularity does bite ------------------------------------
    print("B. Page size and KV bit-width vs burst alignment\n"
          f"   (entry = head_dim 128 x kv_bits / 8; {BURST} B burst)\n")
    print(f"  {'kv_bits':>8} {'entry B':>8} {'page':>6} {'run B':>8} "
          f"{'charged':>9} {'waste':>7}")
    for kv_bits in (4, 3):
        for ps in (1, 4, 16):
            entry_b = 128 * kv_bits // 8
            run_b = ps * entry_b
            charged = -(-run_b // BURST) * BURST
            rows.append({'section': 'B', 'kv_bits': kv_bits, 'page_size': ps,
                         'run_bytes': run_b, 'charged': charged,
                         'ratio': charged / run_b})
            print(f"  {kv_bits:>8} {entry_b:>8} {ps:>6} {run_b:>8} "
                  f"{charged:>9} {charged / run_b:>6.2f}x")
    print("\n  4-bit KV is burst-aligned at every page size, including page = 1\n"
          "  (token-granular, TidalDecode-style).  3-bit KV is 48 B per entry and\n"
          "  is NOT: a single-entry gather pays 1.33x.  Granularity is a bit-width\n"
          "  property here, not a selection property.\n")

    # ---- C. What selection metadata costs ----------------------------------
    print("C. Selection metadata: Quest-style min/max per page, per token\n"
          f"   (context {CONTEXT:,}, 4-bit KV, {BURST} B burst)\n")
    print(f"  {'page':>6} {'read':>7} | {'no cost':>16} {'with metadata':>16} "
          f"{'overhead':>9} {'TPOT':>10}")
    for ps in (16, 64):
        for frac in (0.5, 0.25, 0.1, 0.03):
            free = run_sel(burst=BURST, page_size=ps, select_frac=frac)
            paid = run_sel(burst=BURST, page_size=ps, select_frac=frac,
                           summary_vectors=2)
            oh = paid['dram_read_eff'] / free['dram_read_eff']
            rows.append({'section': 'C', 'page_size': ps, 'select_frac': frac,
                         'free_eff': free['dram_read_eff'],
                         'paid_eff': paid['dram_read_eff'], 'overhead': oh,
                         'tpot_s': paid['tpot_s']})
            print(f"  {ps:>6} {frac:>6.0%} | {free['dram_read_eff']:>16,} "
                  f"{paid['dram_read_eff']:>16,} {oh:>8.3f}x "
                  f"{paid['tpot_s']*1e3:>8.3f} ms")
    print("\n  Metadata scales with the CONTEXT, not with what was selected, so its\n"
          "  share grows as selection gets more aggressive -- the floor under how\n"
          "  far selection can pay.  A larger page amortises it directly.\n")

    # ---- D. End-to-end speedup vs dense ------------------------------------
    print("D. Decode speedup vs dense, and how much of it is real\n"
          f"   (context {CONTEXT:,}, page 16, 4-bit KV)\n")
    print(f"  {'read':>7} | {'flat model':>11} {'+ burst':>10} "
          f"{'+ metadata':>11}")
    for frac in (0.5, 0.25, 0.1, 0.03):
        flat = run_sel(burst=0, page_size=16, select_frac=frac)
        bur = run_sel(burst=BURST, page_size=16, select_frac=frac)
        met = run_sel(burst=BURST, page_size=16, select_frac=frac,
                      summary_vectors=2)
        rows.append({'section': 'D', 'select_frac': frac,
                     'flat_speedup': dense_tpot / flat['tpot_s'],
                     'burst_speedup': dense_tpot / bur['tpot_s'],
                     'metadata_speedup': dense_tpot / met['tpot_s']})
        print(f"  {frac:>6.0%} | {dense_tpot/flat['tpot_s']:>10.3f}x "
              f"{dense_tpot/bur['tpot_s']:>9.3f}x "
              f"{dense_tpot/met['tpot_s']:>10.3f}x")
    print("\n  Burst granularity costs selection nothing at 4-bit KV.  Metadata is\n"
          "  what bends the curve, and it bends it hardest exactly where the paper\n"
          "  numbers look best.\n")
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
    p.add_argument('--csv', default=os.path.join(_here, 'selective.csv'))
    p.add_argument('--report',
                   default=os.path.join(_here, 'selective_report.md'))
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
        f.write("# Select-without-evict: what selection actually costs\n\n")
        f.write("Generated by `analysis/selective_attn/selective_run.py`.\n"
                "Model: LLaMA-3-8B on Omni-LUT.  See the module docstring in\n"
                "`selective_attn.py` for scope and assumptions.\n\n")
        f.write("```\n")
        f.write(buf.getvalue())
        f.write("```\n")

    print(f"Wrote {len(rows)} rows -> {args.csv}")
    print(f"Wrote report      -> {args.report}")


if __name__ == '__main__':
    main()
