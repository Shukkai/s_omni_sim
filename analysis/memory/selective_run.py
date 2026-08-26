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
import csv
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, '..', '..'))
sys.path.insert(0, os.path.join(_root, 'simulator'))
sys.path.insert(0, os.path.join(_root, 'analysis'))
sys.path.insert(0, os.path.join(_root, 'analysis', 'cycle_breakdown'))
sys.path.insert(0, os.path.join(_root, 'analysis', 'compact_breakdown'))
sys.path.insert(0, _here)

from simulator import HardwareConfig, WorkloadConfig     # noqa: E402
from model_configs import get_model_config               # noqa: E402
from kv_budget import KVBudgetSimulator                  # noqa: E402
from selective_attn import SelectiveAttnSimulator        # noqa: E402
from report import Report                               # noqa: E402

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


def sweep(report_path):
    rows = []
    preflight()

    dense = run_sel()
    dense_bytes, dense_tpot = dense['dram_read'], dense['tpot_s']

    rep = Report(
        report_path,
        "Select-without-evict",
        subtitle="What page selection actually costs, once granularity is charged",
        source="analysis/memory/selective_run.py",
        setup=[f"Model: LLaMA-3-8B on Omni-LUT, 4-bit KV, context {CONTEXT:,}, "
               f"{BURST} B DRAM burst.",
               f"Dense decode baseline: {dense_bytes:,} B DRAM read, "
               f"TPOT {dense_tpot * 1e3:.3f} ms."])

    rep.summary([
        "**Selection and compacted eviction are byte-identical** at equal retained "
        "entries, at every `k` tested — so the flat-bandwidth model was right, and "
        "`kv_budget.py`'s decision to defer this study was correct for this hardware.",
        "The reason: a 4-bit KV entry is `128 x 4/8` = **64 B, exactly one "
        "DDR-class burst**, so a page-gathering reader is burst-aligned at every "
        "page size — down to a single token.",
        "**Granularity is a bit-width property here, not a selection property.** At "
        "3-bit KV an entry is 48 B and a single-entry gather does pay 1.33x.",
        "What selection actually costs is **metadata**, which scales with the "
        "context rather than with what was selected — so its share grows as "
        "selection gets more aggressive.",
    ])

    # ---- A ------------------------------------------------------------------
    rep.section(
        "A. Selection vs compacted eviction, at equal retained entries",
        "Page 16, 4-bit KV. If burst granularity separated the two, it would show "
        "here.")
    trows = []
    for k in (16, 64, 256, 1024):
        ps = 16
        e = run_evict(ps * k, burst=BURST)
        sel = run_sel(burst=BURST, page_size=ps, select_pages=k)
        ratio = sel['dram_read_eff'] / e['dram_read_eff']
        rows.append({'section': 'A', 'pages': k, 'entries': ps * k,
                     'evict_eff': e['dram_read_eff'],
                     'select_eff': sel['dram_read_eff'], 'ratio': ratio})
        trows.append([f"{k:,}", f"{ps*k:,}", f"{e['dram_read_eff']:,}",
                      f"{sel['dram_read_eff']:,}", f"{ratio:.3f}x"])
    rep.table(["pages read", "entries", "eviction (eff)", "selection (eff)",
               "ratio"], trows)
    rep.note(
        "Identical. A page of 16 entries is 1,024 B — a whole number of 64 B "
        "bursts — so every page boundary is already burst-aligned and gathering "
        "costs nothing extra.")

    # ---- B ------------------------------------------------------------------
    rep.section(
        "B. Where granularity *does* bite",
        f"Entry width is `head_dim 128 x kv_bits / 8`, against a {BURST} B burst.")
    trows = []
    for kv_bits in (4, 3):
        for ps in (1, 4, 16):
            entry_b = 128 * kv_bits // 8
            run_b = ps * entry_b
            charged = -(-run_b // BURST) * BURST
            rows.append({'section': 'B', 'kv_bits': kv_bits, 'page_size': ps,
                         'run_bytes': run_b, 'charged': charged,
                         'ratio': charged / run_b})
            trows.append([str(kv_bits), f"{entry_b} B", str(ps), f"{run_b} B",
                          f"{charged} B", f"{charged / run_b:.2f}x"])
    rep.table(["kv_bits", "entry", "page", "run", "charged", "waste"], trows)
    rep.note(
        "4-bit KV is burst-aligned at **every** page size, including page = 1 "
        "(token-granular, TidalDecode-style). 3-bit KV is 48 B per entry and is "
        "not: a single-entry gather pays 1.33x.")

    # ---- C ------------------------------------------------------------------
    rep.section(
        "C. What selection metadata costs",
        "Quest-style per-page min/max, read once per page per token over the "
        "**full** context — not over what was selected.")
    trows = []
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
            trows.append([str(ps), f"{frac:.0%}", f"{free['dram_read_eff']:,}",
                          f"{paid['dram_read_eff']:,}", f"{oh:.3f}x",
                          f"{paid['tpot_s']*1e3:.3f} ms"])
    rep.table(["page", "read", "no metadata", "with metadata", "overhead",
               "TPOT"], trows)
    rep.note(
        "Metadata scales with the **context**, not with what was selected, so its "
        "share grows as selection gets more aggressive — a floor under how far "
        "selection can pay. A larger page amortises it directly.")

    # ---- D ------------------------------------------------------------------
    rep.section(
        "D. How much of the paper speedup survives",
        "Page 16, 4-bit KV. Decode TPOT speedup vs dense.")
    trows = []
    for frac in (0.5, 0.25, 0.1, 0.03):
        flat = run_sel(burst=0, page_size=16, select_frac=frac)
        bur = run_sel(burst=BURST, page_size=16, select_frac=frac)
        met = run_sel(burst=BURST, page_size=16, select_frac=frac,
                      summary_vectors=2)
        rows.append({'section': 'D', 'select_frac': frac,
                     'flat_speedup': dense_tpot / flat['tpot_s'],
                     'burst_speedup': dense_tpot / bur['tpot_s'],
                     'metadata_speedup': dense_tpot / met['tpot_s']})
        trows.append([f"{frac:.0%}", f"{dense_tpot/flat['tpot_s']:.3f}x",
                      f"{dense_tpot/bur['tpot_s']:.3f}x",
                      f"{dense_tpot/met['tpot_s']:.3f}x"])
    rep.table(["read", "flat model", "+ burst granularity", "+ metadata"],
              trows)
    rep.note(
        "Burst granularity costs selection **nothing** at 4-bit KV. Metadata is "
        "what bends the curve, and it bends it hardest exactly where the paper "
        "numbers look best.")

    rep.save()
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv', default=os.path.join(_here, 'selective.csv'))
    p.add_argument('--report',
                   default=os.path.join(_here, 'selective_report.md'))
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
