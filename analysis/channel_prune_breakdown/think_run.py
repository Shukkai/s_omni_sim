"""
Channel-pruning breakdown: what ThinK is worth on Omni-LUT.

ThinK prunes KV *channels* (`head_dim`), where the eviction methods in
`compact_breakdown/` prune KV *entries* (`kv_len`).  On the LUT dataflows
those two axes are not symmetric, and this study measures the asymmetry.

Everything is reported per phase, as study.md's tables are, because the two
phases run different dataflows and the same pruning ratio lands on different
geometry in each:

    prefill  LUT_WS     retained channels are columns of one 128-wide tile
    decode   LUT_OS_V   retained channels are lanes out of the full 4096

  A. CYCLE NULL -- sweep retained channels and read the cycle count.
     `attn_v` is flat in BOTH phases, because `head_dim` is its OUTPUT
     dimension N and neither round formula has an N term below one tile.
     `qk` does shrink, because there `head_dim` is the REDUCTION dimension K.
     The direction that saves cycles is the stage that costs nothing.
  B. UTILIZATION -- the same pruning expressed as idle array columns, which is
     where the cost actually lands, against each phase's own denominator.
  C. DRAM -- the one axis where channel pruning pays, fed through the batch x
     context regime map so the payoff is bounded honestly.

Usage:
    python think_run.py
    python think_run.py --contexts 32768 --batches 1,8,32
    python think_run.py --channels 128,90,77,64,38 --modes V,K,KV
    python think_run.py --no-prune-prefill    # strict ThinK: decode only
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

from simulator import (                                          # noqa: E402
    ComputeMode, HardwareConfig, OperationType, Simulator, WorkloadConfig,
)
from model_configs import get_model_config, list_models          # noqa: E402
from cycle_units import (                                        # noqa: E402
    UnitAwareSimulator, compute_stage_cycle_breakdown,
)
from think_prune import (                                        # noqa: E402
    ThinKSimulator, kv_cache_dram_bytes, occupancy, osv_per_round,
    osv_rounds, stage_utilization, ws_per_round,
)

ATTN_STAGES = {'qk_matmul', 'attn_v_matmul', 'softmax'}
PHASES = (('prefill', 'LUT_WS'), ('decode', 'LUT_OS_V'))


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


def resolve_retained(mode: str, d_ret: int, head_dim: int) -> tuple:
    """(d_k_ret, d_v_ret) for a pruning mode: 'K', 'V' or 'KV'."""
    return (d_ret if 'K' in mode else head_dim,
            d_ret if 'V' in mode else head_dim)


def run_point(task):
    """Simulate one (batch, context, mode, retained-channels) point."""
    args, batch, context, mode, d_ret = task
    model = get_model_config(args.model)
    hw = build_hw(args)
    d_k_ret, d_v_ret = resolve_retained(mode, d_ret, model.head_dim)

    workload = WorkloadConfig(
        batch_size=batch, input_tokens=context,
        output_tokens=args.output_tokens, flash_block_size=0,
    )
    sim = ThinKSimulator(hw, d_k_ret=d_k_ret, d_v_ret=d_v_ret,
                         prune_prefill=args.prune_prefill, model_bqu=False)
    results = sim.simulate(model, workload)
    bd = compute_stage_cycle_breakdown(sim, results, workload)

    row = {
        'batch': batch, 'context': context, 'mode': mode,
        'd_ret': d_ret, 'd_k_ret': d_k_ret, 'd_v_ret': d_v_ret,
        'retained_frac': d_ret / model.head_dim,
        'prune_ratio': 1.0 - d_ret / model.head_dim,
    }

    stage_sets = {'prefill': bd['prefill'], 'decode': bd['decode_per_token']}
    phase_metrics = {'prefill': results.prefill, 'decode': results.decode}

    for phase, dataflow in PHASES:
        stages = stage_sets[phase]
        qk = stages.get('qk_matmul', {})
        av = stages.get('attn_v_matmul', {})
        occ = occupancy(hw, d_v_ret, dataflow)
        total_dram = sum(r['dram_bytes'] for r in stages.values())
        attn_dram = sum(r['dram_bytes'] for s, r in stages.items()
                        if s in ATTN_STAGES)

        row.update({
            # A. cycles
            f'{phase}_cycles': sum(r['cycles'] for r in stages.values()),
            f'{phase}_qk_cycles': qk.get('cycles', 0.0),
            f'{phase}_attn_v_cycles': av.get('cycles', 0.0),

            # B. utilization
            f'{phase}_attn_v_utilization': stage_utilization(
                phase_metrics[phase], OperationType.ATTN_V_MATMUL),
            f'{phase}_attn_v_occupied_lanes': occ['occupied_lanes'],
            f'{phase}_attn_v_occupied_frac': occ['occupied_frac'],
            f'{phase}_attn_v_in_round_frac': occ['in_round_frac'],
            f'{phase}_array_lanes': occ['lanes'],
            f'{phase}_dataflow': dataflow,

            # C. DRAM / latency
            f'{phase}_eff_time': sum(r['eff_time'] for r in stages.values()),
            f'{phase}_total_dram': total_dram,
            f'{phase}_attn_dram': attn_dram,
            f'{phase}_kv_dram_share': attn_dram / total_dram if total_dram else 0.0,
            f'{phase}_attn_v_bound': av.get('bound', ''),
            f'{phase}_qk_bound': qk.get('bound', ''),
        })

    # Analytic KV bytes, independent of the stage totals (which also carry the
    # attention-score spill between QK and Attn.V).  Cross-checked in main().
    kv = kv_cache_dram_bytes(model, hw, batch, context, d_k_ret, d_v_ret)
    writes = results.get_kv_cache_writes()
    row.update({
        'kv_bytes_k': kv['k_bytes'],
        'kv_bytes_v': kv['v_bytes'],
        'kv_bytes_total': kv['total_bytes'],
        'prefill_kv_writeback': (writes['prefill_k_bytes']
                                 + writes['prefill_v_bytes']),
        'decode_qk_per_round': osv_per_round(hw, d_k_ret),
        'decode_attn_v_per_round': osv_per_round(hw, context),
        'decode_attn_v_rounds': osv_rounds(hw, d_v_ret),
        'prefill_attn_v_per_round': ws_per_round(hw, context),
    })
    return row


# ---- Reporting --------------------------------------------------------------

def fmt_bytes(b):
    for unit, div in (('GB', 1e9), ('MB', 1e6), ('KB', 1e3)):
        if b >= div:
            return f"{b/div:.2f} {unit}"
    return f"{b:.0f} B"


def report_cycle_null(rows, args, head_dim):
    """A. Cycles vs retained channels, per phase."""
    print("\n" + "=" * 92)
    print("  A. CYCLE NULL  (cycles vs retained channels, by phase)")
    print("=" * 92)
    print("\n  attn_v: head_dim is the OUTPUT dim N.  Neither round formula")
    print("          carries an N term, and one tile covers N <= "
          f"{head_dim} in both phases:")
    print("            prefill  LUT_WS    per_round = M + array_n + array_m + 1 + 2")
    print("            decode   LUT_OS_V  per_round = 3 + ceil(kv_len/4) + 1"
          " + array_n + 2")
    print("  qk:     head_dim is the REDUCTION dim K.  In LUT_OS_V that enters")
    print("          per_round as k_eff = ceil(K/4), so decode qk does shrink.")
    print("          In LUT_WS it does not: the reduction is tiled as")
    print("          k_tiles = ceil(ceil(K/MU)/array_m), one tile per "
          f"{Simulator.MU * args.array_m} elements,")
    print(f"          and head_dim = {head_dim} is exactly one tile.  So PREFILL IS")
    print("          A NULL ON BOTH AXES -- pruning cannot cross a tile boundary")
    print("          it is already sitting on.")
    if not args.prune_prefill:
        print("\n  NOTE: prefill left dense (--no-prune-prefill), so its columns are")
        print("        flat by construction rather than by dataflow.")
    else:
        print("\n  NOTE: prefill pruning is hypothetical -- ThinK's channel selection")
        print("        is query-driven and only available after the prompt.  It is")
        print("        modelled to show the null is a dataflow property, not a phase one.")
    print()

    b0 = min(r['batch'] for r in rows)
    for mode in sorted({r['mode'] for r in rows}):
        for ctx in sorted({r['context'] for r in rows}):
            sel = sorted((r for r in rows if r['batch'] == b0
                          and r['context'] == ctx and r['mode'] == mode),
                         key=lambda r: -r['d_ret'])
            if len(sel) < 2:
                continue
            base = sel[0]
            print(f"  ThinK-{mode}, context {ctx}, batch {b0}:")
            print(f"    {'':14} |{'PREFILL (LUT_WS)':^36}|{'DECODE/tok (LUT_OS_V)':^38}")
            print(f"    {'kept':>6} {'lambda':>6} |{'qk':>12} {'attn_v':>14} "
                  f"{'vs dense':>7} |{'qk':>12} {'attn_v':>14} {'vs dense':>9}")
            for r in sel:
                print(f"    {r['d_ret']:>6} {r['prune_ratio']*100:>5.0f}% |"
                      f"{r['prefill_qk_cycles']:>12,.0f} "
                      f"{r['prefill_attn_v_cycles']:>14,.0f} "
                      f"{base['prefill_cycles']/r['prefill_cycles']:>6.3f}x |"
                      f"{r['decode_qk_cycles']:>12,.0f} "
                      f"{r['decode_attn_v_cycles']:>14,.0f} "
                      f"{base['decode_cycles']/r['decode_cycles']:>8.3f}x")
            print()


def report_utilization(rows, args, head_dim):
    """B. Where the pruning cost lands: idle array columns, per phase."""
    print("=" * 92)
    print("  B. ARRAY UTILIZATION  (attn_v, by phase)")
    print("=" * 92)
    b0 = min(r['batch'] for r in rows)
    ctx0 = max(r['context'] for r in rows)
    sel = sorted((r for r in rows if r['batch'] == b0 and r['context'] == ctx0
                  and 'V' in r['mode']), key=lambda r: -r['d_v_ret'])
    seen, uniq = set(), []
    for r in sel:
        if r['d_v_ret'] not in seen:
            seen.add(r['d_v_ret'])
            uniq.append(r)
    if not uniq:
        return
    ws_lanes = uniq[0]['prefill_array_lanes']
    osv_lanes = uniq[0]['decode_array_lanes']

    print(f"\n  prefill  LUT_WS    a round is one {ws_lanes}-wide output tile, and")
    print(f"                     head_dim = {head_dim} fills it exactly.  Pruning")
    print("                     idles columns in direct proportion.")
    print(f"  decode   LUT_OS_V  a round spans array_m x array_n x NUM_RAC = "
          f"{osv_lanes}")
    print(f"                     lanes, of which head_dim = {head_dim} was the most")
    print("                     ever used.  Pruning widens an already large gap.\n")
    print("    occupied = lanes physically driven (RAC granularity = array_n)")
    print("    useful   = useful MACs / issued lane-cycles, as the simulator")
    print("               reports it in OperationMetrics.utilization\n")

    print(f"  context {ctx0}, batch {b0}:")
    print(f"    {'':14} |{'PREFILL (LUT_WS)':^34}|{'DECODE (LUT_OS_V)':^34}")
    print(f"    {'kept':>6} {'lambda':>6} |{'occupied':>12} {'occ %':>8} "
          f"{'useful %':>9} |{'occupied':>12} {'occ %':>8} {'useful %':>9}")
    for r in uniq:
        print(f"    {r['d_v_ret']:>6} {(1-r['d_v_ret']/head_dim)*100:>5.0f}% |"
              f"{r['prefill_attn_v_occupied_lanes']:>6}/{ws_lanes:<5} "
              f"{r['prefill_attn_v_occupied_frac']*100:>7.2f}% "
              f"{r['prefill_attn_v_utilization']*100:>8.2f}% |"
              f"{r['decode_attn_v_occupied_lanes']:>6}/{osv_lanes:<5} "
              f"{r['decode_attn_v_occupied_frac']*100:>7.2f}% "
              f"{r['decode_attn_v_utilization']*100:>8.2f}%")
    print()


def report_dram(rows, args, head_dim):
    """C. DRAM saving, through the batch x context regime map, per phase."""
    print("=" * 92)
    print(f"  C. DRAM  (at {args.headline_channels} retained channels)")
    print("=" * 92)
    print("\n  Channel pruning only pays where KV is a meaningful share of DRAM")
    print("  traffic AND the stage it lands on is memory-bound.  Those are")
    print("  different conditions, and the K and V paths sit on opposite sides")
    print("  of the second one:\n")

    idx = {(r['batch'], r['context'], r['mode'], r['d_ret']): r for r in rows}
    contexts = sorted({r['context'] for r in rows})
    batches = sorted({r['batch'] for r in rows})

    ref = idx.get((max(batches), max(contexts), rows[0]['mode'], head_dim))
    if ref:
        print(f"    decode qk_matmul     is {ref['decode_qk_bound']}-bound"
              f"  -> pruning Key channels cuts real latency")
        print(f"    decode attn_v_matmul is {ref['decode_attn_v_bound']}-bound"
              f" -> pruning Value channels cuts bytes")
        print(f"    {'':52}that were already hidden under compute\n")

    for mode in sorted({r['mode'] for r in rows}):
        print(f"  ThinK-{mode}:")
        print(f"    {'':17} |{'KV bytes / decode token':^38}|"
              f"{'DECODE/tok time':^26}|{'PREFILL':^28}")
        print(f"    {'batch':>6} {'ctx':>9} |{'KV %DRAM':>9} {'dense':>11} "
              f"{'pruned':>11} {'saved':>5} |{'dense':>10} {'pruned':>10} "
              f"{'up':>4} |{'writeback':>11} {'dense':>9} {'up':>5}")
        for b in batches:
            for c in contexts:
                d = idx.get((b, c, mode, head_dim))
                k = idx.get((b, c, mode, args.headline_channels))
                if not d or not k:
                    continue
                sp = d['decode_eff_time'] / k['decode_eff_time']
                psp = d['prefill_eff_time'] / k['prefill_eff_time']
                saved = 1.0 - k['kv_bytes_total'] / d['kv_bytes_total']
                print(f"    {b:>6} {c:>9} |{d['decode_kv_dram_share']*100:>8.1f}% "
                      f"{fmt_bytes(d['kv_bytes_total']):>11} "
                      f"{fmt_bytes(k['kv_bytes_total']):>11} "
                      f"{saved*100:>4.0f}% |"
                      f"{d['decode_eff_time']*1e3:>9.2f}ms "
                      f"{k['decode_eff_time']*1e3:>9.2f}ms {sp:>4.2f}x |"
                      f"{fmt_bytes(d['prefill_kv_writeback']):>11} "
                      f"{d['prefill_eff_time']:>8.2f}s {psp:>4.2f}x")
        print()


# ---- CLI --------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Channel-pruning (ThinK) breakdown on Omni-LUT.")
    p.add_argument('--model', default='LLaMA-3-8B',
                   help=f"available: {', '.join(list_models())}")
    p.add_argument('--contexts', default='2048,8192,32768')
    p.add_argument('--batches', default='1,8,32')
    p.add_argument('--channels', default='128,90,77,64,38',
                   help="retained channels per head (dense head_dim is added)")
    p.add_argument('--modes', default='V,K,KV',
                   help="which cache to prune: K, V or KV")
    p.add_argument('--headline-channels', type=int, default=77,
                   help="retained channels for the headline table (ThinK lambda=0.4)")
    p.add_argument('--output-tokens', type=int, default=256,
                   help="decode steps to simulate (TPOT is averaged); 256 "
                        "matches study.md, so the dense baseline reproduces "
                        "its Sec. 3 roofline column exactly")
    p.add_argument('--no-prune-prefill', dest='prune_prefill',
                   action='store_false',
                   help="strict ThinK: leave prefill dense (selection is only "
                        "available after the prompt)")
    p.set_defaults(prune_prefill=True)

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
    p.add_argument('--out-prefix', default='channel_prune_breakdown')
    return p.parse_args()


def check_cycle_null_formula(hw, head_dim):
    """The null must hold in the cycle model itself, not just in one figure.

    Asks `Simulator._calculate_cycles` directly for both dataflows, so a future
    edit that gives either an N-dependent term fails here before any number is
    published.
    """
    sim = UnitAwareSimulator(hw, model_bqu=False)
    for mode, M, K in (('LUT_OS_V', 1, 32768), ('LUT_WS', 2048, 2048)):
        cyc = {sim._calculate_cycles(M, K, d, hw.kv_cache_bits,
                                     ComputeMode.AA, mode, 1)
               for d in range(1, head_dim + 1)}
        assert len(cyc) == 1, \
            f"attn_v cycles vary with N under {mode}: {sorted(cyc)[:4]} ..."
    assert osv_rounds(hw, head_dim) == 1
    print(f"  attn_v cycles are N-independent for N <= {head_dim} "
          f"in LUT_WS and LUT_OS_V ✓")


def check_dense_baseline(args, model, hw):
    """A dense ThinKSimulator must reproduce UnitAwareSimulator exactly."""
    workload = WorkloadConfig(
        batch_size=1, input_tokens=max(int(x) for x in args.contexts.split(',')),
        output_tokens=args.output_tokens, flash_block_size=0)

    ref = UnitAwareSimulator(hw, model_bqu=False).simulate(model, workload)
    got = ThinKSimulator(hw, d_k_ret=model.head_dim, d_v_ret=model.head_dim,
                         prune_prefill=True,
                         model_bqu=False).simulate(model, workload)

    for phase in ('prefill', 'decode'):
        a = getattr(ref, phase).get_total_metrics()
        b = getattr(got, phase).get_total_metrics()
        assert a.cycles == b.cycles, f"{phase} cycles differ"
        assert a.dram_read == b.dram_read and a.dram_write == b.dram_write, \
            f"{phase} DRAM differs"
    print("  dense pruning point matches UnitAwareSimulator in both phases ✓")


def check_dram_scaling(rows, head_dim):
    """V-cache bytes must scale exactly linearly with retained channels."""
    for r in rows:
        if 'V' not in r['mode'] or r['d_ret'] == head_dim:
            continue
        dense = next((d for d in rows
                      if d['batch'] == r['batch'] and d['context'] == r['context']
                      and d['mode'] == r['mode'] and d['d_ret'] == head_dim), None)
        if not dense:
            continue
        expected = dense['kv_bytes_v'] * r['d_ret'] / head_dim
        assert abs(r['kv_bytes_v'] - expected) < 1.0, \
            f"V bytes not linear in retained channels at {r['d_ret']}"
    print("  V-cache DRAM scales linearly with retained channels ✓")


def main():
    args = parse_args()
    model = get_model_config(args.model)
    hw = build_hw(args)
    head_dim = model.head_dim

    contexts = [int(x) for x in args.contexts.split(',') if x.strip()]
    batches = [int(x) for x in args.batches.split(',') if x.strip()]
    modes = [m.strip().upper() for m in args.modes.split(',') if m.strip()]
    channels = sorted({int(x) for x in args.channels.split(',') if x.strip()}
                      | {head_dim, args.headline_channels}, reverse=True)
    channels = [c for c in channels if 1 <= c <= head_dim]

    print(f"Model:    {args.model}  (head_dim={head_dim}, "
          f"GQA {model.num_heads}/{model.num_kv_heads})")
    print(f"Hardware: AW={args.aw_mode} AA={args.aa_mode} "
          f"{args.array_m}x{args.array_n} "
          f"W{args.weight_bits}A{args.act_bits}KV{args.kv_bits} "
          f"{args.freq_mhz} MHz {args.dram_bw} GB/s")
    print(f"Sweep:    contexts={contexts} batches={batches} "
          f"modes={modes} channels={channels}")
    print(f"Prefill:  {'pruned (hypothetical)' if args.prune_prefill else 'dense'}")

    print("\nPre-flight checks:")
    check_cycle_null_formula(hw, head_dim)
    check_dense_baseline(args, model, hw)

    tasks = [(args, b, c, m, d)
             for b in batches for c in contexts for m in modes for d in channels]
    print(f"\nRunning {len(tasks)} simulations ...")

    n_workers = max(1, min(len(tasks), cpu_count() - 1))
    with Pool(processes=n_workers) as pool:
        rows = list(pool.imap_unordered(run_point, tasks))

    check_dram_scaling(rows, head_dim)

    report_cycle_null(rows, args, head_dim)
    report_utilization(rows, args, head_dim)
    report_dram(rows, args, head_dim)

    rows.sort(key=lambda r: (r['mode'], r['batch'], r['context'], -r['d_ret']))
    out = os.path.join(_here, args.out_prefix)
    with open(f"{out}.json", 'w') as f:
        json.dump({'config': vars(args), 'head_dim': head_dim, 'rows': rows},
                  f, indent=2)
    with open(f"{out}.csv", 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {args.out_prefix}.json and {args.out_prefix}.csv")


if __name__ == "__main__":
    main()
