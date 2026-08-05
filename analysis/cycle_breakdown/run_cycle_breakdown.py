"""
Stage-by-stage cycle cost of Omni-LUT.

Runs the simulator for a single hardware configuration (default: Omni-LUT-KV4,
32x4 LUT array, W4A16) over one or more context lengths and reports the cycle
cost of every pipeline stage -- q/k/v_proj, qk_matmul, softmax, attn_v, o_proj,
fc1, fc2, rmsnorm, rope, silu, residual, ... -- split into prefill and
decode (per generated token).

Outputs <prefix>.json, <prefix>.csv and a text table on stdout.

Usage:
    python run_cycle_breakdown.py
    python run_cycle_breakdown.py --model LLaMA-3-8B --input-tokens 2048,32768
    python run_cycle_breakdown.py --aw-mode LUT_WS --aa-mode FPE_OS --out-prefix figlut
"""

import argparse
import csv
import json
import os
import sys
from multiprocessing import Pool, cpu_count

# Add repo root and simulator directory to path
_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, _root_dir)
sys.path.insert(0, os.path.join(_root_dir, 'simulator'))

from simulator import Simulator, WorkloadConfig, HardwareConfig
from model_configs import get_model_config, list_models

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cycle_units import (
    UnitAwareSimulator,
    compute_stage_cycle_breakdown,
    compute_unit_cycle_breakdown,
)


# ---- Stage ordering ---------------------------------------------------------
# Pipeline order, used for both the tables and the stacked-bar segments.
STAGE_ORDER = [
    'embedding',
    'rmsnorm_pre_attn',
    'q_proj', 'k_proj', 'v_proj',
    'rope',
    'attn_scale',
    'qk_matmul', 'softmax', 'attn_v_matmul', 'flash_attn',
    'o_proj',
    'residual_attn',
    'rmsnorm_pre_ffn',
    'gate', 'fc1', 'silu', 'gated_mul', 'fc2',
    'residual_ffn',
    'rmsnorm_final',
    'lm_head_softmax',
]


def stage_sort_key(stage: str) -> int:
    return STAGE_ORDER.index(stage) if stage in STAGE_ORDER else len(STAGE_ORDER)


# ---- Hardware unit ordering (Fig. 4 datapath order) -------------------------
UNIT_ORDER = [
    'input_load',
    'lgu',
    'pe_array_compute', 'pe_array_fill_drain',
    'fpe_array_compute', 'fpe_array_fill_drain', 'rescale',
    'accumulator',
    'vpu',
    'bqu_tse', 'bqu_bea',
]

UNIT_LABEL = {
    'input_load': 'Operand issue',
    'lgu': 'LGU (table gen)',
    'pe_array_compute': 'PE array (compute)',
    'pe_array_fill_drain': 'PE array (fill/drain)',
    'fpe_array_compute': 'FPE array (compute)',
    'fpe_array_fill_drain': 'FPE array (fill/drain)',
    'rescale': 'Rescale',
    'accumulator': 'Accumulator',
    'vpu': 'VPU',
    'bqu_tse': 'BQU-TSE (overlapped)',
    'bqu_bea': 'BQU-BEA (overlapped)',
}


def unit_sort_key(unit: str) -> int:
    return UNIT_ORDER.index(unit) if unit in UNIT_ORDER else len(UNIT_ORDER)


# ---- Simulation -------------------------------------------------------------

def build_hw(args) -> HardwareConfig:
    return HardwareConfig(
        array_m=args.array_m,
        array_n=args.array_n,
        replication=args.replication,
        FPE_array_size=args.fpe_size,
        act_bits=args.act_bits,
        accumulate_bits=32,
        weight_bits=args.weight_bits,
        kv_cache_bits=args.kv_bits,
        AW_mode=args.aw_mode,
        AA_mode=args.aa_mode,
        freq_mhz=args.freq_mhz,
        dram_bandwidth_gbps=args.dram_bw,
    )


def run_single(task):
    """Worker: simulate one context length and return its stage breakdown."""
    args, input_tokens = task
    model = get_model_config(args.model)
    hw = build_hw(args)
    workload = WorkloadConfig(
        batch_size=args.batch_size,
        input_tokens=input_tokens,
        output_tokens=args.output_tokens,
        flash_block_size=args.flash_block,
    )

    sim = UnitAwareSimulator(hw, model_bqu=not args.no_bqu,
                             bqu_width=args.bqu_width)
    results = sim.simulate(model, workload)
    breakdown = compute_stage_cycle_breakdown(sim, results, workload)
    units = compute_unit_cycle_breakdown(sim, results, workload)

    # --- Unit cycles must reconcile with the stage view (BQU is overlapped) ---
    for phase_name in ('prefill', 'decode'):
        stage_cycles = sum(r['cycles'] for r in breakdown[phase_name].values())
        assert units['serial_cycles'][phase_name] == stage_cycles, (
            f"{phase_name}: unit serial cycles "
            f"{units['serial_cycles'][phase_name]} != stage cycles {stage_cycles}")

    # --- The subclass must not perturb the original simulator's output ---
    ref_results = Simulator(hw).simulate(model, workload)
    assert (results.get_total_metrics().cycles
            == ref_results.get_total_metrics().cycles), \
        "UnitAwareSimulator changed the cycle total vs the stock Simulator"

    # --- Conservation checks: no stage may be silently dropped ---
    for phase_name, phase in (('prefill', results.prefill), ('decode', results.decode)):
        stages = breakdown[phase_name]
        gemm_cycles = sum(r['cycles'] for r in stages.values()
                          if r['category'] in ('AA', 'AW'))
        non_gemm_cycles = sum(r['cycles'] for r in stages.values()
                              if r['category'] == 'non_gemm')
        assert gemm_cycles == phase.get_total_metrics().cycles, (
            f"{phase_name}: GEMM stage cycles {gemm_cycles} != "
            f"phase total {phase.get_total_metrics().cycles}")
        assert non_gemm_cycles == sum(m.cycles for m in phase.non_gemm_ops), (
            f"{phase_name}: non-GEMM stage cycles mismatch")

    # --- Cross-check the roofline times against the existing TTFT/TPOT path ---
    # compute_roofline_latency covers GEMM only; the *_breakdown variant adds
    # non-GEMM, so it is the one that matches the full stage sum.
    ref = sim.compute_roofline_latency_breakdown(results, workload)
    ttft, tpot = ref['ttft_total'], ref['tpot_total']
    for label, key, stages in (
        ('TTFT', 'ttft_total', breakdown['prefill']),
        ('TPOT', 'tpot_total', breakdown['decode_per_token']),
    ):
        stage_sum = sum(r['eff_time'] for r in stages.values())
        assert abs(stage_sum - ref[key]) <= 1e-9 * max(1.0, abs(ref[key])), \
            f"{label} mismatch: stages {stage_sum} vs roofline {ref[key]}"

    return input_tokens, breakdown, units, ttft, tpot


# ---- Reporting --------------------------------------------------------------

def print_table(title, stages, freq_hz):
    """Print one phase's stage table, sorted by cycles descending."""
    total_cycles = sum(r['cycles'] for r in stages.values())
    total_eff = sum(r['eff_time'] for r in stages.values())

    print(f"\n  {title}")
    print("  " + "-" * 94)
    print(f"  {'Stage':<20} {'Cat':<9} {'Execs':>8} {'Cycles':>16} {'%':>7} "
          f"{'Time':>12} {'Eff.Time':>12} {'Bound':>8}")
    print("  " + "-" * 94)

    for rec in sorted(stages.values(), key=lambda r: -r['cycles']):
        pct = 100.0 * rec['cycles'] / total_cycles if total_cycles else 0.0
        print(f"  {rec['stage']:<20} {rec['category']:<9} {rec['execs']:>8,.0f} "
              f"{rec['cycles']:>16,.0f} {pct:>6.2f}% "
              f"{Simulator._fmt_time(rec['compute_time']):>12} "
              f"{Simulator._fmt_time(rec['eff_time']):>12} {rec['bound']:>8}")

    print("  " + "-" * 94)
    print(f"  {'TOTAL':<20} {'':<9} {'':>8} {total_cycles:>16,.0f} {100.0:>6.2f}% "
          f"{Simulator._fmt_time(total_cycles / freq_hz):>12} "
          f"{Simulator._fmt_time(total_eff):>12}")


def print_unit_table(title, units, serial_cycles, freq_hz):
    """Print one phase's per-hardware-unit table, in datapath order."""
    print(f"\n  {title}")
    print("  " + "-" * 78)
    print(f"  {'Hardware unit':<26} {'Cycles':>18} {'% of serial':>12} {'Time':>14}")
    print("  " + "-" * 78)

    for unit in sorted(units, key=unit_sort_key):
        rec = units[unit]
        label = UNIT_LABEL.get(unit, unit)
        pct = f"{rec['pct_of_serial']:>11.2f}%" if not rec['overlapped'] else \
              f"{'(overlap)':>12}"
        print(f"  {label:<26} {rec['cycles']:>18,.0f} {pct} "
              f"{Simulator._fmt_time(rec['time']):>14}")

    print("  " + "-" * 78)
    print(f"  {'SERIAL TOTAL':<26} {serial_cycles:>18,.0f} {100.0:>11.2f}% "
          f"{Simulator._fmt_time(serial_cycles / freq_hz):>14}")


def write_unit_csv(path, all_units):
    """Flatten every (phase, input_tokens, unit) record into one CSV."""
    fields = ['phase', 'input_tokens', 'unit', 'cycles', 'pct_of_serial',
              'time_s', 'overlapped']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for inp in sorted(all_units):
            u = all_units[inp]
            for phase in ('prefill', 'decode', 'decode_per_token'):
                for unit in sorted(u[phase], key=unit_sort_key):
                    r = u[phase][unit]
                    w.writerow({
                        'phase': phase,
                        'input_tokens': inp,
                        'unit': unit,
                        'cycles': r['cycles'],
                        'pct_of_serial': r['pct_of_serial'],
                        'time_s': r['time'],
                        'overlapped': r['overlapped'],
                    })


def write_csv(path, all_results):
    """Flatten every (phase, input_tokens, stage) record into one CSV."""
    fields = ['phase', 'input_tokens', 'stage', 'category', 'execs', 'cycles',
              'pct_of_phase', 'dram_bytes', 'compute_time_s', 'mem_time_s',
              'eff_time_s', 'bound']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for inp in sorted(all_results):
            bd = all_results[inp]
            for phase in ('prefill', 'decode', 'decode_per_token'):
                stages = bd[phase]
                total = sum(r['cycles'] for r in stages.values())
                for stage in sorted(stages, key=stage_sort_key):
                    r = stages[stage]
                    w.writerow({
                        'phase': phase,
                        'input_tokens': inp,
                        'stage': stage,
                        'category': r['category'],
                        'execs': r['execs'],
                        'cycles': r['cycles'],
                        'pct_of_phase': (100.0 * r['cycles'] / total) if total else 0.0,
                        'dram_bytes': r['dram_bytes'],
                        'compute_time_s': r['compute_time'],
                        'mem_time_s': r['mem_time'],
                        'eff_time_s': r['eff_time'],
                        'bound': r['bound'],
                    })


# ---- CLI --------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage-by-stage cycle cost breakdown for Omni-LUT.")
    p.add_argument('--model', default='LLaMA-3-8B',
                   help=f"model name (available: {', '.join(list_models())})")
    p.add_argument('--input-tokens', default='2048,8192,32768',
                   help="comma-separated prefill lengths")
    p.add_argument('--output-tokens', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--flash-block', type=int, default=0,
                   help="0 = standard attention (qk_matmul / attn_v_matmul are "
                        "separate stages); >0 = FlashAttention tile size, which "
                        "fuses them into a single flash_attn stage")

    p.add_argument('--array-m', type=int, default=32)
    p.add_argument('--array-n', type=int, default=4)
    p.add_argument('--replication', type=int, default=1)
    p.add_argument('--fpe-size', type=int, default=64)

    p.add_argument('--act-bits', type=int, default=16)
    p.add_argument('--weight-bits', type=int, default=4)
    p.add_argument('--kv-bits', type=int, default=4)

    p.add_argument('--aw-mode', default='OMNI',
                   help="VPU, FPE_OS, LUT_OS, LUT_OS_V, LUT_WS, OMNI, TENDER")
    p.add_argument('--aa-mode', default='OMNI')

    p.add_argument('--freq-mhz', type=int, default=500)
    p.add_argument('--dram-bw', type=float, default=51.2)

    p.add_argument('--no-bqu', action='store_true',
                   help="skip BQU (online KV quantization) cycle modelling")
    p.add_argument('--bqu-width', type=int, default=128,
                   help="BCQ Encoder Array throughput, elements per cycle")

    p.add_argument('--out-prefix', default='cycle_breakdown')
    return p.parse_args()


def main():
    args = parse_args()
    input_tokens = [int(x) for x in args.input_tokens.split(',') if x.strip()]

    print(f"Model:    {args.model}")
    print(f"Hardware: AW={args.aw_mode} AA={args.aa_mode}  "
          f"array={args.array_m}x{args.array_n}  "
          f"W{args.weight_bits}A{args.act_bits}KV{args.kv_bits}  "
          f"{args.freq_mhz} MHz  {args.dram_bw} GB/s")
    print(f"Workload: input={input_tokens}  output={args.output_tokens}  "
          f"batch={args.batch_size}  flash_block={args.flash_block}")
    print(f"\nRunning {len(input_tokens)} simulation(s) ...")

    tasks = [(args, inp) for inp in input_tokens]
    n_workers = max(1, min(len(tasks), cpu_count() - 1))

    all_results = {}
    all_units = {}
    latency = {}

    def _store(inp, bd, units, ttft, tpot):
        all_results[inp] = bd
        all_units[inp] = units
        latency[inp] = (ttft, tpot)
        print(f"  input={inp:<6} done   TTFT={ttft*1e3:.2f} ms  "
              f"TPOT={tpot*1e3:.3f} ms")

    if n_workers > 1:
        with Pool(processes=n_workers) as pool:
            for res in pool.imap_unordered(run_single, tasks):
                _store(*res)
    else:
        for task in tasks:
            _store(*run_single(task))

    # ---- Text tables ----
    for inp in sorted(all_results):
        bd, un = all_results[inp], all_units[inp]
        print(f"\n{'=' * 98}")
        print(f"  {args.model}  |  input={inp}  output={args.output_tokens}  "
              f"|  AW={args.aw_mode} AA={args.aa_mode}")
        print('=' * 98)

        print("\n  --- BY PIPELINE STAGE " + "-" * 74)
        print_table("(a) PREFILL", bd['prefill'], bd['freq_hz'])
        print_table(f"(b) DECODE (per token, {bd['num_decode_steps']} steps)",
                    bd['decode_per_token'], bd['freq_hz'])

        print("\n  --- BY HARDWARE UNIT " + "-" * 75)
        print_unit_table("(a) PREFILL", un['prefill'],
                         un['serial_cycles']['prefill'], un['freq_hz'])
        print_unit_table(f"(b) DECODE (per token, {un['num_decode_steps']} steps)",
                         un['decode_per_token'],
                         un['serial_cycles']['decode_per_token'], un['freq_hz'])

    # ---- JSON ----
    out_json = f"{args.out_prefix}.json"
    payload = {
        'config': {
            'model': args.model,
            'input_tokens': input_tokens,
            'output_tokens': args.output_tokens,
            'batch_size': args.batch_size,
            'flash_block_size': args.flash_block,
            'hw': build_hw(args).to_dict(),
        },
        'stage_order': STAGE_ORDER,
        'unit_order': UNIT_ORDER,
        'unit_label': UNIT_LABEL,
        'results': {str(inp): all_results[inp] for inp in sorted(all_results)},
        'units': {str(inp): all_units[inp] for inp in sorted(all_units)},
        'latency': {str(inp): {'ttft': latency[inp][0], 'tpot': latency[inp][1]}
                    for inp in sorted(latency)},
    }
    with open(out_json, 'w') as f:
        json.dump(payload, f, indent=2)

    out_csv = f"{args.out_prefix}.csv"
    write_csv(out_csv, all_results)
    out_unit_csv = f"{args.out_prefix}_units.csv"
    write_unit_csv(out_unit_csv, all_units)

    print(f"\nSaved {out_json}, {out_csv} and {out_unit_csv}")


if __name__ == "__main__":
    main()
