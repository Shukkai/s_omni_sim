"""
Run TTFT / TPOT roofline simulations for OPT-6.7B across hardware configs.

Compares: Tender, FPE, FIGLUT, Omni4, Omni3
Input tokens: 2048, 8192, 32768 (fixed output = 256)
"""

import sys
import os
import json
from multiprocessing import Pool, cpu_count

# Add parent directory and simulator directory to path
_rebuttal_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, _rebuttal_dir)
sys.path.insert(0, os.path.join(_rebuttal_dir, 'simulator'))

from simulator import Simulator, WorkloadConfig, HardwareConfig
from model_configs import get_model_config


# ---- Hardware configurations ------------------------------------------------

def get_hw_configs():
    """Define all hardware configurations to compare."""
    hw_configs = {}

    # Tender
    hw_configs['Tender'] = HardwareConfig(
        array_m=64, array_n=64, FPE_array_size=64,
        act_bits=8, accumulate_bits=32, weight_bits=8, kv_cache_bits=8,
        AW_mode="TENDER", AA_mode="TENDER",
    )

    # FPE (W4A16)
    hw_configs['FPE'] = HardwareConfig(
        array_m=64, array_n=64, FPE_array_size=64,
        act_bits=16, accumulate_bits=32, weight_bits=4, kv_cache_bits=16,
        AW_mode="FPE_OS", AA_mode="FPE_OS",
    )

    # FIGLUT (W4A16 with LUT for AW, FPE for AA)
    # hw_configs['FIGLUT'] = HardwareConfig(
    #     array_m=16, array_n=2, replication=4, FPE_array_size=64,
    #     act_bits=16, accumulate_bits=32, weight_bits=4, kv_cache_bits=16,
    #     AW_mode="LUT_WS", AA_mode="FPE_OS",
    # )
    
    hw_configs['FIGLUT'] = HardwareConfig(
        array_m=32, array_n=4, replication=1, FPE_array_size=64,
        act_bits=16, accumulate_bits=32, weight_bits=4, kv_cache_bits=16,
        AW_mode="LUT_WS", AA_mode="FPE_OS",
    )

    # Omni variants (different KV cache bit-widths)
    for kv_bits in [3, 4]:
        hw_configs[f'Omni{kv_bits}'] = HardwareConfig(
            array_m=32, array_n=4,
            act_bits=16, accumulate_bits=32, weight_bits=4, kv_cache_bits=kv_bits,
            AW_mode="OMNI", AA_mode="OMNI",
        )

    return hw_configs


# ---- Simulation settings ----------------------------------------------------

DEFAULT_FLASH_BLOCK_SIZE = 256
MODEL_NAME = 'OPT-6.7B'
INPUT_TOKENS = [2048, 8192, 32768]
OUTPUT_TOKENS = 256


def run_single(args):
    """Worker: simulate one (hw, input_tokens) pair and return TTFT + TPOT breakdown."""
    hw_name, hw_config, input_tokens = args
    model_config = get_model_config(MODEL_NAME)

    workload = WorkloadConfig(
        batch_size=1,
        input_tokens=input_tokens,
        output_tokens=OUTPUT_TOKENS,
        flash_block_size=DEFAULT_FLASH_BLOCK_SIZE,
    )

    simulator = Simulator(hw_config)
    results = simulator.simulate(model_config, workload)
    ttft, tpot = simulator.compute_roofline_latency(results, workload)
    breakdown = simulator.compute_roofline_latency_breakdown(results, workload)

    return (hw_name, input_tokens, ttft, tpot, breakdown)


def main():
    hw_configs = get_hw_configs()

    # Build task list
    tasks = []
    for hw_name, hw_config in hw_configs.items():
        for inp in INPUT_TOKENS:
            tasks.append((hw_name, hw_config, inp))

    n_workers = max(1, cpu_count() - 1)
    total = len(tasks)
    print(f"Running {total} simulations ({MODEL_NAME}) with {n_workers} workers ...\n")

    # Run in parallel
    all_results = {}
    completed = 0
    with Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(run_single, tasks):
            hw_name, inp, ttft, tpot, breakdown = result
            key = f"{inp}_{OUTPUT_TOKENS}"
            all_results.setdefault(hw_name, {})[key] = {
                'input_tokens': inp,
                'output_tokens': OUTPUT_TOKENS,
                'ttft': ttft,
                'tpot': tpot,
                'ttft_aa': breakdown['ttft_aa'],
                'ttft_aw': breakdown['ttft_aw'],
                'ttft_non_gemm': breakdown['ttft_non_gemm'],
                'tpot_aa': breakdown['tpot_aa'],
                'tpot_aw': breakdown['tpot_aw'],
                'tpot_non_gemm': breakdown['tpot_non_gemm'],
            }
            completed += 1
            print(f"[{completed}/{total}]  {hw_name:<10}  input={inp:<6}  "
                  f"TTFT={ttft*1e3:.2f} ms   TPOT={tpot*1e3:.2f} ms"
                  f"  (AA={breakdown['ttft_aa']*1e3:.1f} AW={breakdown['ttft_aw']*1e3:.1f}"
                  f" non-GEMM={breakdown['ttft_non_gemm']*1e3:.1f} ms)")

    # Sort for deterministic JSON
    from collections import OrderedDict
    sorted_results = OrderedDict()
    for hw in sorted(all_results):
        sorted_results[hw] = OrderedDict(
            sorted(all_results[hw].items(),
                   key=lambda kv: kv[1]['input_tokens']))

    out_json = 'throughput_results.json'
    with open(out_json, 'w') as f:
        json.dump(sorted_results, f, indent=2)
    print(f"\nResults saved to {out_json}")


if __name__ == "__main__":
    main()
