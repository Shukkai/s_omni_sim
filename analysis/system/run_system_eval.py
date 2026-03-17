"""
Run full-system evaluation: throughput (tokens/s) and energy (J)
with AA-GEMM / AW-GEMM / non-GEMM breakdown.

Compares: Tender, FPE, FIGLUT, Omni4, Omni3
Input tokens: 128, 1024, 8192 (fixed output = 256)
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
import vpu_energy_model


# ---- Hardware configurations ------------------------------------------------

def get_hw_configs():
    """Define all hardware configurations to compare."""
    hw_configs = {}

    # Tender
    hw_configs['Tender'] = HardwareConfig(
        array_m=64, array_n=64, FPE_array_size=64,
        act_bits=8, accumulate_bits=32, weight_bits=8, kv_cache_bits=8,
        AW_mode="TENDER", AA_mode="TENDER", tender_m1_opt=True
    )

    # FPE (W4A16)
    hw_configs['FPE'] = HardwareConfig(
        array_m=64, array_n=64, FPE_array_size=64,
        act_bits=16, accumulate_bits=32, weight_bits=4, kv_cache_bits=16,
        AW_mode="FPE_OS", AA_mode="FPE_OS",
    )

    # FIGLUT
    hw_configs['FIGLUT'] = HardwareConfig(
        array_m=32, array_n=4, replication=1, FPE_array_size=64,
        act_bits=16, accumulate_bits=32, weight_bits=4, kv_cache_bits=16,
        AW_mode="LUT_WS", AA_mode="FPE_OS",
    )

    # Omni variants
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
INPUT_TOKENS = [1024, 4096, 16384]
OUTPUT_TOKENS = 512


def run_single(args):
    """Worker: simulate one (hw, input_tokens) pair.

    Returns throughput (tokens/s) and energy breakdown (AA/AW/non-GEMM).
    """
    hw_name, hw_config, input_tokens = args
    model_config = get_model_config(MODEL_NAME)
    vpu_config = vpu_energy_model.DEFAULT_VPU_ENERGY

    workload = WorkloadConfig(
        batch_size=1,
        input_tokens=input_tokens,
        output_tokens=OUTPUT_TOKENS,
        flash_block_size=DEFAULT_FLASH_BLOCK_SIZE,
    )

    simulator = Simulator(hw_config)
    results = simulator.simulate(model_config, workload)

    # ---- Latency breakdown (roofline) ----
    breakdown = simulator.compute_roofline_latency_breakdown(results, workload)
    # Total latency = prefill_total + decode_total
    # (breakdown gives ttft = prefill total, tpot = per-step decode)
    num_steps = max(1, workload.output_tokens - 1)
    total_aa_time = breakdown['ttft_aa'] + breakdown['tpot_aa'] * num_steps
    total_aw_time = breakdown['ttft_aw'] + breakdown['tpot_aw'] * num_steps
    total_ng_time = breakdown['ttft_non_gemm'] + breakdown['tpot_non_gemm'] * num_steps
    total_time = total_aa_time + total_aw_time + total_ng_time
    throughput = OUTPUT_TOKENS / total_time if total_time > 0 else 0.0

    # ---- Energy breakdown ----
    # Calculate VPU energy for non-GEMM ops
    for m in results.prefill.non_gemm_ops:
        m.calculate_energy(vpu_config)
    for m in results.decode.non_gemm_ops:
        m.calculate_energy(vpu_config)

    # AA-GEMM energy (prefill + decode)
    aa_energy = (results.prefill.get_aa_total().total_energy
                 + results.decode.get_aa_total().total_energy)
    # AW-GEMM energy (prefill + decode)
    aw_energy = (results.prefill.get_aw_total().total_energy
                 + results.decode.get_aw_total().total_energy)
    # Non-GEMM energy (prefill + decode)
    ng_energy = sum(m.total_energy for m in results.prefill.non_gemm_ops) \
              + sum(m.total_energy for m in results.decode.non_gemm_ops)

    return (hw_name, input_tokens, {
        'throughput': throughput,
        'total_time': total_time,
        'aa_time': total_aa_time,
        'aw_time': total_aw_time,
        'ng_time': total_ng_time,
        'total_energy': aa_energy + aw_energy + ng_energy,
        'aa_energy': aa_energy,
        'aw_energy': aw_energy,
        'ng_energy': ng_energy,
    })


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
            hw_name, inp, metrics = result
            key = f"{inp}_{OUTPUT_TOKENS}"
            all_results.setdefault(hw_name, {})[key] = {
                'input_tokens': inp,
                'output_tokens': OUTPUT_TOKENS,
                **metrics,
            }
            completed += 1
            print(f"[{completed}/{total}]  {hw_name:<10}  input={inp:<6}  "
                  f"throughput={metrics['throughput']:.1f} tok/s  "
                  f"energy={metrics['total_energy']:.4f} J  "
                  f"(AA={metrics['aa_energy']:.4f} AW={metrics['aw_energy']:.4f} "
                  f"non-GEMM={metrics['ng_energy']:.6f} J)")

    # Sort for deterministic JSON
    from collections import OrderedDict
    sorted_results = OrderedDict()
    for hw in sorted(all_results):
        sorted_results[hw] = OrderedDict(
            sorted(all_results[hw].items(),
                   key=lambda kv: kv[1]['input_tokens']))

    out_json = 'system_eval_results.json'
    with open(out_json, 'w') as f:
        json.dump(sorted_results, f, indent=2)
    print(f"\nResults saved to {out_json}")


if __name__ == "__main__":
    main()
