"""
Run bit-width sensitivity simulations for Pareto (TOPS/W vs PPL) analysis.

Configurations:
  - Tender: W4A8-KV8 and W8A8-KV8
  - Omni:   W4-KV2, W4-KV3, W4-KV4

All on OPT-6.7B, input=8192, output=256, FlashAttention block_size=256.
"""

import sys
import os
import json
from multiprocessing import Pool, cpu_count

# --- Path setup ---
_rebuttal_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, _rebuttal_dir)
sys.path.insert(0, os.path.join(_rebuttal_dir, 'simulator'))

from simulator import Simulator, WorkloadConfig, HardwareConfig
from model_configs import get_model_config

# ============================================================================
# Configuration
# ============================================================================

MODEL_NAME = 'LLaMA-2-7B'
INPUT_TOKENS = 8192
OUTPUT_TOKENS = 256
FLASH_BLOCK_SIZE = 256


def get_hw_configs():
    """Hardware configs for bit-width sweep.

    Returns dict[str, HardwareConfig].
    """
    hw = {}

    # Tender W4A8-KV8
    hw['Tender-W4'] = HardwareConfig(
        array_m=64, array_n=64, FPE_array_size=64,
        act_bits=4, accumulate_bits=32, weight_bits=4, kv_cache_bits=4,
        AW_mode="TENDER", AA_mode="TENDER",
    )

    # Tender W8A8-KV8 (original Tender)
    hw['Tender-W8'] = HardwareConfig(
        array_m=64, array_n=64, FPE_array_size=64,
        act_bits=8, accumulate_bits=32, weight_bits=8, kv_cache_bits=8,
        AW_mode="TENDER", AA_mode="TENDER",
    )

    # Omni W4-KV2
    hw['Omni-KV2'] = HardwareConfig(
        array_m=32, array_n=4,
        act_bits=16, accumulate_bits=32, weight_bits=4, kv_cache_bits=2,
        AW_mode="OMNI", AA_mode="OMNI",
    )

    # Omni W4-KV3
    hw['Omni-KV3'] = HardwareConfig(
        array_m=32, array_n=4,
        act_bits=16, accumulate_bits=32, weight_bits=4, kv_cache_bits=3,
        AW_mode="OMNI", AA_mode="OMNI",
    )

    # Omni W4-KV4
    hw['Omni-KV4'] = HardwareConfig(
        array_m=32, array_n=4,
        act_bits=16, accumulate_bits=32, weight_bits=4, kv_cache_bits=4,
        AW_mode="OMNI", AA_mode="OMNI",
    )

    return hw


# ============================================================================
# Simulation helpers
# ============================================================================

def run_single(args):
    """Worker: run one (hw_name, hw_config) simulation."""
    hw_name, hw_config = args
    model = get_model_config(MODEL_NAME)
    workload = WorkloadConfig(
        batch_size=1,
        input_tokens=INPUT_TOKENS,
        output_tokens=OUTPUT_TOKENS,
        flash_block_size=FLASH_BLOCK_SIZE,
    )

    sim = Simulator(hw_config)
    results = sim.simulate(model, workload)
    total = results.get_total_metrics()

    compute_energy = total.compute_energy

    # Tender energy model uses fixed per-tile energy regardless of weight_bits.
    # W4 halves the multiply energy compared to W8, so apply a 0.5× correction.
    # if hw_config.AW_mode == 'TENDER' and hw_config.weight_bits == 4:
    #     compute_energy *= 0.5

    dram_energy = total.dram_energy
    sram_energy = total.sram_energy
    total_energy = compute_energy + dram_energy + sram_energy

    return hw_name, {
        'model': MODEL_NAME,
        'input_tokens': INPUT_TOKENS,
        'output_tokens': OUTPUT_TOKENS,
        'weight_bits': hw_config.weight_bits,
        'kv_cache_bits': hw_config.kv_cache_bits,
        'act_bits': hw_config.act_bits,
        'total_energy': total_energy,
        'compute_energy': compute_energy,
        'dram_energy': dram_energy,
        'sram_energy': sram_energy,
        'total_cycles': total.cycles,
        'total_flops': total.flops,
    }


def main():
    hw_configs = get_hw_configs()
    tasks = list(hw_configs.items())

    n_workers = min(len(tasks), max(1, cpu_count() - 1))
    print(f"Running {len(tasks)} bit-width simulations on {MODEL_NAME} "
          f"(in={INPUT_TOKENS}, out={OUTPUT_TOKENS}) with {n_workers} workers ...")
    print("=" * 70)

    all_results = {}
    with Pool(processes=n_workers) as pool:
        for hw_name, metrics in pool.imap_unordered(run_single, tasks):
            all_results[hw_name] = metrics
            print(f"  Done: {hw_name:16s}  total_energy = {metrics['total_energy']:.4f} J")

    # Save JSON
    out_json = 'bit_width_results.json'
    with open(out_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_json}")

    # Print summary table
    print(f"\n{'Config':<18} {'W bits':>6} {'KV bits':>7} {'Total Energy (J)':>18} "
          f"{'Compute (J)':>13} {'DRAM (J)':>10} {'SRAM (J)':>10}")
    print("-" * 90)
    for name in sorted(all_results):
        m = all_results[name]
        print(f"{name:<18} {m['weight_bits']:>6} {m['kv_cache_bits']:>7} "
              f"{m['total_energy']:>18.4f} {m['compute_energy']:>13.4f} "
              f"{m['dram_energy']:>10.4f} {m['sram_energy']:>10.4f}")


if __name__ == "__main__":
    main()
