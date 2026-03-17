"""
Run simulations for multiple hardware configurations and save results for plotting
Compares: Tender, FPE, FIGLUT, Omni (with different KV cache bit-widths)
"""

import sys
import os
import json
import csv
from multiprocessing import Pool, cpu_count
from functools import partial

# Add parent directory and simulator directory to path
_rebuttal_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _rebuttal_dir)
sys.path.insert(0, os.path.join(_rebuttal_dir, 'simulator'))

from simulator import (
    Simulator, ModelConfig, WorkloadConfig, HardwareConfig,
    OperationType, ComputeMode,
)
from model_configs import get_model_config, list_models


# Hardware configurations
def get_hw_configs():
    """Define all hardware configurations to compare.
    
    Returns dict[str, HardwareConfig].
    """
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
    hw_configs['FIGLUT'] = HardwareConfig(
        array_m=16, array_n=2, replication=4, FPE_array_size=64,
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


# Default FlashAttention block size (applied to all simulations)
DEFAULT_FLASH_BLOCK_SIZE = 256


def get_model_configs(model_names=None):
    """Get model configurations from the central registry.
    
    Args:
        model_names: List of model names to include.  If None, defaults
                     to ['OPT-6.7B'].
    
    Returns dict[str, ModelConfig].
    """
    if model_names is None:
        model_names = ['OPT-6.7B']
    return {name: get_model_config(name) for name in model_names}


def run_single_simulation(hw_config: HardwareConfig, model_config: ModelConfig,
                          input_tokens: int, output_tokens: int,
                          flash_block_size: int = DEFAULT_FLASH_BLOCK_SIZE):
    """Run a single simulation and extract key metrics"""
    
    workload = WorkloadConfig(
        batch_size=1,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        flash_block_size=flash_block_size,
    )
    
    simulator = Simulator(hw_config)
    results = simulator.simulate(model_config, workload)
    
    # Extract metrics
    prefill = results.prefill
    decode = results.decode
    total = results.get_total_metrics()
    
    # Calculate compute total energy
    compute_total_energy = (prefill.get_aa_total().compute_energy + 
                           decode.get_aa_total().compute_energy + 
                           prefill.get_aw_total().compute_energy + 
                           decode.get_aw_total().compute_energy)
    
    metrics = {
        # Compute Energy
        # 'aa_prefill_compute_energy': prefill.get_aa_total().compute_energy,
        # 'aa_decode_compute_energy': decode.get_aa_total().compute_energy,
        # 'aa_total_compute_energy': prefill.get_aa_total().compute_energy + decode.get_aa_total().compute_energy,
        
        # 'aw_prefill_compute_energy': prefill.get_aw_total().compute_energy,
        # 'aw_decode_compute_energy': decode.get_aw_total().compute_energy,
        # 'aw_total_compute_energy': prefill.get_aw_total().compute_energy + decode.get_aw_total().compute_energy,
        
        'compute_total_energy': compute_total_energy,
        
        # DRAM Energy
        # 'dram_prefill_energy': prefill.get_total_metrics().dram_energy,
        # 'dram_decode_energy': decode.get_total_metrics().dram_energy,
        'dram_total_energy': total.dram_energy,
        
        # SRAM Energy
        # 'sram_prefill_energy': prefill.get_total_metrics().sram_energy,
        # 'sram_decode_energy': decode.get_total_metrics().sram_energy,
        'sram_total_energy': total.sram_energy,
        
        # Total Energy
        # 'prefill_total_energy': prefill.get_total_metrics().total_energy,
        # 'decode_total_energy': decode.get_total_metrics().total_energy,
        'total_energy': total.total_energy,
        
        # FLOPS (for reference)
        # 'aa_total_flops': prefill.get_aa_total().flops + decode.get_aa_total().flops,
        # 'aw_total_flops': prefill.get_aw_total().flops + decode.get_aw_total().flops,
        # 'total_flops': total.flops,
        
        # Cycles
        # 'total_cycles': total.cycles,
        # 'prefill_cycles': prefill.get_total_metrics().cycles,
        # 'decode_cycles': decode.get_total_metrics().cycles,
    }
    
    return metrics


def run_simulation_task(args):
    """Worker function for parallel execution"""
    model_name, model_config, hw_name, hw_config, input_tokens, output_tokens = args
    
    try:
        metrics = run_single_simulation(hw_config, model_config, input_tokens, output_tokens)
        key = f"{input_tokens}_{output_tokens}"
        return (model_name, hw_name, key, {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            **metrics
        })
    except Exception as e:
        print(f"Error in {model_name}/{hw_name} with {input_tokens}/{output_tokens}: {e}")
        return None


def run_all_simulations(output_json='hw_comparison_results.json', output_csv='hw_comparison_results.csv',
                       vary_input=True, fixed_input_tokens=None, fixed_output_tokens=None, n_workers=None):
    """Run simulations for all configurations and save results (parallel version)
    
    Args:
        output_json: Path to save JSON results
        output_csv: Path to save CSV results
        vary_input: If True, vary input tokens and fix output tokens
                   If False, vary output tokens and fix input tokens
        fixed_input_tokens: Fixed input token length (used when vary_input=False)
        fixed_output_tokens: Fixed output token length (used when vary_input=True)
        n_workers: Number of parallel workers (default: CPU count - 1)
    """
    
    hw_configs = get_hw_configs()
    model_configs = get_model_configs()
    
    # Token length configurations to test
    if vary_input:
        # Vary input tokens, fix output tokens
        varying_tokens = [1024, 2048, 4096, 8192, 32768]
        fixed_tokens = fixed_output_tokens if fixed_output_tokens else 256
        token_type = 'input'
    else:
        # Vary output tokens, fix input tokens
        varying_tokens = [256, 512, 1024, 2048, 4096]
        fixed_tokens = fixed_input_tokens if fixed_input_tokens else 2048
        token_type = 'output'
    
    # Determine number of workers
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)
    
    print("Starting simulations...")
    print(f"Mode: Varying {token_type} tokens")
    if vary_input:
        print(f"Fixed output tokens: {fixed_tokens}")
    else:
        print(f"Fixed input tokens: {fixed_tokens}")
    print(f"Using {n_workers} parallel workers")
    print("=" * 80)
    
    # Create task list
    tasks = []
    for model_name, model_config in model_configs.items():
        for hw_name, hw_config in hw_configs.items():
            for varying_token_value in varying_tokens:
                # Determine input and output tokens based on mode
                if vary_input:
                    input_tokens = varying_token_value
                    output_tokens = fixed_tokens
                else:
                    input_tokens = fixed_tokens
                    output_tokens = varying_token_value
                
                tasks.append((model_name, model_config, hw_name, hw_config, 
                            input_tokens, output_tokens))
    
    total_runs = len(tasks)
    print(f"Total simulations to run: {total_runs}")
    print("=" * 80)
    
    # Run simulations in parallel
    all_results = {}
    completed = 0
    
    with Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(run_simulation_task, tasks):
            if result is not None:
                model_name, hw_name, key, metrics = result
                
                # Initialize nested dicts if needed
                if model_name not in all_results:
                    all_results[model_name] = {}
                if hw_name not in all_results[model_name]:
                    all_results[model_name][hw_name] = {}
                
                all_results[model_name][hw_name][key] = metrics
                
                completed += 1
                print(f"[{completed}/{total_runs}] Completed: {model_name}/{hw_name} "
                      f"(input={metrics['input_tokens']}, output={metrics['output_tokens']})")
            else:
                completed += 1
    
    print("\n" + "=" * 80)
    print("All simulations completed!")
    
    # Sort results for consistent JSON output
    sorted_results = sort_results(all_results)
    
    # Save results to JSON
    with open(output_json, 'w') as f:
        json.dump(sorted_results, f, indent=2)
    print(f"\nResults saved to: {output_json}")
    
    # Save results to CSV for easy viewing
    save_to_csv(all_results, output_csv)
    print(f"Results saved to: {output_csv}")
    
    return all_results


def sort_results(results):
    """Sort results dictionary for consistent JSON output"""
    from collections import OrderedDict
    
    sorted_results = OrderedDict()
    
    # Sort models
    for model_name in sorted(results.keys()):
        sorted_results[model_name] = OrderedDict()
        hw_results = results[model_name]
        
        # Sort hardware configs
        for hw_name in sorted(hw_results.keys()):
            sorted_results[model_name][hw_name] = OrderedDict()
            token_results = hw_results[hw_name]
            
            # Sort token configurations by input_tokens, then output_tokens
            sorted_keys = sorted(token_results.keys(), key=lambda k: (
                token_results[k]['input_tokens'],
                token_results[k]['output_tokens']
            ))
            
            for key in sorted_keys:
                sorted_results[model_name][hw_name][key] = token_results[key]
    
    return sorted_results


def save_to_csv(results, output_csv):
    """Save results to CSV format (sorted by model, hardware, and tokens)"""
    
    # Collect all rows
    rows = []
    for model_name, hw_results in results.items():
        for hw_name, token_results in hw_results.items():
            for key, metrics in token_results.items():
                rows.append([
                    model_name,
                    hw_name,
                    metrics['input_tokens'],
                    metrics['output_tokens'],
                    # metrics['aa_total_compute_energy'],
                    # metrics['aw_total_compute_energy'],
                    metrics['compute_total_energy'],
                    metrics['dram_total_energy'],
                    metrics['sram_total_energy'],
                    metrics['total_energy'],
                    # metrics['prefill_total_energy'],
                    # metrics['decode_total_energy'],
                    # metrics['total_flops'],
                    # metrics['total_cycles'],
                ])
    
    # Sort by: model name, hardware name, input tokens, output tokens
    rows.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    
    # Write to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Header
        writer.writerow(['Model', 'Hardware', 'Input Tokens', 'Output Tokens',
                        'AA Compute Energy (J)', 'AW Compute Energy (J)', 'Compute Total Energy (J)',
                        'DRAM Energy (J)', 'SRAM Energy (J)', 'Total Energy (J)',
                        'Prefill Energy (J)', 'Decode Energy (J)',
                        'Total FLOPs', 'Total Cycles'])
        
        # Write sorted data rows
        writer.writerows(rows)


def print_summary(results):
    """Print a summary of results (sorted)"""
    
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    # Sort models
    for model_name in sorted(results.keys()):
        hw_results = results[model_name]
        print(f"\n{model_name}:")
        print("-" * 80)
        
        # Get all token configurations and sort them
        first_hw = list(hw_results.keys())[0]
        token_keys = hw_results[first_hw].keys()
        
        # Sort by input_tokens, then output_tokens
        sorted_keys = sorted(token_keys, key=lambda k: (
            hw_results[first_hw][k]['input_tokens'],
            hw_results[first_hw][k]['output_tokens']
        ))
        
        for key in sorted_keys:
            # Get input/output info from first hw config
            metrics_sample = hw_results[first_hw][key]
            input_tokens = metrics_sample['input_tokens']
            output_tokens = metrics_sample['output_tokens']
            
            print(f"\n  Input Tokens: {input_tokens}, Output Tokens: {output_tokens}")
            print(f"  {'Hardware':<15} {'Total Energy (J)':<20} {'DRAM Energy (J)':<20} {'SRAM Energy (J)':<20}")
            print("  " + "-" * 75)
            
            # Sort hardware names
            for hw_name in sorted(hw_results.keys()):
                metrics = hw_results[hw_name][key]
                print(f"  {hw_name:<15} {metrics['total_energy']:<20.2f} "
                      f"{metrics['dram_total_energy']:<20.2f} "
                      f"{metrics['sram_total_energy']:<20.2f}")


def main():
    """Main entry point"""
    
    # Check command line arguments
    # vary_input = True
    vary_input = False  # Default: vary input tokens
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == 'vary_output' or mode == 'output':
            vary_input = False
        elif mode == 'vary_input' or mode == 'input':
            vary_input = True
        else:
            print("Usage: python run_hw_comparison.py [vary_input|vary_output]")
            print("  vary_input (default): Vary input tokens, fix output tokens at 256")
            print("  vary_output: Vary output tokens, fix input tokens at 32768")
            sys.exit(1)
    
    if vary_input:
        output_json = 'hw_comparison_vary_input.json'
        output_csv = 'hw_comparison_vary_input.csv'
        print("Running simulations with varying INPUT tokens...")
    else:
        output_json = 'hw_comparison_vary_output.json'
        output_csv = 'hw_comparison_vary_output.csv'
        print("Running simulations with varying OUTPUT tokens...")
    
    # Run all simulations
    results = run_all_simulations(
        output_json=output_json,
        output_csv=output_csv,
        vary_input=vary_input
    )
    
    # Print summary
    print_summary(results)
    
    print("\n" + "=" * 80)
    print("Done! You can now use these files for plotting:")
    print(f"  - {output_json} (for Python plotting scripts)")
    print(f"  - {output_csv} (for spreadsheet tools or pandas)")
    print("=" * 80)


if __name__ == "__main__":
    main()
