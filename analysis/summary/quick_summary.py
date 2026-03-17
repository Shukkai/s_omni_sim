"""
Quick summary: OPT-6.7B, in=16384, out=512 for all designs.
Reports total & GEMM-compute energy, power, and latency.
Writes detailed report to file.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')), 'simulator'))

from simulator import Simulator, WorkloadConfig, HardwareConfig
from model_configs import get_model_config
from collections import defaultdict
import vpu_energy_model

MODEL = 'OPT-6.7B'
INPUT_TOKENS = 8192
OUTPUT_TOKENS = 256
FLASH_BLOCK = 256

hw_configs = {
    'FPE': HardwareConfig(
        array_m=64, array_n=64, FPE_array_size=64,
        act_bits=16, accumulate_bits=32, weight_bits=4, kv_cache_bits=16,
        AW_mode="FPE_OS", AA_mode="FPE_OS",
    ),
    'Tender': HardwareConfig(
        array_m=64, array_n=64, FPE_array_size=64,
        act_bits=8, accumulate_bits=32, weight_bits=8, kv_cache_bits=8,
        AW_mode="TENDER", AA_mode="TENDER", tender_m1_opt=True
    ),
    'FIGLUT': HardwareConfig(
        array_m=32, array_n=4, replication=1, FPE_array_size=64,
        act_bits=16, accumulate_bits=32, weight_bits=4, kv_cache_bits=16,
        AW_mode="LUT_WS", AA_mode="FPE_OS",
    ),
    'Omni4': HardwareConfig(
        array_m=32, array_n=4,
        act_bits=16, accumulate_bits=32, weight_bits=4, kv_cache_bits=4,
        AW_mode="OMNI", AA_mode="OMNI",
    ),
    'Omni3': HardwareConfig(
        array_m=32, array_n=4,
        act_bits=16, accumulate_bits=32, weight_bits=4, kv_cache_bits=3,
        AW_mode="OMNI", AA_mode="OMNI",
    ),
}


def fmt_bytes(b):
    """Format bytes to human-readable."""
    if b >= 1e12:
        return f"{b/1e12:.2f} TB"
    elif b >= 1e9:
        return f"{b/1e9:.2f} GB"
    elif b >= 1e6:
        return f"{b/1e6:.2f} MB"
    elif b >= 1e3:
        return f"{b/1e3:.2f} KB"
    return f"{b} B"


def fmt_ops(n):
    """Format op count to human-readable."""
    if n >= 1e12:
        return f"{n/1e12:.2f} TOps"
    elif n >= 1e9:
        return f"{n/1e9:.2f} GOps"
    elif n >= 1e6:
        return f"{n/1e6:.2f} MOps"
    elif n >= 1e3:
        return f"{n/1e3:.2f} KOps"
    return f"{n}"


def write_report(f, name, hw, results, breakdown, vpu_config):
    """Write detailed report for one design."""
    num_steps = max(1, OUTPUT_TOKENS - 1)
    freq = hw.freq_mhz * 1e6

    # --- Latency ---
    total_aa = breakdown['ttft_aa'] + breakdown['tpot_aa'] * num_steps
    total_aw = breakdown['ttft_aw'] + breakdown['tpot_aw'] * num_steps
    total_ng = breakdown['ttft_non_gemm'] + breakdown['tpot_non_gemm'] * num_steps
    total_lat = total_aa + total_aw + total_ng
    gemm_lat = total_aa + total_aw

    # --- GEMM metrics (prefill + decode, AA + AW) ---
    p_aa = results.prefill.get_aa_total()
    p_aw = results.prefill.get_aw_total()
    d_aa = results.decode.get_aa_total()
    d_aw = results.decode.get_aw_total()
    p_total = results.prefill.get_total_metrics()
    d_total = results.decode.get_total_metrics()
    gemm_total_metrics = results.get_total_metrics()

    # --- Non-GEMM aggregation ---
    all_ng = results.prefill.non_gemm_ops + results.decode.non_gemm_ops
    ng_total_compute_energy = sum(m.compute_energy for m in all_ng)
    ng_total_sram_energy = sum(m.sram_energy for m in all_ng)
    ng_total_energy = sum(m.total_energy for m in all_ng)
    ng_total_sram_read = sum(m.sram_read for m in all_ng)
    ng_total_sram_write = sum(m.sram_write for m in all_ng)
    ng_total_cycles = sum(m.cycles for m in all_ng)

    # Aggregate VPU ops
    vpu_ops = defaultdict(int)
    for m in all_ng:
        for op_name in ['ops_add', 'ops_sub', 'ops_mul', 'ops_div', 'ops_exp',
                        'ops_rsqrt', 'ops_sqrt', 'ops_recip', 'ops_sin', 'ops_cos',
                        'ops_sigmoid', 'ops_reduce_sum', 'ops_reduce_max',
                        'ops_square', 'ops_neg', 'ops_max']:
            vpu_ops[op_name] += getattr(m, op_name)
    total_vpu_ops = sum(vpu_ops.values())

    # Aggregate non-GEMM by op type
    ng_by_type = defaultdict(lambda: {'cycles': 0, 'sram_read': 0, 'sram_write': 0,
                                       'compute_energy': 0.0, 'sram_energy': 0.0,
                                       'total_energy': 0.0, 'num_elements': 0, 'count': 0})
    for m in all_ng:
        t = ng_by_type[m.op_type.value]
        t['cycles'] += m.cycles
        t['sram_read'] += m.sram_read
        t['sram_write'] += m.sram_write
        t['compute_energy'] += m.compute_energy
        t['sram_energy'] += m.sram_energy
        t['total_energy'] += m.total_energy
        t['num_elements'] += m.num_elements
        t['count'] += 1

    # --- Totals ---
    total_energy = gemm_total_metrics.total_energy + ng_total_energy
    total_flops = gemm_total_metrics.flops
    total_power = total_energy / total_lat if total_lat > 0 else 0

    # ==================== WRITE ====================
    w = f.write
    sep = "=" * 100
    dash = "-" * 100

    w(f"\n{sep}\n")
    w(f"  {name}\n")
    w(f"  AW: {hw.AW_mode}  |  AA: {hw.AA_mode}  |  "
      f"W{hw.weight_bits}A{hw.act_bits}KV{hw.kv_cache_bits}  |  "
      f"Array: {hw.array_m}x{hw.array_n}  |  Freq: {hw.freq_mhz} MHz\n")
    w(f"{sep}\n")

    # --- 1. LATENCY SUMMARY ---
    w(f"\n  1. LATENCY\n")
    w(f"  {dash}\n")
    w(f"  {'Component':<25} {'Latency (s)':>14} {'Fraction':>10}\n")
    w(f"  {'-'*55}\n")
    w(f"  {'AA-GEMM':<25} {total_aa:>14.2f} {total_aa/total_lat:>10.1%}\n")
    w(f"  {'AW-GEMM':<25} {total_aw:>14.2f} {total_aw/total_lat:>10.1%}\n")
    w(f"  {'Non-GEMM (VPU)':<25} {total_ng:>14.2f} {total_ng/total_lat:>10.1%}\n")
    w(f"  {'-'*55}\n")
    w(f"  {'GEMM Total':<25} {gemm_lat:>14.2f} {gemm_lat/total_lat:>10.1%}\n")
    w(f"  {'TOTAL':<25} {total_lat:>14.2f} {'100.0%':>10}\n")
    w(f"\n")
    w(f"  TTFT:  {breakdown['ttft_total']:.2f} s  "
      f"(AA={breakdown['ttft_aa']:.2f}, AW={breakdown['ttft_aw']:.2f}, "
      f"VPU={breakdown['ttft_non_gemm']:.4f})\n")
    tpot_ms = breakdown['tpot_total'] * 1e3
    w(f"  TPOT:  {tpot_ms:.2f} ms  "
      f"(AA={breakdown['tpot_aa']*1e3:.2f}, AW={breakdown['tpot_aw']*1e3:.2f}, "
      f"VPU={breakdown['tpot_non_gemm']*1e3:.4f} ms)\n")
    throughput = OUTPUT_TOKENS / total_lat if total_lat > 0 else 0
    w(f"  Throughput: {throughput:.2f} tokens/s\n")

    # --- 2. ENERGY SUMMARY ---
    w(f"\n  2. ENERGY\n")
    w(f"  {dash}\n")
    w(f"  {'Component':<30} {'Energy (J)':>12} {'Fraction':>10}\n")
    w(f"  {'-'*58}\n")
    # GEMM breakdown
    w(f"  GEMM:\n")
    w(f"    {'Compute':<28} {gemm_total_metrics.compute_energy:>12.4f} "
      f"{gemm_total_metrics.compute_energy/total_energy:>10.1%}\n")
    w(f"    {'DRAM Read':<28} {gemm_total_metrics.dram_read_energy:>12.4f} "
      f"{gemm_total_metrics.dram_read_energy/total_energy:>10.1%}\n")
    w(f"    {'DRAM Write':<28} {gemm_total_metrics.dram_write_energy:>12.4f} "
      f"{gemm_total_metrics.dram_write_energy/total_energy:>10.1%}\n")
    w(f"    {'SRAM Read':<28} {gemm_total_metrics.sram_read_energy:>12.4f} "
      f"{gemm_total_metrics.sram_read_energy/total_energy:>10.1%}\n")
    w(f"    {'SRAM Write':<28} {gemm_total_metrics.sram_write_energy:>12.4f} "
      f"{gemm_total_metrics.sram_write_energy/total_energy:>10.1%}\n")
    w(f"    {'--- GEMM Subtotal':<28} {gemm_total_metrics.total_energy:>12.4f} "
      f"{gemm_total_metrics.total_energy/total_energy:>10.1%}\n")
    # Non-GEMM breakdown
    w(f"  Non-GEMM (VPU):\n")
    w(f"    {'VPU Compute':<28} {ng_total_compute_energy:>12.4f} "
      f"{ng_total_compute_energy/total_energy:>10.1%}\n")
    w(f"    {'SRAM (Read+Write)':<28} {ng_total_sram_energy:>12.4f} "
      f"{ng_total_sram_energy/total_energy:>10.1%}\n")
    w(f"    {'--- Non-GEMM Subtotal':<28} {ng_total_energy:>12.4f} "
      f"{ng_total_energy/total_energy:>10.1%}\n")
    w(f"  {'-'*58}\n")
    w(f"  {'TOTAL':<30} {total_energy:>12.4f} {'100.0%':>10}\n")
    w(f"\n  Average Power: {total_power:.2f} W\n")

    # --- 3. MEMORY ACCESS ---
    w(f"\n  3. MEMORY ACCESS\n")
    w(f"  {dash}\n")
    # GEMM
    w(f"  GEMM:\n")
    w(f"    {'DRAM Read':<28} {fmt_bytes(gemm_total_metrics.dram_read):>14}\n")
    w(f"    {'DRAM Write':<28} {fmt_bytes(gemm_total_metrics.dram_write):>14}\n")
    w(f"    {'DRAM Total':<28} {fmt_bytes(gemm_total_metrics.dram_read + gemm_total_metrics.dram_write):>14}\n")
    w(f"    {'SRAM Read':<28} {fmt_bytes(gemm_total_metrics.sram_read):>14}\n")
    w(f"    {'SRAM Write':<28} {fmt_bytes(gemm_total_metrics.sram_write):>14}\n")
    w(f"    {'SRAM Total':<28} {fmt_bytes(gemm_total_metrics.sram_read + gemm_total_metrics.sram_write):>14}\n")
    w(f"    {'Peak SRAM':<28} {fmt_bytes(results.peak_sram_bytes):>14}\n")
    # Non-GEMM
    w(f"  Non-GEMM:\n")
    w(f"    {'SRAM Read':<28} {fmt_bytes(ng_total_sram_read):>14}\n")
    w(f"    {'SRAM Write':<28} {fmt_bytes(ng_total_sram_write):>14}\n")
    w(f"    {'SRAM Total':<28} {fmt_bytes(ng_total_sram_read + ng_total_sram_write):>14}\n")
    # Combined
    total_dram = gemm_total_metrics.dram_read + gemm_total_metrics.dram_write
    total_sram = (gemm_total_metrics.sram_read + gemm_total_metrics.sram_write
                  + ng_total_sram_read + ng_total_sram_write)
    w(f"  Combined:\n")
    w(f"    {'Total DRAM':<28} {fmt_bytes(total_dram):>14}\n")
    w(f"    {'Total SRAM':<28} {fmt_bytes(total_sram):>14}\n")

    # --- 4. COMPUTE OPS ---
    w(f"\n  4. COMPUTE OPERATIONS\n")
    w(f"  {dash}\n")
    w(f"  GEMM:\n")
    w(f"    {'Total FLOPs':<28} {fmt_ops(total_flops):>18}\n")
    w(f"    {'Total Cycles':<28} {gemm_total_metrics.cycles:>18,}\n")
    w(f"    {'Utilization':<28} {gemm_total_metrics.utilization:>17.1%}\n")
    # AA vs AW breakdown
    aa_total = p_aa.flops + d_aa.flops
    aw_total = p_aw.flops + d_aw.flops
    w(f"    {'AA FLOPs':<28} {fmt_ops(aa_total):>18}\n")
    w(f"    {'AW FLOPs':<28} {fmt_ops(aw_total):>18}\n")

    w(f"  Non-GEMM (VPU):\n")
    w(f"    {'Total VPU Ops':<28} {fmt_ops(total_vpu_ops):>18}\n")
    w(f"    {'Total VPU Cycles':<28} {ng_total_cycles:>18,}\n")
    # Breakdown by primitive op
    w(f"    VPU Op Breakdown:\n")
    op_display = [
        ('ops_add', 'vec_add'), ('ops_sub', 'vec_sub'), ('ops_mul', 'vec_mul'),
        ('ops_div', 'vec_div'), ('ops_exp', 'vec_exp'), ('ops_rsqrt', 'vec_rsqrt'),
        ('ops_sqrt', 'vec_sqrt'), ('ops_recip', 'vec_recip'), ('ops_sin', 'vec_sin'),
        ('ops_cos', 'vec_cos'), ('ops_sigmoid', 'vec_sigmoid'),
        ('ops_reduce_sum', 'vec_reduce_sum'), ('ops_reduce_max', 'vec_reduce_max'),
        ('ops_square', 'vec_square'), ('ops_neg', 'vec_neg'), ('ops_max', 'vec_max'),
    ]
    for key, label in op_display:
        cnt = vpu_ops[key]
        if cnt > 0:
            w(f"      {label:<24} {cnt:>18,}\n")

    # --- 5. NON-GEMM BREAKDOWN BY OP TYPE ---
    w(f"\n  5. NON-GEMM BREAKDOWN BY OPERATION\n")
    w(f"  {dash}\n")
    w(f"  {'Operation':<22} {'Count':>6} {'Elements':>14} {'Cycles':>14} "
      f"{'Time (s)':>10} {'Energy (J)':>12}\n")
    w(f"  {'-'*84}\n")
    for op_name, t in sorted(ng_by_type.items()):
        time_s = t['cycles'] / freq
        w(f"  {op_name:<22} {t['count']:>6} {t['num_elements']:>14,} "
          f"{t['cycles']:>14,} {time_s:>10.4f} {t['total_energy']:>12.6f}\n")
    w(f"  {'-'*84}\n")
    ng_time = ng_total_cycles / freq
    w(f"  {'TOTAL':<22} {len(all_ng):>6} {sum(m.num_elements for m in all_ng):>14,} "
      f"{ng_total_cycles:>14,} {ng_time:>10.4f} {ng_total_energy:>12.6f}\n")

    w(f"\n")


# ==================== MAIN ====================
model = get_model_config(MODEL)
workload = WorkloadConfig(batch_size=1, input_tokens=INPUT_TOKENS,
                          output_tokens=OUTPUT_TOKENS, flash_block_size=FLASH_BLOCK)
vpu_config = vpu_energy_model.DEFAULT_VPU_ENERGY

report_path = 'detailed_report.txt'

with open(report_path, 'w') as f:
    f.write("=" * 100 + "\n")
    f.write(f"  DETAILED SYSTEM REPORT\n")
    f.write(f"  Model: {MODEL}  |  Input: {INPUT_TOKENS}  |  Output: {OUTPUT_TOKENS}  |  "
            f"Flash: {FLASH_BLOCK}\n")
    f.write("=" * 100 + "\n")

    # Console header
    print(f"{'Design':<12} | {'Latency(s)':>10} {'GEMM Lat(s)':>12} | "
          f"{'Energy(J)':>10} {'GEMM E(J)':>10} {'Compute E(J)':>12} | "
          f"{'Power(W)':>9} {'GEMM Pwr(W)':>12}")
    print("-" * 110)

    for name, hw in hw_configs.items():
        sim = Simulator(hw)
        results = sim.simulate(model, workload)

        # --- Latency (roofline) ---
        breakdown = sim.compute_roofline_latency_breakdown(results, workload)
        num_steps = max(1, OUTPUT_TOKENS - 1)
        total_aa = breakdown['ttft_aa'] + breakdown['tpot_aa'] * num_steps
        total_aw = breakdown['ttft_aw'] + breakdown['tpot_aw'] * num_steps
        total_ng = breakdown['ttft_non_gemm'] + breakdown['tpot_non_gemm'] * num_steps
        total_lat = total_aa + total_aw + total_ng
        gemm_lat = total_aa + total_aw

        # --- VPU energy ---
        for m in results.prefill.non_gemm_ops:
            m.calculate_energy(vpu_config)
        for m in results.decode.non_gemm_ops:
            m.calculate_energy(vpu_config)

        # GEMM totals
        prefill_total = results.prefill.get_total_metrics()
        decode_total = results.decode.get_total_metrics()
        gemm_energy = prefill_total.total_energy + decode_total.total_energy
        gemm_compute = prefill_total.compute_energy + decode_total.compute_energy

        ng_energy = (sum(m.total_energy for m in results.prefill.non_gemm_ops)
                     + sum(m.total_energy for m in results.decode.non_gemm_ops))
        total_energy = gemm_energy + ng_energy

        total_power = total_energy / total_lat if total_lat > 0 else 0
        gemm_power = gemm_energy / gemm_lat if gemm_lat > 0 else 0

        # Console
        print(f"{name:<12} | {total_lat:>10.1f} {gemm_lat:>12.1f} | "
              f"{total_energy:>10.1f} {gemm_energy:>10.1f} {gemm_compute:>12.4f} | "
              f"{total_power:>9.2f} {gemm_power:>12.2f}")

        # Detailed report
        write_report(f, name, hw, results, breakdown, vpu_config)

print(f"\nDetailed report saved to: {report_path}")
