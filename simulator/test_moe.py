"""Quick test: compare Dense vs MoE to verify MoE correctly scales FFN costs."""
import sys
sys.path.insert(0, '.')
from simulator import *

model_dense = ModelConfig(
    num_layers=32, num_heads=32, num_kv_heads=32,
    d_model=4096, d_ffn=16384, head_dim=128,
)
model_moe = ModelConfig(
    num_layers=32, num_heads=32, num_kv_heads=32,
    d_model=4096, d_ffn=16384, head_dim=128,
    num_experts=8, num_active_experts=2,   # Mixtral-style
)

workload = WorkloadConfig(batch_size=1, input_tokens=8192, output_tokens=256)

hw = HardwareConfig(
    array_m=32, array_n=4,
    act_bits=16, accumulate_bits=32, weight_bits=4, kv_cache_bits=4,
    AW_mode="OMNI", AA_mode="OMNI",
)

sim = Simulator(hw)

print(f"Dense: num_experts=1, num_active_experts=1")
print(f"MoE:   num_experts={model_moe.num_experts}, num_active_experts={model_moe.num_active_experts}")
print()

r_dense = sim.simulate(model_dense, workload)
r_moe = sim.simulate(model_moe, workload)

t_dense = r_dense.get_total_metrics()
t_moe = r_moe.get_total_metrics()

print(f"{'Metric':<25s} {'Dense':>20s} {'MoE (8E/2A)':>20s} {'Ratio':>10s}")
print("-" * 77)
for name, d_val, m_val in [
    ("Total Cycles",     t_dense.cycles,         t_moe.cycles),
    ("Total Energy (J)", t_dense.total_energy,    t_moe.total_energy),
    ("DRAM Read (B)",    t_dense.dram_read,       t_moe.dram_read),
    ("DRAM Write (B)",   t_dense.dram_write,      t_moe.dram_write),
    ("SRAM Read (B)",    t_dense.sram_read,       t_moe.sram_read),
    ("Utilization",      t_dense.utilization,     t_moe.utilization),
    ("Peak SRAM (B)",    r_dense.peak_sram_bytes, r_moe.peak_sram_bytes),
]:
    ratio = m_val / d_val if d_val else 0
    if isinstance(d_val, float):
        print(f"{name:<25s} {d_val:>20.4f} {m_val:>20.4f} {ratio:>10.4f}")
    else:
        print(f"{name:<25s} {d_val:>20,} {m_val:>20,} {ratio:>10.4f}")

# Verify FC1/FC2 scale correctly
print("\n--- FC1 details (prefill) ---")
fc1_dense = r_dense.prefill.get_operation_total(OperationType.FC1, ComputeMode.AW)
fc1_moe = r_moe.prefill.get_operation_total(OperationType.FC1, ComputeMode.AW)
print(f"  Dense FC1 execs: {len(r_dense.prefill.aw_ops[OperationType.FC1])}")
print(f"  MoE   FC1 execs: {len(r_moe.prefill.aw_ops[OperationType.FC1])}")
print(f"  Dense FC1 DRAM read: {fc1_dense.dram_read:,}")
print(f"  MoE   FC1 DRAM read: {fc1_moe.dram_read:,}  (ratio={fc1_moe.dram_read/fc1_dense.dram_read:.1f}x)")
print(f"  Dense FC1 cycles:    {fc1_dense.cycles:,}")
print(f"  MoE   FC1 cycles:    {fc1_moe.cycles:,}  (ratio={fc1_moe.cycles/fc1_dense.cycles:.1f}x)")

# Check gate exists
print("\n--- Gate details (prefill) ---")
if OperationType.GATE in r_moe.prefill.aw_ops:
    gate = r_moe.prefill.get_operation_total(OperationType.GATE, ComputeMode.AW)
    gate_list = r_moe.prefill.aw_ops[OperationType.GATE]
    print(f"  Gate execs: {len(gate_list)}")
    print(f"  Gate shape: {gate_list[0].shape}")
    print(f"  Gate cycles: {gate.cycles:,}")
else:
    print("  ERROR: Gate operation not found!")
