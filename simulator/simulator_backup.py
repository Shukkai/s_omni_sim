import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List
from enum import Enum

import dram_power_model
import sram_power_model
import ws_energy_model
import fpe_os_energy_model
import omni_energy_model
import tender_energy_model
import os_energy_model
import os_v_energy_model


class Phase(Enum):
    """Execution phase"""
    PREFILL = "prefill"
    DECODE = "decode"


class OperationType(Enum):
    """Operation type"""
    Q_PROJ = "q_proj"
    K_PROJ = "k_proj"
    V_PROJ = "v_proj"
    QK_MATMUL = "qk_matmul"
    ATTN_V_MATMUL = "attn_v_matmul"
    O_PROJ = "o_proj"
    FC1 = "fc1"
    FC2 = "fc2"


class ComputeMode(Enum):
    """Compute mode for different operations"""
    AA = "AA"  # Activation-Activation
    AW = "AW"  # Activation-Weight


@dataclass
class OperationMetrics:
    """Metrics for a single operation"""
    shape: tuple  # (M, K, N) for matrix multiplication
    cycles: int = 0
    flops: int = 0
    utilization: float = 0.0
    throughput: float = 0.0  # TOPS or tokens/sec
    
    # Memory access
    dram_read: int = 0  # bytes
    dram_write: int = 0  # bytes
    sram_read: int = 0  # bytes
    sram_write: int = 0  # bytes
    
    # Energy
    compute_energy: float = 0.0  # Joules
    dram_read_energy: float = 0.0  # Joules
    dram_write_energy: float = 0.0  # Joules
    sram_read_energy: float = 0.0  # Joules
    sram_write_energy: float = 0.0  # Joules

    @property
    def dram_energy(self):
        return self.dram_read_energy + self.dram_write_energy
    
    @property
    def sram_energy(self):
        return self.sram_read_energy + self.sram_write_energy

    @property
    def total_energy(self):
        return self.compute_energy + self.dram_energy + self.sram_energy


@dataclass
class PhaseMetrics:
    """Metrics for a phase (Prefill or Decode)"""
    phase: Phase
    aa_ops: Dict[OperationType, List[OperationMetrics]] = field(default_factory=lambda: defaultdict(list))
    aw_ops: Dict[OperationType, List[OperationMetrics]] = field(default_factory=lambda: defaultdict(list))
    
    def add_operation(self, op_type: OperationType, compute_mode: ComputeMode, metrics: OperationMetrics):
        """Add an operation execution record"""
        if compute_mode == ComputeMode.AA:
            self.aa_ops[op_type].append(metrics)
        else:
            self.aw_ops[op_type].append(metrics)
    
    def get_total_metrics(self) -> OperationMetrics:
        """Aggregate all operations in this phase"""
        total = OperationMetrics(shape=(0, 0, 0))
        weighted_util = 0.0
        
        for ops_dict in [self.aa_ops, self.aw_ops]:
            for op_list in ops_dict.values():
                for op_metrics in op_list:
                    total.cycles += op_metrics.cycles
                    total.flops += op_metrics.flops
                    total.dram_read += op_metrics.dram_read
                    total.dram_write += op_metrics.dram_write
                    total.sram_read += op_metrics.sram_read
                    total.sram_write += op_metrics.sram_write
                    total.compute_energy += op_metrics.compute_energy
                    total.dram_read_energy += op_metrics.dram_read_energy
                    total.dram_write_energy += op_metrics.dram_write_energy
                    total.sram_read_energy += op_metrics.sram_read_energy
                    total.sram_write_energy += op_metrics.sram_write_energy
                    weighted_util += op_metrics.utilization * op_metrics.cycles
        
        total.utilization = weighted_util / total.cycles if total.cycles > 0 else 0.0
        return total
    
    def get_aa_total(self) -> OperationMetrics:
        """Get total metrics for all AA operations"""
        return self._aggregate_ops(self.aa_ops)
    
    def get_aw_total(self) -> OperationMetrics:
        """Get total metrics for all AW operations"""
        return self._aggregate_ops(self.aw_ops)
    
    def get_operation_total(self, op_type: OperationType, compute_mode: ComputeMode) -> OperationMetrics:
        """Get total metrics for a specific operation type (sum of all executions)"""
        ops_dict = self.aa_ops if compute_mode == ComputeMode.AA else self.aw_ops
        op_list = ops_dict.get(op_type, [])
        
        total = OperationMetrics(shape=(0, 0, 0))
        weighted_util = 0.0
        
        for op_metrics in op_list:
            total.cycles += op_metrics.cycles
            total.flops += op_metrics.flops
            total.dram_read += op_metrics.dram_read
            total.dram_write += op_metrics.dram_write
            total.sram_read += op_metrics.sram_read
            total.sram_write += op_metrics.sram_write
            total.compute_energy += op_metrics.compute_energy
            total.dram_read_energy += op_metrics.dram_read_energy
            total.dram_write_energy += op_metrics.dram_write_energy
            total.sram_read_energy += op_metrics.sram_read_energy
            total.sram_write_energy += op_metrics.sram_write_energy
            weighted_util += op_metrics.utilization * op_metrics.cycles
        
        total.utilization = weighted_util / total.cycles if total.cycles > 0 else 0.0
        return total
    
    def _aggregate_ops(self, ops_dict: Dict[OperationType, List[OperationMetrics]]) -> OperationMetrics:
        """Helper to aggregate operations"""
        total = OperationMetrics(shape=(0, 0, 0))
        weighted_util = 0.0
        
        for op_list in ops_dict.values():
            for op_metrics in op_list:
                total.cycles += op_metrics.cycles
                total.flops += op_metrics.flops
                total.dram_read += op_metrics.dram_read
                total.dram_write += op_metrics.dram_write
                total.sram_read += op_metrics.sram_read
                total.sram_write += op_metrics.sram_write
                total.compute_energy += op_metrics.compute_energy
                total.dram_read_energy += op_metrics.dram_read_energy
                total.dram_write_energy += op_metrics.dram_write_energy
                total.sram_read_energy += op_metrics.sram_read_energy
                total.sram_write_energy += op_metrics.sram_write_energy
                weighted_util += op_metrics.utilization * op_metrics.cycles
        
        total.utilization = weighted_util / total.cycles if total.cycles > 0 else 0.0
        return total


@dataclass
class SimulationResults:
    """Complete simulation results"""
    prefill: PhaseMetrics
    decode: PhaseMetrics
    
    def get_total_metrics(self) -> OperationMetrics:
        """Get total metrics across all phases"""
        prefill_total = self.prefill.get_total_metrics()
        decode_total = self.decode.get_total_metrics()
        
        total = OperationMetrics(shape=(0, 0, 0))
        total.cycles = prefill_total.cycles + decode_total.cycles
        total.flops = prefill_total.flops + decode_total.flops
        total.dram_read = prefill_total.dram_read + decode_total.dram_read
        total.dram_write = prefill_total.dram_write + decode_total.dram_write
        total.sram_read = prefill_total.sram_read + decode_total.sram_read
        total.sram_write = prefill_total.sram_write + decode_total.sram_write
        total.compute_energy = prefill_total.compute_energy + decode_total.compute_energy
        total.dram_read_energy = prefill_total.dram_read_energy + decode_total.dram_read_energy
        total.dram_write_energy = prefill_total.dram_write_energy + decode_total.dram_write_energy
        total.sram_read_energy = prefill_total.sram_read_energy + decode_total.sram_read_energy
        total.sram_write_energy = prefill_total.sram_write_energy + decode_total.sram_write_energy

        # Weighted average utilization
        weighted_util = (prefill_total.utilization * prefill_total.cycles + 
                        decode_total.utilization * decode_total.cycles)
        total.utilization = weighted_util / total.cycles if total.cycles > 0 else 0.0
        
        return total
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "prefill": self._phase_to_dict(self.prefill),
            "decode": self._phase_to_dict(self.decode),
            "total": self._metrics_to_dict(self.get_total_metrics())
        }
    
    def _phase_to_dict(self, phase: PhaseMetrics) -> dict:
        """Convert phase metrics to dictionary"""
        aa_ops_dict = {}
        for op_type, op_list in phase.aa_ops.items():
            aa_ops_dict[op_type.value] = {
                "executions": [self._metrics_to_dict(m) for m in op_list],
                "total": self._metrics_to_dict(phase.get_operation_total(op_type, ComputeMode.AA)),
                "count": len(op_list)
            }
        
        aw_ops_dict = {}
        for op_type, op_list in phase.aw_ops.items():
            aw_ops_dict[op_type.value] = {
                "executions": [self._metrics_to_dict(m) for m in op_list],
                "total": self._metrics_to_dict(phase.get_operation_total(op_type, ComputeMode.AW)),
                "count": len(op_list)
            }
        
        return {
            "aa_operations": aa_ops_dict,
            "aw_operations": aw_ops_dict,
            "aa_total": self._metrics_to_dict(phase.get_aa_total()),
            "aw_total": self._metrics_to_dict(phase.get_aw_total()),
            "phase_total": self._metrics_to_dict(phase.get_total_metrics())
        }
    
    def _metrics_to_dict(self, metrics: OperationMetrics) -> dict:
        """Convert operation metrics to dictionary"""
        return {
            "shape": metrics.shape,
            "cycles": metrics.cycles,
            "flops": metrics.flops,
            "utilization": metrics.utilization,
            "throughput": metrics.throughput,
            "memory": {
                "dram_read": metrics.dram_read,
                "dram_write": metrics.dram_write,
                "sram_read": metrics.sram_read,
                "sram_write": metrics.sram_write
            },
            "energy": {
                "compute": metrics.compute_energy,
                "dram_read": metrics.dram_read_energy,
                "dram_write": metrics.dram_write_energy,
                "sram_read": metrics.sram_read_energy,
                "sram_write": metrics.sram_write_energy,
                "dram_total": metrics.dram_energy,
                "sram_total": metrics.sram_energy,
                "total": metrics.total_energy
            }
        }


class Simulator:
    def __init__(self, hw_config):
        self.hw_config = hw_config
        self.MU = 4
        self.num_RAC = 32

    def simulate(self, model_config, workload_config) -> SimulationResults:
        """Main simulation entry point"""
        # Initialize result containers
        prefill_metrics = PhaseMetrics(phase=Phase.PREFILL)
        decode_metrics = PhaseMetrics(phase=Phase.DECODE)
        
        # Simulate prefill phase
        self._simulate_prefill(model_config, workload_config, prefill_metrics)
        
        # Simulate decode phase
        self._simulate_decode(model_config, workload_config, decode_metrics)
        
        return SimulationResults(prefill=prefill_metrics, decode=decode_metrics)
    
    def _simulate_prefill(self, model_config, workload_config, metrics: PhaseMetrics):
        """Simulate prefill phase"""
        num_layers = model_config["num_layers"]
        batch_size = workload_config["batch_size"]
        seq_len = workload_config["input_tokens"]
        d_model = model_config["d_model"]
        d_ffn = model_config["d_ffn"]
        num_heads = model_config["num_heads"]
        head_dim = model_config["head_dim"]
        
        # Simulate one layer, then multiply by num_layers
        layer_idx = 0
        
        # 1. Q/K/V Projections (AW)
        q_shape = (batch_size * seq_len, d_model, d_model)
        q_metrics = self._simulate_single_operation(
            OperationType.Q_PROJ, ComputeMode.AW, q_shape, layer_idx, token_idx=-1, 
            batch_size=1, is_decode=False, seq_len=seq_len
        )
        for _ in range(num_layers):
            metrics.add_operation(OperationType.Q_PROJ, ComputeMode.AW, q_metrics)
        
        k_shape = (batch_size * seq_len, d_model, d_model)
        k_metrics = self._simulate_single_operation(
            OperationType.K_PROJ, ComputeMode.AW, k_shape, layer_idx, token_idx=-1, 
            batch_size=1, is_decode=False, seq_len=seq_len
        )
        for _ in range(num_layers):
            metrics.add_operation(OperationType.K_PROJ, ComputeMode.AW, k_metrics)
        
        v_shape = (batch_size * seq_len, d_model, d_model)
        v_metrics = self._simulate_single_operation(
            OperationType.V_PROJ, ComputeMode.AW, v_shape, layer_idx, token_idx=-1, 
            batch_size=1, is_decode=False, seq_len=seq_len
        )
        for _ in range(num_layers):
            metrics.add_operation(OperationType.V_PROJ, ComputeMode.AW, v_metrics)
        
        # 2. QK Matmul (AA)
        qk_shape = (seq_len, head_dim, seq_len)
        qk_metrics = self._simulate_single_operation(
            OperationType.QK_MATMUL, ComputeMode.AA, qk_shape, layer_idx, token_idx=-1,
            batch_size=batch_size * num_heads, is_decode=False, seq_len=seq_len
        )
        for _ in range(num_layers):
            metrics.add_operation(OperationType.QK_MATMUL, ComputeMode.AA, qk_metrics)
        
        # 3. Attention * V (AA)
        attn_v_shape = (seq_len, seq_len, head_dim)
        attn_v_metrics = self._simulate_single_operation(
            OperationType.ATTN_V_MATMUL, ComputeMode.AA, attn_v_shape, layer_idx, token_idx=-1,
            batch_size=batch_size * num_heads, is_decode=False, seq_len=seq_len
        )
        for _ in range(num_layers):
            metrics.add_operation(OperationType.ATTN_V_MATMUL, ComputeMode.AA, attn_v_metrics)
        
        # 4. O Projection (AW)
        o_shape = (batch_size * seq_len, d_model, d_model)
        o_metrics = self._simulate_single_operation(
            OperationType.O_PROJ, ComputeMode.AW, o_shape, layer_idx, token_idx=-1,
            batch_size=1, is_decode=False, seq_len=seq_len
        )
        for _ in range(num_layers):
            metrics.add_operation(OperationType.O_PROJ, ComputeMode.AW, o_metrics)
        
        # 5. FFN FC1 (AW)
        fc1_shape = (batch_size * seq_len, d_model, d_ffn)
        fc1_metrics = self._simulate_single_operation(
            OperationType.FC1, ComputeMode.AW, fc1_shape, layer_idx, token_idx=-1,
            batch_size=1, is_decode=False, seq_len=seq_len
        )
        for _ in range(num_layers):
            metrics.add_operation(OperationType.FC1, ComputeMode.AW, fc1_metrics)
        
        # 6. FFN FC2 (AW)
        fc2_shape = (batch_size * seq_len, d_ffn, d_model)
        fc2_metrics = self._simulate_single_operation(
            OperationType.FC2, ComputeMode.AW, fc2_shape, layer_idx, token_idx=-1,
            batch_size=1, is_decode=False, seq_len=seq_len
        )
        for _ in range(num_layers):
            metrics.add_operation(OperationType.FC2, ComputeMode.AW, fc2_metrics)
    
    def _simulate_decode(self, model_config, workload_config, metrics: PhaseMetrics):
        """Simulate decode phase"""
        num_layers = model_config["num_layers"]
        batch_size = workload_config["batch_size"]
        num_output_tokens = workload_config["output_tokens"]
        prefill_len = workload_config["input_tokens"]
        d_model = model_config["d_model"]
        d_ffn = model_config["d_ffn"]
        num_heads = model_config["num_heads"]
        head_dim = model_config["head_dim"]
        
        layer_idx = 0
        
        # For each output token
        for token_idx in range(1, num_output_tokens):
            current_kv_len = prefill_len + token_idx
            
            # 1. Q/K/V Projections (AW)
            q_shape = (batch_size * 1, d_model, d_model)
            q_metrics = self._simulate_single_operation(
                OperationType.Q_PROJ, ComputeMode.AW, q_shape, layer_idx,
                token_idx=token_idx, batch_size=1, is_decode=True,
                seq_len=prefill_len, kv_len=current_kv_len
            )
            for _ in range(num_layers):
                metrics.add_operation(OperationType.Q_PROJ, ComputeMode.AW, q_metrics)
            
            k_shape = (batch_size * 1, d_model, d_model)
            k_metrics = self._simulate_single_operation(
                OperationType.K_PROJ, ComputeMode.AW, k_shape, layer_idx,
                token_idx=token_idx, batch_size=1, is_decode=True,
                seq_len=prefill_len, kv_len=current_kv_len
            )
            for _ in range(num_layers):
                metrics.add_operation(OperationType.K_PROJ, ComputeMode.AW, k_metrics)
            
            v_shape = (batch_size * 1, d_model, d_model)
            v_metrics = self._simulate_single_operation(
                OperationType.V_PROJ, ComputeMode.AW, v_shape, layer_idx,
                token_idx=token_idx, batch_size=1, is_decode=True,
                seq_len=prefill_len, kv_len=current_kv_len
            )
            for _ in range(num_layers):
                metrics.add_operation(OperationType.V_PROJ, ComputeMode.AW, v_metrics)
            
            # 2. QK Matmul (AA)
            qk_shape = (1, head_dim, current_kv_len)
            qk_metrics = self._simulate_single_operation(
                OperationType.QK_MATMUL, ComputeMode.AA, qk_shape, layer_idx,
                token_idx=token_idx, batch_size=batch_size * num_heads, is_decode=True,
                seq_len=prefill_len, kv_len=current_kv_len
            )
            for _ in range(num_layers):
                metrics.add_operation(OperationType.QK_MATMUL, ComputeMode.AA, qk_metrics)
            
            # 3. Attention * V (AA)
            attn_v_shape = (1, current_kv_len, head_dim)
            attn_v_metrics = self._simulate_single_operation(
                OperationType.ATTN_V_MATMUL, ComputeMode.AA, attn_v_shape, layer_idx,
                token_idx=token_idx, batch_size=batch_size * num_heads, is_decode=True,
                seq_len=prefill_len, kv_len=current_kv_len
            )
            for _ in range(num_layers):
                metrics.add_operation(OperationType.ATTN_V_MATMUL, ComputeMode.AA, attn_v_metrics)
            
            # 4. O Projection (AW)
            o_shape = (batch_size * 1, d_model, d_model)
            o_metrics = self._simulate_single_operation(
                OperationType.O_PROJ, ComputeMode.AW, o_shape, layer_idx,
                token_idx=token_idx, batch_size=1, is_decode=True,
                seq_len=prefill_len, kv_len=current_kv_len
            )
            for _ in range(num_layers):
                metrics.add_operation(OperationType.O_PROJ, ComputeMode.AW, o_metrics)
            
            # 5. FFN FC1 (AW)
            fc1_shape = (batch_size * 1, d_model, d_ffn)
            fc1_metrics = self._simulate_single_operation(
                OperationType.FC1, ComputeMode.AW, fc1_shape, layer_idx,
                token_idx=token_idx, batch_size=1, is_decode=True,
                seq_len=prefill_len, kv_len=current_kv_len
            )
            for _ in range(num_layers):
                metrics.add_operation(OperationType.FC1, ComputeMode.AW, fc1_metrics)
            
            # 6. FFN FC2 (AW)
            fc2_shape = (batch_size * 1, d_ffn, d_model)
            fc2_metrics = self._simulate_single_operation(
                OperationType.FC2, ComputeMode.AW, fc2_shape, layer_idx,
                token_idx=token_idx, batch_size=1, is_decode=True,
                seq_len=prefill_len, kv_len=current_kv_len
            )
            for _ in range(num_layers):
                metrics.add_operation(OperationType.FC2, ComputeMode.AW, fc2_metrics)
    
    def _simulate_single_operation(self, op_type: OperationType, compute_mode: ComputeMode, 
                                   shape: tuple, layer_idx: int, token_idx: int = -1, 
                                   batch_size: int = 1, is_decode: bool = False,
                                   seq_len: int = 0, kv_len: int = 0) -> OperationMetrics:
        """Simulate a single operation execution and return its metrics"""
        M, K, N = shape
        
        metrics = OperationMetrics(shape=shape)

        metrics.flops = 2 * M * K * N * batch_size

        # Get mode from hw_config
        mode = self.hw_config["AW_mode"] if compute_mode == ComputeMode.AW else self.hw_config["AA_mode"]
        ori_mode = mode
        if self.hw_config["AW_mode"] == "OMNI":
            if is_decode:
                mode = "LUT_OS_V"
            else:
                mode = "LUT_WS"
        
        # 1. Calculate cycles
        qbit = self.hw_config["weight_bits"] if compute_mode == ComputeMode.AW else self.hw_config["kv_cache_bits"]
        metrics.cycles = self._calculate_cycles(M, K, N, qbit, compute_mode, mode, batch_size)

        # 2. Calculate memory access
        mem_access = self._calculate_memory_access(
            M, K, N, compute_mode, op_type, mode, batch_size,
            is_decode=is_decode, seq_len=seq_len, kv_len=kv_len
        )
        metrics.dram_read = mem_access["dram_read"]
        metrics.dram_write = mem_access["dram_write"]
        metrics.sram_read = mem_access["sram_read"]
        metrics.sram_write = mem_access["sram_write"]
        
        # 3. Calculate utilization
        # lanes_equiv = (self.hw_config["array_m"] * self.hw_config["array_n"] * self.hw_config.get("replication", 1) * self.num_RAC * (self.MU / qbit))
        lanes_equiv = 4096
        metrics.utilization = (metrics.flops / 2) / (metrics.cycles * lanes_equiv) if metrics.cycles > 0 else 0

        # 4. Calculate throughput
        metrics.throughput = self._calculate_throughput(M, K, N, metrics.cycles)

        # 5. Calculate energy
        energy = self._calculate_memory_energy(metrics.cycles, mem_access)
        metrics.dram_read_energy = energy["dram_read"]
        metrics.dram_write_energy = energy["dram_write"]
        metrics.sram_read_energy = energy["sram_read"]
        metrics.sram_write_energy = energy["sram_write"]
        
        qbit = self.hw_config["weight_bits"] if compute_mode == ComputeMode.AW else self.hw_config["kv_cache_bits"]
        if ori_mode == "OMNI":
            omni_mode = "OS" if is_decode else "WS"
            metrics.compute_energy = omni_energy_model.omni_compute_energy(array_m=self.hw_config["array_m"],
                                                                      array_n=self.hw_config["array_n"],
                                                                      M=M, K=K, N=N, batch_size=batch_size,
                                                                      qbit=qbit, mode=omni_mode)
        elif mode == "LUT_WS":
            metrics.compute_energy = ws_energy_model.ws_compute_energy(array_m=self.hw_config["array_m"],
                                                                      array_n=self.hw_config["array_n"],
                                                                      M=M, K=K, N=N, batch_size=batch_size,
                                                                      qbit=qbit)
        elif mode == "FPE_OS":
            fpe_array_size = self.hw_config.get("FPE_array_size", 64)
            metrics.compute_energy = fpe_os_energy_model.fpe_os_compute_energy(
                fpe_array_size=fpe_array_size, M=M, K=K, N=N, batch_size=batch_size
            )
        elif mode == "TENDER":
            fpe_array_size = self.hw_config.get("FPE_array_size", 64)
            metrics.compute_energy = tender_energy_model.tender_compute_energy(
                fpe_array_size=fpe_array_size, M=M, K=K, N=N, batch_size=batch_size
            )
        elif mode == "LUT_OS":
            metrics.compute_energy = os_energy_model.os_compute_energy(array_m=self.hw_config["array_m"],
                                                                      array_n=self.hw_config["array_n"],
                                                                      M=M, K=K, N=N, batch_size=batch_size,
                                                                      qbit=qbit)
        elif mode == "LUT_OS_V":
            metrics.compute_energy = os_v_energy_model.os_v_compute_energy(array_m=self.hw_config["array_m"],
                                                                          array_n=self.hw_config["array_n"],
                                                                          M=M, K=K, N=N, batch_size=batch_size,
                                                                          qbit=qbit)
        else:
            metrics.compute_energy = 0.0

        return metrics

    def _calculate_cycles(self, M: int, K: int, N: int, qbit: int, compute_mode: ComputeMode,
                         mode: str, batch_size: int) -> int:
        """Calculate operation cycles based on hw_config, compute_mode, and mode
        
        Args:
            M, K, N: Matrix dimensions (M x K) @ (K x N)
            compute_mode: AA or AW
            mode: e.g., "LUT_OS", "LUT_OS_V", "LUT_WS", "LUT_AS", "LUT_AS_V", "FPE_OS", "VPU"
            batch_size: Batch size (important for AW vs AA)
                        - AW: batch can share weights, M includes batch dimension
                        - AA: each batch has different activations
        """
        # TODO: Implement cycle calculation based on mode
        # For AW operations: batch_size helps determine weight reuse
        # For AA operations: batch_size means independent computations
        
        array_m = self.hw_config["array_m"]
        array_n = self.hw_config["array_n"]
        fpe_array_size = self.hw_config.get("FPE_array_size", 64)
        replication = self.hw_config.get("replication", 1)
        
        
        # Example structure (needs actual implementation):
        if mode == "LUT_OS":
            # Output stationary dataflow
            k_eff = math.ceil(K / self.MU)
            m_tiles = math.ceil(M / array_m)
            n_tiles = math.ceil(N / (array_n * self.num_RAC))

            LUT_gen_cycles = 3
            output_cycles = 2
            
            if M == 1:
                cycles_per_round = LUT_gen_cycles + k_eff + 1 + array_n + output_cycles
            else:
                cycles_per_round = LUT_gen_cycles + k_eff + array_m + array_n + output_cycles
            rounds_per_bit = math.ceil(m_tiles * n_tiles / replication)
            total_cycles = batch_size * cycles_per_round * rounds_per_bit * qbit
            
            return total_cycles
        
        elif mode == "LUT_OS_V":
            k_eff = math.ceil(K / self.MU)
            m_tiles = math.ceil(M / array_m)
            n_tiles = math.ceil(N / (array_n * self.num_RAC))
            
            LUT_gen_cycles = 3
            output_cycles = 2

            if M == 1:
                cycles_per_round = LUT_gen_cycles + k_eff + 1 + array_n + output_cycles
                rounds_per_bit = math.ceil(n_tiles / array_m / replication)
            else:
                cycles_per_round = LUT_gen_cycles + k_eff + array_m + array_n + output_cycles
                rounds_per_bit = math.ceil(m_tiles * n_tiles / replication)
                
            total_cycles = batch_size * cycles_per_round * rounds_per_bit * qbit
            return total_cycles

        elif mode == "LUT_WS":
            # Weight stationary dataflow
            k_eff = math.ceil(K / self.MU)
            k_tiles = math.ceil(k_eff / array_m)
            n_tiles = math.ceil(N / (array_n * self.num_RAC)) 
            
            input_cycles = 1
            output_cycles = 2
            
            cycles_per_round = M + array_n + array_m + input_cycles + output_cycles
            rounds_per_bit = math.ceil((n_tiles * k_tiles) / replication)
            
            total_cycles = batch_size * rounds_per_bit * cycles_per_round * qbit
            
            return total_cycles
        elif mode == "LUT_AS":
            # Activation stationary dataflow
            pass
        elif mode == "FPE_OS":
            m_tiles = math.ceil(M / fpe_array_size)
            n_tiles = math.ceil(N / fpe_array_size)
            
            input_cycles = 1
            output_cycles = 2
            
            if M == 1:
                cycles_per_tile = K + 1 + fpe_array_size + input_cycles + output_cycles
            else:
                cycles_per_tile = K + fpe_array_size + fpe_array_size + input_cycles + output_cycles
                
            total_cycles = batch_size * m_tiles * n_tiles * cycles_per_tile
            return total_cycles
        elif mode == "FPE_WS":
            k_tiles = math.ceil(K / fpe_array_size)
            n_tiles = math.ceil(N / fpe_array_size) 
            
            input_cycles = 1
            output_cycles = 2
            
            cycles_per_tile = M + fpe_array_size + fpe_array_size + input_cycles + output_cycles
            
            total_cycles = batch_size * k_tiles * n_tiles * cycles_per_tile
            return total_cycles
        elif mode == "VPU":
            total_cycles = (M * K * N * batch_size) // 64
            return total_cycles
        elif mode == "TENDER":
            m_tiles = math.ceil(M / fpe_array_size)
            n_tiles = math.ceil(N / fpe_array_size)
            
            LUT_gen_cycles = 3
            output_cycles = 2

            input_cycles = 1
            output_cycles = 2
            cycles_per_tile = K + fpe_array_size + fpe_array_size + input_cycles + output_cycles
            cycles_per_tile += 16 # Rescale
            
            total_cycles = batch_size * m_tiles * n_tiles * cycles_per_tile
            return total_cycles
        
        return 0
    
    def _calculate_memory_access(self, M: int, K: int, N: int, 
                                 compute_mode: ComputeMode, op_type: OperationType,
                                 mode: str, batch_size: int,
                                 is_decode: bool = False, seq_len: int = 0, kv_len: int = 0) -> dict:
        """Calculate memory access (DRAM/SRAM read/write)
        
        Args:
            M, K, N: Matrix dimensions
            compute_mode: AA or AW
            op_type: Operation type
            mode: Dataflow mode (LUT_OS, etc.)
            batch_size: Batch size
                        - AW: Can reuse weights across batch
                        - AA: No weight reuse, both operands are activations
        """

        act_bits = self.hw_config["act_bits"]
        accumulate_bits = self.hw_config.get("accumulate_bits", act_bits)
        weight_bits = self.hw_config["weight_bits"]
        kv_cache_bits = self.hw_config.get("kv_cache_bits", act_bits)
        
        array_m = self.hw_config["array_m"]
        array_n = self.hw_config["array_n"]
        fpe_array_size = self.hw_config.get("FPE_array_size", 64)
        
        if op_type in [OperationType.K_PROJ, OperationType.V_PROJ]:
            write_dram_bits = kv_cache_bits
        else:
            write_dram_bits = 0

        # A: activations (M x K), B: weights or activations (K x N), C: outputs (M x N)
        A_bits = batch_size * M * K * act_bits
        B_bits = batch_size * K * N * (kv_cache_bits if compute_mode == ComputeMode.AA else weight_bits)
        C_sram_bits = batch_size * M * N * act_bits
        C_sram_accum_bits = batch_size * M * N * accumulate_bits
        C_dram_bits = batch_size * M * N * write_dram_bits

        dram_read_bits = 0
        dram_write_bits = C_dram_bits
        sram_read_bits = 0
        sram_write_bits = 0
        
        qbit = weight_bits if compute_mode == ComputeMode.AW else kv_cache_bits
        # qbit = 1
        
        if mode == "LUT_OS" or mode == "LUT_OS_V":
            m_tiles = math.ceil(M / array_m)
            n_tiles = math.ceil(N / (array_n * self.num_RAC))
            
            sram_read_bits = A_bits * n_tiles * qbit + B_bits * m_tiles
            sram_write_bits = C_sram_accum_bits
        elif mode == "LUT_WS":
            k_eff = math.ceil(K / self.MU)
            k_tiles = math.ceil(k_eff / array_m)
            n_tiles = math.ceil(N / (array_n * self.num_RAC))
             
            sram_read_bits  = A_bits * n_tiles * qbit + B_bits + C_sram_accum_bits * (k_tiles - 1) * qbit
            sram_write_bits = C_sram_accum_bits * k_tiles * qbit
        elif mode == "FPE_OS":
            m_tiles = math.ceil(M / fpe_array_size)
            n_tiles = math.ceil(N / fpe_array_size)
            
            B_bits_fpe = batch_size * K * N * 16  # FPE uses 16 bits for compute
            sram_read_bits = A_bits * n_tiles + B_bits_fpe * m_tiles
            sram_write_bits = C_sram_accum_bits
        elif mode == "TENDER":
            m_tiles = math.ceil(M / fpe_array_size)
            n_tiles = math.ceil(N / fpe_array_size)
            
            sram_read_bits = A_bits * n_tiles + B_bits * m_tiles
            sram_write_bits = C_sram_accum_bits

        # Decode-time KV cache reads (bits) — simple heuristic using provided formula
        if is_decode and op_type in [OperationType.QK_MATMUL, OperationType.ATTN_V_MATMUL] and kv_len > 0:
            if op_type == OperationType.QK_MATMUL:
                # expected shape (1, head_dim, kv_len)
                head_dim = K
                kv_len_val = kv_len
            else:
                # ATTN_V_MATMUL expected shape (1, kv_len, head_dim)
                head_dim = N
                kv_len_val = kv_len

            kv_prev = max(0, kv_len_val - 1)
            dram_kv_read_bits = batch_size * kv_prev * head_dim * kv_cache_bits
            dram_read_bits += dram_kv_read_bits
        elif compute_mode == ComputeMode.AW:
            dram_read_bits = B_bits

        # Convert bits to bytes
        dram_read = dram_read_bits // 8
        dram_write = dram_write_bits // 8
        sram_read = sram_read_bits // 8
        sram_write = sram_write_bits // 8

        return {
            "dram_read": dram_read,
            "dram_write": dram_write,
            "sram_read": sram_read,
            "sram_write": sram_write
        }
    
    def _calculate_throughput(self, M: int, K: int, N: int, cycles: int) -> float:
        """Calculate throughput (TOPS)"""
        # TODO: Implement throughput calculation
        return 0.0
    
    def _calculate_memory_energy(self, cycles: int, mem_access: dict) -> dict:
        """Calculate energy consumption for compute and memory"""
        # TODO: Implement energy calculation based on hw_config
        dram_read_energy = dram_power_model.dram_energy(mem_access["dram_read"], is_write=False)
        dram_write_energy = dram_power_model.dram_energy(mem_access["dram_write"], is_write=True)
        sram_read_energy = sram_power_model.sram_energy(mem_access["sram_read"], is_write=False)
        sram_write_energy = sram_power_model.sram_energy(mem_access["sram_write"], is_write=True)

        return {
            "compute": 0.0,
            "dram_read": dram_read_energy,
            "dram_write": dram_write_energy,
            "sram_read": sram_read_energy,
            "sram_write": sram_write_energy
        }
    
    def save_results(self, results: SimulationResults, output_path: str):
        """Save simulation results to file in both JSON and human-readable format"""
        import os
        
        # Save JSON format
        json_path = output_path.replace(".txt", ".json") if output_path.endswith(".txt") else output_path + ".json"
        # with open(json_path, 'w') as f:
        #     json.dump(results.to_dict(), f, indent=2)
        
        # Save human-readable format
        txt_path = output_path.replace(".json", ".txt") if output_path.endswith(".json") else output_path + ".txt"
        with open(txt_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("SIMULATION RESULTS\n")
            f.write("=" * 80 + "\n\n")
            
            # Hardware configuration
            f.write("Hardware Configuration:\n")
            f.write("-" * 80 + "\n")
            for key, value in self.hw_config.items():
                f.write(f"  {key:20s}: {value}\n")
            f.write("\n")
            
            # Summary
            total = results.get_total_metrics()
            prefill_total = results.prefill.get_total_metrics()
            decode_total = results.decode.get_total_metrics()
            
            f.write("Overall Summary:\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Total Cycles:        {total.cycles:,}\n")
            f.write(f"  Total FLOPs:         {total.flops:,}\n")
            f.write(f"  Avg Utilization:     {total.utilization:.2%}\n")
            f.write(f"  Total DRAM Read:     {total.dram_read:,} bytes ({total.dram_read / (1024**3):.2f} GB)\n")
            f.write(f"  Total DRAM Write:    {total.dram_write:,} bytes ({total.dram_write / (1024**3):.2f} GB)\n")
            f.write(f"  Total SRAM Read:     {total.sram_read:,} bytes ({total.sram_read / (1024**3):.2f} GB)\n")
            f.write(f"  Total SRAM Write:    {total.sram_write:,} bytes ({total.sram_write / (1024**3):.2f} GB)\n")
            f.write(f"  Compute Energy:      {total.compute_energy:.2f} J\n")
            f.write(f"  DRAM Read Energy:    {total.dram_read_energy:.2f} J\n")
            f.write(f"  DRAM Write Energy:   {total.dram_write_energy:.2f} J\n")
            f.write(f"  DRAM Energy:         {total.dram_energy:.2f} J\n")
            f.write(f"  SRAM Read Energy:    {total.sram_read_energy:.2f} J\n")
            f.write(f"  SRAM Write Energy:   {total.sram_write_energy:.2f} J\n")
            f.write(f"  SRAM Energy:         {total.sram_energy:.2f} J\n")
            f.write(f"  Total Energy:        {total.total_energy:.2f} J\n")
            f.write("\n")
            
            # Prefill phase
            self._write_phase_summary(f, "PREFILL PHASE", results.prefill, prefill_total)
            
            # Decode phase
            self._write_phase_summary(f, "DECODE PHASE", results.decode, decode_total)
            
        print(f"Results saved to:\n  - {json_path}\n  - {txt_path}")
    
    def _write_phase_summary(self, f, title: str, phase: PhaseMetrics, phase_total: OperationMetrics):
        """Write phase summary to file"""
        f.write("=" * 80 + "\n")
        f.write(f"{title}\n")
        f.write("=" * 80 + "\n\n")
        
        # Phase totals
        f.write("Phase Totals:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Cycles:              {phase_total.cycles:,}\n")
        f.write(f"  FLOPs:               {phase_total.flops:,}\n")
        f.write(f"  Avg Utilization:     {phase_total.utilization:.2%}\n")
        f.write(f"  DRAM Read:           {phase_total.dram_read:,} bytes ({phase_total.dram_read / (1024**3):.2f} GB)\n")
        f.write(f"  DRAM Write:          {phase_total.dram_write:,} bytes ({phase_total.dram_write / (1024**3):.2f} GB)\n")
        f.write(f"  SRAM Read:           {phase_total.sram_read:,} bytes ({phase_total.sram_read / (1024**3):.2f} GB)\n")
        f.write(f"  SRAM Write:          {phase_total.sram_write:,} bytes ({phase_total.sram_write / (1024**3):.2f} GB)\n")
        f.write(f"  Compute Energy:      {phase_total.compute_energy:.2f} J\n")
        f.write(f"  DRAM Read Energy:    {phase_total.dram_read_energy:.2f} J\n")
        f.write(f"  DRAM Write Energy:   {phase_total.dram_write_energy:.2f} J\n")
        f.write(f"  DRAM Energy:         {phase_total.dram_energy:.2f} J\n")
        f.write(f"  SRAM Read Energy:    {phase_total.sram_read_energy:.2f} J\n")
        f.write(f"  SRAM Write Energy:   {phase_total.sram_write_energy:.2f} J\n")
        f.write(f"  SRAM Energy:         {phase_total.sram_energy:.2f} J\n")
        f.write(f"  Total Energy:        {phase_total.total_energy:.2f} J\n")
        f.write("\n")
        
        # AA operations
        aa_total = phase.get_aa_total()
        f.write("AA Operations Summary:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Total Cycles:        {aa_total.cycles:,}\n")
        f.write(f"  Total FLOPs:         {aa_total.flops:,}\n")
        f.write(f"  Avg Utilization:     {aa_total.utilization:.2%}\n")
        f.write(f"  DRAM Read:           {aa_total.dram_read:,} bytes ({aa_total.dram_read / (1024**3):.2f} GB)\n")
        f.write(f"  DRAM Write:          {aa_total.dram_write:,} bytes ({aa_total.dram_write / (1024**3):.2f} GB)\n")
        f.write(f"  SRAM Read:           {aa_total.sram_read:,} bytes ({aa_total.sram_read / (1024**3):.2f} GB)\n")
        f.write(f"  SRAM Write:          {aa_total.sram_write:,} bytes ({aa_total.sram_write / (1024**3):.2f} GB)\n")
        f.write(f"  Compute Energy:      {aa_total.compute_energy:.2f} J\n")
        f.write(f"  DRAM Read Energy:    {aa_total.dram_read_energy:.2f} J\n")
        f.write(f"  DRAM Write Energy:   {aa_total.dram_write_energy:.2f} J\n")
        f.write(f"  DRAM Energy:         {aa_total.dram_energy:.2f} J\n")
        f.write(f"  SRAM Read Energy:    {aa_total.sram_read_energy:.2f} J\n")
        f.write(f"  SRAM Write Energy:   {aa_total.sram_write_energy:.2f} J\n")
        f.write(f"  SRAM Energy:         {aa_total.sram_energy:.2f} J\n")
        f.write(f"  Total Energy:        {aa_total.total_energy:.2f} J\n")
        f.write("\n")
        
        for op_type, op_list in phase.aa_ops.items():
            op_total = phase.get_operation_total(op_type, ComputeMode.AA)
            # first execution shape
            first_shape = op_list[0].shape if len(op_list) > 0 else None
            f.write(f"  {op_type.value:20s}:\n")
            f.write(f"    First exec shape:   {first_shape}\n")
            f.write(f"    Executions:        {len(op_list):6d}\n")
            f.write(f"    Cycles:            {op_total.cycles:12,}\n")
            f.write(f"    FLOPs:             {op_total.flops:12,}\n")
            f.write(f"    Avg Utilization:   {op_total.utilization:6.2%}\n")
            f.write(f"    DRAM Read:         {op_total.dram_read:12,} bytes ({op_total.dram_read / (1024**2):8.2f} MB)\n")
            f.write(f"    DRAM Write:        {op_total.dram_write:12,} bytes ({op_total.dram_write / (1024**2):8.2f} MB)\n")
            f.write(f"    SRAM Read:         {op_total.sram_read:12,} bytes ({op_total.sram_read / (1024**2):8.2f} MB)\n")
            f.write(f"    SRAM Write:        {op_total.sram_write:12,} bytes ({op_total.sram_write / (1024**2):8.2f} MB)\n")
            f.write(f"    Compute Energy:    {op_total.compute_energy:12.2f} J\n")
            f.write(f"    DRAM Read Energy:  {op_total.dram_read_energy:12.2f} J\n")
            f.write(f"    DRAM Write Energy: {op_total.dram_write_energy:12.2f} J\n")
            f.write(f"    DRAM Energy:       {op_total.dram_energy:12.2f} J\n")
            f.write(f"    SRAM Read Energy:  {op_total.sram_read_energy:12.2f} J\n")
            f.write(f"    SRAM Write Energy: {op_total.sram_write_energy:12.2f} J\n")
            f.write(f"    SRAM Energy:       {op_total.sram_energy:12.2f} J\n")
            f.write(f"    Total Energy:      {op_total.total_energy:12.2f} J\n")
        f.write("\n")
        
        # AW operations
        aw_total = phase.get_aw_total()
        f.write("AW Operations Summary:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Total Cycles:        {aw_total.cycles:,}\n")
        f.write(f"  Total FLOPs:         {aw_total.flops:,}\n")
        f.write(f"  Avg Utilization:     {aw_total.utilization:.2%}\n")
        f.write(f"  DRAM Read:           {aw_total.dram_read:,} bytes ({aw_total.dram_read / (1024**3):.2f} GB)\n")
        f.write(f"  DRAM Write:          {aw_total.dram_write:,} bytes ({aw_total.dram_write / (1024**3):.2f} GB)\n")
        f.write(f"  SRAM Read:           {aw_total.sram_read:,} bytes ({aw_total.sram_read / (1024**3):.2f} GB)\n")
        f.write(f"  SRAM Write:          {aw_total.sram_write:,} bytes ({aw_total.sram_write / (1024**3):.2f} GB)\n")
        f.write(f"  Compute Energy:      {aw_total.compute_energy:.2f} J\n")
        f.write(f"  DRAM Read Energy:    {aw_total.dram_read_energy:.2f} J\n")
        f.write(f"  DRAM Write Energy:   {aw_total.dram_write_energy:.2f} J\n")
        f.write(f"  DRAM Energy:         {aw_total.dram_energy:.2f} J\n")
        f.write(f"  SRAM Read Energy:    {aw_total.sram_read_energy:.2f} J\n")
        f.write(f"  SRAM Write Energy:   {aw_total.sram_write_energy:.2f} J\n")
        f.write(f"  SRAM Energy:         {aw_total.sram_energy:.2f} J\n")
        f.write(f"  Total Energy:        {aw_total.total_energy:.2f} J\n")
        f.write("\n")
        
        for op_type, op_list in phase.aw_ops.items():
            op_total = phase.get_operation_total(op_type, ComputeMode.AW)
            # first execution shape
            first_shape = op_list[0].shape if len(op_list) > 0 else None
            f.write(f"  {op_type.value:20s}:\n")
            f.write(f"    First exec shape:   {first_shape}\n")
            f.write(f"    Executions:        {len(op_list):6d}\n")
            f.write(f"    Cycles:            {op_total.cycles:12,}\n")
            f.write(f"    FLOPs:             {op_total.flops:12,}\n")
            f.write(f"    Avg Utilization:   {op_total.utilization:6.2%}\n")
            f.write(f"    DRAM Read:         {op_total.dram_read:12,} bytes ({op_total.dram_read / (1024**2):8.2f} MB)\n")
            f.write(f"    DRAM Write:        {op_total.dram_write:12,} bytes ({op_total.dram_write / (1024**2):8.2f} MB)\n")
            f.write(f"    SRAM Read:         {op_total.sram_read:12,} bytes ({op_total.sram_read / (1024**2):8.2f} MB)\n")
            f.write(f"    SRAM Write:        {op_total.sram_write:12,} bytes ({op_total.sram_write / (1024**2):8.2f} MB)\n")
            f.write(f"    Compute Energy:    {op_total.compute_energy:12.2f} J\n")
            f.write(f"    DRAM Read Energy:  {op_total.dram_read_energy:12.2f} J\n")
            f.write(f"    DRAM Write Energy: {op_total.dram_write_energy:12.2f} J\n")
            f.write(f"    DRAM Energy:       {op_total.dram_energy:12.2f} J\n")
            f.write(f"    SRAM Read Energy:  {op_total.sram_read_energy:12.2f} J\n")
            f.write(f"    SRAM Write Energy: {op_total.sram_write_energy:12.2f} J\n")
            f.write(f"    SRAM Energy:       {op_total.sram_energy:12.2f} J\n")
            f.write(f"    Total Energy:      {op_total.total_energy:12.2f} J\n")
        f.write("\n")


def main():
    # Example: OPT-1.3B model configuration
    # model_config = {
    #     "num_layers": 24,
    #     "num_heads": 32,
    #     "num_kv_heads": 32,  # MHA (same as num_heads)
    #     "d_model": 2048,
    #     "d_ffn": 8192,
    #     "head_dim": 64,  # 2048 / 32
    # }
    
    # OPT-6.7B
    model_config = {
        "num_layers": 32,
        "num_heads": 32,
        "num_kv_heads": 32,  # MHA (same as num_heads)
        "d_model": 4096,
        "d_ffn": 16384,
        "head_dim": 128,  # 4096 / 32
    }
    
    workload_config = {
        "batch_size": 1,
        "input_tokens": 16384,  # Prefill length
        "output_tokens": 512,  # Number of tokens to generate
    }
    
    # FIGLUT + VPU
    # hw_config = {
    #     "array_m": 16,
    #     "array_n": 2,
    #     "replication": 4, # LUT only
    #     "mu": 4,          # LUT only
    #     "RAC": 32,        # LUT only
    #     "freq_mhz": 1000,
    #     "act_bits": 16,
    #     "accumulate_bits": 32,
    #     "weight_bits": 4,
    #     "kv_cache_bits": 16,
    #     "AW_mode": "LUT_WS",  # VPU, FPE_OS, LUT_OS, LUT_OS_V, LUT_WS, LUT_AS, LUT_AS_V
    #     "AA_mode": "VPU",
    # }
    
    # FIGLUT + FPE
    # hw_config = {
    #     "array_m": 32,
    #     "array_n": 4,
    #     "replication": 1, # LUT only
    #     "mu": 4,          # LUT only
    #     "RAC": 32,        # LUT only
    #     "FPE_array_size": 64,  # FPE only
    #     "freq_mhz": 1000,
    #     "act_bits": 16,
    #     "accumulate_bits": 32,
    #     "weight_bits": 4,
    #     "kv_cache_bits": 16,
    #     "AW_mode": "LUT_OS",  # VPU, FPE_OS, LUT_OS, LUT_OS_V, LUT_WS, LUT_AS, LUT_AS_V
    #     "AA_mode": "FPE_OS",
    # }    

    # FPE
    # hw_config = {
    #     "array_m": 64,
    #     "array_n": 64,
    #     "FPE_array_size": 64,  # FPE only
    #     "replication": 1, # LUT only
    #     "mu": 4,          # LUT only
    #     "RAC": 32,        # LUT only
    #     "freq_mhz": 1000,
    #     "act_bits": 16,
    #     "accumulate_bits": 32,
    #     "weight_bits": 4,
    #     "kv_cache_bits": 16,
    #     "AW_mode": "FPE_OS",  # VPU, FPE_OS, LUT_OS, LUT_OS_V, LUT_WS, LUT_AS, LUT_AS_V
    #     "AA_mode": "FPE_OS",
    # }

    # OMNI
    # hw_config = {
    #     "array_m": 32,
    #     "array_n": 4,
    #     "replication": 1, # LUT only
    #     "mu": 4,          # LUT only
    #     "RAC": 32,        # LUT only
    #     "freq_mhz": 1000,
    #     "act_bits": 16,
    #     "accumulate_bits": 32,
    #     "weight_bits": 4,
    #     "kv_cache_bits": 4,
    #     "AW_mode": "OMNI",  # VPU, FPE_OS, LUT_OS, LUT_OS_V, LUT_WS, LUT_AS, LUT_AS_V
    #     "AA_mode": "OMNI",
    # }

    # OMNI_OS
    hw_config = {
        "array_m": 32,
        "array_n": 4,
        "replication": 1, # LUT only
        "mu": 4,          # LUT only
        "RAC": 32,        # LUT only
        "freq_mhz": 1000,
        "act_bits": 16,
        "accumulate_bits": 32,
        "weight_bits": 4,
        "kv_cache_bits": 4,
        "AW_mode": "OMNI",  # VPU, FPE_OS, LUT_OS, LUT_OS_V, LUT_WS, LUT_AS, LUT_AS_V
        "AA_mode": "OMNI",
    }
    # hw_config = {
    #     "array_m": 32,
    #     "array_n": 4,
    #     "replication": 1, # LUT only
    #     "mu": 4,          # LUT only
    #     "RAC": 32,        # LUT only
    #     "freq_mhz": 1000,
    #     "act_bits": 16,
    #     "accumulate_bits": 32,
    #     "weight_bits": 4,
    #     "kv_cache_bits": 4,
    #     "AW_mode": "LUT_OS",  # VPU, FPE_OS, LUT_OS, LUT_OS_V, LUT_WS, LUT_AS, LUT_AS_V
    #     "AA_mode": "LUT_OS",
    # }

    # hw_config = {
    #     "array_m": 64,
    #     "array_n": 64,
    #     "fpe_array_size": 64,  # FPE/TENDER only
    #     "replication": 1, # LUT only
    #     "mu": 4,          # LUT only
    #     "RAC": 32,        # LUT only
    #     "freq_mhz": 1000,
    #     "act_bits": 8,
    #     "accumulate_bits": 32,
    #     "weight_bits": 8,
    #     "kv_cache_bits": 8,
    #     "AW_mode": "TENDER",  # VPU, FPE_OS, LUT_OS, LUT_OS_V, LUT_WS, LUT_AS, LUT_AS_V
    #     "AA_mode": "TENDER",
    # }
    
    simulator = Simulator(hw_config)
    results = simulator.simulate(model_config, workload_config)

    # Save results
    output_path = "simulation_results"
    simulator.save_results(results, output_path)
    
    print("\nSimulation completed successfully!")


if __name__ == "__main__":
    main()
