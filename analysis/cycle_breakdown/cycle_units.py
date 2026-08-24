"""
Per-unit and per-stage cycle attribution for Omni-LUT, as an add-on layer.

`simulator/simulator.py` is left completely untouched.  Everything here is
either a pure function or a subclass, so the original code structure and all
existing analysis scripts keep working exactly as before.

Two views of the same cycle budget:

  * by pipeline STAGE  -- q_proj, qk_matmul, fc1, softmax, ...
    (`compute_stage_cycle_breakdown`)
  * by hardware UNIT   -- LGU, PE array, accumulator, VPU, BQU, ...
    (`compute_unit_cycle_breakdown`)

Usage:
    from cycle_units import UnitAwareSimulator, compute_unit_cycle_breakdown

    sim = UnitAwareSimulator(hw)              # drop-in for Simulator(hw)
    results = sim.simulate(model, workload)
    units = compute_unit_cycle_breakdown(sim, results, workload)
"""

import math
from dataclasses import dataclass
from typing import Dict, List

from simulator import (
    ComputeMode,
    ModelConfig,
    OperationMetrics,
    PhaseMetrics,
    SimulationResults,
    Simulator,
    WorkloadConfig,
    _aggregate_metrics,
    _aggregate_non_gemm_metrics,
)


# ============================================================================
# Cycle attribution per hardware unit
# ============================================================================

# Same constants as Simulator._calculate_cycles.
LUT_GEN_CYCLES = 3
OUTPUT_CYCLES = 2
INPUT_CYCLES = 1

# Units that run concurrently with the datapath and so do not add to latency.
OVERLAPPED_UNITS = ('bqu_tse', 'bqu_bea')


def cycle_units(hw, M: int, K: int, N: int, qbit: int, mode: str,
                batch_size: int, mu: int = Simulator.MU,
                num_rac: int = Simulator.NUM_RAC,
                pack: int = 1) -> Dict[str, int]:
    """Split the cost of A(M x K) @ B(K x N) across hardware units.

    This mirrors ``Simulator._calculate_cycles`` term by term; summing the
    result reproduces its single-number output exactly.  Each term is charged
    to the block of Fig. 4 that spends it:

      lgu                 LUT Generation Unit (Fig. 7): scaling stage + table
                          generation stage.  Only appears in output-stationary
                          modes -- in LUT_WS the table generation is pipelined
                          into the M-long activation stream and is therefore
                          fully amortized (the original formula has no
                          LUT_GEN term there).
      pe_array_compute    PE array streaming the useful operands.
      pe_array_fill_drain Systolic fill/drain overhead (array_m / array_n).
      fpe_array_*         The same two roles for the floating-point engine.
      rescale             Tender's per-tile rescaling stage.
      input_load          Operand issue into the array.
      accumulator         Final accumulator + write-back.
      vpu                 Vector unit (non-LUT elementwise path).

    ``pack`` packs that many independent OS-V instances into one pass, each
    driven by its own LGU broadcasting to ``array_m / pack`` rows instead of
    one LGU broadcasting to all 32.  It only bites in the ``LUT_OS_V`` M=1
    branch; ``pack=1`` is the original expression term for term, so every
    existing caller is unaffected.  See ``analysis/array_packing/``.
    """
    array_m = hw.array_m
    array_n = hw.array_n
    fpe_size = hw.FPE_array_size
    replication = hw.replication

    if mode in ("LUT_OS", "LUT_OS_V"):
        k_eff = math.ceil(K / mu)
        m_tiles = math.ceil(M / array_m)
        n_tiles = math.ceil(N / (array_n * num_rac))

        per_round = {
            'lgu': LUT_GEN_CYCLES,
            'pe_array_compute': k_eff,
            'pe_array_fill_drain': (1 + array_n) if M == 1 else (array_m + array_n),
            'accumulator': OUTPUT_CYCLES,
        }

        if mode == "LUT_OS_V" and M == 1:
            # Each of the `pack` instances gets its own LGU broadcasting to
            # array_m/pack rows; `pack` instances then retire per pass.
            rows_per_inst = max(1, array_m // pack)
            rounds = math.ceil(n_tiles / rows_per_inst / replication)
            eff_batch = math.ceil(batch_size / pack)
        else:
            rounds = math.ceil(m_tiles * n_tiles / replication)
            eff_batch = batch_size

        scale = eff_batch * rounds * qbit

    elif mode == "LUT_WS":
        k_eff = math.ceil(K / mu)
        k_tiles = math.ceil(k_eff / array_m)
        n_tiles = math.ceil(N / (array_n * num_rac))

        per_round = {
            'lgu': 0,   # pipelined into the M-long activation stream
            'pe_array_compute': M,
            'pe_array_fill_drain': array_n + array_m,
            'input_load': INPUT_CYCLES,
            'accumulator': OUTPUT_CYCLES,
        }
        scale = batch_size * math.ceil(n_tiles * k_tiles / replication) * qbit

    elif mode == "FPE_OS":
        per_round = {
            'fpe_array_compute': K,
            'fpe_array_fill_drain': (1 + fpe_size) if M == 1 else (2 * fpe_size),
            'input_load': INPUT_CYCLES,
            'accumulator': OUTPUT_CYCLES,
        }
        scale = batch_size * math.ceil(M / fpe_size) * math.ceil(N / fpe_size)

    elif mode == "TENDER":
        m1 = M == 1 and hw.tender_m1_opt
        per_round = {
            'fpe_array_compute': K,
            'fpe_array_fill_drain': (1 + fpe_size) if m1 else (2 * fpe_size),
            'input_load': INPUT_CYCLES,
            'accumulator': OUTPUT_CYCLES,
            'rescale': 16,
        }
        scale = batch_size * math.ceil(M / fpe_size) * math.ceil(N / fpe_size)

    elif mode == "FPE_WS":
        per_round = {
            'fpe_array_compute': M,
            'fpe_array_fill_drain': 2 * fpe_size,
            'input_load': INPUT_CYCLES,
            'accumulator': OUTPUT_CYCLES,
        }
        scale = batch_size * math.ceil(K / fpe_size) * math.ceil(N / fpe_size)

    elif mode == "VPU":
        return {'vpu': (M * K * N * batch_size) // 64}

    else:
        return {}

    return {unit: cycles * scale for unit, cycles in per_round.items()}


# ============================================================================
# BQU -- Binary-coding Quantization Unit (Sec. IV-B, Fig. 5)
# ============================================================================

@dataclass
class BQUMetrics:
    """Metrics for one BQU invocation converting a KV tensor into BCQ form.

    Two sub-blocks:
      TSE (Token-Scale Estimator)  -- pipelined min/max reduction tree over a
          token, yielding the per-token zero-point and step size.  Value path
          only; the Key path uses offline-calibrated per-channel scales.
      BEA (BCQ Encoder Array)      -- greedy residual encoder, one pass per
          bit-plane (Eq. 8-9), producing the q x d sign matrix.

    NOTE: the original simulator does not account for the BQU at all, so this
    is a model added on top.  Throughput is assumed `bqu_width` elements/cycle
    with the bit-plane loop serialized.  Energy is NOT modeled -- there is no
    BQU characterization data in the energy models to draw on.
    """
    tensor: str                 # "key" or "value"
    num_tokens: int = 0
    num_elements: int = 0
    bit_planes: int = 0
    tse_cycles: int = 0
    bea_cycles: int = 0

    @property
    def cycles(self) -> int:
        return self.tse_cycles + self.bea_cycles


def bqu_metrics(hw, tensor: str, num_tokens: int, d_kv: int) -> BQUMetrics:
    """Model the BQU converting one KV tensor of `num_tokens` x `d_kv`."""
    qbit = hw.kv_cache_bits
    total_elements = num_tokens * d_kv
    passes = math.ceil(total_elements / getattr(hw, 'bqu_width', 128))

    return BQUMetrics(
        tensor=tensor,
        num_tokens=num_tokens,
        num_elements=total_elements,
        bit_planes=qbit,
        # BEA: one greedy-residual pass per bit-plane (sign + subtract).
        bea_cycles=passes * qbit,
        # TSE: a single min/max reduction pass, Value path only.
        tse_cycles=passes if tensor == "value" else 0,
    )


# ============================================================================
# Unit-aware simulator
# ============================================================================

class UnitAwareSimulator(Simulator):
    """Drop-in `Simulator` that also records where each cycle was spent.

    Nothing in `simulator.py` changes.  Cycle totals, memory traffic, energy
    and latency are produced by the parent class exactly as before; this
    subclass only annotates each `OperationMetrics` with a `unit_cycles` dict
    and collects BQU records.

    Attributes:
        model_bqu: model the BQU (online KV quantization).  Auto-disabled for
            non-LUT dataflows, which have no BQU.
        bqu_width: BCQ Encoder Array throughput in elements per cycle.
    """

    def __init__(self, hw, model_bqu: bool = True, bqu_width: int = 128):
        super().__init__(hw)
        is_lut = any(m == 'OMNI' or m.startswith('LUT_')
                     for m in (hw.AW_mode, hw.AA_mode))
        self.model_bqu = model_bqu and is_lut
        self.bqu_width = bqu_width
        self._pending: List[Dict[str, int]] = []
        self.bqu_ops: Dict[str, List[BQUMetrics]] = {'prefill': [], 'decode': []}

    # ---- Cycle interception -------------------------------------------------

    def _calculate_cycles(self, M, K, N, qbit, compute_mode, mode, batch_size):
        """Record the unit split, then return the parent's total unchanged."""
        units = cycle_units(self.hw, M, K, N, qbit, mode, batch_size,
                            mu=self.MU, num_rac=self.NUM_RAC)
        self._pending.append(units)
        return super()._calculate_cycles(M, K, N, qbit, compute_mode,
                                         mode, batch_size)

    def _attach_units(self, metrics: OperationMetrics) -> OperationMetrics:
        """Fold the pending per-call splits into `metrics.unit_cycles`.

        FlashAttention calls `_calculate_cycles` once per tile shape and then
        multiplies by the tile count outside of it.  Rather than duplicating
        that tiling math, the scale factor is recovered from the ratio between
        the parent's reported total and the sum of the pending splits, so this
        stays correct if the tiling ever changes.
        """
        pending, self._pending = self._pending, []
        raw_total = sum(sum(u.values()) for u in pending)
        if raw_total <= 0:
            metrics.unit_cycles = {}
            return metrics

        scale = metrics.cycles / raw_total
        merged: Dict[str, float] = {}
        for units in pending:
            for unit, c in units.items():
                merged[unit] = merged.get(unit, 0.0) + c * scale

        # Round to integers, then push any residual onto the largest bucket so
        # the split still sums to exactly metrics.cycles.
        rounded = {u: int(round(c)) for u, c in merged.items()}
        residual = metrics.cycles - sum(rounded.values())
        if residual and rounded:
            biggest = max(rounded, key=lambda u: rounded[u])
            rounded[biggest] += residual

        metrics.unit_cycles = rounded
        return metrics

    def _simulate_matmul(self, *args, **kwargs) -> OperationMetrics:
        self._pending = []
        return self._attach_units(super()._simulate_matmul(*args, **kwargs))

    def _simulate_flash_attention(self, *args, **kwargs) -> OperationMetrics:
        self._pending = []
        return self._attach_units(
            super()._simulate_flash_attention(*args, **kwargs))

    # ---- BQU collection -----------------------------------------------------

    def _simulate_transformer_step(self, metrics: PhaseMetrics,
                                   model: ModelConfig, workload: WorkloadConfig,
                                   proj_m: int, *args, **kwargs):
        super()._simulate_transformer_step(metrics, model, workload,
                                           proj_m, *args, **kwargs)
        if not self.model_bqu:
            return

        # The new K and V of this step are quantized online, once per layer.
        # These cycles overlap the PE array, so they are kept out of the
        # phase's own metrics and never enter the serial latency.
        phase = 'decode' if metrics.phase.value == 'decode' else 'prefill'
        hw_view = _BQUView(self.hw, self.bqu_width)
        for tensor in ("key", "value"):
            rec = bqu_metrics(hw_view, tensor, num_tokens=proj_m,
                              d_kv=model.d_kv)
            self.bqu_ops[phase].extend([rec] * model.num_layers)


class _BQUView:
    """Adapter exposing `bqu_width` alongside the real HardwareConfig."""

    def __init__(self, hw, bqu_width):
        self._hw = hw
        self.bqu_width = bqu_width

    def __getattr__(self, name):
        return getattr(self._hw, name)


# ============================================================================
# Breakdowns
# ============================================================================

def compute_stage_cycle_breakdown(sim: Simulator, results: SimulationResults,
                                  workload: WorkloadConfig,
                                  dram_bandwidth_gbps: float | None = None) -> dict:
    """Break the cycle cost down per pipeline stage (per operation type).

    Finer-grained than `Simulator.compute_roofline_latency_breakdown`, which
    only splits AA / AW / non-GEMM.  Every GEMM operation type (q_proj,
    qk_matmul, fc1, ...) and every non-GEMM operation type (rmsnorm, rope,
    softmax, ...) becomes its own stage record.

    Cycles are the primary metric.  The roofline columns (`mem_time`,
    `eff_time`, `bound`) are also reported because a stage can be DRAM-bound --
    notably attention during decode -- in which case raw cycles understate its
    real latency contribution.

    Returns:
        dict with 'prefill', 'decode' and 'decode_per_token' -- each
        {stage: record} -- plus 'num_decode_steps', 'freq_hz', 'dram_bw'.
        Each record has: stage, category ("AW"/"AA"/"non_gemm"), execs,
        cycles, flops, dram_bytes, compute_time, mem_time, eff_time, bound.
    """
    if dram_bandwidth_gbps is None:
        dram_bandwidth_gbps = sim.hw.dram_bandwidth_gbps
    dram_bw = dram_bandwidth_gbps * 1e9
    freq = sim.hw.freq_mhz * 1e6

    def _record(stage, category, execs, cycles, flops, dram_bytes,
                eff_time=None, sram_time=0.0):
        compute_time = cycles / freq
        mem_time = dram_bytes / dram_bw if dram_bw > 0 else 0.0
        if eff_time is None:
            eff_time = max(compute_time, mem_time, sram_time)
        # Three-way, so a stage held up by SRAM throughput is not mislabelled
        # "compute".  `sram_time` is 0.0 unless hw.sram_bandwidth_gbps is set,
        # in which case this reduces to the original two-way test exactly.
        if sram_time > compute_time and sram_time > mem_time:
            bound = 'sram'
        elif mem_time > compute_time:
            bound = 'memory'
        else:
            bound = 'compute'
        return {
            'stage': stage,
            'category': category,
            'execs': execs,
            'cycles': cycles,
            'flops': flops,
            'dram_bytes': dram_bytes,
            'compute_time': compute_time,
            'mem_time': mem_time,
            'sram_time': sram_time,
            'eff_time': eff_time,
            'bound': bound,
        }

    def _phase_stages(phase: PhaseMetrics) -> dict:
        stages = {}

        for ops_dict, category in ((phase.aa_ops, 'AA'), (phase.aw_ops, 'AW')):
            for op_type, op_list in ops_dict.items():
                if not op_list:
                    continue
                total = _aggregate_metrics(op_list)
                # Roofline time is summed per execution (not aggregate-then-max)
                # so these match compute_roofline_latency* exactly.
                eff_time = sum(
                    sim._op_roofline_time(m, freq, dram_bw) for m in op_list
                )
                stages[op_type.value] = _record(
                    op_type.value, category, len(op_list), total.cycles,
                    total.flops, total.dram_read_eff + total.dram_write_eff,
                    eff_time=eff_time,
                    sram_time=sum(sim._sram_time(m) for m in op_list),
                )

        # Non-GEMM (VPU) stages -- no DRAM path, so always compute-bound.
        for op_type, data in _aggregate_non_gemm_metrics(phase.non_gemm_ops).items():
            stages[op_type.value] = _record(
                op_type.value, 'non_gemm', data['count'], data['cycles'], 0, 0)

        return stages

    prefill = _phase_stages(results.prefill)
    decode = _phase_stages(results.decode)
    num_steps = max(1, workload.output_tokens - 1)

    decode_per_token = {}
    for stage, rec in decode.items():
        per = dict(rec)
        for key in ('execs', 'cycles', 'flops', 'dram_bytes',
                    'compute_time', 'mem_time', 'eff_time'):
            per[key] = rec[key] / num_steps
        decode_per_token[stage] = per

    return {
        'prefill': prefill,
        'decode': decode,
        'decode_per_token': decode_per_token,
        'num_decode_steps': num_steps,
        'freq_hz': freq,
        'dram_bw': dram_bw,
    }


def compute_unit_cycle_breakdown(sim: UnitAwareSimulator,
                                 results: SimulationResults,
                                 workload: WorkloadConfig) -> dict:
    """Break the cycle cost down per hardware unit (LGU, PE array, BQU, ...).

    Complements `compute_stage_cycle_breakdown`, which splits by transformer
    stage.  Here the split is by which block of Fig. 4 spends the cycle; see
    `cycle_units` for the unit names.

    The BQU runs on-the-fly alongside the PE array, so `bqu_tse` / `bqu_bea`
    are reported but excluded from `serial_cycles`.

    Returns:
        dict with 'prefill', 'decode', 'decode_per_token' -- each
        {unit: {cycles, time, pct_of_serial, overlapped}} -- plus
        'serial_cycles' per phase, 'num_decode_steps' and 'freq_hz'.
    """
    freq = sim.hw.freq_mhz * 1e6
    bqu_ops = getattr(sim, 'bqu_ops', {'prefill': [], 'decode': []})

    def _phase_units(phase: PhaseMetrics, phase_name: str):
        units: Dict[str, int] = {}

        for ops_dict in (phase.aa_ops, phase.aw_ops):
            for op_list in ops_dict.values():
                for m in op_list:
                    for unit, c in getattr(m, 'unit_cycles', {}).items():
                        units[unit] = units.get(unit, 0) + c

        vpu_cycles = sum(m.cycles for m in phase.non_gemm_ops)
        if vpu_cycles:
            units['vpu'] = units.get('vpu', 0) + vpu_cycles

        for b in bqu_ops.get(phase_name, []):
            units['bqu_tse'] = units.get('bqu_tse', 0) + b.tse_cycles
            units['bqu_bea'] = units.get('bqu_bea', 0) + b.bea_cycles

        serial = sum(c for u, c in units.items() if u not in OVERLAPPED_UNITS)
        return units, serial

    def _records(units, serial, divisor=1.0):
        out = {}
        for unit, cycles in units.items():
            cycles = cycles / divisor
            out[unit] = {
                'unit': unit,
                'cycles': cycles,
                'time': cycles / freq,
                'pct_of_serial': (100.0 * cycles / (serial / divisor)
                                  if serial else 0.0),
                'overlapped': unit in OVERLAPPED_UNITS,
            }
        return out

    prefill_units, prefill_serial = _phase_units(results.prefill, 'prefill')
    decode_units, decode_serial = _phase_units(results.decode, 'decode')
    num_steps = max(1, workload.output_tokens - 1)

    return {
        'prefill': _records(prefill_units, prefill_serial),
        'decode': _records(decode_units, decode_serial),
        'decode_per_token': _records(decode_units, decode_serial, num_steps),
        'serial_cycles': {
            'prefill': prefill_serial,
            'decode': decode_serial,
            'decode_per_token': decode_serial / num_steps,
        },
        'num_decode_steps': num_steps,
        'freq_hz': freq,
    }
