"""
OS-V round counting: the batch-2 cliff, and what fixing it moves.

`analysis/memory/regime_run.py` set out to map what decode is bound by and
instead found that the map's own batch axis was sitting on a cycle-model
defect. This measures the defect and the fix.

**The defect.** `_calculate_cycles` counts `LUT_OS_V` output-stationary rounds
as `ceil(M/array_m) * n_tiles`, which rounds `M` up to a whole 32-row tile
*before* multiplying by `n_tiles`. The array holds
`array_m x (array_n x NUM_RAC)` accumulators, so a round can retire `array_m`
accumulator tiles wherever they come from -- different output rows, different
column tiles, or a mix. The budget therefore allows
`ceil(M * n_tiles / array_m)`, and the two disagree by `array_m / M` for
`2 <= M < array_m`:

    M          1     2     4     8    16    32
    tiled      1    32    32    32    32    32
    packed     1     2     4     8    16    32
    ratio     1x   16x    8x    4x    2x    1x     (N = 4096, n_tiles = 32)

**The tell that it is a defect and not a modelling choice.** The `M == 1`
branch already computes `ceil(n_tiles / array_m)`, which *is*
`ceil(1 x n_tiles / array_m)`. The special case is not special -- it is the one
place the general formula was written down, and `"packed"` restores it for
every `M`.

**Why it matters beyond tidiness.** Decode issues every AW projection with
`M = batch`, so under `"tiled"` decode cycles jump ~33x from batch 1 to batch 2
and are then *flat* to batch 32: the model charges the same compute for 2
sequences as for 32. That is what makes decode look compute-bound from batch 2
upward, which is the conclusion `regime_run.py` could not publish.

**Scope: `LUT_OS_V` only** (see the `os_rounds_model` docstring). `LUT_OS`
carries the same accumulator argument but not the same evidence, and widening
it would move prefill on first principles alone.

Staged per this repo's convention: `hw.os_rounds_model`, default `"tiled"`,
which reproduces every published number bit-for-bit -- the re-captured baseline
moved only `hw.os_rounds_model` (a new key) and the per-entry `full_sha256`,
with **zero** value keys changed.

Usage:
    python rounds_run.py
    python rounds_run.py --csv rounds.csv --report rounds_report.md
"""

import argparse
import csv
import math
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, '..', '..'))
for p in ('simulator', 'analysis', 'analysis/cycle_breakdown',
          'analysis/memory'):
    sys.path.insert(0, os.path.join(_root, *p.split('/')))

from simulator import (                                              # noqa: E402
    HardwareConfig, WorkloadConfig, Simulator, ComputeMode, OperationType,
)
from model_configs import get_model_config                           # noqa: E402
from cycle_units import cycle_units                                  # noqa: E402
from report import Report                                            # noqa: E402

MODEL = 'LLaMA-3-8B'
OUTPUT_TOKENS = 4
CONTEXTS = [2048, 8192, 32768]
BATCHES = [1, 2, 4, 8, 16, 32]
MODELS = ['tiled', 'packed']


def base_hw(rounds_model='tiled', **kw):
    return HardwareConfig(
        array_m=32, array_n=4, FPE_array_size=64,
        act_bits=16, accumulate_bits=32, weight_bits=4, kv_cache_bits=4,
        AW_mode="OMNI", AA_mode="OMNI", os_rounds_model=rounds_model, **kw)


def decode_split(batch, context, rounds_model):
    """Decode compute / DRAM per token under one rounds model."""
    m = get_model_config(MODEL)
    w = WorkloadConfig(batch_size=batch, input_tokens=context,
                       output_tokens=OUTPUT_TOKENS, flash_block_size=0)
    sim = Simulator(base_hw(rounds_model))
    r = sim.simulate(m, w)
    freq = sim.hw.freq_mhz * 1e6
    bw = sim.hw.dram_bandwidth_gbps * 1e9
    steps = max(1, w.output_tokens - 1)
    c = d = 0.0
    per_op = {}
    for grp in (r.decode.aw_ops, r.decode.aa_ops):
        for op, lst in grp.items():
            oc = sum(x.cycles for x in lst)
            per_op[op.value] = oc / steps
            c += oc / freq
            d += sum(x.dram_read_eff + x.dram_write_eff for x in lst) / bw
    _, tpot = sim.compute_roofline_latency(r, w)
    return {'compute_ms': c / steps * 1e3, 'dram_ms': d / steps * 1e3,
            'cd_ratio': c / d, 'regime': 'compute' if c > d else 'memory',
            'tpot_ms': tpot * 1e3, 'per_op': per_op}


# ============================================================================
# Pre-flight
# ============================================================================

def preflight():
    """Six checks. The first two are what make the fix believable."""
    print("Pre-flight")
    hw_t, hw_p = base_hw('tiled'), base_hw('packed')
    sim_t, sim_p = Simulator(hw_t), Simulator(hw_p)

    # 1. "packed" reproduces the M == 1 branch EXACTLY -- the argument that the
    #    special case was never special. If this fails, the general form is not
    #    the general form.
    for N in (128, 1024, 4096, 14336):
        for K in (128, 4096, 32768):
            a = sim_t._calculate_cycles(1, K, N, 4, ComputeMode.AA,
                                        "LUT_OS_V", 1)
            b = sim_p._calculate_cycles(1, K, N, 4, ComputeMode.AA,
                                        "LUT_OS_V", 1)
            assert a == b, (K, N, a, b)
    print("  1. packed == tiled at M == 1, every shape tested ok")

    # 2. The ratio is exactly array_m / M for 2 <= M < array_m -- the defect's
    #    signature, asserted rather than eyeballed off a table.
    N, K, am = 4096, 4096, hw_t.array_m
    n_tiles = math.ceil(N / (hw_t.array_n * Simulator.NUM_RAC))
    for M in (2, 4, 8, 16, 32):
        rt = math.ceil(math.ceil(M / am) * n_tiles)
        rp = math.ceil(M * n_tiles / am)
        assert rp == max(1, M * n_tiles // am), (M, rp)
        assert rt / rp == max(1.0, am / M), (M, rt, rp, am / M)
    print(f"  2. rounds ratio is exactly array_m/M for M in 2..{am} ok")

    # 3. "packed" never charges MORE than "tiled" -- a fix that made some shape
    #    slower would mean the accumulator argument is wrong somewhere.
    for M in (1, 2, 3, 7, 16, 31, 32, 33, 64, 512):
        for N in (128, 4096, 14336):
            a = sim_t._calculate_cycles(M, 4096, N, 4, ComputeMode.AA,
                                        "LUT_OS_V", 1)
            b = sim_p._calculate_cycles(M, 4096, N, 4, ComputeMode.AA,
                                        "LUT_OS_V", 1)
            assert b <= a, (M, N, a, b)
    print("  3. packed <= tiled at every shape tested ok")

    # 4. Standing check 2: the unit breakdown still sums to the single number,
    #    under BOTH models. cycle_units.py duplicates the rounds logic, so this
    #    is the check that catches the two having drifted apart.
    for hw, sim in ((hw_t, sim_t), (hw_p, sim_p)):
        for M in (1, 2, 8, 32, 512):
            for K, N in ((4096, 4096), (4096, 14336), (32768, 128)):
                for mode in ("LUT_OS_V", "LUT_OS", "LUT_WS"):
                    u = cycle_units(hw, M, K, N, 4, mode, 1, mu=Simulator.MU,
                                    num_rac=Simulator.NUM_RAC)
                    tot = sum(u.values()) if isinstance(u, dict) else u
                    ref = sim._calculate_cycles(M, K, N, 4, ComputeMode.AA,
                                                mode, 1)
                    assert tot == ref, (hw.os_rounds_model, M, K, N, mode,
                                        tot, ref)
    print("  4. sum(cycle_units) == _calculate_cycles under both models ok")

    # 5. LUT_OS and LUT_WS are untouched by the switch -- the scope claim.
    for mode in ("LUT_OS", "LUT_WS", "FPE_OS", "TENDER"):
        for M in (1, 2, 8, 32):
            a = sim_t._calculate_cycles(M, 4096, 4096, 4, ComputeMode.AA,
                                        mode, 1)
            b = sim_p._calculate_cycles(M, 4096, 4096, 4, ComputeMode.AA,
                                        mode, 1)
            assert a == b, (mode, M, a, b)
    print("  5. LUT_OS / LUT_WS / FPE_OS / TENDER unaffected ok")

    # 6. Decode attention is unaffected: qk and attn_v are issued with M = 1,
    #    so every KV result in study.md is out of this stage's blast radius.
    t = decode_split(8, 32768, 'tiled')
    p = decode_split(8, 32768, 'packed')
    for op in ('qk_matmul', 'attn_v_matmul'):
        assert t['per_op'][op] == p['per_op'][op], (op, t['per_op'][op],
                                                    p['per_op'][op])
    print("  6. decode qk / attn_v identical -- attention is M=1 ok")
    print()


# ============================================================================
# Sections
# ============================================================================

def sweep(report_path):
    rows = []
    rpt = Report(
        report_path,
        "OS-V round counting",
        "The batch-2 cliff, and what fixing it moves",
        source='analysis/memory/rounds_run.py',
        setup=[
            f"{MODEL}, Omni-LUT-KV4 (32x4, W4A16KV4, 500 MHz, DDR5-6400).",
            "`hw.os_rounds_model`: 'tiled' = every published number; "
            "'packed' = the accumulator-budget form.",
        ],
    )

    # ---- A. the cliff ------------------------------------------------------
    rpt.section(
        "A. The cliff, per operation",
        "Decode cycles per token for each AW projection, batch 1 -> 32, under "
        "'tiled'. Decode issues these with `M = batch`.")
    ops = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'fc1', 'fc2']
    base = {b: decode_split(b, 2048, 'tiled') for b in BATCHES}
    a_rows = []
    for op in ops:
        r1 = base[1]['per_op'][op]
        a_rows.append([op] + [f"{base[b]['per_op'][op]/r1:.2f}x"
                              for b in BATCHES])
    rpt.table(['op'] + [f"b{b}" for b in BATCHES], a_rows, aligns='l')
    rpt.note(
        "**Every projection jumps ~33x from batch 1 to batch 2 for a 2x "
        "workload, then does not move again through batch 32.** The model "
        "charges the same decode compute for 2 sequences as for 32. Both halves "
        "are the same bug: `ceil(M/32)` is 1 for all `M` in 1..32, so `M` "
        "vanishes from the round count entirely, and the `M == 1` branch is the "
        "only thing that kept batch 1 honest.")

    # ---- B. what packed changes -------------------------------------------
    rpt.section(
        "B. Decode compute, both models",
        "Compute time per token, and the ratio between the two round models.")
    b_rows = []
    for ctx in CONTEXTS:
        for b in BATCHES:
            t = decode_split(b, ctx, 'tiled')
            p = decode_split(b, ctx, 'packed')
            b_rows.append([
                f"{ctx//1024}K", str(b),
                f"{t['compute_ms']:.1f}", f"{p['compute_ms']:.1f}",
                f"{t['compute_ms']/p['compute_ms']:.2f}x",
                f"{t['tpot_ms']:.1f}", f"{p['tpot_ms']:.1f}",
                f"{t['tpot_ms']/p['tpot_ms']:.2f}x",
            ])
            rows.append(dict(section='B', context=ctx, batch=b,
                             tiled_compute_ms=t['compute_ms'],
                             packed_compute_ms=p['compute_ms'],
                             tiled_tpot_ms=t['tpot_ms'],
                             packed_tpot_ms=p['tpot_ms'],
                             tiled_cd=t['cd_ratio'], packed_cd=p['cd_ratio'],
                             tiled_regime=t['regime'],
                             packed_regime=p['regime']))
    rpt.table(
        ['ctx', 'batch', 'tiled compute', 'packed compute', 'compute ratio',
         'tiled TPOT', 'packed TPOT', 'TPOT ratio'],
        b_rows, aligns='ll')
    rpt.note(
        "**Batch 1 is bit-identical** -- it always used the correct branch. "
        "Everything from batch 2 moves, most at batch 2 and least at batch 32, "
        "exactly as `array_m / M` predicts.")

    # ---- C. the regime map, corrected -------------------------------------
    rpt.section(
        "C. The regime map under each model",
        "Compute/DRAM ratio. Below 1.0 the array waits on memory.")
    for rm in MODELS:
        c_rows = []
        for b in BATCHES:
            c_rows.append([str(b)] + [
                f"{decode_split(b, c, rm)['cd_ratio']:.2f}" for c in CONTEXTS])
        rpt.table(['batch'] + [f"{c//1024}K" for c in CONTEXTS], c_rows,
                  aligns='l', caption=f"os_rounds_model = '{rm}'")
    rpt.note(
        "**This is what `regime_run.py` was blocked on, and the corrected "
        "picture is a different shape entirely.** Under 'tiled' the boundary is "
        "a cliff between batch 1 and 2 and nothing moves after it. Under "
        "'packed' it is a smooth diagonal: the first compute-bound batch is "
        "**16 at 2K, 4 at 8K, 2 at 32K**. Both axes push the same way -- batch "
        "amortises constant weight traffic, context grows attention compute "
        "quadratically -- so the memory-bound region is a triangle in the "
        "low-batch, short-context corner, not a single row.")
    rpt.note(
        "**The practical consequence is that the memory-bound regime is much "
        "larger than 'tiled' says.** At 2K it reaches batch 8 (C/D 0.93), where "
        "'tiled' claimed 1.93. Any conclusion of the form \"decode is "
        "compute-bound at batch, so KV techniques are pointless there\" was "
        "reading the artefact.")

    rpt.summary([
        "**`ceil(M/array_m) * n_tiles` drops `M` entirely for `M` in 1..32.** "
        "Decode issues AW projections with `M = batch`, so under 'tiled' every "
        "projection jumps ~33x from batch 1 to 2 and is then flat to batch 32 "
        "-- the same compute charged for 2 sequences as for 32.",
        "**The `M == 1` branch was never a special case.** "
        "`ceil(n_tiles/array_m)` is `ceil(1 * n_tiles / array_m)`, the general "
        "accumulator-budget form, which is why batch 1 was right and nothing "
        "else was. Pre-flight 1 asserts the two agree at `M == 1` for every "
        "shape tested.",
        "**Attention is out of the blast radius.** `qk` and `attn_v` are "
        "issued with `M = 1`, so every KV result in `study.md` is untouched "
        "(pre-flight 6). So are `LUT_OS`, `LUT_WS`, `FPE_OS` and `TENDER`, "
        "which the switch does not reach (pre-flight 5).",
        "**Shipped inert.** Default `'tiled'` reproduces every published number "
        "bit-for-bit: the re-captured baseline moved only the new `hw` key and "
        "the per-entry `full_sha256`, with zero value keys changed.",
    ])
    return rpt, rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default=os.path.join(_here, 'rounds.csv'))
    p.add_argument('--report', default=os.path.join(_here, 'rounds_report.md'))
    args = p.parse_args()
    preflight()
    rpt, rows = sweep(args.report)
    rpt.save()
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(args.csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {args.csv}")
    print(f"Wrote report      -> {args.report}")


if __name__ == '__main__':
    main()
