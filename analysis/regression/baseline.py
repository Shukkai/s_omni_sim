"""
Regression gate for changes to the simulator's model.

Memory modelling edits `_calculate_memory_access` and `_simulate_matmul`, which
every published number in `study.md` depends on.  This captures a full dump of
`SimulationResults.to_dict()` across a matrix of hardware configs, models and
workloads, and later re-runs it to prove a change was inert where it was meant
to be inert.

The contract for any new hardware feature is that its *disabled* default
reproduces the archive exactly -- not approximately.  Floats are compared bit
for bit via `repr`, because a model change that is genuinely a no-op produces
identical arithmetic, and anything else deserves to be looked at.

Usage:
    python baseline.py capture --out baseline.json
    python baseline.py check   --ref baseline.json
    python baseline.py check   --ref baseline.json --max-report 40
"""

import argparse
import hashlib
import json
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, '..', '..'))
sys.path.insert(0, os.path.join(_root, 'simulator'))

from simulator import (                                          # noqa: E402
    HardwareConfig, Simulator, WorkloadConfig,
)
from model_configs import get_model_config                       # noqa: E402


# ---- The matrix -------------------------------------------------------------

def hw_configs() -> dict:
    """Every hardware config the analysis scripts actually use."""
    return {
        'Tender': HardwareConfig(
            array_m=64, array_n=64, FPE_array_size=64,
            act_bits=8, accumulate_bits=32, weight_bits=8, kv_cache_bits=8,
            AW_mode="TENDER", AA_mode="TENDER", tender_m1_opt=True),
        'FPE': HardwareConfig(
            array_m=64, array_n=64, FPE_array_size=64,
            act_bits=16, accumulate_bits=32, weight_bits=4, kv_cache_bits=16,
            AW_mode="FPE_OS", AA_mode="FPE_OS"),
        'FIGLUT': HardwareConfig(
            array_m=32, array_n=4, replication=1, FPE_array_size=64,
            act_bits=16, accumulate_bits=32, weight_bits=4, kv_cache_bits=16,
            AW_mode="LUT_WS", AA_mode="FPE_OS"),
        'Omni4': HardwareConfig(
            array_m=32, array_n=4, FPE_array_size=64,
            act_bits=16, accumulate_bits=32, weight_bits=4, kv_cache_bits=4,
            AW_mode="OMNI", AA_mode="OMNI"),
        'Omni3': HardwareConfig(
            array_m=32, array_n=4, FPE_array_size=64,
            act_bits=16, accumulate_bits=32, weight_bits=4, kv_cache_bits=3,
            AW_mode="OMNI", AA_mode="OMNI"),
        'Omni-OSV': HardwareConfig(
            array_m=32, array_n=4, FPE_array_size=64,
            act_bits=16, accumulate_bits=32, weight_bits=4, kv_cache_bits=4,
            AW_mode="LUT_OS_V", AA_mode="LUT_OS_V"),
    }


# (model, batch, input_tokens, output_tokens, flash_block_size)
CASES = [
    ('LLaMA-3-8B', 1, 2048, 32, 0),      # study.md's own setup, short decode
    ('LLaMA-3-8B', 1, 32768, 32, 0),     # long context, standard attention
    ('LLaMA-3-8B', 8, 8192, 16, 0),      # batched
    ('LLaMA-3-8B', 32, 32768, 8, 0),     # the batch-32 row Sec. 4(a) leans on
    ('OPT-6.7B', 1, 8192, 16, 256),      # FlashAttention path
    ('LLaMA-2-7B', 1, 8192, 16, 256),    # MHA + flash
]


def _slim(tree: dict) -> dict:
    """Drop the per-execution dumps, keep every total.

    `to_dict()` emits one record per execution, and an operation is executed
    once per layer per decode step with identical metrics -- 129 MB of exact
    duplicates.  The totals are the readable gate; `full_sha256` below is the
    exact one, so nothing is actually lost by dropping them here.
    """
    for phase in ('prefill', 'decode'):
        for group in ('aa_operations', 'aw_operations'):
            for op in tree['results'][phase][group].values():
                op.pop('executions', None)
    return tree


def run_matrix() -> dict:
    """Simulate every (config, case) pair and dump the metric tree.

    Each entry carries the slimmed tree for human-readable diffs plus a hash
    of the *unslimmed* tree, so a change that somehow moved a per-execution
    record without moving any total would still be caught.
    """
    out = {}
    for hw_name, hw in hw_configs().items():
        for model_name, batch, in_tok, out_tok, flash in CASES:
            key = f"{hw_name}|{model_name}|b{batch}|i{in_tok}|o{out_tok}|f{flash}"
            model = get_model_config(model_name)
            workload = WorkloadConfig(
                batch_size=batch, input_tokens=in_tok,
                output_tokens=out_tok, flash_block_size=flash)
            sim = Simulator(hw)
            results = sim.simulate(model, workload)

            entry = {
                'results': results.to_dict(),
                'roofline': sim.compute_roofline_latency(results, workload),
                'roofline_breakdown': sim.compute_roofline_latency_breakdown(
                    results, workload),
                'hw': hw.to_dict(),
            }
            digest = hashlib.sha256(
                json.dumps(entry, sort_keys=True, default=str).encode()
            ).hexdigest()
            out[key] = _slim(entry)
            out[key]['full_sha256'] = digest
    return out


# ---- Comparison -------------------------------------------------------------

def flatten(obj, prefix=''):
    """Flatten a nested dict/list into {dotted.path: leaf}."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from flatten(v, f"{prefix}[{i}]")
    else:
        yield prefix, obj


def normalize(tree: dict) -> dict:
    """Round a tree through JSON so a fresh run and an archived one compare.

    Without this the gate reports differences that are not differences: an
    energy model returning `np.float64` reprs as `np.float64(0.035...)` fresh
    but `0.035...` after a save/load, and tuples come back as lists.  The
    values are bit-identical; only the types moved.  Normalizing both sides
    keeps the comparison exact while making it immune to whatever any future
    model happens to return.
    """
    return json.loads(json.dumps(tree, sort_keys=True, default=float))


def compare(ref: dict, got: dict, max_report: int) -> int:
    """Exact comparison.  Returns the number of differing leaves."""
    ref_flat = dict(flatten(normalize(ref)))
    got_flat = dict(flatten(normalize(got)))

    missing = sorted(set(ref_flat) - set(got_flat))
    added = sorted(set(got_flat) - set(ref_flat))
    changed = [k for k in sorted(set(ref_flat) & set(got_flat))
               if repr(ref_flat[k]) != repr(got_flat[k])]

    if missing:
        print(f"  {len(missing)} key(s) missing from the new run, e.g.:")
        for k in missing[:max_report]:
            print(f"    - {k}")
    if added:
        print(f"  {len(added)} new key(s) -- expected when a field is added:")
        for k in added[:max_report]:
            print(f"    + {k}")
    if changed:
        print(f"  {len(changed)} value(s) changed:")
        for k in changed[:max_report]:
            print(f"    ~ {k}: {ref_flat[k]!r} -> {got_flat[k]!r}")
        if len(changed) > max_report:
            print(f"    ... and {len(changed) - max_report} more")

    return len(missing) + len(changed)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('capture', 'check'))
    p.add_argument('--out', default=os.path.join(_here, 'baseline.json'))
    p.add_argument('--ref', default=os.path.join(_here, 'baseline.json'))
    p.add_argument('--max-report', type=int, default=20)
    args = p.parse_args()

    n_cases = len(hw_configs()) * len(CASES)
    print(f"Running {n_cases} (config, workload) simulations ...")
    data = run_matrix()

    if args.mode == 'capture':
        with open(args.out, 'w') as f:
            json.dump(data, f, indent=1, sort_keys=True)
        leaves = sum(1 for _ in flatten(data))
        print(f"Captured {n_cases} runs / {leaves:,} values -> {args.out}")
        return

    with open(args.ref) as f:
        ref = json.load(f)
    print(f"Comparing against {args.ref} ...\n")
    n_bad = compare(ref, data, args.max_report)
    if n_bad:
        print(f"\nREGRESSION: {n_bad} value(s) differ from the baseline.")
        sys.exit(1)
    print("Identical to the baseline ✓")


if __name__ == "__main__":
    main()
