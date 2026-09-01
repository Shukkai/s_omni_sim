"""
Per-port SRAM accounting: which on-chip memory actually moves the bytes.

`study.md` §16(c) shipped `sram_bandwidth_gbps` inert because charging the
geometry-implied 128 GB/s made prefill TTFT go **4.35x**, off 113,670 GB of
SRAM traffic against 3 GB of DRAM.  It attributed that to the untiled
activation matrix (the same defect behind §7's 2.1 GB prefill working set) and
parked prefill until prefill tiling lands.

**That attribution was wrong, and this file is the measurement that shows it.**
Decompose the lump by operand and the activations turn out to be *right*:

    A read   7.52 GB    251.2 B/cycle
    B read   0.03 GB      1.0 B/cycle
    C read  14.56 GB    486.7 B/cycle
    C write 15.03 GB    502.4 B/cycle

251.2 B/cycle of activations against an array that consumes
`array_m x MU x act_bits/8` = **256 B/cycle** -- within 2% of the operand port
it was always supposed to be compared against.  Tiling A cannot reduce this and
was never what inflated it.  **79.7% of the lump is accumulator recirculation**:
weight-stationary walks K in `k_tiles` passes and the bit-planes in `qbit`
more, and every pass but the last cycles a full 32-bit partial-sum matrix out
and back -- 128 round trips at prefill shapes.

And those bytes never cross an SRAM port at all.  OMNI_LUT.pdf Fig. 4 draws
three separate memories -- **Unified Buffer**, **Weight Buffer**, **Accumulator**
-- with the accumulator wired straight to the PE array's partial-sum outputs.
`hw.sram_port_model = "ported"` bills each against its own bandwidth and takes
the max, because they move bytes concurrently.

Usage:
    python ports_run.py
    python ports_run.py --csv ports.csv --report ports_report.md
"""

import argparse
import csv
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, '..', '..'))
for p in ('simulator', 'analysis', 'analysis/cycle_breakdown',
          'analysis/memory'):
    sys.path.insert(0, os.path.join(_root, *p.split('/')))

from simulator import HardwareConfig, WorkloadConfig, Simulator      # noqa: E402
from model_configs import get_model_config                           # noqa: E402
from report import Report                                            # noqa: E402

MODEL = 'LLaMA-3-8B'
CONTEXT = 32768
OUTPUT_TOKENS = 4
KV_BITS = 4
SCORE_SRAM_KB = 128
SRAM_BW = [0.0, 128.0, 256.0, 512.0, 1024.0]
ACCUM_BW = [0.0, 2048.0, 1024.0, 512.0, 256.0]
GB = 1e9


def base_hw(sram_bw=0.0, port_model="lumped", accum_bw=0.0):
    return HardwareConfig(
        array_m=32, array_n=4, FPE_array_size=64,
        act_bits=16, accumulate_bits=32, weight_bits=4, kv_cache_bits=KV_BITS,
        AW_mode="OMNI", AA_mode="OMNI",
        sram_bandwidth_gbps=sram_bw, sram_port_model=port_model,
        accum_bandwidth_gbps=accum_bw, score_sram_kb=SCORE_SRAM_KB,
    )


def measure(hw, batch=1, context=CONTEXT):
    sim = Simulator(hw)
    m = get_model_config(MODEL)
    w = WorkloadConfig(batch_size=batch, input_tokens=context,
                       output_tokens=OUTPUT_TOKENS, flash_block_size=0)
    r = sim.simulate(m, w)
    pf = r.prefill.get_total_metrics()
    dec = r.decode.get_total_metrics()
    ttft, tpot = sim.compute_roofline_latency(r, w)
    out = {'ttft_s': ttft, 'tpot_s': tpot,
           'prefill_cycles': pf.cycles, 'decode_cycles': dec.cycles}
    for tag, t in (('prefill', pf), ('decode', dec)):
        out[tag + '_a'] = t.sram_read_a
        out[tag + '_b'] = t.sram_read_b
        out[tag + '_acc'] = t.sram_acc_read + t.sram_acc_write
        out[tag + '_out'] = t.sram_write_out
        out[tag + '_lump'] = t.sram_read + t.sram_write
    return out


def act_port_bytes_per_cycle(hw):
    """What the array geometry says the activation port must sustain."""
    return hw.array_m * Simulator(hw).MU * hw.act_bits // 8


# ============================================================================
# Pre-flight
# ============================================================================

def preflight():
    # 1. "lumped" is the default, and the default is exactly the old model.
    assert HardwareConfig(array_m=32, array_n=4).sram_port_model == "lumped"
    assert HardwareConfig(array_m=32, array_n=4).accum_bandwidth_gbps == 0.0

    # 2. The split partitions the lumps exactly -- no byte gained or lost.
    d = measure(base_hw())
    for tag in ('prefill', 'decode'):
        parts = d[tag + '_a'] + d[tag + '_b'] + d[tag + '_acc'] + d[tag + '_out']
        assert parts == d[tag + '_lump'], \
            f"{tag}: ports {parts} != lump {d[tag + '_lump']}"

    # 3. At unlimited bandwidth the port model cannot matter: both reduce to
    #    max(compute, DRAM).  This is what makes the field safe to add.
    a = measure(base_hw(sram_bw=0.0, port_model="lumped"))
    b = measure(base_hw(sram_bw=0.0, port_model="ported"))
    assert a['ttft_s'] == b['ttft_s'] and a['tpot_s'] == b['tpot_s'], \
        "port model must be inert at sram_bandwidth_gbps = 0"

    # 4. "ported" can never be slower than "lumped" at the same bandwidth: a
    #    max over a partition is at most the sum of its parts.
    for bw in (128.0, 512.0):
        lo = measure(base_hw(sram_bw=bw, port_model="ported"))
        hi = measure(base_hw(sram_bw=bw, port_model="lumped"))
        assert lo['ttft_s'] <= hi['ttft_s'] + 1e-12, \
            f"ported TTFT {lo['ttft_s']} > lumped {hi['ttft_s']} at {bw}"
        assert lo['tpot_s'] <= hi['tpot_s'] + 1e-12, \
            f"ported TPOT {lo['tpot_s']} > lumped {hi['tpot_s']} at {bw}"

    # 5. A finite accumulator bandwidth must actually reach the roofline --
    #    otherwise the term is computed and dropped, which is worse than absent.
    free = measure(base_hw(sram_bw=128.0, port_model="ported", accum_bw=0.0))
    slow = measure(base_hw(sram_bw=128.0, port_model="ported", accum_bw=1.0))
    assert slow['ttft_s'] > free['ttft_s'] * 10, \
        "a 1 GB/s accumulator should dominate; the term is not reaching max()"

    # 6. The activation traffic matches the array's own operand-port rate.
    #    This is the load-bearing claim of the whole file: if A-reads were the
    #    defect §16(c) called them, this ratio would not be ~1.
    hw = base_hw()
    d = measure(hw)
    rate = d['prefill_a'] / d['prefill_cycles']
    geom = act_port_bytes_per_cycle(hw)
    assert 0.9 < rate / geom < 1.05, \
        f"activation port {rate:.1f} B/cyc vs geometry {geom} B/cyc"

    # 7. Output-stationary has no accumulator recirculation -- that is the
    #    definition of the dataflow, so decode (LUT_OS_V) must charge zero.
    assert d['decode_acc'] == 0, \
        f"output-stationary decode should have no accumulator traffic, " \
        f"got {d['decode_acc']}"

    print("pre-flight: 7 checks passed")


# ============================================================================
# Sweep
# ============================================================================

def sweep(report_path):
    rows = []
    preflight()

    hw0 = base_hw()
    geom = act_port_bytes_per_cycle(hw0)
    d = measure(hw0)

    rep = Report(
        report_path,
        "Per-port SRAM accounting",
        subtitle="Which on-chip memory moves the bytes, and what §16(c) "
                 "misattributed",
        source="analysis/memory/ports_run.py",
        setup=[f"Model: LLaMA-3-8B on Omni-LUT, 4-bit KV, context "
               f"{CONTEXT:,}, batch 1, standard attention, scores staged.",
               "Ports follow OMNI_LUT.pdf Fig. 4: Unified Buffer "
               "(activations + results), Weight Buffer (weights / KV), "
               "Accumulator (partial sums)."])

    rep.summary([
        f"**The activation traffic was never the defect.** Prefill reads "
        f"{d['prefill_a'] / d['prefill_cycles']:.1f} B/cycle of activations "
        f"against an array that consumes `array_m x MU x act_bits/8` = "
        f"**{geom} B/cycle**. Within 2% of the port it is charged against, so "
        f"tiling A cannot reduce it — §16(c)'s attribution to the "
        f"untiled-activation defect does not survive decomposition.",
        f"**The accumulator is "
        f"{100 * d['prefill_acc'] / d['prefill_lump']:.1f}% of the lump.** "
        f"Weight-stationary recirculates a full 32-bit partial-sum matrix "
        f"`k_tiles x qbit` = 128 times per prefill GEMM. Those bytes were "
        f"being billed against the *activation* port's bandwidth.",
        "**And they never cross an SRAM port.** Fig. 4 wires the accumulator "
        "straight to the PE array's partial-sum outputs, separate from the "
        "Unified Buffer. Modelling it as a client of that buffer is what "
        "produced the 4.35x.",
        "**Prefill is unparked.** Under `sram_port_model = \"ported\"` at the "
        "geometry-implied 128 GB/s per port, TTFT is a bandwidth result rather "
        "than a modelling artefact — see section C for the number.",
        "**Decode is untouched, as it must be.** Output-stationary keeps "
        "partial sums in the array, so `LUT_OS_V` charges zero accumulator "
        "traffic and the port split changes nothing there.",
    ])

    # ---- A ------------------------------------------------------------------
    rep.section(
        "A. Where the bytes actually go",
        "The same traffic the lumped model reported, decomposed by which "
        "memory of Fig. 4 moves it. Rates are against that phase's own cycle "
        "count, so they are directly comparable to a port width.")
    trows = []
    for tag in ('prefill', 'decode'):
        cyc = d[tag + '_cycles']
        for port, key in (("activation (A read)", '_a'),
                          ("weight (B read)", '_b'),
                          ("accumulator (partial sums)", '_acc'),
                          ("activation (result write)", '_out')):
            b = d[tag + key]
            rows.append({'section': 'A', 'phase': tag, 'port': port,
                         'bytes': b, 'bytes_per_cycle': b / cyc,
                         'share': b / d[tag + '_lump']})
            trows.append([tag, port, f"{b / GB:,.2f} GB",
                          f"{b / cyc:,.1f}", f"{b / d[tag + '_lump']:.1%}"])
    rep.table(["phase", "port", "bytes", "B/cycle", "share of lump"],
              trows, aligns="llrrr")
    rep.note(
        f"**The activation row is the result.** {geom} B/cycle is what the "
        f"array geometry demands and "
        f"{d['prefill_a'] / d['prefill_cycles']:.1f} B/cycle is what the model "
        f"charges — the activation term was correct all along, and 128 GB/s "
        f"was always the *activation port's* number rather than an aggregate. "
        f"Decode's accumulator row is zero because `LUT_OS_V` is "
        f"output-stationary; that asymmetry is the dataflow, not an omission.")

    # ---- B ------------------------------------------------------------------
    rep.section(
        "B. Lumped vs ported, at the same per-port bandwidth",
        "Accumulator unlimited, which is the physical reading: it is a "
        "datapath block, not a buffer client.")
    trows = []
    ref = measure(base_hw(sram_bw=0.0))
    for bw in SRAM_BW:
        lump = measure(base_hw(sram_bw=bw, port_model="lumped"))
        port = measure(base_hw(sram_bw=bw, port_model="ported"))
        rows.append({'section': 'B', 'sram_bw': bw,
                     'ttft_lumped': lump['ttft_s'], 'ttft_ported': port['ttft_s'],
                     'tpot_lumped': lump['tpot_s'], 'tpot_ported': port['tpot_s']})
        label = "unlimited" if bw == 0 else f"{bw:,.0f} GB/s"
        trows.append([
            label,
            f"{lump['ttft_s']:,.1f} s", f"{lump['ttft_s'] / ref['ttft_s']:.2f}x",
            f"{port['ttft_s']:,.1f} s", f"{port['ttft_s'] / ref['ttft_s']:.2f}x",
            f"{1e3 * port['tpot_s']:,.2f} ms",
            f"{port['tpot_s'] / ref['tpot_s']:.3f}x",
        ])
    rep.table(["per-port BW", "TTFT lumped", "vs ∞", "TTFT ported", "vs ∞",
               "TPOT ported", "vs ∞"], trows, aligns="lrrrrrr")
    rep.note(
        "**A max over a partition is at most the sum of its parts**, so ported "
        "can never price above lumped at the same bandwidth — asserted in "
        "pre-flight rather than hoped for. The gap between the two columns is "
        "the accumulator traffic that was being charged to the wrong memory.")

    # ---- C ------------------------------------------------------------------
    rep.section(
        "C. What a finite accumulator would cost",
        "If the partial sums *did* have to cross a bandwidth-limited memory, "
        "how fast would it have to be? Activation and weight ports fixed at "
        "the geometry-implied 128 GB/s.")
    trows = []
    for abw in ACCUM_BW:
        r = measure(base_hw(sram_bw=128.0, port_model="ported", accum_bw=abw))
        rows.append({'section': 'C', 'accum_bw': abw,
                     'ttft_s': r['ttft_s'], 'tpot_s': r['tpot_s']})
        label = "unlimited" if abw == 0 else f"{abw:,.0f} GB/s"
        trows.append([label, f"{r['ttft_s']:,.1f} s",
                      f"{r['ttft_s'] / ref['ttft_s']:.2f}x",
                      f"{1e3 * r['tpot_s']:,.2f} ms",
                      f"{r['tpot_s'] / ref['tpot_s']:.3f}x"])
    rep.table(["accumulator BW", "TTFT", "vs ∞", "TPOT", "vs ∞"],
              trows, aligns="lrrrr")
    acc_rate = d['prefill_acc'] / d['prefill_cycles']
    rep.note(
        f"The accumulator moves {acc_rate:,.0f} B/cycle in prefill, so it needs "
        f"about **{acc_rate * hw0.freq_mhz * 1e6 / GB:,.0f} GB/s** to stay off "
        f"the critical path. That is a real design constraint on the "
        f"accumulator block — but it is a constraint on *that* block, and "
        f"nothing about it belongs in the unified buffer's budget.")

    rep.save()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=os.path.join(_here, 'ports.csv'))
    ap.add_argument('--report', default=os.path.join(_here, 'ports_report.md'))
    args = ap.parse_args()

    rows = sweep(args.report)
    keys = sorted({k for r in rows for k in r})
    with open(args.csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.csv} and {args.report}")


if __name__ == '__main__':
    main()
