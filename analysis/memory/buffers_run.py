"""
The RTL's four buffers against the model's one pool.

`sram_capacity_kb` models on-chip memory as a single pool any operand may draw
from, and §19's `sram_port_model = "ported"` splits its *bandwidth* three ways
while leaving its *capacity* undivided.  The Omni-LUT system block diagram does
neither: four physically separate SRAMs, fixed sizes, fixed word widths, no
trading.  `simulator/buffer_tech.py` holds the geometry; this file measures what
adopting it changes.

**The word widths confirm §19 outright.**  The input buffer word is 2048 b =
256 B = `array_m x MU x act_bits/8` -- one cycle of activation operand -- and
the output word is 4096 b = 512 B = `array_n x NUM_RAC x accum_bits/8`, one
column tile of accumulators.  §19 measured the activation port at 255.7 B/cycle
against a predicted 256 and argued 128 GB/s had always been an activation-port
number rather than an aggregate.  The RTL builds the buffer one cycle wide.
Both identities are asserted in pre-flight rather than admired here.

**Three things the pool could not express**, and all three bind:

  1. Input and output are separate memories *at different widths*, so §19's
     "unified" port both wrongly summed them and understated the output side
     by 2x.
  2. The scale buffer carries an operand the model had no bucket for --
     `hw.model_scale_traffic` adds it.
  3. Capacity cannot be traded, so an operation can fit in 3 MB of total SRAM
     and still not run.

Usage:
    python buffers_run.py
    python buffers_run.py --csv buffers.csv --report buffers_report.md
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
from buffer_tech import (                                            # noqa: E402
    buffer_config, with_buffer_config, DEFAULT_BUFFER_CONFIG,
)
from memory_tech import with_memory_technology                       # noqa: E402
from model_configs import get_model_config                           # noqa: E402
from report import Report                                            # noqa: E402

MODEL = 'LLaMA-3-8B'
CONTEXTS = [2048, 8192, 16384, 32768]
CONTEXT = 8192
OUTPUT_TOKENS = 4
KV_BITS = 4
SCORE_SRAM_KB = 128
TILES = [0, 128, 32, 8]
KB = 1024
GB = 1e9


def pool_hw(m_tile=0):
    return with_memory_technology(HardwareConfig(
        array_m=32, array_n=4, FPE_array_size=64, act_bits=16,
        accumulate_bits=32, weight_bits=4, kv_cache_bits=KV_BITS,
        AW_mode="OMNI", AA_mode="OMNI", score_sram_kb=SCORE_SRAM_KB,
        sram_m_tile=m_tile), 'DDR5-6400')


def rtl_hw(m_tile=0, scale_traffic=True, enforce=True):
    return with_buffer_config(pool_hw(m_tile), DEFAULT_BUFFER_CONFIG,
                              enforce=enforce, scale_traffic=scale_traffic)


def measure(hw, context=CONTEXT, batch=1):
    sim = Simulator(hw)
    m = get_model_config(MODEL)
    w = WorkloadConfig(batch_size=batch, input_tokens=context,
                       output_tokens=OUTPUT_TOKENS, flash_block_size=0)
    r = sim.simulate(m, w)
    ttft, tpot = sim.compute_roofline_latency(r, w)
    pf, dec = r.prefill.get_total_metrics(), r.decode.get_total_metrics()
    return {'ttft_s': ttft, 'tpot_s': tpot,
            'prefill_overflow': pf.sram_overflow_buffers,
            'decode_overflow': dec.sram_overflow_buffers,
            'decode_scale_sram': dec.sram_read_scale,
            'decode_dram': dec.dram_read_eff + dec.dram_write_eff,
            'decode_sram': dec.sram_read + dec.sram_write}


# ============================================================================
# Pre-flight
# ============================================================================

def preflight():
    cfg = buffer_config(DEFAULT_BUFFER_CONFIG)
    stock = HardwareConfig(array_m=32, array_n=4)
    sim = Simulator(stock)

    # 1. "pool" is the default, and every buffer field is inert at 0.
    assert stock.sram_buffer_model == "pool"
    assert stock.model_scale_traffic is False
    assert stock.input_buffer_bytes == 0 and stock.output_buffer_bytes == 0

    # 2. THE identity: one input-buffer word is one cycle of activation
    #    operand.  This is what §19 inferred from a measurement.
    want_in = stock.array_m * sim.MU * stock.act_bits // 8
    assert cfg.input.word_bytes == want_in, (
        f"input word {cfg.input.word_bytes} B != array_m x MU x act_bits/8 "
        f"= {want_in} B")

    # 3. And one output word is one column tile of accumulators.
    want_out = stock.array_n * sim.NUM_RAC * stock.accumulate_bits // 8
    assert cfg.output.word_bytes == want_out, (
        f"output word {cfg.output.word_bytes} B != "
        f"array_n x NUM_RAC x accum_bits/8 = {want_out} B")

    # 4. The input buffer holds exactly `array_m` activation rows, which is
    #    why the machine *is* sram_m_tile = 32 and §20's 512 was fiction.
    rows = cfg.input.bytes // (4096 * stock.act_bits // 8)
    assert rows == stock.array_m, \
        f"input buffer holds {rows} rows of a 4096-wide model, not array_m"

    # 5. A 32K Key cache at 4 bits is the weight buffer, exactly.
    k32 = 32768 * 128 * KV_BITS // 8
    assert k32 == cfg.weight.bytes, \
        f"32K K cache {k32} B vs weight buffer {cfg.weight.bytes} B"

    # 6. Loading the geometry without enforcing it changes nothing.
    a = measure(pool_hw())
    b = measure(rtl_hw(enforce=False, scale_traffic=False))
    for k in ('ttft_s', 'tpot_s'):
        assert a[k] == b[k], f"enforce=False must be inert, {k} moved"

    # 7. The scale term is off by default and non-zero when on.
    assert measure(rtl_hw(scale_traffic=False))['decode_scale_sram'] == 0
    assert measure(rtl_hw(scale_traffic=True))['decode_scale_sram'] > 0

    # 8. Real ports and real capacity can only cost time, never save it.
    for ctx in (2048, 32768):
        p = measure(pool_hw(), context=ctx)
        r = measure(rtl_hw(scale_traffic=False), context=ctx)
        assert r['ttft_s'] >= p['ttft_s'] - 1e-12, "partitioned TTFT below pool"
        assert r['tpot_s'] >= p['tpot_s'] - 1e-12, "partitioned TPOT below pool"

    # 9. The two structural overflows show up at 32K and not before.
    small = measure(rtl_hw(m_tile=32), context=2048)
    big = measure(rtl_hw(m_tile=32), context=32768)
    assert small['decode_overflow'] == '', \
        f"nothing should overflow in decode at 2K, got {small['decode_overflow']}"
    assert 'weight' in big['decode_overflow'], \
        f"the 32K KV tile should not fit the weight buffer, " \
        f"got {big['decode_overflow']}"

    # 10. The input buffer allows a different row block per operation, and
    #     `array_m` is right for exactly one of them.  This pins the fact that
    #     an earlier draft of section 22 got wrong: fc2, not attention, is what
    #     binds below 16K, because its `K` is `d_ffn` rather than `d_model`.
    d_model, d_ffn = 4096, 14336
    rows_dmodel = cfg.input.bytes // (d_model * stock.act_bits // 8)
    rows_dffn = cfg.input.bytes // (d_ffn * stock.act_bits // 8)
    assert rows_dmodel == stock.array_m, "d_model block should be array_m"
    assert rows_dffn == 9, f"d_ffn block should be 9 rows, got {rows_dffn}"
    ov = measure(rtl_hw(m_tile=32), context=2048)['prefill_overflow']
    assert 'input' in ov, \
        f"fc2 should overflow the input buffer at 32 rows even at 2K, got {ov}"

    print("pre-flight: 10 checks passed")


# ============================================================================
# Sweep
# ============================================================================

def sweep(report_path):
    rows = []
    preflight()

    cfg = buffer_config(DEFAULT_BUFFER_CONFIG)
    stock = HardwareConfig(array_m=32, array_n=4)
    sim = Simulator(stock)
    pool = measure(pool_hw())

    rep = Report(
        report_path,
        "RTL buffer partition",
        subtitle="Four fixed SRAMs against the model's one pool",
        source="analysis/memory/buffers_run.py",
        setup=[f"Model: LLaMA-3-8B on Omni-LUT, 4-bit KV, context "
               f"{CONTEXT:,}, batch 1, DDR5-6400, standard attention.",
               "Buffer geometry from the Omni-LUT system block diagram, in "
               "`simulator/buffer_tech.py` with its derivation."])

    rtl32 = measure(rtl_hw(m_tile=32))
    rep.summary([
        f"**The RTL confirms §19's central claim outright.** The input buffer "
        f"word is {cfg.input.word_bytes} B = `array_m × MU × act_bits/8` — the "
        f"buffer is built exactly one cycle of activation operand wide. §19 "
        f"measured 255.7 B/cycle against a predicted 256 and concluded "
        f"128 GB/s had always been an *activation-port* number. It had.",
        f"**But the partition is not a pool, and three things bind.** Input "
        f"and output are separate memories at different widths "
        f"({cfg.input.word_bytes} B vs {cfg.output.word_bytes} B), so §19's "
        f"'unified' port both wrongly summed them and understated the output "
        f"side by 2×. The scale buffer carries an operand the model had no "
        f"bucket for. And capacity cannot be traded.",
        f"**The machine *is* `sram_m_tile = 32`.** The input buffer holds "
        f"{cfg.input.bytes // (4096 * 2)} rows of a 4096-wide model — exactly "
        f"`array_m`. §20 nominated a 512-row block as the operating point; "
        f"that describes a buffer that was never built.",
        "**Two structural overflows fall straight out of the sizes.** A 32K "
        "Key cache at 4 bits is 2,048 KB, which is the weight buffer exactly, "
        "so K alone fills it and K+V needs twice the part. And the input "
        "buffer is sized for a `d_model`-wide operand, so **fc2 — whose `K` is "
        "`d_ffn` — gets 9 rows where fc1 gets 32**, and binds first at every "
        "context below 16K.",
        f"**Total cost of modelling the real part: TTFT "
        f"{pool['ttft_s']:,.1f} s → {rtl32['ttft_s']:,.1f} s "
        f"({rtl32['ttft_s'] / pool['ttft_s']:.2f}×), TPOT "
        f"{1e3 * pool['tpot_s']:,.2f} → {1e3 * rtl32['tpot_s']:,.2f} ms "
        f"({rtl32['tpot_s'] / pool['tpot_s']:.3f}×).** Decode barely moves, "
        "which is §19's result arriving again from the hardware side.",
    ])

    # ---- A ------------------------------------------------------------------
    rep.section(
        "A. The geometry, and the two identities that check it",
        "Transcribed from the block diagram. Depth 1024 is corroborated "
        "independently by the DRAM address map in the same note, whose "
        "address field is `[9:0]`.")
    trows = []
    for name in ('input', 'scale', 'weight', 'output'):
        spec = getattr(cfg, name)
        rows.append({'section': 'A', 'buffer': name, 'depth': spec.depth,
                     'width_bits': spec.width_bits, 'banks': spec.banks,
                     'bytes': spec.bytes})
        trows.append([f"{name} buffer",
                      f"{spec.depth}×{spec.width_bits} b"
                      + (f" ×{spec.banks}" if spec.banks > 1 else ""),
                      f"{spec.kb:,.0f} KB", f"{spec.word_bytes} B"])
    trows.append(["**total**", "", f"**{cfg.total_bytes / KB:,.0f} KB**", ""])
    rep.table(["buffer", "geometry", "capacity", "word"], trows, aligns="llrr")
    rep.note(
        f"**The word widths are the array geometry, and that is the check.**\n\n"
        f"- input `{cfg.input.width_bits} b / act_bits {stock.act_bits}` = "
        f"{cfg.input.width_bits // stock.act_bits} elements = "
        f"`array_m × MU` = {stock.array_m * sim.MU} ✓\n"
        f"- output `{cfg.output.width_bits} b / accum_bits "
        f"{stock.accumulate_bits}` = "
        f"{cfg.output.width_bits // stock.accumulate_bits} elements = "
        f"`array_n × NUM_RAC` = {stock.array_n * sim.NUM_RAC} ✓\n\n"
        f"So the input port is {cfg.input.word_bytes} B/cycle = "
        f"{cfg.input.word_bytes * stock.freq_mhz * 1e6 / GB:,.0f} GB/s and the "
        f"output port is {cfg.output.word_bytes} B/cycle = "
        f"{cfg.output.word_bytes * stock.freq_mhz * 1e6 / GB:,.0f} GB/s. "
        f"**§19 charged both against one 128 GB/s number.**")

    # ---- B ------------------------------------------------------------------
    rep.section(
        "B. What adopting the real part costs",
        "Pool (unlimited capacity, unlimited bandwidth) against the RTL's four "
        "fixed memories, at several row blocks.")
    trows = []
    for mt in TILES:
        r = measure(rtl_hw(m_tile=mt))
        rows.append({'section': 'B', 'm_tile': mt, **r})
        trows.append([
            "untiled" if mt == 0 else f"{mt} rows",
            f"{r['ttft_s']:,.1f} s", f"{r['ttft_s'] / pool['ttft_s']:.2f}×",
            f"{1e3 * r['tpot_s']:,.2f} ms",
            f"{r['tpot_s'] / pool['tpot_s']:.3f}×",
            r['prefill_overflow'] or "—",
        ])
    rep.table(["row block", "TTFT", "vs pool", "TPOT", "vs pool",
               "prefill overflow"], trows, aligns="lrrrrl")
    rep.note(
        "**Decode barely moves and prefill moves a lot**, which is §19's split "
        "arriving from the hardware side rather than from a measurement: "
        "decode is `M=1` and tiling-inert, and its traffic is weight-port "
        "traffic served by 8 banks at 2,048 B/cycle.\n\n"
        "**The `input` overflow does not clear at 32 rows, and what binds is "
        "not what the buffer was sized for.** The A operand is "
        "`m_tile × K × act_bits/8`, so the block a 256 KB buffer allows "
        "depends on that operation's `K`: **32 rows** for the projections and "
        "fc1 (`d_model` 4,096, exactly `array_m` — plainly deliberate), but "
        "only **9** for **fc2**, whose `K` is `d_ffn` = 14,336, and 64 down to "
        "4 for `attn_v` as `kv_len` grows. **fc2 binds at 2K and 8K**; "
        "attention takes over only past 16K, where `kv_len` exceeds `d_ffn`. "
        "So one `sram_m_tile` cannot serve the model, and the conflict is "
        "*inside the FFN* before it is ever between the FFN and attention — "
        "invisible under a pool, where fc2's 896 KB borrowed silently from the "
        "other 2.75 MB.")

    # ---- C ------------------------------------------------------------------
    rep.section(
        "C. Where the capacity runs out",
        "Which buffer overflows, by context, at the 32-row block the input "
        "buffer implies.")
    trows = []
    for ctx in CONTEXTS:
        r = measure(rtl_hw(m_tile=32), context=ctx)
        rows.append({'section': 'C', 'context': ctx, **r})
        kv_kb = ctx * 128 * KV_BITS // 8 / KB
        trows.append([f"{ctx:,}", f"{kv_kb:,.0f} KB",
                      f"{kv_kb / cfg.weight.kb:.2f}×",
                      f"{1e3 * r['tpot_s']:,.2f} ms",
                      r['decode_overflow'] or "—"])
    rep.table(["context", "one K cache", "vs weight buffer", "TPOT",
               "decode overflow"], trows, aligns="lrrrl")
    rep.note(
        "**32K is exactly where this part runs out, and the arithmetic is "
        "exact rather than approximate**: `32768 × 128 × 4 b` = 2,048 KB and "
        "the weight buffer is 2,048 KB. K alone fills it; K+V needs twice the "
        "chip. So §11's on-chip KV residency tops out near **16K** on this "
        "part, and §7's claim that the KV tile becomes binding past ~16K was "
        "right for a reason it could not see — it read that off a pool, and "
        "the real boundary is a buffer wall.\n\n"
        "**The `scale` overflow at 32K is model-derived and wants RTL "
        "confirmation.** Per-token Value scales at `qbit + 1` FP16 each are "
        "320 KB at 32K against a 256 KB scale buffer. The *layout* is inferred "
        "from OMNI_LUT.pdf §IV-B (see `_scale_operand_bits`), not read from "
        "the RTL, so treat it as a question to ask rather than a defect found.")

    # ---- D ------------------------------------------------------------------
    rep.section(
        "D. The scale operand, which the model had no bucket for",
        "A 256 KB SRAM, its own load command and its own DRAM type code — and "
        "zero bytes in every number published before this field.")
    trows = []
    for st in (False, True):
        r = measure(rtl_hw(m_tile=32, scale_traffic=st))
        rows.append({'section': 'D', 'scale_traffic': st, **r})
        trows.append(["on" if st else "off",
                      f"{r['decode_scale_sram'] / GB:,.3f} GB",
                      f"{1e3 * r['tpot_s']:,.2f} ms",
                      f"{r['ttft_s']:,.1f} s"])
    rep.table(["scale traffic", "decode scale SRAM", "TPOT", "TTFT"],
              trows, aligns="lrrr")
    rep.note(
        "**Small in time, real in capacity.** The term moves TPOT by a "
        "fraction of a percent, because the scales ride the same schedule as "
        "B and B's port is eight banks wide. What it is *not* small in is "
        "area: 256 KB, 8.3% of the on-chip budget, for an operand the model "
        "priced at nothing. The interesting output here is the capacity "
        "question in section C, not the latency.")

    rep.save()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=os.path.join(_here, 'buffers.csv'))
    ap.add_argument('--report',
                    default=os.path.join(_here, 'buffers_report.md'))
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
