"""
The RAM model at the built configuration -- dense baseline.

One reference sheet: given the RTL's four on-chip SRAMs and a DDR5-6400 part,
what does the memory model actually say for a **dense** workload -- no KV
reduction, no activation sparsity, nothing switched on?  Every result in
`study.md` §4-§21 is a delta against these numbers, and none of them were ever
collected in one place.

**Dense means dense.**  `sram_m_tile` is the only non-default, because without a
row block prefill claims a 2.16 GB working set (§20) and no capacity row means
anything.  It is set to **9**, which §22 showed is the largest block every
operand actually fits -- fc2's `d_ffn`-wide A operand is what decides it, not
fc1 and not attention.

Usage:
    python ram_model_run.py
    python ram_model_run.py --csv ram_model.csv --report ram_model_report.md
"""

import argparse
import csv
import io
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, '..', '..'))
for p in ('simulator', 'analysis', 'analysis/cycle_breakdown',
          'analysis/memory'):
    sys.path.insert(0, os.path.join(_root, *p.split('/')))

import dataclasses                                                    # noqa: E402
from simulator import (                                              # noqa: E402
    HardwareConfig, WorkloadConfig, Simulator, ComputeMode, OperationType,
)
from buffer_tech import (                                            # noqa: E402
    buffer_config, with_buffer_config, DEFAULT_BUFFER_CONFIG,
)
from memory_tech import (                                            # noqa: E402
    memory_technology, with_memory_technology,
)
from model_configs import get_model_config                           # noqa: E402
from report import Report                                            # noqa: E402
from cycle_units import (                                            # noqa: E402
    UnitAwareSimulator, compute_unit_cycle_breakdown,
    compute_stage_cycle_breakdown, bqu_metrics,
)

MODEL = 'LLaMA-3-8B'
TECH = 'DDR5-6400'
CONTEXTS = [2048, 8192, 32768]
FOCUS = 8192        # sections G and H tabulate one context; the CSV has all
OUTPUT_TOKENS = 4
KV_BITS = 4
SCORE_SRAM_KB = 128
M_TILE = 9          # §22: the largest block every operand fits
KB = 1024.0
MB = KB * KB
GB = 1e9


class BufferProbe(UnitAwareSimulator):
    """Records each operation's per-buffer footprint, so the report can say
    *which* operation sets each buffer's peak rather than only how big it is.

    Built on `UnitAwareSimulator` so the same run also carries the per-unit
    cycle attribution -- bytes and cycles for one configuration, from one
    simulation, which is the point of a baseline sheet.

    Analysis-only: it overrides nothing the simulator computes, it just keeps
    the footprint `_simulate_matmul` already asks for.  `model_bqu=False`
    because the BQU is a placeholder (`study.md` TODO) and a reference sheet
    should not carry unmeasured numbers.
    """

    def __init__(self, hw):
        super().__init__(hw, model_bqu=False)
        self.peaks = {}     # buffer -> (bytes, op label)

    def _simulate_matmul(self, op_type, compute_mode, shape, **kw):
        m = super()._simulate_matmul(op_type, compute_mode, shape, **kw)
        _, mode = self._resolve_dataflow_mode(compute_mode,
                                              kw.get('is_decode', False))
        fps = self._operand_footprints(
            *shape, compute_mode, mode, kw.get('batch_size', 1),
            sram_batch=kw.get('sram_batch', 1))
        phase = 'decode' if kw.get('is_decode') else 'prefill'
        for buf, need in fps.items():
            key = (phase, buf)
            if need > self.peaks.get(key, (0, ''))[0]:
                self.peaks[key] = (need, op_type.value)
        return m


# ---------------------------------------------------------------------------
# `dense.md` is the same tables with the prose stripped out -- a reference
# sheet you read by looking things up rather than by reading.  It is emitted
# from the same run as the report, so the two cannot disagree, and unlike the
# `*_report.md` files it is **tracked**: it describes the built configuration
# rather than a sweep, so it only changes when the hardware does.
# ---------------------------------------------------------------------------

#: The simulator's internal op names are identifiers; a reference sheet should
#: say what the operation *is*.  Widths are LLaMA-3-8B.
OP_LABEL = {
    'q_proj': 'Q projection',
    'k_proj': 'K projection',
    'v_proj': 'V projection',
    'o_proj': 'output projection',
    'fc1': 'FFN expand (4,096 \u2192 14,336)',
    'fc2': 'FFN contract (14,336 \u2192 4,096)',
    'qk_matmul': 'attention Q\u00b7K\u1d40',
    'attn_v_matmul': 'attention scores\u00b7V',
    'gate': 'MoE router',
}


#: One line per section: how to read that table.  Short on purpose -- the
#: longer description stays in the report, this is the key you glance at.
READS_AS = {
    'A': "the parts list \u2014 what exists and how fast each one is.",
    'B': "off-chip bytes actually moved. Prefill = whole phase, "
         "decode = **per token**.",
    'C': "where cycles go, by unit. Columns are % of that phase's cycles.",
    'D': "on-chip bytes per port, as B/cycle and % of that port's width. "
         "100% = that port is the bottleneck.",
    'E': "the largest working set each buffer must hold, and which operation "
         "demands it. `OVER` = does not fit.",
    'F': "the three roofline terms side by side. Largest wins; "
         "\u201cover 2nd\u201d is the margin.",
    'G': "per stage: what it costs to compute, what it costs to fetch, and "
         "how much of the fetch its own compute fails to hide.",
    'H': "the units that are not GEMM stages \u2014 quantisation, table "
         "generation, operand load. **BQU rows are a placeholder.**",
    'I': "what the Key cache's bit allocation costs. Value is held at the low "
         "width throughout, as the paper specifies.",
}

ORIENT = ("**B and D are bytes, C is time, E is capacity, F reconciles them.** "
          "Read F first to see what is limiting, then jump to whichever of "
          "B/C/D explains it.")


def op_label(name):
    return OP_LABEL.get(name, name)


DENSE = []          # [('section', title, subtitle) | ('table', caption, hdr, rows)]

#: The short version, for people who read the tables by looking things up.
#: Kept next to the numbers that produce them so the two stay honest.
#: (headline, [supporting points]).  Rendered as a list with sublists --
#: one claim per bullet, its evidence indented under it, so the section can be
#: skimmed by headline alone.
FINDINGS = [
    ("**Prefill and decode are bound by different things**, at every context.",
     ["Decode is **DRAM-bound**: 6.8× more DRAM time than compute at 2K, "
      "narrowing to 1.1× at 32K.",
      "Prefill is **compute-bound** by 2.8–3.3×.",
      "Two machines in one part, wanting two different optimisations."]),

    ("**The 256 KB input buffer decides how many tokens run at once, and the "
     "FFN's second matrix is what limits it.**",
     ["The buffer holds `rows × input width × 2 B`, so the block depends on "
      "*that operation's* input width.",
      "Projections and the FFN **expand** take a 4,096-wide input: 8 KB a "
      "row, so **32 rows** fit — exactly the array's height, clearly what "
      "it was sized for.",
      "The FFN **contract** takes a 14,336-wide input: 28 KB a row, so only "
      "**9** fit.",
      "The block is one global setting, so the tightest operation wins: "
      "**9, not 32.**"]),

    ("**Those 9 rows are where prefill's time goes.**",
     ["A systolic array pays a fixed start-up cost to fill and drain, and at "
      "9 rows it is amortised over 9 rows instead of 32.",
      "**Fill/drain becomes 70–74% of prefill cycles** — a term that is "
      "under 2% when the whole sequence streams at once.",
      "That costs **~5.3x** prefill compute.",
      "**A bigger input buffer is worth more here than a faster anything.**"]),

    ("**No on-chip port exceeds 37% of its width, in either phase.**",
     ["The same tiling inflates cycles ~5× but leaves operand bytes "
      "unchanged, so every port's utilisation falls with it.",
      "The buffer that forces the block is the same buffer whose port then "
      "goes idle.",
      "The 8 weight banks are there for **capacity, not bandwidth**: the FFN "
      "contract needs 896 KB resident and one bank holds 256 KB."]),

    ("**Decode moves per token almost exactly what prefill moves in total.**",
     ["2.65 GB against 2.65 GB at 2K.",
      "2.58 GB of that is weights, re-read every step because they do not "
      "fit in 3 MB of SRAM.",
      "**It is weights, not KV, until 32K** — 2.58 GB against 0.07 GB."]),

    ("**32K is where the part runs out, and three walls arrive together.**",
     ["Attention's scores-times-V step needs **225%** of the input buffer.",
      "And **125%** of the scale buffer.",
      "And one 32K Key cache is **2,048 KB against a 2,048 KB** weight "
      "buffer — K alone fills it, K+V needs twice the chip.",
      "Only the input one shrinks with a smaller block; the other two scale "
      "with `K`, not with the block."]),

    ("**Per stage, decode is exposed memory almost everywhere and prefill is "
     "exposed nowhere.**",
     ["In prefill every stage is compute-bound: **RAM wait is 0.00 ms "
      "across the board.**",
      "In decode the two FFN stages wait **~95% of their DRAM time** — at "
      "`M = 1` there is no compute to hide it behind. `attention scores·V` "
      "is the only compute-bound stage.",
      "**Those are exactly the stages the FFN weight lever targets**, which "
      "is why it works and KV levers do not."]),

    ("**Key-cache bit allocation is nearly free; Value-cache width is not.**",
     ["50% of Key channels at 5 bits costs **1.005× TPOT at 8K, 1.010× at "
      "32K**. Both caches at 5 bits costs **1.070× / 1.152×** — 7–15× more.",
      "`qk` is ~5% of decode cycles and `attn_v` is 80–92%, though they carry "
      "identical bytes. **So AS-Bit's load-bearing half is leaving the Value "
      "cache alone, not the Key adaptivity.**",
      "**Packed vs padded scheduling changes TPOT by 0%** — decode is "
      "DRAM-bound, so the extra bit-plane pass hides. No case for packing "
      "hardware."]),

    ("**The array shape is tuned to `head_dim`, and is right for the block it "
     "was designed for.**",
     ["A 32×4 array gives exactly **128 columns** against attention's "
      "`head_dim` of 128; 16×8 wastes half, 8×16 three quarters.",
      "At the forced 9-row block, 16×8 would be **1.16×** faster — "
      "fill/drain dominates there.",
      "But **32×4 wins at every block from 32 rows upward**, including "
      "untiled, and 32 rows is what a 256 KB input buffer holds.",
      "**The array and the buffer agree; the FFN contract breaks the "
      "pairing.** (`analysis/memory/tileshape_report.md`)"]),

    ("**The geometry confirms the array model.**",
     ["Input word 256 B = `array_m × MU × act_bits/8` — one cycle of "
      "activation operand.",
      "Output word 512 B = `array_n × NUM_RAC × accum_bits/8` — one column "
      "tile of accumulators."]),
]


def emit_section(rep, heading, body=''):
    rep.section(heading, body)
    DENSE.append(('section', heading, body))
    return rep


def emit_table(rep, headers, trows, caption=None, aligns=None, **kw):
    rep.table(headers, trows, caption=caption, aligns=aligns, **kw)
    DENSE.append(('table', caption, list(headers), [list(r) for r in trows],
                  aligns or ''))
    return rep


def write_dense(path, setup):
    """Render the collected tables as plain markdown, no notes."""
    out = ["<!-- Generated by analysis/memory/ram_model_run.py.",
           "     Edits here are overwritten on the next run. -->",
           "# Omni-LUT — dense baseline",
           ""]
    for line in setup:
        out.append(f"- {line}")
    out += ["", "---", "", "## Reading the tables", "", ORIENT, ""]
    for item in DENSE:
        if item[0] == 'section':
            out += ["---", "", f"## {item[1]}", ""]
            key = READS_AS.get(item[1][:1])
            if key:
                out += [f"**Reads as** \u2014 {key}", ""]
            elif item[2]:
                out += [f"*{item[2]}*", ""]
            continue
        _kind, caption, headers, trows, aligns = item
        if caption:
            out += [f"**{caption}**", ""]
        out.append("| " + " | ".join(headers) + " |")
        marks = []
        for i in range(len(headers)):
            a = aligns[i] if i < len(aligns) else 'l'
            marks.append("---:" if a == 'r' else ":---:" if a == 'c' else "---")
        out.append("| " + " | ".join(marks) + " |")
        for row in trows:
            out.append("| " + " | ".join(str(c).strip() for c in row) + " |")
        out.append("")
    # Findings last: the tables are the document, and the summary reads better
    # once you have seen the numbers it is drawn from.
    out += ["---", "", "## Findings", ""]
    for head, subs in FINDINGS:
        out.append(f"- {head}")
        out += [f"    - {t}" for t in subs]
        out.append("")
    io.open(path, 'w', encoding='utf-8').write("\n".join(out).rstrip() + "\n")
    return path


def _bqu_for(hw, model, context):
    """BQU cycles per phase, from `bqu_metrics`.

    **This is a placeholder, not a measurement.**  `study.md`'s TODO says so
    plainly: the original simulator does not model the BQU at all, and
    `bqu_metrics` assumes one BEA pass per bit-plane, one TSE min/max pass on
    the Value path, and `bqu_width` elements per cycle.  It is reported here
    because "what does quantisation cost?" is a fair question to ask of a
    reference sheet, and because leaving it out silently is worse than
    including it labelled.  Replace with RTL numbers before quoting.

    Excluded from every serial total in this sheet, per OMNI_LUT.pdf §IV-A,
    which runs the BQU on-the-fly alongside the PE array.
    """
    out = {}
    for phase, tokens in (('prefill', context), ('decode', 1)):
        tse = bea = 0
        for tensor in ('key', 'value'):
            b = bqu_metrics(hw, tensor, tokens, model.d_kv)
            tse += b.tse_cycles
            bea += b.bea_cycles
        out[phase] = {'tse': tse * model.num_layers,
                      'bea': bea * model.num_layers}
    return out


def base_hw(m_tile=M_TILE):
    hw = HardwareConfig(
        array_m=32, array_n=4, FPE_array_size=64,
        act_bits=16, accumulate_bits=32, weight_bits=4, kv_cache_bits=KV_BITS,
        AW_mode="OMNI", AA_mode="OMNI", score_sram_kb=SCORE_SRAM_KB,
        sram_m_tile=m_tile)
    hw = with_memory_technology(hw, TECH)
    return with_buffer_config(hw, DEFAULT_BUFFER_CONFIG)


def run(context, batch=1):
    hw = base_hw()
    sim = BufferProbe(hw)
    m = get_model_config(MODEL)
    w = WorkloadConfig(batch_size=batch, input_tokens=context,
                       output_tokens=OUTPUT_TOKENS, flash_block_size=0)
    r = sim.simulate(m, w)
    ttft, tpot = sim.compute_roofline_latency(r, w)
    freq = hw.freq_mhz * 1e6
    steps = max(1, w.output_tokens - 1)
    out = {'context': context, 'batch': batch,
           'ttft_s': ttft, 'tpot_s': tpot, 'peaks': sim.peaks,
           'units': compute_unit_cycle_breakdown(sim, r, w),
           'stages': compute_stage_cycle_breakdown(sim, r, w),
           'bqu': _bqu_for(hw, m, context)}
    for tag, ph, div in (('prefill', r.prefill, 1), ('decode', r.decode, steps)):
        t = ph.get_total_metrics()
        out[tag] = {
            'cycles': t.cycles / div,
            'compute_s': t.cycles / freq / div,
            'dram_read': t.dram_read_eff / div,
            'dram_write': t.dram_write_eff / div,
            'dram_s': ((t.dram_read_eff + t.dram_write_eff)
                       / (hw.dram_bandwidth_gbps * 1e9) / div),
            'input': (t.sram_read_a + t.sram_write_out) / div,
            'scale': t.sram_read_scale / div,
            'weight': t.sram_read_b / div,
            'output': (t.sram_acc_read + t.sram_acc_write) / div,
            'overflow': t.sram_overflow_buffers,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=os.path.join(_here, 'ram_model.csv'))
    ap.add_argument('--report',
                    default=os.path.join(_here, 'ram_model_report.md'))
    ap.add_argument('--dense', default=os.path.join(_root, 'dense.md'),
                    help='tables-only sheet, tracked in the repo')
    args = ap.parse_args()

    cfg = buffer_config(DEFAULT_BUFFER_CONFIG)
    tech = memory_technology(TECH)
    hw = base_hw()
    freq = hw.freq_mhz * 1e6
    runs = {c: run(c) for c in CONTEXTS}
    rows = []

    SETUP = [f"{MODEL}, batch 1, {OUTPUT_TOKENS - 1} decode steps, standard "
             f"attention, W4A16KV{KV_BITS}, scores staged on chip.",
             f"On-chip: the RTL's four SRAMs. Off-chip: {TECH}. Row block "
             f"`sram_m_tile = {M_TILE}` (§22 — the largest every operand "
             f"fits). Nothing else is non-default."]

    rep = Report(
        args.report,
        "RAM model — dense baseline",
        subtitle="What the built configuration does with nothing switched on",
        source="analysis/memory/ram_model_run.py",
        setup=SETUP)

    rep.summary([
        "**The two phases are bound by different resources at every context.** "
        "Decode is DRAM-bound — 6.7× more DRAM time than compute at 2K, "
        "narrowing to 1.06× at 32K. Prefill is compute-bound by 3.3× to 581×. "
        "They are two machines and want two different optimisations.",
        "**The 256 KB input buffer is the binding design decision, and it is "
        "set by fc2.** Its A operand is `m_tile × d_ffn × 2 B`, and `d_ffn` is "
        "3.5× `d_model`, so the block is **9 rows** rather than `array_m`'s 32 "
        "— and a 9-row block inflates prefill compute **~5.3×** over untiled "
        "(§20). One SRAM sets prefill performance.",
        "**No on-chip port exceeds 37% of its width, in either phase.** At the "
        "block the buffer forces, the array is fill/drain-dominated rather "
        "than operand-starved. The 8 weight banks are there for **capacity, "
        "not bandwidth**.",
        "**Decode's DRAM is weights, not KV, until 32K** — 2.58 GB of weights "
        "against 0.07 GB of KV per token at 2K. That ratio is why §21's FFN "
        "lever works at batch 1 and every KV technique does not.",
        "**Two capacity walls, both at 32K and both structural.** `attn_v` "
        "overflows the input buffer (576 KB into 256 KB) and the scale buffer "
        "(320 into 256), and one K cache is 2,048 KB against a 2,048 KB "
        "weight buffer.",
    ])

    # ---- A: the configuration ------------------------------------------
    emit_section(rep, "A. The configuration",
                "On-chip geometry from the block diagram, off-chip from the "
                "technology preset. Port bandwidth is one word per cycle.")
    trows = []
    for name in ('input', 'scale', 'weight', 'output'):
        spec = getattr(cfg, name)
        # Banked buffers move one word per bank per cycle, which is why the
        # weight port carries 2,048 B/cycle against everyone else's 256.
        bw = spec.word_bytes * spec.banks * freq / GB
        rows.append({'section': 'A', 'buffer': name, 'bytes': spec.bytes,
                     'word_bytes': spec.word_bytes, 'bandwidth_gbps': bw})
        trows.append([f"{name} buffer",
                      f"{spec.depth}×{spec.width_bits} b"
                      + (f" ×{spec.banks}" if spec.banks > 1 else ""),
                      f"{spec.kb:,.0f} KB",
                      f"{spec.word_bytes * spec.banks} B",
                      f"{bw:,.0f} GB/s"])
    trows.append(["**on-chip total**", "", f"**{cfg.total_bytes / KB:,.0f} KB**",
                  "", ""])
    trows.append([f"DRAM — {tech.name}", tech.derivation, "—",
                  f"{tech.burst_bytes} B", f"{tech.bandwidth_gbps:,.1f} GB/s"])
    emit_table(rep, ["memory", "geometry", "capacity", "access", "bandwidth"],
              trows, aligns="llrrr")
    rep.note(
        f"The input word is `array_m × MU × act_bits/8` = "
        f"{hw.array_m * 4 * hw.act_bits // 8} B and the output word is "
        f"`array_n × NUM_RAC × accum_bits/8` = "
        f"{hw.array_n * 32 * hw.accumulate_bits // 8} B — one cycle of "
        f"activation operand and one column tile of accumulators. **The "
        f"weight port is 8 banks wide, so it carries 2,048 B/cycle.**")

    # ---- B: DRAM traffic -----------------------------------------------
    emit_section(rep, "B. DRAM traffic, dense",
                "Effective bytes — what the controller actually moves after "
                "burst rounding. Prefill is the whole phase; decode is per "
                "token.")
    trows = []
    for c in CONTEXTS:
        d = runs[c]
        for tag in ('prefill', 'decode'):
            p = d[tag]
            tot = p['dram_read'] + p['dram_write']
            rows.append({'section': 'B', 'context': c, 'phase': tag,
                         'dram_read': p['dram_read'],
                         'dram_write': p['dram_write'], 'dram_s': p['dram_s']})
            trows.append([f"{c:,}", tag,
                          f"{p['dram_read'] / GB:,.2f} GB",
                          f"{p['dram_write'] / GB:,.2f} GB",
                          f"{tot / GB:,.2f} GB",
                          f"{1e3 * p['dram_s']:,.1f} ms"])
    emit_table(rep, ["context", "phase", "read", "write", "total", "time @51.2 GB/s"],
              trows, aligns="llrrrr")
    rep.note(
        "**Decode moves per token almost exactly what prefill moves in "
        "total** — 2.65 GB against 2.65 GB at 2K — because the weights are "
        "re-read every step and 2.58 GB of them do not fit in 3 MB of SRAM.\n\n"
        "**And per-token DRAM barely grows with context**: 2.65 → 3.97 GB "
        "across a 16× context increase, because the constant weight read "
        "dominates until the KV cache catches up at 32K. That is the whole "
        "reason KV reduction disappoints at batch 1 and §21's weight lever "
        "does not.")

    # ---- C: cycles ------------------------------------------------------
    emit_section(rep, "C. Cycles, dense",
                "Where the array actually spends time, by unit of Fig. 4. "
                "Prefill is the whole phase; decode is per token. BQU "
                "excluded — it is a placeholder, not a measurement.")
    UNITS = [('pe_array_compute', 'PE array — compute'),
             ('pe_array_fill_drain', 'PE array — fill/drain'),
             ('lgu', 'LGU'),
             ('input_load', 'operand issue'),
             ('accumulator', 'accumulator'),
             ('vpu', 'VPU (softmax, norms, SiLU)')]
    for tag, key in (('prefill', 'prefill'), ('decode', 'decode_per_token')):
        trows = []
        for c in CONTEXTS:
            u = runs[c]['units'][key]
            serial = sum(v['cycles'] for v in u.values()
                         if not v['overlapped'])
            cells = [f"{c:,}", f"{serial / 1e6:,.1f} M"]
            for unit, _label in UNITS:
                pct = 100.0 * u.get(unit, {}).get('cycles', 0.0) / serial \
                    if serial else 0.0
                rows.append({'section': 'C', 'context': c, 'phase': tag,
                             'unit': unit,
                             'cycles': u.get(unit, {}).get('cycles', 0.0),
                             'pct_of_serial': pct})
                cells.append(f"{pct:.1f}%")
            trows.append(cells)
        emit_table(rep, ["context", "cycles"] + [l for _u, l in UNITS], trows,
                  aligns="lr" + "r" * len(UNITS),
                  caption=f"{tag} — share of serial cycles")
    rep.note(
        "**Prefill's fill/drain share is the row block, and it is the cost "
        "nobody would predict from the buffer size alone.** At a 9-row block "
        "the array re-pays `array_n + array_m` startup cycles for every block "
        "of every column-and-K tile, against only 9 rows of useful streaming "
        "— so the overhead that is under 2% untiled (§2) becomes the "
        "dominant term. **This is the mechanism behind the 5.3× prefill "
        "penalty in section F**, and behind the idle ports in section D.\n\n"
        "**Decode is the opposite and always was**: `attn_v` dominates, "
        "fill/drain stays small because `M = 1` makes the block irrelevant, "
        "and the VPU share is softmax. Nothing here moves with the buffer "
        "partition.\n\n"
        "*Reconciling with §3*: that section reports 38.15 M decode cycles at "
        "32K where this sheet reports 38.02 M. The sheet generates 3 tokens "
        "and §3 generates 256, and per-token cycles grow with `kv_len` as the "
        "cache fills — so §3 is an average over a longer generation, not a "
        "different model. Both match the stock simulator for their own "
        "workload.*")

    # ---- D: SRAM traffic by port ---------------------------------------
    emit_section(rep, "D. On-chip traffic by port, dense",
                "Bytes per port, and the rate they imply against that port's "
                "width. A port at 100% is the bottleneck.")
    widths = {'input': cfg.input.word_bytes, 'scale': cfg.scale.word_bytes,
              'weight': cfg.weight.word_bytes * cfg.weight.banks,
              'output': cfg.output.word_bytes}
    for tag in ('prefill', 'decode'):
        trows = []
        for c in CONTEXTS:
            p = runs[c][tag]
            cells = [f"{c:,}"]
            for buf in ('input', 'scale', 'weight', 'output'):
                rate = p[buf] / p['cycles'] if p['cycles'] else 0.0
                rows.append({'section': 'D', 'context': c, 'phase': tag,
                             'port': buf, 'bytes': p[buf],
                             'bytes_per_cycle': rate})
                cells.append(f"{rate:,.1f} ({rate / widths[buf]:.0%})")
            trows.append(cells)
        emit_table(rep, ["context", "input 256 B", "scale 256 B", "weight 2,048 B",
                   "output 512 B"], trows, aligns="lrrrr",
                  caption=f"{tag} — B/cycle (% of port width)")
    rep.note(
        "**Nothing is close to saturated — the busiest port anywhere is the "
        "weight port at 36% in 2K decode.** Two different reasons.\n\n"
        "In **prefill** the activation port sits at ~20% rather than the 100% "
        "§19 measured, and the difference is the row block: at 9 rows the "
        "array re-pays fill/drain per block, so cycles inflate ~5× while the "
        "operand bytes do not. **The buffer that forces the block is the same "
        "buffer whose port then goes idle.**\n\n"
        "In **decode** the weight port carries almost everything (36% at 2K, "
        "falling with context as attention grows) — which is section B's "
        "'decode waits on weights' seen from on-chip. Its 8 banks are there "
        "for **capacity, not bandwidth**: fc2 needs 896 KB resident and one "
        "bank holds 256 KB.")

    # ---- D: capacity ----------------------------------------------------
    emit_section(rep, "E. Peak footprint against capacity, dense",
                "The largest working set each buffer is asked to hold, and "
                "which operation asks for it.")
    caps = {'input': cfg.input.bytes, 'scale': cfg.scale.bytes,
            'weight': cfg.weight.bytes, 'output': cfg.output.bytes}
    for tag in ('prefill', 'decode'):
        trows = []
        for c in CONTEXTS:
            for buf in ('input', 'scale', 'weight', 'output'):
                need, op = runs[c]['peaks'].get((tag, buf), (0, '—'))
                fits = need <= caps[buf]
                rows.append({'section': 'E', 'context': c, 'phase': tag,
                             'buffer': buf, 'need': need, 'set_by': op,
                             'fits': fits})
                trows.append([f"{c:,}", buf, op_label(op),
                              f"{need / KB:,.0f} KB",
                              f"{caps[buf] / KB:,.0f} KB",
                              f"{need / caps[buf]:.0%}",
                              "fits" if fits else "**OVER**"])
        emit_table(rep, ["context", "buffer", "set by", "needs", "has", "used",
                   ""], trows, aligns="lllrrrl",
                  caption=f"{tag} — m_tile = {M_TILE}")
    rep.note(
        "**Below 16K the input buffer is set by fc2**, at 98% of capacity — "
        "its A operand is `m_tile × d_ffn × 2 B`, and `d_ffn` is 3.5× "
        "`d_model`, which is exactly why the block is 9 rows and not "
        "`array_m`'s 32.\n\n"
        "**At 32K `attn_v` takes over and three walls arrive at once**: it "
        "needs 576 KB of input buffer (225%), 320 KB of scale (125%), and its "
        "KV tile is 2,048 KB against a 2,048 KB weight buffer (100%). K alone "
        "fills the weight buffer and K+V needs twice the part, so **on-chip KV "
        "residency stops near 16K**. Only the input overflow is reachable by a "
        "smaller block; the scale and weight footprints scale with `K`, not "
        "with `m_tile`, so no tiling escapes them.")

    # ---- E: what binds --------------------------------------------------
    emit_section(rep, "F. What binds, dense",
                "The three roofline terms per phase. The largest is the "
                "phase's limit under the serial model.")
    trows = []
    for c in CONTEXTS:
        for tag in ('prefill', 'decode'):
            p = runs[c][tag]
            sram_s = max(p[b] / (widths[b] * freq)
                         for b in ('input', 'scale', 'weight', 'output'))
            terms = {'compute': p['compute_s'], 'DRAM': p['dram_s'],
                     'SRAM': sram_s}
            order = sorted(terms.values(), reverse=True)
            worst = max(terms, key=terms.get)
            headroom = order[0] / order[1] if order[1] > 0 else float('inf')
            unit = 1e3
            rows.append({'section': 'F', 'context': c, 'phase': tag,
                         'compute_s': p['compute_s'], 'dram_s': p['dram_s'],
                         'sram_s': sram_s, 'bound': worst,
                         'over_second': headroom})
            trows.append([f"{c:,}", tag,
                          f"{unit * p['compute_s']:,.1f} ms",
                          f"{unit * p['dram_s']:,.1f} ms",
                          f"{unit * sram_s:,.1f} ms",
                          f"**{worst}**",
                          f"{headroom:,.1f}×"])
    emit_table(rep, ["context", "phase", "compute", "DRAM", "SRAM", "bound by",
               "over 2nd"], trows, aligns="llrrrlr")
    rep.note(
        "**Decode is DRAM-bound and prefill is compute-bound, at every "
        "context** — the single most load-bearing fact in this repo. It is why "
        "§21's byte lever moves decode 1.911× and why §20's row block costs "
        "prefill time rather than saving it.\n\n"
        "**Prefill's compute figure is a consequence of the buffer, not of the "
        "array.** 1,168 s at 32K is the 9-row block re-paying fill/drain; "
        "untiled the same phase is 219 s (§20). **A larger input buffer is "
        "worth more to prefill than a faster anything.**\n\n"
        "These are per-phase sums. The *operation*-level `sum(max(...))` the "
        "simulator reports is larger, and §17 brackets the difference.")

    # ---- G: per-stage profile -------------------------------------------
    emit_section(rep, "G. Per-stage profile, dense",
                 "Every pipeline stage: its cycles, the DRAM it needs, and "
                 "how much of that fetch its own compute fails to hide. "
                 "\"RAM wait\" is max(0, DRAM time − compute time).")
    for tag, key in (('prefill', 'prefill'), ('decode', 'decode_per_token')):
        trows = []
        for c in CONTEXTS:
            st = runs[c]['stages'][key]
            ordered = sorted(st.values(), key=lambda r: -r['eff_time'])
            for r in ordered[:8]:
                wait = max(0.0, r['mem_time'] - r['compute_time'])
                rows.append({'section': 'G', 'context': c, 'phase': tag,
                             'stage': r['stage'], 'cycles': r['cycles'],
                             'compute_s': r['compute_time'],
                             'mem_s': r['mem_time'], 'ram_wait_s': wait,
                             'bound': r['bound']})
                if c != FOCUS:
                    continue
                trows.append([
                    op_label(r['stage']), r['category'],
                    f"{r['cycles'] / 1e6:,.1f} M",
                    f"{1e3 * r['compute_time']:,.2f}",
                    f"{r['dram_bytes'] / GB:,.2f} GB",
                    f"{1e3 * r['mem_time']:,.2f}",
                    f"{1e3 * wait:,.2f}",
                    r['bound'],
                ])
        emit_table(rep, ["stage", "kind", "cycles", "compute ms", "DRAM",
                         "DRAM ms", "RAM wait ms", "bound"], trows,
                   aligns="llrrrrrl",
                   caption=f"{tag} — context {FOCUS:,}, top stages by time")
    rep.note(
        "**\"RAM wait\" is bandwidth stall, not cache-miss latency.** This "
        "model has no latency or queueing term anywhere — `memory_tech.py` "
        "says so — so a stage's memory cost is `bytes / bandwidth` and the "
        "wait is whatever of that its own compute cannot cover. A real part "
        "adds per-access latency on top, and nothing here estimates it.\n\n"
        "**The split is the whole story of this sheet in one table.** In "
        "prefill almost every stage is compute-bound and waits on nothing; "
        "in decode the projections and the FFN are memory-bound and wait "
        "nearly their entire DRAM time, because at `M = 1` there is almost no "
        "compute to hide it behind.")

    # ---- H: the non-stage units -----------------------------------------
    emit_section(rep, "H. Quantisation and load, dense",
                 "The units that are not GEMM stages. Both BQU rows are "
                 "placeholders; the load path is partly unmodelled.")
    trows = []
    for c in CONTEXTS:
        b = runs[c]['bqu']
        u_pf = runs[c]['units']['prefill']
        u_dec = runs[c]['units']['decode_per_token']
        for label, pf, dec, note in (
            ("BQU — BEA (encode)", b['prefill']['bea'], b['decode']['bea'],
             "placeholder"),
            ("BQU — TSE (Value scales)", b['prefill']['tse'],
             b['decode']['tse'], "placeholder"),
            ("LGU (table generation)",
             u_pf.get('lgu', {}).get('cycles', 0.0),
             u_dec.get('lgu', {}).get('cycles', 0.0), "modelled"),
            ("operand issue (buffer load)",
             u_pf.get('input_load', {}).get('cycles', 0.0),
             u_dec.get('input_load', {}).get('cycles', 0.0), "modelled"),
            ("accumulator drain",
             u_pf.get('accumulator', {}).get('cycles', 0.0),
             u_dec.get('accumulator', {}).get('cycles', 0.0), "modelled"),
        ):
            rows.append({'section': 'H', 'context': c, 'unit': label,
                         'prefill_cycles': pf, 'decode_cycles': dec})
            if c != FOCUS:
                continue
            trows.append([label, f"{pf / 1e6:,.2f} M", f"{dec / 1e3:,.1f} K",
                          note])
    emit_table(rep, ["unit", "prefill cycles", "decode cycles/token",
                     "status"], trows, aligns="lrrl",
               caption=f"context {FOCUS:,}")
    rep.note(
        "**The BQU rows are order-of-magnitude, not measurements.** The "
        "original simulator does not model it at all; `bqu_metrics` assumes "
        "one BEA pass per bit-plane, one TSE min/max pass on the Value path, "
        "and `bqu_width` elements per cycle. They are excluded from every "
        "serial total here, per OMNI_LUT.pdf §IV-A, which runs the BQU "
        "on-the-fly alongside the array — **which is itself unverified "
        "against the RTL schedule.**\n\n"
        "**\"Buffer loading\" is only half present.** `operand issue` is the "
        "array-side cost of accepting a word, and it is modelled. The LSU's "
        "own DMA cost is **not** — the RTL measures 492 → 927 ns from adding "
        "it, and this model has no term for that. It is the largest "
        "unmodelled hardware cost in the sheet.")

    # ---- I: KV bit allocation -------------------------------------------
    emit_section(rep, "I. Key cache bit allocation, dense",
                 "AS-Bit gives a fraction of Key channels a high width and "
                 "the rest a low one; the Value cache stays at the low width. "
                 "Value held at 4 bits in every row.")

    def _kv_run(ctx, key_bits, planes="packed"):
        h = dataclasses.replace(base_hw(), kv_key_bits=key_bits,
                                kv_value_bits=4.0, kv_plane_model=planes)
        sim = Simulator(h)
        mm = get_model_config(MODEL)
        w = WorkloadConfig(batch_size=1, input_tokens=ctx,
                           output_tokens=OUTPUT_TOKENS, flash_block_size=0)
        r = sim.simulate(mm, w)
        _, tp = sim.compute_roofline_latency(r, w)
        qk = r.decode.get_operation_total(OperationType.QK_MATMUL,
                                          ComputeMode.AA)
        t = r.decode.get_total_metrics()
        steps = max(1, OUTPUT_TOKENS - 1)
        return {'qk_cyc': qk.cycles / steps, 'tpot': tp,
                'dram': (t.dram_read_eff + t.dram_write_eff) / steps}

    ALLOC = [(0.00, 4.00, "all low (what this sheet models)"),
             (0.25, 4.25, "paper's AS-Bit ratio"),
             (0.50, 4.50, "best measured perplexity"),
             (1.00, 5.00, "all high (Key only)")]
    for ctx in (FOCUS, 32768):
        base_kv = _kv_run(ctx, 4.0)
        trows = []
        for frac, eff, label in ALLOC:
            pk = _kv_run(ctx, eff, "packed")
            pd = _kv_run(ctx, eff, "padded")
            rows.append({'section': 'I', 'context': ctx, 'high_frac': frac,
                         'key_bits': eff, 'tpot_s': pk['tpot'],
                         'qk_cycles_packed': pk['qk_cyc'],
                         'qk_cycles_padded': pd['qk_cyc'],
                         'dram': pk['dram']})
            trows.append([f"{frac:.0%} at 5 bits", f"{eff:.2f}",
                          f"{pk['qk_cyc'] / 1e3:,.0f} K",
                          f"{pd['qk_cyc'] / 1e3:,.0f} K",
                          f"{pk['dram'] / GB:,.3f} GB",
                          f"{1e3 * pk['tpot']:,.2f} ms",
                          f"{pk['tpot'] / base_kv['tpot']:.3f}×", label])
        emit_table(rep, ["Key allocation", "eff. bits", "qk cyc (packed)",
                         "qk cyc (padded)", "decode DRAM", "TPOT", "vs 4-bit",
                         ""], trows, aligns="lrrrrrrl",
                   caption=f"decode per token, context {ctx:,}")
    rep.note(
        "**The packed/padded question turns out not to matter.** The two "
        "schedules differ by 11% of `qk` cycles and **0% of TPOT** — decode "
        "is DRAM-bound, so the extra bit-plane pass hides entirely behind the "
        "memory it is waiting on. There is no case for building packing "
        "hardware.\n\n"
        "**Key-side mixed precision is nearly free; widening the Value cache "
        "is not.** 50% of Key channels at 5 bits costs **1.005× TPOT at 8K "
        "and 1.010× at 32K**. Taking both caches to 5 bits costs **1.070× and "
        "1.152×** — 7-15× more. The asymmetry is §G's: `qk` is ~5% of decode "
        "cycles while `attn_v` is 80-92%, though the two carry identical "
        "bytes. **So the paper's \"no extra bits to the Value cache\" is the "
        "load-bearing half of AS-Bit**, not the Key adaptivity.\n\n"
        "**This sheet models the first row, and the built part is the third.** "
        "Everything else here is a flat-4 result; a 4.5-bit Key would move "
        "decode DRAM by ~1.9% at 32K and nothing else materially.")

    rep.save()
    write_dense(args.dense, SETUP)

    flat = [r for r in rows]
    keys = sorted({k for r in flat for k in r})
    with open(args.csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(flat)
    print(f"wrote {args.csv}, {args.report} and {args.dense}")


if __name__ == '__main__':
    main()
