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
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, '..', '..'))
for p in ('simulator', 'analysis', 'analysis/cycle_breakdown',
          'analysis/memory'):
    sys.path.insert(0, os.path.join(_root, *p.split('/')))

from simulator import (                                              # noqa: E402
    HardwareConfig, WorkloadConfig, Simulator, ComputeMode,
)
from buffer_tech import (                                            # noqa: E402
    buffer_config, with_buffer_config, DEFAULT_BUFFER_CONFIG,
)
from memory_tech import (                                            # noqa: E402
    memory_technology, with_memory_technology,
)
from model_configs import get_model_config                           # noqa: E402
from report import Report                                            # noqa: E402

MODEL = 'LLaMA-3-8B'
TECH = 'DDR5-6400'
CONTEXTS = [2048, 8192, 32768]
OUTPUT_TOKENS = 4
KV_BITS = 4
SCORE_SRAM_KB = 128
M_TILE = 9          # §22: the largest block every operand fits
KB = 1024.0
MB = KB * KB
GB = 1e9


class BufferProbe(Simulator):
    """Records each operation's per-buffer footprint, so the report can say
    *which* operation sets each buffer's peak rather than only how big it is.

    Analysis-only: it overrides nothing the simulator computes, it just keeps
    the footprint `_simulate_matmul` already asks for.
    """

    def __init__(self, hw):
        super().__init__(hw)
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
           'ttft_s': ttft, 'tpot_s': tpot, 'peaks': sim.peaks}
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
    args = ap.parse_args()

    cfg = buffer_config(DEFAULT_BUFFER_CONFIG)
    tech = memory_technology(TECH)
    hw = base_hw()
    freq = hw.freq_mhz * 1e6
    runs = {c: run(c) for c in CONTEXTS}
    rows = []

    rep = Report(
        args.report,
        "RAM model — dense baseline",
        subtitle="What the built configuration does with nothing switched on",
        source="analysis/memory/ram_model_run.py",
        setup=[f"{MODEL}, batch 1, {OUTPUT_TOKENS - 1} decode steps, standard "
               f"attention, W4A16KV{KV_BITS}, scores staged on chip.",
               f"On-chip: the RTL's four SRAMs. Off-chip: {TECH}. Row block "
               f"`sram_m_tile = {M_TILE}` (§22 — the largest every operand "
               f"fits). Nothing else is non-default."])

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
    rep.section("A. The configuration",
                "On-chip geometry from the block diagram, off-chip from the "
                "technology preset. Port bandwidth is one word per cycle.")
    trows = []
    for name in ('input', 'scale', 'weight', 'output'):
        spec = getattr(cfg, name)
        bw = spec.word_bytes * freq / GB
        rows.append({'section': 'A', 'buffer': name, 'bytes': spec.bytes,
                     'word_bytes': spec.word_bytes, 'bandwidth_gbps': bw})
        trows.append([f"{name} buffer",
                      f"{spec.depth}×{spec.width_bits} b"
                      + (f" ×{spec.banks}" if spec.banks > 1 else ""),
                      f"{spec.kb:,.0f} KB", f"{spec.word_bytes} B",
                      f"{bw:,.0f} GB/s"])
    trows.append(["**on-chip total**", "", f"**{cfg.total_bytes / KB:,.0f} KB**",
                  "", ""])
    trows.append([f"DRAM — {tech.name}", tech.derivation, "—",
                  f"{tech.burst_bytes} B", f"{tech.bandwidth_gbps:,.1f} GB/s"])
    rep.table(["memory", "geometry", "capacity", "access", "bandwidth"],
              trows, aligns="llrrr")
    rep.note(
        f"The input word is `array_m × MU × act_bits/8` = "
        f"{hw.array_m * 4 * hw.act_bits // 8} B and the output word is "
        f"`array_n × NUM_RAC × accum_bits/8` = "
        f"{hw.array_n * 32 * hw.accumulate_bits // 8} B — one cycle of "
        f"activation operand and one column tile of accumulators. **The "
        f"weight port is 8 banks wide, so it carries 2,048 B/cycle.**")

    # ---- B: DRAM traffic -----------------------------------------------
    rep.section("B. DRAM traffic, dense",
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
    rep.table(["context", "phase", "read", "write", "total", "time @51.2 GB/s"],
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

    # ---- C: SRAM traffic by port ---------------------------------------
    rep.section("C. On-chip traffic by port, dense",
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
                rows.append({'section': 'C', 'context': c, 'phase': tag,
                             'port': buf, 'bytes': p[buf],
                             'bytes_per_cycle': rate})
                cells.append(f"{rate:,.1f} ({rate / widths[buf]:.0%})")
            trows.append(cells)
        rep.table(["context", "input 256 B", "scale 256 B", "weight 2,048 B",
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
    rep.section("D. Peak footprint against capacity, dense",
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
                rows.append({'section': 'D', 'context': c, 'phase': tag,
                             'buffer': buf, 'need': need, 'set_by': op,
                             'fits': fits})
                trows.append([f"{c:,}", buf, op,
                              f"{need / KB:,.0f} KB",
                              f"{caps[buf] / KB:,.0f} KB",
                              f"{need / caps[buf]:.0%}",
                              "fits" if fits else "**OVER**"])
        rep.table(["context", "buffer", "set by", "needs", "has", "used",
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
    rep.section("E. What binds, dense",
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
            rows.append({'section': 'E', 'context': c, 'phase': tag,
                         'compute_s': p['compute_s'], 'dram_s': p['dram_s'],
                         'sram_s': sram_s, 'bound': worst,
                         'over_second': headroom})
            trows.append([f"{c:,}", tag,
                          f"{unit * p['compute_s']:,.1f} ms",
                          f"{unit * p['dram_s']:,.1f} ms",
                          f"{unit * sram_s:,.1f} ms",
                          f"**{worst}**",
                          f"{headroom:,.1f}×"])
    rep.table(["context", "phase", "compute", "DRAM", "SRAM", "bound by",
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

    rep.save()

    flat = [r for r in rows]
    keys = sorted({k for r in flat for k in r})
    with open(args.csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(flat)
    print(f"wrote {args.csv} and {args.report}")


if __name__ == '__main__':
    main()
