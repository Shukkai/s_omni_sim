"""
Figures for the memory sections of `study.md` (§12 onward), which until now
carried tables only.

Each panel is drawn from the CSV its sweep already writes, so the figures cannot
drift from the reports -- re-run the sweep, re-run this, and both move together.
Nine figures, one per claim that a table states less clearly than a picture:

    overlap.png        the serial/pipelined bracket, and where it is widest
    kv_batch.png       KV technique speedup against batch -- the fan-out
    packing.png        OS-V packing saturating at P=8
    unstructured.png   the layout/axis antisymmetry
    memory_tech.png    the burst cliff moving with the memory technology
    ports.png          which memory moves the bytes, and the 4.35x it explains
    tiling.png         the capacity/latency frontier prefill was hiding
    sparsity.png       activation sparsity, and the batch that spends it
    buffers.png        the RTL's four SRAMs against the model's one pool

The last four read `ports.csv`, `tiling.csv`, `buffers.csv` and
`../act_sparsity/sparsity.csv`.

Usage:
    python plot_memory.py                # all nine, into this directory
    python plot_memory.py --only overlap
    python plot_memory.py --outdir figs
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402

# --- Style, matching analysis/cycle_breakdown/plot_cycle_breakdown.py --------
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['DejaVu Serif']
matplotlib.rcParams['axes.labelcolor'] = 'black'
matplotlib.rcParams['xtick.color'] = 'black'
matplotlib.rcParams['ytick.color'] = 'black'
matplotlib.rcParams['text.color'] = 'black'
matplotlib.rcParams['axes.titlesize'] = 12
matplotlib.rcParams['axes.labelsize'] = 11
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10
matplotlib.rcParams['legend.fontsize'] = 9

AW = '#51827B'     # green  -- AW GEMM / compute / "good" series
AA = '#E68B88'     # red    -- AA GEMM / memory
GREY = '#B0B0B0'
DARK = '#2F4F4F'
ACCENT = '#C97B3C'

_here = os.path.dirname(os.path.abspath(__file__))
_pack_dir = os.path.abspath(os.path.join(_here, '..', 'array_packing'))
_spar_dir = os.path.abspath(os.path.join(_here, '..', 'act_sparsity'))


def load(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def fnum(row, key, default=0.0):
    v = row.get(key, '')
    return float(v) if v not in ('', None) else default


def save(fig, outdir, name):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(outdir, f'{name}.{ext}'), dpi=200,
                    bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf")


# ============================================================================
# 1. Overlap
# ============================================================================

def fig_overlap(outdir):
    rows = [r for r in load(os.path.join(_here, 'overlap.csv'))
            if r['section'] == 'A']
    batches = sorted({int(r['batch']) for r in rows})
    ctxs = sorted({int(r['context']) for r in rows})

    fig, axes = plt.subplots(1, 2, figsize=(11, 4),
                             gridspec_kw={'width_ratios': [1.25, 1]})

    # (a) stacked view of the two roofs at batch 1, with both models marked
    ax = axes[0]
    b1 = {int(r['context']): r for r in rows if int(r['batch']) == 1}
    x = np.arange(len(ctxs))
    comp = [fnum(b1[c], 'serial_decode_compute_s') * 1e3 for c in ctxs]
    dram = [fnum(b1[c], 'serial_decode_dram_s') * 1e3 for c in ctxs]
    ser = [fnum(b1[c], 'serial_tpot') * 1e3 for c in ctxs]
    pipe = [fnum(b1[c], 'pipelined_tpot') * 1e3 for c in ctxs]

    ax.bar(x - 0.19, comp, 0.36, color=AW, label='compute time')
    ax.bar(x + 0.19, dram, 0.36, color=AA, label='DRAM time')
    ax.plot(x, ser, 'o--', color=DARK, ms=6, lw=1.6,
            label='serial  = sum(max)')
    ax.plot(x, pipe, 's-', color=ACCENT, ms=6, lw=1.8,
            label='pipelined = max(sum)')
    for xi, (s, p) in enumerate(zip(ser, pipe)):
        ax.annotate(f'{s/p:.2f}x', (xi, (s + p) / 2), ha='center',
                    va='center', fontsize=9, color=DARK,
                    bbox=dict(boxstyle='round,pad=0.18', fc='white',
                              ec=GREY, lw=0.6))
    ax.set_xticks(x)
    ax.set_xticklabels([f'{c//1024}K' for c in ctxs])
    ax.set_xlabel('context')
    ax.set_ylabel('decode time per token (ms)')
    ax.set_title('(a) Batch 1 — the two roofs meet at 32K')
    ax.legend(frameon=False, loc='upper left')
    ax.grid(axis='y', ls=':', color=GREY, lw=0.6)
    ax.set_axisbelow(True)

    # (b) the gap across the whole grid
    ax = axes[1]
    width = 0.8 / len(batches)
    for i, b in enumerate(batches):
        by = {int(r['context']): r for r in rows if int(r['batch']) == b}
        gaps = [fnum(by[c], 'serial_tpot') / fnum(by[c], 'pipelined_tpot')
                for c in ctxs]
        ax.bar(np.arange(len(ctxs)) + (i - (len(batches) - 1) / 2) * width,
               gaps, width * 0.9, label=f'batch {b}',
               color=[AA, ACCENT, AW][i % 3])
    ax.axhline(2.0, color=DARK, ls='--', lw=1.2)
    ax.annotate('2.00x — hard ceiling, reached when the roofs are equal',
                (len(ctxs) - 0.5, 2.0), xytext=(0, 5),
                textcoords='offset points', ha='right', fontsize=8.5,
                color=DARK)
    ax.axhline(1.0, color=GREY, lw=0.8)
    ax.set_xticks(np.arange(len(ctxs)))
    ax.set_xticklabels([f'{c//1024}K' for c in ctxs])
    ax.set_xlabel('context')
    ax.set_ylabel('serial / pipelined')
    ax.set_ylim(1.0, 2.15)
    ax.set_title('(b) How much "no overlap" overstates')
    ax.legend(frameon=False)
    ax.grid(axis='y', ls=':', color=GREY, lw=0.6)
    ax.set_axisbelow(True)

    fig.suptitle('Compute/memory overlap brackets every latency number',
                 fontsize=13, y=1.02)
    save(fig, outdir, 'overlap')


# ============================================================================
# 2. KV technique speedup vs batch
# ============================================================================

def fig_kv_batch(outdir):
    rows = [r for r in load(os.path.join(_here, 'batch_scaling.csv'))
            if r['section'] == 'B']
    ctxs = sorted({int(r['context']) for r in rows})
    fig, axes = plt.subplots(1, len(ctxs), figsize=(5.2 * len(ctxs), 4),
                             sharey=True)
    if len(ctxs) == 1:
        axes = [axes]

    for ax, ctx in zip(axes, ctxs):
        sub = [r for r in rows if int(r['context']) == ctx]
        techs = sorted({r['technique'] for r in sub})
        for i, t in enumerate(techs):
            pts = sorted(((int(r['batch']), fnum(r, 'speedup'))
                          for r in sub if r['technique'] == t))
            style = '--' if 'think' in t.lower() else '-'
            ax.plot([p[0] for p in pts], [p[1] for p in pts], style,
                    marker='o', ms=5, lw=1.8, label=t)
        ax.axhline(1.0, color=GREY, lw=0.8)
        ax.set_xscale('log', base=2)
        ax.set_yscale('log', base=2)
        ax.set_xticks(sorted({int(r['batch']) for r in sub}))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel('batch size')
        ax.set_title(f'context {ctx:,}')
        ax.grid(ls=':', color=GREY, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('decode TPOT speedup vs dense')
    axes[-1].legend(frameon=False, loc='upper left')
    fig.suptitle('Entry-count techniques fan out with batch; channel pruning '
                 'does not', fontsize=13, y=1.01)
    save(fig, outdir, 'kv_batch')


# ============================================================================
# 3. OS-V packing
# ============================================================================

def fig_packing(outdir):
    rows = [r for r in load(os.path.join(_pack_dir, 'pack.csv'))
            if r['section'] == 'C']
    batches = sorted({int(r['batch']) for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    for i, b in enumerate(batches):
        pts = sorted(((int(r['pack']), fnum(r, 'tpot_s'))
                      for r in rows if int(r['batch']) == b))
        base = pts[0][1]
        ax.plot([p[0] for p in pts], [base / p[1] for p in pts], '-o', ms=5,
                lw=1.8, color=[AA, ACCENT, AW][i % 3], label=f'batch {b}')
    ax.axvline(8, color=DARK, ls='--', lw=1.2)
    ax.annotate('P=8: ceiling', (8, 1.1), xytext=(6, 0),
                textcoords='offset points', fontsize=9, color=DARK)
    ax.set_xscale('log', base=2)
    ax.set_xticks(sorted({int(r['pack']) for r in rows}))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel('pack factor P')
    ax.set_ylabel('decode TPOT speedup')
    ax.set_title('(a) TPOT saturates at P=8')
    ax.legend(frameon=False)
    ax.grid(ls=':', color=GREY, lw=0.6)
    ax.set_axisbelow(True)

    # (b) the SRAM cost, which is what makes P=8 the meeting point
    ax = axes[1]
    d = [r for r in load(os.path.join(_pack_dir, 'pack.csv'))
         if r['section'] == 'D']
    for variant, colour, lbl in (('independent', AA, 'independent'),
                                 ('gqa_shared', AW, 'GQA-shared')):
        pts = sorted(((int(r['pack']), fnum(r, 'decode_peak_sram') / 2**20)
                      for r in d if r.get('variant') == variant))
        ax.plot([p[0] for p in pts], [p[1] for p in pts], '-o', ms=5, lw=1.8,
                color=colour, label=lbl)
    ax.axhline(16, color=DARK, ls='--', lw=1.2)
    ax.annotate('16 MB budget', (1, 16), xytext=(2, 4),
                textcoords='offset points', fontsize=9, color=DARK)
    ax.axvline(8, color=DARK, ls='--', lw=1.2)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    ax.set_xticks(sorted({int(r['pack']) for r in d}))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel('pack factor P')
    ax.set_ylabel('decode peak SRAM (MB)')
    ax.set_title('(b) …and P=8 GQA-shared still fits')
    ax.legend(frameon=False)
    ax.grid(ls=':', color=GREY, lw=0.6)
    ax.set_axisbelow(True)

    fig.suptitle('The speedup ceiling and the capacity limit meet at P=8',
                 fontsize=13, y=1.02)
    save(fig, outdir, 'packing')


# ============================================================================
# 4. Unstructured masks -- the layout antisymmetry
# ============================================================================

def fig_unstructured(outdir):
    all_rows = load(os.path.join(_here, 'unstructured.csv'))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # (a) the 2x2 antisymmetry
    ax = axes[0]
    c = [r for r in all_rows if r['section'] == 'C']
    layouts = ['token_major', 'channel_major']
    axes_lbl = ['token-wise 50%', 'channel-wise 50%']
    width = 0.36
    x = np.arange(len(layouts))
    for i, axis_name in enumerate(axes_lbl):
        vals = []
        for lay in layouts:
            m = [r for r in c if r['layout'] == lay and r['axis'] == axis_name]
            vals.append(fnum(m[0], 'efficiency') * 100 if m else 0.0)
        bars = ax.bar(x + (i - 0.5) * width, vals, width * 0.9,
                      color=[AA, AW][i], label=axis_name)
        for b, v in zip(bars, vals):
            ax.annotate(f'{v:.0f}%', (b.get_x() + b.get_width() / 2, v),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace('_', '-') for l in layouts])
    ax.set_ylabel('% of the byte saving actually collected')
    ax.set_ylim(0, 118)
    ax.set_title('(a) Each layout kills the other axis')
    ax.legend(frameon=False, loc='upper center')
    ax.grid(axis='y', ls=':', color=GREY, lw=0.6)
    ax.set_axisbelow(True)

    # (b) the cliff against channel group
    ax = axes[1]
    b = sorted(((int(r['channel_group']), fnum(r, 'efficiency') * 100)
                for r in all_rows if r['section'] == 'B'))
    ax.step([p[0] for p in b], [p[1] for p in b], where='post', lw=2,
            color=AA)
    ax.plot([p[0] for p in b], [p[1] for p in b], 'o', ms=5, color=AA)
    ax.set_xscale('log', base=2)
    ax.set_xticks([p[0] for p in b])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel('contiguous retained channels per run')
    ax.set_ylabel('% of the byte saving collected')
    ax.set_ylim(-6, 118)
    ax.annotate('nothing below a whole 64 B burst',
                (2, 6), fontsize=9, color=DARK)
    ax.set_title('(b) A cliff, not a slope (token-major, DDR5)')
    ax.grid(ls=':', color=GREY, lw=0.6)
    ax.set_axisbelow(True)

    fig.suptitle('Unstructured masks: the layout decides which axis may work',
                 fontsize=13, y=1.02)
    save(fig, outdir, 'unstructured')


# ============================================================================
# 5. Memory technology
# ============================================================================

def fig_memory_tech(outdir):
    all_rows = load(os.path.join(_here, 'bandwidth.csv'))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # (a) the cliff moving with the burst
    ax = axes[0]
    b = [r for r in all_rows if r['section'] == 'B']
    for tech, colour in (('DDR5-6400', AA), ('HBM3', AW)):
        pts = sorted(((int(r['channel_group']), fnum(r, 'saving_kept') * 100)
                      for r in b if r['tech'] == tech))
        burst = next((int(fnum(r, 'burst_bytes')) for r in all_rows
                      if r['section'] == 'A' and r['tech'] == tech), 0)
        ax.step([p[0] for p in pts], [p[1] for p in pts], where='post', lw=2,
                color=colour, label=f'{tech} ({burst} B burst)')
        ax.plot([p[0] for p in pts], [p[1] for p in pts], 'o', ms=5,
                color=colour)
    ax.set_xscale('log', base=2)
    ax.set_xticks(sorted({int(r['channel_group']) for r in b}))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel('contiguous retained channels per run')
    ax.set_ylabel('% of the byte saving collected')
    ax.set_ylim(-6, 118)
    ax.set_title('(a) Halving the burst halves the group needed')
    ax.legend(frameon=False, loc='upper left')
    ax.grid(ls=':', color=GREY, lw=0.6)
    ax.set_axisbelow(True)

    # (b) bandwidth buys almost nothing
    ax = axes[1]
    c = [r for r in all_rows if r['section'] == 'C' and r['technique']]
    techs, seen = [], set()
    for r in c:
        if r['tech'] not in seen:
            seen.add(r['tech'])
            techs.append(r['tech'])
    techs.sort(key=lambda t: fnum(
        next(r for r in all_rows if r['section'] == 'A' and r['tech'] == t),
        'bandwidth_gbps'))
    bw = [fnum(next(r for r in all_rows
                    if r['section'] == 'A' and r['tech'] == t),
               'bandwidth_gbps') for t in techs]
    dense = []
    for t in techs:
        row = next(r for r in c if r['tech'] == t)
        dense.append(fnum(row, 'tpot_s') * fnum(row, 'speedup') * 1e3)
    # Categorical rather than a log axis: the technologies are six discrete
    # parts, and on a log axis their names collide into an unreadable pile.
    x = np.arange(len(techs))
    colours = [AA if b < 200 else AW for b in bw]
    ax.bar(x, dense, 0.62, color=colours)
    ax.axhline(dense[0], color=DARK, ls='--', lw=1.2)
    for xi, (y_, b_) in enumerate(zip(dense, bw)):
        ax.annotate(f'{dense[0]/y_:.2f}x', (xi, y_), xytext=(0, 4),
                    textcoords='offset points', ha='center', fontsize=9,
                    color=DARK)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{t}\n{b:,.0f} GB/s' for t, b in zip(techs, bw)],
                       fontsize=8, rotation=30, ha='right',
                       rotation_mode='anchor')
    ax.set_ylabel('dense decode TPOT (ms), batch 32')
    ax.set_ylim(0, max(dense) * 1.16)
    ax.set_title('(b) 16x the bandwidth buys 1.10x')
    ax.grid(axis='y', ls=':', color=GREY, lw=0.6)
    ax.set_axisbelow(True)

    fig.suptitle('The memory part decides which pruning axis works — '
                 'not how fast decode runs', fontsize=13, y=1.02)
    save(fig, outdir, 'memory_tech')


# ============================================================================
# 6. Per-port SRAM (§19)
# ============================================================================

def fig_ports(outdir):
    all_rows = load(os.path.join(_here, 'ports.csv'))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # (a) where the bytes go, by port -- the accumulator dominating prefill
    ax = axes[0]
    a = [r for r in all_rows if r['section'] == 'A']
    ports = ['activation (A read)', 'weight (B read)',
             'accumulator (partial sums)', 'activation (result write)']
    labels = ['A read', 'B read', 'accumulator', 'result write']
    colors = [AW, GREY, AA, DARK]
    width = 0.36
    x = np.arange(2)
    bottom = np.zeros(2)
    for port, lbl, col in zip(ports, labels, colors):
        vals = []
        for phase in ('prefill', 'decode'):
            m = [r for r in a if r['phase'] == phase and r['port'] == port]
            vals.append(fnum(m[0], 'share') * 100 if m else 0.0)
        vals = np.array(vals)
        ax.bar(x, vals, width * 1.6, bottom=bottom, color=col, label=lbl,
               edgecolor='white', lw=0.6)
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v > 6:
                ax.annotate(f'{v:.0f}%', (xi, b + v / 2), ha='center',
                            va='center', fontsize=9,
                            color='white' if col != GREY else 'black')
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(['prefill', 'decode'])
    ax.set_ylabel('share of the lumped SRAM traffic')
    ax.set_ylim(0, 108)
    ax.set_title('(a) The lump is mostly the accumulator')
    ax.legend(frameon=False, fontsize=8, ncol=2, loc='upper center')
    ax.grid(axis='y', ls=':', color=GREY, lw=0.6)
    ax.set_axisbelow(True)

    # (b) lumped vs ported TTFT
    ax = axes[1]
    b = sorted(((fnum(r, 'sram_bw'), fnum(r, 'ttft_lumped'),
                 fnum(r, 'ttft_ported'))
                for r in all_rows if r['section'] == 'B'),
               key=lambda t: (t[0] == 0, t[0]))
    ref = [t for t in b if t[0] == 0][0][1]
    finite = [t for t in b if t[0] > 0]
    xs = [t[0] for t in finite]
    ax.plot(xs, [t[1] / ref for t in finite], 'o-', lw=2, color=AA,
            label='lumped (one port)')
    ax.plot(xs, [t[2] / ref for t in finite], 's-', lw=2, color=AW,
            label='ported (three memories)')
    ax.axhline(1.0, ls='--', lw=1, color=GREY)
    ax.set_xscale('log', base=2)
    ax.set_xticks(xs)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel('per-port SRAM bandwidth (GB/s)')
    ax.set_ylabel('TTFT vs unlimited')
    ax.annotate('4.35x -> 1.16x\nprefill unparked', (128, 3.1), fontsize=9,
                color=DARK)
    ax.set_title('(b) What billing the right memory is worth')
    ax.legend(frameon=False)
    ax.grid(ls=':', color=GREY, lw=0.6)
    ax.set_axisbelow(True)

    fig.suptitle('Per-port SRAM: the accumulator was billed to the wrong memory',
                 fontsize=13, y=1.02)
    save(fig, outdir, 'ports')


# ============================================================================
# 7. Prefill row tiling (§20)
# ============================================================================

def fig_tiling(outdir):
    all_rows = load(os.path.join(_here, 'tiling.csv'))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    MB = 1024.0 * 1024.0

    # (a) the frontier: footprint against TTFT
    ax = axes[0]
    a = [r for r in all_rows if r['section'] == 'A']
    untiled = [r for r in a if int(r['m_tile']) == 0][0]
    ref_t = fnum(untiled, 'ttft_s')
    pts = sorted(((int(r['m_tile']), fnum(r, 'prefill_peak') / MB,
                   fnum(r, 'ttft_s') / ref_t)
                  for r in a if int(r['m_tile']) > 0), key=lambda t: t[0])
    ax.plot([p[1] for p in pts], [p[2] for p in pts], 'o-', lw=2, color=AW)
    for mt, peak, tt in pts:
        if mt in (512, 32, 8):
            ax.annotate(f'{mt}', (peak, tt), xytext=(6, 5),
                        textcoords='offset points', fontsize=9, color=DARK)
    ax.plot([fnum(untiled, 'prefill_peak') / MB], [1.0], '*', ms=14, color=AA)
    ax.annotate('untiled\n2.16 GB',
                (fnum(untiled, 'prefill_peak') / MB, 1.0),
                xytext=(-30, 16), textcoords='offset points', fontsize=9,
                color=AA)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    ax.set_xlabel('prefill working set (MB, log)')
    ax.set_ylabel('TTFT vs untiled (log)')
    ax.set_title('(a) A frontier, labelled by row block')
    ax.grid(which='both', ls=':', color=GREY, lw=0.6)
    ax.set_axisbelow(True)

    # (b) the table §7 could not publish
    ax = axes[1]
    b = sorted(((fnum(r, 'capacity_kb'), int(r['m_tile']),
                 fnum(r, 'ttft_s') / ref_t)
                for r in all_rows
                if r['section'] == 'B' and r.get('m_tile') not in ('', None)))
    caps = [t[0] / 1024 for t in b]
    bars = ax.bar(range(len(b)), [t[2] for t in b], 0.6, color=AW)
    for i, (bar, t) in enumerate(zip(bars, b)):
        ax.annotate(f'{int(t[1])} rows',
                    (bar.get_x() + bar.get_width() / 2, t[2]),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', fontsize=9)
    ax.axhline(1.0, ls='--', lw=1, color=GREY)
    ax.set_xticks(range(len(b)))
    ax.set_xticklabels([f'{c:.0f} MB' for c in caps])
    ax.set_xlabel('SRAM capacity')
    ax.set_ylabel('TTFT vs untiled')
    ax.set_title('(b) Largest block that fits, and what it costs')
    ax.grid(axis='y', ls=':', color=GREY, lw=0.6)
    ax.set_axisbelow(True)

    fig.suptitle('Prefill row tiling: capacity and latency trade smoothly',
                 fontsize=13, y=1.02)
    save(fig, outdir, 'tiling')


# ============================================================================
# 8. FFN activation sparsity (§21)
# ============================================================================

def fig_sparsity(outdir):
    all_rows = load(os.path.join(_spar_dir, 'sparsity.csv'))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # (a) speedup against density, batch 1
    ax = axes[0]
    a = sorted(((fnum(r, 'density'), fnum(r, 'tpot_s'))
                for r in all_rows if r['section'] == 'A'), reverse=True)
    dense = [t[1] for t in a if t[0] == 1.0][0]
    ax.plot([t[0] * 100 for t in a], [dense / t[1] for t in a], 'o-', lw=2,
            color=AW)
    ax.axhline(1.0, ls='--', lw=1, color=GREY)
    ax.annotate('16x the DRAM bandwidth\nbuys 1.10x (§16b)', (5.6, 1.10),
                fontsize=9, color=DARK)
    ax.set_xscale('log', base=2)
    ax.set_xticks([t[0] * 100 for t in a])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel('FFN hidden units active per token (%)')
    ax.set_ylabel('decode TPOT speedup, batch 1')
    ax.set_title('(a) It works, and saturates against attention')
    ax.grid(ls=':', color=GREY, lw=0.6)
    ax.set_axisbelow(True)

    # (b) batch spends it
    ax = axes[1]
    d = sorted(((int(r['batch']), fnum(r, 'tpot_dense'),
                 fnum(r, 'tpot_shared'), fnum(r, 'tpot_pertoken'),
                 fnum(r, 'union_density'))
                for r in all_rows if r['section'] == 'D'))
    xs = [t[0] for t in d]
    ax.plot(xs, [t[1] / t[2] for t in d], 's--', lw=2, color=GREY,
            label='shared mask (bound)')
    ax.plot(xs, [t[1] / t[3] for t in d], 'o-', lw=2, color=AW,
            label='per-token (real)')
    ax.axhline(1.0, ls='--', lw=1, color=GREY)
    ax.set_xscale('log', base=2)
    ax.set_xticks(xs)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel('batch')
    ax.set_ylabel('decode TPOT speedup at 10% density')
    ax.annotate('the union $1-(1-d)^M$\nplus compute-bound decode',
                (2.1, 1.35), fontsize=9, color=DARK)
    ax.set_title('(b) Batch spends what sparsity buys')
    ax.legend(frameon=False)
    ax.grid(ls=':', color=GREY, lw=0.6)
    ax.set_axisbelow(True)

    fig.suptitle('FFN activation sparsity: the first lever that moves decode',
                 fontsize=13, y=1.02)
    save(fig, outdir, 'sparsity')


# ============================================================================
# 9. The RTL buffer partition (§22)
# ============================================================================

def fig_buffers(outdir):
    all_rows = load(os.path.join(_here, 'buffers.csv'))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    KBf = 1024.0

    # (a) the partition, and the two identities
    ax = axes[0]
    a = [r for r in all_rows if r['section'] == 'A']
    order = ['input', 'scale', 'weight', 'output']
    caps, words = [], []
    for name in order:
        m = [r for r in a if r['buffer'] == name][0]
        caps.append(fnum(m, 'bytes') / KBf)
        words.append(int(fnum(m, 'width_bits')) // 8)
    bars = ax.bar(range(len(order)), caps, 0.6,
                  color=[AW, ACCENT, AA, DARK])
    for b, c, w in zip(bars, caps, words):
        ax.annotate(f'{c:,.0f} KB\n{w} B/cyc',
                    (b.get_x() + b.get_width() / 2, c),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', fontsize=9)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylabel('capacity (KB)')
    ax.set_ylim(0, max(caps) * 1.28)
    ax.set_title('(a) Four fixed SRAMs, 3.00 MB, no trading')
    ax.grid(axis='y', ls=':', color=GREY, lw=0.6)
    ax.set_axisbelow(True)

    # (b) where the part runs out
    ax = axes[1]
    c = sorted(((int(r['context']), fnum(r, 'tpot_s'), r['decode_overflow'])
                for r in all_rows if r['section'] == 'C'))
    wt_kb = caps[order.index('weight')]
    kv = [ctx * 128 * 4 / 8 / KBf for ctx, _, _ in c]
    bars = ax.bar(range(len(c)), [k / wt_kb for k in kv], 0.6,
                  color=[AA if t[2] else AW for t in c])
    ax.axhline(1.0, ls='--', lw=1.4, color=DARK)
    ax.annotate('weight buffer (2,048 KB)', (-0.4, 1.04), fontsize=9,
                color=DARK)
    for b, t in zip(bars, c):
        if t[2]:
            ax.annotate(t[2].replace(',', ' + '),
                        (b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', fontsize=9, color=AA)
    ax.set_xticks(range(len(c)))
    ax.set_xticklabels([f'{t[0]:,}' for t in c])
    ax.set_xlabel('context')
    ax.set_ylabel('one K cache / weight buffer')
    ax.set_ylim(0, 1.35)
    ax.set_title('(b) 32K is exactly where the part runs out')
    ax.grid(axis='y', ls=':', color=GREY, lw=0.6)
    ax.set_axisbelow(True)

    fig.suptitle("The RTL's four buffers against the model's one pool",
                 fontsize=13, y=1.02)
    save(fig, outdir, 'buffers')


FIGURES = {
    'overlap': fig_overlap,
    'kv_batch': fig_kv_batch,
    'packing': fig_packing,
    'unstructured': fig_unstructured,
    'memory_tech': fig_memory_tech,
    'ports': fig_ports,
    'tiling': fig_tiling,
    'sparsity': fig_sparsity,
    'buffers': fig_buffers,
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outdir', default=_here)
    p.add_argument('--only', choices=sorted(FIGURES), action='append')
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    for name in (args.only or sorted(FIGURES)):
        print(f"{name}:")
        FIGURES[name](args.outdir)


if __name__ == '__main__':
    main()
