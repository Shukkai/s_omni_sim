"""
Plot the channel-pruning breakdown, split by phase as study.md's tables are:
prefill runs LUT_WS and decode runs LUT_OS_V, so the same pruning ratio lands
on different geometry and has to be read against a different denominator.

Rows:  prefill (LUT_WS)  /  decode (LUT_OS_V)
Cols:  (1) the cycle null   (2) the utilization it costs   (3) what it buys

Reads channel_prune_breakdown.json (output of think_run.py).

Usage:
    python think_plot.py
"""

import argparse
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# --- Style, matching the other analysis plots ---
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['DejaVu Serif']
matplotlib.rcParams['axes.labelcolor'] = 'black'
matplotlib.rcParams['xtick.color'] = 'black'
matplotlib.rcParams['ytick.color'] = 'black'
matplotlib.rcParams['text.color'] = 'black'
matplotlib.rcParams['axes.titlesize'] = 11
matplotlib.rcParams['axes.labelsize'] = 10
matplotlib.rcParams['xtick.labelsize'] = 9
matplotlib.rcParams['ytick.labelsize'] = 9
matplotlib.rcParams['legend.fontsize'] = 8

GREEN = '#51827B'
RED = '#E68B88'
GREY = '#B0B0B0'
CTX_COLORS = ['#2F4F4A', GREEN, '#9BBDB6']


def ctx_label(c):
    return f"{c // 1024}K" if c >= 1024 else str(c)


def uniq_by_channel(rows, key):
    """Rows sorted by retained channels, one per distinct value."""
    seen, out = set(), []
    for r in sorted(rows, key=lambda r: r[key]):
        if r[key] not in seen:
            seen.add(r[key])
            out.append(r)
    return out


# ---- Column 1: the cycle null ----------------------------------------------

def panel_cycles(ax, rows, contexts, b0, phase, head_dim, headline):
    """attn_v (pruned as N) against qk (pruned as K), normalized to dense."""
    for i, ctx in enumerate(contexts):
        colour = CTX_COLORS[i % len(CTX_COLORS)]

        av = sorted((r for r in rows if r['batch'] == b0 and r['context'] == ctx
                     and r['mode'] == 'V'), key=lambda r: r['d_ret'])
        if av:
            base = next(r for r in av if r['d_ret'] == head_dim)
            ax.plot([r['d_ret'] for r in av],
                    [r[f'{phase}_attn_v_cycles'] / base[f'{phase}_attn_v_cycles']
                     for r in av],
                    'o-', color=colour, lw=2, ms=5,
                    label=f"attn_v, {ctx_label(ctx)}")

        qk = sorted((r for r in rows if r['batch'] == b0 and r['context'] == ctx
                     and r['mode'] == 'K'), key=lambda r: r['d_ret'])
        if qk:
            base = next(r for r in qk if r['d_ret'] == head_dim)
            ax.plot([r['d_ret'] for r in qk],
                    [r[f'{phase}_qk_cycles'] / base[f'{phase}_qk_cycles']
                     for r in qk],
                    's--', color=colour, lw=1.6, ms=4, alpha=0.85,
                    label=f"qk, {ctx_label(ctx)}")

    ax.axvline(headline, color=GREY, ls=':', lw=1.2)
    ax.text(headline - 2, 0.04, "ThinK $\\lambda$=0.4", color=GREY,
            fontsize=7.5, ha='right', va='bottom')
    ax.set_xlabel(f"retained channels per head (dense = {head_dim})")
    ax.set_ylabel("cycles, normalized to dense")
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(ncol=2, fontsize=7, loc='lower right')


# ---- Column 2: what it costs -----------------------------------------------

def panel_occupancy(ax, rows, ctx0, b0, phase, head_dim):
    """attn_v array occupancy, against this phase's own lane count."""
    uniq = uniq_by_channel([r for r in rows if r['batch'] == b0
                            and r['context'] == ctx0 and 'V' in r['mode']],
                           'd_v_ret')
    if not uniq:
        return
    lanes = uniq[0][f'{phase}_array_lanes']
    d = [r['d_v_ret'] for r in uniq]
    x = np.arange(len(d))
    occ = [r[f'{phase}_attn_v_occupied_frac'] * 100 for r in uniq]
    use = [r[f'{phase}_attn_v_utilization'] * 100 for r in uniq]

    ax.bar(x - 0.19, occ, 0.38, color=GREEN, label='lanes occupied')
    ax.bar(x + 0.19, use, 0.38, color=RED, label='useful MACs / issued')

    top = max(max(occ), max(use))
    for xi, (o, u) in enumerate(zip(occ, use)):
        ax.text(xi, max(o, u) + top * 0.03, f"{u:.2f}%", ha='center',
                fontsize=7, color='black')

    # A 100% line is off-scale in the decode panel by ~25x, which is the point:
    # state the ceiling in text rather than drawing it.
    ax.text(0.5, 0.94, f"full round = {lanes} lanes = 100%",
            transform=ax.transAxes, ha='center', fontsize=8, color=GREY)

    ax.set_xticks(x)
    ax.set_xticklabels(d)
    ax.set_xlabel("retained Value channels per head")
    ax.set_ylabel("% of round doing useful work")
    ax.set_ylim(0, top * 1.35)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='upper right', fontsize=7.5)


# ---- Column 3: what it buys -------------------------------------------------

def panel_prefill_bytes(ax, rows, contexts, b0, head_dim, headline):
    """Prefill KV writeback: the capacity saving, which is all prefill offers."""
    width = 0.36
    x = np.arange(len(contexts))
    dense, pruned = [], []
    for c in contexts:
        d = next((r for r in rows if r['batch'] == b0 and r['context'] == c
                  and r['mode'] == 'KV' and r['d_ret'] == head_dim), None)
        k = next((r for r in rows if r['batch'] == b0 and r['context'] == c
                  and r['mode'] == 'KV' and r['d_ret'] == headline), None)
        dense.append(d['kv_bytes_total'] / 1e9 if d else np.nan)
        pruned.append(k['kv_bytes_total'] / 1e9 if k else np.nan)

    ax.bar(x - width / 2, dense, width, color=GREY, label='dense')
    ax.bar(x + width / 2, pruned, width, color=GREEN,
           label=f'{headline} channels')
    for xi, (dv, pv) in enumerate(zip(dense, pruned)):
        if not np.isnan(dv) and dv > 0:
            ax.text(xi + width / 2, pv, f" -{(1-pv/dv)*100:.0f}%",
                    ha='center', va='bottom', fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels([ctx_label(c) for c in contexts])
    ax.set_xlabel("context")
    ax.set_ylabel("KV cache re-read / decode token (GB)")
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='upper left', fontsize=7.5)


def panel_speedup(ax, fig, rows, contexts, batches, head_dim, headline, mode):
    """Decode roofline speedup over the batch x context grid."""
    idx = {(r['batch'], r['context'], r['mode'], r['d_ret']): r for r in rows}
    grid = np.full((len(batches), len(contexts)), np.nan)
    for i, b in enumerate(batches):
        for j, c in enumerate(contexts):
            d = idx.get((b, c, mode, head_dim))
            k = idx.get((b, c, mode, headline))
            if d and k:
                grid[i, j] = d['decode_eff_time'] / k['decode_eff_time']

    im = ax.imshow(grid, cmap='BuGn', aspect='auto', origin='lower',
                   vmin=1.0, vmax=max(1.02, np.nanmax(grid)))
    ax.set_xticks(range(len(contexts)))
    ax.set_xticklabels([ctx_label(c) for c in contexts])
    ax.set_yticks(range(len(batches)))
    ax.set_yticklabels(batches)
    ax.set_xlabel("context")
    ax.set_ylabel("batch")
    for i in range(len(batches)):
        for j in range(len(contexts)):
            if np.isnan(grid[i, j]):
                continue
            ax.text(j, i, f"{grid[i, j]:.3f}x", ha='center', va='center',
                    fontsize=8.5,
                    color='white' if grid[i, j] > np.nanmean(grid) else 'black')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


# ---- Main -------------------------------------------------------------------

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser()
    p.add_argument('--input',
                   default=os.path.join(here, 'channel_prune_breakdown.json'))
    p.add_argument('--out', default=os.path.join(here, 'channel_prune_breakdown'))
    args = p.parse_args()

    data = json.load(open(args.input))
    cfg = data['config']
    rows = data['rows']
    head_dim = data['head_dim']
    headline = cfg['headline_channels']

    contexts = sorted({r['context'] for r in rows})
    batches = sorted({r['batch'] for r in rows})
    b0, ctx0 = min(batches), max(contexts)
    mode = 'KV' if any(r['mode'] == 'KV' for r in rows) else rows[0]['mode']

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.4))

    for row_i, phase in enumerate(('prefill', 'decode')):
        dataflow = next(r[f'{phase}_dataflow'] for r in rows)
        tag = 'a' if phase == 'prefill' else 'b'

        panel_cycles(axes[row_i][0], rows, contexts, b0, phase,
                     head_dim, headline)
        axes[row_i][0].set_title(
            f"({tag}1) {phase} cycles -- {dataflow}\n"
            + ("both axes flat: head_dim is exactly one tile"
               if phase == 'prefill' else
               "attn_v flat (N); only qk shrinks (K)"))

        panel_occupancy(axes[row_i][1], rows, ctx0, b0, phase, head_dim)
        axes[row_i][1].set_title(
            f"({tag}2) {phase} attn_v occupancy, {ctx_label(ctx0)} ctx")

        if phase == 'prefill':
            panel_prefill_bytes(axes[row_i][2], rows, contexts, b0,
                                head_dim, headline)
            axes[row_i][2].set_title(
                f"({tag}3) KV traffic at {headline} channels\n"
                "the saving is real; the latency saving is not")
        else:
            panel_speedup(axes[row_i][2], fig, rows, contexts, batches,
                          head_dim, headline, mode)
            axes[row_i][2].set_title(
                f"({tag}3) ThinK-{mode} decode speedup, {headline} channels")

    fig.suptitle(
        f"ThinK channel pruning on Omni-LUT -- {cfg['model']}, "
        f"W{cfg['weight_bits']}A{cfg['act_bits']}KV{cfg['kv_bits']}, "
        f"{cfg['array_m']}x{cfg['array_n']} array",
        fontsize=12.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    for ext in ('png', 'pdf'):
        fig.savefig(f"{args.out}.{ext}", dpi=200, bbox_inches='tight')
    print(f"Saved {args.out}.png and {args.out}.pdf")


if __name__ == "__main__":
    main()
