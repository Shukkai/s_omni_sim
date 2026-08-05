"""
Plot the compaction breakdown: payback, regime map, and the fixed-overhead knee.

Reads compact_breakdown.json (output of run_compact_breakdown.py).

Usage:
    python plot_compact_breakdown.py
"""

import argparse
import json

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
matplotlib.rcParams['axes.titlesize'] = 12
matplotlib.rcParams['axes.labelsize'] = 11
matplotlib.rcParams['xtick.labelsize'] = 9
matplotlib.rcParams['ytick.labelsize'] = 9
matplotlib.rcParams['legend.fontsize'] = 9

GREEN = '#51827B'
RED = '#E68B88'
GREY = '#B0B0B0'
CTX_COLORS = ['#2F4F4A', GREEN, '#9BBDB6']


def ctx_label(c):
    return f"{c // 1024}K" if c >= 1024 else str(c)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='compact_breakdown.json')
    p.add_argument('--out', default='compact_breakdown')
    args = p.parse_args()

    data = json.load(open(args.input))
    cfg = data['config']
    rows = data['rows']
    headline = cfg['headline_budget']

    contexts = sorted({r['context'] for r in rows})
    batches = sorted({r['batch'] for r in rows})
    idx = {(r['batch'], r['context'], r['budget_frac']): r for r in rows}

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    # ---- (a) Regime map: speedup heatmap over batch x context ----
    ax = axes[0]
    grid = np.zeros((len(batches), len(contexts)))
    for i, b in enumerate(batches):
        for j, c in enumerate(contexts):
            d, k = idx.get((b, c, 1.0)), idx.get((b, c, headline))
            grid[i, j] = (d['decode_eff_time'] / k['decode_eff_time']
                          if d and k else np.nan)

    im = ax.imshow(grid, cmap='BuGn', aspect='auto', origin='lower',
                   vmin=1.0, vmax=max(2.0, np.nanmax(grid)))
    ax.set_xticks(range(len(contexts)))
    ax.set_xticklabels([ctx_label(c) for c in contexts])
    ax.set_yticks(range(len(batches)))
    ax.set_yticklabels(batches)
    ax.set_xlabel('Context length')
    ax.set_ylabel('Batch size')
    ax.set_title(f'(a) Ceiling speedup @ {headline*100:.0f}% budget')
    for i in range(len(batches)):
        for j in range(len(contexts)):
            v = grid[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}x", ha='center', va='center', fontsize=10,
                        color='white' if v > 0.6 * np.nanmax(grid) else 'black')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='speedup')

    # ---- (b) The knee: fixed per-round overhead vs budget ----
    ax = axes[1]
    b0 = min(batches)
    for n, c in enumerate(contexts):
        sel = sorted((r for r in rows if r['context'] == c and r['batch'] == b0),
                     key=lambda r: r['budget_frac'])
        if not sel:
            continue
        ax.plot([r['budget_tokens'] for r in sel],
                [r['attn_fixed_share'] * 100 for r in sel],
                marker='o', ms=4, lw=1.8,
                color=CTX_COLORS[n % len(CTX_COLORS)],
                label=f"{ctx_label(c)} context")
    ax.set_xscale('log')
    ax.set_xlabel('Retained KV entries')
    ax.set_ylabel('Fixed overhead (% of attention cycles)')
    ax.set_title('(b) Fixed-overhead knee')
    ax.grid(True, linestyle=':', linewidth=0.6, alpha=0.6)
    ax.axvspan(100, 300, color=RED, alpha=0.18, lw=0)
    ax.text(170, ax.get_ylim()[1] * 0.93, 'budgets KV-compression\npapers headline',
            ha='center', va='top', fontsize=8, color='#8a3b38')
    ax.legend(frameon=False)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    # ---- (c) Compaction payback ----
    ax = axes[2]
    fracs = np.array(sorted({r['budget_frac'] for r in rows if r['budget_frac'] < 1.0}))
    payback = (1 + fracs) / (1 - fracs)
    ax.plot(fracs * 100, payback, marker='o', ms=4, lw=1.8, color=GREEN,
            label='compact after prefill')
    ax.plot(fracs * 100, np.zeros_like(fracs), marker='s', ms=4, lw=1.8,
            color=GREY, label='fused into prefill writeback')
    ax.set_xscale('log')
    ax.set_xlabel('KV budget (% of context)')
    ax.set_ylabel('Payback (decode steps)')
    ax.set_title('(c) Cost of compaction')
    ax.set_ylim(-0.4, max(4.0, payback.max() * 1.15))
    ax.grid(True, linestyle=':', linewidth=0.6, alpha=0.6)
    ax.legend(frameon=False, loc='upper left')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    fig.suptitle(
        f"{cfg['model']}  |  AW={cfg['aw_mode']} AA={cfg['aa_mode']}  "
        f"{cfg['array_m']}x{cfg['array_n']}  "
        f"W{cfg['weight_bits']}A{cfg['act_bits']}KV{cfg['kv_bits']}",
        fontsize=11, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    for ext in ('pdf', 'png'):
        fig.savefig(f"{args.out}.{ext}", dpi=300, bbox_inches='tight')
    print(f"Saved {args.out}.pdf and {args.out}.png")


if __name__ == "__main__":
    main()
