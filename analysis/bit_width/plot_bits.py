"""
Pareto scatter plot: Normalized TOPS/W  vs  Perplexity (PPL).

X-axis: TOPS/W ∝ 1 / total_energy, normalized to the worst (lowest TOPS/W).
Y-axis: PPL — fill in manually below.

Reads bit_width_results.json produced by run_bits.py.
"""

import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from brokenaxes import brokenaxes

# --- Style (consistent with project) ---
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['DejaVu Serif']
matplotlib.rcParams['axes.labelcolor'] = 'black'
matplotlib.rcParams['xtick.color'] = 'black'
matplotlib.rcParams['ytick.color'] = 'black'
matplotlib.rcParams['text.color'] = 'black'
matplotlib.rcParams['axes.titlesize'] = 14
matplotlib.rcParams['axes.labelsize'] = 13
matplotlib.rcParams['xtick.labelsize'] = 11
matplotlib.rcParams['ytick.labelsize'] = 11
matplotlib.rcParams['legend.fontsize'] = 11

# ============================================================================
# PPL values — FILL THESE IN MANUALLY
# ============================================================================
PPL = {
    'Tender-W4': 123.10,     # TODO: fill in PPL
    'Tender-W8': 5.76,     # TODO: fill in PPL
    'Omni-KV2':  9.96082,     # TODO: fill in PPL
    'Omni-KV3':  5.82766,     # TODO: fill in PPL
    'Omni-KV4':  5.63,     # TODO: fill in PPL
}

# ============================================================================
# Visual config per point
# ============================================================================
# marker, color, display label
STYLE = {
    'Tender-W4': ('s', '#51827B', 'Tender-int4'),
    'Tender-W8': ('s', '#3A5E58', 'Tender-int8'),
    'Omni-KV2':  ('o', '#F4A582', 'Omni-LUT-KV2'),
    'Omni-KV3':  ('o', '#E68B88', 'Omni-LUT-KV3'),
    'Omni-KV4':  ('o', '#D97573', 'Omni-LUT-KV4'),
}

# ============================================================================
# Load data & compute normalised TOPS/W
# ============================================================================

with open('bit_width_results.json', 'r') as f:
    data = json.load(f)

# TOPS/W ∝ 1 / energy (higher is better)
raw_tops_w = {k: 1.0 / v['total_energy'] for k, v in data.items()}

# Normalize so the minimum TOPS/W maps to 1.0
min_tops_w = min(raw_tops_w.values())
norm_tops_w = {k: v / min_tops_w for k, v in raw_tops_w.items()}

# ============================================================================
# Plot
# ============================================================================

Y_MAX = 10.5   # Upper y region top
Y_BREAK_LO = 6.25  # Break starts
Y_BREAK_HI = 9.5  # Break ends
Y_MIN = 5       # Lower y region bottom
FP16_PPL = 5.47  # FP16 baseline perplexity

fig = plt.figure(figsize=(5, 3.0))
bax = brokenaxes(ylims=((Y_MIN, Y_BREAK_LO), (Y_BREAK_HI, Y_MAX)),
                 hspace=0.08, fig=fig, d=0.012)

# Per-point annotation offsets and alignment
ANNOT_CFG = {
    'Tender-W8': {'xytext': (10, 0),    'ha': 'left',  'va': 'center'},
    'Omni-KV2':  {'xytext': (-10, 0),   'ha': 'right', 'va': 'center'},   # direct left
    'Omni-KV3':  {'xytext': (10, 0),    'ha': 'left',  'va': 'center'},   # direct right
    'Omni-KV4':  {'xytext': (10, 0),    'ha': 'left',  'va': 'center'},   # direct right
}

# Plot all in-range points via bax
for name in data:
    marker, color, label = STYLE[name]
    x = norm_tops_w[name]
    y = PPL[name]

    if name == 'Tender-W4':
        continue  # handle separately below
    cfg = ANNOT_CFG[name]
    bax.scatter([x], [y], marker=marker, color=color, s=220,
                edgecolors='black', linewidths=1.0, zorder=5)
    bax.annotate(label, (x, y), textcoords='offset points',
                 xytext=cfg['xytext'], fontsize=12, color=color,
                 fontweight='bold', ha=cfg['ha'], va=cfg['va'])

# --- Tender-W4: off-chart, marker at top of upper axis with arrow pointing up ---
tw4_marker, tw4_color, tw4_label = STYLE['Tender-W4']
tw4_x = norm_tops_w['Tender-W4']
tw4_ppl = PPL['Tender-W4']
# Place marker slightly below top so it's not clipped
marker_y = Y_MAX - 0.1
bax.scatter([tw4_x], [marker_y], marker=tw4_marker, color=tw4_color, s=220,
            edgecolors='black', linewidths=1.0, zorder=5)
# Arrow from marker upward, extending beyond the axis
top_ax = bax.axs[0]
top_ax.set_clip_on(False)
arrow_tip_y = Y_MAX + 0.3  # above the axis boundary
top_ax.annotate(
    '', xy=(tw4_x, arrow_tip_y), xytext=(tw4_x, marker_y),
    arrowprops=dict(arrowstyle='->', color=tw4_color, lw=2.5, clip_on=False),
    annotation_clip=False, zorder=5,
)
# Label to the left of the marker
top_ax.annotate(
    f'{tw4_label}\n(PPL={tw4_ppl:.1f})', xy=(tw4_x, marker_y),
    textcoords='offset points', xytext=(-10, 0),
    fontsize=12, color=tw4_color, fontweight='bold',
    ha='right', va='center', annotation_clip=False,
)

# --- Connect Omni frontier: KV4 → KV3 → KV2 ---
# omni_frontier = ['Omni-KV4', 'Omni-KV3', 'Omni-KV2']
# frontier_x = [norm_tops_w[k] for k in omni_frontier]
# frontier_y = [PPL[k] for k in omni_frontier]
# bax.plot(frontier_x, frontier_y, '--', color='gray', linewidth=1.2, alpha=0.6, zorder=2)

# --- FP16 baseline ---
bax.axhline(FP16_PPL, color='red', linestyle=':', linewidth=1.5, zorder=3)
# Label below the baseline line (bottom axis only)
bax.axs[-1].text(bax.axs[-1].get_xlim()[1], FP16_PPL - 0.06,
                 f'FP16 baseline (PPL={FP16_PPL})', color='red', fontsize=10,
                 fontweight='bold', ha='right', va='top')

# --- Formatting ---
bax.set_xlabel('Normalized TOPS/W', fontweight='bold',
               fontsize=11, labelpad=20)
bax.set_ylabel('Perplexity (PPL) ↓', fontweight='bold', fontsize=11, labelpad=30)

for ax in bax.axs:
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

plt.savefig('pareto_bits.pdf', bbox_inches='tight', dpi=300)
print('Saved pareto_bits.pdf')
