"""
Plot TTFT/TPOT latency breakdown as stacked bars: non-GEMM, AW-GEMM, AA-GEMM.

Reads throughput_results.json (output of run_throughput.py).
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib
import json

# --- 1. Style Setup ---
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['DejaVu Serif']
matplotlib.rcParams['axes.labelcolor'] = 'black'
matplotlib.rcParams['xtick.color'] = 'black'
matplotlib.rcParams['ytick.color'] = 'black'
matplotlib.rcParams['text.color'] = 'black'
matplotlib.rcParams['axes.titlesize'] = 12
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10
matplotlib.rcParams['legend.fontsize'] = 12

# --- 2. Load Data ---
with open('throughput_results.json', 'r') as f:
    data = json.load(f)

# --- 3. Data Organization ---
hw_order = ['FPE', 'Tender', 'FIGLUT', 'Omni4', 'Omni3']
hw_labels = ['FPE', 'Tender-int8', 'FIGLUT', 'Omni-LUT\n-KV4', 'Omni-LUT\n-KV3']

# Input context configurations (fixed output=256)
input_configs = ['2048_256', '8192_256', '32768_256']
input_labels = ['2K', '8K', '32K']

# Color scheme for latency components (stacked bars)
component_colors = {
    'non_gemm': '#B0B0B0',  # Grey   - non-GEMM (bottom)
    'aw':       '#51827B',  # Green  - AW-GEMM  (middle)
    'aa':       '#E68B88',  # Red    - AA-GEMM  (top)
}

# Two rows: row 0 = TTFT, row 1 = TPOT
metrics_prefixes = ['ttft', 'tpot']
row_labels = ['(a) Normalized TTFT', '(b) Normalized TPOT']

# --- 4. Create Figure: 2 rows x 3 columns ---
fig, axes = plt.subplots(2, 3, figsize=(11, 5), sharey='row')

n_hw = len(hw_order)

# --- 5. First pass: compute per-row max for y-axis scaling ---
row_maxes = [0.0, 0.0]
for row_idx, prefix in enumerate(metrics_prefixes):
    for input_config in input_configs:
        totals = []
        for hw in hw_order:
            if hw in data and input_config in data[hw]:
                d = data[hw][input_config]
                totals.append(d[f'{prefix}_aa'] + d[f'{prefix}_aw'] + d[f'{prefix}_non_gemm'])
            else:
                totals.append(0)
        fpe_total = totals[0]
        if fpe_total > 0:
            normed = [v / fpe_total for v in totals]
        else:
            normed = totals
        row_maxes[row_idx] = max(row_maxes[row_idx], max(normed))

y_limits = [mx * 1.25 for mx in row_maxes]

# --- 6. Plot stacked bars ---
for row_idx, prefix in enumerate(metrics_prefixes):
    for col_idx, input_config in enumerate(input_configs):
        ax = axes[row_idx, col_idx]

        # Extract component latencies (seconds)
        aa_vals = []
        aw_vals = []
        ng_vals = []
        total_vals = []

        for hw in hw_order:
            if hw in data and input_config in data[hw]:
                d = data[hw][input_config]
                aa_vals.append(d[f'{prefix}_aa'])
                aw_vals.append(d[f'{prefix}_aw'])
                ng_vals.append(d[f'{prefix}_non_gemm'])
                total_vals.append(d[f'{prefix}_aa'] + d[f'{prefix}_aw'] + d[f'{prefix}_non_gemm'])
            else:
                aa_vals.append(0)
                aw_vals.append(0)
                ng_vals.append(0)
                total_vals.append(0)

        # Normalize to FPE total
        fpe_total = total_vals[0]
        if fpe_total > 0:
            aa_norm = [v / fpe_total for v in aa_vals]
            aw_norm = [v / fpe_total for v in aw_vals]
            ng_norm = [v / fpe_total for v in ng_vals]
            total_norm = [v / fpe_total for v in total_vals]
        else:
            aa_norm = aa_vals
            aw_norm = aw_vals
            ng_norm = ng_vals
            total_norm = total_vals

        # Create stacked bar chart
        x_pos = np.arange(n_hw)
        bar_width = 0.6

        # Stack: non-GEMM (bottom), AW (middle), AA (top)
        p1 = ax.bar(x_pos, ng_norm, bar_width,
                     label='Non-GEMM', color=component_colors['non_gemm'],
                     edgecolor='black', linewidth=0.8)
        p2 = ax.bar(x_pos, aw_norm, bar_width,
                     bottom=ng_norm,
                     label='AW-GEMM', color=component_colors['aw'],
                     edgecolor='black', linewidth=0.8)
        p3 = ax.bar(x_pos, aa_norm, bar_width,
                     bottom=np.array(ng_norm) + np.array(aw_norm),
                     label='AA-GEMM', color=component_colors['aa'],
                     edgecolor='black', linewidth=0.8)

        # Add value labels on top of bars
        label_offset = y_limits[row_idx] * 0.02
        for i, total in enumerate(total_norm):
            if i == 0:  # FPE: show absolute value too
                if prefix == 'ttft':
                    abs_str = f'({fpe_total:.2f} s)'
                else:
                    abs_str = f'({fpe_total*1e3:.0f} ms)'
                ax.text(x_pos[i], total + label_offset, f'{abs_str}\n{total:.2f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
            else:
                ax.text(x_pos[i], total + label_offset, f'{total:.2f}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Formatting
        ax.set_xticks(x_pos)
        if row_idx == 1:  # Bottom row: show hardware labels
            ax.set_xticklabels(hw_labels, rotation=45, ha='right', fontsize=9)
        else:
            ax.set_xticklabels([])

        # Title: input token count on top row only
        if row_idx == 0:
            ax.set_title(f'Input tokens = {input_labels[col_idx]}', fontweight='bold',
                         fontsize=13, loc='center')

        # Grid
        ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
        ax.set_axisbelow(True)

        # Remove top and right spines
        sns.despine(ax=ax)

        # Set y-axis limits per row
        ax.set_ylim(0, y_limits[row_idx])

        # Y-axis: 2 decimal places
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

# --- 7. Add per-row y-axis labels ---
for row_idx in range(2):
    axes[row_idx, 0].set_ylabel(row_labels[row_idx], fontsize=12, fontweight='bold')

# --- 8. Add Legend ---
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=component_colors['non_gemm'], edgecolor='black', linewidth=0.8, label='Non-GEMM (VPU)'),
    Patch(facecolor=component_colors['aw'], edgecolor='black', linewidth=0.8, label='AW-GEMM'),
    Patch(facecolor=component_colors['aa'], edgecolor='black', linewidth=0.8, label='AA-GEMM'),
]

fig.legend(handles=legend_elements, loc='lower center',
           bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=True,
           fontsize=12, edgecolor='black')

# --- 9. Final Layout and Save ---
plt.subplots_adjust(hspace=0.15, wspace=0.0, bottom=0.12, left=0.10)

plt.savefig("latency_breakdown.pdf", bbox_inches='tight', dpi=300)
print("Figure saved as latency_breakdown.pdf")
