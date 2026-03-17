import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib
import json

# --- 1. Style Setup (matching previous plots) ---
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
matplotlib.rcParams['legend.fontsize'] = 10

# --- 2. Load Data ---
with open('throughput_results.json', 'r') as f:
    data = json.load(f)

# --- 3. Data Organization ---
# Hardware order: FPE, Tender, FIGLUT, Omni4, Omni3
hw_order = ['FPE', 'Tender', 'FIGLUT', 'Omni4', 'Omni3']
hw_labels = ['FPE', 'Tender-int8', 'FIGLUT', 'Omni-LUT-KV4', 'Omni-LUT-KV3']

# Input context configurations (fixed output=256)
input_configs = ['2048_256', '8192_256', '32768_256']
input_labels = ['2K', '8K', '32K']

# Color scheme for hardware (matching other plots)
hw_colors = ['#B0B0B0',   # FPE - Gray
             '#51827B',   # Tender - Teal
             '#7C5151',   # FIGLUT - Brown
             '#E68B88',   # Omni4 - Light red-pink
             '#D97573']   # Omni3 - Darker red-pink

# Two rows: row 0 = TTFT, row 1 = TPOT
metrics = ['ttft', 'tpot']
row_labels = ['(a) Normalized TTFT', '(b) Normalized TPOT']

# --- 4. Create Figure: 2 rows x 3 columns ---
fig, axes = plt.subplots(2, 3, figsize=(11, 5), sharey='row')

n_hw = len(hw_order)

# --- 5. First pass: compute per-row max for y-axis scaling ---
row_maxes = [0.0, 0.0]
for row_idx, metric in enumerate(metrics):
    for input_config in input_configs:
        vals = []
        for hw in hw_order:
            if hw in data and input_config in data[hw]:
                vals.append(data[hw][input_config][metric] * 1e3)
            else:
                vals.append(0)
        fpe_val = vals[0]
        if fpe_val > 0:
            normed = [v / fpe_val for v in vals]
        else:
            normed = vals
        row_maxes[row_idx] = max(row_maxes[row_idx], max(normed))

y_limits = [mx * 1.15 for mx in row_maxes]

# --- 6. Plot bars ---
for row_idx, metric in enumerate(metrics):
    for col_idx, input_config in enumerate(input_configs):
        ax = axes[row_idx, col_idx]

        # Extract metric values in ms
        values_ms = []
        for hw in hw_order:
            if hw in data and input_config in data[hw]:
                values_ms.append(data[hw][input_config][metric] * 1e3)
            else:
                values_ms.append(0)

        # Normalize to FPE (first hardware)
        fpe_val = values_ms[0]
        if fpe_val > 0:
            normalized = [v / fpe_val for v in values_ms]
        else:
            normalized = values_ms

        # Create bar chart
        x_pos = np.arange(n_hw) * 0.8
        bar_width = 0.5

        bars = ax.bar(x_pos, normalized, bar_width,
                      color=hw_colors, edgecolor='black', linewidth=0.8)

        # Add value labels on top of bars
        label_offset = y_limits[row_idx] * 0.02
        for i, val in enumerate(normalized):
            if val > 0:
                ax.text(x_pos[i], val + label_offset, f'{val:.2f}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Formatting
        ax.set_xticks(x_pos)
        ax.set_xticklabels([])

        # Title: input token count on top row only
        if row_idx == 0:
            ax.set_title(f'Input tokens = {input_labels[col_idx]}', fontweight='bold',
                         fontsize=13, loc='center')

        # X-axis label on bottom row only
        # if row_idx == 1:
        #     ax.set_xlabel(f'Input = {input_labels[col_idx]}', fontweight='bold', fontsize=11)

        # Grid
        ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
        ax.set_axisbelow(True)

        # Remove top and right spines
        sns.despine(ax=ax)

        # Set y-axis limits per row
        ax.set_ylim(0, y_limits[row_idx])

# --- 7. Add per-row y-axis labels ---
for row_idx in range(2):
    axes[row_idx, 0].set_ylabel(row_labels[row_idx], fontsize=12, fontweight='bold')

# --- 8. Add Legend ---
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=hw_colors[i], edgecolor='black',
                         linewidth=0.8, label=hw_labels[i])
                   for i in range(n_hw)]

fig.legend(handles=legend_elements, loc='lower center',
           bbox_to_anchor=(0.5, 0), ncol=5, frameon=True,
           fontsize=11, edgecolor='black')

# --- 9. Final Layout and Save ---
plt.subplots_adjust(hspace=0.15, wspace=0.0, bottom=0.10, left=0.10)

# Save as PDF for publication quality
plt.savefig("throughput_analysis.pdf", bbox_inches='tight', dpi=300)
print("Figure saved as throughput_analysis.pdf")
