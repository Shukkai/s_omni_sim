"""
Plot system evaluation: throughput & energy breakdown as stacked bars.

Top row:    Throughput (tokens/s) — stacked by time fraction: non-GEMM, AW, AA
Bottom row: Energy (J)           — stacked: non-GEMM, AW, AA

Reads system_eval_results.json (output of run_system_eval.py).
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
with open('system_eval_results.json', 'r') as f:
    data = json.load(f)

# --- 3. Data Organization ---
hw_order = ['FPE', 'Tender', 'FIGLUT', 'Omni4', 'Omni3']
hw_labels = ['FPE', 'Tender-int8', 'FIGLUT', 'Omni-LUT\n-KV4', 'Omni-LUT\n-KV3']

# Input context configurations (fixed output=256)
input_tokens_list = [1024, 4096, 16384]
input_configs = [f'{inp}_512' for inp in input_tokens_list]
input_labels = ['1K', '4K', '16K']

# Color scheme for stacked bar components
component_colors = {
    'non_gemm': '#B0B0B0',  # Grey   - non-GEMM (bottom)
    # 'aw':       '#51827B',  # Green  - AW-GEMM  (middle)
    'aw':       "#747AA2",  # Green  - AW-GEMM  (middle)
    'aa':       "#D99997",  # Red    - AA-GEMM  (top)
}

row_labels = ['(a) Norm. Latency', '(b) Norm. Energy']

# --- 4. Create Figure: 2 rows x 3 columns ---
fig, axes = plt.subplots(2, 3, figsize=(11, 5), sharey='row')

n_hw = len(hw_order)

# --- 5. First pass: compute per-row max for normalized y-axis scaling ---
row_maxes = [0.0, 0.0]
for col_idx, input_config in enumerate(input_configs):
    # Row 0: latency — normalize to FPE total time
    time_vals = []
    for hw in hw_order:
        if hw in data and input_config in data[hw]:
            d = data[hw][input_config]
            time_vals.append(d['aa_time'] + d['aw_time'] + d['ng_time'])
        else:
            time_vals.append(0)
    fpe_time = time_vals[0]
    if fpe_time > 0:
        row_maxes[0] = max(row_maxes[0], max(v / fpe_time for v in time_vals))

    # Row 1: energy — normalize to FPE
    e_vals = []
    for hw in hw_order:
        if hw in data and input_config in data[hw]:
            d = data[hw][input_config]
            e_vals.append(d['aa_energy'] + d['aw_energy'] + d['ng_energy'])
        else:
            e_vals.append(0)
    fpe_e = e_vals[0]
    if fpe_e > 0:
        row_maxes[1] = max(row_maxes[1], max(v / fpe_e for v in e_vals))

y_limits = [mx * 1.35 for mx in row_maxes]

# --- 6. Plot stacked bars ---
for col_idx, input_config in enumerate(input_configs):

    # ---- Gather data for all hardware ----
    throughput_vals = []
    aa_time_vals, aw_time_vals, ng_time_vals, total_time_vals = [], [], [], []
    aa_energy_vals, aw_energy_vals, ng_energy_vals, total_energy_vals = [], [], [], []

    for hw in hw_order:
        if hw in data and input_config in data[hw]:
            d = data[hw][input_config]
            throughput_vals.append(d['throughput'])
            aa_time_vals.append(d['aa_time'])
            aw_time_vals.append(d['aw_time'])
            ng_time_vals.append(d['ng_time'])
            total_time_vals.append(d['total_time'])
            aa_energy_vals.append(d['aa_energy'])
            aw_energy_vals.append(d['aw_energy'])
            ng_energy_vals.append(d['ng_energy'])
            total_energy_vals.append(d['aa_energy'] + d['aw_energy'] + d['ng_energy'])
        else:
            throughput_vals.append(0)
            aa_time_vals.append(0); aw_time_vals.append(0)
            ng_time_vals.append(0); total_time_vals.append(0)
            aa_energy_vals.append(0); aw_energy_vals.append(0)
            ng_energy_vals.append(0); total_energy_vals.append(0)

    x_pos = np.arange(n_hw)
    bar_width = 0.6

    # ================================================================
    # Row 0: Latency — normalized to FPE total time, stacked AA/AW/non-GEMM
    # ================================================================
    ax = axes[0, col_idx]

    # Normalize time components to FPE total time
    fpe_time = total_time_vals[0]
    if fpe_time > 0:
        ng_t_norm = [v / fpe_time for v in ng_time_vals]
        aw_t_norm = [v / fpe_time for v in aw_time_vals]
        aa_t_norm = [v / fpe_time for v in aa_time_vals]
        total_t_norm = [v / fpe_time for v in total_time_vals]
    else:
        ng_t_norm = ng_time_vals[:]
        aw_t_norm = aw_time_vals[:]
        aa_t_norm = aa_time_vals[:]
        total_t_norm = total_time_vals[:]

    ax.bar(x_pos, ng_t_norm, bar_width,
           color=component_colors['non_gemm'], edgecolor='black', linewidth=0.8)
    ax.bar(x_pos, aw_t_norm, bar_width, bottom=ng_t_norm,
           color=component_colors['aw'], edgecolor='black', linewidth=0.8)
    ax.bar(x_pos, aa_t_norm, bar_width,
           bottom=np.array(ng_t_norm) + np.array(aw_t_norm),
           color=component_colors['aa'], edgecolor='black', linewidth=0.8)

    # Labels: normalized number; FPE also gets absolute anchor (seconds)
    label_offset = y_limits[0] * 0.02
    for i in range(n_hw):
        if i == 0:  # FPE: show absolute latency anchor in seconds
            abs_str = f'({total_time_vals[i]:.0f} s)'
            ax.text(x_pos[i], total_t_norm[i] + label_offset,
                    f'{abs_str}\n{total_t_norm[i]:.2f}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        else:
            ax.text(x_pos[i], total_t_norm[i] + label_offset,
                    f'{total_t_norm[i]:.2f}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    # ================================================================
    # Row 1: Energy — normalized bars, absolute labels (J)
    # ================================================================
    ax2 = axes[1, col_idx]

    # Normalize energy to FPE
    fpe_energy = total_energy_vals[0]
    if fpe_energy > 0:
        ng_e_norm = [v / fpe_energy for v in ng_energy_vals]
        aw_e_norm = [v / fpe_energy for v in aw_energy_vals]
        aa_e_norm = [v / fpe_energy for v in aa_energy_vals]
        total_e_norm = [v / fpe_energy for v in total_energy_vals]
    else:
        ng_e_norm = ng_energy_vals[:]
        aw_e_norm = aw_energy_vals[:]
        aa_e_norm = aa_energy_vals[:]
        total_e_norm = total_energy_vals[:]

    ax2.bar(x_pos, ng_e_norm, bar_width,
            color=component_colors['non_gemm'], edgecolor='black', linewidth=0.8)
    ax2.bar(x_pos, aw_e_norm, bar_width, bottom=ng_e_norm,
            color=component_colors['aw'], edgecolor='black', linewidth=0.8)
    ax2.bar(x_pos, aa_e_norm, bar_width,
            bottom=np.array(ng_e_norm) + np.array(aw_e_norm),
            color=component_colors['aa'], edgecolor='black', linewidth=0.8)

    # Labels: normalized number; FPE also gets absolute anchor
    label_offset2 = y_limits[1] * 0.02
    for i in range(n_hw):
        if i == 0:  # FPE: show absolute anchor
            abs_str = f'({total_energy_vals[i]:.0f} J)'
            ax2.text(x_pos[i], total_e_norm[i] + label_offset2,
                     f'{abs_str}\n{total_e_norm[i]:.2f}',
                     ha='center', va='bottom', fontsize=12, fontweight='bold')
        else:
            ax2.text(x_pos[i], total_e_norm[i] + label_offset2,
                     f'{total_e_norm[i]:.2f}',
                     ha='center', va='bottom', fontsize=12, fontweight='bold')

    # ---- Common formatting for both rows ----
    for row_idx, cur_ax in enumerate([ax, ax2]):
        cur_ax.set_xticks(x_pos)
        if row_idx == 1:  # Bottom row: show hardware labels
            cur_ax.set_xticklabels(hw_labels, rotation=45, ha='right', fontsize=9)
        else:
            cur_ax.set_xticklabels([])

        cur_ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
        cur_ax.set_axisbelow(True)
        sns.despine(ax=cur_ax)
        cur_ax.set_ylim(0, y_limits[row_idx])
        cur_ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

    # Title: input token count on top row only
    ax.set_title(f'Input tokens = {input_labels[col_idx]}', fontweight='bold',
                 fontsize=13, loc='center')

# --- 7. Add per-row y-axis labels ---
axes[0, 0].set_ylabel(row_labels[0], fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel(row_labels[1], fontsize=12, fontweight='bold')

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

plt.savefig("system_eval.pdf", bbox_inches='tight', dpi=300)
print("Figure saved as system_eval.pdf")
