import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib
import json
from matplotlib.patches import Patch

# --- 1. Style Setup ---
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['DejaVu Serif']
matplotlib.rcParams['axes.labelcolor'] = 'black'
matplotlib.rcParams['xtick.color'] = 'black'
matplotlib.rcParams['ytick.color'] = 'black'
matplotlib.rcParams['text.color'] = 'black'
matplotlib.rcParams['axes.titlesize'] = 12
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['legend.fontsize'] = 14

# --- 2. Area Constants ---
FPE_AREA = 640273.2405
TENDER_AREA = 201627.5926
OMNI_AREA = 517703.42 + 9900.516996 + 4102.741429
FIGLUT_AREA = 373146.7753 + 18021.11557 + FPE_AREA  # 32*4

hw_area = {
    'FPE': FPE_AREA,
    'Tender': TENDER_AREA,
    'FIGLUT': FIGLUT_AREA,
    'Omni4': OMNI_AREA,
    'Omni3': OMNI_AREA,
}

# --- 3. Load Data ---
with open('hw_tops-mm2_roofline.json', 'r') as f:
    data = json.load(f)

# --- 4. Configuration ---
input_config = '4096_512'

models_top = ['OPT-1.3B', 'OPT-6.7B', 'OPT-30B']               # Row 0
models_bot = ['LLaMA-3-8B', 'Mixtral-8x7B', 'Qwen3-30B-A3B']    # Row 1

hw_order = ['FPE', 'Tender', 'FIGLUT', 'Omni4', 'Omni3']
hw_labels = ['FPE', 'Tender-int8', 'FIGLUT', 'Omni-LUT-KV4', 'Omni-LUT-KV3']

hw_colors = ['#B0B0B0',   # FPE - Gray
             '#51827B',   # Tender - Teal
             '#7C5151',   # FIGLUT - Brown
             '#E68B88',   # Omni4 - Light red-pink
             '#D97573']   # Omni3 - Darker red-pink

n_hw = len(hw_order)

# --- 5. Create Figure (3 cols × 2 rows) ---
fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharey='row')

all_rows = [models_top, models_bot]

# First pass: compute per-row max for y-axis
row_maxes = [0.0, 0.0]
for row_idx, models in enumerate(all_rows):
    for model in models:
        tops_mm_values = []
        for hw in hw_order:
            if hw in data[model] and input_config in data[model][hw]:
                total_cycles = data[model][hw][input_config]['total_cycles']
                area = hw_area[hw]
                tops_mm = 1.0 / (total_cycles * area) if total_cycles > 0 and area > 0 else 0
                tops_mm_values.append(tops_mm)
            else:
                tops_mm_values.append(0)
        fpe_val = tops_mm_values[0]
        if fpe_val > 0:
            normed = [v / fpe_val for v in tops_mm_values]
        else:
            normed = tops_mm_values
        row_maxes[row_idx] = max(row_maxes[row_idx], max(normed))

y_limits = [mx + 0.5 for mx in row_maxes]

# --- 6. Plot ---
for row_idx, models in enumerate(all_rows):
    for col_idx, model in enumerate(models):
        ax = axes[row_idx, col_idx]

        tops_mm_values = []
        for hw in hw_order:
            if hw in data[model] and input_config in data[model][hw]:
                total_cycles = data[model][hw][input_config]['total_cycles']
                area = hw_area[hw]
                tops_mm = 1.0 / (total_cycles * area) if total_cycles > 0 and area > 0 else 0
                tops_mm_values.append(tops_mm)
            else:
                tops_mm_values.append(0)

        # Normalize to FPE
        fpe_val = tops_mm_values[0]
        if fpe_val > 0:
            normed = [v / fpe_val for v in tops_mm_values]
        else:
            normed = tops_mm_values

        x_pos = np.arange(n_hw) * 0.8
        bar_width = 0.5

        bars = ax.bar(x_pos, normed, bar_width,
                      color=hw_colors, edgecolor='black', linewidth=0.8)

        # Value labels
        label_offset = y_limits[row_idx] * 0.02
        for i, val in enumerate(normed):
            if val > 0:
                ax.text(x_pos[i], val + label_offset, f'{val:.2f}',
                        ha='center', va='bottom', fontsize=13, fontweight='bold')

        ax.set_xticks(x_pos)
        ax.set_xticklabels([])

        # Model name as xlabel
        ax.set_xlabel(model, fontweight='bold', fontsize=12)

        # Grid
        ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        sns.despine(ax=ax)

        ax.set_ylim(0, y_limits[row_idx])

# --- 7. Shared y-axis label ---
fig.text(0.04, 0.5, 'Normalized TOPS/mm$^2$', va='center', rotation='vertical',
         fontsize=16, fontweight='bold')

# --- 8. Row captions ---
fig.text(0.50, 0.01, '(b) GQA / MoE Models', ha='center', va='top',
         fontweight='bold', fontsize=16)
fig.text(0.50, 0.52, '(a) OPT Models', ha='center', va='top',
         fontweight='bold', fontsize=16)

# --- 9. Legend ---
legend_elements = [Patch(facecolor=hw_colors[i], edgecolor='black',
                         linewidth=0.8, label=hw_labels[i])
                   for i in range(n_hw)]

fig.legend(handles=legend_elements, loc='lower center',
           bbox_to_anchor=(0.50, -0.10), ncol=5, frameon=True,
           fontsize=13, edgecolor='black')

# --- 10. Save ---
plt.subplots_adjust(hspace=0.45, wspace=0.0, bottom=0.12, left=0.08)

plt.savefig("tops_mm2_fixed.pdf", bbox_inches='tight', dpi=300)
print("Figure saved as tops_mm2_fixed.pdf")
