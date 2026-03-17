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
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['legend.fontsize'] = 14

FPE_AREA = 640273.2405
TENDER_AREA = 201627.5926
OMNI_AREA = 517703.42 + 9900.516996 + 4102.741429
# FIGLUT_AREA = 118555.518 * 4 + FPE_AREA #16*2
FIGLUT_AREA = 373146.7753 + 18021.11557 + FPE_AREA #32*4

# --- 2. Load Data ---
with open('hw_tops-mm2_roofline.json', 'r') as f:
    data = json.load(f)

# --- 3. Data Organization ---
# Models split into two panels
models_a = ['OPT-1.3B', 'OPT-6.7B', 'OPT-30B']          # Panel (a)
models_b = ['LLaMA-3-8B', 'Mixtral-8x7B', 'Qwen3-30B-A3B']  # Panel (b)

# Hardware order: FPE, Tender, FIGLUT, Omni4, Omni3
hw_order = ['FPE', 'Tender', 'FIGLUT', 'Omni4', 'Omni3']
hw_labels = ['FPE', 'Tender-int8', 'FIGLUT', 'Omni-LUT-KV4', 'Omni-LUT-KV3']

# Input context configurations (fixed output=256)
output_tokens = '512'
input_configs = ['128_'+output_tokens, '1024_'+output_tokens, '8192_'+output_tokens]
input_labels = ['128', '1024', '8192']

# y-axis limits for each row (manually set)
y_limits = [9.39, 5.96, 4.29]  # [row0_ylim, row1_ylim, row2_ylim]
y_limits = [y + 0.5 for y in y_limits]
break_positions = [None, None, None],  # [row0_break, row1_break, row2_break] - None means no break

# Color scheme for hardware (distinct colors, Omni variants similar)
hw_colors = ['#B0B0B0',   # FPE - Gray
             '#51827B',   # Tender - Teal
             '#7C5151',   # FIGLUT - Brown
             '#E68B88',   # Omni4 - Light red-pink
             '#D97573']   # Omni3 - Darker red-pink

# Hardware area mapping
hw_area = {
    'FPE': FPE_AREA,
    'Tender': TENDER_AREA,
    'FIGLUT': FIGLUT_AREA,
    'Omni4': OMNI_AREA,
    'Omni3': OMNI_AREA,
}

# --- 4. Create Figure with Single 3x6 Grid ---
fig, axes = plt.subplots(3, 6, figsize=(24, 6), sharey='row')

n_hw = len(hw_order)
all_models = models_a + models_b   # 6 models total

# --- 5. First pass: compute per-row max for y-axis scaling ---
row_maxes = [0.0] * len(input_configs)
for panel_idx in range(2):
    models = [models_a, models_b][panel_idx]
    for row_idx, input_config in enumerate(input_configs):
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
            fpe_tops_mm = tops_mm_values[0]
            if fpe_tops_mm > 0:
                normed = [v / fpe_tops_mm for v in tops_mm_values]
            else:
                normed = tops_mm_values
            row_maxes[row_idx] = max(row_maxes[row_idx], max(normed))

# Per-row y-limits: max value + padding
y_limits = [mx + 0.5 for mx in row_maxes]

# --- 6. Plot Normalized TOPS/mm^2 for each configuration ---
for panel_idx in range(2):
    models = [models_a, models_b][panel_idx]
    for row_idx, input_config in enumerate(input_configs):
        for col_idx, model in enumerate(models):
            global_col = panel_idx * 3 + col_idx
            ax = axes[row_idx, global_col]

            # Extract total cycles for all hardware and calculate TOPS/mm^2
            tops_mm_values = []

            for hw in hw_order:
                if hw in data[model] and input_config in data[model][hw]:
                    total_cycles = data[model][hw][input_config]['total_cycles']
                    area = hw_area[hw]
                    # TOPS/mm^2 = 1 / cycles / area
                    tops_mm = 1.0 / (total_cycles * area) if total_cycles > 0 and area > 0 else 0
                    tops_mm_values.append(tops_mm)
                else:
                    tops_mm_values.append(0)

            # Normalize to FPE (first hardware)
            fpe_tops_mm = tops_mm_values[0]
            if fpe_tops_mm > 0:
                tops_mm_normalized = [val / fpe_tops_mm for val in tops_mm_values]
            else:
                tops_mm_normalized = tops_mm_values

            # Create bar chart
            x_pos = np.arange(n_hw) * 0.8  # Reduce spacing by multiplying by factor < 1
            bar_width = 0.5

            bars = ax.bar(x_pos, tops_mm_normalized, bar_width,
                          color=hw_colors, edgecolor='black', linewidth=0.8)

            # Add value labels on top of bars
            label_offset = y_limits[row_idx] * 0.02
            for i, val in enumerate(tops_mm_normalized):
                if val > 0:
                    ax.text(x_pos[i], val + label_offset, f'{val:.2f}',
                       ha='center', va='bottom', fontsize=13, fontweight='bold')

            # Formatting
            ax.set_xticks(x_pos)
            ax.set_xticklabels([])  # Never show hardware labels

            # Add input token label on top of center subplot of each row
            if col_idx == 1:
                ax.set_title(f'Input tokens = {input_labels[row_idx]}', fontweight='bold',
                            fontsize=14, loc='center')

            # Add model name on bottom row only
            if row_idx == 2:
                ax.set_xlabel(model, fontweight='bold', fontsize=12)

            # Grid
            ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
            ax.set_axisbelow(True)

            # Remove top and right spines
            sns.despine(ax=ax)

            # Set y-axis limits (per-row)
            ax.set_ylim(0, y_limits[row_idx])

# --- 7. Add shared y-axis label ---
fig.text(0.035, 0.5, 'Normalized TOPS/mm$^2$', va='center', rotation='vertical',
         fontsize=16, fontweight='bold')

# --- 8. Add Panel Captions Above Legend ---
fig.text(0.27, 0.01, '(a) OPT Models', ha='center', va='top',
         fontweight='bold', fontsize=16)
fig.text(0.69, 0.01, '(b) GQA / MoE Models', ha='center', va='top',
         fontweight='bold', fontsize=16)

# --- 9. Add Legend ---
# Create legend patches
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=hw_colors[i], edgecolor='black',
                         linewidth=0.8, label=hw_labels[i])
                   for i in range(n_hw)]

fig.legend(handles=legend_elements, loc='lower center',
          bbox_to_anchor=(0.48, -0.12), ncol=5, frameon=True,
          fontsize=13, edgecolor='black')

# --- 10. Final Layout and Save ---
plt.subplots_adjust(hspace=0.2, wspace=0.0, bottom=0.08, left=0.06)

# Add vertical dotted line between panels (after layout is finalized)
fig.canvas.draw()
pos_left = axes[0, 2].get_position()   # rightmost column of panel (a)
pos_right = axes[0, 3].get_position()  # leftmost column of panel (b)
mid_x = (pos_left.x1 + pos_right.x0) / 2.0
top_y = axes[0, 0].get_position().y1   # top of the top row
bot_y = axes[2, 0].get_position().y0   # bottom of the bottom row
fig.add_artist(plt.Line2D([mid_x, mid_x], [bot_y - 0.10, top_y + 0.05], transform=fig.transFigure,
               color='black', linestyle=':', linewidth=1.5, clip_on=False))

# Save as PDF for publication quality
plt.savefig("tops_mm_analysis.pdf", bbox_inches='tight', dpi=300)
print("Figure saved as tops_mm_analysis.pdf")

# Show the plot
# plt.show()
