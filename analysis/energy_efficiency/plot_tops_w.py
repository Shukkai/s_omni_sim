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

# --- 2. Load Data ---
with open('hw_tops-w.json', 'r') as f:
    data = json.load(f)

# --- 3. Data Organization ---
# Models split into two panels
models_a = ['OPT-1.3B', 'OPT-6.7B', 'OPT-30B']          # Panel (a)
models_b = ['LLaMA-3-8B', 'Mixtral-8x7B', 'Qwen3-30B-A3B']  # Panel (b)

# Hardware order: FPE, Tender, FIGLUT, Omni4, Omni3
hw_order = ['FPE', 'Tender', 'FIGLUT', 'Omni4', 'Omni3']
hw_labels = ['FPE', 'Tender-int8', 'FIGLUT', 'Omni-LUT-KV4', 'Omni-LUT-KV3']

# Input context configurations (fixed output=256)
# input_configs = ['128_256', '1024_256', '8192_256']
input_configs = ['1024_512', '8192_512', '32768_512']
input_labels = ['1024', '8192', '32768']

# input_configs = ['256_512', '2048_512', '16384_512']
# input_labels = ['256', '2048', '16384']

# Color scheme for hardware (distinct colors, Omni variants similar)
hw_colors = ['#B0B0B0',   # FPE - Gray
             '#51827B',   # Tender - Teal
             '#7C5151',   # FIGLUT - Brown
             '#E68B88',   # Omni4 - Light red-pink
             '#D97573']   # Omni3 - Darker red-pink

# --- 4. Create Figure with Single 3x6 Grid ---
fig, axes = plt.subplots(3, 6, figsize=(24, 6), sharey='row')

n_hw = len(hw_order)
all_models = models_a + models_b   # 6 models total

# --- 5. Plot Normalized TOPS/W for each configuration ---
for panel_idx in range(2):
    models = [models_a, models_b][panel_idx]
    for row_idx, input_config in enumerate(input_configs):
        for col_idx, model in enumerate(models):
            global_col = panel_idx * 3 + col_idx
            ax = axes[row_idx, global_col]
        
            # Extract total energy for all hardware
            tops_w_values = []
            
            for hw in hw_order:
                if hw in data[model] and input_config in data[model][hw]:
                    total_energy = data[model][hw][input_config]['total_energy']
                    # TOPS/W = 1 / total_energy
                    tops_w = 1.0 / total_energy if total_energy > 0 else 0
                    tops_w_values.append(tops_w)
                else:
                    tops_w_values.append(0)
            
            # Normalize to FPE (first hardware)
            fpe_tops_w = tops_w_values[0]
            if fpe_tops_w > 0:
                tops_w_normalized = [val / fpe_tops_w for val in tops_w_values]
            else:
                tops_w_normalized = tops_w_values
            
            # Create bar chart
            x_pos = np.arange(n_hw) * 0.8  # Reduce spacing by multiplying by factor < 1
            bar_width = 0.5
            
            bars = ax.bar(x_pos, tops_w_normalized, bar_width,
                          color=hw_colors, edgecolor='black', linewidth=0.8)
            
            # Add value labels on top of bars
            for i, val in enumerate(tops_w_normalized):
                if val > 0:
                    ax.text(x_pos[i], val + 0.05, f'{val:.2f}',
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
            
            # Set y-axis limits
            ax.set_ylim(0, 2.4)

# --- 6. Add shared y-axis label ---
fig.text(0.035, 0.5, 'Normalized TOPS/W', va='center', rotation='vertical',
         fontsize=16, fontweight='bold')

# --- 7. Add Panel Captions Above Legend ---
fig.text(0.27, 0.01, '(a) OPT Models', ha='center', va='top',
         fontweight='bold', fontsize=16)
fig.text(0.69, 0.01, '(b) GQA / MoE Models', ha='center', va='top',
         fontweight='bold', fontsize=16)

# --- 8. Add Legend ---
# Create legend patches
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=hw_colors[i], edgecolor='black',
                         linewidth=0.8, label=hw_labels[i])
                   for i in range(n_hw)]

fig.legend(handles=legend_elements, loc='lower center',
          bbox_to_anchor=(0.48, -0.12), ncol=5, frameon=True,
          fontsize=13, edgecolor='black')

# --- 9. Final Layout and Save ---
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
plt.savefig("tops_w_analysis.pdf", bbox_inches='tight', dpi=300)
print("Figure saved as tops_w_analysis.pdf")

# Show the plot
# plt.show()
