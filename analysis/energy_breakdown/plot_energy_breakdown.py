import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib
import json

# --- 1. Style Setup (matching hybrid_stationary.py) ---
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
with open('hw_comparison_vary_input.json', 'r') as f:
    data_vary_input = json.load(f)

with open('hw_comparison_vary_output.json', 'r') as f:
    data_vary_output = json.load(f)

# --- 3. Data Organization ---
model = "OPT-6.7B"

# Hardware order: FPE, Tender, FIGLUT, Omni4, Omni3
hw_order = ['FPE', 'Tender', 'FIGLUT', 'Omni4', 'Omni3']
hw_labels = ['FPE', 'Tender-int8', 'FIGLUT', 'Omni-LUT\n-KV4', 'Omni-LUT\n-KV3']

# Color scheme for hardware (distinct colors)
hw_colors = ['#E74C3C',   # FPE - Red
             '#F39C12',   # Tender - Orange
             '#3498DB',   # FIGLUT - Blue
             '#9B59B6',   # Omni4 - Purple
             '#1ABC9C']   # Omni3 - Teal

# Color scheme for energy components (stacked bars)
energy_colors = {
    'dram': "#9B6868",      # DRAM - Dark Orange (bottom)
    'sram': "#797171",      # SRAM - Light Orange (middle)
    'compute': "#E68B88"    # Compute - Blue (top)
}

# Vary Input configurations (fixed output=256)
# vary_input_configs = ['1024_256', '2048_256', '4096_256', '8192_256', '32768_256']
vary_input_configs = ['1024_512', '2048_512', '4096_512', '8192_512', '16384_512', '32768_512']
vary_input_labels = ['1024', '2048', '4096', '8192', '16384', '32768']

# Vary Output configurations (fixed input=2048)
vary_output_configs = ['2048_256', '2048_512', '2048_1024', '2048_2048', '2048_4096', '2048_8192']
vary_output_labels = ['256', '512', '1024', '2048', '4096', '8192']

# --- 4. Create Figure with 2 rows x 5 columns ---
fig, axes = plt.subplots(2, 6, figsize=(24, 8), sharey='row')

n_hw = len(hw_order)

# --- 5. Plot Vary Output (Top Row - Fixed Input) ---
for config_idx, config in enumerate(vary_output_configs):
    ax = axes[0, config_idx]
    
    # Extract energy data for all hardware
    dram_energy = []
    sram_energy = []
    compute_energy = []
    total_energy = []
    
    for hw in hw_order:
        if hw in data_vary_output[model] and config in data_vary_output[model][hw]:
            data = data_vary_output[model][hw][config]
            dram_energy.append(data['dram_total_energy'])
            sram_energy.append(data['sram_total_energy'])
            compute_energy.append(data['compute_total_energy'])
            total_energy.append(data['total_energy'])
        else:
            dram_energy.append(0)
            sram_energy.append(0)
            compute_energy.append(0)
            total_energy.append(0)
    
    # Normalize to FPE (first hardware)
    fpe_total = total_energy[0]
    if fpe_total > 0:
        dram_energy_norm = [e / fpe_total for e in dram_energy]
        sram_energy_norm = [e / fpe_total for e in sram_energy]
        compute_energy_norm = [e / fpe_total for e in compute_energy]
        total_energy_norm = [e / fpe_total for e in total_energy]
    else:
        dram_energy_norm = dram_energy
        sram_energy_norm = sram_energy
        compute_energy_norm = compute_energy
        total_energy_norm = total_energy
    
    # Create stacked bar chart
    x_pos = np.arange(n_hw)
    bar_width = 0.6
    
    # Stack: DRAM (bottom), SRAM (middle), Compute (top)
    p1 = ax.bar(x_pos, dram_energy_norm, bar_width, 
                label='DRAM', color=energy_colors['dram'], edgecolor='black', linewidth=0.8)
    p2 = ax.bar(x_pos, sram_energy_norm, bar_width, 
                bottom=dram_energy_norm, label='SRAM', color=energy_colors['sram'], edgecolor='black', linewidth=0.8)
    p3 = ax.bar(x_pos, compute_energy_norm, bar_width, 
                bottom=np.array(dram_energy_norm) + np.array(sram_energy_norm), 
                label='Compute', color=energy_colors['compute'], edgecolor='black', linewidth=0.8)
    
    # Add value labels on top of bars (FPE includes absolute energy)
    for i, total in enumerate(total_energy_norm):
        if i == 0:  # FPE bar: show normalized + absolute
            ax.text(x_pos[i], total + 0.02, f'({fpe_total:.0f} J)\n{total:.2f}',
                    ha='center', va='bottom', fontsize=13, fontweight='bold')
        else:
            ax.text(x_pos[i], total + 0.02, f'{total:.2f}',
                    ha='center', va='bottom', fontsize=13, fontweight='bold')
    
    # Add token label in upper right corner
    output_tokens = vary_output_labels[config_idx]
    ax.text(0.95, 0.95, f'Output tokens =\n{output_tokens}', transform=ax.transAxes,
        ha='right', va='top', fontsize=14)
    
    # Formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels([], rotation=45, ha='right', fontsize=9)  # Hide x-labels for top row
    
    # Set y-axis limits
    ax.set_ylim(0, 1.15)
    
    # Grid
    ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    sns.despine(ax=ax)

# --- 6. Plot Vary Input (Bottom Row - Fixed Output) ---
for config_idx, config in enumerate(vary_input_configs):
    ax = axes[1, config_idx]
    
    # Extract energy data for all hardware
    dram_energy = []
    sram_energy = []
    compute_energy = []
    total_energy = []
    
    for hw in hw_order:
        if hw in data_vary_input[model] and config in data_vary_input[model][hw]:
            data = data_vary_input[model][hw][config]
            dram_energy.append(data['dram_total_energy'])
            sram_energy.append(data['sram_total_energy'])
            compute_energy.append(data['compute_total_energy'])
            total_energy.append(data['total_energy'])
        else:
            dram_energy.append(0)
            sram_energy.append(0)
            compute_energy.append(0)
            total_energy.append(0)
    
    # Normalize to FPE (first hardware)
    fpe_total = total_energy[0]
    if fpe_total > 0:
        dram_energy_norm = [e / fpe_total for e in dram_energy]
        sram_energy_norm = [e / fpe_total for e in sram_energy]
        compute_energy_norm = [e / fpe_total for e in compute_energy]
        total_energy_norm = [e / fpe_total for e in total_energy]
    else:
        dram_energy_norm = dram_energy
        sram_energy_norm = sram_energy
        compute_energy_norm = compute_energy
        total_energy_norm = total_energy
    
    # Create stacked bar chart
    x_pos = np.arange(n_hw)
    bar_width = 0.6
    
    # Stack: DRAM (bottom), SRAM (middle), Compute (top)
    p1 = ax.bar(x_pos, dram_energy_norm, bar_width, 
                label='DRAM', color=energy_colors['dram'], edgecolor='black', linewidth=0.8)
    p2 = ax.bar(x_pos, sram_energy_norm, bar_width, 
                bottom=dram_energy_norm, label='SRAM', color=energy_colors['sram'], edgecolor='black', linewidth=0.8)
    p3 = ax.bar(x_pos, compute_energy_norm, bar_width, 
                bottom=np.array(dram_energy_norm) + np.array(sram_energy_norm), 
                label='Compute', color=energy_colors['compute'], edgecolor='black', linewidth=0.8)
    
    # Add value labels on top of bars (FPE includes absolute energy)
    for i, total in enumerate(total_energy_norm):
        if i == 0:  # FPE bar: show normalized + absolute
            ax.text(x_pos[i], total + 0.02, f'({fpe_total:.0f} J)\n{total:.2f}',
                    ha='center', va='bottom', fontsize=13, fontweight='bold')
        else:
            ax.text(x_pos[i], total + 0.02, f'{total:.2f}',
                    ha='center', va='bottom', fontsize=13, fontweight='bold')
    
    # Add token label in upper right corner (placed above plot area)
    input_tokens = vary_input_labels[config_idx]
    ax.text(0.95, 0.99, f'Input tokens =\n{input_tokens}', transform=ax.transAxes,
            ha='right', va='top', fontsize=14)
    
    # Formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels(hw_labels, rotation=45, ha='right', fontsize=12)  # Show x-labels for bottom row only
    
    # Set y-axis limits to align all subplots in this row
    ax.set_ylim(0, 1.1)
    
    # Grid
    ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    sns.despine(ax=ax)

# --- 7. Add Row Titles and Labels ---
# Add title for top row (fixed input)
fig.text(0.48, 0.96, 'Input tokens = 2048', ha='center', va='top',
         fontsize=15, fontweight='bold')

# Add (a) label for top row
fig.text(0.48, 0.575, '(a)', ha='center', va='top',
         fontsize=15, fontweight='bold')

# Add title for bottom row (fixed output)
# fig.text(0.48, 0.52, 'Output tokens = 256', ha='center', va='top',
#          fontsize=13, fontweight='bold')
fig.text(0.48, 0.53, 'Output tokens = 512', ha='center', va='top',
         fontsize=15, fontweight='bold')

# Add (b) label for bottom row
fig.text(0.48, 0.09, '(b)', ha='center', va='top',
         fontsize=15, fontweight='bold')

# Add horizontal separator line between rows
from matplotlib.lines import Line2D
fig.add_artist(Line2D([0.06, 0.92], [0.54, 0.54], 
                      transform=fig.transFigure, color='black', 
                      linewidth=1.0, linestyle='--'))

# --- 8. Add Legend and Labels ---
handles = [p1, p2, p3]
labels = ['DRAM', 'SRAM', 'Compute unit']
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.02),
          ncol=3, frameon=True, fontsize=13, edgecolor='black')

# Add y-label for entire figure
fig.text(0.03, 0.5, 'Normalized Energy Consumption', va='center', rotation='vertical', 
         fontsize=16, fontweight='bold')

# --- 9. Layout and Save ---
plt.subplots_adjust(hspace=0.35, wspace=0, bottom=0.15, left=0.06, top=0.92)
plt.savefig("energy_breakdown_analysis.pdf", bbox_inches='tight', dpi=300)
print("Figure saved as energy_breakdown_analysis.pdf")

# Show the plot
# plt.show()
