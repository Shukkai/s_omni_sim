"""
Plot the stage-by-stage cycle breakdown as stacked bars.

Reads cycle_breakdown.json (output of run_cycle_breakdown.py) and draws two
panels -- (a) Prefill and (b) Decode (per token) -- with one stacked bar per
context length.  Segments are pipeline stages, colored by category:
green shades = AW GEMM, red shades = AA GEMM, grey shades = non-GEMM (VPU).

Usage:
    python plot_cycle_breakdown.py
    python plot_cycle_breakdown.py --normalize
    python plot_cycle_breakdown.py --input cycle_breakdown.json --out my_fig
"""

import argparse
import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
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
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10
matplotlib.rcParams['legend.fontsize'] = 9

# Base colors, matching analysis/throughput/plot_latency_breakdown.py
CATEGORY_BASE = {
    'AW':       '#51827B',   # Green - AW GEMM
    'AA':       '#E68B88',   # Red   - AA GEMM
    'non_gemm': '#B0B0B0',   # Grey  - non-GEMM (VPU)
}
CATEGORY_LABEL = {
    'AW': 'AW-GEMM',
    'AA': 'AA-GEMM',
    'non_gemm': 'non-GEMM',
}

# --- Hardware-unit view: units grouped into color families ---
# LUT datapath = green, floating-point datapath = red, everything else = grey.
UNIT_CATEGORY = {
    'lgu': 'AW',
    'pe_array_compute': 'AW',
    'pe_array_fill_drain': 'AW',
    'fpe_array_compute': 'AA',
    'fpe_array_fill_drain': 'AA',
    'rescale': 'AA',
    'input_load': 'non_gemm',
    'accumulator': 'non_gemm',
    'vpu': 'non_gemm',
    'bqu_tse': 'non_gemm',
    'bqu_bea': 'non_gemm',
}

# Stages below this share of a panel are folded into an "other" segment.
MIN_SHARE = 0.01


def shade(base_hex, i, n):
    """Return the i-th of n shades of *base_hex*, dark -> light."""
    r, g, b = to_rgb(base_hex)
    if n <= 1:
        return (r, g, b)
    # Spread from 0.75x (darker) to blended-with-white (lighter).
    t = i / (n - 1)
    dark = np.array([r, g, b]) * 0.62
    light = np.array([r, g, b]) * 0.45 + 0.55
    return tuple(dark + t * (light - dark))


def build_panel_data(data, phase_key, input_keys):
    """Collect {stage: [value per input length]} plus per-stage category."""
    results = data['results']
    stage_order = data['stage_order']

    # Union of stages present in this phase, in pipeline order.
    stages, categories = [], {}
    for inp in input_keys:
        for stage, rec in results[inp][phase_key].items():
            if stage not in categories:
                categories[stage] = rec['category']
                stages.append(stage)
    stages.sort(key=lambda s: (stage_order.index(s) if s in stage_order
                               else len(stage_order), s))

    values = {s: [results[inp][phase_key].get(s, {}).get('cycles', 0.0)
                  for inp in input_keys]
              for s in stages}
    return stages, categories, values


def build_unit_panel_data(data, phase_key, input_keys):
    """Collect {unit: [cycles per input length]} for the hardware-unit view.

    Overlapped units (the BQU) are excluded from the stacked bars -- they do not
    contribute to serial latency, so stacking them would misrepresent the total.
    """
    units_by_input = data['units']
    unit_order = data['unit_order']
    labels = data['unit_label']

    units, categories, seen = [], {}, set()
    for inp in input_keys:
        for unit, rec in units_by_input[inp][phase_key].items():
            if rec['overlapped'] or unit in seen:
                continue
            seen.add(unit)
            units.append(unit)
            categories[unit] = UNIT_CATEGORY.get(unit, 'non_gemm')
    units.sort(key=lambda u: (unit_order.index(u) if u in unit_order
                              else len(unit_order), u))

    values = {u: [units_by_input[inp][phase_key].get(u, {}).get('cycles', 0.0)
                  for inp in input_keys]
              for u in units}
    # Drop units that are zero everywhere (e.g. LGU in weight-stationary mode).
    units = [u for u in units if any(values[u])]
    values = {u: values[u] for u in units}

    display = {u: labels.get(u, u) for u in units}
    return units, categories, values, display


def fold_small(stages, categories, values):
    """Fold stages below MIN_SHARE (in every bar) into a per-category 'other'."""
    totals = np.sum([values[s] for s in stages], axis=0)
    totals = np.where(totals == 0, 1.0, totals)

    kept, other = [], {}
    for s in stages:
        share = np.array(values[s]) / totals
        if share.max() >= MIN_SHARE:
            kept.append(s)
        else:
            cat = categories[s]
            other.setdefault(cat, np.zeros(len(totals)))
            other[cat] += np.array(values[s])

    for cat, vals in other.items():
        name = f"other ({CATEGORY_LABEL[cat]})"
        kept.append(name)
        categories[name] = cat
        values[name] = vals
    return kept


def assign_colors(stages, categories):
    """Give each stage a distinct shade of its category color."""
    per_cat = {}
    for s in stages:
        per_cat.setdefault(categories[s], []).append(s)
    colors = {}
    for cat, members in per_cat.items():
        for i, s in enumerate(members):
            colors[s] = shade(CATEGORY_BASE[cat], i, len(members))
    return colors


def fmt_cycles(v):
    for unit, div in (('G', 1e9), ('M', 1e6), ('K', 1e3)):
        if v >= div:
            return f"{v / div:.1f}{unit}"
    return f"{v:.0f}"


def main():
    p = argparse.ArgumentParser(description="Plot stage-by-stage cycle breakdown.")
    p.add_argument('--input', default='cycle_breakdown.json')
    p.add_argument('--out', default='cycle_breakdown')
    p.add_argument('--normalize', action='store_true',
                   help="plot % of phase instead of absolute cycles")
    p.add_argument('--view', choices=('stage', 'unit'), default='stage',
                   help="'stage' = per transformer stage; "
                        "'unit' = per hardware unit (LGU / PE array / VPU / ...)")
    args = p.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    cfg = data['config']
    input_keys = sorted(data['results'], key=int)
    x_labels = [f"{int(k) // 1024}K" if int(k) >= 1024 else k for k in input_keys]

    panels = [
        ('prefill', '(a) Prefill'),
        ('decode_per_token', '(b) Decode (per token)'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))

    # A single global color/legend mapping so both panels agree.
    all_stages, all_categories = [], {}
    display = {}
    panel_data = {}
    for phase_key, _ in panels:
        if args.view == 'unit':
            stages, categories, values, disp = build_unit_panel_data(
                data, phase_key, input_keys)
            display.update(disp)
        else:
            stages, categories, values = build_panel_data(
                data, phase_key, input_keys)
            stages = fold_small(stages, categories, values)
        panel_data[phase_key] = (stages, values)
        for s in stages:
            if s not in all_categories:
                all_categories[s] = categories[s]
                all_stages.append(s)

    order = data['unit_order'] if args.view == 'unit' else data['stage_order']
    all_stages.sort(key=lambda s: (
        {'AW': 0, 'AA': 1, 'non_gemm': 2}[all_categories[s]],
        order.index(s) if s in order else len(order), s))
    display = {s: display.get(s, s) for s in all_stages}
    colors = assign_colors(all_stages, all_categories)

    x = np.arange(len(input_keys))
    for ax, (phase_key, title) in zip(axes, panels):
        stages, values = panel_data[phase_key]
        ordered = [s for s in all_stages if s in stages]

        totals = np.sum([values[s] for s in ordered], axis=0)
        scale = np.where(totals == 0, 1.0, totals) if args.normalize else 1.0

        bottom = np.zeros(len(input_keys))
        for s in ordered:
            vals = np.array(values[s]) / scale * (100.0 if args.normalize else 1.0)
            ax.bar(x, vals, bottom=bottom, width=0.6,
                   color=colors[s], edgecolor='white', linewidth=0.4, label=s)
            bottom += vals

        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel('Context length')
        ax.set_axisbelow(True)
        ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if args.normalize:
            ax.set_ylabel('Cycles (% of serial total)' if args.view == 'unit'
                          else 'Cycles (% of phase)')
            ax.set_ylim(0, 105)
        else:
            ax.set_ylabel('Cycles')
            ax.set_ylim(0, bottom.max() * 1.14 if bottom.max() > 0 else 1)
            for xi, total in zip(x, bottom):
                ax.text(xi, total * 1.02, fmt_cycles(total),
                        ha='center', va='bottom', fontsize=9)

    # --- Shared legend: stages grouped by category ---
    ncol = 4 if args.view == 'unit' else 6
    handles = [Patch(facecolor=colors[s], edgecolor='white', label=display[s])
               for s in all_stages]
    fig.legend(handles=handles, loc='lower center', ncol=ncol,
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    hw = cfg['hw']
    fig.suptitle(
        f"{cfg['model']}  |  AW={hw['AW_mode']} AA={hw['AA_mode']}  "
        f"{hw['array_m']}x{hw['array_n']}  "
        f"W{hw['weight_bits']}A{hw['act_bits']}KV{hw['kv_cache_bits']}  "
        f"|  output={cfg['output_tokens']} tokens",
        fontsize=11, y=0.99)

    n_rows = (len(all_stages) + ncol - 1) // ncol
    fig.tight_layout(rect=(0, 0.04 + 0.032 * n_rows, 1, 0.96))

    for ext in ('pdf', 'png'):
        fig.savefig(f"{args.out}.{ext}", dpi=300, bbox_inches='tight')
    print(f"Saved {args.out}.pdf and {args.out}.png")


if __name__ == "__main__":
    main()
