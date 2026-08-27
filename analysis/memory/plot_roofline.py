"""The decode roofline: why a 14.2 FLOP/byte operation is compute-bound.

The textbook reading of this machine says decode attention is memory-bound:
its arithmetic intensity is 14.2 FLOP/byte against a ridge point of
`2 x LANES_EQUIV x freq / dram_bw` = 80 FLOP/byte, so it sits five and a half
times inside the bandwidth-limited region.

That reading is wrong, and the plot shows why.  The 80 FLOP/byte ridge assumes
all 4,096 lanes are working.  Decode `attn_v` is issued as `(M=1, K=kv_len,
N=head_dim)`, which lights one of 32 PE rows, and it attains **125 GFLOP/s --
3.1% of peak** at every batch and every context.  Against *that* ceiling the
ridge is `125 / 51.2` = **2.4 FLOP/byte**, and an intensity of 14.2 is well
clear of it.  The operation is compute-bound not because it does much
arithmetic but because the array is bad at this shape.

The same construction gives the SRAM roof at the array's own operand port
(`MU x array_n x NUM_RAC x kv_bits` = 256 B/cycle = 128 GB/s), whose nominal
ridge is 32 FLOP/byte and whose effective ridge under the OS-V ceiling is
0.98 FLOP/byte -- below every operation here, which is why the third leg never
sets the roofline.

Run:  python analysis/memory/plot_roofline.py
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

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
matplotlib.rcParams['legend.fontsize'] = 9

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, '..', '..'))
for _p in ('simulator', 'analysis', 'analysis/memory'):
    sys.path.insert(0, os.path.join(_root, *_p.split('/')))

import sram_run as S                                              # noqa: E402
from simulator import Simulator                                   # noqa: E402

FREQ = 500e6
PEAK = Simulator.LANES_EQUIV * 2 * FREQ / 1e9      # 4096 GFLOP/s
DRAM_BW, SRAM_BW = 51.2, 128.0                     # GB/s

#: What a 32-deep request queue actually sustains at 90 ns and a 64 B burst
#: (Little's law).  The datasheet roof is the one every published number here
#: was drawn against; this is the one a real requester reaches.
DRAM_BW_REAL = 32 * 64 / 90e-9 / 1e9               # 22.8 GB/s

ATTN = ('qk_matmul', 'attn_v_matmul')


def ops(batch, ctx):
    """`[(name, I_dram, I_sram, GFLOP/s)]` for one decode step."""
    r = S.simulate(batch, ctx, 0, 0.0)
    out = []
    for group in (r.decode.aw_ops, r.decode.aa_ops):
        for op, lst in group.items():
            f = sum(m.flops for m in lst)
            d = sum(m.dram_read_eff + m.dram_write_eff for m in lst)
            sm = sum(m.sram_read + m.sram_write for m in lst)
            cy = sum(m.cycles for m in lst)
            if not (f and d and cy):
                continue
            out.append((op.value, f / d, f / sm if sm else 0.0,
                        f / (cy / FREQ) / 1e9))
    return out


def panel(ax, batch, ctx):
    I = np.logspace(-0.5, 3, 400)
    ax.plot(I, np.minimum(PEAK, DRAM_BW * I), color='#222', lw=1.8,
            label=f'DRAM roof, datasheet ({DRAM_BW:.1f} GB/s)')
    ax.plot(I, np.minimum(PEAK, DRAM_BW_REAL * I), color='#e08214', lw=1.8,
            label=f'DRAM roof, 32-deep queue ({DRAM_BW_REAL:.1f} GB/s)')
    ax.plot(I, np.minimum(PEAK, SRAM_BW * I), color='#777', lw=1.4, ls='--',
            label=f'SRAM operand port ({SRAM_BW:.0f} GB/s)')

    pts = ops(batch, ctx)
    osv = min(p[3] for p in pts if p[0] == 'attn_v_matmul')
    ax.axhline(osv, color='#c0392b', lw=1.1, ls=':',
               label=f'OS-V M=1 ceiling ({osv:.0f} GFLOP/s, '
                     f'{osv/PEAK*100:.1f}% of peak)')
    # Effective ridge under that ceiling.
    ax.plot([osv / DRAM_BW], [osv], marker='|', ms=11, color='#c0392b')
    ax.annotate(f'effective ridge\n{osv/DRAM_BW:.1f} FLOP/byte',
                xy=(osv / DRAM_BW, osv), xytext=(osv / DRAM_BW * 0.16, osv * 2.4),
                fontsize=8, color='#c0392b',
                arrowprops=dict(arrowstyle='->', color='#c0392b', lw=0.8))

    # Weight-fed operations cluster on one point; label the cluster, not each.
    wt = [(i_d, rate) for name, i_d, _s, rate in pts if name not in ATTN]
    ax.plot([p[0] for p in wt], [p[1] for p in wt], ls='none', marker='s',
            ms=5, color='#2c6fbb', zorder=5, mec='white', mew=0.8,
            label='projections + FFN' if batch == 1 else None)
    hi = max(wt, key=lambda p: p[1])
    lo = min(wt, key=lambda p: p[1])
    ax.annotate('projections + FFN', xy=hi, xytext=(6, 7),
                textcoords='offset points', fontsize=8, color='#2c6fbb')
    if lo[1] < hi[1] * 0.5:
        ax.annotate('k/v_proj', xy=lo, xytext=(6, -4),
                    textcoords='offset points', fontsize=7.5, color='#2c6fbb')

    for name, i_d, _i_s, rate in pts:
        if name not in ATTN:
            continue
        ax.plot([i_d], [rate], marker='o', ms=7.5, color='#c0392b',
                zorder=6, mec='white', mew=0.9,
                label='attention' if (batch == 1 and name == ATTN[0]) else None)
        ax.annotate(name.replace('_matmul', ''), xy=(i_d, rate),
                    xytext=(7, -10 if name == 'attn_v_matmul' else 5),
                    textcoords='offset points', fontsize=8.5, color='#c0392b')

    ax.axvline(PEAK / DRAM_BW, color='#999', lw=0.8, ls='-.')
    ax.axvline(PEAK / DRAM_BW_REAL, color='#e08214', lw=0.8, ls='-.',
               alpha=0.7)
    ax.annotate(f'nominal ridge  {PEAK/DRAM_BW:.0f} FLOP/byte',
                xy=(PEAK / DRAM_BW, PEAK * 1.55), fontsize=8, color='#666',
                ha='right', va='center', xytext=(-6, 0),
                textcoords='offset points')

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(0.3, 2000); ax.set_ylim(15, PEAK * 2.6)
    ax.set_xlabel('arithmetic intensity (FLOP / byte of DRAM)')
    ax.set_title(f'batch {batch}, {ctx // 1024}K context')
    ax.grid(alpha=0.25, which='both', lw=0.4)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    panel(axes[0], 1, 2048)
    panel(axes[1], 32, 32768)
    axes[0].set_ylabel('attained rate (GFLOP/s)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, -0.10))
    fig.suptitle('Decode roofline — two roofs the datasheet does not show: '
                 'the array at M=1, and a finite request queue', y=0.99)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(_here, f'roofline.{ext}'), dpi=200,
                    bbox_inches='tight')
    print('Wrote', os.path.join(_here, 'roofline.png'))

    print()
    print('%-16s %10s %10s %12s %8s'
          % ('op', 'I_dram', 'I_sram', 'GFLOP/s', 'of peak'))
    for name, i_d, i_s, rate in ops(1, 32768):
        print('%-16s %10.1f %10.1f %12.1f %7.1f%%'
              % (name, i_d, i_s, rate, rate / PEAK * 100))


if __name__ == '__main__':
    main()
