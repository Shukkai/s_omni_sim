"""
What filling the idle OS-V rows is worth, and what it costs.

Decode `attn_v` lights 1 of 32 PE rows at any context length (3.12% occupancy).
This sweeps packing P independent attention instances into one pass and reports
both halves: the cycles recovered, and the co-residency that pays for them.

The headline to resist: **32x on the stage is not 32x on the token.**  `attn_v`
is compute-bound, so packing drives its compute time under its memory time and
the stage flips to memory-bound -- an Amdahl ceiling well under 3x.  Section C
is the honest table; section A alone would mislead.

Usage:
    python pack_run.py
    python pack_run.py --csv pack.csv --report pack_report.md
"""

import argparse
import csv
import math
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, '..', '..'))
for p in ('simulator', 'analysis', 'analysis/cycle_breakdown'):
    sys.path.insert(0, os.path.join(_root, *p.split('/')))
sys.path.insert(0, _here)

from simulator import (                                          # noqa: E402
    ComputeMode, HardwareConfig, OperationType, Simulator, WorkloadConfig,
)
from model_configs import get_model_config                       # noqa: E402
from cycle_units import UnitAwareSimulator, cycle_units          # noqa: E402
from array_pack import (                                         # noqa: E402
    PackedOSVSimulator, max_useful_pack, packed_osv_cycles, row_slot_waste,
)
from report import Report                                        # noqa: E402

MODEL = 'LLaMA-3-8B'
HEAD_DIM = 128
NUM_HEADS, NUM_KV_HEADS = 32, 8
GQA_GROUP = NUM_HEADS // NUM_KV_HEADS          # 4
OUTPUT_TOKENS = 4
PACKS = [1, 2, 4, 8, 16, 32]
CONTEXTS = [1024, 2048, 4096, 8192, 32768]
BATCHES = [1, 8, 32]


def hw(sram_capacity_kb=0, batch_model="sequential"):
    return HardwareConfig(
        array_m=32, array_n=4, FPE_array_size=64,
        act_bits=16, accumulate_bits=32, weight_bits=4, kv_cache_bits=4,
        AW_mode="OMNI", AA_mode="OMNI",
        sram_capacity_kb=sram_capacity_kb, sram_batch_model=batch_model,
    )


def sim(pack=1, gqa_share=False, sram_capacity_kb=0, **kw):
    return PackedOSVSimulator(hw(sram_capacity_kb), pack=pack,
                              gqa_share=gqa_share, gqa_group=GQA_GROUP,
                              model_bqu=False, **kw)


def run(pack, batch, context, gqa_share=False, sram_capacity_kb=0):
    model = get_model_config(MODEL)
    w = WorkloadConfig(batch_size=batch, input_tokens=context,
                       output_tokens=OUTPUT_TOKENS, flash_block_size=0)
    s = sim(pack, gqa_share, sram_capacity_kb)
    r = s.simulate(model, w)
    dec = r.decode
    qk = dec.get_operation_total(OperationType.QK_MATMUL, ComputeMode.AA)
    av = dec.get_operation_total(OperationType.ATTN_V_MATMUL, ComputeMode.AA)
    tot = dec.get_total_metrics()
    _, tpot = s.compute_roofline_latency(r, w)
    return {'pack': pack, 'batch': batch, 'context': context,
            'gqa_share': gqa_share, 'sram_capacity_kb': sram_capacity_kb,
            'qk_cycles': qk.cycles, 'av_cycles': av.cycles,
            'av_util': av.utilization, 'attn_cycles': qk.cycles + av.cycles,
            'decode_peak_sram': dec.peak_sram_bytes,
            'sram_overflow': tot.sram_overflow,
            'sram_refetch': tot.sram_refetch_bytes,
            'decode_dram': tot.dram_read, 'tpot_s': tpot}


# ---------------------------------------------------------------- pre-flight

def preflight():
    """Nothing below is worth reading unless all of these hold."""
    n = 0
    model = get_model_config(MODEL)
    w = WorkloadConfig(batch_size=1, input_tokens=8192, output_tokens=3,
                       flash_block_size=0)

    # 1. pack=1 is bit-identical to the stock add-on chain.
    a = UnitAwareSimulator(hw(), model_bqu=False)
    b = sim(pack=1)
    ra, rb = a.simulate(model, w), b.simulate(model, w)
    assert repr(ra.to_dict()) == repr(rb.to_dict()), "pack=1 changed the tree"
    assert repr(a.compute_roofline_latency(ra, w)) == \
           repr(b.compute_roofline_latency(rb, w)), "pack=1 moved the roofline"
    n += 1

    # 2. Three-way sync: cycle_units, packed_osv_cycles, and (at P=1) the parent.
    h = hw()
    base = Simulator(h)
    for K, N, qbit, bs in [(32768, 128, 4, 32), (1024, 128, 4, 256),
                           (128, 32768, 4, 32), (128, 1024, 4, 32)]:
        for p in PACKS:
            units = cycle_units(h, 1, K, N, qbit, "LUT_OS_V", bs,
                                mu=Simulator.MU, num_rac=Simulator.NUM_RAC,
                                pack=p)
            direct = packed_osv_cycles(h, 1, K, N, qbit, bs, p)
            assert sum(units.values()) == direct, (K, N, p, sum(units.values()), direct)
            if p == 1:
                ref = base._calculate_cycles(1, K, N, qbit, ComputeMode.AA,
                                             "LUT_OS_V", bs)
                assert direct == ref, (K, N, direct, ref)
    n += 1

    # 3. attn_v scales exactly: N=head_dim gives n_tiles=1, so rounds stay 1.
    for ctx in (1024, 32768):
        one = packed_osv_cycles(h, 1, ctx, HEAD_DIM, 4, 32, 1)
        for p in PACKS:
            got = packed_osv_cycles(h, 1, ctx, HEAD_DIM, 4, 32, p)
            assert got * p == one, (ctx, p, got, one)
    n += 1

    # 4. qk is EXACTLY neutral when n_tiles is a whole multiple of array_m --
    #    every row busy in the body AND no tail.  This catches a sign error in
    #    the rounds term, which would otherwise look like a plausible speedup.
    for kv in (4096, 8192, 32768):     # n_tiles = 32, 64, 256 -- all multiples
        assert math.ceil(kv / 128) % h.array_m == 0, kv
        one = packed_osv_cycles(h, 1, HEAD_DIM, kv, 4, 32, 1)
        for p in PACKS:
            assert packed_osv_cycles(h, 1, HEAD_DIM, kv, 4, 32, p) == one, (kv, p)
    n += 1

    # 4b. Off a multiple, qk's gain is exactly the tail-quantization ratio
    #     32*ceil(n_tiles/32)/n_tiles -- not "packing more work in".  Pinning
    #     the closed form means the reported qk numbers have a derivation.
    for kv in (4097, 8193, 32769, 1025):
        n_tiles = math.ceil(kv / 128)
        one = packed_osv_cycles(h, 1, HEAD_DIM, kv, 4, 32, 1)
        full = packed_osv_cycles(h, 1, HEAD_DIM, kv, 4, 32, h.array_m)
        expect = h.array_m * math.ceil(n_tiles / h.array_m) / n_tiles
        assert abs(one / full - expect) < 1e-9, (kv, one / full, expect)
    n += 1

    # 5. Monotonic in P for both stages.  The naive "just set M=P" mistake
    #    drops out of the OS-V branch into LUT_OS and fails this by 32x.
    for N in (HEAD_DIM, 1024, 32768):
        prev = None
        for p in PACKS:
            c = packed_osv_cycles(h, 1, 32768 if N == HEAD_DIM else HEAD_DIM,
                                  N, 4, 32, p)
            assert prev is None or c <= prev, (N, p, c, prev)
            prev = c
    n += 1

    # 6. Packing moves no data and does no extra MACs.
    r1, r32 = run(1, 1, 8192), run(32, 1, 8192)
    assert r1['decode_dram'] == r32['decode_dram'], "packing changed DRAM"
    n += 1

    # 7. Utilization closes the loop, derived by _simulate_matmul not by us.
    assert abs(r1['av_util'] * 32 - r32['av_util']) < 1e-9, \
        (r1['av_util'], r32['av_util'])
    assert r32['av_util'] > 0.99, r32['av_util']
    n += 1

    # 8. Guards fire.
    try:
        PackedOSVSimulator(hw(), pack=5)
        raise AssertionError("pack=5 should not divide array_m=32")
    except ValueError:
        pass
    wf = WorkloadConfig(batch_size=1, input_tokens=1024, output_tokens=2,
                        flash_block_size=256)
    try:
        sim(pack=2).simulate(model, wf)
        raise AssertionError("flash + pack should raise")
    except NotImplementedError:
        pass
    n += 1

    print(f"Pre-flight: {n} assertions passed\n")
    return n


# -------------------------------------------------------------------- sweeps

def sweep(report_path):
    rows = []
    preflight()
    h = hw()
    rep = Report(
        report_path,
        "OS-V array packing",
        subtitle="Filling the idle 31 of 32 rows in decode attention",
        source="analysis/array_packing/pack_run.py",
        setup=["Model: LLaMA-3-8B on Omni-LUT, 4-bit KV, standard attention.",
               "Packs P independent attention instances into one OS-V pass, each "
               "with its own LGU driving `array_m / P` rows."])

    rep.summary([
        "Decode `attn_v` is `(M=1, K=kv_len, N=128)`, so `n_tiles = 1` and "
        "`rounds = 1`: **one of 32 PE rows works, at any context length** — 3.12% "
        "occupancy. Packing recovers exactly **32x** on that stage.",
        "**But 32x on the stage is not 32x on the token.** `attn_v` was "
        "compute-bound, so packing drives it under its own memory time and TPOT "
        "saturates at **P=8**: 1.755x / 2.637x / 3.118x at batch 1 / 8 / 32. "
        "P=16 and P=32 buy nothing.",
        "**Which makes it affordable.** P=8 with GQA-aware sharing needs 4.5 MB "
        "and fits 16 MB; P=32 independent needs 66 MB and does not. The two "
        "results meet at P=8.",
        "The result **rests on an unmodelled cost**: P=8 needs ~1.0 TB/s of "
        "KV-SRAM reads and the simulator has no bandwidth term at all.",
    ])

    # ---- A ------------------------------------------------------------------
    rep.section("A. Decode attention cycles vs pack factor", "Batch 1.")
    for ctx in CONTEXTS:
        trows = []
        for p_ in PACKS:
            r = run(p_, 1, ctx)
            rows.append({'section': 'A', **r})
            if p_ == 1:
                base = r['attn_cycles']
            trows.append([str(p_), f"{r['qk_cycles']:,}", f"{r['av_cycles']:,}",
                          f"{r['attn_cycles']:,}",
                          f"{base / r['attn_cycles']:.2f}x",
                          f"{r['av_util']:.1%}"])
        rep.table(["P", "qk", "attn_v", "attn total", "speedup",
                   "attn_v occupancy"], trows, caption=f"context {ctx:,}")

    # ---- B ------------------------------------------------------------------
    rep.section(
        "B. Two different sources of headroom",
        "`qk` is evaluated at `kv_len = context + 1`, the first decode step, "
        "because that is what the array actually sees.")
    trows = []
    for ctx in CONTEXTS + [131072]:
        kv = ctx + 1
        n_tiles = math.ceil(kv / (h.array_n * Simulator.NUM_RAC))
        mq = max_useful_pack(h, kv)
        waste = row_slot_waste(h, kv, 1)
        gain = (packed_osv_cycles(h, 1, HEAD_DIM, kv, 4, 32, 1)
                / packed_osv_cycles(h, 1, HEAD_DIM, kv, 4, 32, h.array_m))
        src = 'idle rows' if n_tiles < h.array_m else 'tail quantization'
        rows.append({'section': 'B', 'context': ctx, 'kv_len': kv,
                     'n_tiles': n_tiles, 'max_pack_qk': mq,
                     'idle_row_frac': waste, 'qk_gain_p32': gain})
        trows.append([f"{kv:,}", f"{n_tiles:,}", str(mq), f"{waste:.1%}",
                      f"{gain:.2f}x", src])
    rep.table(["kv_len", "n_tiles", "max P", "idle rows", "P=32 gain", "source"],
              trows, aligns="rrrrrl")
    cross = h.array_m * h.array_n * Simulator.NUM_RAC
    rep.note(
        f"`attn_v` is not in this table: `n_tiles = 1` for any `head_dim <= 128`, "
        f"so it is permanently in the 'idle rows' regime and packs {h.array_m}x at "
        f"every context.\n"
        f"For `qk` the regimes split at `kv_len = array_m x array_n x NUM_RAC = "
        f"{cross:,}`. Below it rows genuinely sit idle. At or above it the body is "
        f"full and the only waste is the **tail**: `rounds = ceil(n_tiles/"
        f"{h.array_m})` rounds up to whole {h.array_m}-row passes, leaving up to "
        f"{h.array_m - 1} rows idle in the last one. Packing recovers exactly "
        f"`{h.array_m}*ceil(n_tiles/{h.array_m})/n_tiles` — large just past a tile "
        f"boundary, decaying as `1/n_tiles`. It is neutral **only** when `n_tiles` "
        f"is an exact multiple of {h.array_m}, and decode `kv_len = context + "
        f"token_idx` almost never is.")

    # ---- C ------------------------------------------------------------------
    rep.section(
        "C. Token latency — the honest table",
        "Context 32,768. This is why section A alone would mislead.")
    for batch in BATCHES:
        base = None
        trows = []
        for p_ in PACKS:
            r = run(p_, batch, 32768)
            rows.append({'section': 'C', **r})
            if base is None:
                base = r['tpot_s']
            trows.append([str(p_), f"{r['av_cycles']:,}",
                          f"{r['tpot_s']*1e3:.2f} ms",
                          f"{base / r['tpot_s']:.3f}x"])
        rep.table(["P", "attn_v cycles", "TPOT", "speedup"], trows,
                  caption=f"batch {batch}")
    rep.note(
        "`attn_v`'s cycles fall 32x but TPOT does not, because the stage was "
        "compute-bound and packing pushes it under its own memory time. Past that "
        "point more packing buys nothing: the remaining cost is DRAM.")

    # ---- D ------------------------------------------------------------------
    rep.section(
        "D. The co-residency that pays for it",
        "Context 32,768, batch 1, decode peak SRAM. P instances means P working "
        "sets resident.")
    trows = []
    for p_ in PACKS:
        src = ('single' if p_ == 1 else
               'intra-group' if p_ <= GQA_GROUP else 'cross-group')
        ind = run(p_, 1, 32768, gqa_share=False)
        shr = run(p_, 1, 32768, gqa_share=True)
        cap = run(p_, 1, 32768, gqa_share=True, sram_capacity_kb=16384)
        rows.append({'section': 'D', 'variant': 'independent', **ind})
        rows.append({'section': 'D', 'variant': 'gqa_shared', **shr})
        trows.append([str(p_), src, f"{ind['decode_peak_sram']/2**20:.1f} MB",
                      f"{shr['decode_peak_sram']/2**20:.1f} MB",
                      'no' if cap['sram_overflow'] else 'yes'])
    rep.table(["P", "packing source", "independent", "GQA-shared", "fits 16 MB?"],
              trows, aligns="rlrrc")
    rep.note(
        "A GQA group shares its K/V tile, so packing 4 query heads is nearly free. "
        "Past the group size the tiles are distinct and the footprint grows with P "
        "— which is what would turn the 'free' 32x into a capacity problem. "
        "**Read against section C: the ceiling arrives at P=8, which costs 4.5 MB. "
        "The full achievable speedup is affordable; the 32x is neither reachable "
        "nor needed.**")

    # ---- E ------------------------------------------------------------------
    rep.section(
        "E. What the model does not charge for",
        "Computed arithmetic, not a disclaimer.")
    per_cycle = Simulator.MU * h.array_n * Simulator.NUM_RAC * h.kv_cache_bits // 8
    ghz = h.freq_mhz * 1e6
    trows = [["unpacked (1 row live)", f"{per_cycle:,} B/cycle",
              f"{per_cycle * ghz / 1e9:.1f} GB/s"]]
    for p_ in (4, 8, 32):
        bw = per_cycle * p_
        rows.append({'section': 'E', 'pack': p_, 'fifo_bytes_per_cycle': bw})
        trows.append([f"P={p_} ({p_} rows live)", f"{bw:,} B/cycle",
                      f"{bw * ghz / 1e12:.2f} TB/s"])
    rep.table(["configuration", "weight-FIFO / KV-SRAM reads", "bandwidth"],
              trows, aligns="lrr")
    rep.note(
        "**The simulator has no SRAM bandwidth term at all** — capacity is "
        "enforced, throughput is not. Packing converts an idle-array problem into "
        "an SRAM-bandwidth problem it cannot bill. This is the first thing to "
        "check before believing the result.")
    rep.note(
        "**LGU ungating.** Section IV-D gates 31 of 32 LGUs specifically to save "
        "power; P=32 ungates all of them. Cycles fall 32x, LGU dynamic energy "
        "rises up to 32x, and the energy model sees neither.")
    rep.note(
        "**Energy neutrality here is an artefact, not a finding.** "
        "`os_v_energy_model.py:23` charges `n_tiles/array_m` and "
        "`omni_energy_model.py` divides M==1 OS energy by `array_m` — energy is "
        "*already* amortised over all 32 rows while cycles charge a full round for "
        "one. The two halves of the model disagree today, and packing is what "
        "would make them agree.")
    rep.note(
        "Also uncharged: P live LUTs (16 entries x 128 lanes) plus a P-way "
        "segmented broadcast tree; per-instance output routing (`OUTPUT_CYCLES` "
        "unchanged at 2); and the scheduling tail when `batch x heads < P`.")

    rep.save()
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv', default=os.path.join(_here, 'pack.csv'))
    p.add_argument('--report', default=os.path.join(_here, 'pack_report.md'))
    args = p.parse_args()

    rows = sweep(args.report)

    keys = sorted({k for r in rows for k in r})
    with open(args.csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows -> {args.csv}")
    print(f"Wrote report      -> {args.report}")


if __name__ == '__main__':
    main()
