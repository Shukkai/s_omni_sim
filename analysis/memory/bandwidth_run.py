"""
Throughput: what the memory *technology* decides, and what SRAM bandwidth costs.

Two questions, together because they are the same question at two levels of the
hierarchy.

**1. DRAM.**  The simulator carried `dram_bandwidth_gbps` and `dram_burst_bytes`
as independent knobs, which lets a sweep describe a part that does not exist.
A real technology fixes both, and `study.md` §15 showed the *burst* is
what decides whether a pruning axis can collect its saving at all.  So the
interesting column here is not HBM's bandwidth, it is HBM's **32 B access
granularity**: it halves the channel-group size at which channel pruning starts
paying.  `simulator/memory_tech.py` holds the presets with their derivations.

**2. SRAM.**  `sram_capacity_kb` enforced capacity; nothing ever enforced
throughput, so an operation could move unbounded bytes per cycle to and from
SRAM for free.  `hw.sram_bandwidth_gbps` closes that, and the array geometry
implies about `MU x array_n x NUM_RAC x kv_bits` = 256 B/cycle = 128 GB/s.

**The headline result is a warning, and it is why the field ships inert.**  At
128 GB/s decode is essentially unaffected (TPOT 1.074x) but prefill TTFT goes
**4.35x**, because prefill charges **113,670 GB** of SRAM traffic against 3 GB
of DRAM -- a ratio in the tens of thousands to one, which is not physical.  It
is the same untiled-activation defect that makes `_calculate_peak_sram` claim a
2.1 GB prefill working set (`study.md` §7): the SRAM traffic terms are written
against an untiled A, so they count re-reads a tiled loop nest would not
perform.  **Decode is sound -- M=1 makes it tiling-inert -- and prefill is not
usable until that is fixed.**  Reporting a 4.35x TTFT as a bandwidth finding
would be reporting a known modelling bug as a hardware result.

Usage:
    python bandwidth_run.py
    python bandwidth_run.py --csv bandwidth.csv --report bandwidth_report.md
"""

import argparse
import csv
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, '..', '..'))
for p in ('simulator', 'analysis', 'analysis/cycle_breakdown',
          'analysis/compact_breakdown', 'analysis/memory'):
    sys.path.insert(0, os.path.join(_root, *p.split('/')))

from simulator import HardwareConfig, WorkloadConfig, Simulator      # noqa: E402
from memory_tech import (                                            # noqa: E402
    MEMORY_TECHNOLOGIES, DEFAULT_TECHNOLOGY, with_memory_technology,
    memory_technology,
)
from model_configs import get_model_config                           # noqa: E402
from cycle_units import UnitAwareSimulator                           # noqa: E402
from kv_budget import KVBudgetSimulator                              # noqa: E402
from report import Report                                            # noqa: E402
from unstructured_kv import UnstructuredKVSimulator                  # noqa: E402

MODEL = 'LLaMA-3-8B'
CONTEXT = 32768
OUTPUT_TOKENS = 4
HEAD_DIM = 128
KV_BITS = 4
BATCHES = [1, 32]
TECHS = ['DDR5-6400', 'DDR5-6400-x2', 'LPDDR5X-8533', 'GDDR6-16',
         'HBM2E', 'HBM3']
SRAM_BW = [0.0, 128.0, 256.0, 512.0, 1024.0, 2048.0]
GROUPS = [1, 16, 32, 64, 128]
SCORE_SRAM_KB = 128


def base_hw(sram_bw=0.0, score_sram_kb=SCORE_SRAM_KB):
    return HardwareConfig(
        array_m=32, array_n=4, FPE_array_size=64,
        act_bits=16, accumulate_bits=32, weight_bits=4, kv_cache_bits=KV_BITS,
        AW_mode="OMNI", AA_mode="OMNI", sram_bandwidth_gbps=sram_bw,
        score_sram_kb=score_sram_kb,
    )


def measure(sim, batch=1, context=CONTEXT):
    m = get_model_config(MODEL)
    w = WorkloadConfig(batch_size=batch, input_tokens=context,
                       output_tokens=OUTPUT_TOKENS, flash_block_size=0)
    r = sim.simulate(m, w)
    aa = r.decode.get_aa_total()
    pf = r.prefill.get_total_metrics()
    dec = r.decode.get_total_metrics()
    ttft, tpot = sim.compute_roofline_latency(r, w)
    return {'aa_logical': aa.dram_read, 'aa_eff': aa.dram_read_eff,
            'decode_dram': dec.dram_read_eff + dec.dram_write_eff,
            'prefill_sram': pf.sram_read + pf.sram_write,
            'decode_sram': dec.sram_read + dec.sram_write,
            'prefill_dram': pf.dram_read_eff + pf.dram_write_eff,
            'ttft_s': ttft, 'tpot_s': tpot}


def run_tech(tech, batch=1, sram_bw=0.0, cls=None, **kw):
    hw = with_memory_technology(base_hw(sram_bw), tech)
    sim = (cls(hw, **kw) if cls is not None
           else UnstructuredKVSimulator(hw, **kw))
    return measure(sim, batch=batch)


# ============================================================================
# Pre-flight
# ============================================================================

def preflight():
    # 1. The default preset is exactly what the simulator already assumed.
    d = memory_technology(DEFAULT_TECHNOLOGY)
    stock = HardwareConfig(array_m=32, array_n=4)
    assert d.bandwidth_gbps == stock.dram_bandwidth_gbps, \
        f"{DEFAULT_TECHNOLOGY} should match the stock default bandwidth"

    # 2. Applying a technology does not mutate the config it was derived from.
    hw = base_hw()
    hbm = with_memory_technology(hw, 'HBM3')
    assert hw.dram_bandwidth_gbps == 51.2 and hw.dram_burst_bytes == 0, \
        "with_memory_technology must not mutate its argument"
    assert hbm.dram_bandwidth_gbps == 819.2 and hbm.dram_burst_bytes == 32

    # 3. sram_bandwidth_gbps = 0.0 is exactly inert against the stock model.
    a = measure(Simulator(base_hw(sram_bw=0.0)))
    b = measure(Simulator(base_hw(sram_bw=1e9)))
    assert a['ttft_s'] == b['ttft_s'] and a['tpot_s'] == b['tpot_s'], \
        "0.0 (unlimited) and an absurdly high bandwidth must agree"

    # 4. A bandwidth low enough makes everything SRAM-bound -- proves the term
    #    actually reaches the max() rather than being computed and dropped.
    slow = measure(Simulator(base_hw(sram_bw=1.0)))
    assert slow['ttft_s'] > a['ttft_s'] * 10, \
        "1 GB/s SRAM should dominate; the term is not reaching the roofline"

    # 5. The burst cliff moves with the technology: 64 contiguous channels is
    #    burst-aligned on HBM (32 B) and is not on DDR5 (64 B).
    for tech, want in (('DDR5-6400', False), ('HBM3', True)):
        dense = run_tech(tech)
        pruned = run_tech(tech, keep_channels=0.5, channel_group=64)
        kept = ((dense['aa_eff'] - pruned['aa_eff'])
                / (dense['aa_logical'] - pruned['aa_logical']))
        got = kept > 0.9
        assert got == want, \
            f"{tech} channel_group=64 saving kept {kept:.1%}, expected {want}"

    print("pre-flight: 5 checks passed")


# ============================================================================
# Sweep
# ============================================================================

def sweep(report_path):
    rows = []
    preflight()

    rep = Report(
        report_path,
        "Memory throughput",
        subtitle="What the DRAM technology decides, and what SRAM bandwidth costs",
        source="analysis/memory/bandwidth_run.py",
        setup=[f"Model: LLaMA-3-8B on Omni-LUT, 4-bit KV, context {CONTEXT:,}, "
               f"standard attention, attention scores staged on chip.",
               "DRAM presets (bandwidth *and* burst together) live in "
               "`simulator/memory_tech.py` with their derivations."])

    rep.summary([
        "**The burst matters more than the bandwidth.** HBM's 32 B access "
        "granularity halves the channel-group size at which pruning starts "
        "paying: 64 contiguous channels collects **100%** of its saving on HBM "
        "and **0%** on DDR5. The memory part, not the pruning algorithm, "
        "decides whether the saving is collectable.",
        "**More DRAM bandwidth barely helps, which is the interesting half.** "
        "16x the bandwidth (DDR5-6400 to HBM3) buys **1.10x** of decode TPOT, "
        "and the KV techniques hold their value unchanged across every "
        "technology (1.936x to 1.921x). Decode is compute-bound, so no memory "
        "part rescues it — and none devalues KV pruning either.",
        "**The SRAM bandwidth term is decode-safe and prefill-unusable.** At the "
        "geometry-implied 128 GB/s, TPOT moves 1.07x but TTFT moves **4.35x** — "
        "because prefill charges **113,670 GB** of SRAM traffic against 3 GB of "
        "DRAM. A **33,680:1** ratio is the untiled-activation defect, not a "
        "hardware result.",
        "**So the field ships inert (`sram_bandwidth_gbps = 0.0`)** and its "
        "prefill numbers stay parked until prefill tiling lands. Decode is "
        "sound by construction: `M=1` makes it tiling-inert.",
    ])

    # ---- A ------------------------------------------------------------------
    rep.section(
        "A. The presets",
        "One channel or one stack at nominal peak. Derivations are in the "
        "module so they can be checked rather than trusted.")
    trows = []
    for name in TECHS:
        t = MEMORY_TECHNOLOGIES[name]
        rows.append({'section': 'A', 'tech': name,
                     'bandwidth_gbps': t.bandwidth_gbps,
                     'burst_bytes': t.burst_bytes})
        trows.append([name, f"{t.bandwidth_gbps:,.1f} GB/s", f"{t.burst_bytes} B",
                      t.derivation])
    rep.table(["technology", "bandwidth", "burst", "derivation"], trows,
              aligns="lrrl")
    rep.note(
        f"`{DEFAULT_TECHNOLOGY}` is exactly the simulator's own default, "
        f"asserted in pre-flight rather than assumed — so every result "
        f"published before this file was a DDR5-6400 result, whether or not it "
        f"said so.")

    # ---- B ------------------------------------------------------------------
    rep.section(
        "B. Where the channel-pruning cliff sits, by technology",
        "50% of channels retained, token-major, varying how many are "
        "contiguous. 'Saving kept' is the fraction of the logical byte saving "
        "that becomes an effective one.")
    for tech in ('DDR5-6400', 'HBM3'):
        burst = MEMORY_TECHNOLOGIES[tech].burst_bytes
        d = run_tech(tech)
        trows = []
        for g in GROUPS:
            r = run_tech(tech, keep_channels=0.5, channel_group=g)
            kept = ((d['aa_eff'] - r['aa_eff'])
                    / (d['aa_logical'] - r['aa_logical']))
            rows.append({'section': 'B', 'tech': tech, 'channel_group': g,
                         'saving_kept': kept, **r})
            run_b = g * KV_BITS / 8
            trows.append([str(g),
                          f"{run_b:.1f} B" if run_b < 1 else f"{run_b:.0f} B",
                          f"{r['aa_eff'] / d['aa_eff']:.3f}x", f"{kept:.1%}"])
        rep.table(["channel group", "run", "attn DRAM vs dense", "saving kept"],
                  trows, caption=f"{tech} — {burst} B burst")
    rep.note(
        "**The cliff is at one burst, so halving the burst halves the required "
        "group.** A channel mask has to assemble at least one whole burst of "
        "contiguous retained data before it collects anything, and on HBM that "
        "is 64 channels rather than 128. This is the one lever that makes "
        "sub-entry channel pruning viable at all — and it is bought in the "
        "memory subsystem, not in the pruning algorithm.")

    # ---- C ------------------------------------------------------------------
    rep.section(
        "C. What bandwidth does to the KV techniques",
        "Decode TPOT speedup over dense, at batch 32 where KV traffic "
        "dominates. Each technique is measured against the dense baseline *on "
        "its own technology*.")
    trows = []
    ddr5_dense = run_tech(DEFAULT_TECHNOLOGY, batch=32)['tpot_s']
    for tech in TECHS:
        d = run_tech(tech, batch=32)
        cells = [tech]
        for label, kw in (('token 50%', dict(keep_tokens=0.5, token_group=1)),
                          ('token 90%', dict(keep_tokens=0.1, token_group=1))):
            r = run_tech(tech, batch=32, **kw)
            sp = d['tpot_s'] / r['tpot_s']
            rows.append({'section': 'C', 'tech': tech, 'technique': label,
                         'speedup': sp, **r})
            cells.append(f"{sp:.3f}x")
        cells.append(f"{d['tpot_s'] * 1e3:,.0f} ms")
        cells.append(f"{ddr5_dense / d['tpot_s']:.3f}x")
        trows.append(cells)
    rep.table(["technology", "prune 50% tokens", "prune 90% tokens",
               "dense TPOT", "dense vs DDR5"], trows, aligns="lrrrr")
    rep.note(
        "**16x the bandwidth buys 1.10x of decode.** This is the negative "
        "result stated in a new place: `study.md` §13 showed KV *bytes* are "
        "usually not the critical path because `attn_v` is compute-bound, and "
        "the direct consequence is that the memory technology cannot rescue "
        "decode. HBM2E and HBM3 are indistinguishable here — once the DRAM roof "
        "clears the compute roof, further bandwidth is inert.")
    rep.note(
        "**The corollary is the useful one, and it is the opposite of what one "
        "would expect.** A KV technique is usually said to be worth most where "
        "bandwidth is scarcest, so moving to HBM should devalue it. It does "
        "not: 1.936x on DDR5 and 1.921x on HBM3. Token pruning cuts `kv_len`, "
        "which is the `K` of both attention GEMMs, so it removes **cycles** as "
        "well as bytes — and cycles are what decode is actually limited by. "
        "**Its value is portable across memory technologies precisely because "
        "it was never really a bandwidth optimisation.**")

    # ---- D ------------------------------------------------------------------
    rep.section(
        "D. SRAM bandwidth — the term, and why it ships off",
        "DDR5-6400, batch 1. `unlimited` is the shipped default and reproduces "
        "every existing number exactly.")
    trows = []
    base = None
    for bw in SRAM_BW:
        r = measure(Simulator(base_hw(sram_bw=bw)))
        if base is None:
            base = r
        rows.append({'section': 'D', 'sram_bw': bw, **r})
        label = 'unlimited' if bw == 0 else f"{bw:,.0f} GB/s"
        trows.append([label, f"{r['ttft_s']:,.1f} s",
                      f"{r['ttft_s'] / base['ttft_s']:.2f}x",
                      f"{r['tpot_s'] * 1e3:.2f} ms",
                      f"{r['tpot_s'] / base['tpot_s']:.3f}x"])
    rep.table(["SRAM bandwidth", "TTFT", "vs unlimited", "TPOT",
               "vs unlimited"], trows, aligns="lrrrr")

    ref = measure(Simulator(base_hw()))
    rep.table(
        ["phase", "SRAM traffic", "DRAM traffic", "ratio"],
        [["prefill", f"{ref['prefill_sram'] / 2**30:,.0f} GB",
          f"{ref['prefill_dram'] / 2**30:,.0f} GB",
          f"{ref['prefill_sram'] / max(1, ref['prefill_dram']):,.0f} : 1"],
         ["decode", f"{ref['decode_sram'] / 2**30:,.1f} GB",
          f"{ref['decode_dram'] / 2**30:,.1f} GB",
          f"{ref['decode_sram'] / max(1, ref['decode_dram']):,.2f} : 1"]],
        aligns="lrrr", caption="why the two phases disagree")
    rep.note(
        "**Read the two tables together.** The 128 GB/s row is not a finding "
        "about bandwidth; it is the untiled-activation defect becoming visible "
        "through a new term. Prefill's SRAM traffic is written against an "
        "untiled A, so it counts re-reads a real loop nest never performs — the "
        "same defect that makes `_calculate_peak_sram` claim a 2.1 GB prefill "
        "working set (`study.md` §7). **Tens of thousands to one against DRAM "
        "is not physical.** Note the DRAM side is small here precisely because "
        "the scores are correctly staged (Stage 5), which makes the SRAM side's "
        "implausibility all the more visible.")
    rep.note(
        "**Decode is sound and can be believed now.** `M=1` makes decode "
        "tiling-inert, its SRAM:DRAM ratio is a plausible ~2:1, and TPOT moves "
        "only 1.07x even at 128 GB/s. **Decode is not SRAM-throughput-limited** "
        "— a real result, not an artefact, and one that settles the open "
        "question §9 left hanging.")
    rep.note(
        "**And 128 GB/s is itself an over-charge.** It is one *operand port*, "
        "while `sram_read` is a lump of A-reads, B-reads and C accumulator "
        "traffic. Billing the lump against one port's number over-states the "
        "constraint even where the traffic is right. Per-port accounting is the "
        "honest follow-up; a single lumped number is not it.")

    # ---- E ------------------------------------------------------------------
    rep.section(
        "E. The question §9 actually needed: can P=8 packing be fed?",
        "OS-V packing keeps P rows live instead of 1, and each row pulls its "
        "own KV stream. This is the KV port specifically, computed from the "
        "array geometry rather than from the lumped `sram_read`.")
    h = base_hw()
    per_cycle = Simulator.MU * h.array_n * Simulator.NUM_RAC * h.kv_cache_bits // 8
    ghz = h.freq_mhz * 1e6
    trows = []
    for p_ in (1, 4, 8, 16, 32):
        bw = per_cycle * p_ * ghz
        rows.append({'section': 'E', 'pack': p_, 'kv_port_gbps': bw / 1e9})
        verdict = ('one port' if p_ == 1 else
                   'plausible (banked SRAM)' if bw / 1e12 <= 1.5 else
                   'needs a redesign')
        trows.append([f"P={p_}", f"{per_cycle * p_:,} B/cycle",
                      f"{bw / 1e12:.2f} TB/s", verdict])
    rep.table(["packing", "KV bytes/cycle", "required KV-port bandwidth",
               "verdict"], trows, aligns="lrrl")
    rep.note(
        "**P=8 — the point where §9 showed TPOT saturates — needs about 1.0 "
        "TB/s of KV-SRAM reads.** That is roughly 2,048 B/cycle at 500 MHz: "
        "large, but a banked or multi-ported KV SRAM reaches it, and it is the "
        "same order as the register-file bandwidth such arrays already carry. "
        "**The P=8 result survives its own bandwidth check; the P=32 one does "
        "not**, which matters little because §9 showed P=32 buys nothing over "
        "P=8 anyway. The two independent arguments for stopping at P=8 agree.")
    rep.note(
        "This is computed from geometry, not from `sram_read`, deliberately — "
        "the lumped figure is exactly the one section D just showed cannot be "
        "trusted for prefill. A port-level number is answerable; the lump is "
        "not.")

    rep.save()
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv', default=os.path.join(_here, 'bandwidth.csv'))
    p.add_argument('--report',
                   default=os.path.join(_here, 'bandwidth_report.md'))
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
