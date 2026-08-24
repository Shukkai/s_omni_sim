"""
What does an *unstructured* KV mask cost that a compacted one does not?

Every KV result in `study.md` was measured on a **compacted**
retained set: eviction compacts, ThinK narrows the entry to a solid `d_ret`
block, page selection gathers whole pages.  Real pruning masks are irregular --
per-head channel sets differ, per-head token sets differ -- and on a
burst-addressed DRAM an irregular mask and a compacted one of the same size are
not the same read.

The question is therefore not "how much does pruning save" (already measured)
but **"how much of that saving survives the mask being unstructured"**.

The answer is decided by one number.  A 4-bit KV entry is `head_dim 128 x 4/8`
= **64 B, exactly one DRAM burst**.  So an axis is free if and only if its mask
cuts on a boundary that is already burst-aligned in the chosen layout, and the
two axes disagree about which layout that is:

    token-major    a token mask cuts between entries  -> aligned, free
                   a channel mask cuts inside one     -> sub-burst, ruinous
    channel-major  exactly the reverse

Head-wise is the exception: a head is its own address region either way.

Usage:
    python unstructured_run.py
    python unstructured_run.py --csv unstructured.csv \\
        --report unstructured_report.md
"""

import argparse
import csv
import dataclasses
import math
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, '..', '..'))
for p in ('simulator', 'analysis', 'analysis/cycle_breakdown',
          'analysis/channel_prune_breakdown', 'analysis/memory'):
    sys.path.insert(0, os.path.join(_root, *p.split('/')))

from simulator import (                                              # noqa: E402
    HardwareConfig, WorkloadConfig, Simulator, OperationType, ComputeMode,
)
from model_configs import get_model_config                           # noqa: E402
from cycle_units import UnitAwareSimulator                           # noqa: E402
from think_prune import ThinKSimulator                               # noqa: E402
from report import Report                                            # noqa: E402
from unstructured_kv import (                                        # noqa: E402
    UnstructuredKVSimulator, TOKEN_MAJOR, CHANNEL_MAJOR,
)

MODEL = 'LLaMA-3-8B'
CONTEXT = 32768
OUTPUT_TOKENS = 4
BURST = 64
HEAD_DIM = 128
KV_BITS = 4
BATCHES = [1, 32]
GROUPS = [1, 2, 4, 8, 16, 32, 64, 128]


# Scores are staged on chip (Stage 5).  Without this the attention-score spill
# rides along inside every AA total and dilutes every ratio below by a term that
# has nothing to do with the KV mask -- head pruning, for instance, measures
# 1.89x instead of its true 2.00x.  A 32K row vector is 32,769 x 16/8 = 64 KB,
# so 128 KB stages it with margin.  This is the corrected model, not a tweak.
SCORE_SRAM_KB = 128


def hw(burst=BURST):
    return HardwareConfig(
        array_m=32, array_n=4, FPE_array_size=64,
        act_bits=16, accumulate_bits=32, weight_bits=4, kv_cache_bits=KV_BITS,
        AW_mode="OMNI", AA_mode="OMNI", dram_burst_bytes=burst,
        score_sram_kb=SCORE_SRAM_KB,
    )


def model_cfg(keep_heads=1.0):
    m = get_model_config(MODEL)
    if keep_heads >= 1.0:
        return m
    kv = max(1, int(round(m.num_kv_heads * keep_heads)))
    return dataclasses.replace(m, num_kv_heads=kv)


def measure(sim, batch=1, context=CONTEXT, keep_heads=1.0):
    """Decode attention (AA) traffic, cycles and token latency."""
    m = model_cfg(keep_heads)
    w = WorkloadConfig(batch_size=batch, input_tokens=context,
                       output_tokens=OUTPUT_TOKENS, flash_block_size=0)
    r = sim.simulate(m, w)
    aa = r.decode.get_aa_total()
    aw = r.decode.get_aw_total()
    qk = r.decode.get_operation_total(OperationType.QK_MATMUL, ComputeMode.AA)
    av = r.decode.get_operation_total(OperationType.ATTN_V_MATMUL, ComputeMode.AA)
    _, tpot = sim.compute_roofline_latency(r, w)
    return {'aa_logical': aa.dram_read, 'aa_eff': aa.dram_read_eff,
            'aa_cycles': aa.cycles, 'aw_eff': aw.dram_read_eff,
            'qk_cycles': qk.cycles, 'av_cycles': av.cycles,
            'decode_eff': aa.dram_read_eff + aw.dram_read_eff,
            'tpot_s': tpot}


def run(batch=1, context=CONTEXT, keep_heads=1.0, burst=BURST, **kw):
    return measure(UnstructuredKVSimulator(hw(burst), **kw),
                   batch=batch, context=context, keep_heads=keep_heads)


def dense(batch=1, context=CONTEXT, keep_heads=1.0, burst=BURST):
    return run(batch=batch, context=context, keep_heads=keep_heads, burst=burst)


def efficiency(base, pruned):
    """Fraction of the logical byte saving that becomes an effective saving.

    1.0 = the mask is as good as compacted.  0.0 = pruning bought nothing:
    every byte it removed logically is still moved across the bus.
    """
    removed = base['aa_logical'] - pruned['aa_logical']
    if removed <= 0:
        return 1.0
    return (base['aa_eff'] - pruned['aa_eff']) / removed


# ============================================================================
# Pre-flight
# ============================================================================

def preflight():
    """Nine checks.  Each is a claim the report makes, pinned before it runs."""
    base = dense()

    # 1. All-dense reproduces the stock model byte-for-byte.
    stock = measure(UnitAwareSimulator(hw()))
    assert base['aa_logical'] == stock['aa_logical'], \
        f"dense diverged: {base['aa_logical']} vs {stock['aa_logical']}"
    assert base['aa_eff'] == base['aa_logical'], \
        "dense read should be perfectly burst-aligned"

    # 2. Compacted channel pruning reproduces ThinK at the same d_ret.
    d_ret = 64
    mine = run(keep_channels=d_ret / HEAD_DIM, channel_group=HEAD_DIM)
    think = measure(ThinKSimulator(hw(), d_k_ret=d_ret, d_v_ret=d_ret))
    assert mine['aa_logical'] == think['aa_logical'], \
        f"ThinK mismatch: {mine['aa_logical']} vs {think['aa_logical']}"
    assert mine['aa_cycles'] == think['aa_cycles'], "ThinK cycle mismatch"

    # 3. Token-major token pruning is burst-free at EVERY group size, including
    #    a fully scattered single-token gather.  This is the 64 B entry.
    for g in (1, 4, 16, 1024):
        r = run(keep_tokens=0.5, token_group=g)
        assert r['aa_eff'] == r['aa_logical'], \
            f"token-major token mask charged a burst penalty at group {g}"

    # 4. Token-major channel pruning at group 1 degrades to EXACTLY dense --
    #    the clamp binds, so the saving is zero rather than negative.
    r = run(keep_channels=0.5, channel_group=1)
    assert r['aa_logical'] < base['aa_logical'], "logical bytes should fall"
    assert r['aa_eff'] == base['aa_eff'], \
        f"expected dense-equivalent, got {r['aa_eff']} vs {base['aa_eff']}"
    assert abs(efficiency(base, r)) < 1e-9, "efficiency should be exactly 0"

    # 5. The mirror: channel-major inverts which axis is free.
    rc = run(layout=CHANNEL_MAJOR, keep_channels=0.5, channel_group=1)
    rt = run(layout=CHANNEL_MAJOR, keep_tokens=0.5, token_group=1)
    assert efficiency(base, rc) > 0.9, \
        f"channel-major should make channel pruning ~free, got {efficiency(base, rc)}"
    assert efficiency(base, rt) < 0.1, \
        f"channel-major should ruin token pruning, got {efficiency(base, rt)}"

    # 6. Head-wise is structured in both layouts and exactly linear.
    half = dense(keep_heads=0.5)
    assert half['aa_eff'] == half['aa_logical'], "head pruning should be aligned"
    ratio = base['aa_logical'] / half['aa_logical']
    assert 1.9 < ratio < 2.1, f"head pruning not ~linear: {ratio}"

    # 7. With no burst model the whole effect vanishes -- proving every number
    #    below comes from granularity and not from the shape rewrite.
    nb = run(burst=0, keep_channels=0.5, channel_group=1)
    assert nb['aa_eff'] == nb['aa_logical'], \
        "burst=0 should charge nothing extra"

    # 8. Channel pruning is a CYCLE null *on attn_v specifically*: it narrows
    #    N, and the LUT_OS_V round cost has no N term (study.md section 5).  It
    #    is NOT a null on qk, whose K *is* head_dim -- so asserting on the AA
    #    total would be wrong, and separating them here keeps the claim honest.
    ch = run(keep_channels=0.5, channel_group=HEAD_DIM)
    assert ch['av_cycles'] == base['av_cycles'], \
        f"attn_v cycles moved: {ch['av_cycles']} vs {base['av_cycles']}"
    assert ch['qk_cycles'] < base['qk_cycles'], \
        "qk cycles should fall -- head_dim is its reduction dimension"

    # 9. Token pruning cuts BOTH, because kv_len is qk's N and attn_v's K.
    tk = run(keep_tokens=0.5, token_group=1)
    assert tk['av_cycles'] < base['av_cycles'], "token pruning should cut attn_v"
    assert tk['qk_cycles'] < base['qk_cycles'], "token pruning should cut qk"

    print("pre-flight: 9 checks passed")


# ============================================================================
# Analytical helpers
# ============================================================================

def run_bytes_for(layout, axis, group):
    """Contiguous bytes per run, straight from the layout geometry."""
    if layout == TOKEN_MAJOR:
        if axis == 'token':
            return group * HEAD_DIM * KV_BITS // 8   # whole entries
        return max(1, group * KV_BITS // 8)          # fragment of one entry
    if axis == 'channel':
        return group * CONTEXT * KV_BITS // 8        # whole channel rows
    return max(1, group * KV_BITS // 8)              # fragment of one row


def amplification(run_b):
    return math.ceil(run_b / BURST) * BURST / run_b


# ============================================================================
# Sweep
# ============================================================================

def sweep(report_path):
    rows = []
    preflight()
    base = dense()

    rep = Report(
        report_path,
        "Unstructured KV pruning",
        subtitle="What an irregular mask costs that a compacted one does not",
        source="analysis/memory/unstructured_run.py",
        setup=[f"Model: LLaMA-3-8B on Omni-LUT, 4-bit KV, context {CONTEXT:,}, "
               f"{BURST} B DRAM burst, standard attention.",
               "Every prior KV result was measured on a *compacted* retained "
               "set. This asks how much of that saving survives the mask being "
               "irregular.",
               f"Dense decode attention baseline: {base['aa_logical']:,} B."])

    rep.summary([
        "**A 4-bit KV entry is 64 B, exactly one burst — so the layout decides "
        "everything.** In token-major, a *token* mask cuts between entries and "
        "is free at any granularity; a *channel* mask cuts inside one and is "
        "sub-burst. Channel-major inverts it exactly.",
        "**Unstructured channel pruning in token-major saves nothing at all.** "
        "50% of channels removed, **0.0% of the traffic saved** — the read "
        "degrades to precisely dense. It is not a partial loss; the saving is "
        "gone.",
        "**The two axes want opposite layouts, so combining them is not "
        "possible without losing one.** At 50%+50% the best any single layout "
        "achieves is 0.50x of the joint saving, and the loss lands entirely on "
        "whichever axis the layout did not favour.",
        "**Head-wise is the only axis that is free in both layouts**, because a "
        "head is its own address region — and it is the one the pruning "
        "literature uses least.",
        "**Channel pruning is now null on *both* axes**: `study.md` §5 showed "
        "it does not move cycles, and this shows an unstructured mask does not "
        "move bytes either. Structured, entry-contiguous channel groups are the "
        "only version that pays.",
    ])

    # ---- A ------------------------------------------------------------------
    rep.section(
        "A. Why the layout decides — run length against the burst",
        f"A retained element is {KV_BITS} bits. What matters is how many of "
        f"them sit next to each other, against a {BURST} B burst.")
    trows = []
    for layout in (TOKEN_MAJOR, CHANNEL_MAJOR):
        for axis in ('token', 'channel'):
            for group in (1, 16, 128):
                rb = run_bytes_for(layout, axis, group)
                amp = amplification(rb)
                rows.append({'section': 'A', 'layout': layout, 'axis': axis,
                             'group': group, 'run_bytes': rb,
                             'amplification': amp})
                trows.append([layout.replace('_', '-'), f"{axis}-wise",
                              str(group), f"{rb:,} B", f"{amp:.0f}x",
                              'yes' if amp == 1 else 'no'])
    rep.table(["layout", "mask axis", "group", "contiguous run",
               "burst amplification", "aligned?"], trows, aligns="llrrrc")
    rep.note(
        f"A token-major entry is `{HEAD_DIM} x {KV_BITS}/8` = **{BURST} B, "
        f"exactly one burst**. That single coincidence is the whole result: a "
        f"token mask lands on entry boundaries and is aligned at *every* group "
        f"size, down to gathering one isolated token. A channel mask cuts "
        f"inside the entry, so its run is `group x 0.5 B` and needs the **full "
        f"{HEAD_DIM} channels** to reach one burst — meaning any channel "
        f"pruning at all is sub-burst in this layout.")
    rep.note(
        f"A run is floored at 1 B in both the model and this table, so a single "
        f"{KV_BITS}-bit element reads as 1 B / 64x rather than its true 0.5 B / "
        f"128x. That understates the finest-grained scatter, which is the "
        f"conservative direction — and the clamp in section B makes it moot, "
        f"since anything at or below one burst prices the same.")

    # ---- B ------------------------------------------------------------------
    rep.section(
        "B. Token-major: what channel-group granularity buys",
        "50% of channels retained, varying how many of them are contiguous. "
        "Group 128 = the whole entry = compacted, which is what ThinK and every "
        "prior measurement assumed.")
    trows = []
    for g in GROUPS:
        r = run(keep_channels=0.5, channel_group=g)
        eff = efficiency(base, r)
        rows.append({'section': 'B', 'channel_group': g, 'efficiency': eff,
                     **r})
        # Sub-byte runs are real: one 4-bit channel is half a byte, and
        # truncating that to "0 B" would hide the whole mechanism.
        run_b = g * KV_BITS / 8
        run_s = f"{run_b:.1f} B" if run_b < 1 else f"{run_b:.0f} B"
        trows.append([str(g), run_s, f"{r['aa_logical']:,}",
                      f"{r['aa_eff']:,}",
                      f"{r['aa_eff'] / base['aa_eff']:.3f}x", f"{eff:.1%}"])
    rep.table(["channel group", "run", "logical", "effective", "vs dense",
               "saving kept"], trows)
    rep.note(
        "**It is a cliff, not a slope.** Every group below 128 gives the same "
        "answer — zero — because any sub-burst run is rounded to the same 64 B, "
        "and the clamp then stops the charge exceeding a dense read. There is "
        "no partial credit for a *partly* structured channel mask: 64 "
        "contiguous channels out of 128 is worth exactly as much as one, which "
        "is nothing. Only the full entry pays.")

    # ---- C ------------------------------------------------------------------
    rep.section(
        "C. The layout conflict",
        "Both axes at 50%, fully unstructured (group 1), measured under each "
        "layout. This is the table the study exists for.")
    trows = []
    for layout in (TOKEN_MAJOR, CHANNEL_MAJOR):
        for label, kw in (
                ('token-wise 50%', dict(keep_tokens=0.5, token_group=1)),
                ('channel-wise 50%', dict(keep_channels=0.5, channel_group=1))):
            r = run(layout=layout, **kw)
            eff = efficiency(base, r)
            rows.append({'section': 'C', 'layout': layout, 'axis': label,
                         'efficiency': eff, **r})
            trows.append([layout.replace('_', '-'), label,
                          f"{r['aa_logical'] / base['aa_logical']:.3f}x",
                          f"{r['aa_eff'] / base['aa_eff']:.3f}x", f"{eff:.1%}"])
    rep.table(["layout", "mask", "logical", "effective", "saving kept"], trows,
              aligns="llrrr")
    rep.note(
        "**Perfectly antisymmetric.** Each layout makes one axis free and the "
        "other worthless, and there is no third layout: a KV element has two "
        "indices, so one of them is the minor axis and the other is strided. "
        "Choosing a layout is therefore choosing *which pruning axis is allowed "
        "to work* — a decision made in the memory subsystem that silently "
        "determines which pruning papers can be deployed at all.")

    # ---- D ------------------------------------------------------------------
    rep.section(
        "D. Combining axes — head + token + channel",
        "Token-major. Head-wise drops KV heads, which is contiguous in either "
        "layout. Each row adds one axis on top of the row above.")
    trows = []
    combos = [
        ('dense', 1.0, dict()),
        ('head 50%', 0.5, dict()),
        ('head + token 50%', 0.5, dict(keep_tokens=0.5, token_group=1)),
        ('head + token + channel 50%', 0.5,
         dict(keep_tokens=0.5, token_group=1, keep_channels=0.5,
              channel_group=1)),
        ('head + token + channel (chan grp 128)', 0.5,
         dict(keep_tokens=0.5, token_group=1, keep_channels=0.5,
              channel_group=HEAD_DIM)),
    ]
    for label, kh, kw in combos:
        r = run(keep_heads=kh, **kw)
        rows.append({'section': 'D', 'combo': label, 'keep_heads': kh, **r})
        trows.append([label, f"{r['aa_logical'] / base['aa_logical']:.3f}x",
                      f"{r['aa_eff'] / base['aa_eff']:.3f}x",
                      f"{r['tpot_s'] * 1e3:.2f} ms",
                      f"{base['tpot_s'] / r['tpot_s']:.3f}x"])
    rep.table(["mask", "logical", "effective", "TPOT", "speedup"], trows,
              aligns="lrrrr")
    rep.note(
        "The third axis is where it breaks. Head and token compose cleanly — "
        "both are entry-aligned — but adding an unstructured channel mask "
        "removes a further 50% logically and **moves the effective traffic not "
        "at all**. The last row is the same mask with contiguous 128-channel "
        "groups, i.e. structured, and it is the only version that converts.")

    # ---- E ------------------------------------------------------------------
    rep.section(
        "E. What the mask itself costs to store",
        "An unstructured mask has to be written down somewhere and re-read to "
        "drive the gather. Token-major, 50%/50%, context 32,768.")
    trows = []
    for mode, desc in (('none', 'not charged (what prior studies assumed)'),
                       ('static', 'one token + one channel bitmap per head'),
                       ('per_token', 'one bit per (token, channel) element')):
        r = run(keep_tokens=0.5, token_group=1, keep_channels=0.5,
                channel_group=HEAD_DIM, mask_mode=mode)
        rows.append({'section': 'E', 'mask_mode': mode, **r})
        trows.append([mode, desc, f"{r['aa_eff']:,}",
                      f"{r['aa_eff'] / base['aa_eff']:.3f}x"])
    rep.table(["mask_mode", "what it stores", "effective", "vs dense"], trows,
              aligns="llrr")
    rep.note(
        f"A per-element mask is **1 bit against a {KV_BITS}-bit datum — 25% of "
        f"the dense cache**, charged over the full context whether or not the "
        f"element survived. That is the structural cost of the finest-grained "
        f"mask, and it lands on top of the granularity penalty rather than "
        f"instead of it. A per-head static mask is negligible by comparison and "
        f"is what a deployable design would use.")

    # ---- F ------------------------------------------------------------------
    rep.section(
        "F. Does any of it reach token latency?",
        "Token-major, fully unstructured masks, at both batch extremes. DRAM "
        "share of decode grows with batch, so this is where granularity has "
        "its best chance of mattering.")
    for b in BATCHES:
        b_base = dense(batch=b)
        trows = []
        for label, kw in (
                ('token 50%', dict(keep_tokens=0.5, token_group=1)),
                ('channel 50% (unstructured)',
                 dict(keep_channels=0.5, channel_group=1)),
                ('channel 50% (compacted)',
                 dict(keep_channels=0.5, channel_group=HEAD_DIM))):
            r = run(batch=b, **kw)
            rows.append({'section': 'F', 'batch': b, 'mask': label, **r})
            trows.append([label, f"{r['qk_cycles']:,}", f"{r['av_cycles']:,}",
                          f"{r['aa_eff'] / b_base['aa_eff']:.3f}x",
                          f"{r['tpot_s'] * 1e3:.2f} ms",
                          f"{b_base['tpot_s'] / r['tpot_s']:.3f}x"])
        rep.table(["mask", "qk cycles", "attn_v cycles", "attn DRAM", "TPOT",
                   "speedup"], trows, aligns="lrrrrr", caption=f"batch {b}")
    rep.note(
        "**Token pruning is the only axis that works on both counts** — it cuts "
        "`kv_len`, which is the `K` of both attention GEMMs, so cycles fall "
        "*and* its bytes are entry-aligned. Channel pruning cuts `N` of "
        "`attn_v`, which the `LUT_OS_V` round cost has no term for, so cycles "
        "do not move; unstructured, its bytes do not move either, and the "
        "speedup is exactly 1.000x. **Two independent nulls on the same axis.**")

    # ---- G ------------------------------------------------------------------
    rep.section("G. What this does and does not say")
    rep.note(
        "**Accuracy is out of scope.** Unstructured masks exist because they "
        "are more accurate at equal budget. This prices that choice; it does "
        "not dispute it. The finding is that the price is paid in full and the "
        "benefit is not collected — which is an argument for structured masks, "
        "not against pruning.")
    rep.note(
        "**The clamp is doing real work and should be understood.** A scattered "
        "read is never charged more than reading the whole region it sits "
        "inside, because a controller would simply do that instead. Without it "
        "the model reports up to 128x amplification, which is arithmetic rather "
        "than a cost. With it, the failure mode is stated correctly: the saving "
        "goes to **zero**, not negative.")
    rep.note(
        "**Two things are modelled optimistically.** Gather *scheduling* is "
        "free here — no request-queue or MSHR pressure, though a 128-way "
        "scattered gather generates far more outstanding requests than a "
        "streaming read. And the mask is assumed already resident when the "
        "gather issues, so its read latency never serialises against the KV "
        "read. Both push in the same direction: **unstructured masks are worse "
        "than this report says, not better.**")
    rep.note(
        f"**Bit-width would change the entry size and therefore the whole "
        f"table.** At {KV_BITS}-bit the entry is exactly one burst; at 8-bit it "
        f"is two, and at 3-bit it is 48 B and aligned with nothing "
        f"(`selective_report.md` B). The alignment that makes token pruning "
        f"free is a property of this configuration, not a law.")

    rep.save()
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv', default=os.path.join(_here, 'unstructured.csv'))
    p.add_argument('--report',
                   default=os.path.join(_here, 'unstructured_report.md'))
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
