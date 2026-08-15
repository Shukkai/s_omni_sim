"""
The attention score matrix: the largest DRAM term in the model, and an assumption.

`_calculate_memory_access` charged QK to write the full score matrix to DRAM and
Attn.V to read it back, unconditionally, in both phases, regardless of capacity.
That is 99.9% of prefill DRAM at 32K context -- and it is a hardcoded modelling
choice, not a derived result.  It was also internally contradictory in two ways:

  * prefill attention read ZERO K/V from DRAM (the KV branch is gated on
    `is_decode`, the weights branch is an `elif`, so a prefill QK reached
    neither), so the model asserted K/V were resident while asserting the far
    larger score matrix was not;
  * the softmax between QK and Attn.V is a non-GEMM op and `NonGEMMMetrics` has
    no DRAM fields, so the same matrix was DRAM-resident for the matmuls and
    SRAM-resident for the softmax, in the same layer.

`score_sram_kb` stages the scores on chip when a query row's score vector fits;
`prefill_kv_dram_read` charges prefill for the K/V it reads.  Both default off,
so nothing published moves unless asked.

**It is not prefill-only, but decode is a much smaller exposure.**  The score
traffic must be isolated by differencing staged against unstaged: reading
`attn_v`'s `dram_read` directly overstates it ~5x, because that field also
carries the V-cache read.  Differenced properly, the decode score spill is
11.1% of attention DRAM at every context, and 1.2-10.4% of all decode DRAM
depending on batch.  Section D measures it, because `study2.md` sections 7-9
were produced with the spill on -- but the resulting TPOT shift is 1.005x-1.016x,
so those sections are corrected, not overturned.

Usage:
    python prefill_run.py
    python prefill_run.py --csv prefill.csv --report prefill_report.md
"""

import argparse
import csv
import math
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, '..', '..'))
for p in ('simulator', 'analysis', 'analysis/cycle_breakdown',
          'analysis/compact_breakdown'):
    sys.path.insert(0, os.path.join(_root, *p.split('/')))

from simulator import (                                          # noqa: E402
    ComputeMode, HardwareConfig, OperationType, Simulator, WorkloadConfig,
)
from model_configs import get_model_config                       # noqa: E402
from report import Report                                       # noqa: E402

MODEL = 'LLaMA-3-8B'
ACT_BITS = 16
HEAD_DIM = 128
NUM_HEADS, NUM_KV_HEADS = 32, 8
CONTEXTS = [2048, 8192, 32768]
BATCHES = [1, 8, 32]


def hw(score_kb=0, prefill_kv=False, kv_sram_kb=0):
    return HardwareConfig(
        array_m=32, array_n=4, FPE_array_size=64,
        act_bits=ACT_BITS, accumulate_bits=32, weight_bits=4, kv_cache_bits=4,
        AW_mode="OMNI", AA_mode="OMNI",
        score_sram_kb=score_kb, prefill_kv_dram_read=prefill_kv,
        kv_sram_kb=kv_sram_kb,
    )


def run(context, batch=1, score_kb=0, prefill_kv=False, out_tokens=3,
        flash=0):
    model = get_model_config(MODEL)
    w = WorkloadConfig(batch_size=batch, input_tokens=context,
                       output_tokens=out_tokens, flash_block_size=flash)
    s = Simulator(hw(score_kb, prefill_kv))
    r = s.simulate(model, w)
    pre, dec = r.prefill.get_total_metrics(), r.decode.get_total_metrics()
    ttft, tpot = s.compute_roofline_latency(r, w)
    out = {'context': context, 'batch': batch, 'score_sram_kb': score_kb,
           'prefill_kv_dram_read': prefill_kv, 'flash': flash,
           'prefill_dram': pre.dram_read + pre.dram_write,
           'prefill_dram_read': pre.dram_read, 'prefill_dram_write': pre.dram_write,
           'decode_dram': dec.dram_read + dec.dram_write,
           'ttft_s': ttft, 'tpot_s': tpot}
    if flash == 0:
        for tag, op in (('qk', OperationType.QK_MATMUL),
                        ('av', OperationType.ATTN_V_MATMUL)):
            m = r.decode.get_operation_total(op, ComputeMode.AA)
            out[f'dec_{tag}_r'] = m.dram_read
            out[f'dec_{tag}_w'] = m.dram_write
    return out


def score_row_kb(kv_len):
    """KB needed to stage one query row's score vector for one instance."""
    return math.ceil(kv_len * ACT_BITS / 8 / 1024)


def buf_kb(context, out_tokens=3):
    """Buffer that covers the LONGEST score row the run will produce.

    Decode grows `kv_len` to `context + token_idx`, so a buffer sized for
    prefill's `kv_len = context` is short by a row or two once decoding starts
    and the scores silently spill again.  Sizing for the longest row is what a
    real design has to do, and it is the only size at which the field is
    coherent across both phases.
    """
    return score_row_kb(context + out_tokens)


# ---------------------------------------------------------------- pre-flight

def preflight():
    n = 0
    ctx = 8192

    # 1. Both fields at their defaults reproduce the pre-Stage-5 model.  The
    #    constant is the pre-change value for this config, and that it IS the
    #    pre-change value is established independently by the regression gate,
    #    which reports zero changed values at these defaults.
    base = run(ctx, 1)
    assert base['prefill_dram'] == 277_696_479_232, base['prefill_dram']
    n += 1

    # 2. Boundary. One row is kv_len*act/8 bytes; one KB below must still spill
    #    and one KB above must fully stage.  Off-by-one here is the likeliest bug.
    need = buf_kb(ctx)
    row = score_row_kb(ctx)      # prefill's own row, for the boundary test
    below, above = run(ctx, 1, score_kb=row - 1), run(ctx, 1, score_kb=row)
    assert below['prefill_dram'] == base['prefill_dram'], "below threshold leaked"
    assert above['prefill_dram'] < base['prefill_dram'], "at threshold did not stage"
    n += 1

    # 3. Fully staged prefill writes exactly the KV cache and nothing else.
    model = get_model_config(MODEL)
    kv_writeback = 2 * 1 * ctx * (NUM_KV_HEADS * HEAD_DIM) * 4 // 8 \
        * model.num_layers
    assert above['prefill_dram_write'] == kv_writeback, \
        (above['prefill_dram_write'], kv_writeback)
    n += 1

    # 4. Staging is exactly inert on AW ops -- the predicate must not leak
    #    outside attention.
    w = WorkloadConfig(batch_size=1, input_tokens=ctx, output_tokens=3,
                       flash_block_size=0)
    aw0 = Simulator(hw()).simulate(model, w).prefill.get_aw_total()
    aw1 = Simulator(hw(need, True)).simulate(model, w).prefill.get_aw_total()
    assert aw0.dram_read == aw1.dram_read and aw0.dram_write == aw1.dram_write
    n += 1

    # 5. Convergence with FlashAttention.  Fully staged + prefill K/V read must
    #    EXACTLY equal the flash path at one Q block (block = seq), since both
    #    then reduce to Q/K/V read once + KV writeback.
    std = run(ctx, 1, score_kb=need, prefill_kv=True)
    flash1 = run(ctx, 1, score_kb=need, prefill_kv=True, flash=ctx)
    assert std['prefill_dram'] == flash1['prefill_dram'], \
        (std['prefill_dram'], flash1['prefill_dram'])
    # And a small block must cost MORE, by the num_q_blocks re-read factor.
    flash256 = run(ctx, 1, score_kb=need, prefill_kv=True, flash=256)
    assert flash256['prefill_dram'] > flash1['prefill_dram'], "small block not dearer"
    n += 1

    # 6. score_sram_kb>0 with prefill_kv_dram_read=False is a NON-configuration,
    #    and this pins the incoherence precisely rather than asserting it away:
    #    prefill ATTENTION reads exactly nothing from DRAM, while K_PROJ/V_PROJ
    #    in the same run wrote a full KV cache to it.  Data is produced, sent to
    #    DRAM, and then consumed from nowhere.  (Total prefill reads are not the
    #    right probe -- weights dominate them.)
    wl = WorkloadConfig(batch_size=1, input_tokens=ctx, output_tokens=3,
                        flash_block_size=0)
    inc = Simulator(hw(need, False)).simulate(model, wl)
    assert inc.prefill.get_aa_total().dram_read == 0, "expected zero AA read"
    assert inc.prefill.get_aw_total().dram_write == kv_writeback, "no KV written"
    # And enabling the flag is what repairs it.
    fixed = Simulator(hw(need, True)).simulate(model, wl)
    assert fixed.prefill.get_aa_total().dram_read > 0, "flag did not repair it"
    n += 1

    # 7. Decode is NOT inert -- staging removes the decode score write too.
    assert run(ctx, 1, score_kb=need)['decode_dram'] < base['decode_dram'], \
        "decode should lose the score spill as well"
    n += 1

    print(f"Pre-flight: {n} assertions passed\n")
    return n


# -------------------------------------------------------------------- sweeps

def sweep(report_path):
    rows = []
    preflight()
    rep = Report(
        report_path,
        "Attention score staging",
        subtitle="The largest DRAM term in the model was an assumption",
        source="analysis/memory/prefill_run.py",
        setup=["Model: LLaMA-3-8B on Omni-LUT, 4-bit KV, FP16 activations, "
               "standard attention."])

    rep.summary([
        "`QK` wrote the full score matrix to DRAM and `Attn.V` read it back — "
        "**unconditionally, in both phases, regardless of capacity**. That was "
        "**99.9% of prefill DRAM at 32K**.",
        "Staging the scores on chip cuts prefill DRAM **937x** at 32K "
        "(4,401.7 GB to 4.70 GB), and the corrected path then agrees with "
        "FlashAttention **to the byte**.",
        "**No published number moves.** TTFT is unchanged at 1.00x everywhere — "
        "prefill is compute-bound, so removing 4.4 TB of traffic does not move it — "
        "and no tracked markdown contained a prefill-DRAM or TTFT figure.",
        "Decode is a smaller but real exposure: **11.1% of decode attention DRAM**, "
        "1.2–10.4% of all decode DRAM by batch, TPOT shift 1.005x–1.016x.",
    ])

    # ---- A. Prefill --------------------------------------------------------
    rep.section(
        "A. Prefill DRAM, before and after",
        "Batch 1. The buffer is one query row's score vector for **one** "
        "(batch, head) instance — not multiplied by head count.")
    trows = []
    for ctx in CONTEXTS:
        need = buf_kb(ctx)
        a = run(ctx, 1)
        b = run(ctx, 1, score_kb=need)
        c = run(ctx, 1, score_kb=need, prefill_kv=True)
        rows += [{'section': 'A', **x} for x in (a, b, c)]
        trows.append([f"{ctx:,}", f"{need:,} KB", f"{a['prefill_dram']:,}",
                      f"{b['prefill_dram']:,}", f"{c['prefill_dram']:,}",
                      f"{a['prefill_dram']/c['prefill_dram']:.1f}x"])
    rep.table(["context", "row buffer", "spilling", "staged only",
               "staged + K/V read", "reduction"], trows, aligns="rrrrrr")
    rep.note(
        "**'staged only' is a documented non-configuration**, shown to make the "
        "incoherence visible: the scores stay on chip but prefill still reads no "
        "K/V, so it under-reads. The usable column is the last one.\n"
        "The buffer must cover the **longest** row the run produces — context plus "
        "output tokens — because decode grows `kv_len`. Size it for prefill alone "
        "and the scores silently spill again the moment decoding starts.")

    # ---- B. Flash convergence ----------------------------------------------
    rep.section(
        "B. Does the corrected path agree with FlashAttention?",
        "The strongest check that the correction is right: at one Q block both "
        "paths reduce to Q/K/V read once plus KV writeback, so they must agree "
        "exactly, not approximately.")
    trows = []
    for ctx in CONTEXTS:
        need = buf_kb(ctx)
        st = run(ctx, 1, score_kb=need, prefill_kv=True)
        f1 = run(ctx, 1, score_kb=need, prefill_kv=True, flash=ctx)
        f2 = run(ctx, 1, score_kb=need, prefill_kv=True, flash=256)
        rows += [{'section': 'B', **x} for x in (st, f1, f2)]
        trows.append([f"{ctx:,}", f"{st['prefill_dram']:,}",
                      f"{f1['prefill_dram']:,}", f"{f2['prefill_dram']:,}",
                      f"{f2['prefill_dram']/f1['prefill_dram']:.1f}x"])
    rep.table(["context", "standard (corrected)", "flash, block = seq",
               "flash, block = 256", "256 penalty"], trows, aligns="rrrrr")
    rep.note(
        "Exact agreement at one Q block. The `block=256` penalty is flash's real "
        "tiling cost — K/V re-read once per Q block — and is why a larger block is "
        "cheaper. It also confirms the flash path is not being flattered.")

    # ---- C. TTFT ------------------------------------------------------------
    rep.section("C. What it does to TTFT", "Batch 1.")
    trows = []
    for ctx in CONTEXTS:
        need = buf_kb(ctx)
        a = run(ctx, 1)
        c = run(ctx, 1, score_kb=need, prefill_kv=True)
        trows.append([f"{ctx:,}", f"{a['ttft_s']*1e3:.1f} ms",
                      f"{c['ttft_s']*1e3:.1f} ms",
                      f"{a['ttft_s']/c['ttft_s']:.2f}x"])
    rep.table(["context", "TTFT before", "TTFT after", "change"], trows,
              aligns="rrrr")
    rep.note(
        "Unchanged. Prefill is compute-bound, so removing even 4.4 TB of DRAM "
        "traffic does not move the roofline. Nothing published moves either: no "
        "tracked markdown contains a TTFT or prefill-DRAM figure, and every study "
        "that emits TTFT runs the flash path, which never had the spill. **This is "
        "a cleanup before the claim, not a retraction.**")

    # ---- D. Decode ----------------------------------------------------------
    rep.section(
        "D. Decode is not inert either",
        "Scores are FP16 across 32 **query** heads while the KV cache is 4-bit "
        "across only 8 **KV** heads, so the spill is a real if secondary term.")
    trows = []
    for ctx in (8192, 32768):
        for b in BATCHES:
            need = buf_kb(ctx)
            a = run(ctx, b)
            c = run(ctx, b, score_kb=need, prefill_kv=True)
            rows += [{'section': 'D', **x} for x in (a, c)]
            trows.append([f"{ctx:,}", str(b), f"{a['decode_dram']:,}",
                          f"{c['decode_dram']:,}",
                          f"{1 - c['decode_dram']/a['decode_dram']:.1%}",
                          f"{a['tpot_s']/c['tpot_s']:.3f}x"])
    rep.table(["context", "batch", "decode DRAM before", "after", "removed",
               "TPOT change"], trows, aligns="rrrrrr")
    rep.note(
        "Isolated by **differencing** staged against unstaged, the score spill is "
        "11.1% of decode attention DRAM at every context; its share of the decode "
        "total then follows batch, since weights are fixed and attention scales. "
        "Reading `attn_v.dram_read` directly would overstate this ~5x, because that "
        "field also carries the V-cache read.\n"
        "`study2.md` sections 7–9 were produced with the spill on, so section 7's "
        "\"KV share of decode DRAM\" is really *attention* share and is ~11% scores. "
        "At a 1.005x–1.016x TPOT shift those sections need **correcting, not "
        "overturning**.")

    rep.save()
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv', default=os.path.join(_here, 'prefill.csv'))
    p.add_argument('--report', default=os.path.join(_here, 'prefill_report.md'))
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
