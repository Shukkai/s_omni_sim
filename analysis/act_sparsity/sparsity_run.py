"""
FFN activation sparsity: the one lever aimed at what decode actually waits for.

Every KV technique in `study.md` §4-§15 aims at the KV cache, and §13 recorded
why they keep failing alike: decode on this array is compute-bound, so removing
*bytes* buys little.  But §3 said something else that no section followed up --
at 2K context the accelerator idles ~85% of decode waiting on **weights**, and
fc1+fc2 are the largest weight tensors in the model.

Activation sparsity is the lever pointed there.  A gated FFN drives most of its
hidden units to near-zero for any token; skipping unit `j` skips column `j` of
FC1 and row `j` of FC2, so *weights* stop being fetched.  TEAL, CATS and
Deja Vu all produce such a mask.

**Three things decide whether it pays, and only one of them is a design
choice.**  `analysis/act_sparsity/act_sparsity.py` has the model; the axes are
`density`, `mask_source` (which matrices can use the mask), and `share_mask`
(whether the `M` tokens of one GEMM share one).  The third is a fact about the
workload rather than a knob, and it is the one that decides the technique.

Selection cost is excluded, as it is for every technique in §4-§15.

Usage:
    python sparsity_run.py
    python sparsity_run.py --csv sparsity.csv --report sparsity_report.md
"""

import argparse
import csv
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, '..', '..'))
for p in ('simulator', 'analysis', 'analysis/cycle_breakdown',
          'analysis/memory', 'analysis/act_sparsity'):
    sys.path.insert(0, os.path.join(_root, *p.split('/')))

from simulator import HardwareConfig, WorkloadConfig                 # noqa: E402
from memory_tech import with_memory_technology                       # noqa: E402
from model_configs import get_model_config                           # noqa: E402
from report import Report                                            # noqa: E402
from act_sparsity import ActSparsitySimulator                        # noqa: E402

MODEL = 'LLaMA-3-8B'
CONTEXT = 8192
OUTPUT_TOKENS = 4
KV_BITS = 4
SCORE_SRAM_KB = 128
DENSITIES = [1.0, 0.5, 0.25, 0.1, 0.05]
BATCHES = [1, 2, 4, 8, 32]
GROUPS = [1, 4, 16, 64]
GB = 1e9


def base_hw():
    return with_memory_technology(HardwareConfig(
        array_m=32, array_n=4, FPE_array_size=64,
        act_bits=16, accumulate_bits=32, weight_bits=4, kv_cache_bits=KV_BITS,
        AW_mode="OMNI", AA_mode="OMNI", score_sram_kb=SCORE_SRAM_KB,
    ), 'DDR5-6400')


def measure(context=CONTEXT, batch=1, **kw):
    sim = ActSparsitySimulator(base_hw(), **kw)
    m = get_model_config(MODEL)
    w = WorkloadConfig(batch_size=batch, input_tokens=context,
                       output_tokens=OUTPUT_TOKENS, flash_block_size=0)
    r = sim.simulate(m, w)
    ttft, tpot = sim.compute_roofline_latency(r, w)
    dec = r.decode.get_total_metrics()
    pf = r.prefill.get_total_metrics()
    return {'ttft_s': ttft, 'tpot_s': tpot,
            'decode_cycles': dec.cycles,
            'decode_dram_logical': dec.dram_read + dec.dram_write,
            'decode_dram_eff': dec.dram_read_eff + dec.dram_write_eff,
            'prefill_dram_eff': pf.dram_read_eff + pf.dram_write_eff}


def saving_kept(dense, sparse):
    """Fraction of the logical byte saving that becomes an effective one."""
    logical = dense['decode_dram_logical'] - sparse['decode_dram_logical']
    if logical <= 0:
        return 0.0
    kept = (dense['decode_dram_eff'] - sparse['decode_dram_eff']) / logical
    # The covering clamp guarantees a masked read never costs more than the
    # dense one it replaces, so a negative here is float noise around zero --
    # normalise it rather than printing "-0.0%".
    return kept + 0.0 if abs(kept) > 1e-9 else 0.0


# ============================================================================
# Pre-flight
# ============================================================================

def preflight():
    dense = measure()

    # 1. density = 1.0 is exactly the stock simulator -- the hooks are inert.
    one = measure(density=1.0)
    for k in ('ttft_s', 'tpot_s', 'decode_cycles', 'decode_dram_eff'):
        assert one[k] == dense[k], f"density 1.0 must be inert, {k} moved"

    # 2. At batch 1 the union term is the identity, so per-token and shared
    #    masks must agree exactly.  This is what makes batch 1 the clean case.
    a = measure(batch=1, density=0.1, share_mask=True)
    b = measure(batch=1, density=0.1, share_mask=False)
    assert a['tpot_s'] == b['tpot_s'], \
        "share_mask must not matter at M = 1"

    # 3. Sparsity helps decode monotonically at batch 1.
    prev = dense['tpot_s']
    for d in (0.5, 0.25, 0.1, 0.05):
        r = measure(density=d)
        assert r['tpot_s'] < prev, f"TPOT should fall at density {d}"
        prev = r['tpot_s']

    # 4. Neuron-major collects its whole saving at group 1; model-major
    #    collects none.  The layout claim, asserted rather than described.
    nm = measure(density=0.1, weight_layout="neuron_major")
    mm = measure(density=0.1, weight_layout="model_major")
    assert saving_kept(dense, nm) > 0.99, \
        f"neuron-major should keep ~all of it, kept {saving_kept(dense, nm)}"
    assert saving_kept(dense, mm) < 0.01, \
        f"model-major should keep ~none of it, kept {saving_kept(dense, mm)}"

    # 5. And model-major is clamped to the dense read rather than pricing
    #    above it -- a gathering reader always has that fallback.
    assert mm['decode_dram_eff'] <= dense['decode_dram_eff'] * 1.001, \
        "a masked read must never cost more than the dense read it replaces"

    # 6. A mask taken from FC1's output cannot skip FC1, so it must buy
    #    strictly less than one taken from the FFN input.
    inp = measure(density=0.1, mask_source="input")
    out = measure(density=0.1, mask_source="output")
    assert dense['tpot_s'] / out['tpot_s'] < dense['tpot_s'] / inp['tpot_s'], \
        "an output-derived mask must buy less than an input-derived one"

    # 7. The union term is the identity at M=1 and saturating above it, so the
    #    benefit must fall monotonically with batch.
    prev_speedup = None
    for b in BATCHES:
        bd = measure(batch=b)
        bs = measure(batch=b, density=0.1)
        sp = bd['tpot_s'] / bs['tpot_s']
        if prev_speedup is not None:
            assert sp < prev_speedup, f"speedup should fall by batch {b}"
        prev_speedup = sp

    # 8. Prefill with per-token masks collects nothing: `1 - (1-d)^M` at
    #    M = 8192 is 1.0 to every digit that matters.
    pf = measure(density=0.1, share_mask=False)
    assert abs(pf['ttft_s'] - dense['ttft_s']) < dense['ttft_s'] * 1e-6, \
        "per-token masks cannot help prefill"

    print("pre-flight: 8 checks passed")


# ============================================================================
# Sweep
# ============================================================================

def sweep(report_path):
    rows = []
    preflight()

    dense = measure()

    rep = Report(
        report_path,
        "FFN activation sparsity",
        subtitle="The one lever aimed at what decode actually waits for, and "
                 "the axis that decides it",
        source="analysis/act_sparsity/sparsity_run.py",
        setup=[f"Model: LLaMA-3-8B on Omni-LUT, 4-bit KV, context "
               f"{CONTEXT:,}, DDR5-6400, standard attention, scores staged.",
               "Mask from the FFN input (TEAL / Deja Vu) and neuron-major "
               "weights unless a section sweeps them. Selection cost "
               "excluded, as in §4–§15."])

    d10 = measure(density=0.1)
    b32d = measure(batch=32)
    b32s = measure(batch=32, density=0.1)
    mm = measure(density=0.1, weight_layout="model_major")
    out = measure(density=0.1, mask_source="output")

    rep.summary([
        f"**It works, and it is the first thing in this repo that does.** At "
        f"batch 1, 10% density buys "
        f"**{dense['tpot_s'] / d10['tpot_s']:.2f}×** decode TPOT — against "
        f"§16(b)'s finding that 16× the DRAM bandwidth buys 1.10×. It aims at "
        f"weights, which is what §3 said decode actually waits for.",
        f"**Batch destroys it, and §18 says why.** "
        f"{dense['tpot_s'] / d10['tpot_s']:.2f}× at batch 1 falls to "
        f"**{b32d['tpot_s'] / b32s['tpot_s']:.3f}×** at batch 32. Two "
        f"mechanisms compound: per-token masks make the fetched weight set the "
        f"*union* over the batch, and decode is already compute-bound there.",
        f"**The layout question from §15, with the opposite answer.** "
        f"Neuron-major weights keep "
        f"**{saving_kept(dense, measure(density=0.1)):.0%}** of the saving at "
        f"group 1; model-major keeps "
        f"**{saving_kept(dense, mm):.0%}**. The difference from §15 is that a "
        f"weight layout is chosen offline by the compiler, while a KV layout "
        f"is dictated online by an append-only cache. §15's obligation was "
        f"unmeetable; this one is a build-time decision.",
        f"**Where the mask comes from is worth "
        f"{(dense['tpot_s'] / d10['tpot_s']) / (dense['tpot_s'] / out['tpot_s']):.2f}×.** "
        f"An input-derived mask (TEAL, Deja Vu) skips FC1 and FC2 alike; one "
        f"read off FC1's output (CATS) cannot skip the work that produced it.",
        "**Prefill collects nothing at all** — `1 - (1-d)^M` at M = 8,192 is "
        "1.0 to every digit that matters. This is a decode technique, and at "
        "batch 1 specifically.",
    ])

    # ---- A ------------------------------------------------------------------
    rep.section(
        "A. What it buys at batch 1",
        "Decode TPOT and DRAM against the dense baseline, mask from the FFN "
        "input, neuron-major weights.")
    trows = []
    for d in DENSITIES:
        r = measure(density=d)
        rows.append({'section': 'A', 'density': d, 'batch': 1, **r})
        trows.append([f"{d:.0%}", f"{1e3 * r['tpot_s']:,.2f} ms",
                      f"{dense['tpot_s'] / r['tpot_s']:.3f}×",
                      f"{r['decode_dram_eff'] / GB:,.2f} GB",
                      f"{r['decode_cycles'] / 1e6:,.1f} M"])
    rep.table(["density", "TPOT", "speedup", "decode DRAM", "decode cycles"],
              trows, aligns="lrrrr")
    rep.note(
        "**The curve saturates, and what it saturates against is the point.** "
        "Halving density from 10% to 5% buys only 1.911× → 2.013×, because "
        "once the FFN weights stop dominating, attention's own DRAM and "
        "compute become the floor. Cycles barely move at all — the FFN is a "
        "small share of decode cycles (§1) even though it is a large share of "
        "decode *bytes*. **This is a bandwidth technique that works, on an "
        "accelerator where §13 concluded bandwidth techniques do not** — "
        "because it is the only one aimed at weights rather than KV.")

    # ---- B ------------------------------------------------------------------
    rep.section(
        "B. The layout question, and why it is not §15's answer",
        "10% density, varying how many *consecutive* hidden units share a "
        "mask decision. 'Saving kept' is the fraction of the logical byte "
        "saving that survives DDR5's 64 B burst.")
    for layout in ("neuron_major", "model_major"):
        trows = []
        for g in GROUPS:
            r = measure(density=0.1, weight_layout=layout, neuron_group=g)
            kept = saving_kept(dense, r)
            rows.append({'section': 'B', 'layout': layout, 'neuron_group': g,
                         'saving_kept': kept, **r})
            run_b = g * (4096 * 0.5 if layout == "neuron_major" else 0.5)
            trows.append([str(g), f"{run_b:,.1f} B",
                          f"{kept:.1%}",
                          f"{dense['tpot_s'] / r['tpot_s']:.3f}×"])
        rep.table(["neuron group", "contiguous run", "saving kept", "speedup"],
                  trows, aligns="lrrr",
                  caption=f"{layout} — DDR5-6400, 64 B burst")
    rep.note(
        "**Neuron-major is burst-aligned at group 1**: one hidden unit's "
        "weights are `d_model × weight_bits/8` = 2,048 B, or 32 whole bursts, "
        "so a fully *unstructured* mask collects everything. Model-major makes "
        "the same unit a 0.5 B stride and collects nothing — the identical "
        "cliff §15 found for KV channel pruning.\n\n"
        "**The difference is who chooses the layout.** §15's requirement was "
        "that retained KV channels be contiguous and compacted, which an "
        "append-only cache written online cannot promise. A weight matrix is "
        "laid out once, offline, by the compiler. **The same structural "
        "obligation is unmeetable in one case and free in the other**, and "
        "that — not the sparsity pattern — is what separates them.")

    # ---- C ------------------------------------------------------------------
    rep.section(
        "C. Where the mask comes from",
        "10% density, batch 1. The mask has to be computed from something, "
        "and that decides which matrices can use it.")
    trows = []
    for ms, label in (("input", "FFN input (TEAL, Deja Vu)"),
                      ("output", "FC1 output (CATS)")):
        r = measure(density=0.1, mask_source=ms)
        rows.append({'section': 'C', 'mask_source': ms, **r})
        trows.append([label,
                      "FC1 + FC2" if ms == "input" else "FC2 only",
                      f"{1e3 * r['tpot_s']:,.2f} ms",
                      f"{dense['tpot_s'] / r['tpot_s']:.3f}×"])
    rep.table(["mask source", "sparse matrices", "TPOT", "speedup"],
              trows, aligns="llrr")
    rep.note(
        "**FC1 is more than half the win**, so the choice is not a detail. An "
        "output-derived threshold is the cheaper and more accurate mask and "
        "buys 1.31×; an input-derived one buys 1.91× but needs either a "
        "predictor (Deja Vu, whose matmul is not charged here) or a threshold "
        "on the input that no longer sees what it is masking. **That trade is "
        "the real design question this axis poses**, and it is an algorithm "
        "question, not a hardware one.")

    # ---- D ------------------------------------------------------------------
    rep.section(
        "D. Batch, and the union that kills it",
        "10% density. `share_mask` brackets the truth: per-token is what "
        "actually happens, shared is the bound a perfectly correlated batch "
        "would reach.")
    trows = []
    for b in BATCHES:
        bd = measure(batch=b)
        sh = measure(batch=b, density=0.1, share_mask=True)
        pt = measure(batch=b, density=0.1, share_mask=False)
        union = 1.0 - 0.9 ** b
        rows.append({'section': 'D', 'batch': b, 'union_density': union,
                     'tpot_dense': bd['tpot_s'], 'tpot_shared': sh['tpot_s'],
                     'tpot_pertoken': pt['tpot_s']})
        trows.append([str(b), f"{union:.1%}",
                      f"{bd['tpot_s'] / sh['tpot_s']:.3f}×",
                      f"{bd['tpot_s'] / pt['tpot_s']:.3f}×"])
    rep.table(["batch", "union density", "speedup (shared mask)",
               "speedup (per-token)"], trows, aligns="lrrr")
    rep.note(
        "**Two mechanisms compound, and separating them is what the bracket "
        "is for.** The per-token column falls to 1.003× at batch 32; the "
        "shared-mask column still only reaches 1.082×. So the union explains "
        "most of the collapse but not all of it — **the remainder is that "
        "decode is already compute-bound at batch 32**, which is exactly "
        "§18's regime map, and a technique that removes bytes cannot help "
        "there. §13's recurring pattern, arriving for a non-KV technique.\n\n"
        "**This makes activation sparsity the mirror image of KV eviction.** "
        "§8 showed a KV budget buys batch; activation sparsity is *spent* by "
        "batch. They are complementary rather than competing, and a serving "
        "stack running batch 1 for latency is exactly where this one pays.")

    # ---- E ------------------------------------------------------------------
    rep.section(
        "E. Prefill",
        "Same masks, applied to the prefill phase.")
    trows = []
    for sm, label in ((True, "shared mask (fiction at M = 8,192)"),
                      (False, "per-token (what happens)")):
        r = measure(density=0.1, share_mask=sm)
        rows.append({'section': 'E', 'share_mask': sm, **r})
        trows.append([label, f"{r['ttft_s']:,.1f} s",
                      f"{dense['ttft_s'] / r['ttft_s']:.3f}×"])
    rep.table(["mask model", "TTFT", "speedup"], trows, aligns="lrr")
    rep.note(
        "**Prefill collects exactly nothing**, and the arithmetic is not "
        "close: `1 - (1-0.1)^8192` is 1.0 to hundreds of digits. Every weight "
        "column is needed by *some* token in the batch, so none can be "
        "skipped. The shared-mask row is printed only to show that the "
        "mechanism is the union and not the model — the sparsity is fully "
        "available, and the workload is what refuses it.")

    rep.save()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=os.path.join(_here, 'sparsity.csv'))
    ap.add_argument('--report',
                    default=os.path.join(_here, 'sparsity_report.md'))
    args = ap.parse_args()

    rows = sweep(args.report)
    keys = sorted({k for r in rows for k in r})
    with open(args.csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.csv} and {args.report}")


if __name__ == '__main__':
    main()
