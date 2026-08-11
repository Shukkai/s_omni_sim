# Omni-LUT Memory Model Study

Companion to `study.md`, which measured **cycles**. This one measures **memory** —
SRAM capacity, DRAM access granularity, and KV reuse — and re-baselines the KV
reduction results against them.

**Setup.** Identical to `study.md` unless a section says otherwise.

- Model: LLaMA-3-8B — 32 layers, GQA 32/8, d_model 4096, d_ffn 14336.
- Hardware: Omni-LUT-KV4 — 32x4 LUT array, W4A16KV4, `AW=AA=OMNI`,
  500 MHz, 51.2 GB/s.
- Scripts and reports: `analysis/memory/`. Staged record and revert points:
  `ram_sim_plan.md`.

**Method.** Every model change is a `HardwareConfig` field whose *disabled*
default reproduces the previous numbers exactly, checked by
`analysis/regression/baseline.py` (36 configs x workloads, 22,380 values,
compared leaf by leaf). Nothing in `study.md` moves unless asked.

---

## 1. The gate, and what it caught

- `study.md`'s TODO asked for DRAM and SRAM models. Those edit
  `_calculate_memory_access` and `_simulate_matmul` — which every published
  number depends on — so the first step was making accidental change impossible.
- `baseline.py` captures `to_dict()` plus both roofline helpers, drops
  per-execution duplicates (an op runs once per layer per decode step with
  identical metrics — 129 MB of exact copies) and stores a SHA-256 of the
  unslimmed tree so nothing is lost by the slimming.
- **It caught a defect in itself before it could hide a real one.** 816 reported
  "regressions" against an unchanged tree were type artifacts: `interp1d`
  returns `np.float64`, which reprs as `np.float64(0.035…)` fresh and `0.035…`
  after a JSON round-trip. Fixed at the source (`float(energy)`) *and* in the
  comparison. Had this landed mid-change, a genuine regression would have been
  invisible in the noise.

---

## 2. SRAM capacity — what actually fits

`peak_sram_bytes` was computed and printed but never checked against anything.
`sram_capacity_kb` makes it a constraint; on overflow, spill policy v1 re-reads
the resident operand once per column tile (no re-tiling).

| context | dense | any KV budget <= 4096 |
|---|---:|---:|
| 2K / 8K | 1024 KB | 1024 KB |
| 32K | **2176 KB** | **1024 KB** |

- **Decode's working set is floored at 924.5 KB** by the FFN/projection tiles.
  The KV tile only becomes the binding term past ~16K context — below that,
  capacity is not the constraint at all.
- **At 32K a KV budget buys a smaller chip**: 2.1x less SRAM. That is the one
  place the budget pays in capacity rather than latency.
- **Policy v1 is non-monotonic and the flag is the trustworthy output.** Past
  1024 KB at 32K the binding term is the KV tile itself, which needs re-tiling,
  so v1 flags the overflow and charges nothing — the *larger* overflow prices
  lower. Use `sram_overflow`; treat `sram_refetch_bytes` as first-order only.
- **Prefill is excluded, and this is a model gap not a result.**
  `_calculate_peak_sram` holds the entire prefill activation matrix —
  O(seq x d_model), 59 MB at 2K context and 2.1 GB at 32K — so it overflows at
  every plausible capacity and its spill charge is a meaningless ~770 GB
  constant. A real accelerator tiles prefill over the sequence; the model does
  not. **Open.**

---

## 3. Batch as a capacity axis

Enforcing capacity exposed that batch was a *loop* in one half of the model and
a *dimension* in the other: projections issue as one GEMM with
`proj_m = batch x seq_len`, so their footprint scaled with batch, while
attention issues per `(batch, head)` and its footprint did not move at all.
`sram_batch_model="concurrent"` makes attention agree with projections.

Largest batch whose decode working set fits, 32K context (32 = sweep ceiling):

| SRAM | dense | budget 4096 | budget 1024 |
|---|---:|---:|---:|
| 4 MB | 1 | 8 | 32 |
| 8 MB | 2 | 16 | 32 |
| 16 MB | 4 | 32 | 32 |

- **Capacity and batch trade one-for-one**, and at fixed capacity a KV budget
  buys batch directly — 8x at 4 MB for a 4096-entry budget.
- **`"concurrent"` is a scheduling assumption, not a measurement.** It is the
  assumption the projection side already made, which is why adopting it makes
  the model self-consistent — but hardware that serialises batch shows none of
  this. `"sequential"` remains the default.

---

## 4. DRAM burst granularity

DRAM was one flat bandwidth number, so a packed cache and a scattered gather
cost the same. `dram_burst_bytes` rounds each access up to a burst;
`dram_read_eff` / `dram_write_eff` carry bytes actually moved while
`dram_read` / `dram_write` stay logical.

| access | burst | charged | waste |
|---|---:|---:|---:|
| dense 4-bit KV entry, 64 B | 32 B | 64 B | 1.00x |
| ThinK entry (`d_ret=77`), 38 B | 32 B | 64 B | **1.68x** |
| dense entry, 64 B | 128 B | 128 B | **2.00x** |

- **Alignment matters, not run length.** A dense 4-bit KV entry is
  `128 x 4/8` = 64 B — exactly two 32 B bursts — so the term is inert for
  everything else this study models.
- **The one misaligned shape is ThinK's pruned entry.** §5 of `study.md`
  computed ThinK's speedup as K-cache bytes / 51.2 GB/s, which assumes every
  saved byte is a saved transfer. At 38 B per entry that is not true, so a
  compacted ThinK cache plausibly returns part of its saving to burst rounding.
  Quantifying it needs the pruned-entry layout pinned down. **Open.**
- `dram_power_model.dram_energy` already models 1024 B rows and 8 B bursts
  internally. That is a different question — what fetching costs, versus which
  bytes get fetched — so the effective count feeds it rather than competing.

---

## 5. Select-without-evict (Quest / TidalDecode / NSA)

`kv_budget.py` deferred selective reading because on a flat model it is
indistinguishable from compacted eviction. With burst granularity it can finally
be told apart — and the answer is that it still isn't.

| pages read | entries | evict (eff) | select (eff) | ratio |
|---|---:|---:|---:|---:|
| 16 | 256 | 17,913,118,720 | 17,913,118,720 | 1.000x |
| 1024 | 16384 | 21,843,705,856 | 21,843,705,856 | 1.000x |

- **Byte-identical at every `k` tested**, because a 4-bit KV entry is exactly one
  DDR-class burst — so a page-gathering reader is burst-aligned at *every* page
  size, down to `page = 1` (token-granular).
- **Granularity is a bit-width property here, not a selection property.** At
  3-bit KV an entry is 48 B and a single-entry gather does pay 1.33x.
- **`kv_budget.py`'s deferral was correct for this hardware**, not an oversight.
- **What selection actually costs is its metadata.** Quest-style per-page min/max
  scales with the *context*, not with what was selected, so its share grows as
  selection gets more aggressive: 2.2–2.6% of decode DRAM at page 16, 0.5–0.6% at
  page 64. Decode speedup at 3% read goes 2.464x -> 2.404x once it is charged.

---

## 6. KV residency across decode steps

The cache is append-only — entries 1..n-1 are bit-identical between step *t* and
*t+1* — yet the model re-read the whole cache from DRAM every token, with no
reference to on-chip capacity. `kv_sram_kb` removes whatever fits.

At 32K, batch 8, dense:

| buffer | DRAM saved | energy saved | TPOT gain |
|---|---:|---:|---:|
| 8 MB | 2.3% | 1.5% | 0.4% |
| 32 MB | 9.2% | 6.1% | 1.5% |
| 128 MB | **36.8%** | **24.6%** | 6.2% |

- **Residency is an energy and capacity lever, never a throughput one.** The
  bytes go away; the latency does not, because `attn_v` is compute-bound under a
  4-bit KV cache and the traffic removed was not on the critical path.
- **Eviction's advantage is real, not an artifact of the baseline's re-reads.**
  This fix was built to test that and came back negative: `evict-1024` at batch
  32 holds ~16x from a 0 KB buffer to a 128 MB one. The dense K+V working set is
  32 MB per layer at batch 1 and **1,024 MB at batch 32**, so no plausible buffer
  holds a meaningful fraction.
- **The two compose, and the order matters:** eviction shrinks the working set to
  a size a buffer can hold (§3), and residency then removes what is left.
- Steady-state model: the one-time buffer fill after prefill is not charged,
  which slightly favours residency — under 0.4% of KV traffic over 256 output
  tokens.

---

## 7. Where KV reduction actually pays

Decode weight traffic is **constant** in batch (7.65 GB — read once, reused
across the batch); KV traffic scales linearly. So the KV share of decode DRAM,
which is the ceiling on what *any* KV technique can win, moves enormously:

| context | batch | KV share | ceiling |
|---|---:|---:|---:|
| 8K | 1 | **10.1%** | 1.11x |
| 32K | 1 | 30.9% | 1.45x |
| 32K | 8 | 78.2% | 4.58x |
| 32K | 32 | **93.5%** | 15.32x |

Decode TPOT speedup vs dense at 32K:

| technique | batch 1 | batch 8 | batch 32 |
|---|---:|---:|---:|
| evict 1024 | 2.460x | 6.971x | **15.957x** |
| select 3% | 2.404x | 6.358x | **12.854x** |
| evict 4096 | 2.156x | 4.418x | 6.520x |
| ThinK-K `d=77` | 1.034x | 1.049x | 1.054x |

- **Entry-count techniques were measured in the wrong place.** At batch 1 they
  compete for a tenth of decode traffic; their batch-1 numbers understate them
  several-fold.
- **Channel pruning does not recover, and that is a different ceiling.** ThinK
  cuts bytes — decode DRAM falls to 0.825x at batch 32 — but latency barely
  moves, so the bytes were never on the critical path. `attn_v` is compute-bound
  and the `LUT_OS_V` round cost has no N term, so pruning `head_dim` idles array
  columns instead of saving cycles. **`study.md` §5's conclusion stands, for the
  reason it gave.** More batch cannot fix it.

---

## 8. The recurring pattern

Three independent techniques — ThinK channel pruning, select-without-evict, and
KV residency — each removed real DRAM traffic and each produced little or no
speedup, for the same reason: **`attn_v` is compute-bound under a 4-bit KV
cache, so KV bytes are usually not the critical path.**

- The axes differ in whether they touch cycles at all:

  | axis | cycles | DRAM |
  |---|---|---|
  | channel (`head_dim` = N) | **null** — no N term | linear |
  | token (`kv_len` = K) | linear via `k_eff` | linear |
  | **bit-width (`qbit`)** | **linear** | **linear** |

- `cycles = batch_size x per_round x rounds x qbit`, so **bit-width is the only
  axis that is a direct multiplier on cycles as well as bytes**, with no null
  anywhere, and it composes with eviction rather than competing.
- Separately, decode `attn_v` occupancy is 3.12% of 4096 lanes because `M=1`.
  Cycles scale *exactly* linearly with `batch x heads`, so at batch 32 there are
  1,024 instances each lighting 128 of 4,096 lanes, run back-to-back.

---

## TODO

- **Tile prefill in `_calculate_peak_sram`.** It holds the whole activation
  matrix, so prefill capacity claims and its spill charge are both unusable
  (§2). The only outright *bug* the memory work found.
- **Pin down ThinK's pruned-entry layout.** A 38 B entry is the one shape that is
  not burst-aligned, so §5 of `study.md` may overstate its speedup (§4).
- **Profile dynamic bit sparsity.** `qbit` is modelled as static;
  `analysis/bit_width/` sweeps fixed widths. Bit-plane skipping (cycles =
  non-zero planes per round, not a constant `qbit`) and mixed-precision KV
  (KIVI / ZipCache / KVQuant, giving a weighted-average effective `qbit`) both
  land on the one multiplier that touches cycles. Nothing has characterized this
  on a LUT array.
- **Test GQA-group packing.** The 4 query heads of a GQA group attend to the same
  KV and therefore the same LUT, so they could issue as one `M=4` op instead of
  four `M=1` ops. In the cycle model that is 3.98x on decode `attn_v`, and since
  `per_round` charges `array_m=32` for any `M>1`, `M=4` and `M=32` cost the same
  — so a further 8x of work would ride along free (occupancy 3.12% -> 99.5%).
  **Verify against Omni-LUT §IV-D first**: this assumes `array_m` indexes query
  rows and the LUT is KV-derived and broadcast across them. If the LUT is
  query-derived the result collapses.
- **Measure the energy side of channel pruning.** Carried over from `study.md`;
  unchanged by this work.
