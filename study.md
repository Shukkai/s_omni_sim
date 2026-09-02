# Omni-LUT Simulator Study

Where the cycles go, where the bytes go, and which KV-reduction techniques
survive both.

**Setup.** Common to every section unless one says otherwise.

- Model: LLaMA-3-8B — 32 layers, GQA 32/8, d_model 4096, d_ffn 14336.
- Hardware: Omni-LUT-KV4 — 32x4 LUT array, W4A16KV4, `AW=AA=OMNI`,
  500 MHz, 51.2 GB/s (which is exactly `DDR5-6400` — see §16).
- On-chip memory: **unlimited and undifferentiated** in §1–§21. The RTL's
  actual four SRAMs — 256 KB input, 256 KB scale, 2 MB weight, 512 KB output —
  are a preset (`simulator/buffer_tech.py`) that only §22 switches on.
- Workload: batch 1, 256 output tokens, standard attention (no FlashAttention,
  so `qk_matmul` and `attn_v_matmul` stay separate stages), except where a
  section sweeps batch.

**How to read it.**

| | sections | question |
|---|---|---|
| **Cycles** | §1–§3 | where does the array actually spend time? |
| **KV reduction** | §4–§5 | what do eviction and channel pruning cost and buy? |
| **The memory model** | §6–§9 | what was the byte model missing, and what did fixing it change? |
| **Re-measured** | §10–§13 | the KV techniques against the corrected model, and why they fail alike |
| **What actually moves it** | §14–§16 | array packing, mask structure, and the memory part itself |
| **The framing itself** | §17–§18 | what the no-overlap roofline and the OS-V round count cost every number above |
| **The memory model, corrected** | §19–§20 | which memory moves the bytes, and the one outright bug |
| **Off the KV axis** | §21 | activation sparsity — the first lever that moves decode |
| **Against the RTL** | §22 | what the four real SRAMs change, and what they confirm |

**If you read one thing.** §21 is the only technique in this document that
moves decode materially (**1.911×** at batch 1), and §21(e) is why: it is the
first result here that establishes what decode is bound by *by intervention*
rather than by reading a roofline `max()`.

**Method for §6 onward.** Every model change is a `HardwareConfig` field whose
*disabled* default reproduces the previous numbers exactly, checked by
`analysis/regression/baseline.py` (36 configs x workloads, 23,064 values,
compared leaf by leaf). Nothing in §1–§5 moves unless asked. Every stage and its
revert point is in **Appendix A**; the checks run after each one are in
**Appendix B**.

**Where the full tables are.** Sections link to `*_report.md` files under
`analysis/memory/` and `analysis/array_packing/` — those hold every swept row,
while a section quotes only the numbers that carry the argument. **They are
generated, not tracked**: the `*_run.py` sweep is the source and the markdown is
its output, so a fresh checkout has the scripts but not the reports. Rebuild
them all with:

```
for f in analysis/memory/*_run.py analysis/array_packing/pack_run.py \
         analysis/act_sparsity/sparsity_run.py; do python "$f"; done
python analysis/memory/plot_memory.py          # the nine figures
```

The figures are drawn from the CSVs the sweeps write, so they cannot drift from
the tables — re-run the sweep, re-run the plotter, and both move together.

---

## 1. By pipeline stage

![Stage breakdown](analysis/cycle_breakdown/cycle_breakdown_norm.png)

Share of phase cycles, top stages only:

| Stage | Prefill 2K | Prefill 32K | Decode/tok 2K | Decode/tok 32K |
|---|---:|---:|---:|---:|
| fc1 | 30.7% | 9.8% | 12.9% | 1.4% |
| fc2 | 30.7% | 9.8% | 11.2% | 1.2% |
| q_proj / o_proj | 8.8% each | 2.8% each | 3.2% each | 0.4% each |
| qk_matmul | 4.4% | 22.4% | 4.2% | 4.1% |
| attn_v_matmul | 4.4% | 22.4% | **55.5%** | **88.4%** |
| softmax (VPU) | 5.4% | **27.9%** | 2.1% | 3.4% |
| **Total cycles** | 3.12 G | 153.9 G | 4.09 M | 38.2 M |

- **Prefill flips from FFN-bound to attention-bound.** fc1+fc2 are 61% of cycles
  at 2K, falling to 20% at 32K while attention (qk + attn_v + softmax) rises to
  73% — attention grows quadratically, the FFN linearly.
- **Decode is attention-dominated everywhere**, rising 60% -> 93% with context.
- **`attn_v_matmul` costs far more than `qk_matmul`** — in `LUT_OS_V` its N
  dimension is `head_dim`=128 while qk's is `kv_len`, so attn_v serializes over
  the cache in `k_eff` while qk parallelizes across tiles.
- Consequence: any KV-reduction technique is aiming at 62% of decode cycles at
  2K and 96% at 32K.

---

## 2. By hardware unit

![Unit breakdown](analysis/cycle_breakdown/cycle_breakdown_units_norm.png)

Share of serial cycles:

| Unit | Prefill 2K | Prefill 8K | Prefill 32K | Decode/tok 2K | Decode/tok 32K |
|---|---:|---:|---:|---:|---:|
| PE array (compute) | 90.32% | 82.50% | 71.17% | 94.59% | 95.34% |
| PE array (fill/drain) | 1.59% | 0.36% | 0.08% | 1.14% | 0.55% |
| LGU | 0% | 0% | 0% | 0.69% | 0.33% |
| Accumulator | 0.09% | 0.02% | 0.00% | 0.46% | 0.22% |
| Operand issue | 0.04% | 0.01% | 0.00% | — | — |
| VPU | 7.96% | 17.11% | 28.75% | 3.12% | 3.56% |

- **The array is efficient; the overheads are not the problem.** Systolic
  fill/drain stays under 1.6%, the accumulator under 0.5%.
- **The LGU is nearly free at full cache.** Zero in prefill — `LUT_WS` has no
  table-generation term, since generation is pipelined into the M-long
  activation stream and fully amortized. It appears only in decode's `LUT_OS_V`
  (3 cycles/round) and stays below 0.7%.
  - So the scale-aware LGU buys AA-GEMM support at essentially no cycle cost.
  - But see §4(b): this reverses at small KV budgets, where the same fixed
    3 cycles reach ~24% of attention cycles.
- **The VPU is the real long-context threat.** 7.96% -> 28.75% of prefill from
  2K to 32K, almost entirely softmax. At 32K, softmax alone (85.9 s) is the
  single largest prefill stage — larger than any LUT GEMM.
  - Scaling the LUT array would not help; the bottleneck has moved off it.
- **BQU is not measured yet** (see TODO). The placeholder estimate puts online KV
  quantization at 4.7 M cycles at 2K prefill and 75.5 M at 32K (~0.05% of the
  phase) and treats it as concurrent with the PE array, so it is excluded from
  the serial totals above.

---

## 3. Cycles understate decode cost

Decode is DRAM-bound, so raw cycles are not latency:

| Context | Decode cycles/tok | Compute time | Roofline time | Gap |
|---|---:|---:|---:|---:|
| 2K | 4.09 M | 8.18 ms | 55.39 ms | 6.8x |
| 8K | 10.97 M | 21.94 ms | 70.67 ms | 3.2x |
| 32K | 38.15 M | 76.30 ms | 131.82 ms | 1.7x |

- **Every AW stage flags `bound="memory"`** in decode — fc1 is 1.06 ms compute
  against 18.4 ms of DRAM.
- At 2K the accelerator **idles ~85% of decode** waiting on weights, not on KV.
- **KV4 quantization is what keeps attention compute-bound** — it shrinks
  attention's own DRAM traffic enough that the array, not memory, is the limit
  there.
- The gap narrows at 32K only because **attention compute grows**, not because
  memory improves.
- Consequence: any cycle-only speedup claim on decode is inflated by up to 6.8x.
  Report cycles and roofline time together, or not at all.
- **This section states the gap; §21(e) proves it.** Everything here is a
  roofline `max()` — an accounting claim about which term won. §21 removes 60%
  of decode's bytes at nearly fixed cycles and gets 2.52× at 2K, which is the
  same statement made causally.

---

## 4. KV compaction

![Compaction breakdown](analysis/compact_breakdown/compact_breakdown.png)

**Model.** Decode attends to a dense cache of `k` entries — `kv_len -> min(kv_len, k)`.
Covers uniform-budget compacted eviction (H2O, SnapKV, StreamingLLM, TOVA).
Not per-layer/per-head budgets (PyramidKV, Ada-KV), channel pruning (ThinK, §5), or
select-without-evict (Quest, TidalDecode, NSA — §10). Selection cost excluded for all.

### (a) Regime map — where eviction is worth deploying

Ceiling speedup at 20% budget, decode roofline time per token:

| batch \ context | 2K | 8K | 32K |
|---|---:|---:|---:|
| 1 | 1.08x | 1.30x | 1.98x |
| 8 | 1.34x | 2.13x | 3.44x |
| 32 | 1.99x | 3.40x | **4.43x** |

- Driver is KV's share of decode DRAM: **2.9% -> 93.8%** across that grid.
  Weight traffic amortizes over batch; KV traffic scales with it.
- **Batch 1 / 2K is the worst case and the technique is dead there** (1.08x) —
  decode reads 2.6 GB of weights per token vs 80 MB of KV. §1–3 all sit here.
- Bounds every KV-reduction technique, not just eviction.

### (b) Fixed-overhead knee — the one novel result

`attn_v` costs `per_round = 3 (LGU) + ceil(kv_len/4) + 5 (fill/drain) + 2 (accum)`.
The constant 10 does not shrink with the budget:

| Retained entries (32K ctx) | Fixed share of attention cycles |
|---:|---:|
| 32768 (full) | 1.2% |
| 6554 (20%) | 1.7% |
| 656 (2%) | 9.3% |
| 328 (1%) | 14.9% |
| 132 (0.4%) | **23.5%** |

- Fixed overhead goes **1.2% -> 23.5%** exactly across the budgets these papers
  headline (PyramidKV 0.7% cache, SnapKV 128 entries).
- 2K/8K/32K curves **collapse onto one line** against *absolute* retained
  entries — an architectural constant, not a workload artifact.
- Invisible on a GPU; applies to any method reducing attention to `k` operands.
- **So the published accuracy-vs-budget curves have a cost axis that does not
  transfer to LUT-based hardware.**

### (c) Compaction cost — settled, not a tradeoff

- Cost is one-time, benefit repeats every token: payback = `(1+b)/(1-b)` decode
  steps — **1.5** at 20% budget.
- Zero if eviction is decided during prefill: survivors are then the only KV ever
  written, so prefill writeback shrinks too (**-859 MB** at 32K/20%).
- **So the question is "evict before writeback or compact later?"** — the former
  strictly dominates. Not "can I afford to compact?"

---

## 5. KV channel pruning (ThinK)

![Channel-pruning breakdown](analysis/channel_prune_breakdown/channel_prune_breakdown.png)

**Model.** Decode reads a cache narrowed along `head_dim` — `head_dim -> d_ret`,
Key path, Value path or both. Assumes *materialized* pruning: survivors stored
packed, so the datapath sees a dense narrower tensor, never a sparse one.
Selection is static after prefill; its cost is excluded.

At 32K, batch 1, λ=0.4 (77 of 128 channels retained):

| | Prefill (LUT_WS) | Decode (LUT_OS_V) |
|---|---:|---:|
| `qk` cycles | 1.00x | **1.40x** |
| `attn_v` cycles | 1.00x | 1.00x |
| `attn_v` occupancy | 99.9% -> 60.1% | 3.12% -> 1.88% |

- **`attn_v` is exactly flat.** `head_dim` is its *output* dim N and
  `n_tiles = ceil(N/128) = 1` for all N ≤ 128, so pruning never crosses a tile
  boundary. Prefill is a null on both axes: `LUT_WS` tiles the reduction into
  `array_m x MU` = 128 elements, and `head_dim` is exactly one.
- **Only decode `qk` shrinks**, where `head_dim` is the reduction dim and enters
  `per_round` as `k_eff = ceil(K/4)`. It is 4.1% of decode, so 1.40x on it is 1.2%
  of the phase — the axis that saves cycles is the stage that costs nothing.
- **The cost is occupancy**, against a different denominator per phase: a 128-wide
  `LUT_WS` tile, versus OS-V's 4096 lanes of which `head_dim` was the most ever
  live. Head packing cannot fill the rest — §IV-D broadcasts one LGU's LUT to all
  rows, and two heads need two different LUTs.

**The only saving is DRAM.** ThinK-K decode roofline speedup at 77 channels:

| batch \ context | 2K | 8K | 32K |
|---|---:|---:|---:|
| 1 | 1.005x | 1.015x | 1.033x |
| 8 | 1.018x | 1.035x | 1.048x |
| 32 | 1.035x | 1.047x | **1.052x** |

- **How.** Roofline time is `sum of max(cycles/freq, dram_bytes/BW)`; pruning moves
  only `qk`'s memory term, and `qk` stays memory-bound throughout. So the saving is
  exactly `delta K bytes / 51.2 GB/s` — 133.75 ms measured against 133.69 ms
  predicted at batch 32 / 32K. A bytes effect, nothing LUT-specific.
- **Not derivable from the KV-DRAM share.** K is 41.5% of decode DRAM there, which
  predicts 1.199x; `attn_v` compute is 79.8% of the phase and untouchable. §3's
  warning running the other way.
- **ThinK-V is inert on every axis modelled** — `attn_v` is compute-bound under KV4,
  so its Value bytes were already hidden under compute. The byte saving is a
  *capacity* result this simulator cannot cash (see TODO).
- **This was flagged unverified, and §9/§15 verified it — against ThinK.** At the
  time DRAM was one flat bandwidth number, so a packed 77-channel cache and a
  strided read of 77 of 128 looked identical, and the model assumed the former.
  Once granularity is charged, **the assumption turns out to be load-bearing**:
  the table above holds only if the retained channels are contiguous *and*
  compacted. Under an unstructured mask the DRAM saving — the only saving ThinK
  has — is **exactly zero** (§15). Compaction itself pays back in 4 decode tokens
  of 256, or 0 if fused into the prefill writeback.

---

## 6. The gate, and what it caught

- §1–§5's TODO asked for DRAM and SRAM models. Those edit
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

## 7. SRAM capacity — what actually fits

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
- ~~**Prefill is excluded, and this is a model gap not a result.**~~
  **Closed by §20.** `_calculate_peak_sram` held the entire prefill activation
  matrix — O(seq x d_model), 2.1 GB at 32K — so prefill overflowed at every
  plausible capacity and its spill charge was a meaningless constant.
  `hw.sram_m_tile` blocks the row loop, and the table this section could not
  publish is in §20(b).

---

## 8. Batch as a capacity axis

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

## 9. DRAM burst granularity

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
- **The one misaligned shape is ThinK's pruned entry.** §5
  computed ThinK's speedup as K-cache bytes / 51.2 GB/s, which assumes every
  saved byte is a saved transfer. At 38 B per entry that is not true, so a
  compacted ThinK cache plausibly returns part of its saving to burst rounding.
  Quantifying it needs the pruned-entry layout pinned down. **Open.**
- `dram_power_model.dram_energy` already models 1024 B rows and 8 B bursts
  internally. That is a different question — what fetching costs, versus which
  bytes get fetched — so the effective count feeds it rather than competing.

---

## 10. Select-without-evict (Quest / TidalDecode / NSA)

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

## 11. KV residency across decode steps

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
  a size a buffer can hold (§8), and residency then removes what is left.
- Steady-state model: the one-time buffer fill after prefill is not charged,
  which slightly favours residency — under 0.4% of KV traffic over 256 output
  tokens.

---

## 12. Where KV reduction actually pays

![KV reduction vs batch](analysis/memory/kv_batch.png)

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
  columns instead of saving cycles. **§5's conclusion stands, for the
  reason it gave.** More batch cannot fix it.

---

## 13. The recurring pattern

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

## 14. OS-V array packing — the one axis that is not memory

![OS-V packing](analysis/memory/packing.png)

`attn_v` decode is issued as `(M=1, K=kv_len, N=head_dim=128)`, so
`n_tiles = ceil(128/128) = 1` and `rounds = ceil(1/32) = 1`: **one of 32 PE rows
does work, at any context length**, and cycles scale exactly linearly with
`batch x heads`. `PackedOSVSimulator` (`analysis/array_packing/`) packs `P`
instances into one pass, each with its own LGU driving `array_m/P` rows.

Verified against `OMNI_LUT.pdf` §IV-C/§IV-D: the LUT is generated from the
**activation** (query / attention scores), not the KV cache, so packed instances
need `P` distinct LUTs and `P` ungated LGUs. What they share is the K/V
bit-plane stream — the "weight" operand — which is what output-stationary
already shares across rows. OS-V gates 31 of 32 LGUs *precisely because* `M=1`;
packing generalises that broadcast.

- **`attn_v` recovers exactly 32×** at every context, occupancy 3.12% → 99.9%.
- **`qk` has two different mechanisms, and only one is what you'd guess.** Below
  `kv_len = array_m x array_n x NUM_RAC = 4096` rows genuinely sit idle. At or
  above it the body is full and the only waste is the **tail**:
  `rounds = ceil(n_tiles/32)` rounds up to whole 32-row passes, leaving up to 31
  rows idle in the last one. Packing subdivides the array into a finer quantum
  and recovers exactly `32·ceil(n_tiles/32)/n_tiles` — 1.94× just past a tile
  boundary, 1.12× at 32K, decaying as `1/n_tiles`. It is neutral **only** when
  `n_tiles` is an exact multiple of 32, and decode `kv_len = context + token_idx`
  almost never is.

**32× on the stage is not 32× on the token.** `attn_v` was compute-bound, so
packing drives its compute time under its memory time and the stage flips to
memory-bound. Decode TPOT at 32K context:

| P | batch 1 | batch 8 | batch 32 |
|---|---:|---:|---:|
| 2 | 1.353x | 1.604x | 1.701x |
| 4 | 1.643x | 2.297x | 2.617x |
| **8** | **1.755x** | **2.637x** | **3.118x** |
| 16 / 32 | 1.755x | 2.637x | 3.118x |

**The ceiling arrives at P=8, and P beyond that buys literally nothing** — the
remaining cost is DRAM.

**Which is what makes it affordable.** Packing `P` instances means `P` working
sets resident. A GQA group shares its K/V tile; past the group size (4 here) the
tiles are distinct. Decode peak SRAM at 32K, batch 1:

| P | independent | GQA-shared | fits 16 MB? |
|---|---:|---:|:---:|
| 4 | 8.3 MB | 2.3 MB | yes |
| **8** | 16.5 MB | **4.5 MB** | **yes** |
| 32 | 66.0 MB | 18.0 MB | no |

The two tables meet at **P=8, GQA-aware: the full achievable speedup for
4.5 MB.** The 32× cycle figure is both unreachable in latency and unaffordable
in SRAM, and neither fact matters, because nothing above P=8 is worth having.

**What this does not charge for**, computed rather than waved at:

- **Weight-FIFO / KV-SRAM read bandwidth.** One live row consumes
  `MU x array_n x NUM_RAC x kv_bits` = 256 B/cycle ≈ 128 GB/s at 500 MHz. P=8 is
  ~1.0 TB/s; P=32 is 8,192 B/cycle ≈ **4.1 TB/s**. The simulator enforces SRAM
  *capacity* and has no bandwidth term at all, so packing converts an idle-array
  problem into an SRAM-bandwidth problem it cannot bill. This is the first thing
  to check before believing the result.
- **LGU ungating power.** §IV-D gates 31 of 32 LGUs specifically to save power;
  P=32 ungates all of them. Cycles fall 32×, LGU dynamic energy rises up to 32×,
  and the energy model sees neither.
- **Energy neutrality here is an artefact, not a finding.**
  `os_v_energy_model.py:23` charges `n_tiles/array_m` and `omni_energy_model.py`
  divides M==1 OS energy by `array_m` — energy is *already* amortised over all
  32 rows while cycles charge a full round for one. **The two halves of the
  model disagree today, and packing is what would make them agree.**
- P live LUTs plus a P-way broadcast tree; per-instance output routing
  (`OUTPUT_CYCLES` unchanged); and the scheduling tail when `batch x heads < P`.

---

## 15. Unstructured pruning — the layout decides which axis is allowed to work

![Unstructured masks](analysis/memory/unstructured.png)

Every KV result above was measured on a **compacted** retained set: eviction
compacts, ThinK narrows the entry to a solid `d_ret` block, page selection
gathers whole pages. Real masks are irregular. `analysis/memory/unstructured_kv.py`
prices that, and the answer turns on one coincidence: **a 4-bit KV entry is
`128 × 4/8` = 64 B, exactly one DRAM burst.**

An axis is free if and only if its mask cuts on a boundary already burst-aligned
in the chosen layout — and the two axes disagree about which layout that is:

| layout | token-wise mask | channel-wise mask |
|---|---|---|
| **token-major** (today's model) | cuts *between* entries → **100% of the saving kept** | cuts *inside* one → **0.0% kept** |
| **channel-major** (transposed) | **0.0% kept** | **99.9% kept** |

- **Perfectly antisymmetric, and there is no third option.** A KV element has
  two indices; one is the minor axis and the other is strided. **Choosing a KV
  layout is choosing which pruning axis is permitted to work at all** — a
  decision taken in the memory subsystem that silently determines which pruning
  papers are deployable.
- **It is a cliff, not a slope.** In token-major, channel groups of 1, 2, 4, 8,
  16, 32 and 64 all keep **exactly 0%** of the saving. 64 contiguous channels of
  128 is worth precisely as much as one: nothing. Only the full 128-channel
  entry pays. There is no partial credit for a partly-structured mask.
- **The saving goes to zero, not negative.** A gathering reader never pays more
  than streaming the whole covering region, so the model clamps there
  (`_dram_effective_bytes(cap_bytes=...)`). Unstructured channel pruning
  degrades to *precisely* dense: 50% of channels removed, 1.000× the traffic.
- **This makes channel pruning null on both axes at once.** §5
  showed it does not move `attn_v` cycles (no N term in the `LUT_OS_V` round);
  this shows an unstructured mask does not move bytes either. Measured decode
  TPOT speedup is **1.000× at batch 1 and at batch 32**.
- **Head-wise is the only axis free in both layouts** — a head is its own
  address region — and it is exactly linear (2.00× at half the KV heads). It is
  also the axis the pruning literature uses least.
- **Composition breaks at the third axis.** head+token at 50% each reaches
  0.250× traffic and 1.504× TPOT; adding an unstructured channel mask removes a
  further 50% *logically* and moves effective traffic **not at all**. The same
  mask with contiguous 128-channel groups reaches 0.125× and 1.528×.
- **The mask itself is not free.** A per-(token, channel) bitmap is 1 bit
  against a 4-bit datum — **25% of the dense cache**, charged over the full
  context whether or not the element survived (0.250× → 0.375×). A per-head
  static mask is negligible and is what a deployable design would use.

Two costs are modelled optimistically — gather scheduling is free (no
request-queue or MSHR pressure, though a scattered gather has far more
outstanding requests than a stream), and the mask is assumed resident when the
gather issues. **Both push the same way: unstructured masks are worse than this
says, not better.** Accuracy is out of scope throughout; this prices the choice,
it does not dispute it.

Full tables: `analysis/memory/unstructured_report.md`.

---

## 16. Memory technology, and the throughput terms that were never billed

![Memory technology](analysis/memory/memory_tech.png)

`dram_bandwidth_gbps` and `dram_burst_bytes` were independent knobs, so a sweep
could describe a part that does not exist. A technology fixes both.
`simulator/memory_tech.py` holds presets with their derivations — and the first
thing it settles is that **`DDR5-6400` *is* the simulator's default** (51.2 GB/s,
64 B), asserted rather than assumed. **Every number in §1–§15 is a DDR5-6400
result**, whether or not it said so.

### (a) The burst matters more than the bandwidth

| technology | bandwidth | burst | channel group needed to collect any saving |
|---|---:|---:|---:|
| DDR5-6400 | 51.2 GB/s | 64 B | **128** (the whole entry) |
| HBM3 | 819.2 GB/s | 32 B | **64** |

- **The cliff sits at one burst, so halving the burst halves the required
  group.** A channel mask must assemble one whole burst of contiguous retained
  data before it collects anything. On HBM that is 64 channels; on DDR5 it is
  all 128. §15's "channel pruning is worthless unstructured" is therefore a
  DDR5 statement — **HBM makes half-entry channel groups viable.**
- This is bought in the memory subsystem, not in the pruning algorithm.

### (b) 16× the bandwidth buys 1.10× of decode

| technology | dense TPOT | vs DDR5 | prune 50% tokens | prune 90% tokens |
|---|---:|---:|---:|---:|
| DDR5-6400 | 2,568 ms | 1.000× | 1.936× | 7.708× |
| HBM2E | 2,332 ms | 1.101× | 1.921× | 7.506× |
| HBM3 | 2,332 ms | 1.101× | 1.921× | 7.506× |

- **No memory technology rescues decode**, which is §13's compute-bound result
  arriving from a new direction. HBM2E and HBM3 are indistinguishable: once the
  DRAM roof clears the compute roof, more bandwidth is inert.
- **The corollary inverts the usual expectation.** A KV technique is supposed to
  be worth most where bandwidth is scarcest, so HBM should devalue it. It does
  not — 1.936× on DDR5, 1.921× on HBM3. Token pruning cuts `kv_len`, the `K` of
  both attention GEMMs, so it removes **cycles** as well as bytes. **Its value is
  portable across memory technologies precisely because it was never really a
  bandwidth optimisation.**

### (c) SRAM bandwidth — built, wired everywhere, and shipped inert

`hw.sram_bandwidth_gbps` (0 = unlimited, the default) adds the third roofline
term `max(compute, DRAM, SRAM)` at all six sites through one shared
`_op_roofline_time`, so no site can silently miss it.

| SRAM bandwidth | TTFT | vs unlimited | TPOT | vs unlimited |
|---|---:|---:|---:|---:|
| unlimited | 219.3 s | 1.00× | 127.49 ms | 1.000× |
| 128 GB/s | 953.5 s | **4.35×** | 136.94 ms | **1.074×** |
| 512 GB/s | 255.6 s | 1.17× | 127.49 ms | 1.000× |

- **Decode is not SRAM-throughput-limited** — TPOT moves 1.074× even at the
  geometry-implied 128 GB/s. This is trustworthy: `M=1` makes decode
  tiling-inert. **It settles the open question §14 left hanging.**
- **Prefill's 4.35× is not a hardware result and must not be quoted as one.**
  Prefill charges 113,670 GB of SRAM traffic against 3 GB of DRAM.
- **128 GB/s is itself an over-charge**: it is one *operand port*, while
  `sram_read` lumps A-reads, B-reads and C accumulator traffic together.

> **Corrected by §19.** This section attributed the 4.35× to the
> untiled-activation defect of §7 and parked prefill behind prefill tiling.
> **That attribution was wrong.** Decomposing the lump by operand shows the
> activation term is right to 0.1% and the *accumulator* is 73.3% of it —
> a memory the paper's own Fig. 4 draws as a separate block. Prefill is
> unparked at 1.16×, and it was never waiting on tiling. The second bullet
> above was the correct instinct; §19 is what happens when it is measured.

### (d) Can P=8 packing actually be fed?

Computed from array geometry rather than from the lumped `sram_read`, which (c)
just showed cannot be trusted:

| packing | KV bytes/cycle | required KV-port bandwidth | verdict |
|---|---:|---:|---|
| P=8 | 2,048 B/cycle | **1.02 TB/s** | plausible (banked SRAM) |
| P=32 | 8,192 B/cycle | 4.10 TB/s | needs a redesign |

- **§14's P=8 survives its own bandwidth check.** P=32 does not — which costs
  nothing, since §14 already showed P=32 buys no TPOT over P=8. **Two independent
  arguments now agree on the same operating point.**

Full tables: `analysis/memory/bandwidth_report.md`.

---

## 17. Compute/memory overlap — the assumption under every latency number

![Compute/memory overlap](analysis/memory/overlap.png)

The roofline sums `max(compute, memory)` **per operation** and never lets one
operation's memory hide behind another's compute. Real hardware double-buffers.
`hw.overlap_model` supplies both extremes — `"serial"` (today's default, and
every number above) and `"pipelined"` = `max(sum compute, sum memory)`. **They
bracket the truth; neither is it.**

Decode time per token, and how much `"serial"` overstates it:

| context | batch | compute | DRAM | serial | pipelined | overstated |
|---|---:|---:|---:|---:|---:|---:|
| 8K | 1 | 20.9 ms | 55.7 ms | 69.62 ms | 55.71 ms | 1.25× |
| **32K** | **1** | **73.3 ms** | **73.4 ms** | **128.80 ms** | **73.40 ms** | **1.75×** |
| 32K | 8 | 644.4 ms | 238.6 ms | 714.01 ms | 644.41 ms | 1.11× |
| 32K | 32 | 2331.5 ms | 804.8 ms | 2609.91 ms | 2331.50 ms | 1.12× |

- **1.75× is larger than most techniques in this document were measured to
  save.** The modelling assumption outweighs the things being modelled.
- **2× is the hard ceiling and 32K/batch 1 nearly reaches it.** `sum(max)` can
  exceed `max(sum)` by at most 2×, attained exactly when the two resources are
  equal — and there they are **73.3 ms against 73.4 ms**. §3 described decode's
  gap narrowing from 6.8× to 1.7× as attention compute growing to meet the
  memory wall; it lands almost exactly on it. **That is the single worst
  operating point for a no-overlap model, and the one this document quotes
  most.**
- **Prefill barely moves** (1.00–1.07×): compute-bound almost everywhere, so
  there is little memory time to hide.

### It corrects §4 and §12 at batch 1

Decode TPOT speedup over dense at 32K, under each model:

| technique | batch 1 serial | batch 1 pipelined | batch 32 serial | batch 32 pipelined |
|---|---:|---:|---:|---:|
| evict 4096 | 2.156× | 1.391× | 6.520× | 6.403× |
| evict 1024 | 2.460× | **1.452×** | 15.957× | 14.323× |
| evict 256 | 2.538× | 1.468× | 23.209× | 20.733× |

- **About 40% of eviction's batch-1 speedup was the assumption.** Once DRAM
  hides under compute, cutting DRAM further buys nothing — the compute roof
  (73 ms) does not move, and eviction at batch 1 is a pure traffic technique.
- **The three budgets converge** — 1.391× / 1.452× / 1.468× — because all are
  pressed against that same roof. Under `"serial"` they still look separable
  (2.156× / 2.460× / 2.538×), **a distinction the assumption manufactures.**
- **At batch 32 they survive** (15.957× → 14.323×) because there eviction is not
  only a traffic technique: `kv_len` is `attn_v`'s reduction dimension, so it
  lowers the compute roof too. **A technique that moves both roofs is robust to
  how they are combined; one that moves only the slack roof is not.**
- **This sharpens §12 rather than contradicting it.** §12 said entry-count
  techniques were measured in the wrong place because batch 1 gives them a tenth
  of decode traffic to attack. This adds a second, independent reason:
  even the traffic they *do* remove was charged as if none could overlap.
  **Rankings by batch are unchanged; batch-1 magnitudes are upper bounds twice
  over.**

**What neither model captures.** `"pipelined"` assumes buffering it never checks
for — two operand sets resident, which `sram_capacity_kb` would have to allow.
Dependencies are ignored, so the reachable overlap is across *layers* and decode
steps, not within one layer's `qk → softmax → attn_v` chain, and pipeline fill
and drain are uncharged. Non-GEMM work is outside both models entirely, so the
VPU softmax — 27.9% of prefill cycles at 32K (§2) — is absent from both columns.

**Corrected by §18.** The `32K / 8` row above was measured under the OS-V
round-count defect §18 found: decode compute there is **582.9 ms**, not 644.4,
and `"serial"` overstates by **1.17×**, not 1.11×. The batch 1 and batch 32 rows
are unaffected — batch 1 always used the correct `M == 1` branch and at `M = 32`
the two round models agree, so only the middle of the batch axis moves. The
sharpest result here, 1.75× at 32K / batch 1, is untouched. Re-run with
`overlap_run.py --rounds-model packed`.

Full tables: `analysis/memory/overlap_report.md`.

---

## 18. The OS-V round count, and the regime map it was blocking

Asking "what is decode bound by, as a function of (batch, context)?" — the prior
question every section above skipped — turned up a defect in the cycle model
first. `_calculate_cycles` counted `LUT_OS_V` output-stationary rounds as
`ceil(M/array_m) x n_tiles`, and **`ceil(M/32)` is 1 for every `M` in 1..32, so
`M` vanished from the round count entirely.**

The array holds `array_m x (array_n x NUM_RAC)` accumulators, so a round retires
`array_m` accumulator tiles wherever they come from. The budget allows
`ceil(M x n_tiles / array_m)`. The two disagree by `array_m / M`:

| M | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| charged | 1 | 32 | 32 | 32 | 32 | 32 |
| allowed | 1 | 2 | 4 | 8 | 16 | 32 |
| **overcharge** | 1× | **16×** | 8× | 4× | 2× | 1× |

- **The `M == 1` branch was never a special case.** `ceil(n_tiles/array_m)` *is*
  `ceil(1 x n_tiles / array_m)` — it is the one place the general formula was
  written down, which is why batch 1 was right and nothing else was.
- Decode issues AW projections with `M = batch`, so `q_proj` cycles jumped
  **32.96×** from batch 1 to batch 2 for a 2× workload and were then **flat to
  batch 32** — the same compute charged for 2 sequences as for 32.
- `hw.os_rounds_model` (default `"tiled"`, inert) supplies the fix as
  `"packed"`. Scope is `LUT_OS_V` only: `LUT_OS` carries the same accumulator
  argument but not the same evidence, and widening it would move prefill on
  first principles alone.
- **Blast radius, asserted not assumed.** Decode `qk` and `attn_v` are issued
  with `M = 1` and are bit-identical under both models, so **every KV result in
  §4–§15 is untouched**. So are `LUT_OS`, `LUT_WS`, `FPE_OS` and `TENDER`. Only
  §17's C/D split moves, and only in the middle of its batch axis.

### The regime map

Decode compute / DRAM, corrected. Below 1.0 the array waits on memory:

| batch | 2K | 4K | 8K | 16K | 32K |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.15 | 0.23 | 0.38 | 0.62 | 1.00 |
| 2 | 0.28 | 0.43 | 0.67 | **1.04** | 1.51 |
| 4 | 0.52 | 0.76 | **1.12** | 1.56 | 2.02 |
| 8 | 0.93 | **1.28** | 1.69 | 2.10 | 2.44 |
| 16 | **1.57** | 1.93 | 2.27 | 2.54 | 2.73 |
| 32 | 2.37 | 2.60 | 2.74 | 2.84 | 2.90 |

- **The memory-bound region is a triangle, not a row.** First compute-bound
  batch: **16 at 2K, 8 at 4K, 4 at 8K, 2 at 16K and 32K.** Batch amortises
  constant weight traffic (7.65 GB, read once); context grows attention compute
  quadratically. Both axes push the same way.
- **§3's "decode is DRAM-bound" is the batch-1 row and only that row** — a
  batch-1 statement this document never restated and every later section
  inherited.
- **The uncorrected map had the whole grid except one row in the wrong regime**,
  claiming C/D 1.93 at 2K/batch 8 against a true 0.93.

### What any lever could possibly buy

Speedup if a whole resource became free, computed by re-running the roofline
with one term suppressed. `KV bytes` bounds **every KV technique in this
document at once** — eviction, selection, residency, channel pruning:

| batch | ctx | packing | overlap | KV bytes | weight bytes |
|---:|---:|---:|---:|---:|---:|
| 1 | 2K | 1.07× | 1.07× | **1.01×** | **6.80×** |
| 1 | 32K | 1.75× | 1.75× | **1.07×** | 1.57× |
| 32 | 2K | 1.88× | 1.05× | 1.05× | 1.00× |
| 32 | 32K | **3.12×** | 1.12× | 1.12× | 1.00× |

- **Removing *all* KV traffic at batch 1 buys 1.01× at 2K and 1.07× at 32K.**
  That is an upper bound on the entire KV literature at batch 1, independent of
  algorithm — and batch 1 is exactly where §1–§3 and §5 did their measuring.
  **It explains §5, §10, §11 and §13's negative results in one line: they were
  aimed at a resource that was not the bottleneck.**
- **The lever in that corner is weight traffic, and it is worth 6.80×** — the
  86%-idle figure of §3 restated as a ceiling. Every byte worth removing at
  batch 1 / 2K is a weight byte, which is what batching already attacks and what
  quantisation below W4 would attack directly.
- **Outside the triangle, packing is the largest ceiling everywhere**, up to
  3.12×, because the compute-bound regime has one dominant operation running at
  3.12% occupancy (§14). Weight bytes go to exactly 1.00× there — fully
  amortised, nothing left to win.
- **Packing and overlap coincide at batch 1** (1.07× and 1.75×) because both
  press against the same DRAM floor: cutting `attn_v` compute 32× and hiding
  memory under compute reach the identical bound from opposite directions.
- **Two accelerators, not one.** Inside the triangle the lever is weight bytes;
  outside it, array occupancy. This document quotes numbers from both sides
  without ever saying the boundary exists.

These are ceilings, not achievable speedups, and they do not compose — each
suppresses one resource while holding the others. They rank families at a point,
which is what "what is worth researching here" needs, and nothing more.

Full tables: `analysis/memory/rounds_report.md`, `analysis/memory/regime_report.md`.

---

## 19. Per-port SRAM — the accumulator was billed to the wrong memory

![Per-port SRAM](analysis/memory/ports.png)

§16(c) shipped `sram_bandwidth_gbps` inert because 128 GB/s took prefill TTFT
to **4.35×**, and read that as the untiled-activation defect surfacing through
a new term. The fix it named was prefill tiling. **Decomposing the lump by
operand shows it was neither.**

Prefill, batch 1, 32K context, by which memory moves the bytes:

| port | bytes | B/cycle | share of lump |
|---|---:|---:|---:|
| activation (A read) | 28,038 GB | **255.7** | 23.0% |
| weight (B read) | 6.9 GB | 0.1 | 0.0% |
| **accumulator (partial sums)** | **89,473 GB** | **816.0** | **73.3%** |
| activation (result write) | 4,535 GB | 41.4 | 3.7% |

- **The activation traffic was never wrong.** The array consumes
  `array_m × MU × act_bits/8` = **256 B/cycle**; the model charges **255.7**.
  A 0.1% match to the port it is being compared against — and tiling A cannot
  reduce a term that is already exactly the array's own consumption rate.
- **The accumulator is 73.3% of the lump.** `LUT_WS` walks `K` in `k_tiles`
  passes and the bit-planes in `qbit` more, recirculating a full 32-bit
  partial-sum matrix on all but the last: `k_tiles × qbit` = **128 round trips**
  per prefill GEMM.
- **And those bytes never cross an SRAM port.** OMNI_LUT.pdf Fig. 4 draws three
  memories — *Unified Buffer*, *Weight Buffer*, *Accumulator* — with the
  accumulator wired to the PE array's partial-sum outputs. `"lumped"` made it a
  client of the unified buffer, which is the entire 4.35×.

`hw.sram_port_model` (default `"lumped"`, inert) bills the three concurrently
and takes the max. TTFT at the geometry-implied bandwidth, per port:

| per-port BW | TTFT lumped | TTFT ported | TPOT ported |
|---|---:|---:|---:|
| unlimited | 219.3 s | 219.3 s | 127.49 ms |
| **128 GB/s** | **953.5 s (4.35×)** | **254.5 s (1.16×)** | 133.79 ms (1.049×) |
| 256 GB/s | 476.8 s (2.17×) | 219.3 s (1.00×) | 127.49 ms (1.000×) |
| 512 GB/s | 255.6 s (1.17×) | 219.3 s (1.00×) | 127.49 ms (1.000×) |

- **Prefill is unparked.** 1.16× at 128 GB/s per port is a bandwidth result, and
  the whole SRAM term is now usable in both phases rather than one.
- **A max over a partition is at most the sum of its parts**, so `"ported"` can
  never price above `"lumped"` at equal bandwidth — asserted in pre-flight, not
  hoped for.
- **Decode does not move at all, and could not have.** `LUT_OS_V` is
  output-stationary, so partial sums stay in the array and the accumulator row
  is exactly **0 bytes**. The K/V *weight* port carries 87.3% of decode's lump —
  which is §3's "decode waits on weights" arriving from the memory side.
- **The accumulator's own constraint is real and stays stated**: 816 B/cycle
  needs ~**408 GB/s**. That is a design requirement on that block, and nothing
  about it belongs in the unified buffer's budget.

Full tables: `analysis/memory/ports_report.md`.

---

## 20. Prefill row tiling — the one outright bug

![Prefill row tiling](analysis/memory/tiling.png)

`_calculate_peak_sram` held the whole activation matrix, `A_bytes = M x K x
act_bits/8` with `M` the entire prefill sequence. So prefill's claimed working
set was **2.16 GB at 32K**, no capacity fitted it, its spill charge was a
constant, and §7 could publish only a decode table.

**It is a capacity bug and only a capacity bug** — §19 is what settled that.
§16(c) had suspected the same untiled A of corrupting the SRAM *traffic* terms
too, and parked prefill's bandwidth numbers behind this fix; the activation term
turned out to measure 255.7 B/cycle against an array that consumes 256. This
section inherited a smaller problem than it was handed.

`hw.sram_m_tile` (0 = untiled, the default) blocks the row loop.

### (a) The frontier

| row block | prefill peak | smaller by | TTFT | vs untiled |
|---|---:|---:|---:|---:|
| untiled | 2.16 GB | 1× | 219.3 s | 1.000× |
| 2,048 | 129.0 MB | 16× | 223.2 s | 1.018× |
| **512** | **32.3 MB** | **64×** | **235.7 s** | **1.075×** |
| 128 | 8.1 MB | 256× | 285.8 s | 1.303× |
| 32 | 2.0 MB | 1,020× | 486.0 s | 2.216× |
| 8 | 524 KB | 4,033× | 1,286.9 s | 5.868× |

- **The knee is wide, and 512 rows is the operating point**: 64× less on-chip
  memory for 7.5% of TTFT.
- **Tiling is a frontier, not a free fix.** Weight-stationary holds B across the
  whole `M` stream, so a row block re-loads the weight tile and re-pays the
  array's fill/drain once per block. Footprint falls and TTFT rises,
  monotonically in both — asserted, not observed.
- **Below 128 rows the curve turns hard**, because the fill/drain re-paid per
  block starts to dominate the activation stream itself.
- **Output-stationary and FPE pay nothing but the footprint.** Their traffic
  terms already re-read B once per `array_m` (or `FPE_array_size`) row tile,
  which only makes sense if one row tile of A is resident — so for those modes
  the footprint was simply *inconsistent with the loop nest the traffic model
  already described*, and tiling makes the two agree at zero cycles and zero
  bytes. Only `LUT_WS` was ever really untiled.
- **Decode does not move and cannot**: its GEMMs have `M = 1` or `M = batch`.

### (b) §7's prefill row, finally

Largest — so cheapest — row block whose prefill working set fits, at 32K:

| SRAM | largest block that fits | prefill peak | TTFT vs untiled |
|---|---:|---:|---:|
| 1 MB | 8 rows | 0.5 MB | 5.868× |
| 2 MB | 16 rows | 1.0 MB | 3.433× |
| 4 MB | 32 rows | 2.0 MB | 2.216× |
| 8 MB | 64 rows | 4.0 MB | 1.607× |
| 16 MB | 128 rows | 8.1 MB | 1.303× |
| 32 MB | 256 rows | 16.1 MB | 1.151× |

- **Capacity and TTFT now trade smoothly**, where every row of this table was
  "overflow, charge a constant" before.
- **§7's decode floor of 924.5 KB sits underneath all of it**, so a chip sized
  for decode alone pays 5.9× TTFT to run prefill at all. Prefill is what sets
  the SRAM budget on this accelerator, and it could not be said before.

### (c) How the bug scaled with context

| context | untiled peak | 512-row peak | smaller by |
|---|---:|---:|---:|
| 2,048 | 57 MB | 14.3 MB | 4× |
| 8,192 | 228 MB | 14.3 MB | 16× |
| 32,768 | 2,064 MB | 32.3 MB | 64× |

- **The untiled footprint is linear in context and the tiled one nearly flat**,
  so the two diverge without limit — the bug was worst exactly where the
  long-context story this repo is about lives.
- **The tiled column is not quite flat, and the reason matters.** It holds at
  14.3 MB through 8K and rises to 32.3 MB at 32K, because past ~16K the binding
  term stops being the activation block and becomes **attention's KV tile**,
  which grows with `kv_len` and which no row block touches. That is §7's decode
  result arriving in prefill.

**One limitation, stated rather than modelled.** A row block re-reads B from
*SRAM*, not from DRAM. The model has always charged an AW operation's weight
DRAM read exactly once and re-read B from SRAM per row tile, for every mode;
that convention predates this field and §20 keeps it rather than changing it for
the tiled case alone. If the weight matrix does not fit on chip — 29.4 MB for
one LLaMA-3-8B FFN projection — a real machine re-reads it from DRAM per block
too, and the TTFT costs above are optimistic by that amount.

Full tables: `analysis/memory/tiling_report.md`.

---

## 21. FFN activation sparsity — the first lever that moves decode

![FFN activation sparsity](analysis/memory/sparsity.png)

Every technique in §4–§15 aims at the KV cache, and §13 recorded why they fail
alike: decode on this array is compute-bound, so removing bytes buys little.
But §3 said something no section followed up — **decode idles ~85% waiting on
weights**, not on KV — and fc1+fc2 are the largest weight tensors in the model.

A gated FFN drives most of its hidden units to near-zero for any given token.
Skipping unit `j` skips column `j` of FC1 and row `j` of FC2, so **weights**
stop being fetched. TEAL, CATS and Deja Vu all produce such a mask. Three
default-identical hooks carry it (`_ffn_active_neurons`, `_aw_weight_run_bytes`,
`_aw_weight_covering_bytes`); the model is in `analysis/act_sparsity/`.
Selection cost is excluded, as for every technique in §4–§15.

### (a) What it buys, batch 1

| density | TPOT | speedup | decode DRAM | decode cycles |
|---|---:|---:|---:|---:|
| 100% | 69.30 ms | 1.000× | 8.46 GB | 31.4 M |
| 50% | 50.95 ms | 1.360× | 5.64 GB | 29.9 M |
| 25% | 41.77 ms | 1.659× | 4.23 GB | 29.2 M |
| **10%** | **36.27 ms** | **1.911×** | 3.38 GB | 29.0 M |
| 5% | 34.43 ms | 2.013× | 3.10 GB | 28.9 M |

- **1.911× is the largest single-technique decode speedup in this document**,
  against §16(b)'s 1.10× for a 16× bandwidth increase and §13's verdict on the
  KV family.
- **Cycles barely move.** The FFN is a small share of decode *cycles* (§1) and a
  large share of decode *bytes*. **This is a bandwidth technique that works on
  an accelerator where §13 concluded bandwidth techniques do not** — because it
  is the only one aimed at weights rather than KV.
- **It saturates against attention.** 10% → 5% density buys only
  1.911× → 2.013×; once the FFN weights stop dominating, attention's own DRAM
  and compute are the floor.

### (b) The layout question from §15, with the opposite answer

10% density, DDR5-6400, varying how many consecutive units share a decision:

| neuron group | run (neuron-major) | kept | run (model-major) | kept |
|---|---:|---:|---:|---:|
| 1 | 2,048 B | **100.0%** | 0.5 B | 0.0% |
| 4 | 8,192 B | 100.0% | 2.0 B | 0.0% |
| 16 | 32,768 B | 100.0% | 8.0 B | 22.2% |
| 64 | 131,072 B | 100.0% | 32.0 B | 88.9% |

- **Neuron-major is burst-aligned at group 1.** One unit's weights are
  `d_model × weight_bits/8` = 2,048 B — 32 whole bursts — so a fully
  *unstructured* mask collects everything. §15's result for KV was the exact
  opposite.
- **Model-major reproduces §15's cliff precisely**, including its shape: 8 B of
  a 64 B burst keeps 22.2%, 32 B keeps 88.9%.
- **The difference is who chooses the layout, and it is the whole point.** §15's
  requirement was that retained KV channels be contiguous and compacted, which
  an append-only cache written *online* cannot promise. A weight matrix is laid
  out once, *offline*, by the compiler. **The same structural obligation is
  unmeetable in one case and free in the other** — that, not the sparsity
  pattern, is what separates them.

### (c) Where the mask comes from is worth 1.46×

| mask source | sparse matrices | TPOT | speedup |
|---|---|---:|---:|
| FFN input (TEAL, Deja Vu) | FC1 + FC2 | 36.27 ms | 1.911× |
| FC1 output (CATS) | FC2 only | 52.78 ms | 1.313× |

- **FC1 is more than half the win.** You cannot skip the work that produced the
  thing you threshold, so an output-derived mask reaches only FC2.
- The trade is an algorithm question, not a hardware one — and Deja Vu's
  predictor matmul is not charged here.

### (d) Batch destroys it, and §18 says why

10% density. Per-token masks make the fetched weight set the **union** over the
operation's tokens, `1 - (1-d)^M`; `share_mask` brackets that:

| batch | union density | shared-mask bound | per-token (real) |
|---:|---:|---:|---:|
| 1 | 10.0% | 1.911× | **1.911×** |
| 2 | 19.0% | 1.797× | 1.665× |
| 4 | 34.4% | 1.505× | 1.324× |
| 8 | 57.0% | 1.291× | 1.121× |
| 32 | 96.6% | 1.082× | **1.003×** |

- **Two mechanisms compound, and the bracket separates them.** The union
  explains most of the collapse; the remainder is that **decode is already
  compute-bound at batch 32** — exactly §18's regime map — where a technique
  that removes bytes cannot help regardless. §13's recurring pattern, arriving
  for a non-KV technique.
- **Activation sparsity is the mirror image of KV eviction.** §8 showed a KV
  budget *buys* batch; activation sparsity is *spent* by batch. They are
  complementary rather than competing, and a latency-oriented batch-1 serving
  stack is exactly where this one pays.
- **Prefill collects exactly nothing**: `1 - (1-0.1)^8192` is 1.0 to hundreds of
  digits, so every weight column is needed by some token and none can be
  skipped. A shared mask would buy 1.862× — the sparsity is fully available,
  and the workload is what refuses it.

### (e) It settles what decode is bound by

§3 and §18 both said decode is memory-bound at low batch, and both read it off
a roofline `max()` — an accounting statement about which term won. §21 is an
**intervention**: it removes ~60% of decode's DRAM bytes while barely touching
cycles, and asks whether the phase responds.

| context, batch 1 | cycles cut | **TPOT cut** |
|---|---:|---:|
| 2K | 1.268× | **2.521×** |
| 8K | 1.084× | **1.911×** |
| 32K | 1.023× | 1.350× |

- **A compute-bound phase cannot answer a 1.27× cycle reduction with a 2.52×
  latency reduction.** This is the first result in the document that
  establishes what decode waits on causally rather than by construction.
- It agrees with the accounting exactly. Decode compute against DRAM, per token
  at batch 1: **0.15** at 2K, 0.23 at 4K, 0.38 at 8K, 0.64 at 16K, **1.04** at
  32K — memory-bound until 32K, where it crosses. The intervention's payoff
  falls along the same curve.
- **And it is weights, not KV.** At 2K that is 7.67 ms of compute against
  51.12 ms of DRAM — the ~85% idle §3 reported — and §21's lever is the only
  one in this document pointed at it. This is why §4–§16's KV techniques all
  failed and this one did not: they were aimed at the wrong bytes.
- **At batch 32 it inverts, and that is a check rather than a caveat.**
  Compute/DRAM is 2.51–3.23 across every context, and the same lever buys
  1.003×. §16(b)'s "16× the bandwidth buys 1.10×" is a **batch-32** measurement,
  so it never contradicted this — the two describe opposite regimes of §18's map.
- **§19 and §22 close off the alternatives.** Decode is not SRAM-bandwidth-bound
  (1.004–1.049× at the RTL's real port widths) and not capacity-bound below 32K.
  So "decode is memory-bound" now has a precise form: ***DRAM*-bound, on
  *weights*, at low batch, below 32K** — and every other reading has been
  measured and excluded.

Full tables: `analysis/act_sparsity/sparsity_report.md`.

---

## 22. The RTL's four buffers, against the model's one pool

![RTL buffer partition](analysis/memory/buffers.png)

Everything above models on-chip memory as **one pool** any operand may draw
from. §19 split its *bandwidth* three ways and left its *capacity* undivided.
The Omni-LUT system block diagram does neither — four physically separate
SRAMs, fixed sizes, fixed word widths, no trading:

| buffer | geometry | capacity | word |
|---|---|---:|---:|
| input | 1024×2048 b | 256 KB | 256 B |
| scale | 1024×2048 b | 256 KB | 256 B |
| weight | 1024×2048 b ×8 | 2,048 KB | 256 B |
| output | 1024×4096 b | 512 KB | 512 B |
| | | **3,072 KB** | |

`simulator/buffer_tech.py` holds it with its derivation;
`hw.sram_buffer_model = "partitioned"` makes it bind. Depth 1024 is corroborated
independently by the DRAM address map in the same note, whose address field is
`[9:0]`.

### (a) It confirms §19 outright

- **input word** `2048 b / act_bits 16` = 128 elements = `array_m × MU` ✓
- **output word** `4096 b / accum_bits 32` = 128 elements = `array_n × NUM_RAC` ✓

**The input buffer is built exactly one cycle of activation operand wide.** §19
measured the activation port at 255.7 B/cycle against a predicted 256 and
concluded that 128 GB/s had always been an *activation-port* number rather than
an aggregate. It had. Both identities are asserted in pre-flight, not admired.

### (b) Three things a pool could not express

- **Input and output are separate memories at different widths** — 256 B/cycle
  (128 GB/s) and 512 B/cycle (256 GB/s). §19's "unified" port both wrongly
  summed them *and* understated the output side by 2×.
- **The scale buffer has no counterpart in the model.** 256 KB — as large as the
  input buffer, 8.3% of the chip — carrying an operand the RTL gives its own
  load command (`CMD_scale_size`) and DRAM type code (`2'd1`), and which every
  number published before `hw.model_scale_traffic` priced at **zero bytes**.
- **Capacity cannot be traded**, so an operation can fit in 3 MB of total SRAM
  and still not run.

### (c) What adopting the real part costs

| row block | TTFT | vs pool | TPOT | vs pool | prefill overflow |
|---|---:|---:|---:|---:|---|
| untiled | 53.2 s | 1.83× | 69.54 ms | 1.004× | input, output |
| 128 rows | 54.5 s | 1.87× | 69.54 ms | 1.004× | input |
| **32 rows** | **64.3 s** | **2.21×** | 69.54 ms | 1.004× | input |
| 8 rows | 170.3 s | 5.85× | 69.54 ms | 1.004× | — |

- **The input buffer is sized to exactly `array_m` rows**: `256 KB /
  (4096 × 2 B)` = 32 activation rows of a `d_model`-wide operand. **§20
  nominated 512 rows as the operating point; that describes a buffer that was
  never built**, and a 32-row block costs 2.21× rather than 1.075×. But 32 is
  the block for *one* operand shape, not for the model — see the next bullet,
  where the largest globally workable block turns out to be **9**.
- **Decode barely moves (1.004×)**, which is §19's split arriving from the
  hardware side rather than from a measurement: decode is `M=1`, tiling-inert,
  and its traffic is weight-port traffic served by 8 banks at 2,048 B/cycle.
- **The input overflow does not clear at 32 rows, and what binds is not what
  the buffer was sized for.** The A operand is `m_tile × K × act_bits/8`, so the
  row block a 256 KB buffer allows depends entirely on that operation's `K`:

  | operand | its `K` | rows that fit |
  |---|---:|---:|
  | q/k/v/o_proj, fc1 | `d_model` 4,096 | **32** = `array_m` |
  | **fc2** | `d_ffn` 14,336 | **9** |
  | `attn_v` | `kv_len` 2K / 8K / 16K / 32K | 64 / 16 / 8 / 4 |

  **The buffer is sized to exactly `array_m` rows of a `d_model`-wide
  activation** — which serves the projections and fc1, and is plainly
  deliberate. But **the FFN's own second matrix is 3.5× wider in `K`**, so fc2
  gets 9 rows, and it is fc2 that binds at 2K and 8K. Attention takes over only
  past 16K, where `kv_len` exceeds `d_ffn`. Measured: `fc2:input` overflows at
  32 rows for *every* context, and the largest block that fits anywhere is **9**.
- **So one `sram_m_tile` cannot serve the model**, and the conflict is *inside
  the FFN* before it is ever between the FFN and attention. Under a pool it was
  invisible: fc2's 896 KB borrowed silently from the other 2.75 MB.

### (d) Where the part runs out

| context | one K cache | vs weight buffer | decode overflow |
|---|---:|---:|---|
| 2,048 | 128 KB | 0.06× | — |
| 8,192 | 512 KB | 0.25× | — |
| 16,384 | 1,024 KB | 0.50× | — |
| **32,768** | **2,048 KB** | **1.00×** | **scale, weight** |

- **32K is exactly where this part runs out, and the arithmetic is exact:**
  `32768 × 128 × 4 b` = 2,048 KB, and the weight buffer is 2,048 KB. K alone
  fills it; K+V needs twice the chip. **§11's on-chip KV residency tops out near
  16K on this part.**
- §7 said the KV tile becomes binding past ~16K. It was right for a reason it
  could not see: it read that off a pool, and the real boundary is a buffer wall
  at exactly 2 MB.
- **The `scale` overflow is model-derived and wants RTL confirmation.**
  Per-token Value scales at `qbit + 1` FP16 each are 320 KB at 32K against a
  256 KB buffer. The *layout* is inferred from OMNI_LUT.pdf §IV-B (see
  `_scale_operand_bits`), not read from the RTL — **a question to ask, not a
  defect found.**
- **If it is real, no row block escapes it.** The scale footprint scales with
  `K`, not with `m_tile`, so `attn_v:scale` overflows at *every* block from 32
  rows down to 2 at 32K. Tiling is the lever for the input buffer and is no
  lever at all here.

**One RTL number not modelled.** The note's own LSU measurement, 492 → 927 ns
(**1.88×**), says the load path is *not* hidden behind compute. That is evidence
for §17's `"serial"` default and against the `"pipelined"` bound being reachable
— which the 256 KB input buffer independently supports, since it holds one row
tile and has no room for the second one double-buffering needs. It is one
micro-benchmark, so it is recorded rather than turned into a latency term.

Full tables: `analysis/memory/buffers_report.md`.

---

## TODO

### Model gaps

- **Measure the BQU.** Not measured yet — the original simulator does not model
  it at all. `bqu_metrics()` in `cycle_units.py` is a placeholder: BEA one pass
  per bit-plane, TSE one min/max pass (Value path only), assumed `bqu_width`
  elements/cycle. Replace with RTL numbers; treat current rows as
  order-of-magnitude. (`--bqu-width` to tune, `--no-bqu` to drop.)
- **Confirm the BQU is really overlapped.** Excluded from serial latency per
  Sec. IV-A ("on-the-fly"), unverified against the RTL schedule. Only matters if
  the measured BQU is much slower.

### Open questions

- **Confirm the scale operand's layout.** §22(d) predicts the 256 KB scale
  buffer overflows at 32K on the Value path, but `_scale_operand_bits` *derives*
  the size (`K × (qbit + 1) × act_bits`) from OMNI_LUT.pdf §IV-B rather than
  reading it from the RTL. If Values share one α across a token group, or the
  zero point is folded, the term shrinks and the overflow goes away. **This is
  the one place the model now makes a falsifiable claim about the hardware.**
- **A per-operand row block.** §22(c) shows the allowed block is
  `input_buffer / (K × act_bits/8)`, so it differs per operation: 32 rows for
  fc1 and the projections, **9 for fc2**, and 64 down to 4 for `attn_v` as
  context grows. The field is global; the schedule needs it per operation.
- **The LSU / DMA path.** The RTL measures 1.88× from adding the LSU and the
  model has no term for it at all. Not enough data to build one from, but it is
  the largest unmodelled *hardware* cost now that the buffers are in.

- **Charge the mask.** §21 excludes selection cost, as §4–§15 do, but the
  exclusion is heavier here: Deja Vu's predictor is a real matmul per layer and
  CATS' threshold is a real VPU pass over `d_ffn`. §21(c) already shows the two
  differ by 1.46× *before* either is charged, so the ranking between them is
  the thing most likely to move.
- **Weight DRAM re-reads under row blocking.** §20 charges a row block for
  re-reading B from SRAM but not from DRAM, in any mode — the model has always
  assumed an AW weight read happens once. A 29.4 MB FFN weight matrix does not
  stay resident, so §20's TTFT costs are optimistic by whatever that re-read is.

- **Mixed-precision KV.** `qbit` is modelled as static and
  `analysis/bit_width/` sweeps only fixed widths, so per-token / per-channel
  allocation (KIVI / ZipCache / KVQuant), giving a weighted-average effective
  `qbit`, is unexplored — and `qbit` is the one axis that multiplies cycles as
  well as bytes (§13).
  **Bit-plane *skipping* is dead, and should not be attempted.** BCQ bit-planes
  are ±1-valued (§VI-B), so an all-zero plane is not representable and skipping
  is structurally meaningless on this encoding. Reducing `qbit` itself is real;
  eliding planes within a fixed `qbit` is not.
- **Measure the energy side of channel pruning.** Compute energy is charged per
  *tile* (`omni_energy_model.py`), so ThinK-V saves exactly 0 J as modelled while
  ThinK-K saves via `k_eff` — the same K/V asymmetry as cycles. Whether the idle
  RAC columns can actually be power-gated is unmodelled and needs per-column
  characterization; the ceiling is small, since `attn_v` compute is only 2.0% of
  decode energy at batch 1 and 5.0% at batch 32.
- **LGU ungating power under packing.** §14 recovers cycles by ungating all 32
  LGUs; the energy model sees neither that nor the cycle saving, and its two
  halves already disagree (§14's last bullet).

### Closed

- ~~**Tile prefill in `_calculate_peak_sram` — the top item.**~~ Done — §20.
  `hw.sram_m_tile` blocks the row loop, §7's prefill row exists, and the
  frontier has a wide knee (64× less SRAM for 1.075× TTFT). Two things it
  turned out *not* to be: §19 showed it never touched the SRAM traffic terms,
  and only `LUT_WS` was ever untiled — the OS and FPE modes' footprints were
  merely inconsistent with loop nests their own traffic terms already
  described. **Still open underneath it**: weight *DRAM* re-reads per row
  block, which the model does not charge in any mode.

- ~~**No compute/memory overlap.**~~ Both bounds now exist (§17). `"serial"`
  remains the default because `"pipelined"` assumes buffering the model cannot
  verify — but every latency figure in §1–§16 is a `"serial"` figure and is an
  upper bound by 1.00–1.75×, and eviction's **batch-1** speedups specifically
  are inflated ~40%.

- ~~**Build DRAM and SRAM latency models.**~~ Done — §6–§9 and §16.
- ~~**Pin down ThinK's pruned-entry layout.**~~ Answered by §15, and worse than
  the item feared: the risk is not that a 38 B entry is misaligned, it is that
  *any* channel mask below the full 128 is sub-burst in token-major. ThinK's DRAM
  saving requires the retained channels to be **contiguous and compacted**, a
  layout obligation §5 never stated. Unstructured, the saving is exactly zero.
- ~~**Bill the SRAM read bandwidth.**~~ Done in §16(c–d). Decode is not
  SRAM-limited (TPOT 1.074x at 128 GB/s) and P=8's ~1.02 TB/s KV port is
  buildable, so §14 is a design rather than a ceiling.

---

## Appendix A — staged record and revert points

Each memory-model change landed as **one commit** whose *disabled* default
reproduced the previous numbers exactly. A checkpoint line cannot name its own
commit, so each SHA was recorded by the next commit; `git log --oneline` is the
tiebreaker if they ever disagree.

| stage | `HardwareConfig` field | disabled default | § | revert point |
|---|---|---|---|---|
| 0 — regression gate | — | — | §6 | `2342e90` |
| 1 — SRAM capacity | `sram_capacity_kb` | `0` = unlimited | §7 | `9eaa1db` |
| 1b — batch as a capacity axis | `sram_batch_model` | `"sequential"` | §8 | `301f9d5` |
| 2 — DRAM access granularity | `dram_burst_bytes` | `0` = exact bytes | §9 | `569b8ce` |
| 3 — selective attention | *(analysis only)* | — | §10 | `d63e355` |
| 4 — KV residency | `kv_sram_kb` | `0` = no buffer | §11 | `5e3771f` |
| 5 — attention score staging | `score_sram_kb`, `prefill_kv_dram_read` | `0`, `False` | §16 | `00a79b7` |
| 5b — unstructured KV masks | *(three default-identical hooks)* | — | §15 | `40f9071` |
| 7 — memory tech + SRAM bandwidth | `sram_bandwidth_gbps` | `0.0` = unlimited | §16 | `9b0b15c` |
| 10 — compute/memory overlap | `overlap_model` | `"serial"` = no overlap | §17 | `4d593bc` |
| 12 — per-port SRAM | `sram_port_model`, `accum_bandwidth_gbps` | `"lumped"`, `0.0` | §19 | `f9b27f0` |
| 6 — prefill row tiling | `sram_m_tile` | `0` = untiled | §20 | `90dbeb5` |
| 13 — FFN activation sparsity | *(three default-identical hooks)* | — | §21 | `57931d5` |
| 14 — RTL buffer partition | `sram_buffer_model`, `model_scale_traffic`, 9 geometry fields | `"pool"`, `False`, `0` | §22 | `0a78f11` |

```
git revert <sha>            # undo one stage, keep the later ones
git reset --hard <sha>      # rewind to the end of that stage
```

**Stage 6 (prefill tiling) landed out of order**, after Stages 10 and 12, which
is why the table is not in numeric order. It was the top TODO item for the whole
of §1–§18. §16(c) was thought to be blocked on it and was not — Stage 12
unblocked that instead (§19), which is also what shrank Stage 6 to the capacity
fix it always was.

**Two premises died in the building**, recorded here rather than quietly dropped:

- **Stage 2's motivating example was wrong, and Stage 3 disproved it.** The
  premise was that a flat bandwidth model makes 1% selection look ~100x cheaper
  than it is. A 4-bit KV entry is 64 B — exactly one burst — so a page-gathering
  reader is burst-aligned at *every* page size. Building Stage 2 is still what
  made the question answerable, and it did find a genuinely misaligned shape
  (§15), but the example that motivated it was not real.
- **Stage 7 shipped inert against its own plan.** The plan chose
  `sram_bandwidth_gbps = 128.0`; measuring it first is what changed the call, and
  §16(c) states exactly what turning it on costs.

---

## Appendix B — standing checks

Run after *every* model change, not just the one being worked on.

1. `python analysis/regression/baseline.py check` → *Identical to the baseline ✓*
   with all features at their disabled defaults. Re-capture only when a field is
   added, and only after confirming the diff contains nothing but the new `hw.*`
   keys and the per-entry `full_sha256`.
2. `sum(cycle_units(...)) == _calculate_cycles(...)` across the 50-combination
   check — the cycle model is untouched by the memory work, so any movement
   means the wiring leaked.
3. `python analysis/channel_prune_breakdown/think_run.py` — its pre-flight
   assertions pass and the dense baseline still reproduces §3's roofline column
   (55.39 / 70.67 / 131.82 ms).
4. Every sweep's own pre-flight suite still passes: `prefill_run.py` (7),
   `pack_run.py` (9), `unstructured_run.py` (9), `selective_run.py` (5),
   `bandwidth_run.py` (5), `ports_run.py` (7), `tiling_run.py` (8),
   `sparsity_run.py` (8), `buffers_run.py` (9).
