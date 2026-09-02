# Omni-LUT — What We Found

*Presentation cut of `study.md`. Results only.*

**Setup.** LLaMA-3-8B (GQA 32/8) on Omni-LUT-KV4 — 32×4 LUT array, W4A16KV4,
500 MHz, 51.2 GB/s = DDR5-6400. Decode, 32K context unless stated.

---

## The claim

> **Last time: KV bytes are not the critical path, and inside the memory-bound
> triangle the lever is *weight* bytes — worth 6.80×. We identified it and did
> not pull it.**
>
> **This time we pulled it. FFN activation sparsity: 1.911× decode TPOT at
> batch 1** — the largest single-technique decode win in the project, and the
> first that survives contact with the regime map. Along the way the byte model
> had two defects, and the whole thing has now been checked against the RTL.

| technique | cycles | bytes | decode TPOT | verdict |
|---|:--:|:--:|---:|---|
| **FFN activation sparsity** | ~none | **weights, linear** | **1.911× b1 · 1.003× b32** | **NEW — build it, at batch 1** |
| **Array packing** P=8 | **32× on stage** | — | **1.755× b1 · 3.118× b32** | **build it** |
| **Bit-width** (KV4→KV3) | linear | linear | unswept | **only axis that cuts both** |
| **Eviction** (H2O, SnapKV) | linear | linear | 1.45× b1 · **15.96× b32** | works — at batch |
| Select-without-evict (Quest) | linear | linear | 12.85× b32 | **byte-identical to eviction** |
| KV residency (on-chip buffer) | none | −36.8% | 1.06× | energy only |
| **Channel pruning** (ThinK) | **null** | **null** | **1.000×** | **dead** — unless HBM |

**The two techniques that work are complementary, not competing.** Activation
sparsity owns batch 1; packing and eviction own batch 32. Nothing owns both.

---

## 1. Decode is memory-bound in a triangle, not everywhere

Compute vs DRAM time per token (below 1.00 = the array waits on memory):

| batch | 2K | 4K | 8K | 16K | 32K |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.15 | 0.23 | 0.38 | 0.62 | 1.00 |
| 2 | 0.28 | 0.43 | 0.67 | **1.04** | 1.51 |
| 4 | 0.52 | 0.76 | **1.12** | 1.56 | 2.02 |
| 8 | 0.93 | **1.28** | 1.69 | 2.10 | 2.44 |
| 16 | **1.57** | 1.93 | 2.27 | 2.54 | 2.73 |
| 32 | 2.37 | 2.60 | 2.74 | 2.84 | 2.90 |

- **The memory-bound region is a triangle in the low-batch, short-context
  corner.** First compute-bound batch: **16 at 2K, 8 at 4K, 4 at 8K, 2 at
  16K/32K.** Batch amortises constant weight traffic; context grows attention
  compute quadratically. Both axes push the same way.
- **"Decode is DRAM-bound" is the batch-1 row and only that row** — and at
  batch 1 / 2K the bottleneck is **weights, not KV**: 2.6 GB of weights per
  token against 80 MB of KV, array idle 86%.
- **This required fixing a cycle-model defect first** (§9). The uncorrected
  model had the whole grid except one row in the wrong regime.

### What any lever could possibly buy

Speedup if a whole resource became free. `KV bytes` bounds **every KV technique
here at once** — eviction, selection, residency, channel pruning:

| batch | ctx | packing | overlap | KV bytes | weight bytes |
|---:|---:|---:|---:|---:|---:|
| 1 | 2K | 1.07× | 1.07× | **1.01×** | **6.80×** |
| 1 | 32K | 1.75× | 1.75× | **1.07×** | 1.57× |
| 32 | 2K | 1.88× | 1.05× | 1.05× | 1.00× |
| 32 | 32K | **3.12×** | 1.12× | 1.12× | 1.00× |

- **Removing *all* KV traffic at batch 1 buys 1.01× at 2K and 1.07× at 32K.**
  An upper bound on the entire KV literature at batch 1, algorithm-independent —
  and batch 1 is exactly where sections 2–4 did their measuring. **This is the
  one-line explanation for every negative result below.**
- **Inside the triangle the lever is weight bytes (6.80×). Outside it, array
  occupancy (3.12×).** Two accelerators, not one.

## 2. Attention is the only target — and cycles lie about it

![Stage breakdown](analysis/cycle_breakdown/cycle_breakdown_norm.png)

- `attn_v_matmul` alone is **55.5%** of decode cycles at 2K, **88.4%** at 32K.
- **`attn_v` costs far more than `qk`**: in `LUT_OS_V` its N is `head_dim`=128
  while qk's is `kv_len` — attn_v serialises over the cache, qk parallelises.
- **Cycles understate decode latency by up to 6.8×** (8.18 ms compute inside
  55.39 ms roofline at 2K). *Report cycles and roofline together, or not at all.*
- At 32K, VPU **softmax is the single largest prefill stage.** Scaling the LUT
  array would not help — the bottleneck has moved off it.

## 3. The fixed-overhead knee — our one novel result

![Compaction](analysis/compact_breakdown/compact_breakdown.png)

`attn_v` costs `per_round = 3 (LGU) + ceil(kv_len/4) + 5 (fill/drain) + 2 (accum)`.
**The constant 10 does not shrink with the KV budget:**

| Retained entries (32K) | 32768 | 6554 | 656 | 328 | 132 |
|---|---:|---:|---:|---:|---:|
| Fixed share of attn cycles | 1.2% | 1.7% | 9.3% | 14.9% | **23.5%** |

- **1.2% → 23.5%** exactly across the budgets these papers headline
  (PyramidKV 0.7% cache, SnapKV 128 entries).
- 2K/8K/32K curves **collapse onto one line** against *absolute* retained
  entries — an architectural constant, not a workload artefact.
- Invisible on a GPU. Applies to **any** method reducing attention to `k` operands.
- **The published accuracy-vs-budget curves have a cost axis that does not
  transfer to LUT hardware.**

## 4. Channel pruning is null on both axes

![Channel pruning](analysis/channel_prune_breakdown/channel_prune_breakdown.png)

- **Cycles: `attn_v` is exactly 1.00×.** `head_dim` is its *output* dim N, and
  `ceil(N/128) = 1` for all N ≤ 128 — pruning never crosses a tile boundary.
  Only `qk` shrinks, and qk is 4.1% of decode. *The axis that saves cycles is
  the stage that costs nothing.*
- **Bytes: also 1.00×** under an unstructured mask (§6).
- Best case ever measured: **1.052×**, at batch 32 / 32K, DRAM-only.
- More batch cannot fix it — the bytes were never on the critical path.

## 5. The recurring pattern

Channel pruning, select-without-evict and KV residency each removed real DRAM
traffic and each produced little or no speedup, for one reason: **`attn_v` is
compute-bound under a 4-bit KV cache.**

| axis | cycles | DRAM |
|---|---|---|
| channel (`head_dim` = N) | **null** — no N term | linear |
| token (`kv_len` = K) | linear via `k_eff` | linear |
| **bit-width (`qbit`)** | **linear** | **linear** |

- `cycles = batch × per_round × rounds × qbit` — **bit-width is the only axis
  that multiplies cycles as well as bytes**, and it composes with eviction
  rather than competing. **Unmeasured. Highest-value open experiment.**

![KV reduction vs batch](analysis/memory/kv_batch.png)

- **And they were measured in the wrong place.** Weight traffic is constant in
  batch (7.65 GB, read once); KV traffic scales with it. `evict-1024` goes
  **2.46× at batch 1 → 15.96× at batch 32.**

## 6. Layout decides which pruning axis may work

![Unstructured masks](analysis/memory/unstructured.png)

Everything above assumed a **compacted** retained set. Real masks are irregular,
and it turns on one coincidence: **a 4-bit KV entry is `128 × 4/8` = 64 B,
exactly one DRAM burst.**

| layout | token-wise mask | channel-wise mask |
|---|---|---|
| **token-major** (today) | cuts *between* entries → **100% kept** | cuts *inside* one → **0% kept** |
| **channel-major** | **0% kept** | **99.9% kept** |

- **Perfectly antisymmetric, and there is no third option.** An element has two
  indices; one is minor, the other is strided.
- **A cliff, not a slope.** Channel groups of 1, 2, 4, 8, 16, 32 **and 64** all
  keep **exactly 0%**. No partial credit for a partly-structured mask.
- **Head-wise is the only axis free in both layouts** (2.00× at half the KV
  heads) — and the axis the literature uses least.

**Memory technology moves the cliff.**

![Memory technology](analysis/memory/memory_tech.png)

- **The cliff sits at one burst, so halving it halves the group needed.** DDR5
  (64 B) needs all 128 channels; **HBM3 (32 B) needs 64** — "channel pruning is
  worthless" is a *DDR5 statement*.
- **16× the bandwidth buys 1.10× of decode.** Once the DRAM roof clears the
  compute roof, more bandwidth is inert.
- **Token pruning is worth the same on both** (1.936× vs 1.921×) — it cuts
  `kv_len`, the `K` of both attention GEMMs, so it removes **cycles**. *Its
  value is portable precisely because it was never a bandwidth optimisation.*

## 7. Array packing — the only lever that is not memory

![OS-V packing](analysis/memory/packing.png)

`attn_v` decode is `(M=1, K=kv_len, N=128)` — **one of 32 PE rows does work, at
any context.** Packing `P` instances gives each its own LGU driving `array_m/P` rows.

| P | batch 1 | batch 8 | batch 32 |
|---|---:|---:|---:|
| 4 | 1.643× | 2.297× | 2.617× |
| **8** | **1.755×** | **2.637×** | **3.118×** |
| 16 / 32 | 1.755× | 2.637× | 3.118× |

- **`attn_v` recovers exactly 32× on the stage** (occupancy 3.12% → 99.9%) — but
  only **1.755× on the token**, because the stage flips to memory-bound.
- **Ceiling at P=8; beyond it buys literally nothing.**
- **And it fits**: GQA-shared peak SRAM at P=8 is **4.5 MB**, and P=8 needs
  **1.02 TB/s** of KV-port reads — plausible. P=32 needs 4.10 TB/s.
  **Two independent arguments land on the same operating point.**

## 8. The assumption under every number above

![Overlap](analysis/memory/overlap.png)

The roofline never lets one operation's memory hide behind another's compute.
Real hardware double-buffers. `"serial"` and `"pipelined"` **bracket the truth.**

| ctx | batch | compute | DRAM | overstated |
|---|---:|---:|---:|---:|
| 8K | 1 | 20.9 ms | 55.7 ms | 1.25× |
| **32K** | **1** | **73.3 ms** | **73.4 ms** | **1.75×** |
| 32K | 32 | 2331.5 ms | 804.8 ms | 1.12× |

- **1.75× is larger than most techniques here were measured to save.**
- **2× is the hard ceiling and 32K/batch 1 nearly reaches it** — `sum(max)`
  beats `max(sum)` by at most 2×, attained exactly when the roofs are equal, and
  there they are **73.3 vs 73.4 ms**.
- **Pipelining pays *least* where the imbalance is worst**: at 2K/batch 1, the
  most memory-bound point in the grid, overlap buys only **1.069×** — there are
  7.7 ms of compute to hide 51.3 ms of DRAM behind. Bounded by
  `1 + min(C,D)/max(C,D)`.

**It corrects our own eviction numbers at batch 1:**

| technique | b1 serial | b1 pipelined | b32 serial | b32 pipelined |
|---|---:|---:|---:|---:|
| evict 4096 | 2.156× | 1.391× | 6.520× | 6.403× |
| evict 1024 | 2.460× | **1.452×** | 15.957× | 14.323× |
| evict 256 | 2.538× | 1.468× | 23.209× | 20.733× |

- **~40% of eviction's batch-1 speedup was the assumption.**
- **The three budgets converge** (1.391 / 1.452 / 1.468) — under `"serial"` they
  look separable, **a distinction the assumption manufactures.**
- **At batch 32 they survive**, because there eviction also lowers the compute
  roof: `kv_len` is `attn_v`'s reduction dim. *A technique that moves both roofs
  is robust to how they are combined; one that moves only the slack roof is not.*

---

## 9. The defect the regime map found

`_calculate_cycles` counted OS-V rounds as `ceil(M/array_m) × n_tiles` — and
`ceil(M/32)` is **1 for every M in 1..32**, so `M` vanished from the round count.

| M | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| charged | 1 | 32 | 32 | 32 | 32 | 32 |
| allowed | 1 | 2 | 4 | 8 | 16 | 32 |
| **overcharge** | 1× | **16×** | 8× | 4× | 2× | 1× |

- Decode issues projections with `M = batch`, so `q_proj` jumped **32.96×** from
  batch 1 to batch 2 for a 2× workload, then sat **flat to batch 32** — the same
  compute charged for 2 sequences as for 32.
- **The `M == 1` branch was never a special case.** `ceil(n_tiles/array_m)` *is*
  `ceil(1 × n_tiles / array_m)` — the one place the general formula was written
  down, which is why batch 1 was right and nothing else was.
- **Blast radius, asserted not assumed:** decode `qk` and `attn_v` are `M = 1`
  and bit-identical under both models, so **every KV result above is untouched**.
  Only the middle of the batch axis moves.
- Shipped inert as `hw.os_rounds_model`; the baseline moved zero value keys.


## 10. NEW — we pulled the weight-bytes lever

Every technique in sections 3–7 aims at the **KV cache**. Section 1's own lever
table said the batch-1 target was **weight bytes, 6.80×** — and nothing aimed
there. A gated FFN drives most hidden units near zero per token; skipping unit
`j` skips column `j` of FC1 and row `j` of FC2, so **weights** stop being
fetched. (TEAL / CATS / Deja Vu.)

![FFN activation sparsity](analysis/memory/sparsity.png)

| density | TPOT | speedup | decode DRAM | decode cycles |
|---|---:|---:|---:|---:|
| 100% | 69.30 ms | 1.000× | 8.46 GB | 31.4 M |
| **10%** | **36.27 ms** | **1.911×** | 3.38 GB | 29.0 M |
| 5% | 34.43 ms | 2.013× | 3.10 GB | 28.9 M |

- **1.911× against 16× the DRAM bandwidth buying 1.10×.** It is a *bandwidth*
  technique that works on hardware where section 5 concluded bandwidth
  techniques do not — because it is the only one aimed at weights, not KV.
- **The layout question of section 6, with the opposite answer.** One unit's
  weights are `d_model × weight_bits/8` = **2,048 B — 32 whole bursts**, so a
  fully *unstructured* mask keeps **100%** of its saving. ThinK kept 0%.
  **The difference is who chooses the layout**: a weight layout is fixed
  offline by the compiler, a KV layout is dictated online by an append-only
  cache. Same obligation, unmeetable in one case and free in the other.
- **Where the mask comes from is worth 1.46×.** Input-derived (TEAL, Deja Vu)
  reaches FC1 and FC2 → 1.911×. Output-derived (CATS) cannot skip the work that
  produced its own threshold → 1.313×.
- **Batch spends it: 1.911× → 1.003× at batch 32.** Per-token masks make the
  fetched weight set the *union* over the batch, `1 - (1-d)^M`, and decode is
  already compute-bound there. **Prefill collects exactly nothing.**
- **So it is the mirror image of eviction.** Section 5 showed a KV budget *buys*
  batch; this is *spent* by batch. Run both.

## 11. And it proves what decode is bound by

Sections 1 and 2 said decode is memory-bound at low batch by reading a roofline
`max()` — an accounting statement about which term won. Section 10 is an
**intervention**: remove ~60% of decode's DRAM bytes, hold cycles nearly fixed.

| context, batch 1 | cycles cut | **TPOT cut** |
|---|---:|---:|
| 2K | 1.268× | **2.521×** |
| 8K | 1.084× | **1.911×** |
| 32K | 1.023× | 1.350× |

- **A compute-bound phase cannot answer a 1.27× cycle cut with a 2.52× latency
  cut.** First causal evidence in the project.
- It tracks the accounting exactly (compute/DRAM at batch 1: 0.15 → 1.04 from
  2K to 32K) and **inverts at batch 32** (2.51–3.23, same lever buys 1.003×).
- **The precise form**, with SRAM bandwidth and capacity now measured and
  excluded: ***DRAM*-bound, on *weights*, at low batch, below 32K.**

## 12. Two defects in the byte model, found and fixed

**(a) The accumulator was billed to the wrong memory.** Charging the
geometry-implied 128 GB/s took prefill TTFT to 4.35×, which we had blamed on
untiled activations and parked. Decomposing the lump by operand:

![Per-port SRAM](analysis/memory/ports.png)

- Activations measure **255.7 B/cycle against an array that consumes 256** —
  right all along.
- **73.3% of the lump was accumulator recirculation** — partial sums cycled
  `k_tiles × qbit` = 128 times per prefill GEMM — billed against the
  *activation* port. Fig. 4 wires the accumulator outside the unified buffer.
- **Prefill TTFT 4.35× → 1.16×.** Unparked.

**(b) Prefill held the whole activation matrix** — a 2.16 GB claimed working
set, so prefill overflowed at every capacity and had no capacity table at all.
Row blocking gives a real frontier, and says **prefill, not decode, sets the
SRAM budget**.

## 13. The model, checked against the RTL

![RTL buffer partition](analysis/memory/buffers.png)

Four fixed SRAMs — 256 KB input, 256 KB scale, 2 MB weight, 512 KB output —
not the one flexible pool we had modelled.

**It confirmed the sharpest thing we had inferred.** The input buffer word is
**256 B = `array_m × MU × act_bits/8`**: the buffer is built exactly one cycle
of activation operand wide. We had *measured* 255.7 and argued from that 0.1%
gap that 128 GB/s was an activation-port number. It is.

**And it broke four assumptions the pool had hidden:**

- **The row block is per-operand, and `array_m` is right for only one.**
  32 rows for the projections and fc1 (`d_model`), but **9 for fc2** (`d_ffn`
  is 3.5× wider), and 64 → 4 for `attn_v` as context grows. **Largest globally
  workable block is 9.** The conflict is *inside the FFN*.
- **32K is exactly the wall.** `32768 × 128 × 4 b` = **2,048 KB**; the weight
  buffer is **2,048 KB**. K alone fills it, K+V needs twice the chip — so
  on-chip KV residency tops out near **16K**.
- **A 256 KB scale buffer we charged zero bytes for** — 8.3% of the chip, its
  own load command and DRAM type code.
- **Input and output are separate memories at different widths** (256 vs
  512 B/cycle), so our "unified port" both summed them and understated output 2×.

**One falsifiable prediction.** Derived per-token Value scales are 320 KB at 32K
against a 256 KB buffer — and no row block escapes it, since the scale footprint
scales with `K`, not `m_tile`. **The layout is inferred from the paper, not read
from the RTL.** If Values share an α across a token group, it disappears.

## What to do

- **Ship activation sparsity for batch-1 serving.** 1.911× at 10% density, the
  largest decode win we have, and it needs a **neuron-major weight layout** —
  a build-time decision, free if made and unrecoverable if not.
- **Know which regime you are in first.** Inside the memory-bound triangle
  (low batch, short context) the lever is weight bytes — worth **6.80×**, and
  section 10 now collects 1.911× of it. Outside it, array occupancy — worth
  **3.12×**. Nothing else comes close in either.
- **Stop aiming KV techniques at batch 1.** Removing *all* KV traffic there buys
  **1.01–1.07×**. That is the whole literature's ceiling, and it is where most
  of our own measuring happened.
- **Build P=8 packing.** Largest lever outside the triangle, 3.118× at batch 32,
  fits in 4.5 MB, and survives its own 1.02 TB/s bandwidth check.
- **Prune bit-width, not channels.** The only axis that multiplies cycles and
  bytes, composes with eviction, and is still unmeasured.
- **Pick the KV layout before the pruning algorithm.** It silently decides which
  pruning literature is deployable at all.

- **Size the row block per operand, not globally.** One `sram_m_tile` cannot
  serve fc1 (32 rows), fc2 (9) and `attn_v` (4 at 32K) at once.
- **Confirm the scale-buffer layout.** It is the one place the model now makes a
  falsifiable claim about the hardware, and it decides whether 32K is reachable.

*Full derivations, model-change record and open gaps: `study.md`
(sections 19–22 are this week).*
