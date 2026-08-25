# Omni-LUT — What We Found

*Presentation cut of `study.md`. Results only.*

**Setup.** LLaMA-3-8B (GQA 32/8) on Omni-LUT-KV4 — 32×4 LUT array, W4A16KV4,
500 MHz, 51.2 GB/s = DDR5-6400. Decode, 32K context unless stated.

---

## The claim

> **KV-reduction papers are scored in bytes. On this hardware bytes are usually
> not the critical path — so most published wins do not transfer. What moves
> decode is the array, the memory layout, and the batch you run at.**

| technique | cycles | bytes | decode TPOT | verdict |
|---|:--:|:--:|---:|---|
| **Array packing** P=8 | **32× on stage** | — | **1.755× b1 · 3.118× b32** | **build it** |
| **Bit-width** (KV4→KV3) | linear | linear | unswept | **only axis that cuts both** |
| **Eviction** (H2O, SnapKV) | linear | linear | 1.45× b1 · **15.96× b32** | works — at batch |
| Select-without-evict (Quest) | linear | linear | 12.85× b32 | **byte-identical to eviction** |
| KV residency (on-chip buffer) | none | −36.8% | 1.06× | energy only |
| **Channel pruning** (ThinK) | **null** | **null** | **1.000×** | **dead** — unless HBM |

---

## 1. Decode is memory-bound only at batch 1

Compute vs DRAM time per token (C/D > 1 = compute-bound):

| batch | 2K | 8K | 32K |
|---:|---:|---:|---:|
| **1** | **0.15** | **0.38** | **1.00** |
| 8 | 1.93 | 2.32 | 2.70 |
| 32 | 2.37 | 2.74 | 2.90 |

- **At batch ≥ 8 the array is compute-bound everywhere.** "Decode is DRAM-bound"
  is a **batch-1 statement**.
- **At batch 1 / 2K the bottleneck is weights, not KV** — 2.6 GB of weights per
  token against 80 MB of KV. The array idles **86%** of decode.
- **It self-corrects with context**: attention compute grows quadratically,
  weight traffic is constant, so C/D goes 0.15 → 1.00 by 32K.
- **This sets who each technique is for.** Anything aimed at KV bytes is aimed
  at 2.9% of traffic at batch 1 / 2K.

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

## What to do

- **Build P=8 packing.** Largest lever that is not a memory technique, 3.118× at
  batch 32, and it fits in 4.5 MB.
- **Prune bit-width, not channels.** The only axis that multiplies cycles and
  bytes, composes with eviction, and is unmeasured.
- **Pick the KV layout first.** It silently decides which pruning literature is
  deployable on this chip.
- **Quote batch-32 numbers.** Batch-1 magnitudes are upper bounds twice over —
  wrong share of traffic, and no overlap credited.

*Full derivations, model-change record and open gaps: `study.md`.*
