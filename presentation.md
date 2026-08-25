# Omni-LUT — What We Found

*Presentation cut of `study.md`. Findings only; the full derivations, model
changes and open gaps stay there.*

**Setup.** LLaMA-3-8B (32 layers, GQA 32/8) on Omni-LUT-KV4 — 32×4 LUT array,
W4A16KV4, 500 MHz, 51.2 GB/s = DDR5-6400. Batch 1, 256 output tokens, standard
attention, except where a section sweeps batch.

---

## The claim

> **KV-reduction papers are scored in bytes. On this hardware bytes are usually
> not the critical path — so most published wins do not transfer, and what
> actually moves decode is the array, the memory layout, and the modelling
> assumptions.**

### Five results that carry it

- **A fixed per-round cost of 10 cycles does not shrink with the KV budget.**
  Overhead goes **1.2% → 23.5%** of attention cycles exactly across the budgets
  these papers headline. Invisible on a GPU. *The published accuracy-vs-budget
  curves have a cost axis that does not transfer to LUT hardware.*
- **Channel pruning (ThinK) is null on both axes at once** — no cycles (no N
  term in the `LUT_OS_V` round) and, under an unstructured mask, no bytes
  either. Measured decode TPOT: **1.000×**.
- **Choosing a KV layout is choosing which pruning axis is permitted to work.**
  Token-major and channel-major are perfectly antisymmetric, and there is no
  third option.
- **Array packing is the only lever that is not about memory** — and it caps at
  P=8, **1.755×** on the token, not the 32× it recovers on the stage.
- **The no-overlap assumption is worth up to 1.75×** — larger than most
  techniques here were measured to save.

### Verdict per technique

| technique | cycles | bytes | decode TPOT | holds? |
|---|:--:|:--:|---:|---|
| **Eviction** (H2O, SnapKV, TOVA) | linear | linear | 2.46× b1 · **15.96× b32** | **yes** — but batch-1 is an upper bound twice over |
| **Bit-width** (KV4 → KV3) | linear | linear | not swept | **the only axis that multiplies both** — unexplored |
| **Array packing** (P=8) | **32× on stage** | — | **1.755× b1 · 3.118× b32** | **yes** — ceiling at P=8 |
| **Select-without-evict** (Quest, NSA) | linear | linear | 2.40× b1 · 12.85× b32 | yes, but **byte-identical to eviction** here |
| **KV residency** (on-chip buffer) | none | −36.8% | 1.06× | **energy/capacity lever only** |
| **Channel pruning** (ThinK) | **null** | **null** unstructured | **1.000×** | **no** — unless HBM (see §6) |

---

## 1. Attention is the only target worth aiming at

![Stage breakdown](analysis/cycle_breakdown/cycle_breakdown_norm.png)

| Stage | Prefill 2K | Prefill 32K | Decode/tok 2K | Decode/tok 32K |
|---|---:|---:|---:|---:|
| fc1 + fc2 | 61.4% | 19.6% | 24.1% | 2.6% |
| qk_matmul | 4.4% | 22.4% | 4.2% | 4.1% |
| attn_v_matmul | 4.4% | 22.4% | **55.5%** | **88.4%** |
| softmax (VPU) | 5.4% | **27.9%** | 2.1% | 3.4% |

- **Decode is attention-dominated everywhere** — 60% of cycles at 2K rising to
  93% at 32K; counting softmax, any KV technique is aiming at 62% → 96%.
- **Prefill flips from FFN-bound to attention-bound** with context.
- **`attn_v` costs far more than `qk`**: in `LUT_OS_V` its N is `head_dim`=128
  while qk's is `kv_len`, so attn_v serialises over the cache and qk parallelises.
- **At 32K, VPU softmax alone is the single largest prefill stage.** Scaling the
  LUT array would not help — the bottleneck has moved off it.

## 2. Cycles understate decode by up to 6.8×

| Context | Cycles/tok | Compute | Roofline | Gap |
|---|---:|---:|---:|---:|
| 2K | 4.09 M | 8.18 ms | 55.39 ms | **6.8×** |
| 8K | 10.97 M | 21.94 ms | 70.67 ms | 3.2× |
| 32K | 38.15 M | 76.30 ms | 131.82 ms | 1.7× |

- **Every AW stage is memory-bound in decode.** At 2K the array idles ~85% of
  decode waiting on *weights*, not on KV.
- **KV4 quantisation is what keeps attention compute-bound** — it shrinks
  attention's own traffic until the array, not memory, is the limit.
- **Report cycles and roofline time together, or not at all.**

## 3. Eviction works — and has a knee nobody has priced

![Compaction](analysis/compact_breakdown/compact_breakdown.png)

Ceiling speedup at 20% budget (decode roofline/token):

| batch \ ctx | 2K | 8K | 32K |
|---|---:|---:|---:|
| 1 | 1.08× | 1.30× | 1.98× |
| 8 | 1.34× | 2.13× | 3.44× |
| 32 | 1.99× | 3.40× | **4.43×** |

- Driver is KV's share of decode DRAM: **2.9% → 93.8%** across that grid.
- **Batch 1 / 2K is the worst case and the technique is dead there.**

**The knee — the one novel result.** `attn_v` costs
`per_round = 3 (LGU) + ceil(kv_len/4) + 5 (fill/drain) + 2 (accum)`. The
constant 10 does not shrink with the budget:

| Retained (32K) | 32768 | 6554 | 656 | 328 | 132 |
|---|---:|---:|---:|---:|---:|
| Fixed share of attn cycles | 1.2% | 1.7% | 9.3% | 14.9% | **23.5%** |

- **1.2% → 23.5%** exactly across the budgets these papers headline
  (PyramidKV 0.7% cache, SnapKV 128 entries).
- 2K/8K/32K curves **collapse onto one line** against *absolute* retained
  entries — an architectural constant, not a workload artefact.
- Applies to **any** method reducing attention to `k` operands.

**Compaction cost is settled, not a tradeoff.** Payback is `(1+b)/(1-b)` decode
steps — **1.5** at 20% budget — and **zero** if eviction is decided during
prefill, which also shrinks prefill writeback (−859 MB at 32K/20%).

## 4. Channel pruning is a null

![Channel pruning](analysis/channel_prune_breakdown/channel_prune_breakdown.png)

At 32K, batch 1, 77 of 128 channels retained:

| | Prefill | Decode |
|---|---:|---:|
| `qk` cycles | 1.00× | **1.40×** |
| `attn_v` cycles | 1.00× | **1.00×** |
| `attn_v` occupancy | 99.9% → 60.1% | 3.12% → 1.88% |

- **`attn_v` is exactly flat.** `head_dim` is its *output* dim N, and
  `ceil(N/128) = 1` for all N ≤ 128 — pruning never crosses a tile boundary.
- **Only decode `qk` shrinks**, and qk is 4.1% of decode: 1.40× on it is 1.2% of
  the phase. **The axis that saves cycles is the stage that costs nothing.**
- **The cost is occupancy.** Head packing cannot fill the rest — one LGU's LUT is
  broadcast to all rows, and two heads need two different LUTs.
- Best case DRAM-only saving: **1.005× (b1/2K) → 1.052× (b32/32K).**

## 5. The recurring pattern

Three independent techniques — channel pruning, select-without-evict, KV
residency — each removed real DRAM traffic and each produced little or no
speedup, for one reason: **`attn_v` is compute-bound under a 4-bit KV cache, so
KV bytes are usually not the critical path.**

| axis | cycles | DRAM |
|---|---|---|
| channel (`head_dim` = N) | **null** — no N term | linear |
| token (`kv_len` = K) | linear via `k_eff` | linear |
| **bit-width (`qbit`)** | **linear** | **linear** |

- `cycles = batch × per_round × rounds × qbit`, so **bit-width is the only axis
  that multiplies cycles as well as bytes**, with no null anywhere — and it
  composes with eviction rather than competing. **Unexplored.**

**And they were measured in the wrong place.**

![KV reduction vs batch](analysis/memory/kv_batch.png)

Weight traffic is constant in batch (7.65 GB, read once); KV traffic scales with
it. Decode TPOT at 32K:

| technique | batch 1 | batch 8 | batch 32 |
|---|---:|---:|---:|
| evict 1024 | 2.460× | 6.971× | **15.957×** |
| select 3% | 2.404× | 6.358× | **12.854×** |
| evict 4096 | 2.156× | 4.418× | 6.520× |
| ThinK-K `d=77` | 1.034× | 1.049× | 1.054× |

- **Channel pruning does not recover with batch, and that is a different
  ceiling** — the bytes were never on the critical path.

## 6. Layout decides which pruning axis is allowed to work

![Unstructured masks](analysis/memory/unstructured.png)

Everything above assumed a **compacted** retained set. Real masks are irregular,
and the answer turns on one coincidence: **a 4-bit KV entry is `128 × 4/8` = 64 B,
exactly one DRAM burst.**

| layout | token-wise mask | channel-wise mask |
|---|---|---|
| **token-major** (today) | cuts *between* entries → **100% kept** | cuts *inside* one → **0.0% kept** |
| **channel-major** | **0.0% kept** | **99.9% kept** |

- **Perfectly antisymmetric, and there is no third option.** An element has two
  indices; one is minor, the other is strided.
- **It is a cliff, not a slope.** In token-major, channel groups of 1, 2, 4, 8,
  16, 32 **and 64** all keep **exactly 0%**. No partial credit.
- **The saving goes to zero, not negative** — a gather never costs more than
  streaming the covering region.
- **Head-wise is the only axis free in both layouts** (2.00× at half the KV
  heads) — and the axis the literature uses least.
- **The mask is not free.** A per-(token, channel) bitmap is 1 bit against a
  4-bit datum: **25% of the dense cache**, charged whether or not the element
  survived.

**Memory technology moves the cliff.**

![Memory technology](analysis/memory/memory_tech.png)

| technology | bandwidth | burst | channel group needed to collect anything |
|---|---:|---:|---:|
| DDR5-6400 | 51.2 GB/s | 64 B | **128** (the whole entry) |
| HBM3 | 819.2 GB/s | 32 B | **64** |

- **Halving the burst halves the required group.** "Channel pruning is worthless
  unstructured" is a **DDR5 statement** — HBM makes half-entry groups viable.
- **16× the bandwidth buys 1.10× of decode.** HBM2E and HBM3 are
  indistinguishable: once the DRAM roof clears the compute roof, more bandwidth
  is inert.
- **The corollary inverts expectation.** Token pruning is worth 1.936× on DDR5
  and 1.921× on HBM3 — it cuts `kv_len`, the `K` of both attention GEMMs, so it
  removes **cycles**. **Its value is portable precisely because it was never a
  bandwidth optimisation.**

## 7. Array packing — the one lever that is not memory

![OS-V packing](analysis/memory/packing.png)

`attn_v` decode is `(M=1, K=kv_len, N=128)`, so **one of 32 PE rows does work, at
any context length.** Packing `P` instances into one pass gives each its own LGU
driving `array_m/P` rows.

- **`attn_v` recovers exactly 32×** — occupancy 3.12% → 99.9%.
- **`qk` recovers the tail**, not idle rows: `rounds = ceil(n_tiles/32)` rounds up
  to whole passes. 1.94× just past a tile boundary, 1.12× at 32K.

**But 32× on the stage is not 32× on the token** — packing drives `attn_v`
compute under its memory time and the stage flips to memory-bound:

| P | batch 1 | batch 8 | batch 32 |
|---|---:|---:|---:|
| 2 | 1.353× | 1.604× | 1.701× |
| 4 | 1.643× | 2.297× | 2.617× |
| **8** | **1.755×** | **2.637×** | **3.118×** |
| 16 / 32 | 1.755× | 2.637× | 3.118× |

- **The ceiling arrives at P=8; beyond it buys literally nothing.**
- **Which is what makes it affordable.** GQA-shared peak SRAM at P=8 is
  **4.5 MB** (vs 16.5 MB independent) — the full achievable speedup, on chip.
- **P=8 survives its own bandwidth check** at **1.02 TB/s** of KV-port reads;
  P=32 needs 4.10 TB/s and a redesign. Two independent arguments land on P=8.

**Not charged for:** LGU ungating power (gating 31 of 32 LGUs is a deliberate
power decision), and packing energy neutrality is an artefact — energy is
already amortised over 32 rows while cycles charge a full round for one.

## 8. The assumption under every number above

![Overlap](analysis/memory/overlap.png)

The roofline sums `max(compute, memory)` **per operation** and never lets one
op's memory hide behind another's compute. Real hardware double-buffers.
`"serial"` and `"pipelined"` **bracket the truth; neither is it.**

| ctx | batch | compute | DRAM | serial | pipelined | overstated |
|---|---:|---:|---:|---:|---:|---:|
| 8K | 1 | 20.9 ms | 55.7 ms | 69.62 ms | 55.71 ms | 1.25× |
| **32K** | **1** | **73.3 ms** | **73.4 ms** | **128.80 ms** | **73.40 ms** | **1.75×** |
| 32K | 32 | 2331.5 ms | 804.8 ms | 2609.91 ms | 2331.50 ms | 1.12× |

- **1.75× is larger than most techniques here were measured to save.**
- **2× is the hard ceiling and 32K/batch 1 nearly reaches it** — `sum(max)`
  exceeds `max(sum)` by at most 2×, attained exactly when the roofs are equal,
  and there they are **73.3 ms against 73.4 ms**. *The single worst operating
  point for a no-overlap model, and the one this work quotes most.*

**It corrects the eviction numbers at batch 1:**

| technique | b1 serial | b1 pipelined | b32 serial | b32 pipelined |
|---|---:|---:|---:|---:|
| evict 4096 | 2.156× | 1.391× | 6.520× | 6.403× |
| evict 1024 | 2.460× | **1.452×** | 15.957× | 14.323× |
| evict 256 | 2.538× | 1.468× | 23.209× | 20.733× |

- **~40% of eviction's batch-1 speedup was the assumption.** Once DRAM hides
  under compute, cutting DRAM further buys nothing — the compute roof does not move.
- **The three budgets converge** (1.391 / 1.452 / 1.468). Under `"serial"` they
  look separable — **a distinction the assumption manufactures.**
- **At batch 32 they survive**, because there eviction lowers the compute roof
  too: `kv_len` is `attn_v`'s reduction dimension. **A technique that moves both
  roofs is robust to how they are combined; one that moves only the slack roof
  is not.**

---

## What this says to do next

- **Prune bit-width, not channels.** It is the only axis that multiplies cycles
  and bytes, it composes with eviction, and it is unmeasured here.
- **Pick the KV layout first.** It silently decides which pruning literature is
  deployable on this chip.
- **Build P=8 packing.** Largest single lever that is not a memory technique,
  and it fits in 4.5 MB.
- **Quote batch-32 numbers.** Batch-1 magnitudes are upper bounds twice over —
  wrong share of traffic, and no overlap credited.

*Full derivations, model-change record, standing checks and open gaps:
`study.md`.*
