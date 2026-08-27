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

## 5. The recurring pattern — and the axis that escapes it

Channel pruning, select-without-evict and KV residency each removed real DRAM
traffic and each produced little or no speedup, for one reason: **`attn_v` is
compute-bound under a 4-bit KV cache.** Eviction removed the same kind of
traffic and **did** produce speedup. One table explains both.

A KV cache has exactly three reducible dimensions, and
`cycles = batch × per_round × rounds × qbit` says what each one touches:

| axis | technique | cycles | DRAM | measured |
|---|---|---|---|---:|
| channel (`head_dim` = N) | ThinK | **null** — N enters only via `ceil(N/128)` | linear | **1.000×** |
| token (`kv_len` = K) | **eviction** (H2O, SnapKV) | **linear** via `k_eff` | linear | **1.45× b1 · 15.96× b32** |
| token, read-only | Quest, TidalDecode | linear | linear | 12.85× b32 |
| **bit-width (`qbit`)** | KV3 / KV2 | **linear** — outer multiplier | **linear** | **unmeasured** |

- **The table predicts the successes as well as the failures, which is what
  makes it a model rather than an excuse.** Channel pruning has no cycle term,
  so it is null *before* anything is measured. **Eviction is on the token axis,
  cuts `k_eff`, and therefore removes cycles — so it works.** Same bytes, same
  hardware, opposite outcome, one line of arithmetic apart.
- **The discriminator is never "how many bytes", it is "does it reach `k_eff`
  or `qbit`".** Every technique here that touches only DRAM lands in §1's
  1.01–1.07× ceiling; every one that lowers the compute roof survives.
- **Select-without-evict is the control that proves it.** Quest reads fewer
  blocks but stores the same cache — byte-identical to eviction, and it does
  lower `k_eff`, so it tracks eviction at batch 32 (12.85× vs 15.96×). The gap
  between them is storage, not compute.
- **Bit-width is the only axis with no null anywhere.** `qbit` is an outer
  multiplier, so no `ceil` can absorb it — KV4→KV3 is 0.75× cycles *and* 0.75×
  bytes unconditionally. And since `cycles ∝ k_eff × qbit`, it **multiplies with
  eviction instead of competing with it**: the two axes are orthogonal, so
  evict-1024 at KV3 is roughly their product. **Unmeasured. Highest-value open
  experiment.**

![KV reduction vs batch](analysis/memory/kv_batch.png)

- **And all of it was measured in the wrong place.** Weight traffic is constant
  in batch (7.65 GB, read once); KV traffic scales with it, so KV's *share* of
  DRAM — and the payoff for cutting it — grows with batch. `evict-1024` goes
  **2.46× at batch 1 → 15.96× at batch 32.**
- **Which is why eviction is the exception and still obeys the rule.** At batch
  1 it is worth 1.45× (§8), inside the same ceiling as everything else. Its 16×
  is a *batch-32* result, earned where §1's triangle says bytes finally bind.

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


## 10. GNNs — the knee as a workload, and the one place packing wins

Same array, different workload. A GCN layer is `H' = σ(Â(HW))`: **Combine** is a
dense GEMM, **Aggregate** is the sparse `Â @ ·`. Six standard benchmarks.

**Both halves mapped onto the existing model with no new operation type.**
Aggregate written as a pull is `(M=1, K=deg(v), N=F)` — **the shape decode
`attn_v` is issued as.** Proved, not asserted: a real `attn_v` at
`(kv_len, head_dim) = (deg, F)` returns the identical cycle count over 30 pairs.
So §3's knee is not *analogous* to aggregation's, it **is** aggregation's.

### The knee we engineered toward in §3 is where a citation graph starts

10 cycles per round are fixed; `ceil(deg/4)` is useful:

| deg(v) | cycles | fixed % | vs VPU |
|---:|---:|---:|---:|
| 1 | 176 | **90.9%** | 88.0× worse |
| 4 | 176 | 90.9% | 22.0× |
| 32 | 288 | 55.6% | 4.5× |
| 492 | 2,128 | 7.5% | 2.2× |

- **Cora's mean degree is 3.9 → 90.9% overhead**, against the 23.5% that was the
  most extreme point in the entire KV study. §3 needed a 0.4% KV budget to reach
  that regime; a GNN is born there.
- **Our stage-1 hypothesis was wrong twice, and both are results.** "Crossover
  near degree 50" was a **4-bit statement** — true within 10% at `qbit=4`, false
  on every graph at 16. The real condition is on *width*: a crossover exists iff
  `F > 32 × qbit`. And aggregation is **compute-bound, not memory-bound** —
  0.9–1.2 FLOP/byte was right about the FLOPs, but the LUT does not spend its
  cycles on FLOPs.

### Packing reverses the verdict — on the large graphs only

`M = 1` means 1 of 32 rows works (§7's 3.12% occupancy). Packing `P` destinations
into one pass recovers **exactly `array_m / n_tiles`** — measured `P*` equals the
bound at every width, with no slack, and is **1.00× at F=4096** where it buys
nothing. Packing and the N-null end at the same width, for the same reason.

| graph | avg deg | F_out | P\* | LUT vs VPU |
|:---|---:|---:|---:|---:|
| Reddit | 492.0 | 256 | 16 | **7.35×** |
| ogbn-products | 50.5 | 256 | 16 | **4.35×** |
| ogbn-arxiv | 13.8 | 256 | 16 | **1.94×** |
| Cora / CiteSeer / PubMed | 2.7–4.5 | 3–16 | 32 | 0.02–0.10× |

- **The band's edge moves to `32 × qbit / P`; the crossover degree never moves —
  43 at every P.** Packing divides LUT cycles per node by `P` *and* the
  qualifying width by `P`, so the VPU term falls equally. **Packing widens the
  band without ever making a sparse graph cheap to gather.**
- **Both variables are load-bearing** — ogbn-arxiv wins at `F=256` and loses at
  `F=40` on the same degree. Stage 2 blamed width alone because it measured at
  `P=1`, where the threshold is 512 and nothing reaches it.
- **Sort the packs; do not group them.** A pass costs its maximum degree, which
  sounds like the hard part and is not: degree-sorted greedy filling lands
  within **4%** of the unreachable bound (15.40–15.95× vs 16.00×). Grouping by
  *exactly equal* degree instead hits **0.05× on ogbn-arxiv — 20× slower than
  not packing** — a power-law tail has thousands of sub-`P` buckets, each still
  costing a whole pass.
- **The bill: 8.19 TB/s** of KV-SRAM port at `P*=16`. That is **4×** §7's figure
  for the same `P`, because attention packs at 4-bit and aggregation runs at 16.
  The speedups are a compute-side ceiling; the port decides reachability.

> **Verdict.** Omni-LUT is an excellent Combine engine, and the wrong shape for
> Aggregate *at P=1*. The shape problem is `M=1`, it is fixable by packing, and
> what it costs is bandwidth — which is the same sentence §7 ended on, reached
> from a different workload.


## What to do

- **Know which regime you are in first.** Inside the memory-bound triangle
  (low batch, short context) the lever is weight bytes — worth **6.80×**.
  Outside it, array occupancy — worth **3.12×**. Nothing else comes close in
  either.
- **Stop aiming KV techniques at batch 1.** Removing *all* KV traffic there buys
  **1.01–1.07×**. That is the whole literature's ceiling, and it is where most
  of our own measuring happened.
- **Build P=8 packing.** Largest lever outside the triangle, 3.118× at batch 32,
  fits in 4.5 MB, and survives its own 1.02 TB/s bandwidth check.
- **Prune bit-width, not channels.** The only axis that multiplies cycles and
  bytes, composes with eviction, and is still unmeasured.
- **Pick the KV layout before the pruning algorithm.** It silently decides which
  pruning literature is deployable at all.
- **Packing is the lever in both workloads, and both times the open question
  is the SRAM port.** It is worth 3.12× on decode at batch 32 and 7.35× on
  Reddit aggregation — and needs 1.02 TB/s for the first, 8.19 TB/s for the
  second. **Cost that port next**; every compute-side win above depends on it.

*Full derivations, model-change record and open gaps: `study.md`.*
