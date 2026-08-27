# Omni-LUT — The Memory Model, and What Bounds Decode

## 1. Is decode compute-bound or memory-bound?

### 1.1 The reason

Every KV-reduction paper rests on one premise: decode is memory-bound, so removing bytes buys time. A technique aimed at the wrong side of that boundary cannot work, however well implemented. So the boundary has to be found before anything else is judged.

**The machine:**

| | value |
|---|---|
| PE array | **32 × 4**, 32 RACs/PE, MU=4 = **4,096 lanes** |
| clock | 500 MHz |
| precision | W4 A16 **KV4** |
| DRAM | **51.2 GB/s** peak, 64 B burst, ~90 ns |
| request queue | swept 16 → 128 *(not stated in the paper)* |
| operand port | **128 GB/s** = `MU × array_n × NUM_RAC × kv_bits` |
| unified buffer | swept 256 KB → unlimited *(not stated in the paper)* |
| model | LLaMA-3-8B, **GQA 32:8**, 32 layers, head_dim 128 |

**The datapath, and why the KV cache is the interesting operand:**

```
DRAM ──► unified buffer ──► BQU ──► LGU ──► PE array ──► accumulator
(KV cache,   (staging,    (quantise  (build     (32×4,
 weights)     not a cache)  to BCQ)   the LUT)   32 RAC/PE)
```

- **KV never lives on-chip.** The whole cache is re-read from DRAM every decode step.
- One entry = `128 × 4/8` = **64 B = exactly one DDR5 burst**; per token, all layers = **32 KB**.
- The buffer holds **one instance's tile — 2.06 MB, constant at batch 1, 8 and 32.** Instances stream one at a time; the full working set would be 512 MB.

| KV read per step | 2K ctx | 32K ctx |
|---|---:|---:|
| batch 1 | 0.06 GB (2.4% of DRAM) | 1.00 GB (28.2%) |
| batch 32 | 2.00 GB (44.0%) | 32.00 GB (**92.6%**) |

Weight traffic, by contrast, is **constant** — read once per step whatever the batch or context.

### 1.2 The picture

![Decode roofline](analysis/memory/roofline.png)

**How to read it.** Diagonal = memory-limited, flat top = compute-limited, and where they meet is the ridge. A point sits *on* or *below* a roof, never above.

| op | intensity | attained | of peak |
|---|---:|---:|---:|
| projections + FFN, batch 1 | 4.0 | 1,014–4,085 GFLOP/s | 25–99% |
| projections + FFN, batch 32 | **128.0** | ~3,940 GFLOP/s | 96% |
| `qk_matmul` | 14.2 | 2,774 GFLOP/s | 68% |
| **`attn_v_matmul`** | 14.2 | **128 GFLOP/s** | **3.1%** |

**The textbook reading is wrong.** Attention runs at 14.2 FLOP/byte against an 80 FLOP/byte ridge — five and a half times inside the memory-limited region, so it *looks* memory-bound. But 80 assumes all 4,096 lanes work. `attn_v` is `(M=1, K=kv_len, N=head_dim)`: it lights **one of 32 rows**, attains **3.1% of peak**, and against that ceiling the ridge is `125/51.2` = **2.4 FLOP/byte** — which 14.2 clears.

> **`attn_v` is compute-bound not because it does much arithmetic, but because the array is bad at this shape.**

Three more things the picture shows:

- **Batching moves the projections across the ridge** — 4.0 → **128.0**. Weights are read once per step, so batch multiplies compute by N and DRAM by less.
- **Batching can never move attention** — 14.2 at *every* batch, because its FLOPs and its KV bytes both scale with N. Context cannot move it sideways either: **14.23 at 2K, 14.22 at 32K**, since `kv_len` cancels the same way.
- **Context moves it straight up**: `qk` 1,562 → 2,774 GFLOP/s as `N = kv_len` fills more tiles; `attn_v` creeps 125.4 → 127.8, asymptotic to 1/32 of peak.

> ⚠️ **A roofline plots two ratios, so it hides duration.** From 2K to 32K, `attn_v` FLOPs ×16, bytes ×16, **time ×15.7 — 4.28 ms to 67.20 ms**. The dot sits still while everything about it grows 16×. Read it for *what limits you*, never *how long you wait*.

The orange roof is the same machine with a realistic request queue (§1.3): it drags the ridge from 80 to **180 FLOP/byte**, and `attn_v`'s own effective ridge from 2.4 to **5.5** — still under its 14.2, so the one compute-bound operation stays compute-bound while everything else gets worse.

### 1.3 The results

**SRAM never decides it.** Charged at the array's own 128 GB/s port, datasheet DRAM:

| batch | ctx | compute | DRAM | SRAM | SRAM/max |
|---:|---:|---:|---:|---:|---:|
| 1 | 2K | 7.7 ms | **51.3 ms** | 24.9 ms | 0.49 |
| 1 | 32K | 73.3 ms | **73.4 ms** | 61.3 ms | 0.83 |
| 8 | 2K | 57.5 ms | **61.6 ms** | 59.6 ms | **0.97** |
| 32 | 32K | **2331.5 ms** | 804.8 ms | 1342.5 ms | 0.58 |

- The 0.97 near-miss is **itself an ideal-DRAM artefact** — under a finite queue it falls to **0.78×**.
- **Capacity is nearly inert on decode**: unchanged from unlimited down to **1 MB**. Prefill rises **3.3× at 8 MB** — decode re-reads everything anyway, so there is no reuse for a small buffer to lose.

**DRAM latency decides it.** 51.2 GB/s is a peak; reaching it needs enough reads in flight to cover the round trip — `reachable = min(peak, outstanding × burst / latency)`. At 90 ns that is **72 reads in flight**.

| queue depth | reachable | memory-bound cells | C/D at b1/32K |
|---:|---:|---:|---:|
| none (ideal) | 51.2 GB/s | **11 of 30** | 1.00 |
| 64 | 45.5 GB/s | 13 of 30 | 0.89 |
| **32** | **22.8 GB/s** (44%) | **21 of 30** | **0.44** |
| **16** | **11.4 GB/s** (22%) | **30 of 30** | **0.22** |

> **Bandwidth is a property of the DRAM *and the requester*. The compute-bound region was an artefact of assuming an infinitely deep queue.**

Weight DRAM stays constant at every cell either way — **49.81 ms** per token at datasheet, **112.1 ms** at a 32-deep queue. Only the size of the floor moves.

**The grid.** Compute ÷ DRAM at datasheet bandwidth; below 1.00 the array waits:

| batch | 2K | 4K | 8K | 16K | 32K |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.15 | 0.23 | 0.38 | 0.62 | 1.00 |
| 2 | 0.28 | 0.43 | 0.67 | **1.04** | 1.51 |
| 4 | 0.52 | 0.76 | **1.12** | 1.56 | 2.02 |
| 8 | 0.93 | **1.28** | 1.69 | 2.10 | 2.44 |
| 16 | **1.57** | 1.93 | 2.27 | 2.54 | 2.73 |
| 32 | 2.37 | 2.60 | 2.74 | 2.84 | 2.90 |

> **Verdict.** Ideal memory: memory-bound in a low-batch, short-context triangle, **11 of 30 cells**. A 32-deep queue: **21 of 30**. A 16-deep queue: **everywhere**. **SRAM decides none of it** — what decides it is DRAM, and how deep a queue you can keep full.

**And the resource that binds is weights, not KV.** At 2K / batch 1 the byte split is **97.1% weights, 2.9% KV**. Latency scales both legs alike, so it cannot change that ratio — it only deepens the wait, with the array idle **86% → 94%**.

### 1.4 Scope

**This is a GQA result.** The Omni-LUT paper evaluates OPT-1.3B…30B and LLaMA2-7B/13B — **all MHA**:

| configuration | max `C/D` over 30 cells | compute-bound cells |
|---|---:|---|
| LLaMA-3-8B GQA, KV4 | 2.90 | the triangle |
| OPT-6.7B **MHA**, KV4 | 1.07 | 1 of 30 |
| OPT-6.7B **MHA**, KV16 | **0.94** | **none** |

**Two independent reasons this model disagreed with the literature, either sufficient:** the paper measures MHA, and no roofline here charged for latency. Under either correction, decode is memory-bound where the paper says it is.

Still idealised:

1. **The latency term is a throughput clamp, not a latency model** — no dependent chains, row misses or refresh. **A scattered KV read does worse than modelled here.**
2. **KV buffer pressure is inexpressible** — the spill charge is `A_bytes × (n_tiles − 1)` and `attn_v` has `n_tiles = 1`, so it is **zero at any buffer size** against a 2.06 MB working set.
3. **The prefill spill charge prices an assumed loop order** — read it as *the predicate fires here*, not as a cost.

---

## 2. Which pruning axis survives

**The claim**

> - Channel pruning, select-without-evict and KV residency each removed real DRAM traffic and each produced little or no speedup.
> - Eviction removed the same kind of traffic and **did** produce speedup.
> - One line of arithmetic predicts both, in advance.

- The three failures share nothing algorithmically — ThinK drops channels, Quest skips reads, residency moves bytes on-chip.
- They share only **which resource they attack**: KV DRAM bytes.
- Three unrelated methods failing identically ⇒ the failure is a property of the **resource**, not of any method.

### 2.1 The formula

```
cycles    = batch_size × per_round × rounds × qbit
per_round = LUT_GEN(3) + ceil(K / MU) + 1 + array_n + OUTPUT(2)
rounds    = ceil(n_tiles / array_m)
n_tiles   = ceil(N / (array_n × NUM_RAC))
```

For decode `attn_v` — `(M=1, K=kv_len, N=head_dim=128)`:

```
per_round = ceil(kv_len/4) + 10      ← the fixed 10 is the knee constant
n_tiles   = ceil(128/128) = 1
rounds    = ceil(1/32)   = 1
```

### 2.2 The axis table

| axis | technique | where it enters | cycles | DRAM | measured |
|---|---|---|---|---|---:|
| channel (`head_dim` = N) | ThinK | only `ceil(N/128)` | **null** | linear | **1.000×** |
| token (`kv_len` = K) | **eviction** (H2O, SnapKV) | `k_eff = ceil(K/4)` | **linear** | linear | **1.45× b1 · 15.96× b32** |
| token, read-only | Quest, TidalDecode | `k_eff`, storage unchanged | linear | linear | 12.85× b32 |
| **bit-width** (`qbit`) | KV3 / KV2 | outer multiplier | **linear** | **linear** | **~1.07–1.14× energy** * |

<sub>\* KV2 vs KV4, from the Omni-LUT paper's Fig. 10 energy bars — the one row measured elsewhere, and in a different unit. Every other figure is TPOT from our own model **at datasheet DRAM bandwidth**; §2.4 re-measures the ceilings under a finite request queue.</sub>

### 2.3 Row by row

**Channel — the null**

- `head_dim` is `attn_v`'s **output** dim `N`, reaching cycles through exactly one term: `ceil(N/128)`.
- For every `N ≤ 128` that term is **1**. Pruning 128 → 64, or → 8, does not change it.
- `N` is not weakly present. It is **absent** — which is why the result is *exactly* 1.000×.
- Channel pruning does shrink `qk` — but `qk` is **4.1% of decode**.
- **The axis that saves cycles is the stage that costs nothing.**

**Token — the survivor, and the proof the table is honest**

- `kv_len` is `attn_v`'s reduction dim `K`, entering via `k_eff = ceil(K/MU)` — linear and unbounded.
- So eviction **must** remove cycles, and therefore **must** work. Measured: **15.96× at batch 32.**
- A framework that only explains failures is unfalsifiable. This one commits in **both** directions from the same arithmetic.
- **Same bytes. Same hardware. Opposite outcome. One line of arithmetic apart.**

**Token, read-only — the control**

- Quest reads fewer blocks but stores the full cache → **byte-identical to eviction** on the read path.
- If bytes decided, the two would be indistinguishable. If `k_eff` decides, they should nearly coincide — and they do: **12.85× vs 15.96×**. The residual gap is **storage, not compute**.

**Bit-width — the axis with no null anywhere**

- `qbit` multiplies the *entire* expression; no `ceil` sits outside it to absorb it.
- KV4 → KV3 is 0.75× cycles **and** 0.75× bytes, unconditionally.
- **Channel pruning has a rounding boundary to die on. Bit-width does not.**

### 2.4 The ceilings, under a realistic memory profile

The bound that makes this quantitative: speedup if that resource became **entirely free**. Latency makes DRAM bind harder, so a KV technique should be worth *more* — the question is how much.

**Delete the entire KV cache:**

| profile | b1 / 2K | b1 / 32K | b32 / 2K | b32 / 32K |
|---|---:|---:|---:|---:|
| ideal memory | 1.01× | 1.07× | 1.05× | 1.12× |
| 90 ns, 64 deep | 1.01× | 1.08× | 1.07× | 1.14× |
| **90 ns, 32 deep** | **1.01×** | 1.13× | 1.16× | 1.32× |
| **90 ns, 16 deep** | **1.01×** | 1.17× | 1.26× | **1.65×** |

**Delete all weight traffic:**

| profile | b1 / 2K | b1 / 32K | b32 / 2K | b32 / 32K |
|---|---:|---:|---:|---:|
| ideal memory | 6.80× | 1.57× | 1.00× | 1.00× |
| 90 ns, 32 deep | **13.14×** | 2.13× | 1.11× | 1.01× |
| 90 ns, 16 deep | **21.78×** | 2.79× | 1.44× | 1.04× |

- **The realistic profile sharpens §2's conclusion rather than softening it.** Latency makes memory bind harder — but the memory that binds at low batch is *weights*. The weight ceiling triples, 6.80× → 21.78×; the KV ceiling at b1/2K does not move **at all**, staying **1.01× under every profile**.
- KV work does gain where KV already dominated: **1.12× → 1.65×** at batch 32 / 32K. Real, and still not the order of magnitude the byte counts suggest.
- **The one-line rule survives verbatim:** deleting the entire cache buys 1.01× at batch 1 / 2K whether DRAM runs at 51.2 GB/s or 11.4.

### 2.5 The discriminator

> The question is never *"how many bytes does this remove?"*
> It is *"does it reach `k_eff` or `qbit`?"*

- Touches only DRAM → lands inside the ceiling above.
- Lowers the compute roof → survives.
- Checkable on paper before a single simulation runs.

### 2.6 Bit-width: confirmed by the paper, bounded by it

- **The mechanism is confirmed by construction.** The paper builds KV3 and KV2 and states it in our terms: KV2 "eliminates 33% (vs. KV3) and 50% (vs. KV4) of the computational workload from these AA-GEMM bottlenecks" — that is `cycles ∝ qbit`.
- **The payoff is smaller than the halving implies.** Its Fig. 10 energy bars put KV2 within ≈ **1.07×** of KV4 on a decode-heavy workload and ≈ **1.14×** on a prefill-heavy one — for an axis that halves bytes *and* cycles.
- **It composes with eviction rather than competing.** `cycles ∝ k_eff × qbit` → orthogonal, so they **multiply**; evict-1024 at KV3 ≈ the *product*.
- **Still open on our side:** the paper measures *energy*. The **latency** effect of KV3/KV2, and how it stacks with eviction, is what our model can add.

### 2.7 All of it was measured in the wrong place

![KV reduction vs batch](analysis/memory/kv_batch.png)

- **Weight traffic is constant in batch** — read once, however many sequences are in flight.
- **KV traffic scales with batch** — every sequence carries its own cache.
- So KV's *share* of DRAM, and the payoff for cutting it, **grows with batch**.
- `evict-1024`: **2.46× at batch 1 → 15.96× at batch 32** (datasheet bandwidth; §2.4 shows what a finite queue does to the ceiling these sit under).

### 2.8 Eviction is the exception that still obeys the rule

- Its batch-1 figure is **1.45×** under the pipelined model — inside the same ceiling as everything else.
- The **15.96× is a batch-32 result**, earned exactly where bytes finally bind.
- At batch 32 it wins twice: eviction also lowers the **compute** roof, since `kv_len` is `attn_v`'s reduction dim.
- **A technique that moves both roofs is robust. One that moves only the slack roof is not.**

### 2.9 What §2 does *not* establish

- **The channel null is `attn_v`-specific** — it depends on `head_dim ≤ 128` making `n_tiles = 1`.
- **Scoped to GQA + KV4** (§1.4). On MHA at KV16 the KV share is 25–84%, not 2.9%, and the ideal-memory ceiling is 1.11–1.46× before any latency correction.
- **The batch-1 numbers assume the serial roofline.** Under pipelining, eviction's 2.46× becomes 1.452×.
- **A scattered read is charged optimistically** (§1.4). Selective-KV methods that gather non-contiguous blocks cannot keep the request queue as full as a streaming read, so their true cost is above what any table here shows.
