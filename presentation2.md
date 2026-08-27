# Omni-LUT — The Memory Model, and What Bounds Decode

## 1. The memory model

### 1.1 Configuration

| | value |
|---|---|
| PE array | **32 × 4**, 32 RACs/PE, MU=4 = **4,096 lanes** |
| clock | 500 MHz |
| precision | W4 A16 **KV4** |
| DRAM | **51.2 GB/s** peak, 64 B burst, ~90 ns |
| request queue | swept 16 → 128 *(not in the paper)* |
| operand port | **128 GB/s** |
| unified buffer | swept 256 KB → unlimited *(not in the paper)* |
| model | LLaMA-3-8B, **GQA 32:8**, 32 layers, head_dim 128 |

### 1.2 Datapath

```
DRAM ──► unified buffer ──► BQU ──► LGU ──► PE array ──► accumulator
(KV cache,   (staging,    (quantise  (build     (32×4,
 weights)     not a cache)  to BCQ)   the LUT)   32 RAC/PE)
```

- **KV never lives on-chip** — whole cache re-read from DRAM every step.
- One entry = **64 B = one DDR5 burst**. Per token, all layers = **32 KB**.
- Working set **2.06 MB — constant in batch**. One tile at a time, not the 512 MB full set.

**KV read per step vs constant weight read:**

| | 2K ctx | 32K ctx |
|---|---:|---:|
| batch 1 | 0.06 GB (2.4% of DRAM) | 1.00 GB (28.2%) |
| batch 32 | 2.00 GB (44.0%) | 32.00 GB (**92.6%**) |

### 1.3 Roofline

![Decode roofline](analysis/memory/roofline.png)

| op | intensity | attained | of peak |
|---|---:|---:|---:|
| projections + FFN, batch 1 | 4.0 | 1,014–4,085 GFLOP/s | 25–99% |
| projections + FFN, batch 32 | **128.0** | ~3,940 GFLOP/s | 96% |
| `qk_matmul` | 14.2 | 2,774 GFLOP/s | 68% |
| **`attn_v_matmul`** | 14.2 | **128 GFLOP/s** | **3.1%** |

- Nominal ridge **80 FLOP/byte** assumes all 4,096 lanes work.
- `attn_v` lights **1 of 32 rows** → 3.1% of peak → real ridge **2.4**.
- 14.2 > 2.4 ⇒ **compute-bound, because the array is bad at this shape.**
- **Batching moves projections across the ridge**: 4.0 → **128.0**.
- **Batching cannot move attention**: 14.2 at every batch (FLOPs and bytes both scale with N).
- Context also cannot move it sideways — both scale with `kv_len` too.
- Context moves it **up**: `qk` 1,562 → 2,774 GFLOP/s; `attn_v` 125.4 → 127.8.
- ⚠️ **Roofline plots ratios, so it hides duration.** 2K→32K: FLOPs ×16, bytes ×16, **time ×15.7 (4.28 → 67.20 ms)**. Dot stays still, everything grows.
- Orange roof = 32-deep queue. Ridge 80 → **180**.

### 1.4 SRAM — never decides it

| batch | ctx | compute | DRAM | SRAM | SRAM/max |
|---:|---:|---:|---:|---:|---:|
| 1 | 2K | 7.7 ms | **51.3 ms** | 24.9 ms | 0.49 |
| 1 | 32K | 73.3 ms | **73.4 ms** | 61.3 ms | 0.83 |
| 8 | 2K | 57.5 ms | **61.6 ms** | 59.6 ms | **0.97** |
| 32 | 32K | **2331.5 ms** | 804.8 ms | 1342.5 ms | 0.58 |

- 0.97 near-miss is **an ideal-DRAM artefact** → **0.78×** under a finite queue.
- Capacity nearly inert: decode unchanged from unlimited to **1 MB**.
- Prefill rises **3.3× at 8 MB** — decode has no reuse to lose, prefill does.

### 1.5 DRAM latency — decides it

`reachable = min(peak, outstanding × burst / latency)` — at 90 ns, **72 reads in flight** needed.

| queue depth | reachable | memory-bound cells | C/D at b1/32K |
|---:|---:|---:|---:|
| none (ideal) | 51.2 GB/s | **11 of 30** | 1.00 |
| 64 | 45.5 GB/s | 13 of 30 | 0.89 |
| **32** | **22.8 GB/s** (44%) | **21 of 30** | **0.44** |
| **16** | **11.4 GB/s** (22%) | **30 of 30** | **0.22** |

- **Bandwidth is a property of the DRAM *and the requester*.**
- **The compute-bound triangle was an artefact of an infinite queue.**
- Weight DRAM constant either way: 49.81 ms/token → **112.1 ms** at 32-deep.

### 1.6 Verdict

Compute ÷ DRAM at datasheet bandwidth; below 1.00 the array waits:

| batch | 2K | 4K | 8K | 16K | 32K |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.15 | 0.23 | 0.38 | 0.62 | 1.00 |
| 2 | 0.28 | 0.43 | 0.67 | **1.04** | 1.51 |
| 4 | 0.52 | 0.76 | **1.12** | 1.56 | 2.02 |
| 8 | 0.93 | **1.28** | 1.69 | 2.10 | 2.44 |
| 16 | **1.57** | 1.93 | 2.27 | 2.54 | 2.73 |
| 32 | 2.37 | 2.60 | 2.74 | 2.84 | 2.90 |

- **Ideal memory** → memory-bound in **11 of 30**.
- **32-deep queue** → **21 of 30**. **16-deep** → **everywhere**.
- **SRAM decides none of it.** DRAM and queue depth do.
- **Inside the memory-bound region the bottleneck is weights, not KV**: 2K/b1 = **97.1% weights, 2.9% KV**.
- Latency scales both legs alike → cannot change that ratio; only deepens the wait (array idle **86% → 94%**).

### 1.7 Scope

| configuration | max `C/D` | compute-bound cells |
|---|---:|---|
| LLaMA-3-8B GQA, KV4 | 2.90 | the triangle |
| OPT-6.7B **MHA**, KV4 | 1.07 | 1 of 30 |
| OPT-6.7B **MHA**, KV16 | **0.94** | **none** |

- **This is a GQA result.** The paper evaluates OPT and LLaMA2 — **all MHA**.
- **Two independent reasons we disagreed with the literature, either sufficient**: they measure MHA; we never charged for latency.

Still idealised:

- Latency term is a **throughput clamp, not a latency model** — **scattered reads cost more than modelled**.
- **KV buffer pressure inexpressible**: `attn_v` has `n_tiles = 1` → spill charge **zero at any buffer size**.
- Prefill spill charge prices an assumed loop order — read as *predicate fires*, not as cost.

---

## 2. Which pruning axis survives

- Channel pruning, select-without-evict, KV residency: all removed real DRAM traffic, all produced ~nothing.
- Eviction removed the same traffic and **worked**.
- **One line of arithmetic predicts both, in advance.**

### 2.1 The formula

```
cycles    = batch × per_round × rounds × qbit
per_round = LUT_GEN(3) + ceil(K/MU) + 1 + array_n + OUTPUT(2)
rounds    = ceil(n_tiles / array_m),  n_tiles = ceil(N / 128)
```

For decode `attn_v` — `(M=1, K=kv_len, N=head_dim=128)`: `per_round = ceil(kv_len/4) + 10`, `n_tiles = 1`, `rounds = 1`.

### 2.2 The axis table

| axis | technique | enters via | cycles | DRAM | measured |
|---|---|---|---|---|---:|
| channel (`head_dim` = N) | ThinK | only `ceil(N/128)` | **null** | linear | **1.000×** |
| token (`kv_len` = K) | **eviction** | `k_eff = ceil(K/4)` | **linear** | linear | **1.45× b1 · 15.96× b32** |
| token, read-only | Quest | `k_eff`, storage unchanged | linear | linear | 12.85× b32 |
| **bit-width** (`qbit`) | KV3 / KV2 | outer multiplier | **linear** | **linear** | **~1.07–1.14× energy** * |

<sub>\* KV2 vs KV4, from the paper's Fig. 10 energy bars. Every other figure is our own TPOT at datasheet bandwidth; §2.3 re-measures under a finite queue.</sub>

- **Channel = null.** `N ≤ 128` ⇒ `ceil(N/128) = 1` always. Not weak — **absent**. Hence *exactly* 1.000×.
- Channel does shrink `qk`, but `qk` is **4.1% of decode**. *The axis that saves cycles is the stage that costs nothing.*
- **Token = survivor.** `k_eff` is linear and unbounded ⇒ eviction **must** remove cycles ⇒ **must** work. 15.96× at batch 32.
- **Same bytes, same hardware, opposite outcome — one line of arithmetic apart.**
- **Quest is the control**: byte-identical to eviction, still lowers `k_eff`, tracks it (12.85× vs 15.96×). Gap = storage, not compute.
- **Bit-width has no null**: `qbit` is an outer multiplier, no `ceil` can absorb it.

### 2.3 Ceilings under a realistic memory profile

**Delete the entire KV cache:**

| profile | b1/2K | b1/32K | b32/2K | b32/32K |
|---|---:|---:|---:|---:|
| ideal memory | 1.01× | 1.07× | 1.05× | 1.12× |
| 90 ns, 32 deep | **1.01×** | 1.13× | 1.16× | 1.32× |
| 90 ns, 16 deep | **1.01×** | 1.17× | 1.26× | **1.65×** |

**Delete all weight traffic:**

| profile | b1/2K | b1/32K | b32/2K | b32/32K |
|---|---:|---:|---:|---:|
| ideal memory | 6.80× | 1.57× | 1.00× | 1.00× |
| 90 ns, 32 deep | **13.14×** | 2.13× | 1.11× | 1.01× |
| 90 ns, 16 deep | **21.78×** | 2.79× | 1.44× | 1.04× |

- **Latency sharpens the conclusion, doesn't soften it.**
- Weight ceiling **triples** (6.80× → 21.78×). KV ceiling at b1/2K **never moves — 1.01× under every profile**.
- KV does gain where KV already dominated: **1.12× → 1.65×** at b32/32K.

### 2.4 The discriminator

> Not *"how many bytes does it remove?"* — but *"does it reach `k_eff` or `qbit`?"*

- DRAM only → inside the ceiling above.
- Lowers the compute roof → survives.
- Checkable on paper before any simulation.

### 2.5 Bit-width

- Paper builds KV3/KV2 and confirms the mechanism: KV2 "eliminates … 50% (vs. KV4) of the computational workload" = `cycles ∝ qbit`.
- Payoff modest: KV2 within **1.07×–1.14×** of KV4 in total energy.
- **Composes with eviction** — `cycles ∝ k_eff × qbit`, orthogonal, so they **multiply**.
- Open on our side: the **latency** effect, and how it stacks with eviction.

### 2.6 Measured in the wrong place

![KV reduction vs batch](analysis/memory/kv_batch.png)

- Weight traffic **constant** in batch; KV traffic **scales** with it.
- So KV's share of DRAM — and the payoff for cutting it — **grows with batch**.
- `evict-1024`: **2.46× at batch 1 → 15.96× at batch 32** (datasheet bandwidth).

### 2.7 Eviction — the exception that obeys the rule

- Batch-1 figure is **1.45×** under pipelining — inside the same ceiling as everything else.
- The **15.96× is a batch-32 result**, earned where bytes finally bind.
- It wins twice there: `kv_len` is also `attn_v`'s reduction dim, so it lowers the **compute** roof too.
- **Moves both roofs ⇒ robust. Moves only the slack roof ⇒ artefact.**

### 2.8 What §2 does not establish

- Channel null is **`attn_v`-specific** — depends on `head_dim ≤ 128`.
- **Scoped to GQA + KV4** (§1.7). MHA/KV16 ⇒ KV share 25–84%, ceiling 1.11–1.46×.
- Batch-1 numbers assume the **serial** roofline; pipelined turns 2.46× into 1.452×.
- **Scattered reads charged optimistically** — selective-KV costs more than any table here shows.
