# Omni-LUT — What Bounds Decode, and Which Pruning Axis Survives

## 1. Is decode compute-bound or memory-bound?

**Why it matters**

- Every KV-reduction paper rests on one premise: decode is memory-bound, so removing bytes buys time.
- A technique aimed at the wrong side of the boundary cannot work, however well implemented.

**Verdict**

> - On **GQA + KV4**: memory-bound only in a low-batch, short-context corner — and inside that corner the binding resource is **weights, not KV**.
> - On **MHA + KV16**: memory-bound **everywhere**. §1.5 shows GQA is the reason.

*§1.1–§1.4 are LLaMA-3-8B (GQA 32:8, KV4). §1.5 varies both.*

---

### 1.1 It is a triangle, not a property

Decode compute ÷ DRAM per token. **Below 1.00 the array waits on memory:**

| batch | 2K | 4K | 8K | 16K | 32K |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.15 | 0.23 | 0.38 | 0.62 | 1.00 |
| 2 | 0.28 | 0.43 | 0.67 | **1.04** | 1.51 |
| 4 | 0.52 | 0.76 | **1.12** | 1.56 | 2.02 |
| 8 | 0.93 | **1.28** | 1.69 | 2.10 | 2.44 |
| 16 | **1.57** | 1.93 | 2.27 | 2.54 | 2.73 |
| 32 | 2.37 | 2.60 | 2.74 | 2.84 | 2.90 |

- Memory-bound cells form a **triangle** in the low-batch, short-context corner.
- First compute-bound batch: **16 at 2K · 8 at 4K · 4 at 8K · 2 at 16K · 2 at 32K.**
- **"Decode is DRAM-bound" is the batch-1 row, and only that row.**
- Even batch 1 / 32K is *exactly 1.00* — balanced, not memory-bound.

### 1.2 The boundary is diagonal because weight traffic is constant

- **Weight DRAM = 49.81 ms per token at every cell of the grid.** Every batch, every context, no exception.
- Weights are read **once per token** regardless of batch or context → DRAM has a large fixed floor.
- Compute has no floor: it grows with **batch** (more sequences) *and* **context** (attention's reduction dim).

At batch 1:

| ctx | compute | DRAM | weights' share | C/D |
|---:|---:|---:|---:|---:|
| 2K | 7.67 ms | 51.28 ms | 97.1% | 0.15 |
| 8K | 20.94 ms | 55.71 ms | 89.4% | 0.38 |
| 32K | 73.33 ms | 73.40 ms | 67.9% | **1.00** |

- **DRAM moves 1.43× (51→73 ms). Compute moves 9.56× (7.67→73.33 ms).** That asymmetry is the mechanism.
- Crossover = *"when does compute exceed the constant ~50 ms of weight traffic?"*
- Both axes push it the same way → the boundary runs **diagonally**, not at a fixed batch or context.
- At 32K / batch 1 the roofs are **73.33 vs 73.40 ms** — the most balanced point in the grid.

### 1.3 Inside the triangle, the bottleneck is weights — not KV

At **2K / batch 1**, the most memory-bound cell in the whole grid:

| | time | share |
|---|---:|---:|
| weight DRAM | 49.81 ms | **97.1%** |
| KV DRAM | ~1.5 ms | **2.9%** |
| compute | 7.67 ms | array idle **86%** |

- ≈ **2.6 GB of weights** per token against **80 MB of KV**.
- Picking the *most* memory-bound cell is deliberate — if KV is 2.9% here, it is less everywhere else in the triangle.
- **At batch 1 you can attack the KV cache as hard as you like and still be optimising 2.9% of the bottleneck.**

### 1.4 What any lever could possibly buy

Speedup if that resource became **entirely free** — a ceiling no algorithm can beat:

| batch | ctx | packing | overlap | KV bytes | weight bytes |
|---:|---:|---:|---:|---:|---:|
| 1 | 2K | 1.07× | 1.07× | **1.01×** | **6.80×** |
| 1 | 32K | 1.75× | 1.75× | **1.07×** | 1.57× |
| 32 | 2K | 1.88× | 1.05× | 1.05× | 1.00× |
| 32 | 32K | **3.12×** | 1.12× | 1.12× | 1.00× |

- **Deleting the entire KV cache at batch 1 buys 1.01× at 2K, 1.07× at 32K.** Not "eviction buys little" — removing *all* of it buys 7%.
- Algorithm-independent ceiling on the whole KV literature at batch 1 → the one-line explanation for every negative result in §2.
- The levers that are **not** bounded that way:

| regime | lever | worth |
|---|---|---:|
| inside the triangle (low batch, short ctx) | weight bytes | **6.80×** |
| outside it (high batch, long ctx) | array occupancy | **3.12×** |

- Neither is a KV technique.
- The two regimes want different machines → **two accelerators, not one.**

### 1.5 The triangle is a GQA result, not an Omni-LUT result

- §1.1–§1.4 run **LLaMA-3-8B = GQA 32:8** (8 KV heads). That choice, not the accelerator, does most of the work.
- The Omni-LUT paper evaluates OPT-1.3B/2.7B/6.7B/13B/30B and LLaMA2-7B/13B — **all MHA**, 32 KV heads, 4× the KV bytes with no change to attention compute.

| configuration | max `C/D` over 30 cells | compute-bound cells |
|---|---:|---|
| LLaMA-3-8B GQA, KV4 *(§1.1–§1.4)* | 2.90 | the triangle |
| LLaMA-3-8B GQA, KV16 | 3.22 | almost all |
| OPT-6.7B **MHA**, KV4 | 1.07 | **1 of 30** (batch 32 / 2K) |
| OPT-6.7B **MHA**, KV16 | **0.94** | **none** |

- **On MHA at KV16 — the baseline every KV paper argues against — decode is memory-bound in every cell. There is no triangle.** The literature's premise is right for the configuration it was written about.
- **GQA is the dominant term.** At batch 1 / 32K: `C/D` = 1.00 on GQA KV4, 0.49 on MHA KV4. GQA divides KV traffic by 4 and attention compute by nothing → 4× the arithmetic intensity.
- **More KV bits make decode *more* compute-bound, not less.** GQA KV4 → KV16 takes `C/D` from 1.00 → 2.08 at batch 1 / 32K, because `cycles ∝ qbit` on a bit-serial array quadruples AA compute while quadrupling only the KV slice of DRAM.
- **Under MHA the boundary stops being diagonal.** On GQA, `C/D` rises with context at every batch. On MHA at batch 32 it **falls** — 1.07 at 2K → 0.83 at 32K — because KV bytes outgrow attention compute.

What that does to §1.3 and §1.4:

| | GQA KV4 | MHA KV4 | MHA KV16 |
|---|---:|---:|---:|
| KV share of decode DRAM, b1 / 2K | 2.9% | 7.9% | **25.2%** |
| KV share of decode DRAM, b1 / 32K | 32.1% | 57.9% | **84.3%** |
| ceiling if all KV free, b1 / 2K | 1.01× | 1.03× | **1.11×** |
| ceiling if all KV free, b1 / 32K | 1.07× | 1.30× | **1.46×** |

> **§1.4's ceiling is not a universal bound on KV work — it is what remains once GQA and 4-bit KV have already been applied.** A compliment to the design, not a refutation of the literature.

- **Caveat the other way:** the paper never states a batch size, and its edge framing implies **batch 1** — the memory-bound row in *every* configuration above, ours included. At the paper's operating point our grid agrees with it. The compute-bound half of §1.1 lives at batch ≥ 2, which the paper does not evaluate.

### 1.6 How this was measured

- Roofline per op: `time = max(cycles / freq, dram_bytes / bandwidth)`.
- Sum over every decode GEMM → compute `C`, DRAM `D`. Regime = `C/D`.
- Balance point, computed in code so it cannot drift: `2 × 4096 × 500e6 / 51.2e9` = **80 FLOP/byte**.
- Decode attention sits far below 80 FLOP/byte — which is why "memory-bound" *sounds* obviously true. But 80 assumes all 4,096 lanes work; at `M=1` only **3.12%** do, so the effective balance point is ≈ **2.5 FLOP/byte**, while `attn_v` under GQA 32/8 at KV4 runs at **16 FLOP/byte**. Below 80, above 2.5 — hence the measurement disagrees with the intuition.

**Two things that nearly went wrong:**

- **The obvious split is wrong.** "AW = weights, AA = KV" fails: `k_proj`/`v_proj` are AW ops that *write the KV cache*. It made weight DRAM drift 0.04% between batch 1 and 32 — a quantity that must be exactly constant, which is what made it detectable. Split **by read/write, not by operation.**
- **The first version of this grid was wrong.** Every row but batch 1 sat in the wrong regime, from a cycle defect that dropped the batch term (`ceil(M/32) = 1` for all M in 1..32). Reported, not quietly corrected.

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

For decode `attn_v` — `(M=1, K=kv_len, N=head_dim=128)`, `MU=4`, `array_m=32`, `array_n=4`, `NUM_RAC=32`:

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

<sub>\* KV2 vs KV4, read from the Omni-LUT paper's Fig. 10 energy bars — the one row measured elsewhere, and in a different unit. Every other figure is TPOT from our own model.</sub>

### 2.3 Row by row

**Channel — the null**

- `head_dim` is `attn_v`'s **output** dim `N`, and `N` reaches cycles through exactly one term: `ceil(N/128)`.
- For every `N ≤ 128` that term is **1**. Pruning 128 → 64, or → 8, does not change it.
- `N` is not weakly present. It is **absent** — which is why the result is *exactly* 1.000×, not 1.03×.
- Channel pruning does shrink `qk` — but `qk` is **4.1% of decode**, because in `LUT_OS_V` qk's `N` is `kv_len` (parallelised) while attn_v's is `head_dim` (serialised over the cache).
- **The axis that saves cycles is the stage that costs nothing.**

**Token — the survivor, and the proof the table is honest**

- `kv_len` is `attn_v`'s reduction dim `K`, entering via `k_eff = ceil(K/MU)` — linear and unbounded.
- So eviction **must** remove cycles, and therefore **must** work. Measured: **15.96× at batch 32.**
- A framework that only explains failures is unfalsifiable. This one commits in **both** directions from the same arithmetic.
- **Same bytes. Same hardware. Opposite outcome. One line of arithmetic apart.**

**Token, read-only — the control**

- Quest reads fewer blocks but stores the full cache → **byte-identical to eviction** on the read path.
- If bytes decided, the two would be indistinguishable. If `k_eff` decides, they should nearly coincide.
- They do: **12.85× vs 15.96×** at batch 32. The residual gap is **storage, not compute**.

**Bit-width — the axis with no null anywhere**

- `qbit` multiplies the *entire* expression; no `ceil` sits outside it to absorb it.
- KV4 → KV3 is 0.75× cycles **and** 0.75× bytes — unconditionally, every context, every batch.
- **Channel pruning has a rounding boundary to die on. Bit-width does not.**

### 2.4 The discriminator

> The question is never *"how many bytes does this remove?"*
> It is *"does it reach `k_eff` or `qbit`?"*

- Touches only DRAM → lands inside §1.4's **1.01–1.07×** batch-1 ceiling.
- Lowers the compute roof → survives.
- Checkable on paper before a single simulation runs.

### 2.5 Bit-width: confirmed by the paper, and bounded by it

- **The mechanism is confirmed by construction.** The Omni-LUT paper builds KV3 and KV2 and states it in our terms: KV2 "eliminates 33% (vs. KV3) and 50% (vs. KV4) of the computational workload from these AA-GEMM bottlenecks" — that is `cycles ∝ qbit`.
- **The payoff is smaller than the halving implies.** Its Fig. 10 energy bars put KV2 within ≈ **1.07×** of KV4 on a decode-heavy workload (2K in, 2048 out) and ≈ **1.14×** on a prefill-heavy one (8K in, 256 out) — for an axis that halves bytes *and* cycles.
- Same lesson as §1.4, on a different metric: even the axis with no null runs into the fact that neither KV bytes nor AA cycles are the whole of decode.
- **It composes with eviction rather than competing.** `cycles ∝ k_eff × qbit` → the two axes are orthogonal and **multiply**; evict-1024 at KV3 ≈ the *product*, not the larger.
- Contrast: channel + token pruning both claim bytes from the same tensor, so their savings sub-add.
- **Still open on our side:** the paper measures *energy*. The **latency** effect of KV3/KV2, and how it stacks with eviction, is what our model can add.

### 2.6 All of it was measured in the wrong place

![KV reduction vs batch](analysis/memory/kv_batch.png)

- **Weight traffic is constant in batch** — 7.65 GB, read once, however many sequences are in flight.
- **KV traffic scales with batch** — every sequence carries its own cache.
- So KV's *share* of DRAM, and the payoff for cutting it, **grows with batch**.
- `evict-1024`: **2.46× at batch 1 → 15.96× at batch 32.**

### 2.7 Eviction is the exception that still obeys the rule

- Its batch-1 figure is **1.45×** under the pipelined model — inside the same ceiling as everything else here.
- The **15.96× is a batch-32 result**, earned exactly where §1's triangle says bytes finally bind.
- And at batch 32 it wins twice: eviction also lowers the **compute** roof, since `kv_len` is `attn_v`'s reduction dim.
- **A technique that moves both roofs is robust. One that moves only the slack roof is not.**

### 2.8 What §2 does *not* establish

- **The channel null is `attn_v`-specific** — it depends on `head_dim ≤ 128` making `n_tiles = 1`. A wider head, or `qk` instead of `attn_v`, changes the row.
- **"Compute-bound" is scoped to GQA + KV4** (§1.5). On MHA at KV16 the KV share is 25–84%, not 2.9%, and the ceiling is 1.11–1.46×.
- **The batch-1 numbers assume the serial roofline.** Under pipelining, eviction's 2.46× becomes 1.452× — which strengthens §2.3 and weakens nothing in §2.6.

---

*Findings-only cut: `presentation.md`. Full derivations, model-change record, staged revert points and open gaps: `study.md`.*
