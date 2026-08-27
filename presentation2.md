# Omni-LUT — The Memory Model, and What Bounds Decode

## 1. The memory model

### 1.1 The configuration

| | value | source |
|---|---|---|
| PE array | **32 × 4**, 32 RACs/PE, MU=4 | = **4,096 lanes** |
| clock | **500 MHz** | TSMC 7 nm |
| precision | W4 A16 **KV4** | Omni-LUT-KV4 |
| DRAM | **51.2 GB/s** peak, 64 B burst, ~90 ns | DDR5-6400 |
| request queue | swept 16 → 128 outstanding | *not stated in the paper* |
| operand port | **256 B/cycle = 128 GB/s** | `MU × array_n × NUM_RAC × kv_bits` |
| unified buffer | swept 256 KB → unlimited | *not stated in the paper* |
| model | LLaMA-3-8B, 32 layers, **GQA 32:8**, head_dim 128 | §1.8 scopes this |

### 1.2 The datapath

```
DRAM ──► unified buffer ──► BQU ──► LGU ──► PE array ──► accumulator
(KV cache,   (on-chip     (quantise   (build     (32×4,
 weights)     staging)     to BCQ)     the LUT)   32 RAC/PE)
                                                       │
        weight buffer ─────────────────────────────────┘
```

- **Unified buffer** — on-chip staging for activations and KV. Not a cache: no tags, no replacement policy, no reuse tracking.
- **BQU** — quantises K/V to BCQ online, reading from and writing back to the buffer.
- **LGU** — builds the lookup table from a 4-activation group. In OS-V decode **one** LGU broadcasts to all 32 rows; the rest are gated off.
- **PE array** — each PE holds a LUT for a 4-activation group, shared by 32 binary weights read-and-accumulated per cycle.

Three legs are priced — DRAM, SRAM bytes, SRAM capacity — plus, now, the queue depth needed to reach DRAM's peak.

### 1.3 How the KV cache reaches the array

**It does not live on-chip.** The entire KV cache is re-read from DRAM on **every decode step**:

- one KV entry (one token, one head) = `128 × 4/8` = **64 B — exactly one DDR5 burst**
- per token per layer = `2 × 8 kv_heads × 128 × 0.5 B` = **1,024 B**
- per token, all 32 layers = **32 KB**

Per-step KV read = `32 KB × context × batch`, against a **constant 2.55 GB of weights**:

| | 2K ctx | 32K ctx |
|---|---:|---:|
| batch 1 | 0.06 GB (2.4% of DRAM) | 1.00 GB (28.2%) |
| batch 32 | 2.00 GB (44.0%) | 32.00 GB (**92.6%**) |

**The buffer holds one instance's tile, not all of them.** `attn_v` is issued once per `(batch, kv_head)` instance, each streaming its own V tile — 2.00 MB at 32K. Measured peak working set is **2.06 MB at batch 1, 8 and 32 alike** — *constant in batch*. That is the model stating its pipeline assumption: instances stream one at a time, so the buffer needs one tile, never the **512 MB** the full working set would occupy at batch 32.

### 1.4 What scales with batch, and what with context

| | with batch | with context |
|---|---|---|
| **compute** | ×N | grows — `kv_len` is attention's reduction dim |
| **KV DRAM** | ×N | ×N |
| **weight DRAM** | **×1** — read once per step, serves all N | **×1** |

Measured: **weight DRAM is 49.81 ms per token at every cell of the grid.**

Batching multiplies compute by N but DRAM by *less* than N — and how much less is entirely how much of DRAM was weights.

### 1.5 The roofline

![Decode roofline](analysis/memory/roofline.png)

The textbook reading says decode attention is memory-bound: intensity **14.2 FLOP/byte** against a ridge of `2 × 4096 × 500e6 / 51.2e9` = **80 FLOP/byte**, five and a half times inside the bandwidth-limited region.

**That reading is wrong, and the plot shows why.** The 80 FLOP/byte ridge assumes all 4,096 lanes work. `attn_v` is `(M=1, K=kv_len, N=head_dim)` — it lights **one of 32 PE rows** and attains **125 GFLOP/s, 3.1% of peak**, at every batch and every context. Against *that* ceiling the ridge is `125 / 51.2` = **2.4 FLOP/byte**, and 14.2 clears it comfortably.

| op | intensity | attained | of peak |
|---|---:|---:|---:|
| projections + FFN (batch 1) | 4.0 | 1,014–4,085 GFLOP/s | 25–99% |
| projections + FFN (batch 32) | **128.0** | ~3,940 GFLOP/s | 96% |
| `qk_matmul` | 14.2 | 2,774 GFLOP/s | 68% |
| **`attn_v_matmul`** | 14.2 | **128 GFLOP/s** | **3.1%** |

- **`attn_v` is compute-bound not because it does much arithmetic, but because the array is bad at this shape.**
- **Batching moves the projections across the ridge**: intensity 4.0 at batch 1 (weight-dominated, DRAM-bound) → 128.0 at batch 32 (compute-bound). That single number is the whole batch story.
- Attention's intensity is **14.2 at every batch** — both its FLOPs and its KV bytes scale with N, so batching cannot move it.
- The identity `C/D = intensity ÷ effective ridge` reproduces the grid exactly: at b1/2K, `4.3 ÷ 28.7` = 0.15; at b1/32K, `7.3 ÷ 7.29` = 1.00.

### 1.6 The third leg: SRAM

Charged at the array's own 128 GB/s operand port:

| batch | ctx | compute | DRAM | SRAM | SRAM/max |
|---:|---:|---:|---:|---:|---:|
| 1 | 2K | 7.7 ms | **51.3 ms** | 24.9 ms | 0.49 |
| 1 | 32K | 73.3 ms | **73.4 ms** | 61.3 ms | 0.83 |
| 8 | 2K | 57.5 ms | **61.6 ms** | 59.6 ms | **0.97** |
| 32 | 32K | **2331.5 ms** | 804.8 ms | 1342.5 ms | 0.58 |

- **SRAM never sets the roofline — but by 3%, not comfortably.** Its worst cell flips at **124 GB/s against a 128 GB/s port**.
- The test **over-charges**: A-reads, B-reads and accumulator traffic are lumped against a *single* port. A real design splits them.
- **A finite buffer barely touches decode.** DRAM is unchanged from unlimited down to **1 MB**, moving 2.7% only at 256 KB — and never from attention. Prefill DRAM rises **3.3× at 8 MB**: decode re-reads everything every step, so a small buffer destroys no reuse; prefill has reuse and loses it.

### 1.7 The fourth leg: DRAM latency — and it decides the verdict

51.2 GB/s is a **peak**. Reaching it needs enough reads in flight to cover the round trip (Little's law):

```
reachable_bw = min(peak, outstanding × burst / latency)
```

| latency | reads in flight needed for 51.2 GB/s |
|---:|---:|
| 60 ns | 48 |
| **90 ns** | **72** (4.5 KB in the air at all times) |
| 120 ns | 96 |

| queue depth | reachable | of peak |
|---:|---:|---:|
| 16 | 11.4 GB/s | 22% |
| **32** | **22.8 GB/s** | **44%** |
| 64 | 45.5 GB/s | 89% |
| 72+ | 51.2 GB/s | 100% |

**Bandwidth is a property of the DRAM *and the requester*.** And it moves the grid:

| profile | effective BW | memory-bound cells | C/D at b1/32K |
|---|---:|---:|---:|
| no latency term | 51.2 GB/s | **11 of 30** | 1.00 |
| 90 ns, 128 deep | 51.2 GB/s | 11 of 30 | 1.00 |
| 90 ns, 64 deep | 45.5 GB/s | 13 of 30 | 0.89 |
| **90 ns, 32 deep** | **22.8 GB/s** | **21 of 30** | **0.44** |
| **90 ns, 16 deep** | **11.4 GB/s** | **30 of 30** | **0.22** |

> **The compute-bound triangle was an artefact of assuming an infinitely deep request queue.**

### 1.8 The verdict

> **Under ideal memory: memory-bound in a low-batch, short-context triangle (11 of 30 cells), compute-bound elsewhere.**
> **Under a realistic 32-deep request queue: memory-bound in 21 of 30 cells.**
> **Under a 16-deep queue: memory-bound everywhere.**
> **SRAM never decides it — neither capacity above ~2 MB nor the operand port at its design rate. What decides it is DRAM, and how deep a queue you can afford to keep full.**

Ideal-memory grid (compute ÷ DRAM; below 1.00 the array waits):

| batch | 2K | 4K | 8K | 16K | 32K |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.15 | 0.23 | 0.38 | 0.62 | 1.00 |
| 2 | 0.28 | 0.43 | 0.67 | **1.04** | 1.51 |
| 4 | 0.52 | 0.76 | **1.12** | 1.56 | 2.02 |
| 8 | 0.93 | **1.28** | 1.69 | 2.10 | 2.44 |
| 16 | **1.57** | 1.93 | 2.27 | 2.54 | 2.73 |
| 32 | 2.37 | 2.60 | 2.74 | 2.84 | 2.90 |

At a 32-deep queue the same grid:

| batch | 2K | 4K | 8K | 16K | 32K |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.07 | 0.10 | 0.17 | 0.28 | 0.44 |
| 2 | 0.12 | 0.19 | 0.30 | 0.46 | 0.67 |
| 4 | 0.23 | 0.34 | 0.50 | 0.70 | 0.90 |
| 8 | 0.42 | 0.57 | 0.75 | 0.94 | **1.09** |
| 16 | 0.70 | 0.86 | **1.01** | **1.13** | **1.21** |
| 32 | **1.05** | **1.16** | **1.22** | **1.26** | **1.29** |

**And inside the memory-bound region the binding resource is weights, not KV.** At 2K / batch 1: weights **97.1%** of DRAM, KV **2.9%**, array idle **86%** — and latency does not change that, it deepens it.

### 1.9 Scope, and what is still idealised

**This is a GQA result.** The Omni-LUT paper evaluates OPT-1.3B/2.7B/6.7B/13B/30B and LLaMA2-7B/13B — **all MHA**:

| configuration | max `C/D` over 30 cells | compute-bound cells |
|---|---:|---|
| LLaMA-3-8B GQA, KV4 | 2.90 | the triangle |
| OPT-6.7B **MHA**, KV4 | 1.07 | 1 of 30 |
| OPT-6.7B **MHA**, KV16 | **0.94** | **none** |

**So there were two independent reasons this model disagreed with the literature about bound-ness, and either alone is sufficient:** the paper measures MHA, and no roofline here charged for latency. Under either correction, decode is memory-bound where the paper says it is.

**What remains idealised:**

1. **The latency term is a throughput clamp, not a latency model.** Steady-state, streaming reads. No dependent chains, no row misses, no refresh, and no allowance for a gather whose requests cannot all be in flight. **A scattered KV read does worse than modelled here.**
2. **KV buffer pressure is inexpressible.** The spill charge is `A_bytes × (n_tiles − 1)`, and `attn_v` has `N = head_dim = 128` → `n_tiles = 1` → **identically zero at any buffer size**, against a 2.06 MB working set. The op that is 88% of decode cycles is the one that cannot be charged for buffering.
3. **The prefill spill charge prices an assumed loop order.** Read prefill capacity numbers as *the predicate fires here*, not as *prefill costs this much*.

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

<sub>\* KV2 vs KV4, from the Omni-LUT paper's Fig. 10 energy bars — the one row measured elsewhere, and in a different unit. Every other figure is TPOT from our own model.</sub>

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
- `evict-1024`: **2.46× at batch 1 → 15.96× at batch 32.**

### 2.8 Eviction is the exception that still obeys the rule

- Its batch-1 figure is **1.45×** under the pipelined model — inside the same ceiling as everything else.
- The **15.96× is a batch-32 result**, earned exactly where bytes finally bind.
- At batch 32 it wins twice: eviction also lowers the **compute** roof, since `kv_len` is `attn_v`'s reduction dim.
- **A technique that moves both roofs is robust. One that moves only the slack roof is not.**

### 2.9 What §2 does *not* establish

- **The channel null is `attn_v`-specific** — it depends on `head_dim ≤ 128` making `n_tiles = 1`.
- **Scoped to GQA + KV4** (§1.9). On MHA at KV16 the KV share is 25–84%, not 2.9%, and the ideal-memory ceiling is 1.11–1.46× before any latency correction.
- **The batch-1 numbers assume the serial roofline.** Under pipelining, eviction's 2.46× becomes 1.452×.
- **A scattered read is charged optimistically** (§1.9). Selective-KV methods that gather non-contiguous blocks cannot keep the request queue as full as a streaming read, so their true cost is above what any table here shows.
