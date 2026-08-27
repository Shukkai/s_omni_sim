# Omni-LUT — The Memory Model, and What Bounds Decode

## 1. The memory model

### 1.1 The configuration being simulated

| | value | where it comes from |
|---|---|---|
| PE array | **32 × 4**, 32 RACs/PE, MU=4 | = **4,096 lanes**, matched to a 64×64 MAC array |
| clock | **500 MHz** | paper, TSMC 7 nm |
| precision | W4 A16 **KV4** | Omni-LUT-KV4 |
| DRAM | **51.2 GB/s**, 64 B burst | DDR5-6400 |
| operand port | **256 B/cycle = 128 GB/s** | `MU × array_n × NUM_RAC × kv_bits` |
| unified buffer | swept 256 KB → unlimited | *not stated in the paper* |
| model | LLaMA-3-8B, 32 layers, **GQA 32:8**, head_dim 128 | §1.8 scopes this |

Balance point: `2 × 4096 × 500e6 / 51.2e9` = **80 FLOP/byte**.

### 1.2 The datapath

```
DRAM ──► unified buffer ──► BQU ──► LGU ──► PE array ──► accumulator
(KV cache,   (on-chip     (quantise   (build     (32×4,
 weights)     staging)     to BCQ)     the LUT)   32 RAC/PE)
                                                       │
        weight buffer ─────────────────────────────────┘
```

- **Unified buffer** — on-chip staging for activations and KV. Not a cache: no tags, no replacement policy, no reuse tracking in this model.
- **BQU** — quantises K/V to BCQ online, reading from and writing back to the unified buffer.
- **LGU** — builds the lookup table from a 4-activation group. In OS-V decode, **one** LGU broadcasts to all 32 rows and the rest are gated off.
- **PE array** — each PE holds a LUT for a 4-activation group, shared by 32 binary weights read-and-accumulated per cycle.

### 1.3 What "buffering" means here — and what it does not

The model prices **three** things and ignores a fourth:

| priced | how |
|---|---|
| DRAM **bytes** | `dram_read_eff + dram_write_eff`, with burst rounding |
| SRAM **bytes** | `sram_read + sram_write`, ÷ `sram_bandwidth_gbps` |
| SRAM **capacity** | `peak_sram_bytes` vs `sram_capacity_kb`, spilling to a refetch charge |
| **not priced** | **latency.** No tRC/tRCD/CAS, no queueing, no MSHRs, no bank conflicts. |

That last row is the load-bearing limitation, and §1.8 returns to it.

### 1.4 How the KV cache actually reaches the array

**It does not live on-chip.** `kv_sram_kb = 0` by default, so **the entire KV cache is re-read from DRAM on every decode step.** For LLaMA-3-8B at KV4:

- one KV entry (one token, one head) = `128 × 4/8` = **64 B — exactly one DDR5 burst**
- per token per layer = `2 × 8 kv_heads × 128 × 0.5 B` = **1,024 B**
- per token, all 32 layers = **32 KB**

So the per-step KV read is `32 KB × context × batch`:

| | 2K ctx | 32K ctx |
|---|---:|---:|
| batch 1 | 0.06 GB | 1.00 GB |
| batch 8 | 0.50 GB | 8.00 GB |
| batch 32 | 2.00 GB | 32.00 GB |

against a **constant 2.55 GB of weights** per step. KV goes from **2.4%** of the DRAM bill to **92.6%** across that grid.

**The buffer holds one instance's tile, not all of them.** `attn_v` is issued once per `(batch, kv_head)` instance, and each streams its own V tile:

| | one V tile | all 8 kv_heads × batch 32 |
|---|---:|---:|
| 2K | 0.12 MB | 32 MB |
| 8K | 0.50 MB | 128 MB |
| 32K | **2.00 MB** | **512 MB** |

Measured `peak_sram` for decode at 32K is **2.06 MB at batch 1, 8 and 32 alike** — *constant in batch*. That is the model telling you its own pipeline assumption: **instances are streamed one at a time, so the buffer needs one tile, never the 512 MB the whole working set would occupy.**

### 1.5 What scales with batch, and what scales with context

Three quantities, three behaviours — this is the whole mechanism:

| | with batch | with context |
|---|---|---|
| **compute** | ×N — each sequence needs its own projections and attention | grows — `kv_len` is attention's reduction dim |
| **KV DRAM** | ×N — each sequence has its own cache | ×N — cache grows with length |
| **weight DRAM** | **×1** — read once per step, serves all N | **×1** — independent of length |

Measured: **weight DRAM is 49.81 ms per token at every cell of the grid.** Every batch, every context, no exception.

So batching multiplies compute by N but DRAM by *less* than N — and how much less depends entirely on how much of DRAM was weights.

### 1.6 The three legs: what actually binds

Decode, per token, with the operand port charged at 128 GB/s:

| batch | ctx | compute | DRAM | SRAM | SRAM/max | **bound by** |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 2K | 7.7 ms | **51.3 ms** | 24.9 ms | 0.49 | DRAM |
| 1 | 8K | 20.9 ms | **55.7 ms** | 32.2 ms | 0.58 | DRAM |
| 1 | 32K | 73.3 ms | **73.4 ms** | 61.3 ms | 0.83 | DRAM |
| 8 | 2K | 57.5 ms | **61.6 ms** | 59.6 ms | **0.97** | DRAM |
| 8 | 8K | **163.7 ms** | 97.0 ms | 117.8 ms | 0.72 | compute |
| 8 | 32K | **582.9 ms** | 238.6 ms | 350.6 ms | 0.60 | compute |
| 32 | 2K | **230.2 ms** | 97.0 ms | 178.6 ms | 0.78 | compute |
| 32 | 8K | **654.8 ms** | 238.6 ms | 411.4 ms | 0.63 | compute |
| 32 | 32K | **2331.5 ms** | 804.8 ms | 1342.5 ms | 0.58 | compute |

- **The operand port never wins — but by 3%, not comfortably.** Its worst cell is 0.97× at batch 8 / 2K, which flips at **124 GB/s against a 128 GB/s port.** Per-cell flip bandwidths: 62 (b1/2K), 107 (b1/32K), **124 (b8/2K)**, 99 (b32/2K), 74 (b32/32K).
- That test **over-charges**: `sram_read + sram_write` lumps A-reads, B-reads and accumulator traffic against a *single* port. A real design splits them, so the true margin is wider.
- **A finite buffer barely touches decode.** DRAM is unchanged from unlimited down to **1 MB**, moving 2.7% only at 256 KB — and never from attention. Prefill DRAM rises **3.3× at 8 MB**. Decode re-reads the whole cache every step, so a small buffer has no reuse to destroy; prefill has reuse and loses it.
- **The regime map is identical at 4 MB + 128 GB/s to the ideal-memory numbers, cell for cell.**

### 1.7 The verdict

> **Memory-bound at low batch and short context; compute-bound everywhere else. The on-chip memory system decides neither.**

| batch | 2K | 4K | 8K | 16K | 32K |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.15 | 0.23 | 0.38 | 0.62 | 1.00 |
| 2 | 0.28 | 0.43 | 0.67 | **1.04** | 1.51 |
| 4 | 0.52 | 0.76 | **1.12** | 1.56 | 2.02 |
| 8 | 0.93 | **1.28** | 1.69 | 2.10 | 2.44 |
| 16 | **1.57** | 1.93 | 2.27 | 2.54 | 2.73 |
| 32 | 2.37 | 2.60 | 2.74 | 2.84 | 2.90 |

- Compute ÷ DRAM. Below 1.00 the array waits on memory. First compute-bound batch: **16 at 2K · 8 at 4K · 4 at 8K · 2 at 16K · 2 at 32K.**
- **The boundary is diagonal because weight DRAM is a constant floor** (49.81 ms) while compute grows on *both* axes. At batch 1, DRAM moves 1.43× from 2K→32K while compute moves 9.56×.
- **And inside the memory-bound corner, the binding resource is weights, not KV.** At 2K / batch 1: weights **97.1%** of DRAM, KV **2.9%**, array idle **86%**.
- **Ceiling if a resource were free:** deleting the *entire* KV cache at batch 1 buys **1.01× at 2K, 1.07× at 32K**. Weight bytes buy **6.80×** inside the triangle; array occupancy **3.12×** outside it. Neither is a KV technique.

### 1.8 Scope, and what the model idealises

**This is a GQA result.** The Omni-LUT paper evaluates OPT-1.3B/2.7B/6.7B/13B/30B and LLaMA2-7B/13B — **all MHA**:

| configuration | max `C/D` over 30 cells | compute-bound cells |
|---|---:|---|
| LLaMA-3-8B GQA, KV4 *(above)* | 2.90 | the triangle |
| OPT-6.7B **MHA**, KV4 | 1.07 | **1 of 30** |
| OPT-6.7B **MHA**, KV16 | **0.94** | **none** |

- On MHA at KV16 decode is memory-bound **in every cell**. The literature's premise is right for the configuration it was written about.
- GQA divides KV traffic by 4 and attention compute by nothing → 4× the arithmetic intensity.
- **More KV bits make decode *more* compute-bound** (`cycles ∝ qbit` on a bit-serial array).
- So the 1.01–1.07× KV ceiling is **what remains after GQA and KV4 have been applied** — on MHA/KV16 it is 1.11×/1.46×.

**Three idealisations, in order of how much they matter:**

1. **No latency term anywhere.** No tRC/tRCD/CAS, no queueing, no MSHRs, no contention. Every number is a bandwidth-and-count roofline.
2. **KV buffer pressure is inexpressible.** The spill charge is `A_bytes × (n_tiles − 1)`, and `attn_v` has `N = head_dim = 128` → `n_tiles = 1` → the charge is **identically zero at any buffer size**, against a 2.06 MB working set. At a 256 KB buffer every other decode op pays a refetch charge and attention pays nothing. **The op that is 88% of decode cycles is the one the model cannot charge for buffering.** Zero extra *bytes* is defensible — there is no reuse to lose — but what a small buffer really costs is keeping the array fed across DRAM latency, and see (1).
3. **The prefill spill charge prices an assumed loop order**, re-reading `A` once per N-tile. Read prefill capacity numbers as *the overflow predicate fires here*, not as *prefill costs this much*.

> **What this licenses:** the compute/memory verdict is robust to on-chip capacity above ~2 MB and to any operand port at or above the array's own feed rate.
> **What it does not:** any claim about how small the unified buffer can be.

*Measured by `analysis/memory/regime_run.py` and `analysis/memory/sram_run.py`.*

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

- Touches only DRAM → lands inside §1.7's **1.01–1.07×** batch-1 ceiling.
- Lowers the compute roof → survives.
- Checkable on paper before a single simulation runs.

### 2.5 Bit-width: confirmed by the paper, and bounded by it

- **The mechanism is confirmed by construction.** The Omni-LUT paper builds KV3 and KV2 and states it in our terms: KV2 "eliminates 33% (vs. KV3) and 50% (vs. KV4) of the computational workload from these AA-GEMM bottlenecks" — that is `cycles ∝ qbit`.
- **The payoff is smaller than the halving implies.** Its Fig. 10 energy bars put KV2 within ≈ **1.07×** of KV4 on a decode-heavy workload (2K in, 2048 out) and ≈ **1.14×** on a prefill-heavy one (8K in, 256 out) — for an axis that halves bytes *and* cycles.
- Same lesson as §1.7, on a different metric: even the axis with no null runs into the fact that neither KV bytes nor AA cycles are the whole of decode.
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
- **"Compute-bound" is scoped to GQA + KV4** (§1.8). On MHA at KV16 the KV share is 25–84%, not 2.9%, and the ceiling is 1.11–1.46×.
- **The batch-1 numbers assume the serial roofline.** Under pipelining, eviction's 2.46× becomes 1.452× — which strengthens §2.3 and weakens nothing in §2.6.

---

*Findings-only cut: `presentation.md`. Full derivations, model-change record, staged revert points and open gaps: `study.md`.*
