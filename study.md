# Omni-LUT Cycle Breakdown Study

**Setup.**

- Model: LLaMA-3-8B — 32 layers, GQA 32/8, d_model 4096, d_ffn 14336.
- Hardware: Omni-LUT-KV4 — 32x4 LUT array, W4A16KV4, `AW=AA=OMNI`,
  500 MHz, 51.2 GB/s.
- Workload: batch 1, 256 output tokens, standard attention (no FlashAttention,
  so `qk_matmul` and `attn_v_matmul` stay separate stages).

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

---

## 4. KV compaction

![Compaction breakdown](analysis/compact_breakdown/compact_breakdown.png)

**Model.** Decode attends to a dense cache of `k` entries — `kv_len -> min(kv_len, k)`.
Covers uniform-budget compacted eviction (H2O, SnapKV, StreamingLLM, TOVA).
Not per-layer/per-head budgets (PyramidKV, Ada-KV), channel pruning (ThinK), or
select-without-evict (Quest, TidalDecode, NSA). Selection cost excluded for all.

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

## TODO

- **Measure the BQU.** Not measured yet — the original simulator does not model
  it at all. `bqu_metrics()` in `cycle_units.py` is a placeholder: BEA one pass
  per bit-plane, TSE one min/max pass (Value path only), assumed `bqu_width`
  elements/cycle. Replace with RTL numbers; treat current rows as
  order-of-magnitude. (`--bqu-width` to tune, `--no-bqu` to drop.)
- **Confirm the BQU is really overlapped.** Excluded from serial latency per
  Sec. IV-A ("on-the-fly"), unverified against the RTL schedule. Only matters if
  the measured BQU is much slower.
- **Build DRAM and SRAM latency models.** DRAM is one flat bandwidth number, so
  scattered and contiguous KV reads look identical. SRAM capacity is reported but
  never enforced, so §4(a)'s batch-32 row assumes memory that may not exist.
  **Capacity enforcement is the higher-value half** — it turns "2x faster per
  token" into "4x the batch fits", the claim this study cannot currently make.
