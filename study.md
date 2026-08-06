# Omni-LUT Cycle Breakdown Study

**Setup.** LLaMA-3-8B (32 layers, GQA 32/8, d_model 4096, d_ffn 14336),
Omni-LUT-KV4 (32x4 LUT array, W4A16KV4, `AW=AA=OMNI`), 500 MHz, 51.2 GB/s,
batch 1, 256 output tokens, standard attention.

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

**Prefill flips from FFN-bound to attention-bound.** At 2K, fc1+fc2 are 61% of
cycles. At 32K they fall to 20% while attention (qk + attn_v + softmax) takes
73% — attention grows quadratically, the FFN linearly.

**Decode is attention-dominated everywhere**, rising from 60% to 93% as context
grows. `attn_v_matmul` costs far more than `qk_matmul` because in `LUT_OS_V` its
N dimension is `head_dim`=128 while qk's is `kv_len`.

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

**The array is efficient; the overheads are not the problem.** Systolic
fill/drain stays under 1.6% and the accumulator under 0.5%.

**The LGU is nearly free.** It costs *zero* in prefill: `LUT_WS` has no
table-generation term because generation is pipelined into the M-long activation
stream and fully amortized. It only appears in decode's `LUT_OS_V` (3 cycles per
round) and even there stays below 0.7%. The scale-aware LGU buys AA-GEMM support
at essentially no cycle cost.

**The VPU is the real long-context threat.** It grows 7.96% -> 28.75% of prefill
from 2K to 32K, almost entirely softmax. At 32K, softmax alone (85.9 s) is the
single largest prefill stage — larger than any LUT GEMM. Scaling the LUT array
would not help; the bottleneck has moved off it.

**BQU is not measured yet** (see TODO). The placeholder estimate puts online KV
quantization at 4.7 M cycles at 2K prefill and 75.5 M at 32K — ~0.05% of the
phase — and treats it as concurrent with the PE array, so it is excluded from
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

- Covers any uniform-budget, compacted eviction: H2O, SnapKV, StreamingLLM, TOVA.
- H2O keeps the cache dense by refilling evicted slots with new KV, so budget-`k`
  is simply a length-`k` cache.
- Not covered: per-layer / per-head budgets (PyramidKV, Ada-KV), channel pruning
  (ThinK), and select-without-evict (Quest, TidalDecode, NSA) — those keep the
  cache resident, so DRAM and capacity do not shrink with the compute budget.
- Selection/bookkeeping cost is excluded for every method. That is where the
  methods differ from each other; here they all cost the same, which is not true.

### (a) Regime map — where eviction is worth deploying

Ceiling speedup at 20% budget, decode roofline time per token:

| batch \ context | 2K | 8K | 32K |
|---|---:|---:|---:|
| 1 | 1.08x | 1.30x | 1.98x |
| 8 | 1.34x | 2.13x | 3.44x |
| 32 | 1.99x | 3.40x | **4.43x** |

- Driver is KV's share of decode DRAM: **2.9% -> 93.8%** across that grid.
- At batch 1 / 2K, decode reads **2.6 GB of weights** per token vs **80 MB of KV**.
  Perfect eviction buys 1.08x. The technique is dead here.
- Weight traffic amortizes over batch; KV traffic scales with it.
- **Batch 1 is the worst possible case** — and it is what §1–3 simulate.
- Method-independent: this bounds every KV-reduction technique, not just eviction.

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

- The LGU measures 0.33% of decode cycles on a full cache; it reaches ~24% of
  attention cycles at the budgets KV-compression papers headline (PyramidKV
  claims 0.7% cache; SnapKV runs 128 entries).
- 2K / 8K / 32K curves **collapse onto one line** against *absolute* retained
  entries — the knee depends on how many entries survive, not on the original
  context. It is an architectural constant, not a workload artifact.
- Invisible on a GPU, where kernels hide the grouping. Applies to *any* method
  that reduces attention to `k` operands, including select-without-evict.
- Implication: the cost axis of the published accuracy-vs-budget curves does not
  transfer to LUT-based hardware.

### (c) Compaction cost — settled, not a tradeoff

- One-time cost: stream the cache once, write back the survivors.
  Per-token benefit: decode re-reads the whole cache every step.
- Payback = `(1+b)/(1-b)` decode steps — **3.0** at 50% budget, **1.5** at 20%,
  **1.2** at 10%.
- If eviction is decided during prefill, the survivors are the only KV ever
  written to DRAM: no gather at all, and prefill writeback shrinks too
  (**-859 MB** at 32K / 20%).
- So the question is not *"can I afford to compact?"* but **"evict before
  writeback, or write everything and compact later?"** — the former strictly
  dominates.
- Eviction-specific: select-without-evict methods have nothing to compact.

---

## TODO

- **Measure the BQU.** BQU cycles are *not measured yet* — the original
  simulator does not model the BQU at all, so there was nothing to read.
  What the current code does instead: `bqu_metrics()` in `cycle_units.py` is a
  placeholder that charges the BEA one pass per bit-plane
  (`ceil(tokens x d_kv / bqu_width) x q`, from the greedy residual loop of
  Eq. 8-9) and the TSE one min/max reduction pass, Value path only. Throughput
  is *assumed* to be `bqu_width` elements/cycle (default 128) with the bit-plane
  loop serialized. Replace with real numbers from the BQU RTL; until then treat
  the BQU rows as an order-of-magnitude estimate. (`--bqu-width` to tune,
  `--no-bqu` to drop.)
- **Confirm the BQU really is overlapped.** Currently excluded from serial
  latency per Sec. IV-A ("on-the-fly"), but not verified against the RTL
  schedule. At ~0.05% of cycles the choice barely matters today; it would matter
  if the measured BQU turns out much slower.
