# Memory model plan — SRAM capacity + DRAM access granularity

Working record for the memory-model work. Each stage is **one commit**, pushed
before the next begins, so any stage can be undone on its own:

```
git revert <stage sha>            # undo one stage, keep the later ones
git reset --hard <previous sha>   # rewind to the end of the previous stage
```

Checkpoint SHAs are filled in as each stage lands.

---

## Why

Every KV-reduction claim in `study.md` bottoms out in `bytes / 51.2 GB/s`. Two
gaps follow, both already in that file's TODO:

- **SRAM capacity is computed but never enforced.** `_calculate_peak_sram`
  (`simulator/simulator.py`) fills `OperationMetrics.peak_sram_bytes`, which is only
  ever *reported*; `HardwareConfig` has no capacity field. §4(a)'s batch-8 and
  batch-32 rows therefore assume memory that may not exist, and the study cannot
  make the claim it most wants: "N× the batch fits", not "N× faster per token".
- **DRAM is one flat bandwidth number**, so a packed cache and a scattered gather
  cost the same. Everything modelled so far is genuinely contiguous, so this is
  currently defensible (see §5's "Unverified" bullet) — but it breaks the moment the
  study covers select-without-evict (Quest, TidalDecode, NSA), where a flat model
  makes 1% selection look ~100× cheaper than it is.

Goal is the minimum change that makes the next study honest, not a full memory
simulator. No claim here needs bank conflicts or refresh modelling.

**Architectural note.** This is the first change belonging in `simulator/` rather
than another add-on layer. `cycle_units.py` → `kv_budget.py` → `think_prune.py` is a
subclass chain that worked because each layer only *rewrote shapes*. Memory
modelling edits `_calculate_memory_access` itself — a model correction, not an
analysis — so every existing study re-runs against it. Hence Stage 0 first.

---

## Stage 0 — regression gate ✅

**Goal.** Make it impossible to change the model by accident. Every later stage
adds a feature whose *disabled* default must reproduce today's numbers exactly.

**Files.** `analysis/regression/baseline.py`, `analysis/regression/baseline.json`,
`.gitignore`, `simulator/omni_energy_model.py`.

- `baseline.py` captures `SimulationResults.to_dict()` plus both roofline helpers
  across 6 hardware configs × 6 workloads. Per-execution records are dropped (an op
  runs once per layer per decode step with identical metrics — 129 MB of exact
  duplicates), and a SHA-256 of the *unslimmed* tree is stored so the slimming loses
  nothing. 36 runs / 18,216 values / 670 KB / ~5 s.
- `omni_energy_model.py` now returns `float(energy)`. `interp1d` was returning
  `np.float64`, which reprs as `np.float64(0.035…)` fresh but `0.035…` after a JSON
  round-trip — the gate reported 816 "regressions" that were type artifacts, not
  value changes. Coercing at the source stops numpy scalars leaking into every
  downstream consumer. Numerically inert: `float(np.float64(x)) == x`.
- `compare()` also normalizes both trees through JSON, so the gate is immune to
  whatever any future model returns.
- `.gitignore` gains `!analysis/regression/baseline.json` — analysis outputs stay
  ignored, but the reference archive has to be tracked.

**Verification.** `python analysis/regression/baseline.py check` reports
*Identical to the baseline ✓* — including the full-tree hashes, and against an
archive captured **before** the `float()` coercion, which is what proves that
coercion changed no number.

**Checkpoint.** `2342e90` — revert point for everything that follows.

> A checkpoint line cannot name its own commit, so each stage's SHA is recorded by
> the *next* commit. `git log --oneline` is the tiebreaker if they ever disagree.

---

## Stage 1 — SRAM capacity enforcement ✅

**Goal.** Convert existing latency results into capacity claims — what the
accelerator story actually needs.

**Files.** `simulator/simulator.py`, new `analysis/capacity/capacity_run.py`,
`analysis/regression/baseline.json` (re-captured).

- `HardwareConfig`: `sram_capacity_kb: int = 0` (`0` = unlimited → today's
  behaviour is the default).
- `OperationMetrics`: `sram_overflow: bool` and `sram_refetch_bytes: int`, OR'd /
  summed in `_aggregate_metrics` and emitted by `_metrics_to_dict`.
- `_simulate_matmul`: after `peak_sram_bytes` is set, `_apply_sram_capacity`
  compares against capacity and folds the re-fetch into `dram_read` *before*
  energy and roofline see the operation.
- **Spill policy v1** — no re-tiling. A is the only operand the footprint assumes
  stays resident (B and C are already charged per tile), so on overflow A is
  re-read once per column tile: `A_bytes × (n_tiles − 1)`.
- `_simulate_flash_attention` sets the flag only: it is already tiled to a fixed
  block, so there is no resident operand to spill and the fix would be a smaller
  `Br`/`Bc` — a re-tiling, out of scope for v1.
- `_write_overall_summary` gains a fits/overflows line when capacity is finite.

**Two model gaps this exposed** — both pre-existing in `_calculate_peak_sram`,
both surfaced only once capacity was enforced, and both recorded in
`capacity_run.py`'s docstring rather than papered over:

1. **Prefill holds the entire activation matrix.** `A_bytes = M·K·act/8` with `M`
   = full prefill length, so prefill's working set is O(seq × d_model): 59 MB at
   2K context, 2.1 GB at 32K. It overflows at every plausible capacity, so its
   spill charge (a constant ~770 GB) prices a wrong assumption and is not usable.
   A real accelerator tiles prefill over the sequence; the model does not.
2. **Batch is a loop in one half of the model and a dimension in the other.**
   `_calculate_peak_sram` has no batch term ("peak is per element"), yet
   projections are issued as one GEMM with `proj_m = batch × seq_len`, so the
   footprint scales with batch anyway. Peak went 239 MB → 1.91 GB from batch 1 → 8.

Because of (1), the well-posed capacity question is **decode**, and there the
result is the one the study wanted:

| context | dense | any KV budget ≤ 4096 |
|---------|-------|----------------------|
| 2K / 8K | 1024 KB | 1024 KB |
| 32K     | **2176 KB** | **1024 KB** |

Decode's working set is floored at 924.5 KB by the FFN/projection tiles; the KV
tile only becomes binding past ~16K context. That is exactly where a KV budget
buys a smaller chip — 2.1× less SRAM at 32K — and below it, capacity is not the
constraint at all.

**Policy v1's known non-monotonicity** (documented in-code and in the report):
past 1024 KB at 32K the binding term is the KV tile itself, which needs
re-tiling, so v1 flags the overflow and charges nothing — the *larger* overflow
prices lower. `sram_overflow` is the trustworthy output; `sram_refetch_bytes` is
a first-order cost.

**Verification.**
- Gate at `sram_capacity_kb=0`: **zero pre-existing values changed, zero keys
  missing**; the only differences were 1,644 added keys (all three new fields) and
  the 36 tree hashes that necessarily follow. Baseline then re-captured
  (19,860 values, 736 KB) and re-checked clean.
- Overflow hand-check: capacity one KB below a known 17,833,984 B peak charges
  `16,777,216 × (32 − 1) = 520,093,696` B — exact — folds into `dram_read`
  exactly, reaches the energy model, and a capacity above the peak is inert.
- Standing checks 2 and 3 pass; `think_run.py`'s dense baseline still reproduces
  §3's roofline column (55.39 / 70.67 / 131.82 ms).

**Checkpoint.** `9eaa1db` — recorded by the following commit, per the note
under Stage 0.

---

## Stage 2 — DRAM access granularity ⬜

**Goal.** One term, not a timing model: round every access up to a burst, so a
packed cache and a scattered gather stop costing the same.

**Files.** `simulator/simulator.py`, `analysis/cycle_breakdown/cycle_units.py`.

- `HardwareConfig`: add `dram_burst_bytes: int = 0` (`0` = disabled → the gate).
- `OperationMetrics`: add `dram_read_eff` / `dram_write_eff` — bytes actually moved.
  Keep `dram_read` / `dram_write` as *logical* bytes so existing reports do not
  silently change meaning and both can be shown side by side.
- `_calculate_memory_access`: alongside each byte count, derive the access's
  contiguous *run length* — for a packed KV read, `d_ret * kv_bits / 8` per entry;
  for a selective read, one page. Add
  `_dram_effective_bytes(logical, run, burst) = logical * ceil(run/burst)*burst/run`.
- Rewire the six roofline sites to `*_eff`: `_roofline_analyze_ops`,
  `compute_roofline_latency`, `compute_roofline_latency_breakdown` (×2), and
  `cycle_units.compute_stage_cycle_breakdown` (×2).
- `_calculate_memory_energy` takes `*_eff` too — moving a burst costs its energy
  whether or not the bytes were wanted.

**Found during Stage 1, relevant here.** `dram_power_model.dram_energy` already
models granularity — 1024 B rows, 8 B bursts, plus a fixed per-call `ACT` term —
which is why energy is not exactly linear in bytes (Stage 1's hand-check saw a
63.0× byte ratio give a 62.9999× energy ratio). So the *energy* side is partly
granular already and the *latency* side is not at all. Stage 2 must reconcile the
two rather than add a second, independent burst notion on top.

**Verification.** Gate clean at `dram_burst_bytes=0`. Hand-check one burst: a
scattered KV read of `d_ret * kv_bits / 8` = 38 B against a 32 B burst charges 64 B.

**Checkpoint.** `<stage-2-sha>`

---

## Stage 3 — selective attention study ⬜

**Goal.** The first study that needs Stage 2 to be honest.

**Files.** New `analysis/selective_attn/`.

Same add-on pattern as `compact_breakdown/` and `channel_prune_breakdown/`: a
subclass reading `k` of `n` KV *pages* per token rather than a prefix — Quest,
TidalDecode, NSA. The selection cost and the burst term are the whole result; on a
flat bandwidth model this is indistinguishable from compacted eviction, which is
exactly why `kv_budget.py` deferred it to "a separate mask-granularity study".

**Verification.** Gate still clean (the study adds no `simulator/` changes). At page
size = full context the selective simulator must reproduce the dense baseline
exactly, the same way `ThinKSimulator` does at `d_ret = head_dim`.

**Checkpoint.** `<stage-3-sha>`

---

## Standing checks — run after *every* stage

1. `python analysis/regression/baseline.py check` → *Identical to the baseline ✓*
   with all new features at their disabled defaults.
2. `sum(cycle_units(...)) == _calculate_cycles(...)` — the cycle model is untouched,
   so a break means the wiring leaked.
3. `analysis/channel_prune_breakdown/think_run.py` — its three pre-flight assertions
   pass, and the dense baseline still reproduces §3's roofline column
   (55.39 / 70.67 / 131.82 ms).
