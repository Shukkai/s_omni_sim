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

  > **This premise turned out to be wrong, and Stage 3 is what disproved it.** A
  > 4-bit KV entry is `128 × 4/8` = 64 B — exactly one DDR-class burst — so a
  > page-gathering reader is burst-aligned at *every* page size, down to a single
  > token. Selection costs nothing extra on this axis. Building Stage 2 was still
  > what made the question answerable, and it did find a genuinely misaligned case
  > (ThinK's 38 B pruned entry), but the motivating example was not real.

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
2. **Batch was a loop in one half of the model and a dimension in the other.**
   `_calculate_peak_sram` has no batch term ("peak is per element"), yet
   projections are issued as one GEMM with `proj_m = batch × seq_len`, so their
   footprint scaled with batch anyway while attention's did not.
   **Resolved in Stage 1b below.**

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

## Stage 1b — batch as a real capacity axis ✅

**Goal.** Close gap (2) above, so "how much batch fits" becomes answerable.

**Files.** `simulator/simulator.py`, `analysis/capacity/capacity_run.py`,
`analysis/regression/baseline.json` (re-captured).

- `HardwareConfig.sram_batch_model: str = "sequential"` — `"sequential"` is the
  original per-instance behaviour and stays the default; `"concurrent"` makes
  batch elements co-resident during attention, heads still sequential.
- `_calculate_peak_sram` takes `sram_batch` (the workload batch, heads excluded)
  and multiplies the per-instance working set by it — **for AA ops only**. AW
  already carries batch in `M`, so scaling it there too would count batch twice.
- `sram_batch` threaded through `_simulate_matmul` and `_simulate_flash_attention`
  to the QK / Attn·V / flash call sites.

**Result — the claim the study could not previously make.** Largest batch whose
decode working set fits, at 32K context (sweep ceiling 32, so "32" means "≥32"):

| SRAM | dense | budget 4096 | budget 1024 |
|------|-------|-------------|-------------|
| 4 MB | 1 | 8 | 32 |
| 8 MB | 2 | 16 | 32 |
| 16 MB | 4 | 32 | 32 |

Capacity and batch trade one-for-one, and at fixed capacity a KV budget buys
batch directly — 8× at 4 MB for a 4096-entry budget.

**Caveat, stated in the report too.** `"concurrent"` is a *scheduling
assumption*, not a measurement. It is the assumption the projection side of the
model already made, which is why adopting it makes the model self-consistent —
but a design that serialises batch would show none of this.

**Verification.**
- Gate at the `"sequential"` default: zero pre-existing values changed, zero keys
  missing; the sole added key is the new config field. Re-captured (19,896
  values) and re-checked clean.
- Hand-checks: an AA op's peak is exactly linear in batch under `"concurrent"`
  (1×…32×) and exactly flat under `"sequential"`; an AW op is byte-identical
  under both, proving batch is not double-counted; end-to-end decode peak grows
  2.06 → 16.50 MB from batch 1 → 8 at 32K.
- Standing checks 2 and 3 still pass, and Stage 1's overflow hand-check still
  passes unchanged.

**Checkpoint.** `301f9d5` — recorded by the following commit.

---

## Stage 2 — DRAM access granularity ✅

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

**As landed.** `_dram_effective_bytes(logical, run)` rounds each access up to a
burst; `_calculate_memory_access` now tracks DRAM reads as `(bits, run_bytes)`
components — KV, weights and attention scores have very different access shapes,
and lumping them would average that away. `_kv_dram_run_entries` is the hook a
page-selective reader overrides (Stage 3); it defaults to the whole per-head
block, which is what a dense or compacted cache actually reads.

**Result — alignment matters, not just run length.** A dense 4-bit KV entry is
`128 × 4 / 8` = 64 B: exactly two 32 B bursts, so the term is **inert**. It bites
only when the run is unaligned or the burst is coarse:

| access | burst | charged | waste |
|--------|-------|---------|-------|
| dense entry, 64 B | 32 B | 64 B | 1.00× |
| ThinK entry (`d_ret=77`), 38 B | 32 B | 64 B | **1.68×** |
| dense entry, 64 B | 128 B | 128 B | **2.00×** |

End-to-end at 8K context with a 128 B burst: contiguous KV is inert
(1.00×, TPOT 69.629 ms), while a per-entry reader pays 1.09× on decode DRAM and
72.252 ms TPOT — 3.8% slower. That gap is the thing a flat bandwidth model
cannot see, and it is why Stage 3 needed this first.

**Note this sharpens §5's "unverified" caveat rather than settling it.** ThinK's
speedup was computed from K-cache bytes ÷ 51.2 GB/s. A pruned entry is 38 B — the
one shape in the table that is *not* burst-aligned — so a compacted ThinK cache
plausibly gives back part of its saving to burst rounding. Quantifying that needs
the pruned-entry layout pinned down, which the current model does not specify.

**Verification.**
- Gate at `dram_burst_bytes=0`: zero pre-existing values changed, zero keys
  missing; the only additions are `dram_burst_bytes`, `dram_read_eff` and
  `dram_write_eff`. Re-captured (21,540 values) and re-checked clean.
- Hand-checks: the plan's worked example (38 B run, 32 B burst → 64 B) exact;
  16 MB contiguous is inert; `burst=0` is exactly inert; alignment cases as
  tabled above; end-to-end a `_kv_dram_run_entries → 1` subclass moves both DRAM
  bytes and roofline TPOT, proving the rewiring reaches the latency path.
- Standing checks 2 and 3 pass; `think_run.py` still reproduces 55.39 / 70.67 /
  131.82 ms.

**Checkpoint.** `569b8ce` — recorded by the following commit.

---

## Stage 3 — selective attention study ✅

**Goal.** The first study that needs Stage 2 to be honest.

**Files.** New `analysis/selective_attn/selective_attn.py`, `selective_run.py`,
`selective_report.md`. No `simulator/` changes — pure add-on, as predicted.

`SelectiveAttnSimulator(UnitAwareSimulator)` reads `k` of `n` KV *pages* per
token: it clamps the decode GEMMs to the selected entries, overrides
`_kv_dram_run_entries` to one page (the Stage 2 hook), and adds Quest-style
per-page min/max metadata reads over the **full** context.

**Headline: the burst term is inert, and that is the finding.**

A 4-bit KV entry is `128 × 4/8` = **64 B — exactly one DDR-class burst**. So a
page-gathering reader is burst-aligned at every page size, down to `page = 1`
(token-granular, TidalDecode-style). Selection and compacted eviction come out
**byte-identical** at equal retained entries, at every `k` tested:

| pages read | entries | evict (eff) | select (eff) | ratio |
|-----------|---------|-------------|--------------|-------|
| 16 | 256 | 17,913,118,720 | 17,913,118,720 | 1.000× |
| 1024 | 16384 | 21,843,705,856 | 21,843,705,856 | 1.000× |

Granularity here is a **bit-width** property, not a selection property. At 3-bit
KV an entry is 48 B and a single-entry gather does pay 1.33× — but at 4 bits,
`kv_budget.py`'s decision to defer this study turns out to have been *correct
for this hardware*, not an oversight.

**What actually costs: selection metadata.** It scales with the context, not
with what was selected, so its share grows as selection gets more aggressive —
a floor under how far selection can pay. Larger pages amortise it directly:

| page | metadata overhead on decode DRAM |
|------|----------------------------------|
| 16 | 2.2 – 2.6% (grows as `k` shrinks) |
| 64 | 0.5 – 0.6% |

Decode speedup vs dense (page 16, 4-bit KV, 32K context) — burst costs nothing,
metadata takes the top off:

| read | flat model | + burst | + metadata |
|------|-----------|---------|-----------|
| 25% | 1.849× | 1.849× | 1.815× |
| 3% | 2.464× | 2.464× | **2.404×** |

**Verification.** Gate clean **without re-capturing** — the study adds no
`simulator/` changes, so the archive should not move, and it did not. Five
pre-flight assertions in `selective_run.py`: no selection parameters reproduces
dense exactly (bytes and TPOT); page size = full context with one page selected
reproduces dense; selecting every page reproduces dense at any page size; with
burst off, selection and compacted eviction agree exactly; and effective ==
logical whenever the burst term is off. Standing checks 2 and 3 pass.

**Checkpoint.** `d63e355` — recorded by the following commit.

---

## Standing checks — run after *every* stage

1. `python analysis/regression/baseline.py check` → *Identical to the baseline ✓*
   with all new features at their disabled defaults.
2. `sum(cycle_units(...)) == _calculate_cycles(...)` — the cycle model is untouched,
   so a break means the wiring leaked.
3. `analysis/channel_prune_breakdown/think_run.py` — its three pre-flight assertions
   pass, and the dense baseline still reproduces §3's roofline column
   (55.39 / 70.67 / 131.82 ms).
