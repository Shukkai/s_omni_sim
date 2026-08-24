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

**Files.** `simulator/simulator.py`, new `analysis/memory/capacity_run.py`,
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
`analysis/memory/capacity_run.py`'s docstring rather than papered over:

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

**Files.** `simulator/simulator.py`, `analysis/memory/capacity_run.py`,
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

**Files.** New `analysis/memory/selective_attn.py`, `selective_run.py`,
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

## Stage 4 — KV residency across decode steps ✅

**Goal.** Stop charging for re-reads a real design would not pay.

**Files.** `simulator/simulator.py`, new `analysis/memory/residency_run.py`,
`analysis/regression/baseline.json` (re-captured).

**The gap.** `_calculate_memory_access` charged the decode KV read as
`eff_kv_batch × kv_prev × head_dim × kv_bits` on *every decode step*, with no
reference to on-chip capacity — even though the cache is append-only and
entries 1..n−1 are bit-identical between step *t* and *t+1*. Every decode DRAM
number in `study.md` therefore assumed **zero** KV reuse across steps. Stage 1
built the capacity machinery and nothing used it to suppress re-reads.

- `HardwareConfig.kv_sram_kb: int = 0` — bytes of KV held on chip between decode
  steps (`0` = old behaviour). A carve-out of `sram_capacity_kb`, not extra memory.
- `OperationMetrics.kv_resident_bytes` for visibility, summed in `_aggregate_metrics`.
- `_kv_resident_bytes(kv_bytes, share)` — whatever fits is not re-read. QK reads K
  and Attn·V reads V, so each gets half the buffer; FlashAttention reads both in
  one op and passes `share=1.0`.
- **Steady-state model:** the one-time buffer fill after prefill is not charged,
  which slightly favours residency — under 0.4% of KV traffic over 256 output
  tokens, and a constant rather than a per-token term.

**Result 1 — eviction's advantage is real, not a modelling artifact.** This is
what the fix was built to test, and it came back negative. `evict-1024` at batch
32 holds ~16× from a 0 KB buffer to a 128 MB one; at batch 1 it slips only
2.460× → 2.295×. The dense K+V working set is 32 MB per layer at batch 1 and
**1,024 MB at batch 32**, so no plausible buffer holds a meaningful fraction.
Residency cannot rescue a cache that was never going to fit.

**Result 2 — residency is an energy/capacity lever, not a latency one.** At 32K,
batch 8, dense:

| buffer | DRAM saved | energy saved | TPOT gain |
|--------|-----------|--------------|-----------|
| 8 MB | 2.3% | 1.5% | 0.4% |
| 32 MB | 9.2% | 6.1% | 1.5% |
| 128 MB | **36.8%** | **24.6%** | 6.2% |

The bytes go away; the latency mostly does not. `attn_v` is compute-bound under
a 4-bit KV cache, so removing its DRAM traffic moves a term that was not on the
critical path — the same reason ThinK's byte saving never became speed. Report
residency as energy and capacity, never as throughput.

**So eviction and residency are complementary, and the order matters:** eviction
shrinks the working set to a size a buffer can hold (the Stage 1b capacity
result), and residency then removes what is left of the traffic.

**Verification.**
- Gate at `kv_sram_kb=0`: zero pre-existing values changed, zero keys missing;
  only `kv_sram_kb` and `kv_resident_bytes` added. Re-captured (22,380 values)
  and re-checked clean.
- Hand-checks: disabled is exactly inert; a buffer ≥ the working set eliminates
  the KV read entirely; a half-sized buffer removes exactly half, to the byte;
  prefill is untouched; and end-to-end the weight traffic does not move while
  decode DRAM falls.
- All prior stages' hand-checks still pass; `think_run.py` still reproduces
  55.39 / 70.67 / 131.82 ms.

**Checkpoint.** `5e3771f` — recorded by the following commit.

---

## Stage 5 — attention score staging + prefill K/V read ✅

**Goal.** The largest DRAM term in the model was a hardcoded assumption, applied
unconditionally and inconsistently.

**Files.** `simulator/simulator.py`, new `analysis/memory/prefill_run.py`,
`analysis/regression/baseline.json` (re-captured).

**The gap.** `QK_MATMUL` wrote the full score matrix to DRAM and `ATTN_V_MATMUL`
read it back, in both phases, regardless of capacity — **99.9% of prefill DRAM at
32K**. Two contradictions rode along: prefill attention read *zero* K/V from DRAM
(the KV branch is gated on `is_decode`, the weights branch is an `elif`, so a
prefill QK reached neither), and the softmax between them read the spilled matrix
for free (`NonGEMMMetrics` has no DRAM fields).

- `score_sram_kb: int = 0` — scores stay on chip iff one query row's vector fits.
  Row granularity, not whole-matrix: a 32K score matrix is 2 GB per instance, so a
  whole-matrix test would be false at every plausible size and inert exactly where
  the traffic is.
- `prefill_kv_dram_read: bool = False` — charges prefill for the K/V it reads.
  Lands **with** the staging: staging alone would make prefill DRAM wrong by an
  unbounded factor *low*, which is more dangerous than the 100×-high it replaces.
- `_score_staged(N)` is consulted by both the write and the read-back, so the two
  sides cannot disagree. It tests the **unmultiplied** width — the write site
  multiplies by `batch_size = batch × num_heads`, and testing the multiplied count
  would scale the buffer with head count so staging never fires.
- Mirrored into `_simulate_flash_attention`, which reads K/V once per Q block.
  Without that, flash would become the artificially cheap path.

**Result — prefill DRAM, batch 1:**

| context | spilling | staged + K/V read | reduction |
|---|---:|---:|---:|
| 2K | 19.8 GB | 2.68 GB | 7.4× |
| 8K | 277.7 GB | 3.09 GB | 90× |
| 32K | **4,401.7 GB** | **4.70 GB** | **937×** |

**Nothing published moves.** Verified by grep: no tracked markdown contains a TTFT
or prefill-DRAM figure, and every study emitting TTFT runs on the flash path, which
never had the spill. TTFT itself is unchanged (1.00× at every context) — prefill is
compute-bound, so removing even 4.4 TB of DRAM traffic does not move it.

**Decode is a much smaller exposure than first thought, and the first estimate was
wrong.** The score traffic must be isolated by *differencing* staged against
unstaged; reading `attn_v.dram_read` directly overstates it ~5× because that field
also carries the V-cache read. Differenced: **11.1% of decode attention DRAM** at
every context, 1.2–10.4% of all decode DRAM depending on batch, TPOT shift
1.005×–1.016×. `study2.md` §7's "KV share" is really *attention* share and is ~11%
scores — a correction to those sections, not an overturning.

**Verification.**
- Gate at both defaults: zero pre-existing values changed, zero keys missing, only
  the two config keys added. Re-captured (22,452 values) and re-checked clean.
- 7 pre-flight assertions, including: the boundary (one KB below the row still
  spills, at the row fully stages); staged prefill writes *exactly* the KV
  writeback and nothing else; staging is inert on AW ops; and the
  **non-configuration** (`score_sram_kb>0`, `prefill_kv_dram_read=False`) is pinned
  by asserting prefill attention reads exactly 0 while the same run wrote a full KV
  cache — the incoherence is demonstrated, not asserted away.
- **Flash convergence is exact**: fully staged + prefill K/V read equals the flash
  path at one Q block, to the byte, at every context. At `block=256` flash costs
  1.2×/3.7×/30× more — its real K/V re-read-per-Q-block cost, and a signal the flash
  path is not being flattered.
- All prior stages' hand-checks pass; `think_run.py` still reproduces
  55.39 / 70.67 / 131.82 ms; `selective_run.py` (5) and `pack_run.py` (9) still pass.

**Design note worth carrying forward.** The buffer must be sized for the *longest*
row the run produces — `context + output_tokens` — because decode grows `kv_len`.
Sized for prefill alone, the scores silently spill again the moment decoding starts.

**Checkpoint.** `00a79b7` — recorded by the following commit.

---

## Stage 5b — unstructured KV masks ✅

**Not in the original plan.** Every KV result so far was measured on a *compacted*
retained set; real masks are irregular, and on a burst-addressed DRAM those are
not the same read. Three hooks, all default-identical, so no field was added and
the gate stayed green without a re-capture:

| hook | default | why it exists |
|---|---|---|
| `_kv_dram_run_bytes(...)` | whole entries | sub-entry runs, so a channel mask is expressible at all |
| `_kv_covering_bytes(...)` | `0` (no clamp) | a gather never costs more than streaming the region it sits in |
| `_dram_effective_bytes(cap_bytes=0)` | `0` | the clamp itself |

**The clamp is the load-bearing part.** Without it a fine mask prices *above* a
dense read — arithmetic, not a cost. With it the failure mode states correctly:
the saving goes to **zero**, not negative.

**Result.** A 4-bit entry is 64 B = one burst, so token-major keeps 100% of a
token mask's saving and 0.0% of a channel mask's; channel-major inverts it
exactly. There is no third layout. Channel pruning is now null on *both* axes
(`study.md` §5 for cycles, this for bytes) — decode TPOT 1.000× at batch 1 and 32.

**Checkpoint.** `40f9071` — recorded by the following commit.

---

## Stage 7 — memory technology + SRAM bandwidth ✅

Landed **before** Stage 6, deliberately: the point of the SRAM term was to find
out whether it is trustworthy, and the answer turned out to be *"in decode yes,
in prefill not until Stage 6 lands"*. Running it first is what established that.

**`simulator/memory_tech.py`** — presets fixing bandwidth *and* burst together,
because they are not independent in hardware and §10 showed the burst is what
decides whether a pruning axis can collect anything. `DDR5-6400` is asserted to
be the simulator's own default, so every prior result is a DDR5 result.

**`sram_bandwidth_gbps: float = 0.0`** — the roofline becomes
`max(compute, DRAM, SRAM)` at all six sites via one shared `_op_roofline_time`,
rather than six inlined `max` calls. **A term that reaches five of six sites is
worse than no term**: the numbers stay plausible and stop being consistent.
`cycle_units._record` gained `sram_time` and a three-way `bound`.

**Shipped inert, against the plan's `128.0`.** The plan chose 128 GB/s with the
consequence in the headline table. Measuring it first is what changed the call:
at 128 GB/s decode moves 1.074× (sound, and a real result) but prefill moves
**4.35×** off a 113,670 GB SRAM figure that is the untiled-A defect, not
hardware. Defaulting it on would have written a known modelling bug into every
TTFT in the repo. **Flipping the default is a one-line change once Stage 6
lands**, and §11(c) states exactly what it costs.

**Baseline re-captured** — only the 36 new `hw.sram_bandwidth_gbps` keys and the
36 per-entry `full_sha256` moved; no metric value changed, verified by filtering
the diff.

**Checkpoint.** Recorded by the following commit.

---

## Standing checks — run after *every* stage

1. `python analysis/regression/baseline.py check` → *Identical to the baseline ✓*
   with all new features at their disabled defaults.
2. `sum(cycle_units(...)) == _calculate_cycles(...)` — the cycle model is untouched,
   so a break means the wiring leaked.
3. `analysis/channel_prune_breakdown/think_run.py` — its three pre-flight assertions
   pass, and the dense baseline still reproduces §3's roofline column
   (55.39 / 70.67 / 131.82 ms).
