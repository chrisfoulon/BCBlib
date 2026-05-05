# Implementation Plan: Robust Statistics for best_overlap

*LAD Phase 1 — Plan Document*
*Feature slug*: `best_overlap_robust_stats`
*Branch*: `feat/best-overlap-robust-stats`

---

## Task Complexity Assessment

**Task Complexity**: MEDIUM  
**Implementation Approach**: Enhance existing module — change outcome variable, swap likelihood, add ROPE, dual CSV output. No architectural changes to the pipeline.  
**Key Challenges**: (1) outcome variable change cascades into documentation and interpretation; (2) ROPE bounds need user input before implementation; (3) model summary text needs updating to reflect new outcome scale; (4) existing tests assume Normal likelihood and SumProb outcome — all must be updated.  
**Resource Requirements**: ~3–4 h implementation; ~1 h documentation update; ~1 h test update.

---

## Progress Update Requirements

**CRITICAL**: After completing any task:
1. Mark checkbox `[x]` in this file immediately
2. Update TodoWrite status to `completed`
3. Run tests to verify no regressions: `pytest bcblib/tests/test_best_overlap_validation.py -x -q`
4. Only mark complete after green tests

---

## Task List

### T1 — Fix bug: double-escaped newlines in `generate_model_summary` ║ `test_best_overlap_validation.py` ║ S

**Background**: Lines like `f.write("="*60 + "\\n")` write literal `\n` as two characters, not a newline. The output `.txt` file is therefore unreadable.

- [ ] T1.1: Replace all `"\\n"` with `"\n"` in `generate_model_summary`
- [ ] T1.2: Add test asserting that the model summary file contains actual newline characters

---

### T2 — Fix bug: `df` modified in-place in `generate_report` ║ existing test ║ S

- [ ] T2.1: Add `df = df.copy()` at the start of `generate_report`
- [ ] T2.2: Verify the caller's dataframe is not mutated (existing test or new assertion)

---

### T3 — Fix 1: Change outcome variable to `WeightedClusterContribution` ║ `test_best_overlap_validation.py` ║ M

This is the primary statistical fix. `SumProb` is replaced by `WeightedClusterContribution = SumProb / ClusterVolume` as the regression outcome.

- [ ] T3.1: In `run_bayesian_model`, change outcome:
  ```python
  EPS = 1e-9
  y = np.log(df["WeightedClusterContribution"] + EPS)
  ```
- [ ] T3.2: Update docstring of `run_bayesian_model` to reflect new outcome, scale, and interpretation
- [ ] T3.3: Update `generate_report` — `PosteriorMean` now lives on log(WCC) scale; update column comment and plot axis label
- [ ] T3.4: Update `generate_model_summary` — outcome description, scale note
- [ ] T3.5: Update existing tests to expect log(WCC) predictions, not log(SumProb+1)
- [ ] T3.6 [USER_INPUT]: Confirm ROPE default bounds (see context.md Open Questions #1) before proceeding to T5

---

### T4 — Fix 2: Replace Normal with Student-t likelihood ║ `test_best_overlap_validation.py` ║ S

- [ ] T4.1: Add `nu = pm.Exponential("nu", lam=1/29)` prior in `run_bayesian_model`
- [ ] T4.2: Replace `pm.Normal("obs", ...)` with `pm.StudentT("obs", ..., nu=nu, ...)`
- [ ] T4.3: Extract and store posterior `nu` summary in `model_results` dict
- [ ] T4.4: Update `generate_model_summary` to report estimated `nu` with interpretation:
  - `ν < 5`: Strong evidence of heavy tails / outliers present
  - `ν 5–30`: Moderate heteroscedasticity
  - `ν > 30`: Data consistent with Normal assumption
- [ ] T4.5: Add `nu` to `effect_sizes` dict (or separate `robustness` key) in `model_results`
- [ ] T4.6: Update tests — monkeypatch must now handle `StudentT` sampling; check `nu` is present in trace

---

### T5 — Fix 4: ROPE classification ║ `test_best_overlap_validation.py` ║ M

[USER_INPUT — requires ROPE bounds decision from context.md #1 before starting]

- [ ] T5.1: Implement `compute_rope_classification(ci_lower, ci_upper, rope_low, rope_high) -> str`:
  - `"Above_ROPE"`: `ci_lower > rope_high` — connectivity clearly above negligible
  - `"Below_ROPE"`: `ci_upper < rope_low` — connectivity clearly below negligible (negative artifact)
  - `"In_ROPE"`: `ci_lower >= rope_low and ci_upper <= rope_high` — negligible
  - `"Inconclusive"`: CI overlaps ROPE boundary
  - Add NumPy docstring with Kruschke 2018 reference
- [ ] T5.2: Add `rope_bounds` parameter to `run_connectivity_analysis` (default: `None` → compute as ±0.1 SD of `y`)
- [ ] T5.3: Pass `rope_bounds` through to `generate_report`
- [ ] T5.4: In `generate_report`, call `compute_rope_classification` for each row; add `ROPE_Category` column
- [ ] T5.5: Add `ROPE_Bounds` columns (`ROPE_Low`, `ROPE_High`) to raw CSV so analysis is fully reproducible
- [ ] T5.6: Add `--rope_bounds` CLI argument to `run_best_overlap.py` (two floats: low high)
- [ ] T5.7: Unit tests for `compute_rope_classification` (all four categories + edge cases)
- [ ] T5.8: Integration test: `ROPE_Category` column present in output CSV

---

### T6 — Dual CSV output ║ `test_best_overlap_validation.py` ║ M

[USER_INPUT — confirm two-file vs single-file approach from context.md #2]

Assuming two-file approach:

- [ ] T6.1: Define `SIMPLIFIED_COLS` constant:
  ```python
  SIMPLIFIED_COLS = [
      "Target_Region_ID", "Source_Cluster",
      "PosteriorMean", "CredibleInterval_2.5", "CredibleInterval_97.5",
      "ROPE_Category", "StatisticalQuality",
      "RatioOverlap", "WeightedClusterContribution", "P90",
  ]
  # + "Target_Region_Name" if atlas_txt provided
  ```
- [ ] T6.2: In `generate_report`, save simplified CSV as `*_analysis_report.csv` (replaces current)
- [ ] T6.3: Save raw (all-columns) CSV as `*_analysis_report_full.csv`
- [ ] T6.4: Update `translate_cluster_labels` to handle both files (or generate translated versions of both)
- [ ] T6.5: Update `run_connectivity_analysis` docstring — document both output files
- [ ] T6.6: Test: both files exist after run; simplified file contains exactly `SIMPLIFIED_COLS`; full file contains all original columns

---

### T7 — Optional: size-normalised and absolute detection-limit filters ║ `run_best_overlap.py` + `best_overlap.py` ║ S

Two complementary, both-off-by-default filters addressing different sources of noise:

| Parameter | Criterion | What it guards against | Biological justification |
|-----------|-----------|----------------------|--------------------------|
| `--min_ratio_overlap` | `RatioOverlap >= threshold` | Anatomically negligible coverage of a region | Size-normalised: requires a minimum *fraction* of region voxels to have preserved connections, regardless of absolute region size. Avoids penalising small regions unfairly. |
| `--min_overlap_voxels` | `OverlapVolume >= threshold` | Single-voxel partial-volume / registration artefacts | Absolute floor: one voxel of overlap is indistinguishable from registration error at 2 mm isotropic resolution in any region. |

Both filters applied with **AND** logic: a pair must satisfy *both* thresholds to be retained. When only one is set, only that one is applied.

**Why `RatioOverlap` and not `WeightedClusterContribution`?** WCC is the regression *outcome*; filtering on it before fitting would mean "exclude observations below a minimum signal level on the response variable" — a direct analogue of p-hacking. `RatioOverlap` is a *predictor*, not the outcome, and represents a spatial coverage criterion (what fraction of the region has any preserved connection at all), which is clearly a data-quality / anatomical relevance question independent of the signal magnitude.

- [ ] T7.1: Add `min_ratio_overlap: float | None = None` and `min_overlap_voxels: int | None = None` parameters to `run_connectivity_analysis`
- [ ] T7.2: Apply both filters after feature computation, before model fitting:
  ```python
  mask = pd.Series(True, index=combined_df.index)
  if min_ratio_overlap is not None:
      mask &= combined_df["RatioOverlap"] >= min_ratio_overlap
  if min_overlap_voxels is not None:
      mask &= combined_df["OverlapVolume"] >= min_overlap_voxels
  n_removed = (~mask).sum()
  if n_removed > 0:
      pct = 100 * n_removed / len(combined_df)
      print(f"  Detection-limit filter removed {n_removed} pairs ({pct:.1f}%)")
      if pct > 20:
          print("  [WARNING] > 20% of pairs removed. Verify thresholds are "
                "based on spatial resolution, not statistical outcomes.")
  combined_df = combined_df[mask]
  ```
- [ ] T7.3: Add CLI flags to `run_best_overlap.py` with explicit warnings in help text:
  - `--min_ratio_overlap FLOAT`: *"Size-normalised detection filter: exclude region-cluster pairs where fewer than this fraction of region voxels show preserved connections (e.g. 0.01 = 1%). Set based on anatomical relevance grounds, not to improve statistical results."*
  - `--min_overlap_voxels INT`: *"Absolute detection floor: exclude pairs with fewer than N overlap voxels regardless of region size. Guards against single-voxel partial-volume artefacts."*
- [ ] T7.4: Record both filter values (or `None`) in `command_used.txt` and in a `FilteringParameters` section of the model summary
- [ ] T7.5: Unit test: filter applied correctly; > 20% warning triggers; neither filter applied when both are None

---

### T8 — Update `docs/best_overlap.md` ║ documentation ║ M

- [ ] T8.1: Update "Bayesian Regression Model" section — new outcome variable, equation, rationale
- [ ] T8.2: Add "Robust Likelihood" subsection explaining Student-t and `ν` interpretation
- [ ] T8.3: Add "ROPE Classification" subsection with Kruschke 2018 reference and decision table
- [ ] T8.4: Add "Output Files" subsection documenting simplified vs full CSV with column descriptions
- [ ] T8.5: Update "Statistical Quality Categories" section — note that `StatisticalQuality` is now supplemented by `ROPE_Category`
- [ ] T8.6: Update References section — add Lange et al. 1989, Kruschke 2018, Kruschke & Liddell 2018, Makowski et al. 2019
- [ ] T8.7: Update FAQ: "What about negative PosteriorMean values?" — after Fix 1 this should be rarer; update accordingly
- [ ] T8.8: Add "How to cite the statistical methods" section listing all referenced papers

---

## Dependency Order

```
T1 (bug fix) → independent
T2 (bug fix) → independent
T3 (outcome change) → T4 (likelihood change) can follow in parallel
T5 (ROPE) → requires T3 complete (needs outcome scale to set ROPE bounds)
T6 (dual CSV) → requires T5 complete (ROPE_Category is a column in simplified CSV)
T7 (min_overlap_voxels) → requires T3 + T6 complete
T8 (docs) → requires all code tasks complete
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| ROPE bounds choice affects all results | High | Medium | Make user-specifiable; document default clearly; show bounds in output |
| Student-t `ν` sampling is slower | Low | Low | PyMC 5 HMC handles this efficiently; monitor wall-clock time in tests |
| Changing outcome breaks existing test data | High | Medium | Update all test assertions; keep old tests in commented form for reference |
| `log(WCC + ε)` with very small WCC → large negative values | Medium | Medium | Choose ε carefully (1e-9); document that very sparse pairs will have large negative PM and ROPE will flag them |
| Dual CSV adds complexity for downstream scripting | Low | Low | Keep simplified CSV as primary; full CSV is opt-in for advanced users |

---

## Acceptance Criteria

1. No negative PosteriorMean values resulting from the SumProb bimodality (Fix 1 resolves this by changing the scale)
2. Patient results show > 0.1 PosteriorMean variation for more than 50 % of pairs (measurable improvement in patient sensitivity)
3. `ROPE_Category = "Below_ROPE"` correctly identifies the 51 pairs that were systematically negative in the original data
4. Model summary reports estimated `ν` with interpretation
5. Simplified CSV contains ≤ 12 columns; full CSV contains all original columns + new additions
6. All existing tests pass (updated to new outcome scale)
7. New tests: 90 %+ coverage on new functions
8. `docs/best_overlap.md` references all cited methods with DOIs

---

## Sub-plan Split Assessment

7 tasks, ~25 sub-tasks total. Within manageable range. **No split needed.**

---

*[USER_INPUT] required before T5, T6, T7 — see context.md Open Questions*
