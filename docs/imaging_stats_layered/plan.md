# imaging_stats_layered — Implementation Plan

## Complexity Assessment

**Task Complexity**: MEDIUM  
**Branch**: `refactor/imaging-stats-layered-design`  
**Implementation Approach**: module by module, tests first. Extend existing test files
rather than creating new ones. Each split is mechanical: extract body into `_fn_array`,
update public wrapper to call it. No logic changes.  
**Key Challenges**: `laterality_ratio` and `image_stats` need a small signature adaptation
(pass orientation tuple / voxel_vol float to the core); `extract_roi` is deferred.  
**Resource Requirements**: no new dependencies.

---

## Progress Update Requirements

**CRITICAL**: After completing any task:
1. Mark checkbox `[x]` in this file immediately
2. Run `pytest bcblib/tests/ -x -q 2>&1 | tail -n 30` to verify no regressions
3. Only mark complete after green tests

---

## Task Breakdown

### R1 — `stats.py` refactor ║ bcblib/tests/test_imaging_stats.py ║ M

*Highest value — these are the functions most likely to be called with pre-loaded arrays.*

- [ ] R1.1: Add `_centre_of_gravity_array(data, round_coord)` → tuple; update `centre_of_gravity` to delegate
- [ ] R1.2: Add `_volume_count_array(data, threshold, ratio)` → int|float; update `volume_count`
- [ ] R1.3: Add `_laterality_ratio_array(data, orientation)` → float; update `laterality_ratio`
  to extract `nib.aff2axcodes(nii.affine)` before delegating
- [ ] R1.4: Add `_reduce_axis_array(data, method, axis)` → np.ndarray; update `reduce_axis`
- [ ] R1.5: Add `_image_stats_array(data, voxel_vol)` → dict; update `image_stats`
  to compute `voxel_vol = float(np.abs(np.linalg.det(nii.affine[:3, :3])))` before delegating
- [ ] R1.6: Add `_percentile_array(data, q)` → float; update `percentile`
- [ ] R1.7: Add `_robust_range_array(data, low, high)` → tuple; update `robust_range`
- [ ] R1.8: Add `_histogram_array(data, bins)` → (counts, edges); update `histogram`
- [ ] R1.9: Write / extend `test_imaging_stats.py` — for each `_fn_array`, add one test
  that calls the array-level function directly with a synthetic numpy array; verify the
  public wrapper test still passes (no logic change → existing tests cover it)
- [ ] R1.10: All R1 tests green, no regressions

---

### R2 — `math.py` arithmetic refactor ║ bcblib/tests/test_imaging_math.py ║ S

*The morphology ops are already layered. Only arith + binarize/threshold need work.*

- [ ] R2.1: Add `_binarize_array(data, thr)` → np.ndarray; update `binarize`
- [ ] R2.2: Add `_threshold_array(data, low, high)` → np.ndarray; update `threshold`
- [ ] R2.3: Add `_add_arrays(a, b)`, `_subtract_arrays(a, b)`, `_multiply_arrays(a, b)`;
  update `add`, `subtract`, `multiply`
- [ ] R2.4: Extend test file — array-level tests for each new core function
- [ ] R2.5: All R2 tests green, no regressions

---

### R3 — `manipulate.py` partial refactor ║ bcblib/tests/test_imaging_manipulate.py ║ S

- [ ] R3.1: Add `_merge_arrays(arrays, axis)` → np.ndarray; update `merge_images`
- [ ] R3.2: Add `_split_array(data, axis)` → list[np.ndarray]; update `split_image`
- [ ] R3.3: Extend test file — array-level tests for each new core function
- [ ] R3.4: `extract_roi` — leave as-is (deferred; affine update makes array-only core awkward)
- [ ] R3.5: All R3 tests green, no regressions

---

### R4 — Finalisation ║ — ║ S

- [ ] R4.1: Run full test suite — `pytest bcblib/tests/ -q 2>&1 | tail -n 50`
- [ ] R4.2: Run flake8 on modified files; fix any issues
- [ ] R4.3: Commit on `refactor/imaging-stats-layered-design` with conventional commit message

---

## Acceptance Criteria

1. Every modified function has a `_fn_array` counterpart callable with plain numpy arrays
2. All existing tests pass unchanged (no public API change)
3. New array-level tests cover all extracted cores
4. flake8 clean on modified files
5. `orient.py`, `info.py`, `extract_roi`, `copy_geometry` are untouched

---

## Relationship to damage_profile

- `damage_profile` T5.0 adds `fraction_covered` and `weighted_region_mean` to `stats.py`
  using the correct layered pattern — those are new functions, not part of this refactor
- `damage_profile._stats` calls `_fraction_covered_array` and `_weighted_region_mean_array`
  directly — the same pattern this refactor establishes for existing functions
- This refactor can be merged independently, before or after `damage_profile`
