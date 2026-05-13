# damage_profile — Implementation Plan

## Complexity Assessment

**Task Complexity**: MEDIUM
**Implementation Approach**: Bottom-up TDD. Pure functions first (stats, atlas loading),
then space handling, then integration (core), then download manager, then CLI.
**Key Challenges**: atlas format detection; download consent flow (mock in tests);
nilearn resample across template families; memory for large 4D atlases.
**Resource Requirements**: no new dependencies; nibabel, numpy, pandas, nilearn all present.

---

## Progress Update Requirements

**CRITICAL**: After completing any task:
1. Mark checkbox `[x]` in this file immediately
2. Update TodoWrite status to `completed`
3. Run `pytest bcblib/tests/test_damage_profile.py -x` to verify
4. Only mark complete after green tests

---

## Task Breakdown

### T1 — Data structures and package skeleton ║ test_damage_profile.py ║ M

- [x] T1.1: Create `AtlasSpec` dataclass in `_atlas.py` (source, name, threshold=0.0, label_file=None, space=None)
- [x] T1.2: Create `AtlasInfo` dataclass in `_atlas_manager.py` (full_name, url, size_mb, format, space, citation)
- [x] T1.3: Create stub files: `_atlas.py`, `_atlas_manager.py`, `_space.py`, `_stats.py`, `_core.py`
- [x] T1.4: Update `__init__.py` to export `damage_profile`, `AtlasSpec`, `get_preset_atlas`, `list_preset_atlases`

---

### T2 — Atlas loading: directory format ║ test_damage_profile.py::TestAtlasLoading ║ M

*TDD: write failing tests first.*

- [x] T2.1: Write `TestAtlasLoading::test_directory_loads_all_files` — tmp dir with 3 synthetic NIfTI files
- [x] T2.2: Write `TestAtlasLoading::test_directory_threshold_applied` — values below threshold zeroed/excluded
- [x] T2.3: Write `TestAtlasLoading::test_directory_empty_regions_excluded` — all-zero file excluded from dict
- [x] T2.4: Implement `detect_atlas_format(source)` → `'directory'` | `'4d_nifti'` | `'label_nifti'`
  using `bcblib.imaging.io.is_nifti` for NIfTI detection (reuse existing tool)
- [x] T2.5: Implement `_load_directory(spec)` → `dict[str, np.ndarray]`
- [x] T2.6: Implement `load_atlas(spec)` dispatcher — calls correct loader; all tests green

---

### T3 — Atlas loading: 4D NIfTI format ║ test_damage_profile.py::TestAtlasLoading ║ M

- [x] T3.1: Write `TestAtlasLoading::test_4d_nifti_loads_correct_n_regions`
- [x] T3.2: Write `TestAtlasLoading::test_4d_nifti_names_from_label_file`
- [x] T3.3: Write `TestAtlasLoading::test_4d_nifti_names_auto_numbered_without_label_file`
- [x] T3.4: Implement `_load_4d_nifti(spec)` — one volume per region, uses label_file if provided
- [x] T3.5: All T3 tests green

---

### T4 — Atlas loading: label NIfTI format ║ test_damage_profile.py::TestAtlasLoading ║ S

- [x] T4.1: Write `TestAtlasLoading::test_label_nifti_correct_n_regions`
- [x] T4.2: Write `TestAtlasLoading::test_label_nifti_names_from_label_file`
- [x] T4.3: Implement `_load_label_nifti(spec)` — integer labels, binary weights, uses label_file
- [x] T4.4: All T4 tests green
  **Note**: `_parse_label_file` also handles FSL XML format (`.xml` atlas files bundled with FSL)

---

### T5 — Region statistics ║ test_damage_profile.py::TestRegionStats + test_imaging_stats.py ║ M

*Core algorithm — most important test coverage.*

#### T5.0 — Design audit and new public functions in `bcblib.imaging.stats`

**See also**: `docs/imaging_stats_layered/` — separate branch/plan to retrofit the existing
functions in `stats.py` and `math.py` with the same pattern. That work is independent and
does not need to happen before or after `damage_profile`.

**DESIGN FLAW CONFIRMED**: all existing functions in `bcblib/imaging/stats.py` load a
`NiftiLike` directly and operate on it in a single layer — no underlying numpy-level functions
exist. This means you cannot reuse the core logic with a pre-loaded array.
**Flag for future update**: existing functions (`image_stats`, `percentile`, `reduce_axis`, …)
should each be refactored into a fast numpy core + NiftiLike wrapper. Do NOT refactor them now
(out of scope), but implement the two new functions below with the correct pattern from the start.

- [x] T5.0a: Add `_fraction_covered_array(subject_data: np.ndarray, mask_data: np.ndarray) → float`
  to `bcblib/imaging/stats.py` — pure numpy, no I/O
- [x] T5.0b: Add public `fraction_covered(subject: NiftiLike, mask: NiftiLike) → float` wrapper
  that calls `_fraction_covered_array`
- [x] T5.0c: Add `_weighted_region_mean_array(subject_data: np.ndarray, weight_data: np.ndarray) → float`
  to `bcblib/imaging/stats.py` — pure numpy, returns `nan` when weight sum is 0
- [x] T5.0d: Add public `weighted_region_mean(subject: NiftiLike, weight_map: NiftiLike) → float`
  wrapper that calls `_weighted_region_mean_array`
- [x] T5.0e: Tests in `test_damage_profile.py::TestImagingStatsNewFunctions` covering array-level
  and NiftiLike-level; edge cases: empty mask, uniform weights ≡ mean, weight sum = 0 → nan
- [x] T5.0f: Export `fraction_covered` and `weighted_region_mean` from `bcblib.imaging.__init__`

#### T5.1–T5.9 — `compute_region_stats` in `_stats.py`

*Note: `_stats.py` calls `_fraction_covered_array` and `_weighted_region_mean_array` directly
(the numpy-level functions, not the NiftiLike wrappers) since data is already loaded.*

- [x] T5.1: Write `TestRegionStats::test_mean_overlap_known_values`
- [x] T5.2: Write `TestRegionStats::test_weighted_mean_overlap`
- [x] T5.3: Write `TestRegionStats::test_weighted_mean_uniform_weights_equals_mean`
- [x] T5.4: Write `TestRegionStats::test_fraction_covered`
- [x] T5.5: Write `TestRegionStats::test_min_overlap_voxels_filter`
- [x] T5.6: Write `TestRegionStats::test_output_sorted_descending`
- [x] T5.7: Write `TestRegionStats::test_empty_atlas_returns_empty_df`
- [x] T5.8: Implement `compute_region_stats` using `_fraction_covered_array` and `_weighted_region_mean_array`
- [x] T5.9: All T5 tests green

---

### T6 — Space handling ║ test_damage_profile.py::TestSpaceHandling ║ M

**Three tiers** (determined by `AtlasSpec.space` and subject image space):
1. Same affine → passthrough (no copy)
2. Same template family, different resolution → nilearn `resample_to_img(..., interpolation='nearest')`
3. Cross-template family (MNI152NLin2009cAsym ↔ MNI152NLin6Asym) → TemplateFlow + nitransforms warp

**Dependency decision (RESOLVED)**: add `templateflow` and `nitransforms` to `install_requires`
in `setup.py`. They are hard requirements. T6.8–T6.9 can import them at the top of `_space.py`
without a lazy-import guard.

**Template identifiers used**:
- `MNI152NLin6Asym` = FSL MNI (Disconnectomes, Rojkova, Tian 1mm, Buckner)
- `MNI152NLin2009cAsym` = SPM MNI (Schaefer, AAL, Yeh HCP1065)

- [x] T6.1: Write `TestSpaceHandling::test_same_space_no_resample`
- [x] T6.2: Write `TestSpaceHandling::test_different_resolution_resamples`
- [ ] T6.3: `test_shape_mismatch_raises` — deferred; tier-2 delegates to nilearn which handles shape gracefully
- [x] T6.4: Write `TestSpaceHandling::test_affine_mismatch_error_mode`
- [x] T6.5: Write `TestSpaceHandling::test_affine_mismatch_warn_mode`
- [x] T6.6: Write `TestSpaceHandling::test_cross_template_uses_templateflow`
- [x] T6.7: Write `TestSpaceHandling::test_cross_template_result_shape_matches_subject`
- [x] T6.8: Implement `_detect_template_family(img)` — shape-based detection of MNI6Asym vs MNI2009cAsym
- [x] T6.9: Implement `_apply_templateflow_warp` using `tflow.get()` + `nitransforms`
- [x] T6.10: Implement `check_and_resample` — three tiers dispatched correctly
- [x] T6.11: All T6 tests green

---

### T7 — Core integration ║ test_damage_profile.py::TestDamageProfile ║ M

- [x] T7.1: Write `TestDamageProfile::test_returns_dict_keyed_by_atlas_name`
- [x] T7.2: Write `TestDamageProfile::test_csv_written_per_atlas`
- [x] T7.3: Write `TestDamageProfile::test_accepts_path_string`
- [x] T7.4: Write `TestDamageProfile::test_multiple_atlases_independent`
- [x] T7.5: Write `TestDamageProfile::test_empty_result_when_no_overlap`
- [x] T7.6: Implement `damage_profile(subject_map, atlases, ...)` in `_core.py`
- [x] T7.7: All T7 tests green

---

### T8 — Atlas download manager ║ test_damage_profile.py::TestAtlasManager ║ M

- [x] T8.1: Write `TestAtlasManager::test_get_atlas_dir_returns_home_path`
- [x] T8.2: Write `TestAtlasManager::test_get_preset_atlas_from_cache`
- [x] T8.3: Write `TestAtlasManager::test_get_preset_atlas_jhu_from_fsldir`
- [x] T8.4: Write `TestAtlasManager::test_download_prompts_user`
- [x] T8.5: Write `TestAtlasManager::test_download_skipped_when_user_says_no`
- [x] T8.6: Write `TestAtlasManager::test_assume_yes_skips_prompt`
- [x] T8.7: Write `TestAtlasManager::test_unknown_atlas_name_raises`
- [x] T8.8: Implement `get_atlas_dir()`, `list_preset_atlases()`, `get_preset_atlas()`
- [x] T8.9: Implement `_download_atlas(info, dest)` — urllib download + zip/gz extraction
- [x] T8.10: `PRESET_ATLASES` registry populated (all 7 atlases, URLs confirmed)
  **Note**: label file paths guarded — silently omitted if file absent (e.g. FSL XML not present)
- [x] T8.11: All T8 tests green

---

### T9 — CLI ║ test_damage_profile.py::TestCLI ║ S

- [x] T9.1: Write `TestCLI::test_parse_args_minimal`
- [x] T9.2: Write `TestCLI::test_parse_args_preset`
- [x] T9.3: Write `TestCLI::test_main_writes_csv`
- [x] T9.4: Write `TestCLI::test_parse_args_missing_atlas_raises` and `test_parse_args_atlas_name_count_mismatch_raises`
- [x] T9.5: Implement `parse_args()` and `main()` in `run_damage_profile.py`
- [x] T9.6: Register `bcb-damage-profile` entry point in `setup.py`
- [x] T9.7: All T9 tests green

---

### T10 — Finalisation ║ — ║ S

- [x] T10.1: Full test suite — 250 passed, 0 new errors (pre-existing `test_parcitron` fixture error unchanged)
- [x] T10.2: flake8 clean on all new files
- [x] T10.3: `templateflow` and `nitransforms` added to `install_requires` in `setup.py`
- [x] T10.4: README updated — `bcb-damage-profile` row in CLI table + two usage examples + dependency note
- [ ] T10.5: Commit on `feat/best-overlap-robust-stats` branch (current working branch)

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Download URLs change / go offline | Medium | Medium | Pin URLs; user can always pass path directly via AtlasSpec |
| Yeh atlas (157×189×136) is MNI2009cAsym — needs TemplateFlow warp to FSL MNI | **Confirmed** | Medium | T6 TemplateFlow tier; document that Yeh requires `templateflow` + `nitransforms` |
| Buckner atlas (141×95×87 cerebellar box) needs resample | **Confirmed** | Low | Same MNI world space, just a tight bounding box; nilearn resample_to_img handles cleanly |
| Tian label file format: plain text, one name per line (1-indexed) | **Confirmed** | Low | Parse with `Path.read_text().splitlines()`; label N → line N-1 |
| Buckner label file is TSV with generic names (Network1–7) | **Confirmed** | Low | Parse TSV; consider allowing user to provide a custom label file |
| JHU has no standalone download URL (FSL-bundled only) | **Confirmed** | Medium | If $FSLDIR not set: tell user to install FSL or provide path manually |
| `templateflow` + `nitransforms` install weight | Low | Low | Hard requirements in setup.py; add to T10 checklist |
| `_check_image_space` moves in future best_overlap refactor | Low | Low | Document the import in context.md; acceptable for now |
| `bcblib.imaging.stats` layered design flaw | **Confirmed** | Low | New functions implement correct pattern; existing functions flagged but not refactored (out of scope) |

---

## Milestone Checkpoints

- **After T5** (stats complete): validate hand-calculated values match implementation — pause for review if unexpected behaviour
- **After T7** (core complete): end-to-end test with real Rojkova atlas from BCBToolKit path — `[USER_INPUT]`
- **T8.10**: populate download URLs — requires manual lookup per atlas, flag for user confirmation `[USER_INPUT]`
- **After T10**: full suite + flake8 clean before merge

---

## Acceptance Criteria

1. `damage_profile(map, [AtlasSpec(source=rojkova_path, name='rojkova')])` returns a DataFrame with 68 rows and all required columns
2. `damage_profile` with a binary lesion map produces `weighted_mean_overlap == NaN` (uniform weights = binary)
3. Calling with a preset name (`get_preset_atlas('jhu_wm_prob')`) works when `$FSLDIR` is set
4. All tests pass with ≥ 90% coverage on new code
5. flake8 clean (max-complexity 10)
