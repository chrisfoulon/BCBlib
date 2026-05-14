# lesion_features — Implementation Plan

## Complexity Assessment

**Task Complexity**: MEDIUM-LARGE
**Implementation Approach**: Bottom-up TDD. BIDS utilities and space normalization
first (pure functions, easy to test), then subprocess wrapper (mocked), then
orchestration pipelines, then CLIs.
**Key Challenges**: BIDS entity parsing edge cases; subprocess + file-system
coordination for run_disco.sh; optional `ses-` propagation; EBRAINS atlas set
requires adding new presets to `damage_profile._atlas_manager` (separate sub-task).
**Resource Requirements**: no new dependencies; all tools already installed.

---

## Prerequisites (external — must be resolved before marked tasks)

| # | What | Status | Blocks |
|---|------|--------|--------|
| P1 | BCBToolKit `run_disco.sh`: add `-r FROM:TO` flag | PENDING — see `bcbtoolkit_run_disco_instruction.md` | T3 |
| P2 | EBRAINS atlas set: user confirms which preset atlases to include | USER_INPUT_REQUIRED | T6 |

---

## Progress Update Requirements

**CRITICAL**: After completing any task:
1. Mark checkbox `[x]` in this file immediately
2. Run `pytest bcblib/tests/test_lesion_features.py -x` to verify
3. Only mark complete after green tests

---

## Task Breakdown

---

### T1 — Package skeleton and BIDS utilities ║ test_lesion_features.py::TestBidsUtils ║ M ✓

- [x] T1.1–T1.5: Package created; `__init__.py`, `_bids.py`, `_preprocess.py`,
      `_disco.py`, `_pipeline.py`, `_constants.py` implemented
- [x] T1.6–T1.13: TestBidsUtils tests written (13 tests)
- [x] T1.14–T1.18: BIDS utility functions implemented and passing

---

### T2 — Space normalization ║ test_lesion_features.py::TestPreprocess ║ M ✓

- [x] T2.1–T2.8: TestPreprocess tests written (10 tests)
- [x] T2.9–T2.13: All normalisation functions implemented and passing

---

### T3 — Disconnectome runner ║ test_lesion_features.py::TestDiscoRunner ║ M ✓

**Prerequisite**: BCBToolKit P1 must be complete (run_disco.sh `-r` flag).

- [x] T3.1–T3.7: TestDiscoRunner tests written (6 tests)
- [x] T3.8–T3.11: All disco runner functions implemented and passing

---

### T4 — Preprocessing pipeline orchestration ║ test_lesion_features.py::TestPipelines ║ M ✓

- [x] T4.1–T4.3: TestPipelines preprocessing tests written (3 tests)
- [x] T4.4: `preprocess_batch` implemented and passing

---

### T5 — Feature extraction pipeline ║ test_lesion_features.py::TestPipelines ║ M

- [ ] T5.1: Write `TestPipelines::test_extract_features_one_writes_csvs`
      (synthetic lesion + disconnectome, one atlas, confirm CSV written)
- [ ] T5.2: Write `TestPipelines::test_extract_features_one_writes_tsvs`
- [ ] T5.3: Write `TestPipelines::test_extract_features_one_with_ses`
      (ses directory and ses entity in filename)
- [ ] T5.4: Write `TestPipelines::test_extract_features_batch_all_subjects`
- [ ] T5.5: Implement `extract_features_one(sub_id, ses_id, lesion_path, disco_path,
      atlases, output_dir)` → `dict`
      - Calls `damage_profile(lesion_path, atlases, output_dir=None)` →
        writes per-atlas CSVs using `build_lf_csv_path(..., feature_variant="lesion")`
      - Calls `damage_profile(disco_path, atlases, output_dir=None)` →
        writes per-atlas CSVs using `build_lf_csv_path(..., feature_variant="disconnectome")`
      - Saves `_subject_map_stats` DataFrames as TSVs using `build_lf_tsv_path`
      - Returns dict of written file paths
- [ ] T5.6: Implement `extract_features_batch(prep_dir, atlases, output_dir, force=False)`
      → `dict[str, dict]`
      - Discovers normalised lesions in `prep_dir/sub-*/[ses-*/]anat/`
          matching `*_label-lesion_mask.nii.gz`
      - Discovers matching disconnectomes `*_desc-disconnectome.nii.gz`
      - Warns and skips subjects where disconnectome is absent
      - Calls `extract_features_one` for each subject

---

### T6 — EBRAINS atlas preset set ║ damage_profile._atlas_manager ║ S–M

**Prerequisite**: User confirms atlas list (P2).

This task extends `bcblib/tools/damage_profile/_atlas_manager.py` with the atlases
needed for the EBRAINS default run, and defines `EBRAINS_ATLAS_SPECS` in
`bcblib/tools/lesion_features/_constants.py`.

Candidate atlases (to be confirmed by user — match the FC/SC atlas set in EBRAINS doc):
- `aal` — AAL116 (cortical + subcortical, 116 regions)
- `brainnetome` — Brainnetome246 (not yet in presets)
- `jhu_wm_prob` — already in presets ✓
- `jhu_wm_labels` — already in presets ✓
- `tian_s1` — already in presets ✓
- `schaefer_200_7n`, `schaefer_400_7n` — not yet in presets

- [x] T6.1: User confirms final atlas list — 5 presets confirmed (aal, schaefer_200_7n,
      schaefer_400_7n, schaefer_200_tian_s1, schaefer_400_tian_s1); Brainnetome and
      surface-based atlases skipped (can be provided via --atlas)
- [x] T6.2: Added 5 AtlasInfo entries to PRESET_ATLASES; added label_url field to
      AtlasInfo; generalized _download_atlas to use label_url; extended label parsers
      (_parse_csv_labels, _parse_alternating_labels) in _atlas.py
- [x] T6.3: Added 12 new tests in test_damage_profile.py covering new parsers and
      all 5 new preset keys — 82/82 passing
- [x] T6.4: `EBRAINS_ATLAS_SPECS` defined in `_constants.py` (8 keys including
      aal, jhu_wm_prob, jhu_wm_labels, tian_s1, schaefer_200_7n, schaefer_400_7n,
      schaefer_200_tian_s1, schaefer_400_tian_s1)
- [x] T6.5: `get_ebrains_atlas_specs(assume_yes=False)` implemented in `_pipeline.py`

---

### T7 — CLIs ║ bcb-lf-preprocess + bcb-lesion-features ║ S ✓

- [x] T7.1–T7.3: TestCLI smoke tests written (6 tests)
- [x] T7.4: `bcblib/scripts/run_lf_preprocess.py` implemented with all required args
- [x] T7.5: `bcblib/scripts/run_lesion_features.py` implemented with all required args
- [x] Entry points added to setup.py

---

### T8 — Finalization ║ coverage + flake8 + commit ║ S

- [ ] T8.1: Run full test suite, confirm ≥90% coverage for new modules
- [ ] T8.2: `flake8 bcblib/tools/lesion_features/ bcblib/scripts/run_lf_preprocess.py
      bcblib/scripts/run_lesion_features.py` — zero violations
- [ ] T8.3: Update `MAINTENANCE_REGISTRY.md` if any tech debt introduced
- [ ] T8.4: Conventional commit:
      `feat(lesion_features): add bcb-lf-preprocess + bcb-lesion-features pipeline`
- [ ] T8.5: Push branch and open PR against `devel`
