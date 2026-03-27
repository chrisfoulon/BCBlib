# Implementation Plan — Balanced Dataset Splitting

**Feature slug**: `dataset-splitting`
**Complexity**: MEDIUM
**Approach**: TDD — write failing tests first, implement until green
**Baseline**: 167 tests collected, 166 passed, 1 pre-existing error (test_parcitron.py, unrelated)
**Target**: 167 + 13 new tests = 180 collected, all passed

---

## Progress Update Requirements

**CRITICAL**: After completing any task:
1. Mark checkbox `[x]` in this file immediately
2. Update TodoWrite status to `completed`
3. Run `pytest -q --tb=short` to verify no regressions
4. Only mark complete after tests pass

---

## Tasks

- [ ] **Task 1** ║ `bcblib/tools/misc.py` ║ Add frozen-implementation comment ║ S
  - [ ] 1.1: Add comment block above `create_balanced_split`
  - [ ] 1.2: Add comment block above `permutation_balanced_splits`
  - [ ] 1.3: Verify no logic changes (`git diff` shows only comment additions)

- [ ] **Task 2** ║ `bcblib/tests/test_dataset_splitting.py` ║ Write full test suite (TDD red phase) ║ M
  - [ ] 2.1: `TestOutputStructure.test_output_structure` — n_splits folds, total count preserved, no index duplicated across folds
  - [ ] 2.2: `TestCountBalance.test_count_balance` — each fold has floor/ceil subjects per group (±1)
  - [ ] 2.3: `TestCovariatBalance.test_covariate_balance` — fixed seed, optimised split scores ≥ single random split
  - [ ] 2.4: `TestReproducibility.test_reproducibility` — same seed → same fold indices
  - [ ] 2.5: `TestReproducibility.test_different_seeds` — different seeds → different fold indices
  - [ ] 2.6: `TestCovariates.test_single_covariate` — covariates dict with one entry
  - [ ] 2.7: `TestCovariates.test_multiple_covariates` — covariates dict with two+ entries
  - [ ] 2.8: `TestScoring.test_degenerate_score` — `_score_split` returns 0.0 when KW cannot be computed
  - [ ] 2.9: `TestScoring.test_score_range` — best_score is in [0, 1]
    - [ ] 2.10: `TestValidation.test_too_few_subjects_raises` — ValueError when a group has < n_splits subjects, message includes offending group name and count
  - [ ] 2.11: `TestValidation.test_strict_false_skips_check` — `strict=False` does not raise even when a group has < n_splits subjects
  - [ ] 2.12: Confirm all 11 new tests are collected and fail (import error expected at this stage)

- [ ] **Task 3** ║ `bcblib/tools/dataset_splitting.py` ║ Implement until tests green ║ M
  - [ ] 3.1: Implement `_score_split(folds, covariates)` — explicit pre-check before calling KW:
        check each covariate has ≥2 non-empty folds; if not, return 0.0 (worst score)
  - [ ] 3.2: Implement `permutation_balanced_splits` input validation — explicit pre-check with
        `strict=True` (default): raise `ValueError` listing each offending group with its count,
        e.g. `"Group 'True' has 3 subjects but n_splits=5"`. When `strict=False`, skip the check
        and let degenerate splits score 0.0 naturally via `_score_split`.
  - [ ] 3.3: Implement round-robin fold assignment within groups
  - [ ] 3.4: Implement permutation loop with `numpy.random.default_rng(seed)`
  - [ ] 3.5: Run `pytest bcblib/tests/test_dataset_splitting.py -v` — all tests green
  - [ ] 3.6: Run `flake8 bcblib/tools/dataset_splitting.py` — zero violations

- [ ] **Task 4** ║ `bcblib/tests/test_backward_compat.py` ║ Add misc.py regression tests ║ S
  - [ ] 4.1: Add `TestMiscSplitRegression` class
  - [ ] 4.2: `test_import_still_works` — `from bcblib.tools.misc import permutation_balanced_splits` succeeds
  - [ ] 4.3: `test_output_format` — misc.py version returns a list of 5 dicts (original format)
  - [ ] 4.4: Run `pytest bcblib/tests/test_backward_compat.py -v` — all passing

- [ ] **Task 5** ║ `bcblib/scripts/run_dataset_splitting.py` + `setup.py` ║ CLI ║ S
  - [ ] 5.1: Implement `parse_args()` — `--input`, `--group-col`, `--covariate-cols`, `--n-splits`, `--n-permutations`, `--seed`, `--output`
  - [ ] 5.2: Implement `main()` — read CSV, call `permutation_balanced_splits`, write CSV + fold column
  - [ ] 5.3: Register `bcb-dataset-split = bcblib.scripts.run_dataset_splitting:main` in `setup.py`
  - [ ] 5.4: Smoke test: `python -m bcblib.scripts.run_dataset_splitting --help` exits 0

---

## Final Validation

- [ ] `pytest -q --tb=short` — 180 collected, all passed, 1 pre-existing error unchanged
- [ ] `flake8 bcblib/tools/dataset_splitting.py bcblib/scripts/run_dataset_splitting.py` — zero violations
- [ ] `git diff bcblib/tools/misc.py` — comment additions only, no logic changes

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| KW edge cases (single-value groups, NaN) | Medium | Explicit pre-check in `_score_split` (≥2 non-empty folds per covariate); return 0.0 if not met |
| test_parcitron.py pre-existing error bleeds | Low | Run new tests in isolation first |
| `bcb-split` name conflict | None | Already identified; using `bcb-dataset-split` |
| misc.py regression (comment misplaced) | Low | `git diff` check in Task 1.3 |

---

## Resolved Decisions

- **Task 3.1/3.2**: KW degenerate handling — explicit pre-check (not try/except). `strict=True`
  raises `ValueError` with per-group counts; `strict=False` skips validation and lets degenerate
  splits score 0.0 naturally.
- **Task 5.3**: Register `bcb-dataset-split` in `setup.py` entry_points.
