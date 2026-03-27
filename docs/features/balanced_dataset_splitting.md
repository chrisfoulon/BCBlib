# Feature: Balanced Dataset Splitting

## Status
Planned — implement as `bcblib/tools/dataset_splitting.py`

## Background

A version of this algorithm exists in `bcblib/tools/misc.py` as
`create_balanced_split` + `permutation_balanced_splits`. That version
was used in a published paper and must remain untouched.

This feature creates a **new, clean module** that is a direct
generalisation of the original: `create_balanced_split` groups subjects
by `lesion_cluster` and assigns them round-robin to folds; the new
module does the same with any categorical label array. The chronic/acute
split in the lesionsplit project is a 2-class instance of that same
algorithm. Known issues in the original are also fixed (see Step 1).

The two modules are independent tools — there is no deprecation
relationship. The `misc.py` functions are frozen as the published
implementation; the new module is a general-purpose replacement for new
code.

The lesionsplit project (`/home/chrisfoulon/neuro_apps/lesionsplit`)
will import from this module instead of reimplementing.

---

## What the algorithm does

Monte Carlo permutation search for the most balanced k-way dataset split:

1. Separate subjects into groups by a categorical label array (e.g.
   lesion territory, or chronic vs acute-only). Each group is guaranteed
   a near-equal count per fold (hard constraint, handled by round-robin
   assignment after shuffle).
2. Shuffle within each group and assign round-robin to folds.
3. Score the split using Kruskal-Wallis tests on one or more continuous
   covariates (e.g. acute lesion volume, chronic lesion volume).
4. Repeat N times, keep the split with the highest score.
5. Return the best split (as fold indices) and its score.

This is a form of **covariate-constrained rerandomisation** (Morgan &
Rubin 2012; Pocock & Simon 1975 for the clinical trial analogue).
Lesion volumes are right-skewed / log-normal, so Kruskal-Wallis is
appropriate over ANOVA.

### Relation to the original

`create_balanced_split` in `misc.py` builds a `cluster_dict` keyed by
`lesion_cluster`, shuffles each cluster's subjects, and interleaves them
round-robin into folds. The new `permutation_balanced_splits` does
exactly the same with a generic `groups` array. The `bilat_offset`
logic in the original (hemisphere disambiguation for a unilateral atlas)
is domain-specific preprocessing that the caller is responsible for
— the new function operates on opaque categorical labels only.

---

## Step 0: Add comment to misc.py (do this first, no logic changes)

In `bcblib/tools/misc.py`, add a comment above `create_balanced_split`
and `permutation_balanced_splits` explaining they are the original
published versions. Do **not** change any logic. Example text:

```
NOTE: This is the original implementation used in [paper citation].
It is kept frozen as the published implementation. For new code use
`bcblib.tools.dataset_splitting.permutation_balanced_splits`, which
generalises this algorithm to arbitrary categorical groupings and
multiple covariates.
```

---

## Step 1: Create `bcblib/tools/dataset_splitting.py`

### Known issues in the original to fix

1. **Conjunctive selection criterion (critical)**

   Original:
   ```python
   if best_mean_range > means_range and best_std_range > stds_range and best_pvalue < pval:
   ```
   This requires ALL three conditions to improve simultaneously, causing
   most valid improvements to be rejected. Fix: replace with a single
   composite score. Use the **minimax p-value** (minimum KW p-value
   across all covariates) as the primary score. This handles any number
   of covariates uniformly.

2. **Single covariate only**

   Original aggregates all cluster volumes into one list. The new
   version accepts a dict of covariate arrays and scores each
   independently, then takes the minimum p-value.

3. **No reproducibility**

   Add a `seed` parameter (default 42). Use `numpy.random.default_rng`
   instead of `random.shuffle`.

4. **`best_st` is tracked but unused**

   Remove it. Return `(fold_indices, best_score)` only.

5. **Input and output are tied to the BCBlib cluster concept**

   The new version takes a generic `groups` array (one categorical label
   per subject) and a `covariates` dict (one array per covariate).
   It returns **fold indices** — lists of subject indices — so the
   caller can map them back to whatever data structure they use.
   Any domain-specific preprocessing (e.g. hemisphere disambiguation,
   filtering subjects per covariate) is the caller's responsibility.

6. **print statement in library code**

   The original has a bare `print(...)` call. The new module must not
   use print; use `logging` if progress output is needed.

### Public API to implement

```python
def permutation_balanced_splits(
    groups: "array-like of shape (n_subjects,)",
    covariates: "dict[str, array-like of shape (n_subjects,)]",
    n_splits: int = 5,
    n_permutations: int = 50000,
    seed: int = 42,
    strict: bool = True,
) -> "tuple[list[list[int]], float]":
    """Find a balanced k-way split by Monte Carlo permutation search.

    Generalises the round-robin splitting strategy from
    ``bcblib.tools.misc.create_balanced_split`` (which groups subjects
    by lesion territory) to any categorical grouping.

    Balances both group counts (hard constraint via round-robin) and
    continuous covariate distributions (optimised via Kruskal-Wallis).

    Parameters
    ----------
    groups : array-like, shape (n_subjects,)
        Categorical label for each subject (any hashable values).
        Subjects are shuffled within each unique label and assigned to
        folds round-robin, guaranteeing near-equal counts per fold per
        group. Any preprocessing of these labels (e.g. hemisphere
        disambiguation) is the caller's responsibility.
    covariates : dict[str, array-like]
        Continuous variables to balance across folds, keyed by name.
        Each array must have length n_subjects. Kruskal-Wallis is
        applied to each covariate independently. The selection score
        is min(p_1, p_2, ...) — maximised.
    n_splits : int
        Number of folds. Default 5.
    n_permutations : int
        Number of random shuffles to try. Default 50000.
    seed : int
        Random seed for reproducibility. Default 42.
    strict : bool
        If True (default), raise ValueError before running if any group has
        fewer subjects than n_splits. The error message lists every offending
        group with its subject count, e.g.:
            "The following groups have fewer subjects than n_splits=5:
             - group 'True': 3 subjects
             - group 'ctrl': 2 subjects"
        If False, skip the check. Groups too small to fill every fold will
        produce some folds with 0 subjects for that group; those splits score
        0.0 via _score_split and will not be selected as the best split.
        Use strict=False only when you have verified that underfilled folds
        are acceptable for your use case.

    Returns
    -------
    best_folds : list of list of int
        The best split found: a list of n_splits folds, each fold being
        a list of subject indices into the original input arrays.
    best_score : float
        The minimax KW p-value of the best split. Higher = better
        balanced. Not interpretable as a probability; used as a
        selection criterion only.

    Raises
    ------
    ValueError
        If strict=True and any group has fewer subjects than n_splits.
        Message includes every offending group name and its subject count.

    Notes
    -----
    Score is min(KW_p_covariate_1, ..., KW_p_covariate_N) — maximising
    this minimises the worst-case detectable imbalance across all
    covariates. KW is appropriate for skewed distributions (e.g. lesion
    volumes).

    The round-robin assignment guarantees that each fold receives
    floor(n_group / n_splits) or ceil(n_group / n_splits) subjects from
    each group — count imbalance of at most 1 per group per fold.

    References
    ----------
    Morgan & Rubin (2012). Rerandomization to improve covariate balance
    in experiments. Annals of Statistics.
    """
```

### Internal helper to implement

```python
def _score_split(
    folds: "list[list[int]]",
    covariates: "dict[str, array-like]",
) -> float:
    """Compute minimax Kruskal-Wallis p-value across covariates.

    Parameters
    ----------
    folds : list of list of int
        Subject indices for each fold.
    covariates : dict[str, array-like]
        Covariate arrays indexed by subject position.

    Returns
    -------
    float
        min(p_1, ..., p_K) across all covariates. Higher = better.
        Returns 0.0 (worst score) if the pre-check fails for any
        covariate (fewer than 2 non-empty folds), preventing degenerate
        splits from being selected. The check is explicit (not
        try/except): count non-empty folds per covariate before calling
        kruskal.
    """
```

### Design decisions to preserve

- **Round-robin within groups** — this is the key insight from the
  original that guarantees count balance.
- **Shuffle happens before round-robin** — randomness comes from the
  order within each group, not from random fold assignment.
- **No binning** — covariates are used as continuous values throughout.
- **Degenerate splits score 0.0** — if KW cannot be computed (e.g.
  only one non-empty fold for a covariate), return the worst possible
  score rather than a misleadingly high one.

---

## Step 2: Adapt `balance_splits.py` in lesionsplit

After the BCBlib module is implemented, update
`/home/chrisfoulon/neuro_apps/lesionsplit/lesionsplit/balance_splits.py`
to use `bcblib.tools.dataset_splitting.permutation_balanced_splits`.

The lesionsplit call will be:
```python
import numpy as np
from bcblib.tools.dataset_splitting import permutation_balanced_splits

# Build parallel arrays — one entry per subject
has_chronic = np.array([...])      # bool or int group labels
acute_vols  = np.array([...])      # continuous covariate
chronic_vols = np.array([...])     # continuous covariate (caller decides
                                   # how to handle acute-only subjects,
                                   # e.g. filter them out before passing)

fold_indices, score = permutation_balanced_splits(
    groups=has_chronic,
    covariates={'acute_volume': acute_vols, 'chronic_volume': chronic_vols},
    n_splits=5,
    n_permutations=50000,
    seed=42,
)
```

---

## Step 3: Tests (`bcblib/tests/test_dataset_splitting.py`)

Follow the existing BCBlib test style (pytest classes, tmp_path fixture).

Required tests:

- `test_output_structure`: returns list of n_splits folds (lists of
  int), total subject count preserved, no index appears in more than
  one fold
- `test_count_balance`: each fold has floor or ceil subjects from each
  group (not off by more than 1)
- `test_covariate_balance`: with fixed seed, the optimised split scores
  higher than a single random split (deterministic check, no flakiness)
- `test_reproducibility`: same seed → same fold indices
- `test_different_seeds`: different seeds → different fold indices
- `test_single_covariate`: works when covariates has one entry
- `test_multiple_covariates`: works when covariates has two+ entries
- `test_degenerate_score`: `_score_split` returns 0.0 when KW cannot
  be computed (e.g. only one non-empty fold for a covariate)
- `test_score_range`: best_score is between 0 and 1
- `test_too_few_subjects_raises`: ValueError when a group has fewer
  subjects than n_splits

---

## Step 4: Regression test for misc.py

In `bcblib/tests/test_backward_compat.py`, add a class verifying that
the frozen `misc.py` functions still produce correct output:
- `from bcblib.tools.misc import permutation_balanced_splits` still works
- The misc.py version produces a list of 5 dicts (original output format)

No migration or compatibility shims are needed — the two modules are
independent.

---

## Step 5: CLI (`bcblib/scripts/run_dataset_splitting.py`)

Add a command-line interface following the `argparse` pattern used by
other scripts in `bcblib/scripts/`.

Inputs:
- `--input`: path to a CSV file with one row per subject; must contain
  a group column and one or more covariate columns
- `--group-col`: name of the CSV column to use as group labels
- `--covariate-cols`: one or more column names to use as covariates
- `--n-splits`: number of folds (default 5)
- `--n-permutations`: number of permutations (default 50000)
- `--seed`: random seed (default 42)
- `--output`: path to output CSV (input CSV with a `fold` column added)

Output: the input CSV with an extra `fold` column (0-indexed fold
assignment for each subject).

---

## BCBlib code conventions to follow

- NumPy-style docstrings
- Flake8, max-complexity 10
- No print statements in library code (use `logging` if needed)
- `tqdm` is acceptable for progress (already a dependency)
- Type hints are encouraged but not required for all internals
- Tests use pytest, no mocking of core logic

---

## Files to create/modify

| File | Action |
|------|--------|
| `bcblib/tools/misc.py` | Add frozen-implementation comment to 2 functions |
| `bcblib/tools/dataset_splitting.py` | Create (new module) |
| `bcblib/tests/test_dataset_splitting.py` | Create (new tests) |
| `bcblib/tests/test_backward_compat.py` | Add misc.py regression tests |
| `bcblib/scripts/run_dataset_splitting.py` | Create (CLI) |
