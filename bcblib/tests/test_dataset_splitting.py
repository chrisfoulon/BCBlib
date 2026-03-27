"""Tests for bcblib.tools.dataset_splitting."""

import numpy as np
import pytest

from bcblib.tools.dataset_splitting import (
    _score_split,
    permutation_balanced_splits,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_subjects(n_per_group, rng=None):
    """Return groups array and covariates dict for a balanced 2-group dataset.

    Parameters
    ----------
    n_per_group : int
        Number of subjects in each of the two groups (0 and 1).
    rng : numpy.random.Generator, optional
        RNG for covariate values. Defaults to a fixed seed.

    Returns
    -------
    groups : np.ndarray, shape (2 * n_per_group,)
    covariates : dict[str, np.ndarray]
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n = 2 * n_per_group
    groups = np.array([0] * n_per_group + [1] * n_per_group)
    covariates = {
        'vol_a': rng.exponential(scale=100, size=n),
        'vol_b': rng.exponential(scale=50, size=n),
    }
    return groups, covariates


# ---------------------------------------------------------------------------
# TestOutputStructure
# ---------------------------------------------------------------------------

class TestOutputStructure:
    def test_output_structure(self):
        groups, covariates = _make_subjects(15)
        folds, score = permutation_balanced_splits(
            groups, covariates, n_splits=5, n_permutations=20, seed=0
        )
        assert len(folds) == 5
        all_indices = [idx for fold in folds for idx in fold]
        assert sorted(all_indices) == list(range(30))
        # no duplicates across folds
        assert len(all_indices) == len(set(all_indices))


# ---------------------------------------------------------------------------
# TestCountBalance
# ---------------------------------------------------------------------------

class TestCountBalance:
    def test_count_balance(self):
        n_per_group = 13  # intentionally not divisible by n_splits=5
        groups, covariates = _make_subjects(n_per_group)
        folds, _ = permutation_balanced_splits(
            groups, covariates, n_splits=5, n_permutations=10, seed=0
        )
        for g in [0, 1]:
            counts = [sum(1 for idx in fold if groups[idx] == g) for fold in folds]
            assert max(counts) - min(counts) <= 1, (
                f"Group {g} fold counts differ by more than 1: {counts}"
            )


# ---------------------------------------------------------------------------
# TestCovariateBalance
# ---------------------------------------------------------------------------

class TestCovariateBalance:
    def test_covariate_balance(self):
        """Optimised split should score >= a single random (unseeded) split."""
        rng = np.random.default_rng(99)
        groups, covariates = _make_subjects(25, rng=rng)

        _, best_score = permutation_balanced_splits(
            groups, covariates, n_splits=5, n_permutations=500, seed=42
        )

        # Build one naive random split for comparison
        indices = list(range(50))
        rng2 = np.random.default_rng(1)
        rng2.shuffle(indices)
        chunk = len(indices) // 5
        naive_folds = [indices[i * chunk:(i + 1) * chunk] for i in range(5)]
        naive_score = _score_split(naive_folds, covariates)

        assert best_score >= naive_score


# ---------------------------------------------------------------------------
# TestReproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_reproducibility(self):
        groups, covariates = _make_subjects(15)
        folds_a, score_a = permutation_balanced_splits(
            groups, covariates, n_splits=5, n_permutations=50, seed=7
        )
        folds_b, score_b = permutation_balanced_splits(
            groups, covariates, n_splits=5, n_permutations=50, seed=7
        )
        assert folds_a == folds_b
        assert score_a == score_b

    def test_different_seeds(self):
        groups, covariates = _make_subjects(20)
        folds_a, _ = permutation_balanced_splits(
            groups, covariates, n_splits=5, n_permutations=50, seed=1
        )
        folds_b, _ = permutation_balanced_splits(
            groups, covariates, n_splits=5, n_permutations=50, seed=2
        )
        assert folds_a != folds_b


# ---------------------------------------------------------------------------
# TestCovariates
# ---------------------------------------------------------------------------

class TestCovariates:
    def test_single_covariate(self):
        groups, covariates = _make_subjects(15)
        single = {'vol_a': covariates['vol_a']}
        folds, score = permutation_balanced_splits(
            groups, single, n_splits=5, n_permutations=20, seed=0
        )
        assert len(folds) == 5
        assert 0.0 <= score <= 1.0

    def test_multiple_covariates(self):
        groups, covariates = _make_subjects(15)
        folds, score = permutation_balanced_splits(
            groups, covariates, n_splits=5, n_permutations=20, seed=0
        )
        assert len(folds) == 5
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# TestScoring
# ---------------------------------------------------------------------------

class TestScoring:
    def test_degenerate_score(self):
        """_score_split returns 0.0 when fewer than 2 folds have data."""
        # Only fold 0 has any subjects — cannot run KW
        folds = [[0, 1, 2, 3, 4], [], [], [], []]
        covariates = {'vol': np.array([10.0, 20.0, 30.0, 40.0, 50.0])}
        assert _score_split(folds, covariates) == 0.0

    def test_score_range(self):
        groups, covariates = _make_subjects(15)
        folds, score = permutation_balanced_splits(
            groups, covariates, n_splits=5, n_permutations=20, seed=0
        )
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# TestValidation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_too_few_subjects_raises(self):
        """strict=True (default): ValueError with per-group counts in message."""
        groups = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])  # group 0 has 3 < 5
        covariates = {'vol': np.arange(10, dtype=float)}
        with pytest.raises(ValueError, match="0") as exc_info:
            permutation_balanced_splits(
                groups, covariates, n_splits=5, n_permutations=10, seed=0
            )
        # Message must mention the count
        assert "3" in str(exc_info.value)

    def test_strict_false_skips_check(self):
        """strict=False: no error raised even when a group is undersized."""
        groups = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])  # group 0 has 3 < 5
        covariates = {'vol': np.arange(10, dtype=float)}
        # Should not raise
        folds, score = permutation_balanced_splits(
            groups, covariates, n_splits=5, n_permutations=10, seed=0,
            strict=False
        )
        assert len(folds) == 5
