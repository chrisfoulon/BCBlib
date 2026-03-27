"""Balanced dataset splitting via Monte Carlo permutation search.

Generalises the round-robin splitting strategy from
``bcblib.tools.misc.create_balanced_split`` (which groups subjects by lesion
territory) to any categorical grouping variable.

References
----------
Morgan & Rubin (2012). Rerandomization to improve covariate balance in
experiments. Annals of Statistics.
"""

import logging
from collections import defaultdict

import numpy as np
from scipy.stats import kruskal
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _score_split(folds, covariates):
    """Compute minimax Kruskal-Wallis p-value across covariates.

    For each covariate, an explicit pre-check verifies that at least 2 folds
    contain subjects before calling Kruskal-Wallis. If the check fails for any
    covariate, 0.0 (worst score) is returned immediately so that degenerate
    splits are never selected.

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
        Returns 0.0 if the pre-check fails for any covariate (fewer than
        2 non-empty folds), preventing degenerate splits from being selected.
    """
    min_p = 1.0
    for values in covariates.values():
        values = np.asarray(values)
        groups = [values[fold] for fold in folds if len(fold) > 0]
        if len(groups) < 2:
            return 0.0
        _, p = kruskal(*groups)
        if p < min_p:
            min_p = p
    return min_p


def _descriptive_stats(values):
    """Compute descriptive statistics for a 1-D array.

    Parameters
    ----------
    values : array-like

    Returns
    -------
    dict with keys mean, std, min, max, median (all float)
    """
    arr = np.asarray(values, dtype=float)
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'median': float(np.median(arr)),
    }


def _build_report(
    best_folds,
    best_score,
    groups,
    covariates,
    group_indices,
    convergence,
    n_splits,
    n_permutations,
    seed,
):
    """Build the full report dict for the best split found.

    Parameters
    ----------
    best_folds : list of list of int
    best_score : float
    groups : np.ndarray
    covariates : dict[str, array-like]
    group_indices : dict
    convergence : list of dict
        Each entry: {'permutation': int, 'score': float}
    n_splits, n_permutations, seed : int

    Returns
    -------
    dict
    """
    # Dataset-level stats
    dataset_groups = {
        str(label): {'n': len(idxs)}
        for label, idxs in group_indices.items()
    }
    dataset_covariates = {
        key: _descriptive_stats(vals)
        for key, vals in covariates.items()
    }

    # Per-covariate KW stats for the best split
    kw_stats = {}
    for key, vals in covariates.items():
        arr = np.asarray(vals)
        fold_groups = [arr[fold] for fold in best_folds if len(fold) > 0]
        if len(fold_groups) >= 2:
            H, p = kruskal(*fold_groups)
            kw_stats[key] = {'H': float(H), 'p': float(p)}
        else:
            kw_stats[key] = {'H': None, 'p': None}

    # Per-fold stats
    folds_report = []
    for i, fold in enumerate(best_folds):
        fold_arr = np.asarray(fold)
        fold_group_counts = {
            str(label): int(np.sum(groups[fold_arr] == label))
            for label in group_indices
        }
        fold_covariates = {
            key: _descriptive_stats(np.asarray(vals)[fold_arr])
            for key, vals in covariates.items()
        }
        folds_report.append({
            'fold': i,
            'n_subjects': len(fold),
            'groups': fold_group_counts,
            'covariates': fold_covariates,
        })

    best_found_at = convergence[-1]['permutation'] if convergence else 0

    return {
        'parameters': {
            'n_splits': n_splits,
            'n_permutations': n_permutations,
            'seed': seed,
        },
        'dataset': {
            'n_subjects': int(len(groups)),
            'groups': dataset_groups,
            'covariates': dataset_covariates,
        },
        'search': {
            'best_score': best_score,
            'best_found_at_permutation': best_found_at,
            'convergence': convergence,
        },
        'best_split': {
            'kruskal_wallis': kw_stats,
            'folds': folds_report,
        },
    }


def permutation_balanced_splits(
    groups,
    covariates,
    n_splits=5,
    n_permutations=50000,
    seed=42,
    strict=True,
):
    """Find a balanced k-way split by Monte Carlo permutation search.

    Generalises the round-robin splitting strategy from
    ``bcblib.tools.misc.create_balanced_split`` (which groups subjects by
    lesion territory) to any categorical grouping. The chronic/acute split used
    in lesionsplit is a 2-class instance of this algorithm.

    Balances both group counts (hard constraint via round-robin) and continuous
    covariate distributions (optimised via Kruskal-Wallis minimax p-value).

    Parameters
    ----------
    groups : array-like, shape (n_subjects,)
        Categorical label for each subject (any hashable values). Subjects are
        shuffled within each unique label and assigned to folds round-robin,
        guaranteeing near-equal counts per fold per group. Any preprocessing of
        these labels (e.g. hemisphere disambiguation) is the caller's
        responsibility.
    covariates : dict[str, array-like]
        Continuous variables to balance across folds, keyed by name. Each
        array must have length n_subjects. Kruskal-Wallis is applied to each
        covariate independently. The selection score is min(p_1, p_2, ...)
        — maximised.
    n_splits : int
        Number of folds. Default 5.
    n_permutations : int
        Number of random shuffles to try. Default 50000.
    seed : int
        Random seed for reproducibility. Default 42.
    strict : bool
        If True (default), raise ValueError before running if any group has
        fewer subjects than n_splits. The error message lists every offending
        group with its subject count, e.g.::

            The following groups have fewer subjects than n_splits=5:
              - group '0': 3 subjects
              - group 'ctrl': 2 subjects

        If False, skip the check. Groups too small to fill every fold will
        produce some folds with 0 subjects for that group; those splits score
        0.0 via _score_split and will not be selected as the best split.
        Use strict=False only when you have verified that underfilled folds
        are acceptable for your use case.

    Returns
    -------
    best_folds : list of list of int
        The best split found: a list of n_splits folds, each fold being a list
        of subject indices into the original input arrays.
    best_score : float
        The minimax KW p-value of the best split. Higher = better balanced.
        Not interpretable as a probability; used as a selection criterion only.
    report : dict
        Full report of the search and best split. Keys:

        - ``parameters``: n_splits, n_permutations, seed
        - ``dataset``: n_subjects, per-group counts, per-covariate descriptive
          stats (mean, std, min, max, median) for the whole dataset
        - ``search``: best_score, permutation at which it was found, and
          convergence history (list of improvements only — each entry is
          ``{'permutation': int, 'score': float}``)
        - ``best_split``: per-covariate Kruskal-Wallis results (H, p) and
          per-fold breakdown (n_subjects, group counts, per-covariate
          descriptive stats)

    Raises
    ------
    ValueError
        If strict=True and any group has fewer subjects than n_splits. Message
        includes every offending group name and its subject count.

    Notes
    -----
    Score is min(KW_p_covariate_1, ..., KW_p_covariate_N) — maximising this
    minimises the worst-case detectable imbalance across all covariates. KW is
    appropriate for skewed distributions (e.g. lesion volumes).

    The round-robin assignment guarantees that each fold receives
    floor(n_group / n_splits) or ceil(n_group / n_splits) subjects from each
    group — count imbalance of at most 1 per group per fold.

    References
    ----------
    Morgan & Rubin (2012). Rerandomization to improve covariate balance in
    experiments. Annals of Statistics.
    """
    groups = np.asarray(groups)
    n_subjects = len(groups)

    # Validate covariate lengths
    for key, vals in covariates.items():
        if len(vals) != n_subjects:
            raise ValueError(
                f"Covariate '{key}' has {len(vals)} values but groups has "
                f"{n_subjects} subjects."
            )

    # Group indices by label
    group_indices = defaultdict(list)
    for idx, label in enumerate(groups):
        group_indices[label].append(idx)

    if strict:
        offenders = [
            (label, len(idxs))
            for label, idxs in group_indices.items()
            if len(idxs) < n_splits
        ]
        if offenders:
            lines = "\n".join(
                f"  - group {str(label)!r}: {count} subjects"
                for label, count in offenders
            )
            raise ValueError(
                f"The following groups have fewer subjects than "
                f"n_splits={n_splits}:\n{lines}"
            )

    rng = np.random.default_rng(seed)
    best_folds = None
    best_score = -1.0
    convergence = []

    with tqdm(range(n_permutations), desc="Permutation search") as pbar:
        for perm in pbar:
            folds = [[] for _ in range(n_splits)]
            for idxs in group_indices.values():
                shuffled = list(idxs)
                rng.shuffle(shuffled)
                for i, idx in enumerate(shuffled):
                    folds[i % n_splits].append(idx)

            score = _score_split(folds, covariates)
            if score > best_score:
                best_score = score
                best_folds = [list(fold) for fold in folds]
                convergence.append({'permutation': perm, 'score': best_score})
                pbar.set_postfix(best=f'{best_score:.4f}',
                                 improved=len(convergence))

    report = _build_report(
        best_folds, best_score, groups, covariates,
        group_indices, convergence, n_splits, n_permutations, seed,
    )
    logger.info(
        "Best split: score=%.4f found at permutation %d/%d (%d improvements)",
        best_score,
        convergence[-1]['permutation'] if convergence else 0,
        n_permutations,
        len(convergence),
    )
    return best_folds, best_score, report
