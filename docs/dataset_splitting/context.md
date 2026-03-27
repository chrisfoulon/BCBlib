# Context — Balanced Dataset Splitting

## Level 1: Plain English

BCBlib already has a Monte Carlo balanced-split function in `misc.py`, written for a
specific paper. It groups subjects by `lesion_cluster`, shuffles within each group, and
assigns them round-robin to folds. The scoring uses a conjunctive criterion (all three
metrics must improve at once) which causes most valid improvements to be rejected.

The new module generalises this to any categorical label array, fixes the scoring to use
a single minimax KW p-value, adds a seed parameter for reproducibility, and returns fold
indices (integers) so the caller can map them back to any data structure they prefer.

The two modules are independent. No deprecation. `misc.py` functions are frozen as the
published implementation.

---

## Level 2: Key API Table

### Existing code (`bcblib/tools/misc.py`)

| Symbol | Purpose | Inputs | Outputs | Notes |
|--------|---------|--------|---------|-------|
| `create_balanced_split(info_dict_keys, info_dict, num_splits=5)` | One round-robin split | list of keys, dict keyed by those keys, int | `(split_dict, mean_splits, std_splits, st, pval)` | Groups by `info_dict[k]['lesion_cluster']`; bilateral offset hardcoded |
| `permutation_balanced_splits(info_dict_keys, info_dict, num_permutations)` | Monte Carlo best split | same + int permutations | `list[dict]` — 5 dicts | Conjunctive criterion bug; bare print; `best_st` unused; no seed |

### New code (`bcblib/tools/dataset_splitting.py`)

| Symbol | Purpose | Inputs | Outputs | Notes |
|--------|---------|--------|---------|-------|
| `permutation_balanced_splits(groups, covariates, n_splits, n_permutations, seed)` | Monte Carlo best split (generalised) | array-like, dict[str→array-like], int, int, int | `(list[list[int]], float)` | Returns fold indices; raises ValueError if any group < n_splits subjects |
| `_score_split(folds, covariates)` | Minimax KW p-value | list[list[int]], dict[str→array-like] | float | Returns 0.0 for degenerate cases (can't run KW) |

### CLI (`bcblib/scripts/run_dataset_splitting.py`)

| Entry point | Input | Output |
|-------------|-------|--------|
| `bcb-dataset-split` | CSV with group col + covariate cols | Same CSV + `fold` column |

### Test files to modify

| File | Change |
|------|--------|
| `bcblib/tests/test_backward_compat.py` | Add `TestMiscSplitRegression` class |

---

## Level 3: Integration Points

### Round-robin pattern from misc.py (to generalise)

```python
# misc.py create_balanced_split — the core pattern we generalise:
cluster_dict = defaultdict(list)
for k in info_dict_keys:
    clu_name = info_dict[k]['lesion_cluster']
    cluster_dict[clu_name].append(k)

split_dict = defaultdict(dict)
for clu_name, clu_keys in cluster_dict.items():
    for i, k in enumerate(clu_keys):
        split_dict[i % num_splits][k] = info_dict[k]
# → in new module: groups array replaces cluster_dict lookup; indices replace subject dicts
```

### KW scoring pattern from misc.py (to replace)

```python
# misc.py — conjunctive criterion (broken):
if best_mean_range > means_range and best_std_range > stds_range and best_pvalue < pval:

# new module — single minimax KW p-value (correct):
score = min(kruskal(*[covariate[fold] for fold in folds]).pvalue
            for covariate in covariates.values())
# if degenerate (can't run kruskal): score = 0.0
```

### argparse script pattern (from run_best_overlap.py)

```python
def parse_args():
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument("--input", required=True, ...)
    ...
    return parser.parse_args()

def main():
    args = parse_args()
    ...

if __name__ == "__main__":
    main()
```

### Entry point registration (setup.py)

```python
# NOTE: 'bcb-split' is already taken by bcblib.scripts.imaging_cli:bcb_split
# New entry point:
'bcb-dataset-split = bcblib.scripts.run_dataset_splitting:main'
```

---

## Maintenance Opportunities in Target Files

### misc.py (touched only to add comment — no logic changes)

- `misc.py:117` — bare `print(...)` in library code (not fixing, frozen file)
- `misc.py:100` — `best_st` tracked but never returned (not fixing, frozen file)
