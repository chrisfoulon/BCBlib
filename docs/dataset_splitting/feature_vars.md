# Feature Variables — Balanced Dataset Splitting

```
FEATURE_SLUG=dataset-splitting
PROJECT_NAME=BCBlib
FEATURE_DESCRIPTION="New general-purpose module bcblib/tools/dataset_splitting.py implementing a Monte Carlo permutation search for balanced k-way dataset splits. Generalises the round-robin strategy from bcblib.tools.misc.create_balanced_split (lesion territories) to any categorical grouping. Fixes conjunctive selection criterion, adds multi-covariate support, reproducibility seed, and a CLI entry point."

TASK_COMPLEXITY=MEDIUM
IMPLEMENTATION_APPROACH=TDD — write failing tests first, implement until green, flake8 clean
KEY_CHALLENGES="Correct degenerate-case handling in _score_split; ensuring round-robin count guarantee is testable; CLI entry point name conflict (bcb-split already taken → use bcb-dataset-split)"

BASELINE_TESTS=167
BASELINE_ERRORS=1  # pre-existing in test_parcitron.py, unrelated
PYTHON_VERSION=3.12.12
GIT_BRANCH=dev
NEW_DEPENDENCIES=none  # numpy, scipy, tqdm already in install_requires
```
