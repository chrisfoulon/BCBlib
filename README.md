# BCBlib

[![PyPI version](https://badge.fury.io/py/bcblib.svg)](https://badge.fury.io/py/bcblib)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

A collection of neuroimaging utilities for MRI data analysis, developed by
[Chris Foulon](https://github.com/chrisfoulon) at the [BCBlab](http://bcblab.com).
The library covers NIfTI image processing, FSL workflow integration, and
research-grade statistical tools — several of which have been used in published
studies on structural connectivity, lesion-symptom mapping, and stroke outcome
prediction.

```bash
pip install bcblib
```

## Quick Start

```python
# Load and inspect a NIfTI image
from bcblib.imaging import load_nifti, image_stats
nii = load_nifti("subject01_lesion.nii.gz")
stats = image_stats(nii)

# Balanced dataset splitting for cross-validation
from bcblib.tools.dataset_splitting import permutation_balanced_splits
folds, score, report = permutation_balanced_splits(
    groups=has_chronic,
    covariates={"acute_vol": acute_volumes, "chronic_vol": chronic_volumes},
    n_splits=5,
    n_permutations=50000,
    seed=42,
)

# Prepare FSL randomise inputs from a spreadsheet
from bcblib.tools.randomise_helper import spreadsheet_to_mat_and_file_list
spreadsheet_to_mat_and_file_list(
    "subjects.csv", columns=["age", "score"],
    output_dir="randomise_inputs/", filenames_column="image_path"
)
```

## Modules

### Imaging (`bcblib.imaging`)

A modern FSL-equivalent API for NIfTI image processing. All functions accept
either a file path or an already-loaded `Nifti1Image`.

| Module | Description |
|--------|-------------|
| `bcblib.imaging.io` | Load, save, resave NIfTI images; format detection |
| `bcblib.imaging.info` | Header inspection — equivalent to `fslinfo` |
| `bcblib.imaging.stats` | Centre of gravity, volume, histogram, laterality |
| `bcblib.imaging.math` | Binarize, dilate, erode, apply/invert masks |
| `bcblib.imaging.orient` | Get and reorient image orientation |
| `bcblib.imaging.manipulate` | Extract ROI, merge, split 4-D images |
| `bcblib.imaging.convert` | Convert between `.nii` and `.nii.gz` |

> **Backward compatibility:** `bcblib.tools.nifti_utils`, `bcblib.tools.images_utils`,
> and `bcblib.tools.nii_stats` are kept as shims. New code should import from
> `bcblib.imaging` directly.

### Research Tools (`bcblib.tools`)

**`best_overlap`**
Compute preserved structural connectivity from patient and cluster disconnectome
maps using Bayesian modelling (PyMC). Outputs latent connectivity scores with
uncertainty estimates per brain region.

**`dataset_splitting`**
Monte Carlo permutation search for the most balanced k-way dataset split.
Balances group counts (round-robin hard constraint) and continuous covariate
distributions (Kruskal-Wallis minimax score). Returns fold indices, best score,
and a full JSON report including convergence history and per-fold descriptive
stats. See [Publications](#publications).

**`randomise_helper`**
Build FSL `randomise` inputs from a spreadsheet: generates `.mat` design files,
concatenates 4-D NIfTI stacks, and manages file lists.
See [Publications](#publications).

**`split_clusters`** — Split a multi-label NIfTI atlas into one file per label value.

**`divide_mask`** — Cluster a binary mask into spatially separate components by proximity.

**`shapes`** — Generate geometric shapes (hyperspheres, arbitrary forms) as NIfTI
arrays, useful for creating synthetic phantom data.

**`constants`** — Pre-computed MNI 1 mm and 2 mm affines and shapes;
`empty_MNI1MM()` / `empty_MNI2MM()` convenience constructors.

**`general_utils`** — JSON I/O with support for NumPy arrays, UUIDs, and datetime objects.

**`spreadsheet_io_utils`** — Load CSV and Excel files with column selection helpers.

**`dataframe_filtering`** — Remove constant columns, filter by completeness
threshold, handle datetime columns for ML pipelines.

**`mat_transform`** — Connectivity matrix preprocessing: log₂ transform,
z-score normalisation, rank transform.

**`arrays_utils`** — Coordinate validation and centroid calculations for NumPy arrays.

**`umap_utils`** — UMAP dimensionality reduction wrappers tuned for neuroimaging data.

**`visualisation`** — Wrappers for MRIcron, matplotlib, and TensorBoard visualisation.
> Requires external tools (MRIcron, TensorBoard) installed separately depending
> on which functions are used.

## CLI Tools

| Command | Example | Description |
|---------|---------|-------------|
| `bcb-info` | `bcb-info brain.nii.gz` | NIfTI header summary (`fslinfo` equivalent) |
| `bcb-header` | `bcb-header brain.nii.gz` | Full NIfTI header inspection |
| `bcb-stats` | `bcb-stats lesion.nii.gz` | Image statistics (`fslstats` equivalent) |
| `bcb-orient` | `bcb-orient -g brain.nii.gz` | Get or set image orientation |
| `bcb-roi` | `bcb-roi brain.nii.gz mask.nii.gz` | Extract a region of interest |
| `bcb-merge` | `bcb-merge -o 4d.nii.gz *.nii.gz` | Merge NIfTI images along a dimension |
| `bcb-split` | `bcb-split 4d.nii.gz -o out/` | Split a 4-D NIfTI along the volume axis |
| `bcb-convert` | `bcb-convert brain.nii` | Convert between `.nii` and `.nii.gz` |
| `bcb-dataset-split` | see below | Balanced dataset splitting from a CSV file |
| `randomise_helper` | — | Build FSL randomise design files from a spreadsheet |
| `pick_up_matched_synth_lesions` | — | Select synthetic lesions matching a size distribution |

```bash
# Balanced dataset split — writes splits.csv and splits_report.json
bcb-dataset-split \
    --input subjects.csv \
    --group-col has_chronic \
    --covariate-cols acute_volume chronic_volume \
    --n-splits 5 --n-permutations 50000 --seed 42 \
    --output splits.csv
```

## Dependencies

Core dependencies installed automatically:

- `nibabel`, `numpy`, `scipy`, `nilearn`, `scikit-learn`
- `pandas`, `openpyxl`
- `tqdm`, `joblib`, `statsmodels`
- `matplotlib`, `rich`
- `pymc >= 5`, `arviz` (required for `best_overlap`)
- `umap-learn` (required for `umap_utils`)
- `mne`

External tools (not installed by pip):

- **FSL** — required for `randomise_helper` to call `randomise` itself
- **MRIcron** — required for `visualisation` MRIcron wrappers
- **TensorBoard** — required for `visualisation` TensorBoard integration

## Publications

Tools from BCBlib have been used in the following publications:

- **`randomise_helper`** — Giampiccolo D, Binding LP, Caciagli L, Rodionov R,
  Foulon C, et al. (2023). Thalamostriatal disconnection underpins long-term
  seizure freedom in frontal lobe epilepsy surgery. *Brain*, 146(6):2377–2388.
  https://doi.org/10.1093/brain/awad085

- **`pick_up_matched_synth_lesions`** — Thiebaut de Schotten M, Foulon C,
  Nachev P. (2020). Brain disconnections link structural connectivity with
  function and behaviour. *Nature Communications*, 11:5094.
  https://doi.org/10.1038/s41467-020-18920-9

- **`dataset_splitting`** — Foulon C, Gray R, Ruffle JK, et al. (2025).
  Generalizable automated ischaemic stroke lesion segmentation with vision
  transformers. arXiv:2502.06939.
  https://arxiv.org/abs/2502.06939

- **BCBlib** was used to participate in the Neural-CUP benchmark —
  Matsulevits A, Alvez P, Atzori M, et al. (2024). A global effort to
  benchmark predictive models and reveal mechanistic diversity in long-term
  stroke outcomes. bioRxiv.
  https://doi.org/10.1101/2024.10.17.618691

- **BCBlib** is used in — Foulon C, Ovando-Tellez M, Talozzi L, Corbetta M,
  Matsulevits A, Thiebaut de Schotten M. (2024). Emerging-properties Mapping
  Using Spatial Embedding Statistics: EMUSES. arXiv:2406.14309 *(preprint)*.
  https://arxiv.org/abs/2406.14309

## Links

- [Source](https://github.com/chrisfoulon/BCBlib)
- [Bug reports](https://github.com/chrisfoulon/BCBlib/issues)
- [BCBlab](http://bcblab.com)
- [PyPI](https://pypi.org/project/bcblib/)
