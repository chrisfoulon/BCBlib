# Lesion Features Pipeline

Extract atlas-based overlap profiles from patient lesion masks and disconnectome maps.
The pipeline is split into two CLI commands that can be run independently.

---

## Overview

```
Stage 1  bcb-lf-preprocess
         BIDS lesion masks → normalised 1 mm MNI6 masks + disconnectome maps

Stage 2  bcb-lesion-features
         Normalised masks + disconnectomes → per-atlas overlap CSVs
```

---

## Prerequisites

- **BCBToolKit** installed and accessible (provides `run_disco.sh`)
- **Tractography atlas** (1 mm tracks), typically bundled with BCBToolKit at
  `<BCBToolKit>/Tools/extraFiles/tracks_1mm`
- **BCBlib** installed: `pip install bcblib`

---

## Input data format (BIDS)

The pipeline reads a standard BIDS derivatives directory.
Each lesion mask must follow this naming convention:

```
<bids_dir>/
  sub-001/
    anat/
      sub-001_space-MNI152NLin6Asym_res-1_label-lesion_mask.nii.gz
  sub-002/
    anat/
      sub-002_space-MNI152NLin6Asym_res-2_label-lesion_mask.nii.gz
  sub-003/
    anat/
      sub-003_space-MNI152NLin2009cAsym_res-1_label-lesion_mask.nii.gz
```

Required BIDS filename entities:

| Entity | Values | Effect |
|--------|--------|--------|
| `space-` | `MNI152NLin6Asym` (or alias), `MNI152NLin2009cAsym` | Controls template-space normalisation |
| `res-` | `1`, `2` | Controls resolution resampling (optional — detected from shape if absent) |
| `label-lesion` | fixed | Identifies the file as a lesion mask |
| suffix `_mask` | fixed | Required |

Sessions are supported: `sub-001/ses-01/anat/sub-001_ses-01_space-…_mask.nii.gz`.

### Supported input spaces

| `space-` entity | Handling |
|-----------------|---------|
| `MNI152NLin6Asym` at 1 mm | Passthrough — used as-is |
| `MNI152NLin6Asym` at 2 mm | Resampled to 1 mm preserving input orientation |
| `MNI152NLin2009cAsym` | Warped to MNI6 via TemplateFlow, then orientation corrected |

---

## Stage 1 — Normalise lesions and compute disconnectomes

```bash
bcb-lf-preprocess \
  --bids-dir /path/to/bids \
  --output-dir /path/to/prep \
  --bcbtoolkit /path/to/BCBToolKit \
  --tracks-dir /path/to/BCBToolKit/Tools/extraFiles/tracks_1mm \
  --ncores 8
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--bids-dir` | required | Input BIDS directory |
| `--output-dir` | `./lesion_features_prep` | Output derivatives directory |
| `--bcbtoolkit` | `$BCBTOOLKIT_PATH` or compiled-in default | BCBToolKit root directory |
| `--tracks-dir` | BCBToolKit default | Tractography atlas directory (`-T` flag for `run_disco.sh`) |
| `--ncores` | BCBToolKit default | Parallel cores for disconnectome computation |
| `--tmpdir` | `$TMPDIR` or `/tmp` | Scratch directory for `run_disco.sh` intermediate files |
| `--skip-existing` | off | Skip subjects whose normalised lesion already exists |
| `--dry-run` | off | Print subjects that would be processed, then exit |

> **Note for JupyterHub / HPC environments**: if `/tmp` is not writable or is too
> small, set `--tmpdir` to a directory on your home or scratch filesystem:
> ```bash
> bcb-lf-preprocess ... --tmpdir /home/user/scratch/disco_tmp
> ```

BCBToolKit is found in this order:
1. `--bcbtoolkit` flag
2. `$BCBTOOLKIT_PATH` environment variable
3. Compiled-in default path

### Outputs

```
<output-dir>/
  sub-001/
    anat/
      sub-001_space-MNI152NLin6Asym_res-1_label-lesion_mask.nii.gz   ← normalised lesion
      sub-001_space-MNI152NLin6Asym_res-1_desc-disconnectome.nii.gz  ← disconnectome
  sub-002/
    anat/
      sub-002_space-MNI152NLin6Asym_res-1_label-lesion_mask.nii.gz
      sub-002_space-MNI152NLin6Asym_res-1_desc-disconnectome.nii.gz
```

All outputs are in **MNI152NLin6Asym 1 mm**, preserving the input's voxel orientation
convention (radiological or neurological).

---

## Stage 2 — Extract atlas overlap features

```bash
bcb-lesion-features \
  --prep-dir /path/to/prep \
  --output-dir /path/to/features \
  --ebrains \
  --assume-yes
```

### Options

| Flag | Description |
|------|-------------|
| `--prep-dir` | Stage 1 output directory (required) |
| `--output-dir` | Where to write CSVs and TSVs (default: `./lesion_features`) |
| `--ebrains` | Use the full EBRAINS atlas set (AAL, Schaefer 200/400, Schaefer+Tian S1 200/400) |
| `--assume-yes` | Skip download consent prompts (non-interactive / scripted use) |
| `--skip-existing` | Skip subjects whose output already exists |
| `--min-overlap-voxels` | Minimum overlap voxels to include a region (default: 1) |
| `--preset NAME` | Add a named preset atlas (repeatable) |
| `--name / --atlas / --threshold / --label-file` | Add a custom atlas (repeatable group) |

### EBRAINS atlas set

When `--ebrains` is passed, five atlases are used automatically:

| Key | Atlas | Regions |
|-----|-------|---------|
| `aal` | AAL (neuroparc) | 116 |
| `schaefer_200_7n` | Schaefer 2018, 200 parcels, 7 networks | 200 |
| `schaefer_400_7n` | Schaefer 2018, 400 parcels, 7 networks | 400 |
| `schaefer_200_tian_s1` | Schaefer 200 + Tian Subcortex S1 | 216 |
| `schaefer_400_tian_s1` | Schaefer 400 + Tian Subcortex S1 | 416 |

Atlases are downloaded on first use to `~/.bcblib/atlases/`.
To use a shared atlas cache (e.g. on a multi-user server), set the environment variable
`BCBLIB_ATLAS_DIR` to a shared path before running *(see note below)*.

> **Atlas cache location**: currently hard-coded to `~/.bcblib/atlases/`.
> If you need a system-wide shared cache, download the atlases once as the admin
> user and set `BCBLIB_ATLAS_DIR=/shared/path` in the environment before running
> the pipeline.  This requires a one-line change to `get_atlas_dir()` in
> `_atlas_manager.py` — contact the BCBlib maintainers or open an issue.

### Outputs

```
<output-dir>/
  sub-001/
    anat/
      sub-001_space-MNI152NLin6Asym_LF-lesion_atlas-aal.csv
      sub-001_space-MNI152NLin6Asym_LF-lesion_atlas-schaefer_200_7n.csv
      sub-001_space-MNI152NLin6Asym_LF-lesion_atlas-schaefer_400_7n.csv
      sub-001_space-MNI152NLin6Asym_LF-lesion_atlas-schaefer_200_tian_s1.csv
      sub-001_space-MNI152NLin6Asym_LF-lesion_atlas-schaefer_400_tian_s1.csv
      sub-001_space-MNI152NLin6Asym_LF-disconnectome_atlas-aal.csv
      …
      sub-001_space-MNI152NLin6Asym_desc-lesion_mapstats.tsv
      sub-001_space-MNI152NLin6Asym_desc-disconnectome_mapstats.tsv
```

**Per-atlas CSV** (`LF-lesion_atlas-*.csv` and `LF-disconnectome_atlas-*.csv`):

| Column | Description |
|--------|-------------|
| `region_name` | Region label from atlas |
| `n_voxels_region` | Total voxels in the atlas region |
| `n_voxels_overlap` | Voxels with subject map > 0 |
| `fraction_covered` | `n_voxels_overlap / n_voxels_region` |
| `mean_overlap` | Mean map value across all region voxels |
| `weighted_mean_overlap` | Atlas-weight-weighted mean |
| `sum_overlap` | Sum of map values within region |
| `p90_overlap` | 90th percentile over non-zero voxels |
| `p95_overlap` | 95th percentile over non-zero voxels |

Rows are sorted by `mean_overlap` descending and filtered to `n_voxels_overlap >= min_overlap_voxels`.

**Map stats TSV** (`desc-lesion_mapstats.tsv`, `desc-disconnectome_mapstats.tsv`):
One-row summary per subject map — voxel count, volume (mm³), sum, mean, percentiles.
Useful for normalising overlap metrics at analysis time.

---

## Full example

```bash
# Stage 1: normalise and compute disconnectomes
bcb-lf-preprocess \
  --bids-dir ~/data/stroke_cohort \
  --output-dir ~/data/stroke_cohort_prep \
  --bcbtoolkit ~/BCBToolKit \
  --tracks-dir ~/BCBToolKit/Tools/extraFiles/tracks_1mm \
  --ncores 16 \
  --tmpdir ~/scratch/disco_tmp

# Stage 2: extract features
bcb-lesion-features \
  --prep-dir ~/data/stroke_cohort_prep \
  --output-dir ~/data/stroke_cohort_features \
  --ebrains \
  --assume-yes
```

---

## Troubleshooting

**`/tmp` not writable or too small**
Pass `--tmpdir /path/to/scratch` to `bcb-lf-preprocess`. This maps to the `-w` flag
of `run_disco.sh`. The directory is created if it does not exist and cleaned up after the run.

**TemplateFlow download fails**
The MNI2009c→MNI6 warp requires TemplateFlow data. On systems without internet access,
pre-populate the cache by running on a connected machine first:
```python
import templateflow.api as tflow
tflow.get("MNI152NLin6Asym", suffix="xfm", extension=".h5", **{"from": "MNI152NLin2009cAsym"})
```
The cache is stored in `$TEMPLATEFLOW_HOME` (default: `~/.cache/templateflow`).

**Atlas download prompts block non-interactive runs**
Pass `--assume-yes` to `bcb-lesion-features`, or set `BCBLIB_YES=1` in the environment.

**BCBToolKit not found**
Set the `BCBTOOLKIT_PATH` environment variable to the BCBToolKit root directory, or pass
`--bcbtoolkit /path/to/BCBToolKit` explicitly.
