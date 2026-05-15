# BCBlib + BCBToolKit — Server Setup Instructions

## 1. Update BCBToolKit

```bash
cd /path/to/BCBToolKit
git pull
```

No other changes needed for BCBToolKit.

---

## 2. Update BCBlib

Install or upgrade directly from the GitHub devel branch.
If BCBlib is installed in a shared conda/virtual environment, activate it first.

```bash
pip install --upgrade "git+https://github.com/chrisfoulon/BCBlib.git@devel"
```

---

## 3. Set up shared atlas cache

This avoids every user downloading their own copy of the atlases (~50 MB total).

Create a shared directory and download the five atlases into it once:

```bash
mkdir -p /shared/bcblib_atlases

BCBLIB_ATLAS_DIR=/shared/bcblib_atlases \
  bcb-lesion-features \
    --prep-dir /tmp \
    --output-dir /tmp/throwaway_lf \
    --ebrains \
    --assume-yes \
    --skip-existing

chmod -R a+rX /shared/bcblib_atlases
```

The command will print warnings about finding no subjects in /tmp — that is expected.
The atlases will be downloaded and cached in /shared/bcblib_atlases.

Then make the path permanent for all users by adding this line
to `/etc/environment` (or wherever the server sets system-wide environment variables):

```
BCBLIB_ATLAS_DIR=/shared/bcblib_atlases
```

---

## 4. Pre-download TemplateFlow warps

The pipeline uses TemplateFlow to normalise lesions that are not already in MNI152NLin6Asym.
Pre-downloading the warp server-wide avoids each user downloading ~200 MB on their first run.
MNI152NLin2009cAsym is the only adult template TemplateFlow can warp to MNI6.

```bash
mkdir -p /shared/templateflow

TEMPLATEFLOW_HOME=/shared/templateflow python -c "
import templateflow.api as tflow
tflow.get('MNI152NLin6Asym', suffix='xfm', extension='.h5',
          **{'from': 'MNI152NLin2009cAsym'})
tflow.get('MNI152NLin6Asym', resolution=1, desc='brain', suffix='T1w')
"

chmod -R a+rX /shared/templateflow
```

Then add to `/etc/environment`:

```
TEMPLATEFLOW_HOME=/shared/templateflow
```

---

## How users run the pipeline

The pipeline runs in two stages from the terminal.

**Input:** lesion masks in BIDS format:

```
bids_dir/
  sub-001/anat/sub-001_space-MNI152NLin6Asym_res-1_label-lesion_mask.nii.gz
  sub-002/anat/sub-002_space-MNI152NLin6Asym_res-2_label-lesion_mask.nii.gz
  sub-003/anat/sub-003_space-MNI152NLin2009cAsym_res-1_label-lesion_mask.nii.gz
```

The `space-` and `res-` entities are used automatically to normalise each lesion to MNI6 1mm.

### Stage 1 — Normalise lesions and compute disconnectomes

```bash
bcb-lf-preprocess \
  --bids-dir ~/data/bids \
  --output-dir ~/results/prep \
  --bcbtoolkit /path/to/BCBToolKit \
  --tracks-dir /path/to/BCBToolKit/Tools/extraFiles/tracks_1mm \
  --ncores 8 \
  --tmpdir ~/scratch/disco_tmp
```

`--tmpdir` points `run_disco.sh` to a writable scratch directory (any directory on the
user's home or scratch filesystem). It is created automatically and cleaned up after the run.

### Stage 2 — Extract atlas overlap features

```bash
bcb-lesion-features \
  --prep-dir ~/results/prep \
  --output-dir ~/results/features \
  --ebrains \
  --assume-yes
```

**Output:** for each subject, one CSV per atlas per map type (lesion + disconnectome)
and one TSV with descriptive stats for each map.
