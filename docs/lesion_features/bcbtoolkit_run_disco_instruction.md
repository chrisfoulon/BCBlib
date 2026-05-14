# BCBToolKit — run_disco.sh: BIDS naming modification

## Context

This instruction is for a separate Claude session working on the BCBToolKit package.
It must be completed before the `bcb-lf-preprocess` pipeline in BCBlib can produce
correct BIDS-compliant output filenames.

**File to modify**: `/home/chrisfoulon/neuro_apps/BCBToolKitLINUX/BCBToolKit/run_disco.sh`

---

## Required changes

### Change 1 — Folder/CSV mode: add `-r FROM TO` rename flag

**What it does**: When `-r FROM TO` is given, each output stem has the first
occurrence of `FROM` replaced with `TO` (bash `${stem/FROM/TO}` substitution).

**Where to add it**: in the argument-parsing block (look for the `getopts` or
explicit `while` loop that parses `-l`, `-o`, `-t`, `-n`, `-T`, `-p`, `-w`, `-d`).
Add a new option `-r`:

```bash
-r)  RENAME_FROM="$OPTARG"
     # read the replacement as the next positional argument
     # (see Note below for two-arg pattern)
     ;;
```

Because `getopts` only accepts a single argument per flag, the cleanest approach
for a two-value flag is to split on a delimiter.  Use `-r FROM:TO` (colon-separated):

```bash
-r)  RENAME_FROM="${OPTARG%%:*}"
     RENAME_TO="${OPTARG##*:}"
     ;;
```

Initialise at the top of the variable block:
```bash
RENAME_FROM=""
RENAME_TO=""
```

Apply the substitution in **both** `discover_folder` and `discover_csv` functions,
immediately after the `stem` is computed (lines ~248, ~310):

```bash
# Apply optional rename substitution
if [[ -n "$RENAME_FROM" ]]; then
    stem="${stem//$RENAME_FROM/$RENAME_TO}"
fi
```

**Do NOT apply to BIDS mode** (`discover_bids`) — that has its own naming logic
(Change 2 below).

**Usage example** (how the BCBlib pipeline will call it):
```bash
run_disco.sh -l /tmp/prep_lesions -o /tmp/disco_out \
    -r "_label-lesion_mask:_desc-disconnectome"
```

Input `sub-001_space-MNI152NLin6Asym_res-1_label-lesion_mask.nii.gz`
→ output `sub-001_space-MNI152NLin6Asym_res-1_desc-disconnectome.nii.gz`

---

### Change 2 — BIDS mode: rename output from `_les_SDC` to `_desc-disconnectome`

**Current line** (≈ line 358 in `discover_bids`):
```bash
out_stem="$participant_dir/features/lesion/${participant_id}_les_SDC"
```

**Replace with**:
```bash
out_stem="$participant_dir/features/lesion/${participant_id}_desc-disconnectome"
```

Also update the log file reference at ≈ line 99 in the header comment:
```
# Old: <participant_dir>/features/lesion/logs/<id>_les_SDC.txt (BIDS)
# New: <participant_dir>/features/lesion/logs/<id>_desc-disconnectome.txt (BIDS)
```

And update the synopsis comment at line 45:
```
# Old: ROOT/…/<participant_id>/features/lesion/<participant_id>_les_SDC.nii.gz
# New: ROOT/…/<participant_id>/features/lesion/<participant_id>_desc-disconnectome.nii.gz
```

---

### Change 3 — Update the synopsis/options block

Add the `-r` option to the SYNOPSIS and OPTIONS sections at the top of the file:

In SYNOPSIS:
```
#   Folder mode : run_disco.sh -l DIR   -o OUTDIR [-r FROM:TO] [options]
#   CSV mode    : run_disco.sh -l FILE  -o OUTDIR [-r FROM:TO] [options]
```

In OPTIONS (after `-p`):
```
#   -r FROM:TO  Rename: replace FROM with TO in every output filename stem.
#               Applies only in folder and CSV mode (not BIDS mode).
#               Colon-separated. Example:
#                 -r "_label-lesion_mask:_desc-disconnectome"
```

---

## Verification

After making the changes, test with a dry run:
```bash
cd /home/chrisfoulon/neuro_apps/BCBToolKitLINUX/BCBToolKit
echo "running dry-run test..."
# Create two dummy nii.gz file names to test stem substitution logic:
mkdir -p /tmp/test_disco_rename
touch "/tmp/test_disco_rename/sub-001_space-MNI152NLin6Asym_res-1_label-lesion_mask.nii.gz"
touch "/tmp/test_disco_rename/sub-002_space-MNI152NLin6Asym_res-1_label-lesion_mask.nii.gz"
bash run_disco.sh -l /tmp/test_disco_rename -o /tmp/disco_out \
    -r "_label-lesion_mask:_desc-disconnectome" -d
# Expected dry-run output should show:
#   sub-001_space-MNI152NLin6Asym_res-1_desc-disconnectome
#   sub-002_space-MNI152NLin6Asym_res-1_desc-disconnectome
rm -rf /tmp/test_disco_rename /tmp/disco_out
```

Expected output lines should contain `_desc-disconnectome`, NOT `_label-lesion_mask`.

---

## Commit

BCBToolKit does not appear to be a git repository.
Save the modified file and notify the BCBlib maintainer that the change is complete
so the lesion_features pipeline plan can proceed.

If version control is added later, the appropriate conventional commit message would be:
```
feat(run_disco): add -r FROM:TO rename flag for folder/CSV output stems;
rename BIDS mode output from _les_SDC to _desc-disconnectome
```
