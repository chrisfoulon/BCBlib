# lesion_features — Feature Variables

## Identity

```
FEATURE_NAME=lesion_features
FEATURE_SLUG=lesion_features
PACKAGE_PATH=bcblib/tools/lesion_features/
SCRIPTS_PATH=bcblib/scripts/run_lf_preprocess.py, bcblib/scripts/run_lesion_features.py
TEST_FILE=bcblib/tests/test_lesion_features.py
CLI_ENTRY_1=bcb-lf-preprocess
CLI_ENTRY_2=bcb-lesion-features
BRANCH=feat/lesion-features
```

## Complexity Assessment

```
TASK_COMPLEXITY=MEDIUM-LARGE
IMPLEMENTATION_APPROACH=bottom-up TDD; BIDS utilities first, then space
  normalization, then disco runner, then orchestration pipelines, then CLIs
KEY_CHALLENGES=BIDS path parsing; subprocess management for run_disco.sh;
  batch orchestration across subjects; EBRAINS atlas preset set (new presets
  needed in damage_profile._atlas_manager); ses- optional propagation
RESOURCE_REQUIREMENTS=nibabel, numpy, pandas, nilearn (existing); subprocess
  (stdlib); templateflow, nitransforms (existing via damage_profile._space);
  damage_profile (existing BCBlib tool)
BCBTOOLKIT_PATH=/home/chrisfoulon/neuro_apps/BCBToolKitLINUX/BCBToolKit/run_disco.sh
```

## Prerequisites

```
1. BCBToolKit run_disco.sh modification: add -r FROM:TO flag (folder/CSV mode)
   See: docs/lesion_features/bcbtoolkit_run_disco_instruction.md
   Status: [PENDING — must be done before T3]

2. EBRAINS atlas set confirmed by user: which atlases to include
   Status: [USER_INPUT_REQUIRED — needed before T6]
```

## Public API

```python
# bcblib/tools/lesion_features/__init__.py exports:
EXPORTS=preprocess_batch, extract_features_batch, EBRAINS_ATLAS_SPECS,
        parse_bids_entities, build_lf_output_path
```

## BIDS naming convention (LF pipeline)

```
Input  lesion: sub-001_space-<X>_[res-<N>_]label-lesion_mask.nii.gz
Output lesion: sub-001_space-MNI152NLin6Asym_res-1_label-lesion_mask.nii.gz
Output disco:  sub-001_space-MNI152NLin6Asym_res-1_desc-disconnectome.nii.gz
LF CSV:        sub-001_space-MNI152NLin6Asym_LF-lesion_atlas-<Name>.csv
               sub-001_space-MNI152NLin6Asym_LF-disconnectome_atlas-<Name>.csv
LF TSV:        sub-001_space-MNI152NLin6Asym_desc-lesion_mapstats.tsv
               sub-001_space-MNI152NLin6Asym_desc-disconnectome_mapstats.tsv

With session:  sub-001/ses-01/anat/sub-001_ses-01_space-MNI152NLin6Asym_...
```

## BIDS directory layout (two derivatives)

```
derivatives/
├── lesion_features_prep/         # stage 1 output (preprocessing)
│   ├── sub-001/
│   │   └── anat/
│   │       ├── sub-001_space-MNI152NLin6Asym_res-1_label-lesion_mask.nii.gz
│   │       └── sub-001_space-MNI152NLin6Asym_res-1_desc-disconnectome.nii.gz
│   └── sub-002/...
└── lesion_features/              # stage 2 output (LF features)
    ├── sub-001/
    │   └── anat/
    │       ├── sub-001_space-MNI152NLin6Asym_LF-lesion_atlas-JHU.csv
    │       ├── sub-001_space-MNI152NLin6Asym_LF-disconnectome_atlas-JHU.csv
    │       ├── sub-001_space-MNI152NLin6Asym_desc-lesion_mapstats.tsv
    │       └── sub-001_space-MNI152NLin6Asym_desc-disconnectome_mapstats.tsv
    └── sub-002/...
```

## Internal module map

```
_bids.py        → parse_bids_entities, build_lf_csv_path, build_lf_tsv_path,
                  build_prep_path, iter_bids_lesions
_preprocess.py  → extract_space_from_filename, extract_resolution_from_filename,
                  detect_resolution_from_shape, normalise_lesion_to_mni6,
                  preprocess_one
_disco.py       → find_bcbtoolkit, predict_disco_output,
                  run_disco_batch, collect_disco_outputs
_pipeline.py    → preprocess_batch, extract_features_batch
_constants.py   → TARGET_SPACE, TARGET_RES, DEFAULT_BCBTOOLKIT, EBRAINS_ATLAS_SPECS
```

## Reused components

```
bcblib.tools.damage_profile._space._apply_templateflow_warp
bcblib.tools.damage_profile._space._detect_template_family
bcblib.tools.damage_profile._space.check_and_resample
bcblib.tools.damage_profile.damage_profile
bcblib.tools.damage_profile._atlas_manager.PRESET_ATLASES, get_preset_atlas
bcblib.imaging.io.load_nifti
nilearn.image.resample_to_img  (NN for lesions, linear for disconnectomes)
```
