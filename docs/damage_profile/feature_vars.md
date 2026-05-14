# damage_profile — Feature Variables

## Identity

```
FEATURE_NAME=damage_profile
FEATURE_SLUG=damage_profile
PACKAGE_PATH=bcblib/tools/damage_profile/
SCRIPTS_PATH=bcblib/scripts/run_damage_profile.py
TEST_FILE=bcblib/tests/test_damage_profile.py
CLI_ENTRY=bcb-damage-profile
```

## Complexity Assessment

```
TASK_COMPLEXITY=MEDIUM
IMPLEMENTATION_APPROACH=bottom-up TDD; pure functions first (stats, atlas loading),
  then space handling, then integration (core), then download manager, then CLI
KEY_CHALLENGES=atlas format detection; download consent flow; nilearn resample
  behaviour across template families; 4D memory usage for large atlases
RESOURCE_REQUIREMENTS=nibabel, numpy, pandas, nilearn (all existing deps); no new deps
```

## Public API

```python
# bcblib/tools/damage_profile/__init__.py exports:
EXPORTS=damage_profile, AtlasSpec, get_preset_atlas, list_preset_atlases
```

## Data structures

```python
ATLAS_SPEC_FIELDS=source, name, threshold, label_file
ATLAS_INFO_FIELDS=full_name, url, size_mb, format, space, citation
PRESET_ATLAS_KEYS=jhu_wm_prob, jhu_wm_labels, rojkova, tian_s1, tian_s2,
                  buckner_7n, yeh_hcp1065, schaefer_200_7n, schaefer_300_7n,
                  schaefer_400_7n, aal
```

## Output DataFrame columns

```
COLUMNS=region_id, region_name, n_voxels_region, n_voxels_overlap,
        fraction_covered, mean_overlap, weighted_mean_overlap,
        sum_overlap, p90_overlap, p95_overlap
```

## Cache directory

```
CACHE_ROOT=~/.bcblib/
ATLAS_CACHE=~/.bcblib/atlases/<name>/
```

## Internal module map

```
_atlas.py          → detect_atlas_format, load_atlas
_atlas_manager.py  → PRESET_ATLASES, get_preset_atlas, list_preset_atlases,
                      get_atlas_dir, _download_atlas
_space.py          → check_and_resample
_stats.py          → compute_region_stats
_core.py           → damage_profile (top-level)
```

## Reused components

```
bcblib.tools.best_overlap._check_image_space  → imported directly into _space.py
bcblib.imaging.io.load_nifti                  → NiftiLike→Nifti1Image helper
bcblib.imaging._types.NiftiLike               → type annotation
nilearn.image.resample_to_img                 → atlas resampling
```

## Integration strategy

```
STRATEGY=NEW (greenfield within existing package)
COEXISTENCE=no conflict with best_overlap; damage_profile is a separate tool
DEPRECATION=none; replaces the old disco_tract_profile plan only (folder renamed)
```
