# imaging_stats_layered — Feature Variables

## Identity

```
FEATURE_NAME=imaging_stats_layered
FEATURE_SLUG=imaging_stats_layered
BRANCH=refactor/imaging-stats-layered-design
TARGET_FILES=bcblib/imaging/stats.py, bcblib/imaging/math.py, bcblib/imaging/manipulate.py
TEST_FILE=bcblib/tests/test_imaging_stats.py (extend), bcblib/tests/test_imaging_math.py (extend)
```

## Complexity Assessment

```
TASK_COMPLEXITY=MEDIUM
IMPLEMENTATION_APPROACH=module-by-module refactor; stats.py first (highest value),
  then math.py arithmetic ops, then manipulate.py merge/split.
  orient.py and info.py are excluded — pattern does not apply (header/affine ops).
KEY_CHALLENGES=keeping public API backward-compatible; laterality_ratio needs
  orientation tuple from header (core signature is array + ornt, not array alone);
  extract_roi returns (array, affine) pair, not pure array
RESOURCE_REQUIREMENTS=no new dependencies
```

## Design Pattern

```
PATTERN=two-layer design
LAYER_1=_fn_array(data: np.ndarray, ...) -> result
  Pure numpy, no I/O, no nibabel objects.
  Named with leading underscore to signal internal use.
LAYER_2=fn(img: NiftiLike, ...) -> result
  Loads image with load_nifti, extracts array, delegates to _fn_array.
  Preserves existing public signature exactly.
```

## Scope per module

```
stats.py — 9 functions to split:
  centre_of_gravity         → _centre_of_gravity_array(data, affine)
  volume_count              → _volume_count_array(data, threshold)
  laterality_ratio          → _laterality_ratio_array(data, orientation)
  reduce_axis               → _reduce_axis_array(data, method, axis)
  image_stats               → _image_stats_array(data, voxel_vol)
  percentile                → _percentile_array(data, q)
  robust_range              → _robust_range_array(data, low, high)
  histogram                 → _histogram_array(data, bins)
  centre_of_gravity_distance → delegates to centre_of_gravity (no array core needed)

math.py — 5 functions to split (arith + binarize/threshold):
  binarize   → _binarize_array(data, thr)
  threshold  → _threshold_array(data, low, high)
  add        → _add_arrays(a, b)
  subtract   → _subtract_arrays(a, b)
  multiply   → _multiply_arrays(a, b)
  (dilate, erode, apply_mask already layered — no change)

manipulate.py — 2 trivial splits:
  merge_images → _merge_arrays(arrays, axis)
  split_image  → _split_array(data, axis)
  (extract_roi: borderline — affine update needed, defer)
  (copy_geometry: header op — excluded)

orient.py — excluded (nibabel orientation objects, no array core)
info.py    — excluded (header field access, no array core)
```

## New public functions added by damage_profile (T5.0)

```
fraction_covered and weighted_region_mean are added to stats.py
as part of damage_profile T5.0, using the correct layered design.
This refactor task does NOT add those — they are handled in damage_profile.
This task retrofits the existing functions only.
```
