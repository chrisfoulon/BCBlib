# imaging_stats_layered — Integration Context

## Level 1: Plain English

All functions in `bcblib/imaging/stats.py`, and the arithmetic/binarize functions in
`bcblib/imaging/math.py`, currently load a `NiftiLike` input and operate on it in a single
layer. There are no underlying numpy-level functions. This means callers who already hold a
loaded `np.ndarray` (e.g. the inner loop of `damage_profile._stats`) must go through the full
image-load path on every call.

The refactor splits each function into two layers:
1. A private `_fn_array(data, ...)` that operates on a plain `np.ndarray`.
2. The existing public `fn(img, ...)` which now just calls `load_nifti` → `_fn_array`.

The public API is unchanged. No callers need updating. The array-level functions become
available for import by internal consumers (like `_stats.py` in `damage_profile`).

`math.py` already does this correctly for morphology ops (`dilate`, `erode`, `apply_mask`).
Those are left untouched; only the remaining functions are brought in line.

`orient.py` and `info.py` are excluded — their operations inherently require nibabel header
and affine objects, so no meaningful array-only core can be extracted.

---

## Level 2: Function Map

### stats.py

| Public function | Array core | Core signature |
|-----------------|------------|---------------|
| `centre_of_gravity` | `_centre_of_gravity_array` | `(data, round_coord) → tuple` |
| `volume_count` | `_volume_count_array` | `(data, threshold, ratio) → int\|float` |
| `laterality_ratio` | `_laterality_ratio_array` | `(data, orientation: tuple) → float` |
| `reduce_axis` | `_reduce_axis_array` | `(data, method, axis) → np.ndarray` |
| `image_stats` | `_image_stats_array` | `(data, voxel_vol) → dict` |
| `percentile` | `_percentile_array` | `(data, q) → float` |
| `robust_range` | `_robust_range_array` | `(data, low, high) → tuple` |
| `histogram` | `_histogram_array` | `(data, bins) → (counts, edges)` |
| `centre_of_gravity_distance` | — | delegates to `centre_of_gravity`; no array core needed |

Note: `_image_stats_array` takes `voxel_vol` (a float computed from the affine) as a
parameter rather than the full affine, keeping the core free of nibabel objects.

### math.py

| Public function | Array core | Core signature |
|-----------------|------------|---------------|
| `binarize` | `_binarize_array` | `(data, thr) → np.ndarray` |
| `threshold` | `_threshold_array` | `(data, low, high) → np.ndarray` |
| `add` | `_add_arrays` | `(a, b) → np.ndarray` |
| `subtract` | `_subtract_arrays` | `(a, b) → np.ndarray` |
| `multiply` | `_multiply_arrays` | `(a, b) → np.ndarray` |

### manipulate.py

| Public function | Array core | Core signature |
|-----------------|------------|---------------|
| `merge_images` | `_merge_arrays` | `(arrays: list[np.ndarray], axis) → np.ndarray` |
| `split_image` | `_split_array` | `(data, axis) → list[np.ndarray]` |

---

## Level 3: Key Considerations

### Backward compatibility
The public signatures do not change. Existing callers, tests, and CLI commands are unaffected.
The only observable change is the addition of new private functions in the module namespace.

### `laterality_ratio` — orientation dependence
The current implementation reads `nib.aff2axcodes(nii.affine)` to determine left/right.
The array core therefore takes `orientation: tuple` (e.g. `('R', 'A', 'S')`) as an explicit
parameter. The public wrapper extracts this from the loaded image before calling the core.

### `image_stats` — voxel volume
The current implementation computes `voxel_vol` from `np.abs(np.linalg.det(nii.affine[:3,:3]))`.
The core takes this as a float argument; the wrapper computes it before delegating.

### `extract_roi` — deferred
`extract_roi` also updates the affine origin to reflect the spatial crop. A pure array core
would need to return `(cropped_array, new_affine)`, making the signature asymmetric with all
other array cores. Defer this to a future pass; the function remains as-is for now.

### `copy_geometry` — excluded
Header-only operation. No array transformation involved.

### `orient.py` / `info.py` — excluded
All functions operate on nibabel orientation codes, affine matrices, or header fields.
There is no numpy array computation to isolate.

---

## Maintenance Note

This refactor was identified as a design flaw during `damage_profile` planning (T5.0).
The two new functions added by `damage_profile` (`fraction_covered`, `weighted_region_mean`)
already implement the correct pattern and do not need to be touched by this refactor.
