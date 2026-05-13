# damage_profile — Integration Context

## Level 1: Plain English

`damage_profile` is a new subpackage with no existing logic to inherit. It relies on two
existing bcblib components: the `_check_image_space` function from `best_overlap.py` (space
validation + orientation checks) and `load_nifti` / `NiftiLike` from `bcblib.imaging.io`
(path-or-image input normalisation). All other logic is new.

The download manager writes to `~/.bcblib/atlases/` using only stdlib (`urllib`, `zipfile`,
`pathlib`) — no new runtime dependencies.

---

## Level 2: API Table

| Symbol | Module | Purpose | Inputs | Outputs | Side-effects |
|--------|--------|---------|--------|---------|--------------|
| `_check_image_space` | `best_overlap` | Validate shape, affine, orientation of two images | two Nifti1Images + names + on_mismatch | dict with issues list | raises ValueError or prints warning |
| `load_nifti` | `bcblib.imaging.io` | Normalise NiftiLike → Nifti1Image | str / Path / Nifti1Image | Nifti1Image | none |
| `is_nifti` | `bcblib.imaging.io` | Check whether a path is a NIfTI file | str / Path | bool | none |
| `NiftiLike` | `bcblib.imaging._types` | Type alias for accepted image inputs | — | Union[str, bytes, PathLike, Nifti1Image] | — |
| `resample_to_img` | `nilearn.image` | Resample atlas to subject voxel grid | source_img, target_img, interpolation | Nifti1Image | none |
| `detect_atlas_format` | `_atlas` | Identify atlas format from source path | Path or str | `'directory'` / `'4d_nifti'` / `'label_nifti'` | none |
| `load_atlas` | `_atlas` | Load atlas into name→weight dict | AtlasSpec | dict[str, np.ndarray] | none |
| `check_and_resample` | `_space` | Validate space + resample atlas if needed | subject_img, atlas_img, name, on_mismatch | np.ndarray (resampled data) | warning print if mismatch |
| `compute_region_stats` | `_stats` | Core overlap stats per region | subject_data np.ndarray, atlas_dict, min_overlap_voxels | pd.DataFrame | none |
| `get_preset_atlas` | `_atlas_manager` | Resolve a named atlas: cache → nilearn → download | name str, path optional, assume_yes bool | dict[str, np.ndarray] | may download to `~/.bcblib/atlases/` |
| `damage_profile` | `_core` | Top-level: map + atlases → DataFrames | subject_map NiftiLike, atlases list[AtlasSpec], kwargs | dict[str, pd.DataFrame] | writes CSVs if output_dir given |

---

## Level 3: Key Integration Points

### Reusing `_check_image_space`

```python
# _space.py
from bcblib.tools.best_overlap import _check_image_space

def check_and_resample(subject_img, atlas_img, atlas_name, on_space_mismatch="error"):
    info = _check_image_space(
        subject_img, "subject_map",
        atlas_img, atlas_name,
        on_mismatch=on_space_mismatch,
    )
    if info["issues"]:          # affine/orientation mismatch — resample
        from nilearn.image import resample_to_img
        atlas_img = resample_to_img(atlas_img, subject_img, interpolation="nearest")
    return atlas_img.get_fdata()
```

### NiftiLike input normalisation

```python
# _core.py
from bcblib.imaging.io import load_nifti

def damage_profile(subject_map, atlases, ...):
    subject_img = load_nifti(subject_map)   # accepts str/Path/Nifti1Image
    subject_data = subject_img.get_fdata()
    ...
```

### Atlas format detection using `is_nifti`

```python
# _atlas.py
from bcblib.imaging.io import is_nifti

def detect_atlas_format(source):
    p = Path(source)
    if p.is_dir():
        return 'directory'
    if is_nifti(str(p)):                    # reuses existing bcblib check
        img = nib.load(str(p))
        return '4d_nifti' if img.ndim == 4 else 'label_nifti'
    raise ValueError(f"Cannot determine atlas format for: {source}")
```

### Why imaging.stats is NOT reused for per-region stats

`bcblib.imaging.stats.image_stats(img, mask)` takes NiftiLike inputs and operates on full
images. For `compute_region_stats` we already have loaded numpy arrays and need to slice by
atlas weights per region — building a temporary Nifti1Image per region just to call
`image_stats` adds object-creation overhead with no benefit. `weighted_mean_overlap` and
`fraction_covered` are also not in `image_stats`. Numpy directly on masked arrays is simpler
and faster.

### Label file parsing — three formats

`_parse_label_file` (in `_atlas.py`) is split into two helpers:
- `_parse_fsl_xml_labels(text)` — handles FSL atlas XML (`<label index="N">Name</label>`)
- `_parse_text_labels(text)` — handles plain text (one name per line, 1-indexed) and TSV
  (first column = integer index, second column = name)

Detection is by checking whether the file starts with `<`. JHU label files are FSL XML;
Tian label files are plain text; Buckner label file is TSV.

Label files are **optional at load time**: `get_preset_atlas` checks `Path.exists()` before
building the label path and passes `None` if absent, giving auto-numbered region names.

### Atlas loading — directory format (Rojkova pattern)

```python
# _atlas.py
def _load_directory(spec):
    files = sorted(Path(spec.source).glob("*.nii*"))
    result = {}
    for f in files:
        name = f.name.replace(".nii.gz", "").replace(".nii", "")
        data = nib.load(f).get_fdata()
        if spec.threshold > 0:
            data = np.where(data >= spec.threshold, data, 0.0)
        if data.any():
            result[name] = data
    return result
```

### compute_region_stats inner loop

```python
# _stats.py
def compute_region_stats(subject_data, atlas_dict, min_overlap_voxels=1):
    rows = []
    for region_name, weights in atlas_dict.items():
        mask = weights > 0
        n_total = int(mask.sum())
        overlap_vals = subject_map[mask]
        n_nonzero = int((overlap_vals > 0).sum())
        if n_nonzero < min_overlap_voxels:
            continue
        rows.append({
            "region_name": region_name,
            "n_voxels_region": n_total,
            "n_voxels_overlap": n_nonzero,
            "fraction_covered": n_nonzero / n_total if n_total else 0.0,
            "mean_overlap": float(subject_data[mask].mean()),
            "weighted_mean_overlap": (
                float((subject_data[mask] * weights[mask]).sum() / weights[mask].sum())
                if weights[mask].sum() > 0 else float("nan")
            ),
            "sum_overlap": float(subject_data[mask].sum()),
            "p90_overlap": float(np.percentile(overlap_vals[overlap_vals > 0], 90)),
            "p95_overlap": float(np.percentile(overlap_vals[overlap_vals > 0], 95)),
        })
    return pd.DataFrame(rows).sort_values("mean_overlap", ascending=False)
```

---

## Maintenance Opportunities in Target Files

### Medium Priority (Boy Scout Rule — consider during implementation)

- [ ] `best_overlap.py:106` — `_check_image_space` is private but needed here.
  Long-term home is a shared `bcblib/tools/_image_utils.py`. Note this in a comment
  but do not move it now (out of scope, would break existing tests).
- [ ] `best_overlap.py:1` — no `__all__` defined; `_check_image_space` import
  will work but is technically accessing a private name across modules. Acceptable for now.
