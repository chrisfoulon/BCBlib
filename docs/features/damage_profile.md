# damage_profile — Feature Specification

## What this feature does

Given a single-subject neuroimaging map (binary lesion or probability disconnectome) and one or
more brain atlases, compute per-region descriptive overlap statistics and write one CSV per atlas.
The tool is atlas-agnostic: it accepts any standard format and handles resolution mismatches
automatically.

---

## Scientific rationale

Existing atlas-overlap tools either:
- Report only the highest-probability atlas voxel hit by the lesion (misleading for
  partial-overlap cases), or
- Are tied to a specific atlas or pipeline (e.g. FSLeyes, MRIcron, LQT).

This module computes principled overlap statistics for any combination of subject map type and
atlas, making them directly comparable across atlases and usable for downstream group-level
analysis.

**Subject map types:**

| Type | Values | Interpretation |
|------|--------|----------------|
| Binary lesion | {0, 1} | Is voxel v in the lesion? |
| Graded/probability lesion | [0, 1] | How likely/severe is involvement at v? |
| Disconnectome | [0, 1] | Fraction of healthy subjects with connectivity through v |

All types use the same formulas — the distinction is only in interpretation.

**Core statistics per region R:**

```
mean_overlap(R)          = mean(map[voxels_of_R])           # unweighted
weighted_mean_overlap(R) = sum(map[v] * w[v]) / sum(w[v])   # w = atlas probability
fraction_covered(R)      = n_nonzero_voxels / n_total_voxels
```

The denominator is always all voxels in R (not just nonzero ones), so `mean_overlap` combines
coverage and intensity into a single number — which is appropriate for both binary and probability
maps.

---

## Inputs

| Argument | Type | Description |
|----------|------|-------------|
| `subject_map` | path or Nifti1Image | Binary lesion or probability map, values ≥ 0 |
| `atlases` | list of AtlasSpec | One or more atlas specifications (see below) |
| `min_overlap_voxels` | int, default 1 | Minimum nonzero overlap voxels to include a region in output |
| `output_dir` | path or None | If provided, save one CSV per atlas here |

### Atlas specification (`AtlasSpec`)

An `AtlasSpec` is a small dataclass (or plain dict) with:

```python
@dataclass
class AtlasSpec:
    source: str | Path          # file path (4D, label, or directory) — see formats below
    name: str                   # short name used in output filenames and column headers
    threshold: float = 0.0      # minimum atlas probability to include a voxel in a region
                                # (relevant for probabilistic atlases; JHU convention: 0.25)
    label_file: str | Path | None = None  # CSV/JSON: label_id → region_name mapping
```

### Atlas file formats (in order of priority)

1. **4D NIfTI** — one volume per region, values in [0, 1] or binary.  
   Region names from `label_file` or auto-numbered.

2. **Label NIfTI** — single 3D volume with integer labels.  
   Region names from `label_file` (required for meaningful output).

3. **Directory of binary masks** — one `.nii.gz` per region, filename = region name.

A `load_atlas()` helper detects the format automatically and returns
`dict[str, np.ndarray]` mapping region name → voxel-weight array.

---

## Outputs

One `pandas.DataFrame` per atlas (saved as CSV if `output_dir` is given):

| Column | Description |
|--------|-------------|
| `region_id` | Integer label or volume index |
| `region_name` | Name from label file, or filename/auto |
| `n_voxels_region` | Total voxels in the atlas region |
| `n_voxels_overlap` | Voxels with `subject_map > 0` |
| `fraction_covered` | `n_voxels_overlap / n_voxels_region` |
| `mean_overlap` | Mean subject_map value across all region voxels |
| `weighted_mean_overlap` | Atlas-prob-weighted mean (NaN for binary atlases) |
| `sum_overlap` | Sum of subject_map values within region |
| `p90_overlap` | 90th percentile of subject_map within nonzero voxels |
| `p95_overlap` | 95th percentile |

Rows with `n_voxels_overlap < min_overlap_voxels` are dropped.
Output is sorted by `mean_overlap` descending.

The function returns `dict[str, pd.DataFrame]` keyed by atlas name.

---

## Space handling

The tool operates in the subject map's voxel space. When an atlas has a different
shape or affine:

1. **Same shape, same affine** → use directly.
2. **Different resolution, same template family** (affines differ only by a scale factor) →
   resample atlas to subject space using `nilearn.image.resample_to_img`.
3. **Cross-template-family mismatch** (e.g. MNI152NLin6Asym vs MNI152NLin2009cAsym) →
   raise `ValueError` by default; `on_space_mismatch='warn'` downgrades to a warning and
   proceeds with affine resampling (approximate — user's responsibility).

Detection uses the same `_check_image_space` logic already in `best_overlap.py`.

---

## Atlas data management

### User cache directory

Preset atlases are stored in `~/.bcblib/atlases/<name>/`, created on first download.
`pathlib.Path.home() / '.bcblib'` resolves correctly on Linux, macOS, and Windows.

```
~/.bcblib/
  atlases/
    jhu_wm/          # from FSL install or downloaded
    rojkova/         # downloaded from BCBlab Dropbox
    tian_s1/         # downloaded from GitHub
    tian_s2/
    buckner_7n/      # downloaded from Diedrichsen lab GitHub
    yeh_hcp1065/     # downloaded from brain.labsolver.org
```

Nilearn-managed atlases (Schaefer, AAL) use nilearn's own cache and are not duplicated here.

### Resolution order for finding an atlas

When the user requests a preset atlas by name, the lookup proceeds in this order:

1. `~/.bcblib/atlases/<name>/` — previously downloaded cache
2. **JHU only**: `$FSLDIR/data/atlases/JHU/` — FSL installation (most users already have it)
3. **Schaefer / AAL**: `nilearn.datasets.fetch_*` — uses nilearn's cache transparently
4. Not found → prompt user for consent to download, then download + extract to cache

### Consent and non-interactive use

Before any download, the tool prints the atlas name, approximate size, and source URL,
then asks: `Download? [y/N]`. The user types `y` to proceed.

For scripted/non-interactive use: `assume_yes=True` in the Python API or `--yes` on the CLI
skips the prompt. The environment variable `BCBLIB_YES=1` also skips it (useful in pipelines).

### Preset atlas registry

```python
PRESET_ATLASES = {
    "jhu_wm": AtlasInfo(
        full_name="JHU DTI White Matter Tractography Atlas",
        url="...",          # NeuroVault or FSL mirror
        size_mb=5,
        format="4d_nifti",
        space="MNI152NLin6Asym",
        citation="Hua et al. NeuroImage 2008",
    ),
    "rojkova": AtlasInfo(
        full_name="Rojkova et al. Functional White Matter Atlas",
        url="https://www.dropbox.com/s/dnbt3gdm1iledkv/Atlas_Rojkova.zip?dl=1",
        size_mb=None,       # to be confirmed after inspecting zip
        format="directory", # assumed — to confirm after download
        space="MNI152NLin6Asym",  # to confirm
        citation="Rojkova et al. Neuropsychologia 2016",
    ),
    "tian_s1": AtlasInfo(...),
    "tian_s2": AtlasInfo(...),
    "buckner_7n": AtlasInfo(...),
    "yeh_hcp1065": AtlasInfo(...),
}
```

A few fields for Rojkova are marked "to confirm" — the zip contents and exact template space
need to be verified by downloading and inspecting the file before the registry entry is finalised.

### User-provided atlases (no download)

The `AtlasSpec` `source` field accepts any of:
- Path to a **directory** of `.nii` / `.nii.gz` files (one per region)
- Path to a **4D NIfTI** (one volume per region)
- Path to a **label NIfTI** (integer-labelled 3D volume) + optional `label_file`

No download, no registry. Works entirely from user-supplied files.

---

## Implementation plan

```
bcblib/
  tools/
    damage_profile/
      __init__.py          ← public: damage_profile(), AtlasSpec, get_preset_atlas()
      _atlas.py            ← load_atlas(), detect_atlas_format()
      _atlas_manager.py    ← PRESET_ATLASES, get_preset_atlas(), _download_atlas()
      _space.py            ← check_and_resample(), wraps best_overlap._check_image_space
      _stats.py            ← compute_region_stats() → DataFrame
      _core.py             ← damage_profile() top-level function
  scripts/
    run_damage_profile.py  ← CLI: bcb-damage-profile
  tests/
    test_damage_profile.py ← TDD, synthetic arrays
```

### Function signatures (draft)

```python
def damage_profile(
    subject_map: str | Path | nib.Nifti1Image,
    atlases: list[AtlasSpec],
    min_overlap_voxels: int = 1,
    output_dir: str | Path | None = None,
    on_space_mismatch: str = "error",
) -> dict[str, pd.DataFrame]:
    ...
```

```python
def load_atlas(
    spec: AtlasSpec,
) -> dict[str, np.ndarray]:
    # detect format, apply threshold, return name→weight_array
    ...
```

```python
def compute_region_stats(
    subject_data: np.ndarray,
    atlas_dict: dict[str, np.ndarray],
    min_overlap_voxels: int = 1,
) -> pd.DataFrame:
    ...
```

### CLI (bcb-damage-profile)

```
bcb-damage-profile \
  --map patient_disconnectome.nii.gz \
  --atlas jhu:/path/to/JHU-ICBM-tracts-prob-2mm.nii.gz \
  --atlas schaefer:/path/to/Schaefer100.nii.gz:schaefer_labels.csv \
  --min-overlap-voxels 5 \
  --output-dir results/patient01/
```

---

## Open decisions

| # | Question | Default proposal |
|---|----------|-----------------|
| 1 | Should `weighted_mean_overlap` be computed for binary atlases too (weight=1.0, so same as mean)? | Return NaN for binary atlases to signal "not meaningful" |
| 2 | Should the output CSV include zero-overlap regions (with NaN stats)? | Drop them by default; `--keep-zeros` flag for CLI |
| 3 | TemplateFlow integration (query atlases by name)? | Phase 2 — v1 requires file paths |
| 4 | Support passing both a lesion AND a disconnectome in one call? | Phase 2 — v1 takes one map at a time; caller can loop |

---

## Dependencies

No new dependencies:
- `nibabel` — NIfTI I/O
- `numpy` — array operations
- `pandas` — output DataFrame
- `nilearn` — resampling (already in bcblib deps)

---

## Relationship to existing modules

- Replaces the `disco_tract_profile` plan (folder renamed to `damage_profile`).
- Reuses `_check_image_space` from `best_overlap.py` (refactor to shared utility later if needed).
- No Bayesian model — descriptive stats only. Users run their own group-level stats on the CSVs.
- The `best_overlap.py` workflow (preserved connections + Bayesian regression) remains separate.

---

## Atlas evaluation

The atlases proposed by the team (Schaefer, Tian, Buckner, Glasser, AAL, Yeh) are assessed
below for access method, template space, and practical feasibility for this module.

### The space problem

Disconnectome maps and lesions (after pre-processing) will be in **FSL MNI 1mm
(MNI152NLin6Asym)**. Several cortical atlases live in **SPM MNI (MNI152NLin2009cAsym)** — a
different non-linear registration. Simple affine resampling (`nilearn.resample_to_img`) between
these two will introduce ~1–2 mm spatial error, which is typically acceptable for coarse atlas
overlap but is not a registration. The tool should warn the user when this cross-family situation
is detected. A proper solution (TemplateFlow warp transforms) is deferred to Phase 2.

---

### Cortical + whole-brain parcellation atlases

#### Schaefer 200 / 300 / 400 (7 Networks)

| Property | Detail |
|----------|--------|
| Coverage | Cortex only (no subcortex, no cerebellum) |
| Template space | **MNI152NLin2009cAsym (SPM MNI)** |
| Resolutions | 1 mm and 2 mm |
| Access | `nilearn.datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7)` — zero manual steps, auto-download and caching. Also in TemplateFlow. |
| Format | Label NIfTI + TSV label file |
| License | Free, cite Schaefer et al. 2018 |
| **Feasibility** | **Very easy** — already a nilearn dependency |
| Space note | SPM MNI ≠ FSL MNI. Affine resample adds ~1–2 mm error. Acceptable for most use cases. |

#### AAL (116 regions / AAL3)

| Property | Detail |
|----------|--------|
| Coverage | Cortex (90) + subcortical basal ganglia + cerebellum — whole brain in one atlas |
| Template space | **MNI152NLin2009cAsym (SPM MNI)** (designed with SPM) |
| Resolutions | 1 mm and 2 mm |
| Access | `nilearn.datasets.fetch_atlas_aal(version='SPM12')` — zero manual steps. AAL3 adds brainstem. |
| Format | Label NIfTI + CSV label file |
| License | Free, cite Tzourio-Mazoyer et al. 2002 / Rolls et al. 2020 (AAL3) |
| **Feasibility** | **Very easy** — already a nilearn dependency |
| Caveat | Regions are large and coarse (especially cerebellum). Suitable for a broad-brush view. |

#### Glasser HCP-MMP1 (360 cortical regions) — excluded from v1

The Glasser atlas is fundamentally a surface parcellation (vertices from multimodal
surface-based registration). Volumetric projections exist but the original authors explicitly
caution against using them for volumetric analyses — the projection is lossy, particularly in
sulci where surface geometry does not map cleanly to voxels. For lesion and disconnectome
profiling in volume space this is a conceptual mismatch, not just a resolution issue. The
generic `AtlasSpec` will accept it as a file if a user provides one, but it will not be
bundled or documented as a supported atlas. Recommend Schaefer instead (volumetric NIfTI is
first-class output of the Schaefer pipeline).

---

### Subcortical atlas

#### Tian Subcortical (S1: 16 regions / S2: 32 regions)

| Property | Detail |
|----------|--------|
| Coverage | Subcortex only (basal ganglia, thalamus, amygdala, hippocampus, etc.) |
| Template space | **Available in both** MNI152NLin6Asym (FSL MNI) and MNI152NLin2009cAsym — use the FSL MNI version directly with disconnectomes |
| Access | Manual download from [github.com/yetianmed/subcortex](https://github.com/yetianmed/subcortex) or NITRC |
| Format | Label NIfTI |
| License | Free, cite Tian et al. *Nature Neuroscience* 2020 |
| **Feasibility** | **Easy** — file download only; FSL MNI version means no space mismatch |
| Note | This is a subcortical-only atlas, always combined with a cortical atlas (Schaefer) and cerebellar atlas (Buckner) for whole-brain coverage |

---

### Cerebellar atlas

#### Buckner 7 Networks (cerebellum)

| Property | Detail |
|----------|--------|
| Coverage | Cerebellum only |
| Template space | **MNI152NLin6Asym (FSL MNI)** available from the Diedrichsen lab cerebellar atlas collection |
| Access | Manual download from [github.com/DiedrichsenLab/cerebellar_atlases](https://github.com/DiedrichsenLab/cerebellar_atlases) |
| Format | Label NIfTI |
| License | Free, cite Buckner et al. *Journal of Neurophysiology* 2011 |
| **Feasibility** | **Easy** — file download only; FSL MNI version available |
| Note | Cerebellum only. Typically used as the cerebellar component alongside Schaefer (cortex) + Tian (subcortex) |

---

### White matter tract atlases

#### JHU DTI White Matter Tractography Atlas

| Property | Detail |
|----------|--------|
| Coverage | 20 major WM tracts (probabilistic) + 48 WM regions (label) |
| Template space | **MNI152NLin6Asym (FSL MNI)** — direct match with disconnectomes |
| Resolutions | 1 mm and 2 mm |
| Access | Pre-installed with FSL at `$FSLDIR/data/atlases/JHU/`. Fallback download if FSL absent. |
| Format | 4D probabilistic NIfTI (tracts) + label NIfTI (regions), both with XML label files |
| License | Free, cite Hua et al. *NeuroImage* 2008 and Mori et al. 2005 |
| **Feasibility** | **Very easy** — most users already have it via FSL; no download needed |
| Note | The most widely used WM atlas in stroke/lesion research. Two flavours: 20 probabilistic tracts (use for weighted overlap) and 48 white matter labels (use for label NIfTI mode). |

#### Rojkova et al. — Functional White Matter Atlas

| Property | Detail |
|----------|--------|
| Coverage | 68 tracts: major association, projection, commissural, and U-fibre bundles, left/right separated |
| Template space | **MNI152NLin6Asym (FSL MNI) at 1mm** — 182×218×182, direct match with disconnectomes |
| Values | Probabilistic [0, 1] per voxel — supports weighted overlap |
| Access | Hosted by BCBlab on Dropbox — download on first use to `~/.bcblib/atlases/rojkova/` |
| Format | **Directory of individual `.nii.gz` files**, one per tract (`TractName_Left/Right.nii.gz`) |
| License | BCBlab distribution; cite Rojkova et al. *Neuropsychologia* 2016 |
| **Feasibility** | **Easy** — one-time download, auto-cached, no resampling ever needed |
| Note | Users with BCBToolKit already have the atlas locally — pass the `Tracts/` folder path directly via `AtlasSpec(source=...)`, no download needed |

#### Yeh et al. 2022 — HCP1065 population-based tractography atlas

| Property | Detail |
|----------|--------|
| Coverage | White matter tracts (number unspecified in docs, but includes major association, projection, and commissural tracts) |
| Template space | **ICBM MNI152 2009a Nonlinear Asymmetric (MNI152NLin2009cAsym / SPM MNI)** — same as Schaefer, not FSL MNI |
| Access | Manual download from [brain.labsolver.org](https://brain.labsolver.org/hcp_trk_atlas.html) — NIfTI probability volumes available alongside DSI Studio format |
| Format | Probabilistic NIfTI (one volume per tract) + Excel label files |
| License | Appears freely available; cite Yeh et al. *Nature Communications* 2022 |
| **Feasibility** | **Moderate** — NIfTI files available but no Python API; need to parse file structure and convert Excel labels |
| Space note | SPM MNI space. Same ~1–2 mm affine-resample issue as Schaefer when used with FSL MNI disconnectomes |
| Note vs HCP842 | This 2022 atlas (HCP1065, N=1065 subjects) supersedes the earlier HCP842 atlas. Probabilistic, so supports the weighted-overlap computation out of the box |

---

### Summary table

| Atlas | Regions | Coverage | Space | Auto-fetch | Space match (FSL MNI) | Feasibility |
|-------|---------|----------|-------|------------|----------------------|-------------|
| Schaefer 200/300/400 (7N) | 200–400 | Cortex | SPM MNI | `nilearn` | No (affine resample) | Very easy |
| AAL / AAL3 | 116–170 | Whole brain | SPM MNI | `nilearn` | No (affine resample) | Very easy |
| Tian S1 / S2 | 16 / 32 | Subcortex | Both available | `~/.bcblib` cache | Yes (FSL version) | Easy |
| Buckner 7N | 7 | Cerebellum | FSL MNI | `~/.bcblib` cache | Yes | Easy |
| JHU WM | 20 tracts / 48 regions | White matter | FSL MNI | FSL install / `~/.bcblib` | Yes | Very easy |
| Rojkova | 68 tracts | White matter | FSL MNI 1mm | `~/.bcblib` cache / BCBToolKit | Yes | Easy |
| Yeh 2022 (HCP1065) | 64 tracts | White matter | MNI2009cAsym cropped (157×189×136) | `~/.bcblib` cache | No (resample needed) | Moderate |
| Glasser MMP1 | 360 | Cortex | — | — | — | **Excluded** (surface atlas) |

### Practical recommendation for v1

The two nilearn-fetchable atlases (Schaefer, AAL) have the lowest barrier to entry and should
be the first supported with auto-download capability. The others require a local file path, which
is exactly what the generic `AtlasSpec` already handles — no special treatment needed.

The team's intended combined parcellation (Schaefer + Tian + Buckner as separate atlases for
cortex, subcortex, and cerebellum) is fully supported by the current design: run `damage_profile`
once per atlas, get three CSVs, merge in downstream analysis.

---

## References

Foulon C et al. Advanced lesion symptom mapping analyses and implementation as BCBtoolkit.
*GigaScience*. 2018;7(3):giy004. https://doi.org/10.1093/gigascience/giy004

Griffis JC et al. Lesion Quantification Toolkit. *NeuroImage: Clinical*. 2021;30:102639.
https://doi.org/10.1016/j.nicl.2021.102639

Ciric R et al. TemplateFlow: a community archive of imaging templates and atlases for improved
consistency in neuroimaging. *Nature Methods*. 2022;19:316–318.
https://doi.org/10.1038/s41592-022-01681-2

Schaefer A et al. Local-global parcellation of the human cerebral cortex from intrinsic
functional connectivity MRI. *Cerebral Cortex*. 2018;28(9):3095–3114.
https://doi.org/10.1093/cercor/bhx179

Tian Y et al. Topographic organization of the human subcortex unveiled with functional
connectivity gradients. *Nature Neuroscience*. 2020;23:1421–1432.
https://doi.org/10.1038/s41593-020-00711-6

Buckner RL et al. The organization of the human cerebellum estimated by intrinsic functional
connectivity. *Journal of Neurophysiology*. 2011;106(5):2322–2345.
https://doi.org/10.1152/jn.00339.2011

Glasser MF et al. A multi-modal parcellation of human cerebral cortex. *Nature*.
2016;536:171–178. https://doi.org/10.1038/nature18933

Yeh F-C et al. Population-based tract-to-region connectome of the human brain and its
hierarchical topology. *Nature Communications*. 2022;13:4933.
https://doi.org/10.1038/s41467-022-32595-4
