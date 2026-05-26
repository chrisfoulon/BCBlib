"""Orchestration pipelines for preprocessing and feature extraction."""

import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

from bcblib.tools.lesion_features._constants import EBRAINS_ATLAS_SPECS, LF_SUBDIR, TARGET_SPACE
from bcblib.tools.lesion_features._bids import (
    build_lf_csv_path,
    build_lf_tsv_path,
    iter_bids_lesions,
    iter_flat_lesions,
    parse_bids_entities,
)
from bcblib.tools.lesion_features._preprocess import preprocess_one


def get_ebrains_atlas_specs(assume_yes: bool = False):
    """Return AtlasSpec objects for the EBRAINS default atlas set.

    Parameters
    ----------
    assume_yes : bool
        Skip download consent prompts.

    Returns
    -------
    list[AtlasSpec]
    """
    from bcblib.tools.damage_profile._atlas_manager import get_preset_atlas
    from bcblib.tools.damage_profile import AtlasSpec
    from bcblib.tools.damage_profile._atlas_manager import get_atlas_dir, PRESET_ATLASES

    specs = []
    for key in EBRAINS_ATLAS_SPECS:
        try:
            get_preset_atlas(key, assume_yes=assume_yes)
        except RuntimeError:
            warnings.warn(f"Skipping atlas '{key}': download declined.", RuntimeWarning)
            continue
        info = PRESET_ATLASES[key]
        cache = get_atlas_dir() / key
        label_file = None
        if info.label_file and (cache / info.label_file).exists():
            label_file = str(cache / info.label_file)
        specs.append(AtlasSpec(
            source=str(_preset_cache_path(key)),
            name=key,
            space=TARGET_SPACE,
            label_file=label_file,
        ))
    return specs


def _preset_cache_path(key: str) -> Path:
    """Return the cache directory for a preset atlas key."""
    from bcblib.tools.damage_profile._atlas_manager import get_atlas_dir, PRESET_ATLASES
    info = PRESET_ATLASES[key]
    cache = get_atlas_dir() / key
    if info.nifti_path:
        return cache / info.nifti_path
    return cache


def preprocess_batch(
    bids_dir,
    prep_dir,
    force: bool = False,
    suffix: str = "*_label-lesion_mask.nii.gz",
    flat: bool = False,
) -> Dict[str, Path]:
    """Normalise all lesions to MNI152NLin6Asym 1 mm.

    Parameters
    ----------
    bids_dir : str or Path
        Input directory.  With ``flat=False`` (default) this must follow BIDS
        conventions (``sub-*/anat/``).  With ``flat=True`` lesion files are
        expected directly under ``sub-*/`` with no ``anat/`` subdirectory.
    prep_dir : str or Path
        Output BIDS derivatives directory.
    force : bool
        Reprocess even if output already exists.
    suffix : str
        Glob pattern for lesion mask filenames.
    flat : bool
        When ``True``, use :func:`iter_flat_lesions` instead of
        :func:`iter_bids_lesions`.  Suited for pipeline outputs such as
        StrokeBrain that place files directly under ``sub-*/``.

    Returns
    -------
    dict[str, Path]
        ``sub_id[_desc-X]`` → normalised lesion path.
    """
    from bcblib.tools.lesion_features._bids import build_prep_path

    iterator = (
        iter_flat_lesions(bids_dir, suffix=suffix)
        if flat
        else iter_bids_lesions(bids_dir, suffix=suffix)
    )

    subjects = list(iterator)
    n_total = len(subjects)

    results: Dict[str, Path] = {}
    for i, (sub_id, ses_id, lesion_path) in enumerate(subjects, 1):
        ses_part = f"_ses-{ses_id}" if ses_id else ""
        label = f"sub-{sub_id}{ses_part}"
        lesion_desc = parse_bids_entities(lesion_path).get("desc")
        if lesion_desc:
            label += f" desc-{lesion_desc}"

        out_path = build_prep_path(
            prep_dir, sub_id, ses_id, "mask", "label-lesion", lesion_desc=lesion_desc
        )
        key = sub_id + (f"_desc-{lesion_desc}" if lesion_desc else "")

        if out_path.exists() and not force:
            print(f"  [{i}/{n_total}] {label} — skipped (already exists)")
            results[key] = out_path
            continue

        print(f"  [{i}/{n_total}] {label} ...", end="", flush=True)
        t0 = time.perf_counter()
        out_path = preprocess_one(lesion_path, prep_dir, sub_id, ses_id)
        elapsed = time.perf_counter() - t0
        print(f" done ({elapsed:.1f}s)")
        results[key] = out_path

    return results


def extract_features_one(
    sub_id: str,
    ses_id: Optional[str],
    lesion_path,
    disco_path,
    atlases,
    output_dir,
    lesion_desc: Optional[str] = None,
) -> Dict[str, Path]:
    """Extract lesion and disconnectome features for one subject.

    Parameters
    ----------
    atlases : list[AtlasSpec]
    output_dir : str or Path
    lesion_desc : str or None
        Value of the ``desc-`` entity from the lesion filename (e.g. ``'core'``,
        ``'edema'``).  Propagated into output filenames to distinguish multiple
        lesion types per subject (glioma).

    Returns
    -------
    dict mapping output file descriptions to paths.
    """
    from bcblib.tools.damage_profile import damage_profile

    output_dir = Path(output_dir)
    written: Dict[str, Path] = {}

    for variant, map_path in (("lesion", lesion_path), ("disconnectome", disco_path)):
        if map_path is None:
            continue
        dp_results = damage_profile(map_path, atlases)

        for atlas_spec in atlases:
            df = dp_results.get(atlas_spec.name)
            if df is None:
                continue
            csv_path = build_lf_csv_path(
                output_dir, sub_id, ses_id, TARGET_SPACE,
                variant, atlas_spec.name, lesion_desc=lesion_desc,
            )
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(str(csv_path), index=False)
            written[f"{variant}_{atlas_spec.name}_csv"] = csv_path

        stats_df = dp_results.get("_subject_map_stats")
        if stats_df is not None:
            tsv_path = build_lf_tsv_path(
                output_dir, sub_id, ses_id, TARGET_SPACE, variant,
                lesion_desc=lesion_desc,
            )
            tsv_path.parent.mkdir(parents=True, exist_ok=True)
            stats_df.to_csv(str(tsv_path), sep="\t", index=False)
            written[f"{variant}_mapstats_tsv"] = tsv_path

    return written


def extract_features_batch(
    prep_dir,
    atlases: List,
    output_dir,
    force: bool = False,
) -> Dict[str, Dict]:
    """Extract features for all subjects in a prep derivatives directory.

    Parameters
    ----------
    prep_dir : str or Path
    atlases : list[AtlasSpec]
    output_dir : str or Path
    force : bool

    Returns
    -------
    dict[str, dict]
        sub_id → dict of written file paths.
    """
    prep_dir = Path(prep_dir)

    sub_dirs = sorted(d for d in prep_dir.glob("sub-*") if d.is_dir())
    n_total = len(sub_dirs)
    results: Dict[str, Dict] = {}

    for i, sub_dir in enumerate(sub_dirs, 1):
        sub_id = sub_dir.name[4:]
        print(f"  [{i}/{n_total}] sub-{sub_id} ...", end="", flush=True)
        t0 = time.perf_counter()
        before = len(results)
        _process_sub_dirs(sub_id, sub_dir, atlases, output_dir, results, force)
        elapsed = time.perf_counter() - t0
        n_written = len(results) - before
        if n_written:
            print(f" done ({elapsed:.1f}s, {n_written} file(s))")
        else:
            print(f" skipped ({elapsed:.1f}s)")

    return results


def _resolve_lesion_dir(parent_dir: Path) -> Optional[Path]:
    """Return the lesion NIfTI directory inside *parent_dir*, or None.

    Tries LF_SUBDIR ("lesion") first, then falls back to "anat" for prep
    directories written by older versions of BCBlib.
    """
    preferred = parent_dir / LF_SUBDIR
    if preferred.is_dir():
        return preferred
    legacy = parent_dir / "anat"
    if legacy.is_dir():
        warnings.warn(
            f"Prep directory uses legacy 'anat/' layout: {parent_dir}. "
            "Re-running bcb-lf-preprocess will migrate it to 'lesion/'.",
            UserWarning,
            stacklevel=4,
        )
        return legacy
    return None


def _process_sub_dirs(sub_id, sub_dir, atlases, output_dir, results, force):
    """Process all session or sessionless lesion directories for one subject."""
    lesion_dir = _resolve_lesion_dir(sub_dir)
    if lesion_dir is not None:
        _process_anat(sub_id, None, lesion_dir, atlases, output_dir, results, force)
    for ses_dir in sorted(sub_dir.glob("ses-*")):
        if not ses_dir.is_dir():
            continue
        ses_id = ses_dir.name[4:]
        lesion_dir = _resolve_lesion_dir(ses_dir)
        if lesion_dir is not None:
            _process_anat(sub_id, ses_id, lesion_dir, atlases, output_dir, results, force)


def _process_anat(sub_id, ses_id, anat_dir, atlases, output_dir, results, force):
    """Locate all lesion + disconnectome pairs and run extract_features_one for each."""
    lesion_files = sorted(anat_dir.glob("*_label-lesion_mask.nii.gz"))
    if not lesion_files:
        return

    for lesion_path in lesion_files:
        lesion_desc = parse_bids_entities(lesion_path).get("desc")

        # find the matching disconnectome: desc-X-disconnectome for glioma, else desc-disconnectome
        if lesion_desc:
            disco_pattern = f"*_desc-{lesion_desc}-disconnectome.nii.gz"
        else:
            disco_pattern = "*_desc-disconnectome.nii.gz"
        disco_files = sorted(anat_dir.glob(disco_pattern))

        if not disco_files:
            warnings.warn(
                f"No disconnectome found for sub-{sub_id}"
                + (f" desc-{lesion_desc}" if lesion_desc else "")
                + f" in {anat_dir}; skipping.",
                RuntimeWarning,
                stacklevel=3,
            )
            continue
        disco_path = disco_files[0]

        key = (
            f"{sub_id}"
            + (f"_ses-{ses_id}" if ses_id else "")
            + (f"_desc-{lesion_desc}" if lesion_desc else "")
        )
        written = extract_features_one(
            sub_id, ses_id, lesion_path, disco_path, atlases, output_dir,
            lesion_desc=lesion_desc,
        )
        results[key] = written
