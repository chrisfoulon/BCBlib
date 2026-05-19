"""BIDS path utilities for the lesion_features pipeline."""

import re
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

from bcblib.tools.lesion_features._constants import LF_SUBDIR


def parse_bids_entities(path) -> Dict[str, Optional[str]]:
    """Parse BIDS entities from a filename.

    Returns
    -------
    dict with keys: sub, ses, space, res, label, desc, suffix, extension
    Missing optional entities are None.
    """
    name = Path(path).name
    entities: Dict[str, Optional[str]] = {
        "sub": None, "ses": None, "space": None, "res": None,
        "label": None, "desc": None, "suffix": None, "extension": None,
    }
    # extract key-value entities (match both leading and internal)
    for m in re.finditer(r'(?:^|_)([a-zA-Z]+)-([^_\.]+)', name):
        key, val = m.group(1), m.group(2)
        if key in entities:
            entities[key] = val
    # extract suffix and extension
    # suffix is the last _-separated part before the first dot
    parts = name.split(".")
    stem = parts[0]
    extension = "." + ".".join(parts[1:]) if len(parts) > 1 else ""
    entities["extension"] = extension if extension else None
    suffix_m = re.search(r'_([^_\-]+)$', stem)
    entities["suffix"] = suffix_m.group(1) if suffix_m else None
    return entities


def build_lf_csv_path(
    output_dir,
    sub: str,
    ses: Optional[str],
    space: str,
    feature_variant: str,
    atlas_name: str,
    lesion_desc: Optional[str] = None,
) -> Path:
    """Build output path for a lesion-features CSV file.

    Example
    -------
    sub-001_space-MNI152NLin6Asym_LF-lesion_atlas-JHU.csv
    sub-001_ses-01_space-MNI152NLin6Asym_LF-disconnectome_atlas-JHU.csv
    sub-001_space-MNI152NLin6Asym_LF-lesion-core_atlas-JHU.csv  (glioma desc-core)
    """
    ses_part = f"_ses-{ses}" if ses else ""
    variant_str = f"{feature_variant}-{lesion_desc}" if lesion_desc else feature_variant
    name = f"sub-{sub}{ses_part}_space-{space}_LF-{variant_str}_atlas-{atlas_name}.csv"
    sub_dir = Path(output_dir) / f"sub-{sub}"
    if ses:
        sub_dir = sub_dir / f"ses-{ses}"
    return sub_dir / LF_SUBDIR / name


def build_lf_tsv_path(
    output_dir,
    sub: str,
    ses: Optional[str],
    space: str,
    map_type: str,
    lesion_desc: Optional[str] = None,
) -> Path:
    """Build output path for a lesion-features mapstats TSV.

    Example
    -------
    sub-001_space-MNI152NLin6Asym_desc-lesion_mapstats.tsv
    sub-001_space-MNI152NLin6Asym_desc-core-lesion_mapstats.tsv  (glioma desc-core)
    """
    ses_part = f"_ses-{ses}" if ses else ""
    desc_val = f"{lesion_desc}-{map_type}" if lesion_desc else map_type
    name = f"sub-{sub}{ses_part}_space-{space}_desc-{desc_val}_mapstats.tsv"
    sub_dir = Path(output_dir) / f"sub-{sub}"
    if ses:
        sub_dir = sub_dir / f"ses-{ses}"
    return sub_dir / LF_SUBDIR / name


def build_prep_path(
    prep_dir,
    sub: str,
    ses: Optional[str],
    suffix: str,
    label_or_desc: str,
    lesion_desc: Optional[str] = None,
) -> Path:
    """Build output path for a preprocessed NIfTI in the prep derivatives.

    Parameters
    ----------
    suffix : str
        BIDS suffix, e.g. ``'mask'``.
    label_or_desc : str
        Either ``'label-lesion'`` or ``'desc-disconnectome'`` (full entity string).
    lesion_desc : str or None
        Value of the ``desc-`` entity from the source lesion filename (e.g.
        ``'core'``, ``'edema'``).  Included before ``label_or_desc`` when set.

    Returns
    -------
    Path
        e.g. ``prep_dir/sub-001/anat/sub-001_space-MNI152NLin6Asym_res-1_label-lesion_mask.nii.gz``
        or ``prep_dir/sub-001/anat/sub-001_space-MNI152NLin6Asym_res-1_desc-core_label-lesion_mask.nii.gz``
    """
    from bcblib.tools.lesion_features._constants import TARGET_SPACE, TARGET_RES
    ses_part = f"_ses-{ses}" if ses else ""
    desc_part = f"_desc-{lesion_desc}" if lesion_desc else ""
    name = (
        f"sub-{sub}{ses_part}"
        f"_space-{TARGET_SPACE}_res-{TARGET_RES}"
        f"{desc_part}_{label_or_desc}_{suffix}.nii.gz"
    )
    sub_dir = Path(prep_dir) / f"sub-{sub}"
    if ses:
        sub_dir = sub_dir / f"ses-{ses}"
    return sub_dir / LF_SUBDIR / name


def iter_bids_lesions(
    bids_dir,
    suffix: str = "*_label-lesion_mask.nii.gz",
    subdir: str = "anat",
) -> Iterator[Tuple[str, Optional[str], Path]]:
    """Yield ``(sub_id, ses_id_or_None, lesion_path)`` for every lesion mask.

    Parameters
    ----------
    bids_dir : str or Path
    suffix : str
        Glob pattern for lesion mask filenames.  Default matches the standard
        ``*_label-lesion_mask.nii.gz`` convention; override if your project
        uses a different naming (e.g. ``*_label-tumor_mask.nii.gz``).
    subdir : str
        Name of the per-subject subdirectory that contains the NIfTI files.
        Defaults to ``"anat"`` for standard BIDS input; use ``LF_SUBDIR``
        when iterating over a lesion-features prep/output directory.
    """
    bids_root = Path(bids_dir)
    for sub_dir in sorted(bids_root.glob("sub-*")):
        if not sub_dir.is_dir():
            continue
        sub_id = sub_dir.name[4:]  # strip "sub-"
        img_dir = sub_dir / subdir
        if img_dir.is_dir():
            for f in sorted(img_dir.glob(suffix)):
                yield sub_id, None, f
        for ses_dir in sorted(sub_dir.glob("ses-*")):
            if not ses_dir.is_dir():
                continue
            ses_id = ses_dir.name[4:]  # strip "ses-"
            img_dir = ses_dir / subdir
            if img_dir.is_dir():
                for f in sorted(img_dir.glob(suffix)):
                    yield sub_id, ses_id, f


def iter_flat_lesions(
    root_dir,
    suffix: str = "*_lesion_mask.nii.gz",
) -> Iterator[Tuple[str, Optional[str], Path]]:
    """Yield ``(sub_id, None, lesion_path)`` for non-BIDS flat directories.

    Expects lesion files directly under ``sub-*/`` with no ``anat/``
    subdirectory — the layout produced by pipelines such as StrokeBrain.

    Parameters
    ----------
    root_dir : str or Path
    suffix : str
        Glob pattern for lesion mask filenames inside each ``sub-*/``
        directory.  Default matches ``*_lesion_mask.nii.gz``.
    """
    root = Path(root_dir)
    for sub_dir in sorted(root.glob("sub-*")):
        if not sub_dir.is_dir():
            continue
        sub_id = sub_dir.name[4:]  # strip "sub-"
        for f in sorted(sub_dir.glob(suffix)):
            yield sub_id, None, f
