"""Top-level damage_profile function."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import nibabel as nib
import pandas as pd

from bcblib.imaging._types import NiftiLike
from bcblib.imaging.io import load_nifti, is_nifti
from bcblib.tools.damage_profile._atlas import AtlasSpec, load_atlas
from bcblib.tools.damage_profile._space import check_and_resample
from bcblib.tools.damage_profile._stats import compute_region_stats, compute_subject_stats


def _load_atlas_reference(spec: AtlasSpec) -> Optional[nib.Nifti1Image]:
    """Load one representative NIfTI from the atlas for affine/space information.

    Returns
    -------
    Nifti1Image or None
        None if no NIfTI file can be found.
    """
    p = Path(spec.source)
    if p.is_dir():
        niftis = sorted(p.glob("*.nii*"))
        return nib.load(str(niftis[0])) if niftis else None
    if is_nifti(str(p)):
        return nib.load(str(p))
    return None


def _resample_atlas_dict(
    atlas_dict: Dict[str, np.ndarray],
    atlas_affine: np.ndarray,
    subject_img: nib.Nifti1Image,
    spec: AtlasSpec,
    on_space_mismatch: str,
) -> Dict[str, np.ndarray]:
    """Resample every region in *atlas_dict* to the subject voxel grid.

    Returns the dict unchanged if subject and atlas are already on the same grid.

    Parameters
    ----------
    atlas_dict : dict[str, np.ndarray]
    atlas_affine : np.ndarray
        Affine of the atlas (used to build per-region NIfTI headers).
    subject_img : Nifti1Image
    spec : AtlasSpec
    on_space_mismatch : str
        Passed through to ``check_and_resample``.

    Returns
    -------
    dict[str, np.ndarray]
        Same keys as *atlas_dict*; values resampled to *subject_img* grid.
    """
    # Quick check: are they already on the same grid?
    if (
        subject_img.shape[:3] == tuple(atlas_dict[next(iter(atlas_dict))].shape[:3])
        and np.allclose(subject_img.affine, atlas_affine, atol=1e-3)
    ):
        return atlas_dict

    resampled: Dict[str, np.ndarray] = {}
    for region_name, weights in atlas_dict.items():
        region_img = nib.Nifti1Image(weights.astype(np.float32), atlas_affine)
        resampled[region_name] = check_and_resample(
            subject_img, region_img, region_name, on_space_mismatch
        )
    return resampled


def damage_profile(
    subject_map: NiftiLike,
    atlases: List[AtlasSpec],
    min_overlap_voxels: int = 1,
    on_space_mismatch: str = "error",
    output_dir: Optional[Union[str, os.PathLike]] = None,
) -> Dict[str, pd.DataFrame]:
    """Compute overlap statistics between a subject map and one or more atlases.

    Parameters
    ----------
    subject_map : NiftiLike
        Subject lesion map or disconnectome probability map.
    atlases : list of AtlasSpec
        Atlases to profile against.
    min_overlap_voxels : int
        Regions with fewer overlapping non-zero voxels are excluded.
    on_space_mismatch : str
        ``'error'`` (default) or ``'warn'``.  Passed to ``check_and_resample``
        for same-family space mismatches.
    output_dir : path-like, optional
        If provided, one CSV per atlas is written as
        ``<atlas_name>_damage_profile.csv`` and a ``subject_map_stats.csv``
        containing descriptive statistics for the subject map is also written.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keyed by ``AtlasSpec.name`` for per-region overlap results, plus the
        reserved key ``'_subject_map_stats'`` for the subject descriptive stats
        (see :func:`compute_subject_stats`).
    """
    subject_img = load_nifti(subject_map)
    subject_data = subject_img.get_fdata()

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, pd.DataFrame] = {}

    subject_stats = compute_subject_stats(subject_img)
    results["_subject_map_stats"] = subject_stats
    if output_dir is not None:
        subject_stats.to_csv(output_dir / "subject_map_stats.csv", index=False)

    for spec in atlases:
        atlas_ref = _load_atlas_reference(spec)
        atlas_dict = load_atlas(spec)

        if atlas_ref is not None and atlas_dict:
            atlas_dict = _resample_atlas_dict(
                atlas_dict, atlas_ref.affine, subject_img, spec, on_space_mismatch
            )

        df = compute_region_stats(subject_data, atlas_dict, min_overlap_voxels)
        results[spec.name] = df

        if output_dir is not None:
            csv_path = output_dir / f"{spec.name}_damage_profile.csv"
            df.to_csv(csv_path, index=False)

    return results
