"""Space normalisation for lesion images."""

import re
from pathlib import Path
from typing import Optional, Union

import nibabel as nib
import numpy as np
from nilearn.image import resample_img

from bcblib.tools.damage_profile._space import _apply_templateflow_warp
from bcblib.tools.lesion_features._constants import TARGET_RES, TARGET_SPACE

# MNI152NLin6Asym canonical shapes at 1 mm and 2 mm.
# (181, 217, 181) is the SPM12 MNI152 1 mm template — same affine as FSL's
# (182, 218, 182) but one voxel smaller FOV on each axis.  It is listed here
# for resolution detection only; normalise_lesion_to_mni6 resamples it to the
# canonical FSL grid before saving so that run_disco.sh and atlas lookups both
# see the expected (182, 218, 182) volume.
_MNI6_SHAPES = {
    1: {(182, 218, 182), (193, 229, 193), (181, 217, 181)},
    2: {(91, 109, 91), (97, 115, 97)},
}

# Canonical FSL MNI152NLin6Asym 1 mm output shape.
_MNI6_CANONICAL_SHAPE = (182, 218, 182)

_MNI6_ALIASES = {
    "MNI152NLin6Asym", "MNI152NLin6", "MNI6",
}
_MNI2009C_ALIASES = {
    "MNI152NLin2009cAsym", "MNI152NLin2009c", "MNI2009c",
}


def extract_space_from_filename(path) -> Optional[str]:
    """Return the ``space-`` entity value from a BIDS filename, or None."""
    m = re.search(r'_space-([^_\.]+)', Path(path).name)
    return m.group(1) if m else None


def extract_resolution_from_filename(path) -> Optional[int]:
    """Return the ``res-`` entity value as int from a BIDS filename, or None."""
    m = re.search(r'_res-(\d+)', Path(path).name)
    return int(m.group(1)) if m else None


def detect_resolution_from_shape(img: nib.Nifti1Image) -> int:
    """Guess voxel resolution (mm) from image shape.

    Parameters
    ----------
    img : nib.Nifti1Image

    Returns
    -------
    int
        1 or 2 (mm).

    Raises
    ------
    ValueError
        If shape is not a recognised MNI shape.
    """
    shape = img.shape[:3]
    for res, shapes in _MNI6_SHAPES.items():
        if shape in shapes:
            return res
    raise ValueError(
        f"Unrecognised image shape {shape}; cannot determine resolution."
    )


def normalise_lesion_to_mni6(
    img_or_path: Union[nib.Nifti1Image, str, Path],
    source_space: Optional[str],
    source_res: Optional[int],
) -> nib.Nifti1Image:
    """Return a lesion image resampled to MNI152NLin6Asym 1 mm.

    Parameters
    ----------
    img_or_path
        Input NIfTI image or path.
    source_space
        Template space identifier.  If None, assumed to be MNI6.
    source_res
        Voxel resolution in mm.  If None, detected from image shape.

    Returns
    -------
    nib.Nifti1Image
        In MNI152NLin6Asym 1 mm, nearest-neighbour interpolation.

    Raises
    ------
    ValueError
        For unsupported spaces.
    """
    if isinstance(img_or_path, (str, Path)):
        img = nib.load(str(img_or_path))
    else:
        img = img_or_path

    if source_res is None:
        source_res = detect_resolution_from_shape(img)

    # Normalise space alias
    in_mni6 = (source_space is None) or (source_space in _MNI6_ALIASES)
    in_mni2009c = source_space in _MNI2009C_ALIASES if source_space else False

    if in_mni6:
        if source_res == TARGET_RES:
            if img.shape[:3] == _MNI6_CANONICAL_SHAPE:
                return img
            # Non-canonical FOV (e.g. SPM12 181×217×181).  SPM and FSL MNI
            # share the same affine origin, so padding to (182, 218, 182) with
            # the same affine is exact — no interpolation error, just one extra
            # zero-filled voxel at each boundary.  Required so that run_disco.sh
            # and atlas lookups both see the expected canonical grid.
            return resample_img(
                img,
                target_affine=img.affine,
                target_shape=_MNI6_CANONICAL_SHAPE,
                interpolation="nearest",
            )
        # 2 mm → 1 mm: derive the 1 mm affine from the input's own affine by
        # normalising the voxel size to 1 mm.  This preserves the input's
        # orientation convention (radiological or neurological) exactly.
        vox_size = np.sqrt((img.affine[:3, :3] ** 2).sum(axis=0))
        target_affine = img.affine.copy()
        target_affine[:3, :3] = img.affine[:3, :3] / vox_size
        target_shape = tuple(np.round(np.array(img.shape[:3]) * vox_size).astype(int))
        return resample_img(
            img,
            target_affine=target_affine,
            target_shape=target_shape,
            interpolation="nearest",
        )

    if in_mni2009c:
        # warp to MNI6 1 mm (output inherits TemplateFlow's orientation
        # convention, which may differ from the input).
        warped = _apply_templateflow_warp(img, source_space, TARGET_SPACE)
        # restore input's orientation convention via nibabel reorientation
        in_ornt = nib.io_orientation(img.affine)
        out_ornt = nib.io_orientation(warped.affine)
        if not np.array_equal(in_ornt, out_ornt):
            ornt_xfm = nib.orientations.ornt_transform(out_ornt, in_ornt)
            warped = warped.as_reoriented(ornt_xfm)
        return warped

    raise ValueError(
        f"Unsupported source space '{source_space}'. "
        f"Supported: MNI152NLin6Asym (1 mm or 2 mm) and MNI152NLin2009cAsym."
    )


def preprocess_one(lesion_path, prep_dir, sub: str, ses: Optional[str] = None) -> Path:
    """Normalise a single lesion to MNI152NLin6Asym 1 mm and save it.

    Parameters
    ----------
    lesion_path : str or Path
    prep_dir : str or Path
    sub : str
        Subject ID without the ``sub-`` prefix.
    ses : str or None
        Session ID without the ``ses-`` prefix.

    Returns
    -------
    Path
        Output file path.
    """
    from bcblib.tools.lesion_features._bids import build_prep_path, parse_bids_entities

    lesion_path = Path(lesion_path)
    lesion_desc = parse_bids_entities(lesion_path).get("desc")
    space = extract_space_from_filename(lesion_path)
    res = extract_resolution_from_filename(lesion_path)

    img = nib.load(str(lesion_path))
    if res is None:
        try:
            res = detect_resolution_from_shape(img)
        except ValueError:
            res = None

    norm = normalise_lesion_to_mni6(img, space, res)

    out_path = build_prep_path(prep_dir, sub, ses, "mask", "label-lesion", lesion_desc=lesion_desc)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(norm, str(out_path))
    return out_path
