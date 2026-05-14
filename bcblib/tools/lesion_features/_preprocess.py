"""Space normalisation for lesion images."""

import re
from pathlib import Path
from typing import Optional, Union

import nibabel as nib
from nilearn.image import resample_to_img

from bcblib.tools.damage_profile._space import _apply_templateflow_warp
from bcblib.tools.lesion_features._constants import TARGET_RES, TARGET_SPACE

# MNI152NLin6Asym canonical shapes at 1 mm and 2 mm
_MNI6_SHAPES = {
    1: {(182, 218, 182), (193, 229, 193)},
    2: {(91, 109, 91), (97, 115, 97)},
}
# MNI152NLin2009cAsym canonical shapes
_MNI2009C_SHAPES = {
    1: {(182, 218, 182), (193, 229, 193)},
    2: {(91, 109, 91), (97, 115, 97)},
}

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
            return img
        # 2 mm → 1 mm nearest-neighbour resample
        import templateflow.api as tflow
        ref_paths = tflow.get(
            TARGET_SPACE, resolution=TARGET_RES, suffix="T1w", desc=None, extension=".nii.gz"
        )
        ref_paths = [ref_paths] if not isinstance(ref_paths, list) else ref_paths
        ref_img = nib.load(str(ref_paths[0]))
        return resample_to_img(img, ref_img, interpolation="nearest")

    if in_mni2009c:
        # warp to MNI6, then resample to 1 mm if needed
        warped = _apply_templateflow_warp(img, source_space, TARGET_SPACE)
        if source_res == TARGET_RES:
            return warped
        import templateflow.api as tflow
        ref_paths = tflow.get(
            TARGET_SPACE, resolution=TARGET_RES, suffix="T1w", desc=None, extension=".nii.gz"
        )
        ref_paths = [ref_paths] if not isinstance(ref_paths, list) else ref_paths
        ref_img = nib.load(str(ref_paths[0]))
        return resample_to_img(warped, ref_img, interpolation="nearest")

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
    from bcblib.tools.lesion_features._bids import build_prep_path

    lesion_path = Path(lesion_path)
    space = extract_space_from_filename(lesion_path)
    res = extract_resolution_from_filename(lesion_path)

    img = nib.load(str(lesion_path))
    if res is None:
        try:
            res = detect_resolution_from_shape(img)
        except ValueError:
            res = None

    norm = normalise_lesion_to_mni6(img, space, res)

    out_path = build_prep_path(prep_dir, sub, ses, "mask", "label-lesion")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(norm, str(out_path))
    return out_path
