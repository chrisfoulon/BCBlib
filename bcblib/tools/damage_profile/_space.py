"""Space handling: template detection, resampling, and cross-template warps."""

import numpy as np
import nibabel as nib
import templateflow.api as tflow
from nitransforms.io.itk import ITKCompositeH5
from nitransforms.nonlinear import DenseFieldTransform
import nitransforms.resampling as ntres

from bcblib.tools.best_overlap import _check_image_space

# Known canonical shapes for quick template family detection
# (x, y, z) voxel counts at 1mm and 2mm resolutions
_MNI6_SHAPES = {(182, 218, 182), (91, 109, 91)}
_MNI2009C_SHAPES = {(193, 229, 193), (97, 115, 97)}


def _detect_template_family(img: nib.Nifti1Image) -> str:
    """Identify the MNI template family of *img* from its shape.

    Returns
    -------
    str
        ``'MNI152NLin6Asym'``, ``'MNI152NLin2009cAsym'``, or ``'unknown'``.
    """
    shape = tuple(img.shape[:3])
    if shape in _MNI6_SHAPES:
        return "MNI152NLin6Asym"
    if shape in _MNI2009C_SHAPES:
        return "MNI152NLin2009cAsym"
    return "unknown"


def _apply_templateflow_warp(
    atlas_img: nib.Nifti1Image,
    source_space: str,
    target_space: str,
) -> nib.Nifti1Image:
    """Apply a TemplateFlow non-linear warp to resample *atlas_img*.

    Parameters
    ----------
    atlas_img : Nifti1Image
        Atlas image in *source_space*.
    source_space, target_space : str
        TemplateFlow template identifiers.

    Returns
    -------
    nibabel.Nifti1Image
        Atlas resampled into *target_space*.
    """
    warp_files = tflow.get(
        target_space,
        suffix="xfm",
        extension=".h5",
        **{"from": source_space},
    )
    if not warp_files:
        raise RuntimeError(
            f"TemplateFlow has no warp from {source_space} to {target_space}. "
            f"Check that the templateflow data cache is populated."
        )
    warp_path = warp_files if isinstance(warp_files, str) else str(warp_files)

    target_ref = tflow.get(target_space, resolution=1, desc="brain", suffix="T1w")
    if not target_ref:
        target_ref = tflow.get(target_space, resolution=1, suffix="T1w")
    ref_img = nib.load(str(target_ref))

    # TemplateFlow H5 files are ITK composite transforms (affine + displacement
    # field).  ITKCompositeH5.from_filename returns a list; element [1] is the
    # displacement field as a Nifti1Image.  We apply it via the current API.
    composite = ITKCompositeH5.from_filename(warp_path)
    disp_xfm = DenseFieldTransform(composite[1])
    resampled = ntres.apply(disp_xfm, atlas_img, reference=ref_img, order=0)
    return nib.Nifti1Image(resampled.get_fdata().astype(np.float32), ref_img.affine)


def check_and_resample(
    subject_img: nib.Nifti1Image,
    atlas_img: nib.Nifti1Image,
    atlas_name: str,
    on_space_mismatch: str = "error",
) -> np.ndarray:
    """Validate space compatibility and resample *atlas_img* to match *subject_img*.

    Three tiers:
    1. Identical affine + shape → passthrough (no copy).
    2. Same template family, different resolution → nilearn resample.
    3. Cross-template family (MNI6Asym ↔ MNI2009cAsym) → TemplateFlow warp.

    Parameters
    ----------
    subject_img : Nifti1Image
    atlas_img : Nifti1Image
    atlas_name : str
        Used in error/warning messages.
    on_space_mismatch : str
        ``'error'`` (default) or ``'warn'``.  Controls behaviour for
        same-family affine mismatches (tier 2); tier 3 always resamples.

    Returns
    -------
    np.ndarray
        Atlas data resampled to subject voxel grid.

    Raises
    ------
    ValueError
        If shapes are incompatible and cannot be resampled, or if
        ``on_space_mismatch='error'`` and a mismatch is detected.
    """
    # Tier 1: identical grid
    if (
        subject_img.shape[:3] == atlas_img.shape[:3]
        and np.allclose(subject_img.affine, atlas_img.affine, atol=1e-3)
    ):
        return atlas_img.get_fdata()

    # Tier 3: cross-template family
    subj_family = _detect_template_family(subject_img)
    atlas_family = _detect_template_family(atlas_img)
    cross_template_pairs = {
        ("MNI152NLin6Asym", "MNI152NLin2009cAsym"),
        ("MNI152NLin2009cAsym", "MNI152NLin6Asym"),
    }
    if (subj_family, atlas_family) in cross_template_pairs:
        warped = _apply_templateflow_warp(atlas_img, atlas_family, subj_family)
        return warped.get_fdata()

    # Tier 2: same family (or unknown), different resolution — use nilearn
    info = _check_image_space(
        subject_img, "subject_map",
        atlas_img, atlas_name,
        on_mismatch=on_space_mismatch,
    )
    if info.get("issues"):
        if on_space_mismatch == "error":
            raise ValueError(
                f"Space mismatch between subject and atlas '{atlas_name}': "
                f"{info['issues']}"
            )
        import warnings
        warnings.warn(
            f"Space mismatch for atlas '{atlas_name}'; resampling with nilearn. "
            f"Issues: {info['issues']}"
        )

    from nilearn.image import resample_to_img as _resample_to_img
    resampled = _resample_to_img(atlas_img, subject_img, interpolation="nearest")
    return resampled.get_fdata()
