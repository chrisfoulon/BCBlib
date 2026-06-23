"""Optional Tract Density Index (TDI) hook.

The TDI script and atlas map are provided by EBRAINS but cannot be bundled
with BCBlib or the BCBToolKit distribution (EBRAINS patent office
restriction). Centers install them privately on PHI; if they are absent,
TDI is silently skipped.
"""

import importlib.util
import os
import tempfile
import warnings
from pathlib import Path
from typing import Callable, Optional

import nibabel as nib

from bcblib.tools.lesion_features._constants import DEFAULT_TDI_DIR

_SCRIPT_NAME = "tdi.py"
_ATLAS_NAME = "tdi_map_1mm.nii"


def find_tdi_dir(path_hint: Optional[str] = None) -> Optional[Path]:
    """Locate the private directory containing tdi.py and tdi_map_1mm.nii.

    Search order: *path_hint* -> ``TDI_DIR`` env var -> DEFAULT_TDI_DIR.
    Returns None (and warns) if neither file is found at any candidate.
    """
    candidates = [path_hint, os.environ.get("TDI_DIR"), DEFAULT_TDI_DIR]
    for c in candidates:
        if c is None:
            continue
        p = Path(c)
        if (p / _SCRIPT_NAME).exists() and (p / _ATLAS_NAME).exists():
            return p
    warnings.warn(
        "TDI script/atlas not found (looked in TDI_DIR and "
        f"{DEFAULT_TDI_DIR}); skipping Tract Density Index.",
        RuntimeWarning,
    )
    return None


def _reorient_to_match(source_img: nib.Nifti1Image, target_img: nib.Nifti1Image) -> nib.Nifti1Image:
    """Losslessly reorder/flip axes of *source_img* to match *target_img*'s orientation.

    No resampling — only valid when both images already share shape and voxel size.
    """
    transform = nib.orientations.ornt_transform(
        nib.io_orientation(source_img.affine), nib.io_orientation(target_img.affine)
    )
    data = nib.orientations.apply_orientation(source_img.get_fdata(), transform)
    affine = source_img.affine.dot(
        nib.orientations.inv_ornt_aff(transform, source_img.shape)
    )
    return nib.Nifti1Image(data, affine)


def load_tdi_function(path_hint: Optional[str] = None) -> Optional[Callable[[str], float]]:
    """Return a ``lesion_path -> tdi_value`` callable, or None if TDI is unavailable.

    The TDI atlas (provided by EBRAINS, FSL convention) and our pipeline's
    MNI152NLin6Asym lesion masks describe the same physical space but store
    voxels in opposite L/R axis order. ``tdi.py`` refuses mismatched
    orientations rather than resample, so the lesion mask is losslessly
    reoriented to match the atlas before being handed to the private script.
    """
    tdi_dir = find_tdi_dir(path_hint)
    if tdi_dir is None:
        return None
    spec = importlib.util.spec_from_file_location(
        "_bcblib_tdi_private", tdi_dir / _SCRIPT_NAME
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    atlas_path = str(tdi_dir / _ATLAS_NAME)
    atlas_img = nib.load(atlas_path)

    def _compute(lesion_path) -> float:
        lesion_img = nib.load(str(lesion_path))
        if nib.aff2axcodes(lesion_img.affine) != nib.aff2axcodes(atlas_img.affine):
            lesion_img = _reorient_to_match(lesion_img, atlas_img)
            fd, tmp_path = tempfile.mkstemp(suffix=".nii.gz")
            os.close(fd)
            try:
                nib.save(lesion_img, tmp_path)
                return module.calculate_tdi(atlas_path, tmp_path)
            finally:
                os.unlink(tmp_path)
        return module.calculate_tdi(atlas_path, str(lesion_path))

    return _compute
