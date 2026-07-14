"""Streamline ratio computation for the Yeh HCP1065 tractography atlas."""

import os
import tempfile
import warnings
from pathlib import Path
from typing import Callable, Optional

import nibabel as nib
import numpy as np

try:
    from dipy.io.streamline import load_tractogram as _load_tractogram
    from dipy.tracking.utils import target as _dipy_target
    _HAS_DIPY = True
except ImportError:
    _HAS_DIPY = False


def _compute_streamline_ratio(roi_path: str, tracts_dir: str):
    """Return a DataFrame with region_name / streamline_ratio per tract.

    Parameters
    ----------
    roi_path : str
        Path to a binary lesion mask in the same space as the TRK files.
    tracts_dir : str
        Directory containing ``.trk`` files.

    Returns
    -------
    pandas.DataFrame or None
        Columns: ``region_name``, ``streamline_ratio``.  ``None`` if no
        tractograms could be processed.
    """
    import pandas as pd

    roi = nib.load(roi_path)
    roi_data = roi.get_fdata()
    if roi_data.min() != 0 or roi_data.max() != 1:
        roi_data = (roi_data > 0).astype(np.uint8)
    affine = roi.affine

    scores = {}
    for fname in sorted(os.listdir(tracts_dir)):
        if not fname.endswith(".trk"):
            continue
        tract_path = os.path.join(tracts_dir, fname)
        try:
            sft = _load_tractogram(tract_path, reference="same", bbox_valid_check=False)
        except Exception as exc:
            warnings.warn(f"Could not load {fname}: {exc}", RuntimeWarning)
            continue
        if len(sft.streamlines) == 0:
            continue
        intersecting = list(_dipy_target(
            streamlines=sft.streamlines,
            target_mask=roi_data,
            affine=affine,
            include=True,
        ))
        scores[fname[:-4]] = len(intersecting) / len(sft.streamlines)

    if not scores:
        return None
    return pd.DataFrame({
        "region_name": list(scores.keys()),
        "streamline_ratio": list(scores.values()),
    })


def load_streamline_ratio_function(trk_dir) -> Optional[Callable]:
    """Return a ``(lesion_path -> DataFrame)`` callable, or None.

    The callable warps the binary lesion mask from MNI152NLin6Asym to
    MNI152NLin2009cAsym (the space of the Yeh HCP1065 TRK files) before
    running the streamline intersection computation.

    Parameters
    ----------
    trk_dir : str or Path
        Directory containing the HCP1065 ``.trk`` files.

    Returns
    -------
    callable or None
        Returns ``None`` (with a ``RuntimeWarning``) when dipy is not
        installed or no ``.trk`` files are found in *trk_dir*.
    """
    if not _HAS_DIPY:
        warnings.warn(
            "dipy is not installed; streamline ratio will be skipped. "
            "Install with: pip install bcblib[dipy]",
            RuntimeWarning,
        )
        return None

    trk_dir = Path(trk_dir)
    if not trk_dir.is_dir() or not any(trk_dir.glob("*.trk")):
        warnings.warn(
            f"No .trk files found in {trk_dir}; streamline ratio will be skipped.",
            RuntimeWarning,
        )
        return None

    from bcblib.tools.damage_profile._space import warp_binary_mask

    _MNI6 = "MNI152NLin6Asym"
    _MNI2009C = "MNI152NLin2009cAsym"
    _trk_dir = str(trk_dir)

    def _compute(lesion_path):
        lesion_img = nib.load(str(lesion_path))
        warped = warp_binary_mask(lesion_img, _MNI6, _MNI2009C)
        fd, tmp_path = tempfile.mkstemp(suffix=".nii.gz")
        os.close(fd)
        try:
            nib.save(warped, tmp_path)
            return _compute_streamline_ratio(tmp_path, _trk_dir)
        finally:
            os.unlink(tmp_path)

    return _compute
