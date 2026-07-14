"""Streamline ratio computation for the Yeh HCP1065 tractography atlas."""

import os
import tempfile
import warnings
from pathlib import Path
from typing import Callable, Optional

import nibabel as nib
import numpy as np

try:
    from dipy.tracking.utils import target as _dipy_target
    _HAS_DIPY = True
except ImportError:
    _HAS_DIPY = False

# Yeh HCP1065 TRK files use abbreviated names (AF_L, CST_L …) that differ
# from the probability atlas NIfTI names (Arcuate_Fasciculus_L …).  Streamline
# ratios are therefore written to a *separate* CSV rather than merged into the
# atlas overlap CSV.  The TRK zip organises files into category subdirectories
# (association/, cerebellum/, …) and uses the .trk.gz extension.


def _iter_trk_files(tracts_dir: Path):
    """Yield (region_name, path) for every .trk.gz / .trk file under tracts_dir."""
    for path in sorted(tracts_dir.rglob("*.trk.gz")):
        yield path.name[:-7], path
    for path in sorted(tracts_dir.rglob("*.trk")):
        if not path.name.endswith(".trk.gz"):
            yield path.name[:-4], path


def _compute_streamline_ratio(roi_path: str, tracts_dir: str):
    """Return a DataFrame with tract / streamline_ratio per tractogram.

    Parameters
    ----------
    roi_path : str
        Path to a binary lesion mask in the same space as the TRK files.
    tracts_dir : str
        Root directory containing TRK (or TRK.gz) files, possibly in
        category subdirectories.

    Returns
    -------
    pandas.DataFrame or None
        Columns: ``tract``, ``streamline_ratio``.  ``None`` if no tractograms
        could be processed.
    """
    import pandas as pd

    roi = nib.load(roi_path)
    roi_data = roi.get_fdata()
    if roi_data.min() != 0 or roi_data.max() != 1:
        roi_data = (roi_data > 0).astype(np.uint8)
    affine = roi.affine

    scores = {}
    for region, path in _iter_trk_files(Path(tracts_dir)):
        try:
            # nibabel natively supports .trk.gz via gzip; dipy load_tractogram does not.
            tfile = nib.streamlines.load(str(path), lazy_load=False)
            streamlines = tfile.streamlines
        except Exception as exc:
            warnings.warn(f"Could not load {path.name}: {exc}", RuntimeWarning)
            continue
        if len(streamlines) == 0:
            continue
        intersecting = list(_dipy_target(
            streamlines=streamlines,
            target_mask=roi_data,
            affine=affine,
            include=True,
        ))
        scores[region] = len(intersecting) / len(streamlines)

    if not scores:
        return None
    return pd.DataFrame({
        "tract": list(scores.keys()),
        "streamline_ratio": list(scores.values()),
    })


def load_streamline_ratio_function(trk_dir) -> Optional[Callable]:
    """Return a ``(lesion_path -> DataFrame)`` callable, or None.

    The callable warps the binary lesion mask from MNI152NLin6Asym to
    MNI152NLin2009cAsym (the space of the Yeh HCP1065 TRK files) before
    computing streamline intersections.

    The returned DataFrame has columns ``tract`` (abbreviated HCP1065 name,
    e.g. ``AF_L``) and ``streamline_ratio``.  It is written to a *separate*
    CSV from the probability atlas overlap results because the two datasets
    use different naming conventions.

    Parameters
    ----------
    trk_dir : str or Path
        Root directory containing the HCP1065 ``.trk.gz`` files (possibly in
        category subdirectories).

    Returns
    -------
    callable or None
        Returns ``None`` (with a ``RuntimeWarning``) when dipy is not installed
        or no TRK files are found under *trk_dir*.
    """
    if not _HAS_DIPY:
        warnings.warn(
            "dipy is not installed; streamline ratio will be skipped. "
            "Install with: pip install bcblib[dipy]",
            RuntimeWarning,
        )
        return None

    trk_dir = Path(trk_dir)
    has_trks = any(trk_dir.rglob("*.trk.gz")) or any(trk_dir.rglob("*.trk"))
    if not trk_dir.is_dir() or not has_trks:
        warnings.warn(
            f"No .trk or .trk.gz files found under {trk_dir}; "
            "streamline ratio will be skipped.",
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
