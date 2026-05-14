"""NIfTI statistical utilities: summary stats, centre of gravity, histograms."""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import nibabel as nib
from scipy.spatial.distance import euclidean
from scipy.ndimage import center_of_mass

from bcblib.imaging._types import NiftiLike
from bcblib.imaging.io import load_nifti


# ---------------------------------------------------------------------------
# Centre of gravity (moved from nifti_utils)
# ---------------------------------------------------------------------------

def centre_of_gravity(img: NiftiLike, round_coord: bool = False):
    """Centre of mass of non-zero voxels in world-index coordinates.

    Parameters
    ----------
    img : NiftiLike
    round_coord : bool
        Round result to nearest integer voxel.

    Returns
    -------
    tuple
    """
    nii = load_nifti(img)
    data = nii.get_fdata()
    if not data.any():
        return tuple(np.zeros(len(nii.shape)))
    abs_data = np.nan_to_num(np.abs(data))
    com = center_of_mass(abs_data)
    if round_coord:
        return np.round(com)
    return com


def centre_of_gravity_distance(
    img: NiftiLike,
    reference,
    round_centres: bool = False,
) -> float:
    """Euclidean distance between two images' centres of gravity.

    Parameters
    ----------
    img : NiftiLike
    reference : NiftiLike or tuple/list of coordinates
    round_centres : bool

    Returns
    -------
    float
    """
    if not isinstance(reference, (list, tuple, set)):
        reference = centre_of_gravity(reference, round_centres)
    cog = centre_of_gravity(img, round_centres)
    if len(cog) != len(reference):
        raise ValueError(
            f"Image ({cog}) and reference ({reference}) must have the same dimensionality"
        )
    return euclidean(cog, reference)


# ---------------------------------------------------------------------------
# Volume / voxel counting (moved from nifti_utils)
# ---------------------------------------------------------------------------

def volume_count(
    img: NiftiLike,
    ratio: bool = False,
    threshold: float = 0,
) -> Union[int, float]:
    """Count non-zero voxels above *threshold*.

    Parameters
    ----------
    img : NiftiLike
    ratio : bool
        If True, return the fraction rather than the count.
    threshold : float

    Returns
    -------
    int or float
    """
    data = load_nifti(img).get_fdata()
    count = int(np.count_nonzero(data > threshold))
    if ratio:
        return count / np.prod(data.shape)
    return count


# ---------------------------------------------------------------------------
# Laterality (moved from nifti_utils)
# ---------------------------------------------------------------------------

def laterality_ratio(img: NiftiLike) -> float:
    """Compute left-minus-right laterality ratio of non-zero voxels.

    Parameters
    ----------
    img : NiftiLike

    Returns
    -------
    float
        Positive = left-dominant, negative = right-dominant.
    """
    nii = load_nifti(img)
    ori = nib.aff2axcodes(nii.affine)
    data = nii.get_fdata()
    mid_x = int(data.shape[0] / 2)
    coord = np.where(data)
    if ((ori[0] == "L" and ori[1] == "A") or (ori[0] == "R" and ori[1] == "P")) and ori[2] == "S":
        right_les = len(coord[0][coord[0] <= mid_x])
        left_les = len(coord[0][coord[0] > mid_x])
    else:
        right_les = len(coord[0][coord[0] > mid_x])
        left_les = len(coord[0][coord[0] <= mid_x])
    total = len(coord[0])
    return (left_les / total) - (right_les / total)


# ---------------------------------------------------------------------------
# reduce_axis (modernised from nii_stats.simple_stats)
# ---------------------------------------------------------------------------

def reduce_axis(
    img: NiftiLike,
    method: str = "mean",
    axis: int = 3,
) -> nib.Nifti1Image:
    """Apply a NumPy reduction along *axis* (e.g. temporal mean of a 4-D image).

    Parameters
    ----------
    img : NiftiLike
    method : str
        Any NumPy function name accepting ``(array, axis)``
        — e.g. ``'mean'``, ``'median'``, ``'std'``.
    axis : int
        Axis to reduce (default 3, the time dimension).

    Returns
    -------
    nibabel.Nifti1Image
    """
    nii = load_nifti(img)
    data = nii.get_fdata()
    out_data = getattr(np, method)(data, axis)
    return nib.Nifti1Image(out_data, nii.affine)


# ---------------------------------------------------------------------------
# New helper functions
# ---------------------------------------------------------------------------

def image_stats(
    img: NiftiLike,
    mask: Optional[NiftiLike] = None,
) -> Dict[str, Any]:
    """Compute summary statistics for an image (like ``fslstats``).

    Parameters
    ----------
    img : NiftiLike
    mask : NiftiLike, optional
        If given, stats are computed only within the mask.

    Returns
    -------
    dict
        Keys: ``min``, ``max``, ``mean``, ``std``, ``median``,
        ``robust_min``, ``robust_max``, ``nonzero_voxels``,
        ``total_voxels``, ``volume_mm3``.
    """
    nii = load_nifti(img)
    data = nii.get_fdata()
    if mask is not None:
        mask_data = load_nifti(mask).get_fdata().astype(bool)
        data = data[mask_data]
    else:
        data = data.ravel()

    voxel_vol = float(np.abs(np.linalg.det(nii.affine[:3, :3])))

    return {
        "min": float(np.nanmin(data)),
        "max": float(np.nanmax(data)),
        "mean": float(np.nanmean(data)),
        "std": float(np.nanstd(data)),
        "median": float(np.nanmedian(data)),
        "robust_min": float(np.nanpercentile(data, 2)),
        "robust_max": float(np.nanpercentile(data, 98)),
        "nonzero_voxels": int(np.count_nonzero(data)),
        "total_voxels": int(data.size),
        "volume_mm3": float(np.count_nonzero(data) * voxel_vol),
    }


def percentile(
    img: NiftiLike,
    q: float,
    mask: Optional[NiftiLike] = None,
) -> float:
    """Return the *q*-th percentile of voxel intensities.

    Parameters
    ----------
    img : NiftiLike
    q : float
        Percentile in ``[0, 100]``.
    mask : NiftiLike, optional
    """
    data = load_nifti(img).get_fdata()
    if mask is not None:
        data = data[load_nifti(mask).get_fdata().astype(bool)]
    return float(np.nanpercentile(data, q))


def robust_range(
    img: NiftiLike,
    mask: Optional[NiftiLike] = None,
    low: float = 2.0,
    high: float = 98.0,
) -> Tuple[float, float]:
    """Return the robust intensity range (default 2nd–98th percentile).

    Parameters
    ----------
    img : NiftiLike
    mask : NiftiLike, optional
    low, high : float
    """
    data = load_nifti(img).get_fdata()
    if mask is not None:
        data = data[load_nifti(mask).get_fdata().astype(bool)]
    return (
        float(np.nanpercentile(data, low)),
        float(np.nanpercentile(data, high)),
    )


def histogram(
    img: NiftiLike,
    bins: int = 100,
    mask: Optional[NiftiLike] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a histogram of voxel intensities.

    Parameters
    ----------
    img : NiftiLike
    bins : int
    mask : NiftiLike, optional

    Returns
    -------
    (counts, bin_edges) : tuple of arrays
    """
    data = load_nifti(img).get_fdata()
    if mask is not None:
        data = data[load_nifti(mask).get_fdata().astype(bool)]
    else:
        data = data.ravel()
    data = data[~np.isnan(data)]
    return np.histogram(data, bins=bins)


# ---------------------------------------------------------------------------
# Layered overlap stats (array cores + NiftiLike wrappers)
# ---------------------------------------------------------------------------

def _fraction_covered_array(
    subject_data: np.ndarray,
    mask_data: np.ndarray,
) -> float:
    """Fraction of *mask_data* voxels that are non-zero in *subject_data*.

    Parameters
    ----------
    subject_data : np.ndarray
    mask_data : np.ndarray
        Boolean or float array; non-zero elements define the region.

    Returns
    -------
    float
        0.0 if the mask is empty.
    """
    mask = mask_data.astype(bool)
    n_total = int(mask.sum())
    if n_total == 0:
        return 0.0
    return float((subject_data[mask] > 0).sum()) / n_total


def fraction_covered(
    subject: NiftiLike,
    mask: NiftiLike,
) -> float:
    """Fraction of *mask* voxels that are non-zero in *subject*.

    Parameters
    ----------
    subject : NiftiLike
    mask : NiftiLike

    Returns
    -------
    float
    """
    return _fraction_covered_array(
        load_nifti(subject).get_fdata(),
        load_nifti(mask).get_fdata(),
    )


def _weighted_region_mean_array(
    subject_data: np.ndarray,
    weight_data: np.ndarray,
) -> float:
    """Weighted mean of *subject_data* values within *weight_data* > 0.

    ``sum(subject[v] * weight[v]) / sum(weight[v])`` for all voxels v
    where weight[v] > 0.

    Parameters
    ----------
    subject_data : np.ndarray
    weight_data : np.ndarray
        Non-negative weights; uniform weights give the same result as the
        unweighted mean.

    Returns
    -------
    float
        ``nan`` when the total weight is zero (empty region or all-zero weights).
    """
    mask = weight_data > 0
    w = weight_data[mask]
    w_sum = float(w.sum())
    if w_sum == 0.0:
        return float("nan")
    return float((subject_data[mask] * w).sum() / w_sum)


def weighted_region_mean(
    subject: NiftiLike,
    weight_map: NiftiLike,
) -> float:
    """Weighted mean of *subject* intensities using *weight_map* as weights.

    Parameters
    ----------
    subject : NiftiLike
    weight_map : NiftiLike

    Returns
    -------
    float
        ``nan`` when total weight is zero.
    """
    return _weighted_region_mean_array(
        load_nifti(subject).get_fdata(),
        load_nifti(weight_map).get_fdata(),
    )
