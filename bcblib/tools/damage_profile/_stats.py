"""Per-region overlap statistics and subject map descriptive stats."""

from typing import Dict

import nibabel as nib
import numpy as np
import pandas as pd

from bcblib.imaging.stats import _fraction_covered_array, _weighted_region_mean_array


_EMPTY_COLUMNS = [
    "region_name", "n_voxels_region", "n_voxels_overlap",
    "fraction_covered", "mean_overlap", "weighted_mean_overlap",
    "sum_overlap", "p90_overlap", "p95_overlap",
]


def compute_region_stats_from_labels(
    subject_data: np.ndarray,
    label_array: np.ndarray,
    label_names: Dict[int, str],
    min_overlap_voxels: int = 1,
) -> pd.DataFrame:
    """Compute per-region overlap statistics from a compact integer label atlas.

    Processes one region at a time so peak memory is O(voxels), not O(N×voxels).

    Parameters
    ----------
    subject_data : np.ndarray
        3D array (subject lesion or disconnectome map).
    label_array : np.ndarray
        3D int32 array of region labels (0 = background).
    label_names : dict[int, str]
        Index → region name mapping.
    min_overlap_voxels : int
        Regions with fewer overlapping non-zero voxels are excluded.

    Returns
    -------
    pd.DataFrame
        Same columns as :func:`compute_region_stats`, sorted by ``mean_overlap``.
    """
    rows = []
    for idx, region_name in label_names.items():
        mask = (label_array == idx)
        n_total = int(mask.sum())
        if n_total == 0:
            continue

        overlap_vals = subject_data[mask]
        nonzero_mask = overlap_vals > 0
        n_nonzero = int(nonzero_mask.sum())

        if n_nonzero < min_overlap_voxels:
            continue

        nonzero_vals = overlap_vals[nonzero_mask]
        weights = mask.astype(np.float32)
        rows.append({
            "region_name": region_name,
            "n_voxels_region": n_total,
            "n_voxels_overlap": n_nonzero,
            "fraction_covered": _fraction_covered_array(subject_data, mask),
            "mean_overlap": float(subject_data[mask].mean()),
            "weighted_mean_overlap": _weighted_region_mean_array(subject_data, weights),
            "sum_overlap": float(subject_data[mask].sum()),
            "p90_overlap": float(np.percentile(nonzero_vals, 90)),
            "p95_overlap": float(np.percentile(nonzero_vals, 95)),
        })

    if not rows:
        return pd.DataFrame(columns=_EMPTY_COLUMNS)

    return (
        pd.DataFrame(rows)
        .sort_values("mean_overlap", ascending=False)
        .reset_index(drop=True)
    )


_PROB_COLUMNS = [
    "region_name", "n_voxels_region", "n_voxels_overlap",
    "fraction_covered", "mean_overlap", "weighted_mean_overlap",
    "sum_overlap", "sum_atlas_in_tract", "pwll_normalised",
    "max_atlas_prob_in_overlap", "continuous_dice",
    "p90_overlap", "p95_overlap",
]


def compute_region_stats(
    subject_data: np.ndarray,
    atlas_dict: Dict[str, np.ndarray],
    min_overlap_voxels: int = 1,
) -> pd.DataFrame:
    """Compute per-region overlap statistics between a subject map and an atlas.

    Parameters
    ----------
    subject_data : np.ndarray
        3D array (the subject lesion or disconnectome map).
    atlas_dict : dict[str, np.ndarray]
        Region name → 3D weight array mapping, as returned by ``load_atlas``.
        When atlas values are probabilities (e.g. tractography atlases),
        probabilistic metrics are also computed.
    min_overlap_voxels : int
        Regions with fewer than this many non-zero overlapping voxels are excluded.

    Returns
    -------
    pd.DataFrame
        One row per region, sorted by ``mean_overlap`` descending.
        Columns: region_name, n_voxels_region, n_voxels_overlap, fraction_covered,
        mean_overlap, weighted_mean_overlap, sum_overlap, sum_atlas_in_tract,
        pwll_normalised, max_atlas_prob_in_overlap, continuous_dice,
        p90_overlap, p95_overlap.

    Notes
    -----
    ``pwll_normalised`` (probability-weighted lesion load, normalised) is
    ``weighted_overlap / sum_atlas_in_tract``, where ``weighted_overlap`` is
    ``Σ(subject_data × atlas_weight)`` over the region — the expected fraction
    of the tract's total probability mass that is overlapped by the subject
    map, weighted by both the subject map's own magnitude and the atlas's
    per-voxel probability.  For a disconnectome map this quantifies the
    expected fraction of the tract that is functionally severed.  Bounded in
    ``[0, 1]`` when the subject map itself is bounded in ``[0, 1]`` (a binary
    lesion mask or an already-probabilistic disconnectome map — the normal
    inputs to this function); an unnormalised subject map (e.g. raw
    streamline counts) is out of contract and will not respect that bound.

    ``continuous_dice`` is ``2 × weighted_overlap / (n_voxels_overlap + sum_atlas_in_tract)``,
    a bidirectional size-normalised overlap metric that accounts for both the
    subject map extent and the atlas probability mass. Also bounded in
    ``[0, 1]`` under the same subject-map assumption, since
    ``weighted_overlap ≤ min(n_voxels_overlap, sum_atlas_in_tract)``.

    Both metrics use the *probability-weighted* overlap (``subject_data ×
    atlas_weight``), not the raw ``sum_overlap`` column — dividing by
    ``sum_atlas_in_tract`` without that weighting can exceed 1 whenever the
    atlas has non-uniform, low per-voxel probability spread over many
    voxels (the normal case for real tractography atlases).
    """
    rows = []
    for region_name, weights in atlas_dict.items():
        mask = weights > 0
        n_total = int(mask.sum())
        if n_total == 0:
            continue

        overlap_vals = subject_data[mask]
        nonzero_mask = overlap_vals > 0
        n_nonzero = int(nonzero_mask.sum())

        if n_nonzero < min_overlap_voxels:
            continue

        nonzero_vals = overlap_vals[nonzero_mask]
        sum_ov = float(subject_data[mask].sum())
        sum_atlas = float(weights[mask].sum())
        # Probability-weighted overlap: subject magnitude × atlas probability,
        # not the raw (unweighted) sum_ov — see Notes above.
        weighted_overlap = float((subject_data[mask] * weights[mask]).sum())
        pwll_norm = weighted_overlap / sum_atlas if sum_atlas > 0 else float("nan")
        denom = n_nonzero + sum_atlas
        cont_dice = (2.0 * weighted_overlap / denom) if denom > 0 else float("nan")
        subj_nonzero_mask = subject_data > 0
        weights_at_overlap = weights[subj_nonzero_mask & mask]
        max_atlas_prob = float(weights_at_overlap.max()) if weights_at_overlap.size > 0 else 0.0

        rows.append({
            "region_name": region_name,
            "n_voxels_region": n_total,
            "n_voxels_overlap": n_nonzero,
            "fraction_covered": _fraction_covered_array(subject_data, mask),
            "mean_overlap": float(subject_data[mask].mean()),
            "weighted_mean_overlap": _weighted_region_mean_array(subject_data, weights),
            "sum_overlap": sum_ov,
            "sum_atlas_in_tract": sum_atlas,
            "pwll_normalised": pwll_norm,
            "max_atlas_prob_in_overlap": max_atlas_prob,
            "continuous_dice": cont_dice,
            "p90_overlap": float(np.percentile(nonzero_vals, 90)),
            "p95_overlap": float(np.percentile(nonzero_vals, 95)),
        })

    if not rows:
        return pd.DataFrame(columns=_PROB_COLUMNS)

    return (
        pd.DataFrame(rows)
        .sort_values("mean_overlap", ascending=False)
        .reset_index(drop=True)
    )


def compute_subject_stats(subject_img: nib.Nifti1Image) -> pd.DataFrame:
    """Compute descriptive statistics for the subject map itself.

    Intended to be saved alongside the per-region overlap results so that
    users can normalise overlap metrics by lesion volume or disconnectome
    integral at analysis time.

    Parameters
    ----------
    subject_img : Nifti1Image
        The loaded subject lesion or disconnectome map.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with columns:

        - ``n_nonzero_voxels`` — number of voxels with value > 0
        - ``voxel_volume_mm3`` — volume of one voxel (from the affine)
        - ``map_volume_mm3`` — ``n_nonzero_voxels × voxel_volume_mm3``
        - ``map_sum`` — sum of all voxel values (disconnectome integral)
        - ``map_mean_nonzero`` — mean over non-zero voxels
        - ``map_min_nonzero`` — minimum non-zero value
        - ``map_max`` — maximum value
        - ``map_p50_nonzero`` — median over non-zero voxels
        - ``map_p90_nonzero`` — 90th percentile over non-zero voxels
        - ``map_p95_nonzero`` — 95th percentile over non-zero voxels
    """
    data = subject_img.get_fdata()
    voxel_volume = float(np.abs(np.linalg.det(subject_img.affine[:3, :3])))

    nonzero = data[data > 0]
    n_nonzero = int(nonzero.size)

    if n_nonzero == 0:
        return pd.DataFrame([{
            "n_nonzero_voxels": 0,
            "voxel_volume_mm3": voxel_volume,
            "map_volume_mm3": 0.0,
            "map_sum": 0.0,
            "map_mean_nonzero": float("nan"),
            "map_min_nonzero": float("nan"),
            "map_max": 0.0,
            "map_p50_nonzero": float("nan"),
            "map_p90_nonzero": float("nan"),
            "map_p95_nonzero": float("nan"),
        }])

    return pd.DataFrame([{
        "n_nonzero_voxels": n_nonzero,
        "voxel_volume_mm3": voxel_volume,
        "map_volume_mm3": n_nonzero * voxel_volume,
        "map_sum": float(data.sum()),
        "map_mean_nonzero": float(nonzero.mean()),
        "map_min_nonzero": float(nonzero.min()),
        "map_max": float(data.max()),
        "map_p50_nonzero": float(np.percentile(nonzero, 50)),
        "map_p90_nonzero": float(np.percentile(nonzero, 90)),
        "map_p95_nonzero": float(np.percentile(nonzero, 95)),
    }])
