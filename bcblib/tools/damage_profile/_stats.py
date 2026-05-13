"""Per-region overlap statistics."""

from typing import Dict

import numpy as np
import pandas as pd

from bcblib.imaging.stats import _fraction_covered_array, _weighted_region_mean_array


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
    min_overlap_voxels : int
        Regions with fewer than this many non-zero overlapping voxels are excluded.

    Returns
    -------
    pd.DataFrame
        One row per region, sorted by ``mean_overlap`` descending.
        Columns: region_name, n_voxels_region, n_voxels_overlap, fraction_covered,
        mean_overlap, weighted_mean_overlap, sum_overlap, p90_overlap, p95_overlap.
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
        rows.append({
            "region_name": region_name,
            "n_voxels_region": n_total,
            "n_voxels_overlap": n_nonzero,
            "fraction_covered": _fraction_covered_array(subject_data, mask),
            "mean_overlap": float(subject_data[mask].mean()),
            "weighted_mean_overlap": _weighted_region_mean_array(
                subject_data, weights
            ),
            "sum_overlap": float(subject_data[mask].sum()),
            "p90_overlap": float(np.percentile(nonzero_vals, 90)),
            "p95_overlap": float(np.percentile(nonzero_vals, 95)),
        })

    if not rows:
        return pd.DataFrame(columns=[
            "region_name", "n_voxels_region", "n_voxels_overlap",
            "fraction_covered", "mean_overlap", "weighted_mean_overlap",
            "sum_overlap", "p90_overlap", "p95_overlap",
        ])

    return (
        pd.DataFrame(rows)
        .sort_values("mean_overlap", ascending=False)
        .reset_index(drop=True)
    )
