"""bcblib.imaging -- NIfTI image utilities organised by domain.

This subpackage provides FSL-equivalent NIfTI utilities for header
inspection, orientation, statistics, manipulation, format conversion,
and voxel-wise maths.

Public API is re-exported here so that ``from bcblib.imaging import load_nifti``
works.  The canonical per-module imports (e.g.
``from bcblib.imaging.io import load_nifti``) also work and are preferred
in library code.
"""

# -- io -------------------------------------------------------------------
from bcblib.imaging.io import is_nifti, load_nifti, resave_nifti, resave_nifti_list

# -- info -----------------------------------------------------------------
from bcblib.imaging.info import header_summary, header_dump, header_field

# -- orient ---------------------------------------------------------------
from bcblib.imaging.orient import (
    get_orientation,
    reorient_to_standard,
    reorient_list,
    set_orientation,
    swap_dimensions,
)

# -- stats ----------------------------------------------------------------
from bcblib.imaging.stats import (
    centre_of_gravity,
    centre_of_gravity_distance,
    volume_count,
    laterality_ratio,
    reduce_axis,
    image_stats,
    percentile,
    robust_range,
    histogram,
    fraction_covered,
    weighted_region_mean,
)

# -- manipulate -----------------------------------------------------------
from bcblib.imaging.manipulate import (
    extract_roi,
    merge_images,
    split_image,
    copy_geometry,
)

# -- convert --------------------------------------------------------------
from bcblib.imaging.convert import convert_format, create_image

# -- math -----------------------------------------------------------------
from bcblib.imaging.math import (
    binarize,
    threshold,
    dilate,
    erode,
    apply_mask,
    apply_inverse_mask,
    add,
    subtract,
    multiply,
)

__all__ = [
    # io
    "is_nifti",
    "load_nifti",
    "resave_nifti",
    "resave_nifti_list",
    # info
    "header_summary",
    "header_dump",
    "header_field",
    # orient
    "get_orientation",
    "reorient_to_standard",
    "reorient_list",
    "set_orientation",
    "swap_dimensions",
    # stats
    "centre_of_gravity",
    "centre_of_gravity_distance",
    "volume_count",
    "laterality_ratio",
    "reduce_axis",
    "image_stats",
    "percentile",
    "robust_range",
    "histogram",
    "fraction_covered",
    "weighted_region_mean",
    # manipulate
    "extract_roi",
    "merge_images",
    "split_image",
    "copy_geometry",
    # convert
    "convert_format",
    "create_image",
    # math
    "binarize",
    "threshold",
    "dilate",
    "erode",
    "apply_mask",
    "apply_inverse_mask",
    "add",
    "subtract",
    "multiply",
]
