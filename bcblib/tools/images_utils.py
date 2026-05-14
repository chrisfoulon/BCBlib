import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion

# Private import for functions that STAY in this module
from bcblib.imaging.io import load_nifti as _load_nifti


def get_mask_perimeter(array, binarise_thr=0.5):
    """
    Returns a binary mask of the input mask's perimeter with a thickness of one voxel
    Parameters
    ----------
    array
    binarise_thr

    Returns
    -------

    """
    bin_array = np.where(array >= binarise_thr, 1, 0)
    erode_array = np.copy(bin_array)
    erode_array = binary_erosion(erode_array)
    perimeter_array = bin_array - erode_array
    return perimeter_array


# =========================================================================
# Deprecation shims for functions moved to bcblib.imaging.math
# =========================================================================

def dilate_mask(array, connectivity, dimensions=3):
    """Deprecated. Use ``bcblib.imaging.math._dilate_array`` (array-level) or
    ``bcblib.imaging.math.dilate`` (NIfTI-level) instead."""
    warnings.warn(
        "bcblib.tools.images_utils.dilate_mask is deprecated. "
        "Use bcblib.imaging.math.dilate (NIfTI) or "
        "bcblib.imaging.math._dilate_array (array) instead.",
        DeprecationWarning, stacklevel=2,
    )
    from bcblib.imaging.math import _dilate_array
    return _dilate_array(array, connectivity, dimensions)


def mask_in_array(array, mask):
    """Deprecated. Use ``bcblib.imaging.math._mask_in_array`` (array-level) or
    ``bcblib.imaging.math.apply_mask`` (NIfTI-level) instead."""
    warnings.warn(
        "bcblib.tools.images_utils.mask_in_array is deprecated. "
        "Use bcblib.imaging.math.apply_mask (NIfTI) or "
        "bcblib.imaging.math._mask_in_array (array) instead.",
        DeprecationWarning, stacklevel=2,
    )
    from bcblib.imaging.math import _mask_in_array
    return _mask_in_array(array, mask)


def mask_out_array(array, mask):
    """Deprecated. Use ``bcblib.imaging.math._mask_out_array`` (array-level) or
    ``bcblib.imaging.math.apply_inverse_mask`` (NIfTI-level) instead."""
    warnings.warn(
        "bcblib.tools.images_utils.mask_out_array is deprecated. "
        "Use bcblib.imaging.math.apply_inverse_mask (NIfTI) or "
        "bcblib.imaging.math._mask_out_array (array) instead.",
        DeprecationWarning, stacklevel=2,
    )
    from bcblib.imaging.math import _mask_out_array
    return _mask_out_array(array, mask)


# =========================================================================
# Functions that STAY in this module
# =========================================================================

def get_outer_inner_intensity_ratio(inner_edge_mask, outer_edge_mask):
    mean_outer = np.mean(outer_edge_mask)
    return (np.mean(inner_edge_mask) - mean_outer) / mean_outer


def get_roi_outer_inner_ratio(image, mask, dilation_connectivity=2, binarise_thr=0.5):
    """
        Calculate the perimeter of the mask, dilate it and then extracts the intensity of the input image within this mask
        to compute the outer-inner intensity ratio.

        Parameters
        ----------
        image: str
            Path to the input image in Nifti format.
        mask: str
            Path to the mask image in Nifti format.
        dilation_connectivity: int, optional
            Dilation connectivity of the mask (default is 2).
        binarise_thr: float, optional
            Threshold for binarizing the mask (default is 0.5).

        Returns
        -------
        ratio: float
            The outer-inner intensity ratio.
        inner_image: Nifti1Image
            The image data within the inner perimeter of the mask.
        outer_image: Nifti1Image
            The image data within the outer perimeter of the mask.
    """
    from bcblib.imaging.math import _dilate_array, _mask_in_array, _mask_out_array
    image_hdr = _load_nifti(image)
    image_data = image_hdr.get_fdata()
    mask_hdr = _load_nifti(mask)
    mask_data = mask_hdr.get_fdata()
    inner_edge_data = get_mask_perimeter(mask_data, binarise_thr=binarise_thr)
    if np.count_nonzero(inner_edge_data) == 0:
        inner_edge_data = mask_data
    inner_edge_masked_image_data = _mask_in_array(image_data, inner_edge_data)
    dilated_perim_data = _dilate_array(inner_edge_data, dilation_connectivity)
    outer_edge_data = _mask_out_array(dilated_perim_data, mask_data)
    outer_edge_masked_image_data = _mask_in_array(image_data, outer_edge_data)
    return get_outer_inner_intensity_ratio(
        inner_edge_masked_image_data,
        outer_edge_masked_image_data), nib.Nifti1Image(
        inner_edge_masked_image_data, image_hdr.affine), nib.Nifti1Image(outer_edge_masked_image_data, image_hdr.affine)
