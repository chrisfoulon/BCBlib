from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation, generate_binary_structure

from bcblib.tools.nifti_utils import load_nifti


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


def dilate_mask(array, connectivity, dimensions=3):
    """
    Dilates a mask using a structure determined by the connectivity (it can be a cube, a cross etc ...)
    structure ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.generate_binary_structure.html
    Parameters
    ----------
    array
    connectivity
    dimensions

    Returns
    -------
    dilated array with the same type as input array
    """
    binary_structure = generate_binary_structure(dimensions, connectivity)
    return binary_dilation(array, binary_structure).astype(array.dtype)


def mask_in_array(array, mask):
    out_array = np.copy(array)
    out_array[mask == 0] = 0
    return out_array


def mask_out_array(array, mask):
    out_array = np.copy(array)
    out_array[mask == 1] = 0
    return out_array


def get_outer_inner_intensity_ratio(inner_edge_mask, outer_edge_mask):
    mean_outer = np.mean(outer_edge_mask)
    return (np.mean(inner_edge_mask) - mean_outer) / mean_outer


def get_roi_outer_inner_ratio(image, mask, dilation_connectivity=2, binarise_thr=0.5):
    """
    Calculate the perimeter of the mask, dilate it and then extracts the intensity of the input image withing this mask
    to compute the min-max range
    Parameters
    ----------
    image
    mask
    dilation_connectivity
    binarise_thr

    Returns
    -------

    """
    image_hdr = load_nifti(image)
    image_data = image_hdr.get_fdata()
    mask_hdr = load_nifti(mask)
    mask_data = mask_hdr.get_fdata()
    inner_edge_data = get_mask_perimeter(mask_data, binarise_thr=binarise_thr)
    if np.count_nonzero(inner_edge_data) == 0:
        inner_edge_data = mask_data
    inner_edge_masked_image_data = mask_in_array(image_data, inner_edge_data)
    dilated_perim_data = dilate_mask(inner_edge_data, dilation_connectivity)
    outer_edge_data = mask_out_array(dilated_perim_data, mask_data)
    outer_edge_masked_image_data = mask_in_array(image_data, outer_edge_data)
    return get_outer_inner_intensity_ratio(
        inner_edge_masked_image_data,
        outer_edge_masked_image_data), nib.Nifti1Image(
        inner_edge_masked_image_data, image_hdr.affine), nib.Nifti1Image(outer_edge_masked_image_data, image_hdr.affine)
