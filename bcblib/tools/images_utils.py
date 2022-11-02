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


def mask_array(array, mask):
    array[mask == 0] = 0
    return array


def get_masked_img_min_max_range(mask_arr):
    return np.max(mask_arr) - np.min(mask_arr)


def get_roi_intensity_range(image, mask, dilation_connectivity=2, binarise_thr=0.5):
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
    perim_data = get_mask_perimeter(mask_data, binarise_thr=binarise_thr)
    perim_data = dilate_mask(perim_data, dilation_connectivity)
    masked_image_data = mask_array(image_data, perim_data)
    return get_masked_img_min_max_range(masked_image_data), nib.Nifti1Image(masked_image_data, image_hdr.affine)
