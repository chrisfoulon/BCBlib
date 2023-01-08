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

import numpy as np
def compute_disconnectome(img):
    return img

"""
Write a python function. For the paths, use the pathlib. The function takes a path and a number as input. 
First it tests whether the file exists, if not, test if it's a folder and replace path by folder + 'my_image.nii'. 
Then, it calls another function called 'compute_disconnectome' of which result is store into a variable my_disco. 
My_disco is a nifti image. Get the data of my_disco and multiply it by the number from the input. 
Return a new nifti image with the result of the multiplication as data.
"""

import pathlib
from nibabel import load, save

def multiply_disconnectome(path: str, number: float):
    # Create a Path object from the input path
    p = pathlib.Path(path)

    # Check if the path points to a file
    if p.is_file():
        # Load the file as a nifti image
        my_disco = load(p)
    # If the path is a directory, assume you want to use 'my_image.nii' in that directory
    elif p.is_dir():
        # Construct the path to the file 'my_image.nii' in the directory
        p = p / 'my_image.nii'
        # Load the file as a nifti image
        my_disco = load(p)
    else:
        # If the path is not a file or directory, raise an error
        raise ValueError(f"{path} is not a valid file or directory path")

    # Call the compute_disconnectome function and store the result in my_disco
    my_disco = compute_disconnectome(my_disco)

    # Get the data of my_disco
    data = my_disco.get_fdata()

    # Multiply the data by the input number
    result = data * number

    # Create a new nifti image with the result data
    result_image = my_disco.__class__(result, my_disco.affine, my_disco.header)

    # Return the new nifti image
    return result_image


mask = '/data/Chris/lesionsFormated/patient01.nii.gz'
print('Values in input mask: ')
print(np.unique(nib.load(mask).get_fdata()))
res_img = multiply_disconnectome(mask, 42)
print('Values in output mask: ')
print(np.unique(res_img.get_fdata()))
