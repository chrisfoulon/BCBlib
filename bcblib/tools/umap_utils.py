from pathlib import Path
from typing import Sequence, List

import nibabel as nib
import numpy as np
import joblib
import umap
from scipy.ndimage import gaussian_filter

from bcblib.tools.nifti_utils import reorient_to_canonical


def fwhm_to_sigma(fwhm):
    return fwhm / np.sqrt(8 * np.log(2))


def sigma_to_fwhm(sigma):
    return sigma * np.sqrt(8 * np.log(2))


def nifti_dataset_to_matrix(nifti_list: List[nib.Nifti1Image]):
    """
    Converts a list of nifti images into a matrix where each row is a flattened nifti image
    after reorienting it to canonical space
    Parameters
    ----------
    nifti_list : list of nibabel.Nifti1Image

    Returns
    -------
    out_matrix : np.ndarray
        matrix where each row is a flattened nifti image
    Notes
    -----
    The order of the images in the matrix is the same as the order of the images in the input list
    """
    # reorient the images and then flatten them into a matrix
    out_matrix = None
    for nii in nifti_list:
        nii = reorient_to_canonical(nii)
        flattened_nii = np.ravel(nii.get_fdata())
        if out_matrix is None:
            out_matrix = flattened_nii
        else:
            out_matrix = np.vstack((out_matrix, flattened_nii))
    return out_matrix


def train_umap(input_matrix, **umap_param):
    """
    Trains a UMAP model on the input matrix
    Parameters
    ----------
    input_matrix : np.ndarray
        matrix where each row is a flattened nifti image
    umap_param : dict
        parameters for the UMAP model
        (e.g. n_neighbors=5, random_state=42, disconnection_distance=0.1)

    Returns
    -------
    umap_model : umap.UMAP
        trained UMAP model
    """
    umap_model = umap.UMAP(**umap_param)
    umap_model.fit(input_matrix)
    return umap_model


def find_scaling_factor(out_cell_nb, max_coordinates):
    """
    Finds the scaling factor to rescale zero-centered coordinates to a grid with out_cell_nb cells
    Parameters
    ----------
    out_cell_nb : int
        number of cells in the output nd-grid
    max_coordinates : Sequence
        sequence of the maximum coordinates in each dimension of the input nd-grid

    Returns
    -------
    output_mult : float
    Notes: with the help of Guillaume Corré (https://github.com/notdryft)
    """
    output_mult = (out_cell_nb / np.prod(max_coordinates)) ** (1 / len(max_coordinates))
    return output_mult



def smooth_morphospace_points(morphospace, sigma=None, fwmm=None):

    # First, we need to separate the pixels into individual 2D arrays

    # TODO that would be good to have a smoothing proportional to the size of the morphospace
    # if either sigma or fwhm is not None, and not equal to 0, then smooth the morphospace
    if sigma is not None or fwmm is not None:
        # get the sigma value
        if sigma is not None:
            sigma = sigma
        else:
            sigma = fwhm_to_sigma(fwmm)
        # smooth the morphospace
        morphospace = gaussian_filter(morphospace, sigma=sigma)


def rescale_morphostace_coord(input_matrix, trained_umap, out_cell_nb=10000, fwhm=None, sigma=None, filling_value=1):
    """
    Creates a morphospace from the input matrix and the trained UMAP model
    Parameters
    ----------
    input_matrix : np.ndarray
        matrix where each row is a flattened nifti image
    trained_umap : umap.UMAP
        trained UMAP model
    out_cell_nb : int, optional
        number of cells in the output nd-grid (default is 10000)

    Returns
    -------
    output_coordinates : np.ndarray
        coordinates of the input matrix in the UMAP space rescaled to the output nd-grid
    output_mult : float
        scaling factor to rescale zero-centered coordinates to a grid with out_cell_nb cells

    """
    # fwhm and sigma are mutually exclusive
    if fwhm is not None and sigma is not None:
        raise ValueError("fwhm and sigma are mutually exclusive")

    # get the coordinates of the input matrix in the UMAP space
    input_coordinates = trained_umap.transform(input_matrix)
    # get the maximum coordinates in each dimension of the input matrix
    max_coordinates = np.max(input_coordinates, axis=0)
    # find the scaling factor to rescale zero-centered coordinates to a grid with out_cell_nb cells
    output_mult = find_scaling_factor(out_cell_nb, max_coordinates)
    # rescale the coordinates
    output_coordinates = input_coordinates * output_mult

    return output_coordinates, output_mult

def get_rounded_coord(output_coordinates):
    """
    Round up the coordinates to the nearest integer
    Parameters
    ----------
    output_coordinates

    Returns
    -------

    """
    # round the coordinates to the nearest integer (bounding box)
    output_coordinates_rounded = np.round(output_coordinates).astype(int)

    max_coordinates_rounded = np.max(output_coordinates_rounded, axis=0)
    # create the morphospace
    morphospace = np.zeros(max_coordinates_rounded)
    # fill the morphospace with the coordinates
    morphospace[tuple(output_coordinates_rounded.T)] = filling_value

    return morphospace


def create_morphospace(input_matrix, out_cell_nb=10000, fwhm=None, sigma=None, filling_value=1, **umap_param):
    """
    Creates a morphospace from the input matrix
    Parameters
    ----------
    input_matrix
    out_cell_nb
    fwhm
    sigma
    filling_value
    umap_param

    Returns
    -------

    Notes
    -----
    The columns of the input_matrix are the input observation and their order is preserved in the output morphospace
    """
    # first we train the UMAP model
    trained_umap = train_umap(input_matrix, **umap_param)
    # then we transform the input matrix into the UMAP space
    orig_umap_space = trained_umap.transform(input_matrix)
    # then we rescale the coordinates to the output nd-grid
    rescaled_coords, scaling_factor = rescale_morphostace_coord(
        input_matrix, trained_umap, out_cell_nb, fwhm, sigma, filling_value)
    # round the coordinates
    rescaled_coords_rounded = np.round(rescaled_coords).astype(int)
    # the bounding box is the maximum coordinates rounded up in each dimension of rescaled_coords
    bounding_box = [np.max(np.ceil(rescaled_coords[:, i])) for i in range(rescaled_coords.shape[1])]
    print(f'bounding box: {bounding_box}')
