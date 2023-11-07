from pathlib import Path
from typing import Sequence, List

import nibabel as nib
import numpy as np
import joblib
import umap
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr, spearmanr
import statsmodels.stats.multitest as smm
from tqdm import tqdm

from bcblib.tools.nifti_utils import reorient_to_canonical


def fwhm_to_sigma(fwhm):
    return fwhm / np.sqrt(8 * np.log(2))


def sigma_to_fwhm(sigma):
    return sigma * np.sqrt(8 * np.log(2))


def nifti_dataset_to_matrix(nifti_list: List[nib.Nifti1Image], pre_allocate_memory=True) -> np.ndarray:
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
    for i, nii in tqdm(enumerate(nifti_list)):
        nii = reorient_to_canonical(nii)
        flattened_nii = np.ravel(nii.get_fdata())
        if out_matrix is None:
            if pre_allocate_memory:
                out_matrix = np.zeros((len(nifti_list), len(flattened_nii)))
                out_matrix[0, :] = flattened_nii
        else:
            out_matrix[i, :] = flattened_nii
            # out_matrix = np.vstack((out_matrix, flattened_nii))
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


def smooth_morphospace_points(morphospace, sigma=None, fwmm=None, mode='reflect'):

    # First, we need to separate the pixels into individual 2D arrays

    # TODO that would be good to have a smoothing proportional to the size of the morphospace
    # if either sigma or fwhm is not None, and not equal to 0, then smooth the morphospace

    # TODO a solution to the increased value at the edge of the space could be to increase the space to like twice
    # the fwhm, smooth and then crop the space to the original wanted size
    if sigma is not None or fwmm is not None:
        # get the sigma value
        if sigma is not None:
            sigma = sigma
        else:
            sigma = fwhm_to_sigma(fwmm)
        # smooth the morphospace
        morphospace = gaussian_filter(morphospace, sigma=sigma, mode=mode)
    return morphospace


def rescale_morphostace_coord(input_matrix, trained_umap, out_cell_nb=10000):
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
    # get the coordinates of the input matrix in the UMAP space
    input_coordinates = trained_umap.transform(input_matrix)
    # get the maximum coordinates in each dimension of the input matrix
    max_coordinates = np.max(input_coordinates, axis=0)
    # find the scaling factor to rescale zero-centered coordinates to a grid with out_cell_nb cells
    output_mult = find_scaling_factor(out_cell_nb, max_coordinates)
    # rescale the coordinates
    output_coordinates = input_coordinates * output_mult

    return output_coordinates, output_mult


def get_3d_space(rounded_coordinates, bounding_box, filling_value=1):
    """
    Round up the coordinates to the nearest integer
    Parameters
    ----------
    rounded_coordinates : np.ndarray
        coordinates of the input matrix in the UMAP space (usually rescaled) rounded up to the nearest integer
    bounding_box : Sequence
        bounding box of the 2D arrays to create
    filling_value : float, optional
        value to fill the 2D cells with (default is 1)

    Returns
    -------

    """
    # create a 3D array of len(rounded_coordinates) 2D arrays of shape bounding_box
    volume = np.zeros((len(rounded_coordinates), *bounding_box))
    # fill the 2D arrays with the filling_value at the rounded coordinates
    for i, coord in enumerate(rounded_coordinates):
        volume[i, coord[0], coord[1]] = filling_value
    return volume


def compute_heatmap(smoothed_3d_coord, dependant_variable_values, method='spearmanr'):
    """
    Computes a heatmap of the morphospace
    Parameters
    ----------
    smoothed_3d_coord : np.ndarray
        smoothed morphospace
    dependant_variable_values : np.ndarray
        values of the dependant variable
    method : str, optional
        method to compute the heatmap (default is 'spearmanr')

    Returns
    -------
    heatmaps : np.ndarray

    """
    # compute the heatmap so the 3D smoothed morphospace becomes a 2D array with each cell containing the statistic
    # of the correlation between the columns of the smoothed morphospace and the dependant variable
    if method == 'spearmanr':
        # heatmap = np.apply_along_axis(
        #     lambda x: spearmanr(x, dependant_variable_values).correlation, 0, smoothed_3d_coord)
        heatmaps = np.apply_along_axis(
            lambda x: spearmanr(x, dependant_variable_values), 0, smoothed_3d_coord)
    elif method == 'pearsonr':
        # heatmap = np.apply_along_axis(
        #     lambda x: pearsonr(x, dependant_variable_values).correlation, 0, smoothed_3d_coord)
        heatmaps = np.apply_along_axis(
            lambda x: pearsonr(x, dependant_variable_values), 0, smoothed_3d_coord)
    else:
        raise ValueError(f'Unknown method: {method}')
    # heatmaps should be 2 2D arrays (correlation coefficient/statistics and p-value)
    return heatmaps


def create_morphospace(input_matrix, dependent_variable, out_cell_nb=10000, fwhm=None, sigma=None, filling_value=1,
                       **umap_param):
    """
    Creates a morphospace from the input matrix
    Parameters
    ----------
    input_matrix
    dependent_variable: np.ndarray
        values of the dependant variable
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
    print('UMAP trained')
    # then we transform the input matrix into the UMAP space
    orig_umap_space = trained_umap.transform(input_matrix)
    print('Input matrix transformed into the UMAP space')
    # recenter the x and y coordinates to 0
    orig_umap_space -= np.min(orig_umap_space, axis=0)

    # plot the space
    plt.scatter(orig_umap_space[:, 0], orig_umap_space[:, 1], cmap='Spectral', s=20)
    plt.show()
    # then we rescale the coordinates to the output nd-grid
    rescaled_coords, scaling_factor = rescale_morphostace_coord(
        input_matrix, trained_umap, out_cell_nb)
    # round the coordinates
    rescaled_coords_rounded = np.round(rescaled_coords).astype(int)
    # plot
    plt.scatter(rescaled_coords_rounded[:, 0], rescaled_coords_rounded[:, 1], cmap='Spectral', s=20)
    plt.show()
    # the bounding box is the maximum coordinates rounded up in each dimension of rescaled_coords
    bounding_box = [np.max(np.ceil(rescaled_coords[:, i])) for i in range(rescaled_coords.shape[1])]
    bounding_box = np.array(bounding_box).astype(int)
    # add 1 to the bounding box to account for the 0 index
    bounding_box += 1
    print(f'Min max rescaled coordinates in each dimension: ')
    print(f'Min : {[np.min(np.ceil(rescaled_coords[:, i])) for i in range(rescaled_coords.shape[1])]}, Max: {[np.max(np.ceil(rescaled_coords[:, i])) for i in range(rescaled_coords.shape[1])]}')
    print(f'Min max rounded coordinates in each dimension: ')
    print(f'Min : {[np.min(rescaled_coords_rounded[:, i]) for i in range(rescaled_coords_rounded.shape[1])]}, Max: {[np.max(rescaled_coords_rounded[:, i]) for i in range(rescaled_coords_rounded.shape[1])]}')
    print(f'bounding box: {bounding_box}')
    # get the 3D space
    morphospace = get_3d_space(rescaled_coords_rounded, bounding_box, filling_value)
    print(f'morphospace shape: {morphospace.shape}')
    # display scatter plot of the morphospace's sum
    sum_morphospace = np.sum(morphospace, axis=0)
    points_coord = np.argwhere(sum_morphospace > 0)
    plt.scatter(points_coord[:, 0], points_coord[:, 1], cmap='Spectral', s=20)
    plt.show()
    # smooth the morphospace
    # modes = ['reflect', 'constant', 'nearest', 'mirror', 'wrap']
    # for m in modes:
    #     morphospace = smooth_morphospace_points(morphospace, sigma, fwhm, mode=m)
    #     print(f'mode: {m}')
    #     print(f'Minimum value: {np.min(morphospace)}, maximum value: {np.max(morphospace)}')
    #     print(f'Sum of the morphospace: {np.sum(morphospace)}')
    #     print(f'Average value: {np.mean(morphospace)}')
    #     sum_morphospace = np.sum(morphospace, axis=0)
    #     # plot the sum of the morphospace
    #     plt.imshow(sum_morphospace.T, origin='lower')
    #     plt.show()

    morphospace = smooth_morphospace_points(morphospace, sigma, fwhm, mode='reflect')
    sum_morphospace = np.sum(morphospace, axis=0)
    # save sum_morphospace as  nifti
    sum_morphospace_nii = nib.Nifti1Image(sum_morphospace, np.eye(4))
    nib.save(sum_morphospace_nii, 'sum_morphospace.nii.gz')
    # plot the sum of the morphospace
    plt.imshow(sum_morphospace.T, origin='lower')
    plt.show()
    # len of dependent variable must be equal to the number of columns of the input matrix
    print(f'len(dependent_variable): {len(dependent_variable)}')
    print(f'input_matrix.shape: {input_matrix.shape}')
    if len(dependent_variable) != input_matrix.shape[0]:
        raise ValueError(f'len(dependent_variable) must be equal to the number of columns of the input matrix')
    heatmaps = compute_heatmap(morphospace, dependent_variable, method='pearsonr')
    # plot the heatmaps, heatmaps[0] is the correlation coefficient, heatmaps[1] is the p-value
    plt.imshow(heatmaps[0].T, origin='lower')
    # with a colourbar
    plt.colorbar()
    plt.show()
    plt.imshow(heatmaps[1].T, origin='lower')
    # with a colourbar
    plt.colorbar()
    plt.show()
    # statsmodels.stats.multitest.multipletests(pvals, alpha=0.05, method='hs', maxiter=1, is_sorted=False,
    #                                           returnsorted=False)
    # use fdr_bh method to correct for multiple comparisons the p-values and then plot the heatmap
    corrected_p_values = smm.multipletests(heatmaps[1].flatten(), alpha=0.05, method='fdr_bh')
    corrected_p_values = corrected_p_values[1].reshape(heatmaps[1].shape)

    plt.imshow(corrected_p_values.T, origin='lower')
    # with a colourbar
    plt.colorbar()
    plt.show()
    return heatmaps

