import os
from copy import deepcopy
from pathlib import Path
from typing import Sequence, List

import nibabel as nib
import numpy as np
import joblib
import umap
import matplotlib as mpl
from matplotlib import pyplot as plt
from numpy.linalg import svd
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr, spearmanr, mannwhitneyu
import statsmodels.stats.multitest as smm
from tqdm import tqdm
import mne

from bcblib.tools.nifti_utils import reorient_to_canonical


def fwhm_to_sigma(fwhm):
    return fwhm / np.sqrt(8 * np.log(2))


def sigma_to_fwhm(sigma):
    return sigma * np.sqrt(8 * np.log(2))


def nifti_dataset_to_matrix(nifti_list: List[nib.Nifti1Image], pre_allocate_memory: bool = True,
                            disable_progress: bool = False) -> np.ndarray:
    """
    Converts a list of nifti images into a matrix where each row is a flattened nifti image
    after reorienting it to canonical space
    Parameters
    ----------
    nifti_list : list of nibabel.Nifti1Image
        list of nifti images
    pre_allocate_memory : bool, optional
        whether to pre-allocate the memory for the output matrix (default is True)
    disable_progress : bool, optional
        whether to disable the progress bar (default is False)

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
    for i, nii in tqdm(enumerate(nifti_list), disable=disable_progress):
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
        # create a tuple with the sigma for each dimension besides the first one which
        # should not be smoothed
        sigma_tuple = (0,) + (sigma,) * (morphospace.ndim - 1)
        # smooth the morphospace along the first axis
        morphospace = gaussian_filter(morphospace, sigma=sigma_tuple, mode=mode)
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
    # TODO WE DO NOT NEED TO EMBED THE INPUT IF IT WAS THE TRAINING DATA!
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
    for i, coord in tqdm(enumerate(rounded_coordinates)):
        volume[i, coord[0], coord[1]] = filling_value
    return volume


def compute_scores_heatmap(smoothed_3d_coord, dependant_variable_values):
    # weight each slice of smoother_3d_coord by the dependant_variable_values
    weighted_smoothed_3d_coord = deepcopy(smoothed_3d_coord)
    for i in range(smoothed_3d_coord.shape[0]):
        weighted_smoothed_3d_coord[i, :, :] *= dependant_variable_values[i]
    return weighted_smoothed_3d_coord


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
    # TODO TRY https://mne.tools/stable/generated/mne.stats.permutation_cluster_test.html
    if method == 'mne':
        # try with the pearsonr with dependant_variable_values
        def stat_func(x):
            pearsonr(x, dependant_variable_values)
            # mannwhitneyu(x, dependant_variable_values)

        morphospace_list = []
        for i in range(smoothed_3d_coord.shape[0]):
            morphospace_list.append(smoothed_3d_coord[i, :, :].flatten())
        heatmaps = mne.stats.permutation_cluster_test(morphospace_list, n_permutations=1000)
    # compute the heatmap so the 3D smoothed morphospace becomes a 2D array with each cell containing the statistic
    # of the correlation between the columns of the smoothed morphospace and the dependant variable
    elif method == 'spearmanr':
        # heatmap = np.apply_along_axis(
        #     lambda x: spearmanr(x, dependant_variable_values).correlation, 0, smoothed_3d_coord)
        heatmaps = np.apply_along_axis(
            lambda x: spearmanr(x, dependant_variable_values), 0, smoothed_3d_coord)
    elif method == 'pearsonr':
        # heatmap = np.apply_along_axis(
        #     lambda x: pearsonr(x, dependant_variable_values).correlation, 0, smoothed_3d_coord)
        heatmaps = np.apply_along_axis(
            lambda x: pearsonr(x, dependant_variable_values), 0, smoothed_3d_coord)
    elif method == 'mannwhitneyu':
        heatmaps = np.apply_along_axis(
            lambda x: mannwhitneyu(x, dependant_variable_values), 0, smoothed_3d_coord)
    else:
        raise ValueError(f'Unknown method: {method}')
    # heatmaps should be 2 2D arrays (correlation coefficient/statistics and p-value)
    return heatmaps


def get_overlaping_input_indices(p_map, slices):
    """
    Gets the inputs that are in the morphospace
    Parameters
    ----------
    p_map : np.ndarray
        corrected p-values
    slices : np.ndarray
        3D array of the 2D smoothed slices of corresponding to the embedded coordinates in the UMAP space

    Returns
    -------
    overlaping_inputs : list
    """
    # for each 2D in the morphospace check which overlaps with the corrected p-values
    overlaping_inputs = []
    for i in range(slices.shape[0]):
        if np.nansum(slices[i, :, :] * p_map) > 0:
            overlaping_inputs.append(i)
    return overlaping_inputs


def clean_pipeline(input_matrix, dependent_variable, output_folder, matrix_for_embeddings=None, trained_umap=None,
                   out_cell_nb=10000, fwhm=None, sigma=None, filling_value=1, show=-1,
                   stats_method='pearsonr', points_overlap_proportion_threshold=0.8, prefix='',
                   **umap_param):
    """
    1) Train or load a UMAP model
    2) Transform the input matrix into the UMAP space
    3) Rescale the coordinates to the output nd-grid (check the overlap proportion)
    3.5) If matrix_for_embeddings is provided, transform it into the UMAP space and rescale the coordinates to the
    output nd-grid
    4) Create the stacked 2D slices for the morphospace (using the matrix_for_embeddings if provided)
    5) Smooth the morphospace
    6) Compute the dependent variable heatmap
    7) Run the stats_method on the heatmap
    8) Correct the p-values
    Parameters
    ----------
    input_matrix
    dependent_variable
    output_folder
    matrix_for_embeddings
    trained_umap
    out_cell_nb
    fwhm
    sigma
    filling_value
    show
    stats_method
    points_overlap_proportion_threshold
    prefix
    umap_param

    Returns
    -------

    """
    if not Path(output_folder).is_dir():
        Path(output_folder).mkdir(parents=True)

    # first we train the UMAP model
    if trained_umap is None:
        trained_umap = train_umap(input_matrix, **umap_param)
        joblib.dump(trained_umap, Path(output_folder).joinpath('trained_umap.sav'))
        print(f'Trained UMAP model saved in {Path(output_folder).joinpath("trained_umap.sav")}')
    else:
        # if trained_umap is a path load it, otherwise assume it is a trained UMAP model
        if isinstance(trained_umap, os.PathLike) or isinstance(trained_umap, str):
            print('Using the provided trained UMAP model')
            trained_umap = joblib.load(trained_umap)
        else:
            if not isinstance(trained_umap, umap.UMAP):
                print(f'trained_umap type: {type(trained_umap)}')
                raise ValueError(f'trained_umap must be a path to a trained UMAP model or a trained UMAP model')

    # then we transform the input matrix into the UMAP space
    orig_umap_space = trained_umap.transform(input_matrix)
    print('Input matrix transformed into the UMAP space')
    # recenter the x and y coordinates to 0
    orig_umap_space -= np.min(orig_umap_space, axis=0)
    # then we rescale the coordinates to the output nd-grid
    rescaled_coords, scaling_factor = rescale_morphostace_coord(
        input_matrix, trained_umap, out_cell_nb)
    # round the coordinates
    rescaled_coords_rounded = np.round(rescaled_coords).astype(int)
    # the bounding box is the maximum coordinates rounded up in each dimension of rescaled_coords
    bounding_box = [np.max(np.ceil(rescaled_coords[:, i])) for i in range(rescaled_coords.shape[1])]
    bounding_box = np.array(bounding_box).astype(int)
    # add 1 to the bounding box to account for the 0 index
    bounding_box += 1
    # if matrix_for_embeddings is provided, transform it into the UMAP space and rescale the coordinates to the
    # output nd-grid
    if matrix_for_embeddings is not None:
        # if this is a path load it, otherwise assume it is a matrix
        if isinstance(matrix_for_embeddings, os.PathLike) or isinstance(matrix_for_embeddings, str):
            print('Loading the matrix for embeddings')
            matrix_for_embeddings = np.load(matrix_for_embeddings)
        embeddings_for_stats = trained_umap.transform(matrix_for_embeddings)
        # centre using the min of orig_umap_space and rescale using the scaling factor found with the input_matrix
        embeddings_for_stats -= np.min(orig_umap_space, axis=0)
        embeddings_for_stats *= scaling_factor
        # round the coordinates
        embeddings_for_stats_rounded = np.round(embeddings_for_stats).astype(int)
        # the bounding box is the maximum coordinates rounded up in each dimension of rescaled_coords
        bounding_box_for_stats = [np.max(np.ceil(embeddings_for_stats[:, i])) for i in range(embeddings_for_stats.shape[1])]


def test_pipeline(input_matrix, dependent_variable, output_folder, trained_umap=None,
                  out_cell_nb=10000, fwhm=None, sigma=None, filling_value=1, show=-1,
                  stats_method='pearsonr', points_overlap_proportion_threshold=0.8, prefix='',
                  **umap_param):
    mpl.rcParams['image.cmap'] = 'plasma'

    if not Path(output_folder).is_dir():
        Path(output_folder).mkdir(parents=True)

    # first we train the UMAP model
    if trained_umap is None:
        trained_umap = train_umap(input_matrix, **umap_param)
        joblib.dump(trained_umap, Path(output_folder).joinpath('trained_umap.sav'))
        print(f'Trained UMAP model saved in {Path(output_folder).joinpath("trained_umap.sav")}')
    else:
        # if trained_umap is a path load it, otherwise assume it is a trained UMAP model
        if isinstance(trained_umap, os.PathLike) or isinstance(trained_umap, str):
            print('Using the provided trained UMAP model')
            trained_umap = joblib.load(trained_umap)
        else:
            if not isinstance(trained_umap, umap.UMAP):
                print(f'trained_umap type: {type(trained_umap)}')
                raise ValueError(f'trained_umap must be a path to a trained UMAP model or a trained UMAP model')

    # then we transform the input matrix into the UMAP space
    orig_umap_space = trained_umap.transform(input_matrix)
    print('Input matrix transformed into the UMAP space')
    # save a figure of the UMAP space as a png file without changing the ratio
    plt.scatter(orig_umap_space[:, 0], orig_umap_space[:, 1], cmap='Spectral', s=20)
    plt.title('UMAP space')
    plt.axis('scaled')
    plt.savefig(Path(output_folder, 'umap_space.png'), bbox_inches='tight')
    plt.close()
    # recenter the x and y coordinates to 0
    orig_umap_space -= np.min(orig_umap_space, axis=0)
    # plot the space
    if show == -1 or 1 in show:
        plt.scatter(orig_umap_space[:, 0], orig_umap_space[:, 1], cmap='Spectral', s=20)
        plt.title('UMAP space')
        plt.show()
    # then we rescale the coordinates to the output nd-grid
    rescaled_coords, scaling_factor = rescale_morphostace_coord(
        input_matrix, trained_umap, out_cell_nb)
    # round the coordinates
    rescaled_coords_rounded = np.round(rescaled_coords).astype(int)
    # plot
    if show == -1 or 2 in show:
        plt.scatter(rescaled_coords_rounded[:, 0], rescaled_coords_rounded[:, 1], cmap='Spectral', s=20)
        plt.show()
    # the bounding box is the maximum coordinates rounded up in each dimension of rescaled_coords
    bounding_box = [np.max(np.ceil(rescaled_coords[:, i])) for i in range(rescaled_coords.shape[1])]
    bounding_box = np.array(bounding_box).astype(int)
    # add 1 to the bounding box to account for the 0 index
    bounding_box += 1
    # get the 3D space
    print('Creating the stacked 2D slices of the morphospace')
    slices = get_3d_space(rescaled_coords_rounded, bounding_box, filling_value)
    slices_nii = nib.Nifti1Image(slices, np.eye(4))
    nib.save(slices_nii, Path(output_folder, 'slices.nii.gz'))
    print(f'morphospace shape: {slices.shape}')
    # display scatter plot of the slices' sum
    sum_slices = np.sum(slices, axis=0)
    points_coord = np.argwhere(sum_slices > 0)
    print(f'points_coord.shape: {points_coord.shape}')
    # if the first dimension of the points_coord is less than points_overlap_proportion_threshold of
    # the number observations in input_matrix, then throw a warning (very visible)
    if points_coord.shape[0] < points_overlap_proportion_threshold * input_matrix.shape[0]:
        raise Warning(f'Less than {points_overlap_proportion_threshold} of the observations in input_matrix are '
                      f'in the morphospace. You might want to increase the number of cells in the output nd-grid')
    if show == -1 or 3 in show:
        plt.scatter(points_coord[:, 0], points_coord[:, 1], cmap='Spectral', s=20)
        plt.title('Sum of the slices')
        # save the figure as a png file without changing the ratio
        plt.axis('scaled')
        plt.savefig(Path(output_folder, f'points_coord.png'), bbox_inches='tight')
        plt.show()
    smoothed_slices = smooth_morphospace_points(slices, sigma, fwhm, mode='reflect')
    sum_smoothed_slices = np.sum(smoothed_slices, axis=0)
    print(f'smoothed_slices.shape: {smoothed_slices.shape}')
    # plot sum_smoothed_slices
    if show == -1 or 4 in show:
        plt.imshow(sum_smoothed_slices.T, origin='lower')
        plt.title('Sum of the smoothed slices')
        plt.show()
    # save sum_morphospace as  nifti
    smoothed_slices_nii = nib.Nifti1Image(smoothed_slices, np.eye(4))
    nib.save(smoothed_slices_nii, Path(output_folder, 'smoothed_slices.nii.gz'))

    # len of dependent variable must be equal to the number of columns of the input matrix
    print(f'len(dependent_variable): {len(dependent_variable)}')
    print(f'input_matrix.shape: {input_matrix.shape}')
    if len(dependent_variable) != input_matrix.shape[0]:
        raise ValueError(f'len(dependent_variable) must be equal to the number of columns of the input matrix')
        # dependent_variable = distance_from_selected_point.flatten()
    heatmaps = compute_heatmap(smoothed_slices, dependent_variable, method=stats_method)
    # plot the heatmaps, heatmaps[0] is the correlation coefficient, heatmaps[1] is the p-value
    if show == -1 or 5 in show:
        plt.imshow(heatmaps[0].T, origin='lower')
        # with a colourbar
        plt.colorbar(shrink=0.5)
        plt.title('Correlation coefficient')
        # save the figure as a png file without changing the ratio
        plt.axis('scaled')
        plt.savefig(Path(output_folder, f'{prefix}correlation_coefficient.png'),
                    bbox_inches='tight')
        plt.show()
        # threshold the heatmap[0] at -0.3 and under
        heatmaps[0][heatmaps[0] > -0.3] = np.nan
        plt.imshow(heatmaps[0].T, origin='lower')
        # with a colourbar
        plt.colorbar(shrink=0.5)
        plt.title('Correlation coefficient thresholded at -0.3')
        plt.show()

    if show == -1 or 6 in show:
        thresholded = True
        if thresholded:
            # threshold the heatmap to only keep the significant correlations (p < 0.05)
            heatmaps[1][heatmaps[1] >= 0.05] = np.nan
        # # save heatmaps[1] as  nifti
        # heatmaps_nii = nib.Nifti1Image(np.array(heatmaps[1]), np.eye(4))
        # nib.save(heatmaps_nii, Path(output_folder, 'p_values.nii.gz'))
        # # save slice 1 as nifti
        # slice_1_nii = nib.Nifti1Image(smoothed_slices[1, :, :], np.eye(4))
        # nib.save(slice_1_nii, Path(output_folder, 'slice_1.nii.gz'))
        #
        # # save slice 10 as nifti
        # slice_10_nii = nib.Nifti1Image(smoothed_slices[10, :, :], np.eye(4))
        # nib.save(slice_10_nii, Path(output_folder, 'slice_10.nii.gz'))

        correlated_input_indices = get_overlaping_input_indices(heatmaps[1], smoothed_slices)
        print(f'Indices of correlated (uncorrected) inputs: {correlated_input_indices}')

        # show the points from correlated_input_indices these are the indices in
        # rescaled_coords_rounded
        # create a heatmap with these coordinates to plot it with imshow
        correlated_points_heatmap = np.zeros((bounding_box[0], bounding_box[1]))
        for i in correlated_input_indices:
            print(f'Correlated point {i}: {rescaled_coords_rounded[i, :]}')
            correlated_points_heatmap[rescaled_coords_rounded[i, 0], rescaled_coords_rounded[i, 1]] = 1
        mpl.rcParams['image.cmap'] = 'plasma'
        plt.imshow(correlated_points_heatmap.T, origin='lower')

        plt.colorbar(shrink=0.5)
        plt.title('Points from correlated_input_indices')
        plt.show()
        mpl.rcParams['image.cmap'] = 'plasma'
        plt.imshow(heatmaps[1].T, origin='lower')
        # with a colourbar
        plt.colorbar(shrink=0.5)
        plt.title('p-values')
        plt.show()

    scores_heatmap = compute_scores_heatmap(smoothed_slices, dependent_variable)
    print(f'scores_heatmap.shape: {scores_heatmap.shape}')
    # save scores_heatmap as  nifti
    scores_heatmap_nii = nib.Nifti1Image(scores_heatmap, np.eye(4))
    nib.save(scores_heatmap_nii, Path(output_folder, 'scores_heatmap.nii.gz'))
    # plot sum of scores_heatmap
    sum_scores_heatmap = np.sum(scores_heatmap, axis=0)
    print(f'sum_scores_heatmap.shape: {sum_scores_heatmap.shape}')
    if show == -1 or 7 in show:
        plt.imshow(sum_scores_heatmap.T, origin='lower')
        plt.title('Sum of the scores heatmap')
        plt.colorbar(shrink=0.5)
        plt.show()

    methods = ['holm']
    corrected_p_maps = {}
    correlated_input_indices = []
    for m in methods:
        corrected_p_values = smm.multipletests(heatmaps[1].flatten(), alpha=0.05, method=m, maxiter=-1)
        # if corrected_p_values doesn't contain anything between 0 < p < 0.05 skip the method
        if np.sum(corrected_p_values[1] < 0.05) == 0:
            print(f'No p-values between 0 and 0.05 using the {m} method')
            continue
        corrected_p_values = corrected_p_values[1].reshape(heatmaps[1].shape)
        # threshold the corrected p-values to only keep the significant ones (< 0.05)
        corrected_p_values[corrected_p_values >= 0.05] = np.nan

        corrected_p_maps[m] = corrected_p_values

        correlated_input_indices = get_overlaping_input_indices(corrected_p_values, smoothed_slices)
        print(f'Indices of correlated inputs: {correlated_input_indices}')
        # plot the correlation values where the p-values are < 0.05
        significant_correlations = np.zeros(corrected_p_values.shape)
        significant_correlations[corrected_p_values < 0.05] = heatmaps[0][corrected_p_values < 0.05]
        # we need nans for the values that are not significant
        significant_correlations[significant_correlations == 0] = np.nan
        plt.imshow(significant_correlations.T, origin='lower')
        # add a colorbard with hot colours for the significant p-values
        plt.colorbar(shrink=0.5)

        # plt.title(f'Correlation value (where p-values < 0.05 after Bonferroni-Holm correction)')
        # save the figure as a png file without changing the ratio
        plt.axis('scaled')
        plt.savefig(Path(output_folder, f'{prefix}correlated_points.png'),
                    bbox_inches='tight')
        plt.show()
    return slices, smoothed_slices, sum_smoothed_slices, correlated_input_indices


def align_embeddings(embedding1, embedding2):
    # TODO modify it to use from scipy.spatial import procrustes
    # work on copies of the embeddings
    embedding1 = embedding1.copy()
    embedding2 = embedding2.copy()
    # Translate the embeddings
    center_of_mass1 = np.mean(embedding1, axis=0)
    center_of_mass2 = np.mean(embedding2, axis=0)
    embedding1 -= center_of_mass1
    embedding2 -= center_of_mass2

    # Scale the embeddings
    average_norm1 = np.mean(np.linalg.norm(embedding1, axis=1))
    average_norm2 = np.mean(np.linalg.norm(embedding2, axis=1))
    embedding1 /= average_norm1
    embedding2 /= average_norm2

    # Compute the matrix product of the first embedding and the second embedding transposed
    matrix_product = np.dot(embedding1.T, embedding2)

    # Perform a singular value decomposition on this matrix product
    u, _, vt = svd(matrix_product)

    # Compute the rotation matrix
    rotation_matrix = np.dot(vt.T, u.T)

    # Rotate the first embedding
    rotated_embedding1 = np.dot(embedding1, rotation_matrix)

    return rotated_embedding1, embedding2
