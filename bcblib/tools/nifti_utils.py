import os
import warnings
from pathlib import Path
import csv
from typing import Union

from matplotlib import font_manager, pyplot as plt
from nilearn import plotting
from nilearn.image import image
from nilearn.regions import connected_regions
from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.ndimage import center_of_mass

import nibabel as nib

# Private imports from the new imaging package -- used by functions that
# STAY in this module so they never emit deprecation warnings themselves.
from bcblib.imaging.io import (
    is_nifti as _is_nifti,
    load_nifti as _load_nifti,
)


# =========================================================================
# Deprecation shims for functions moved to bcblib.imaging
# =========================================================================

def is_nifti(filename):
    """Deprecated. Use ``bcblib.imaging.io.is_nifti`` instead."""
    warnings.warn(
        "bcblib.tools.nifti_utils.is_nifti is deprecated. "
        "Use bcblib.imaging.io.is_nifti instead.",
        DeprecationWarning, stacklevel=2,
    )
    from bcblib.imaging.io import is_nifti as _fn
    return _fn(filename)


def load_nifti(img):
    """Deprecated. Use ``bcblib.imaging.io.load_nifti`` instead."""
    warnings.warn(
        "bcblib.tools.nifti_utils.load_nifti is deprecated. "
        "Use bcblib.imaging.io.load_nifti instead.",
        DeprecationWarning, stacklevel=2,
    )
    from bcblib.imaging.io import load_nifti as _fn
    return _fn(img)


def resave_nifti(nifti, output=None):
    """Deprecated. Use ``bcblib.imaging.io.resave_nifti`` instead."""
    warnings.warn(
        "bcblib.tools.nifti_utils.resave_nifti is deprecated. "
        "Use bcblib.imaging.io.resave_nifti instead.",
        DeprecationWarning, stacklevel=2,
    )
    from bcblib.imaging.io import resave_nifti as _fn
    return _fn(nifti, output=output)


def resave_nifti_list(nifti_list, output_dir=None, save_in_place=False, discard_errors=False):
    """Deprecated. Use ``bcblib.imaging.io.resave_nifti_list`` instead."""
    warnings.warn(
        "bcblib.tools.nifti_utils.resave_nifti_list is deprecated. "
        "Use bcblib.imaging.io.resave_nifti_list instead.",
        DeprecationWarning, stacklevel=2,
    )
    from bcblib.imaging.io import resave_nifti_list as _fn
    return _fn(nifti_list, output_dir=output_dir, save_in_place=save_in_place,
               discard_errors=discard_errors)


# =========================================================================
# Functions that STAY in this module (already deprecated or domain-specific)
# =========================================================================

def file_to_list(file_path, delimiter=' '):
    warnings.warn("bcblib: file_to_list is deprecated and will be removed in a future version",
                  DeprecationWarning)
    file_path = Path(file_path)
    if not file_path.is_file():
        raise ValueError(f'{file_path} does not exist.')
    if file_path.name.endswith('.csv'):
        with open(file_path, 'r') as csv_file:
            dir_list = []
            for row in csv.reader(csv_file):
                if len(row) > 1:
                    dir_list += [r for r in row]
                else:
                    dir_list.append(row[0])
    else:
        # default delimiter is ' ', it might need to be changed
        dir_list = np.loadtxt(str(file_path), dtype=str, delimiter=delimiter)
    return dir_list


def get_nifti_orientation(img):
    """Deprecated. Use ``bcblib.imaging.orient.get_orientation`` instead."""
    warnings.warn(
        "bcblib.tools.nifti_utils.get_nifti_orientation is deprecated. "
        "Use bcblib.imaging.orient.get_orientation instead.",
        DeprecationWarning, stacklevel=2,
    )
    from bcblib.imaging.orient import get_orientation
    return get_orientation(img)


def reorient_to_canonical(img, save=False):
    """Deprecated. Use ``bcblib.imaging.orient.reorient_to_standard`` instead."""
    warnings.warn(
        "bcblib.tools.nifti_utils.reorient_to_canonical is deprecated. "
        "Use bcblib.imaging.orient.reorient_to_standard instead.",
        DeprecationWarning, stacklevel=2,
    )
    from bcblib.imaging.orient import reorient_to_standard
    return reorient_to_standard(img, save=save)


def reorient_nifti_list(nifti_list, output_dir=None, save_in_place=False, discard_errors=False):
    """Deprecated. Use ``bcblib.imaging.orient.reorient_list`` instead."""
    warnings.warn(
        "bcblib.tools.nifti_utils.reorient_nifti_list is deprecated. "
        "Use bcblib.imaging.orient.reorient_list instead.",
        DeprecationWarning, stacklevel=2,
    )
    from bcblib.imaging.orient import reorient_list
    return reorient_list(nifti_list, output_dir=output_dir,
                         save_in_place=save_in_place,
                         discard_errors=discard_errors)


def get_centre_of_mass(nifti, round_coord=False):
    """Deprecated. Use ``bcblib.imaging.stats.centre_of_gravity`` instead."""
    warnings.warn(
        "bcblib.tools.nifti_utils.get_centre_of_mass is deprecated. "
        "Use bcblib.imaging.stats.centre_of_gravity instead.",
        DeprecationWarning, stacklevel=2,
    )
    from bcblib.imaging.stats import centre_of_gravity
    return centre_of_gravity(nifti, round_coord=round_coord)


def centre_of_mass_difference(nifti, reference, round_centres=False):
    """Deprecated. Use ``bcblib.imaging.stats.centre_of_gravity_distance`` instead."""
    warnings.warn(
        "bcblib.tools.nifti_utils.centre_of_mass_difference is deprecated. "
        "Use bcblib.imaging.stats.centre_of_gravity_distance instead.",
        DeprecationWarning, stacklevel=2,
    )
    from bcblib.imaging.stats import centre_of_gravity_distance
    return centre_of_gravity_distance(nifti, reference, round_centres=round_centres)


def centre_of_mass_difference_list(nifti_list, reference, fname_filter=None, round_centres=False):
    if fname_filter is not None:
        nifti_list = [f for f in nifti_list if fname_filter in Path(f).name]
    distance_dict = {}
    for f in nifti_list:
        distance_dict[str(f)] = centre_of_mass_difference(f, reference, round_centres)
    return distance_dict


def nifti_overlap_images(input_images, filter_pref='', recursive=False, mean=False, mean_for_std=None):
    if not isinstance(input_images, list):
        if Path(input_images).is_file():
            input_images = [str(p) for p in file_to_list(input_images) if _is_nifti(p)]
        elif Path(input_images).is_dir():
            if recursive:
                input_images = [str(p) for p in Path(input_images).rglob('*') if _is_nifti(p)]
            else:
                input_images = [str(p) for p in Path(input_images).iterdir() if _is_nifti(p)]
        else:
            raise ValueError('Wrong input (must be a file/directory path of a list of paths)')
    if mean_for_std is not None and mean:
        raise ValueError('mean_for_std can only be used if mean is False')
    if mean_for_std is not None:
        # if mean_for_std is pathlike, load the image
        if isinstance(mean_for_std, str) or isinstance(mean_for_std, Path) or _is_nifti(mean_for_std):
            mean_for_std = _load_nifti(mean_for_std).get_fdata()
        elif not isinstance(mean_for_std, np.ndarray):
            raise ValueError('mean_for_std must be a pathlike object, a nifti image or a numpy array')
    if filter_pref:
        input_images = [p for p in input_images if Path(p).name.startswith(filter_pref)]
    if not input_images:
        print(' The image list is empty')
        return None
    temp_overlap = None
    temp_overlap_data = None
    for img in tqdm(input_images):
        nii = nib.load(img)
        if temp_overlap is None:
            temp_overlap = nii
            temp_overlap_data = nii.get_fdata()
        else:
            if mean:
                temp_overlap_data = temp_overlap_data + nii.get_fdata()
            if mean_for_std is not None:
                temp_overlap_data += np.square(nii.get_fdata() - mean_for_std)
            else:
                temp_overlap_data += nii.get_fdata()
    if mean_for_std is not None:
        temp_overlap_data = np.sqrt(temp_overlap_data / len(input_images))
    if mean:
        temp_overlap_data = temp_overlap_data / len(input_images)
    temp_overlap = nib.Nifti1Image(temp_overlap_data, temp_overlap.affine)
    return temp_overlap


def overlaps_subfolders(root_folder, filter_pref='', subfolders_recursive=True,
                        subfolders_overlap=False, output_pref='overlap_',
                        save_in_root=True, mean=False, std_pref=None):
    """
    Compute the overlap of all the images in the subfolders of a root folder
    Notes: subfolders_overlap and subfolders_recursive are interacting with each other.
    If both are True, each subfolder overlap will be the overlap of all the subfolders of the subfolder
    Parameters
    ----------
    root_folder : str
    filter_pref : str
    subfolders_recursive : bool
    subfolders_overlap : bool
    output_pref : str
    save_in_root : bool
    mean : bool
    std_pref : str

    Returns
    -------

    """
    if not Path(root_folder).is_dir():
        raise ValueError('root_folder must be a directory')
    if subfolders_recursive and subfolders_overlap:
        raise ValueError('subfolders_recursive and subfolders_overlap cannot be True together')
    if subfolders_overlap:
        folder_list = [p for p in Path(root_folder).rglob('*') if p.is_dir()]
    else:
        folder_list = [p for p in Path(root_folder).iterdir() if p.is_dir()]
    if not mean and std_pref is not None:
        print('WARNING: std_pref is set but mean is False, mean will be computed anyway (if not already present)')
        mean = True

    for subfolder in folder_list:
        if save_in_root:
            output_folder = Path(root_folder)
        else:
            output_folder = Path(root_folder, subfolder.relative_to(root_folder).parent)
        overlap_path = output_folder / (output_pref + subfolder.name + '.nii.gz')
        if std_pref is not None:
            std_output_path = Path(root_folder, subfolder.relative_to(root_folder).parent,
                                   std_pref + subfolder.name + '.nii.gz')
            if overlap_path.is_file():
                print(f'Mean image found for [{subfolder.name}], computing std')
                overlap_nifti = _load_nifti(overlap_path)
            else:
                print(f'Computing mean image for [{subfolder.name}]')
                overlap_nifti = nifti_overlap_images(subfolder, filter_pref, recursive=subfolders_recursive, mean=True)
                if overlap_nifti is not None:
                    nib.save(overlap_nifti, overlap_path)
            if overlap_nifti is not None:
                print(f'Computing std image for [{subfolder.name}]')
                std_nifti = nifti_overlap_images(subfolder, filter_pref, recursive=subfolders_recursive,
                                                 mean_for_std=overlap_nifti.get_fdata())
                if std_nifti is not None:
                    nib.save(std_nifti, std_output_path)
        else:
            output_type = 'Overlap'
            if mean:
                output_type = 'Mean'
            print(f'{output_type} of [{subfolder.name}]')
            overlap_nifti = nifti_overlap_images(subfolder, filter_pref, recursive=subfolders_recursive, mean=mean)
            if overlap_nifti is not None:
                nib.save(overlap_nifti, overlap_path)


def binarize_nii(nii: Union[os.PathLike, nib.Nifti1Image], thr: Union[float, int] = None):
    """Deprecated. Use ``bcblib.imaging.math.binarize`` instead."""
    warnings.warn(
        "bcblib.tools.nifti_utils.binarize_nii is deprecated. "
        "Use bcblib.imaging.math.binarize instead.",
        DeprecationWarning, stacklevel=2,
    )
    from bcblib.imaging.math import binarize
    return binarize(nii, thr=thr)


def laterality_ratio(image):
    """Deprecated. Use ``bcblib.imaging.stats.laterality_ratio`` instead."""
    warnings.warn(
        "bcblib.tools.nifti_utils.laterality_ratio is deprecated. "
        "Use bcblib.imaging.stats.laterality_ratio instead.",
        DeprecationWarning, stacklevel=2,
    )
    from bcblib.imaging.stats import laterality_ratio as _fn
    return _fn(image)


def has_big_enough_cluster(img, min_cluster_size=4):
    hdr = _load_nifti(img)
    data, _ = connected_regions(hdr, min_region_size=1, extract_type='connected_components', smoothing_fwhm=0)
    data = data.get_fdata()
    max_cluster_size = np.max([np.count_nonzero(data[..., i]) for i in range(data.shape[-1])])
    return max_cluster_size >= min_cluster_size


def get_volume(img, ratio=False, threshold=0):
    """Deprecated. Use ``bcblib.imaging.stats.volume_count`` instead."""
    warnings.warn(
        "bcblib.tools.nifti_utils.get_volume is deprecated. "
        "Use bcblib.imaging.stats.volume_count instead.",
        DeprecationWarning, stacklevel=2,
    )
    from bcblib.imaging.stats import volume_count
    return volume_count(img, ratio=ratio, threshold=threshold)


def mask_statistical_image(stat_img: nib.Nifti1Image, mask_img: nib.Nifti1Image, verbose: bool=True) -> nib.Nifti1Image:
    """
    Apply a mask to a statistical image, setting voxels outside the mask range to zero.

    This function resamples a mask image to the space of the statistical image and then applies the mask
    to the statistical image. Voxels in the statistical image that fall outside the masked region
    are set to zero.

    Parameters:
    -----------
    stat_img : nib.Nifti1Image
        The statistical image to be masked.

    mask_img : nib.Nifti1Image
        The mask image to be resampled and applied to the statistical image.

    Returns:
    --------
    masked_stat_img : nib.Nifti1Image
        The masked statistical image, with non-masked voxels set to zero.

    Notes:
    ------
    The function assumes that the mask is a binary image, where non-zero voxels indicate regions of
    interest, and zeros indicate regions to be masked out. The mask is resampled to match the affine
    and shape of the statistical image before being applied.
    """
    # Resample the template to the space of the statistical image
    resampled_template_img = image.resample_img(mask_img, target_affine=stat_img.affine,
                                                target_shape=stat_img.shape)

    # Get the data from the resampled template image and the statistical image
    resampled_template_data = resampled_template_img.get_fdata()
    stat_data = stat_img.get_fdata()

    # Apply the mask: set values to zero where the template data is outside the desired range
    # resampled_template_data = (resampled_template_data >= 40) & (resampled_template_data <= 80)
    masked_stat_data = np.where(resampled_template_data, stat_data, 0)
    if verbose:
        print(f'Number of voxels removed by the mask: '
              f'{np.count_nonzero(stat_data) - np.count_nonzero(masked_stat_data)}')

    # Create a new Nifti1Image for the masked data
    masked_stat_img = nib.Nifti1Image(masked_stat_data, stat_img.affine)

    return masked_stat_img


def plot_statistical_map(stat_img_path: str, template_path: str, output_folder: str,
                         otf_font_path: str = None, title: str = '', cmap: str = 'viridis',
                         prefix: str = '', low_threshold: float = None,
                         mask_brain: Union[str, nib.Nifti1Image] = None, grid_size: str = '3x5') -> None:
    """
    Plot a statistical map overlaying a template image and save the resulting figure.

    This function loads a statistical image and a template image, optionally applies a mask
    and threshold to the statistical image, and then plots the statistical map using
    a specified colormap. The output is saved as a PNG image in the specified output folder.

    Parameters:
    -----------
    stat_img_path : str
        Path to the statistical image file (NIfTI format).

    template_path : str
        Path to the template image file (NIfTI format).

    output_folder : str
        Directory where the output image will be saved.

    otf_font_path : str, optional
        Path to the .otf font file to be used for text in the plot. If not provided, the default
        font will be used (default is None).

    title : str, optional
        Title for the plot (default is an empty string).

    cmap : str, optional
        Colormap to be used for the statistical map (default is 'viridis').

    prefix : str, optional
        Prefix for the output image filename (default is an empty string).

    low_threshold : float, optional
        Threshold value to apply to the statistical image. Voxels with values below this
        threshold are set to zero (default is None, meaning no threshold is applied).

    mask_brain : str or nib.Nifti1Image, optional
        A path to a NIfTI image or a Nifti1Image object to use as a mask. If provided,
        the mask will be applied to the statistical image using the mask_statistical_image
        function (default is None, meaning no mask is applied).

    grid_size : str, optional
        Grid size for the plot. Accepted values are '2x4', '3x4', and '3x5'
        (default is '3x5').

    Returns:
    --------
    None
        The function saves the plot as a PNG file and does not return anything.

    Notes:
    ------
    The function assumes that the template image is in MNI space and that the mask (if applied)
    is binary. The function supports different grid layouts for plotting based on the specified
    cut coordinates. The output image is saved in the specified folder with the given prefix.
    """

    # Load the custom font if provided, otherwise use default font
    font_prop = font_manager.FontProperties(fname=otf_font_path) if otf_font_path else None

    # Load the template
    mni_template = image.load_img(template_path)

    # Apply a threshold to the template (between 40 and 80)
    mni_data = mni_template.get_fdata()
    mni_data[(mni_data < 40) | (mni_data > 80)] = 0
    mni_template = nib.Nifti1Image(mni_data, mni_template.affine)

    # Load the statistical image
    stat_img = image.load_img(stat_img_path)
    stat_img_data = stat_img.get_fdata()

    # Threshold the statistical image
    if low_threshold is not None:
        stat_img_data[stat_img_data <= low_threshold] = 0
        stat_img = nib.Nifti1Image(stat_img_data, stat_img.affine)

    # Process the mask_brain parameter
    if mask_brain is not None:
        if isinstance(mask_brain, str):
            mask_img = nib.load(mask_brain)
        elif isinstance(mask_brain, nib.Nifti1Image):
            mask_img = mask_brain
        else:
            raise ValueError("mask_brain must be a path to a NIfTI image or a Nifti1Image object.")

        print('Masking the statistical image')
        stat_img = mask_statistical_image(stat_img, mask_img)
        stat_img_data = stat_img.get_fdata()

    # Determine the vmax based on the stat_img data
    vmax = stat_img_data.max()
    vmin = stat_img_data.min()

    # Set up the cut coordinates and grid based on grid_size
    if grid_size == '2x4':
        cut_coords = [-54, -36, -18, -2, 16, 34, 52, 70]
        fig, axs = plt.subplots(2, 4, figsize=(20, 10), facecolor='white')
    elif grid_size == '3x4':
        cut_coords = [-60, -46, -34, -22, -10, 2, 14, 26, 38, 50, 62, 74]
        fig, axs = plt.subplots(3, 4, figsize=(20, 15), facecolor='white')
    elif grid_size == '3x5':
        cut_coords = [-62, -52, -42, -32, -22, -12, -2, 8, 18, 28, 38, 48, 56, 66, 76]
        fig, axs = plt.subplots(3, 5, figsize=(20, 15), facecolor='white')
    else:
        raise ValueError(f"Invalid grid_size '{grid_size}'. Choose from '2x4', '3x4', or '3x5'.")

    # Adjust the title position to align with the first plot
    fig.subplots_adjust(left=0.07, right=0.9, top=0.9, wspace=-0.3, hspace=0.05)
    fig.text(0.08, 0.95, title, fontsize=16, fontproperties=font_prop, va='top', ha='left')

    # Plot the statistical map overlaying the template using the custom colormap
    for i, ax in enumerate(axs.flatten()):
        display = plotting.plot_roi(
            stat_img,
            bg_img=mni_template,
            display_mode='z',
            cut_coords=[cut_coords[i]],
            cmap=cmap,  # Use the custom colormap
            axes=ax,
            colorbar=False,  # Disable individual colorbars
            black_bg=False,
        )
        if font_prop:  # Apply custom font if provided
            for text in ax.texts:  # Accessing the texts in the Axes
                text.set_fontproperties(font_prop)

    # Create a ScalarMappable object to use for the shared colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []  # Fake data for colorbar initialization

    # Manually add the colorbar with separate axes
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])  # Adjust the [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_ticks([1, vmax])  # Set the colorbar ticks
    cbar.set_ticklabels([str(int(1)), str(int(vmax))])  # Set the colorbar tick labels

    # Save and show the figure
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, f'{prefix}stat_map.png')
    plt.savefig(output_path, facecolor='white')
    plt.show()


def get_dispersion(img):
    nii = _load_nifti(img)
    # zeros to non-zero values ratios
    data = nii.get_fdata()
    zero_nonzero_ratio = np.count_nonzero(data) / np.prod(data.shape)
    mean_to_median_ratio = np.mean(data) / np.median(data)
    """
    Returns:

    regions_extracted_imgnibabel.nifti1.Nifti1Image

        Gives the image in 4D of extracted brain regions. Each 3D image consists of only one separated region.
    index_of_each_mapnumpy.ndarray

        An array of list of indices where each index denotes the identity of each extracted region to their family of brain maps.

    """
    clusters_4d, cluster_indices = connected_regions(nii, min_region_size=1, extract_type='connected_components', fwhm=0)
    clusters_4d = clusters_4d.get_fdata()
    mean_cluster_size = np.mean([np.count_nonzero(clusters_4d[..., i]) for i in range(clusters_4d.shape[-1])])
    # distance between the centre of mass of every cluster to the centre (coordinate) of the image
    centre_of_image = np.array(nii.shape) / 2
    # TODO
    # mean_distance_to_center =
    # return zero_nonzero_ratio, mean_to_median_ratio, mean_cluster_size, distance_to_center
