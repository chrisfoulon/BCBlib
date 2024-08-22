"""
Generate clustered datasets for use in neuroimaging analyses

Authors: Chris Foulon & Michel Thiebaut de Scotten

Description:
------------
This script is designed to generate clustered datasets, particularly synthetic lesions, for neuroimaging analyses.
It supports multiple clustering methods, including KMeans and a custom compactor method that can enforce contiguity
between clusters. The script can process a list of NIfTI images to create a combined mask and then apply clustering
based on user-defined parcel sizes or a specified number of parcels.

Features:
---------
- Supports both KMeans clustering and a custom compactor method.
- Clusters can be defined by size or by the number of desired parcels.
- The compactor method allows for contiguous or non-contiguous clusters.
- The script can process multiple images to create a unified mask for clustering.
- Clustering can be configured with different strategies for size distribution (fixed or balanced).
- Outputs the resulting clusters as NIfTI images with associated metadata.

Usage:
------
- Command-line usage with various options to specify input paths, output directories, clustering methods, and more.
- The script includes options to fix the random seed for reproducibility, enforce contiguity in clusters, and apply smoothing.

Examples:
---------
The script supports different scenarios for creating clustered datasets. Examples of command-line usage include:

1. KMeans with Parcel Size List:
   parcitron -p "$path" -o "${output_path}/KMeans_parcel_sizes" --method KMeans -rsl 30000,50000 --random_state 42

2. KMeans with Number of Parcels:
   parcitron -p "$path" -o "${output_path}/KMeans_num_parcels" --method KMeans -np 50 --random_state 42

3. Compactor with Fixed Size (Non-Contiguous):
   parcitron -p "$path" -o "${output_path}/Compactor_fixed_noncontig" --method compactor -rsl 30000 --strategy fixed_size --random_state 42

4. Compactor with Fixed Size (Contiguous):
   parcitron -p "$path" -o "${output_path}/Compactor_fixed_contig" --method compactor -rsl 30000 --strategy fixed_size --contiguous --random_state 42

5. Compactor with Balanced Size (Non-Contiguous):
   parcitron -p "$path" -o "${output_path}/Compactor_balanced_noncontig" --method compactor -rsl 30000 --strategy balanced_size --random_state 42

6. Compactor with Balanced Size (Contiguous):
   parcitron -p "$path" -o "${output_path}/Compactor_balanced_contig" --method compactor -rsl 30000 --strategy balanced_size --contiguous --random_state 42

7. Compactor with Number of Parcels (Non-Contiguous):
   parcitron -p "$path" -o "${output_path}/Compactor_num_parcels_noncontig" --method compactor -np 20,30,50,100 --random_state 42

8. Compactor with Number of Parcels (Contiguous):
   parcitron -p "$path" -o "${output_path}/Compactor_num_parcels_contig" --method compactor -np 50 --contiguous --random_state 42

"""
import os
import argparse
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import json
import csv

import nibabel as nib
import nilearn
from nilearn.masking import compute_multi_background_mask, intersect_masks
from nilearn.image import threshold_img
from sklearn.cluster import KMeans

from bcblib.tools.general_utils import file_to_list, str_to_lower
from bcblib.tools.divide_mask import divide_compactor
from bcblib.tools.nifti_utils import is_nifti, load_nifti


def determine_parcels(total_points, num_parcels=None, parcel_size=None, strategy="equal_size"):
    """
    Determine the number of parcels and their sizes based on the desired strategy.

    Parameters
    ----------
    total_points : int
        Total number of points to be divided into parcels.
    num_parcels : int, optional
        Number of parcels desired.
    parcel_size : int, optional
        Desired size of each parcel.
    strategy : str, optional
        Strategy to use for parceling.
        Options are:
        - "equal_size": (default) Create exactly `num_parcels` parcels of roughly equal size (+- 1).
        - "fixed_size": Create parcels of size `parcel_size`, with a leftover parcel if necessary.
        - "balanced_size": Create parcels of size close to `parcel_size`, balancing the sizes to avoid a very small
        leftover parcel.

    Returns
    -------
    list of int
        A list containing the sizes of each parcel.
    """
    if strategy == "equal_size":
        if num_parcels is None:
            raise ValueError("`num_parcels` must be provided for the 'equal_size' strategy.")
        base_size = total_points // num_parcels
        remainder = total_points % num_parcels
        parcel_sizes = [base_size + 1 if i < remainder else base_size for i in range(num_parcels)]

    elif strategy == "fixed_size":
        if parcel_size is None:
            raise ValueError("`parcel_size` must be provided for the 'fixed_size' strategy.")
        num_parcels = total_points // parcel_size
        remainder = total_points % parcel_size
        parcel_sizes = [parcel_size] * num_parcels
        if remainder > 0:
            parcel_sizes.append(remainder)

    elif strategy == "balanced_size":
        if parcel_size is None:
            raise ValueError("`parcel_size` must be provided for the 'balanced_size' strategy.")
        num_parcels = total_points // parcel_size
        base_size = total_points // num_parcels
        remainder = total_points % num_parcels
        parcel_sizes = [base_size + 1 if i < remainder else base_size for i in range(num_parcels)]

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return parcel_sizes



def create_coverage_mask(image_path_list):
    """
    Create a combined coverage mask from a list of NIfTI image paths.

    Parameters
    ----------
    image_path_list : list of str
        List of paths to NIfTI images.

    Returns
    -------
    Nifti1Image
        A combined coverage mask of the input images.
    """
    nii_list = []

    for f in image_path_list:
        if not os.path.isfile(f):
            raise ValueError('{} is not an existing file'.format(f))
        if not nii_list:
            nii_list = [load_nifti(f)]
        else:
            nii_list.append(load_nifti(f))
    return compute_multi_background_mask(nii_list, threshold=0, connected=False, n_jobs=-1)


def create_parcel_set(coverage_mask, parcel_sizes=None, num_parcels=None, output_path=None, method='KMeans',
                      strategy="equal_size"):
    """
    Create a set of parcels from a coverage mask using the specified clustering method.

    Parameters
    ----------
    coverage_mask : Nifti1Image
        The NIfTI mask image to be parcelated.
    parcel_sizes : int or Sequence[int], optional
        Desired size of each parcel, or a list of sizes for each parcel.
    num_parcels : int, optional
        Number of parcels to generate.
    output_path : str, optional
        Path to save the output NIfTI image.
    method : str, optional
        Clustering method to use: 'KMeans' or 'compactor'.
    strategy : str, optional
        Strategy for parceling with compactor when using parcel_size ('fixed_size', 'balanced_size').

    Returns
    -------
    Nifti1Image
        The parcelated NIfTI image.
    """

    if output_path is not None:
        output_path = os.path.abspath(output_path)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
    mask_coord = np.array(np.where(coverage_mask.get_fdata())).T

    # Determine the sizes based on the input parameters
    if isinstance(parcel_sizes, int):
        sizes = determine_parcels(len(mask_coord), parcel_size=parcel_sizes, strategy=strategy)
    elif isinstance(parcel_sizes, Sequence):
        sizes = parcel_sizes
        if sum(sizes) > len(mask_coord):
            raise ValueError("The sum of custom sizes exceeds the number of voxels in the mask.")
    elif num_parcels is not None:
        sizes = determine_parcels(len(mask_coord), num_parcels=num_parcels, strategy=strategy)
    else:
        raise ValueError("Either parcel_sizes or num_parcels must be provided.")
    print(f'Method: {method}')
    if method.lower() == 'compactor':
        # print(f'Running divide_compactor with sizes = {sizes}')
        print(f'Size of the mask (non-zero): {mask_coord.shape[0]}')
        parcels_img = divide_compactor(coverage_mask, sizes=sizes, random_labels=True)
        new_data = parcels_img.get_fdata()
    else:  # Default to KMeans
        k = len(sizes)  # Use the number of parcels determined by sizes
        print(f'Running KMeans with k = {k}')
        kmeans = KMeans(k, n_init='auto').fit(mask_coord)
        kmeans_labels_img = kmeans.labels_
        new_data = np.zeros(coverage_mask.shape, int)
        for ind, c in enumerate(mask_coord):
            # KMeans labels start at 0, to avoid the first cluster to be in the 0 background of the image we add 1
            new_data[tuple(c)] = kmeans_labels_img[ind] + 1

    # set new_data dtype to the same as the coverage_mask
    new_data = new_data.astype(coverage_mask.get_fdata().dtype)
    new_nii = nib.Nifti1Image(new_data, coverage_mask.affine)
    if output_path is not None and output_path != '':
        # add .nii.gz extension to the output_path if it does not have it
        if not output_path.endswith('.nii.gz'):
            output_path = output_path + '.nii.gz'
        nib.save(new_nii, output_path)
    return new_nii


def split_labels(labels_img, output_folder=None):
    """
    Split a labeled NIfTI image into individual images, each containing one label.

    Parameters
    ----------
    labels_img : Nifti1Image
        The labeled NIfTI image.
    output_folder : str, optional
        Directory to save the split label images.

    Returns
    -------
    list of Nifti1Image
        A list of NIfTI images, one for each label in the input image.
    """
    if not isinstance(labels_img, nib.Nifti1Image):
        raise TypeError('labels_img must be an instance of nibabel.Nifti1Image')

    data = labels_img.get_fdata()
    affine = labels_img.affine
    o_max = np.amax(data)

    label_img_list = []
    if output_folder is not None:
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
    for i in np.arange(1, o_max + 1):
        label = np.array(np.where(data == i))
        mask = np.zeros(data.shape)
        mask[label[0, ], label[1, ], label[2, ]] = i
        nii_label = nib.Nifti1Image(mask, affine)
        label_img_list.append(nii_label)
        if output_folder is not None:
            path = os.path.join(output_folder, 'label_{}.nii.gz'.format(str(i)))
            nib.save(nii_label, path)
    return label_img_list


def print_imgs_avg_size(list_img):
    """
    Print the average size of the clusters in a list of NIfTI images.

    Parameters
    ----------
    list_img : list of Nifti1Image
        List of NIfTI images to analyze.
    """
    sizes = []
    for img in list_img:
        sizes.append(len(np.where(img.get_fdata())[0]))
    print('Mean size of the images: {}'.format(np.mean(sizes)))


def main():
    """
    Main entry point for the script. Parses command-line arguments and coordinates the clustering process.

    Handles various input scenarios (e.g., parcel size list, number of parcels), applies clustering,
    and saves the resulting NIfTI images and metadata.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(description='Generate matched synthetic lesions dataset')
    paths_group = parser.add_mutually_exclusive_group(required=True)
    paths_group.add_argument('-p', '--input_path', type=str, help='Root folder of the lesion dataset')
    paths_group.add_argument('-li-', '--input_list', type=str,
                             help='Text file containing the list of lesion files')
    paths_group.add_argument('-m', '--mask', type=str,
                             help='Region where the synthetic lesions will be generated')

    parser.add_argument('-o', '--output', type=str, help='Output folder')
    parser.add_argument('-fwhm', '--smoothing_param', type=int, default=0,
                        help='FWHM parameter to nilearn smooth_img function')
    parser.add_argument('-thr', '--smoothing_threshold', type=float, default=0.5,
                        help='Threshold applied on the smoothing')
    parser.add_argument('--random_state', type=int, default=None,
                        help='Fix random seed for reproducibility')
    parser.add_argument('--method', type=str_to_lower, default='KMeans',
                        choices=['kmeans', 'compactor'],
                        help='Clustering method to use: "KMeans", "compactor". Default is "KMeans".')
    parser.add_argument('-c', '--contiguous', action='store_true',
                        help='Compactor option: force the clusters to be contiguous. If set, clusters will only include'
                             ' spatially connected voxels.')

    # New arguments for parcel_size_list
    # Create a mutually exclusive group for parcel_size_list and num_parcel_list
    parcel_group = parser.add_mutually_exclusive_group(required=True)
    parcel_group.add_argument('-rsl', '--parcel_size_list', type=str,
                              help='Comma-separated list of parcel sizes, '
                                   'or path to file containing the list of parcel sizes')
    parcel_group.add_argument('-np', '--num_parcel_list', type=str,
                              help='Number of parcels to generate')
    parcel_group.add_argument('-cs', '--custom_sizes', type=str,
                    help='Comma-separated list of custom sizes for the parcels.')
    # Add the strategy argument only for the parcel_size case
    parser.add_argument('--strategy', type=str, choices=['fixed_size', 'balanced_size'],
                        default='fixed_size',
                        help="Strategy for parceling with compactor when using parce_size "
                             "('fixed_size', 'balanced_size'). Default is 'fixed_size'.")

    args = parser.parse_args()
    args.output = os.path.abspath(args.output)
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    if args.random_state is not None:
        np.random.seed(args.random_state)
        random.seed(args.random_state)

    if args.num_parcel_list is not None:
        strategy = 'equal_size'  # Automatically use 'equal_size' strategy when num_parcel_list is specified
    else:
        strategy = args.strategy  # Use the strategy provided for parcel_size

    # Process the parcel_size_list or num_parcel_list argument
    custom_sizes = None
    if args.parcel_size_list:
        if os.path.exists(args.parcel_size_list):
            parcel_size_list = file_to_list(args.parcel_size_list)
        else:
            parcel_size_list = [s.strip() for s in args.parcel_size_list.split(',')]
        parcel_size_list = [int(s) for s in parcel_size_list]
        num_parcel_list = [None] * len(parcel_size_list)
    elif args.num_parcel_list:
        if os.path.exists(args.num_parcel_list):
            num_parcel_list = file_to_list(args.num_parcel_list)
        else:
            num_parcel_list = [s.strip() for s in args.num_parcel_list.split(',')]
        num_parcel_list = [int(s) for s in num_parcel_list]
        parcel_size_list = [None] * len(num_parcel_list)
    else:
        if os.path.exists(args.custom_sizes):
            custom_sizes = file_to_list(args.custom_sizes)
        else:
            custom_sizes = [s.strip() for s in args.custom_sizes.split(',')]
        custom_sizes = [int(s) for s in custom_sizes]
        parcel_size_list = [None]
        num_parcel_list = [None]
        if args.method != 'compactor':
            raise ValueError('Custom sizes can only be used with the compactor method')

    # if args.parcel_size_list is used the suffix will be parcel_size_{parcel_size}
    if args.parcel_size_list:
        output_subfolder_suffix = 'parcel_size'
    elif args.num_parcel_list:
        output_subfolder_suffix = 'num_parcels'
    else:
        output_subfolder_suffix = 'custom_sizes'

    if args.mask is not None:
        args.mask = os.path.abspath(args.mask)
        if not os.path.exists(args.mask):
            raise ValueError('The mask {} does not exist'.format(args.mask))
        coverage_mask = nib.load(args.mask)
    else:
        if args.input_path is not None:
            les_list = [os.path.join(args.input_path, f) for f in os.listdir(args.input_path)]
        else:
            if not os.path.exists(args.input_list):
                raise ValueError(args.input_list + ' does not exist.')
            if args.input_list.endswith('.csv'):
                with open(args.input_list, 'r') as csv_file:
                    les_list = []
                    for row in csv.reader(csv_file):
                        if len(row) > 1:
                            les_list += [r for r in row]
                        else:
                            les_list.append(row[0])
            else:
                # default delimiter is ' ', it might need to be changed
                les_list = np.loadtxt(args.input_list, dtype=str, delimiter=' ')
        les_list = [os.path.abspath(f) for f in les_list if is_nifti(f)]
        coverage_mask = create_coverage_mask(les_list)
        nib.save(coverage_mask, os.path.join(args.output, 'coverage_mask.nii.gz'))
    thr = args.smoothing_threshold
    # match +-10% size random in the pool
    parcels_size_dict = {}
    for parcel_sizes, number_parcels in zip(parcel_size_list, num_parcel_list):
        if output_subfolder_suffix == 'parcel_size':
            output_subfolder = Path(args.output, f'{output_subfolder_suffix}_{parcel_sizes}')
        elif output_subfolder_suffix == 'num_parcels':
            output_subfolder = Path(args.output, f'{output_subfolder_suffix}_{number_parcels}_parcels')
        else:
            output_subfolder = Path(args.output, f'{output_subfolder_suffix}')
        if parcel_sizes is not None or custom_sizes is not None:
            parcel_sizes = int(parcel_sizes) if parcel_sizes is not None else custom_sizes
            # if we use custom sizes we just say "custom sizes" in the print
            string_end = f'custom sizes' if custom_sizes is not None else f'{parcel_sizes}'
            print(f'Running {args.method} with parcel size: {string_end}')
            parcels_img = create_parcel_set(
                coverage_mask, parcel_sizes=parcel_sizes,
                output_path=output_subfolder,
                method=args.method, strategy=strategy
            )
        else:
            print(f'Running {args.method} with num_parcels = {number_parcels}')
            parcels_img = create_parcel_set(
                coverage_mask, num_parcels=number_parcels,
                output_path=output_subfolder,
                method=args.method, strategy=strategy
            )
        if parcels_img is None:
            print('cluster size too big compared to the mask')
            continue
        parcel_img_list = split_labels(parcels_img)
        if args.smoothing_param > 0:
            smoothed_label_list = [nilearn.image.smooth_img(label_img, args.smoothing_param)
                                   for label_img in parcel_img_list]
            smoothed_thr_label_list = [threshold_img(nii, thr) for nii in smoothed_label_list]
            smoothed_thr_binarized_label_list = [nilearn.image.math_img('img > {}'.format(thr), img=img)
                                                 for img in smoothed_thr_label_list]
            smoothed_thr_binarized_masked_label_list = [
                intersect_masks(
                    [nii, coverage_mask], 1, True)
                for nii in smoothed_thr_binarized_label_list]
            final_label_list = smoothed_thr_binarized_masked_label_list
        else:
            final_label_list = parcel_img_list

        print_imgs_avg_size(final_label_list)
        for parcel in final_label_list:
            parcel_data = parcel.get_fdata()
            parcel_sizes = len(np.where(parcel_data)[0])
            parcel_max = int(np.max(parcel_data))

            file_name = f'parcel_{parcel_sizes}_cluster{parcel_max}.nii.gz'
            file_path = Path(output_subfolder, file_name)
            if parcel_sizes in parcels_size_dict:
                parcels_size_dict[parcel_sizes].append(str(file_path))
            else:
                parcels_size_dict[parcel_sizes] = [str(file_path)]

            nib.save(parcel, file_path)
        with open(Path(output_subfolder, '__parcels_dict.json'), 'w+') as out_file:
            json.dump(parcels_size_dict, out_file, indent=4)


if __name__ == '__main__':
    """
    # KMeans with Parcel Size List
    parcitron -p "$path" -o "${output_path}/KMeans_parcel_sizes" --method KMeans -rsl 30000,50000 --random_state 42

    # KMeans with Number of Parcels
    parcitron -p "$path" -o "${output_path}/KMeans_num_parcels" --method KMeans -np 50 --random_state 42

    # Compactor with Fixed Size (Non-Contiguous)
    parcitron -p "$path" -o "${output_path}/Compactor_fixed_noncontig" --method compactor -rsl 30000 --strategy fixed_size --random_state 42

    # Compactor with Fixed Size (Contiguous)
    parcitron -p "$path" -o "${output_path}/Compactor_fixed_contig" --method compactor -rsl 30000 --strategy fixed_size --contiguous --random_state 42

    # Compactor with Balanced Size (Non-Contiguous)
    parcitron -p "$path" -o "${output_path}/Compactor_balanced_noncontig" --method compactor -rsl 30000 --strategy balanced_size --random_state 42

    # Compactor with Balanced Size (Contiguous)
    parcitron -p "$path" -o "${output_path}/Compactor_balanced_contig" --method compactor -rsl 30000 --strategy balanced_size --contiguous --random_state 42
    
    # Compactor with Number of Parcels (Non-Contiguous)
    parcitron -p "$path" -o "${output_path}/Compactor_num_parcels_noncontig" --method compactor -np 20,30,50,100 --random_state 42
    
    # Compactor with Number of Parcels (Contiguous)
    parcitron -p "$path" -o "${output_path}/Compactor_num_parcels_contig" --method compactor -np 50 --contiguous --random_state 42

    # Compactor with Custom Sizes (Contiguous)
    parcitron -p "$path" -o "${output_path}/Compactor_custom_sizes_contig" --method compactor -cs 2000,4000,5000,2500 --contiguous --random_state 42

    # Compactor with Custom Sizes (Non-Contiguous)
    parcitron -p "$path" -o "${output_path}/Compactor_custom_sizes_noncontig" --method compactor -cs 2000,4000,5000,2500 --random_state 42

    """
    main()
