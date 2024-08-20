"""
Generate matched synthetic lesions dataset

Authors: Chris Foulon & Michel Thiebaut de Scotten
"""
import os
import argparse
import random
from pathlib import Path

import numpy as np
import json
import csv

import nibabel as nib
import nilearn
from nilearn.masking import compute_multi_background_mask, intersect_masks
from nilearn.image import threshold_img
from sklearn.cluster import KMeans

from bcblib.tools.general_utils import file_to_list
from bcblib.tools.divide_mask import divide_compactor


def determine_parcels(M, N=None, S=None, strategy="equal_size"):
    """
    Determine the number of parcels and their sizes based on the desired strategy.

    Parameters
    ----------
    M : int
        Total number of points to be divided into parcels.
    N : int, optional
        Number of parcels desired.
    S : int, optional
        Desired size of each parcel.
    strategy : str, optional
        Strategy to use for parceling.
        Options are:
        - "equal_size": (default) Create exactly N parcels of roughly (+- 1) equal size.
        - "fixed_size": Create parcels of size S, with a leftover parcel if necessary.
        - "balanced_size": Create parcels of size close to S, balancing the sizes to avoid a very small leftover parcel.

    Returns
    -------
    list of int
        A list containing the sizes of each parcel.
    """
    if strategy == "equal_size":
        if N is None:
            raise ValueError("N must be provided for the 'equal_size' strategy.")
        base_size = M // N
        remainder = M % N
        sizes = [base_size + 1 if i < remainder else base_size for i in range(N)]

    elif strategy == "fixed_size":
        if S is None:
            raise ValueError("S must be provided for the 'fixed_size' strategy.")
        N = M // S
        remainder = M % S
        sizes = [S] * N
        if remainder > 0:
            sizes.append(remainder)

    elif strategy == "balanced_size":
        if S is None:
            raise ValueError("S must be provided for the 'balanced_size' strategy.")
        N = M // S
        base_size = M // N
        remainder = M % N
        sizes = [base_size + 1 if i < remainder else base_size for i in range(N)]

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return sizes


def create_coverage_mask(image_path_list):
    nii_list = []

    for f in image_path_list:
        if not os.path.isfile(f):
            raise ValueError('{} is not an existing file'.format(f))
        if not nii_list:
            nii_list = [nib.load(f)]
        else:
            nii_list.append(nib.load(f))
    return compute_multi_background_mask(nii_list, threshold=0, connected=False, n_jobs=-1)


def create_parcel_set(coverage_mask, parcel_size=None, num_parcels=None, output_path=None, method='KMeans',
                      strategy="equal_size"):
    if output_path is not None:
        output_path = os.path.abspath(output_path)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
    mask_coord = np.array(np.where(coverage_mask.get_fdata())).T

    # Determine the number of parcels and sizes using the specified strategy
    if num_parcels is not None:
        sizes = determine_parcels(len(mask_coord), N=num_parcels, strategy=strategy)
    elif parcel_size is not None:
        sizes = determine_parcels(len(mask_coord), S=parcel_size, strategy=strategy)
    else:
        raise ValueError("Either parcel_size or num_parcels must be provided.")
    print(f'Method = {method}')
    if method == 'compactor':
        print(f'Running divide_compactor with sizes = {sizes}')
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
            new_data[c] = kmeans_labels_img[ind] + 1

    # set new_data dtype to the same as the coverage_mask
    new_data = new_data.astype(coverage_mask.get_fdata().dtype)
    new_nii = nib.Nifti1Image(new_data, coverage_mask.affine)
    if output_path is not None and output_path != '':
        nib.save(new_nii, output_path)
    return new_nii


def split_labels(labels_img, output_folder=None):
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
        print(f'Label {i} has {len(label[0])} voxels')
        mask = np.zeros(data.shape)
        mask[label[0, ], label[1, ], label[2, ]] = i
        print(f'Label {i} has {len(np.where(mask)[0])} voxels')
        nii_label = nib.Nifti1Image(mask, affine)
        label_img_list.append(nii_label)
        if output_folder is not None:
            path = os.path.join(output_folder, 'label_{}.nii.gz'.format(str(i)))
            nib.save(nii_label, path)
    return label_img_list


def print_imgs_avg_size(list_img):
    sizes = []
    for img in list_img:
        sizes.append(len(np.where(img.get_fdata())[0]))
    print('Mean size of the images: {}'.format(np.mean(sizes)))


def main():
    """

    Returns
    -------

    Examples:
    parcel_size_list = ['300000', '200000', '120000', '110000', '100000', '90000', '80000', '70000', '60000', '50000',
                     '40000', '30000', '20000', '10000', '9000', '8000', '7000', '6000', '5000', '4000', '3000', '2000',
                     '1000', '900', '800', '700', '600', '500', '400', '300', '200', '100', '35000', '25000', '15000']
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
    parser.add_argument('--method', type=str, default='KMeans',
                        choices=['KMeans', 'compactor'],
                        help='Clustering method to use: "KMeans" or "compactor". Default is "KMeans".')
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
    if args.parcel_size_list:
        if os.path.exists(args.parcel_size_list):
            parcel_size_list = file_to_list(args.parcel_size_list)
        else:
            parcel_size_list = [s.strip() for s in args.parcel_size_list.split(',')]
        parcel_size_list = [int(s) for s in parcel_size_list]
        num_parcel_list = [None] * len(parcel_size_list)
    else:
        if os.path.exists(args.num_parcel_list):
            num_parcel_list = file_to_list(args.num_parcel_list)
        else:
            num_parcel_list = [s.strip() for s in args.num_parcel_list.split(',')]
        num_parcel_list = [int(s) for s in num_parcel_list]
        parcel_size_list = [None] * len(num_parcel_list)

    # if args.parcel_size_list is used the suffix will be parcel_size_{parcel_size}
    if args.parcel_size_list:
        output_subfolder_suffix = 'parcel_size'
    else:
        output_subfolder_suffix = 'num_parcels'

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
        les_list = [os.path.abspath(f) for f in les_list]
        coverage_mask = create_coverage_mask(les_list)
        nib.save(coverage_mask, os.path.join(args.output, 'coverage_mask.nii.gz'))
    thr = args.smoothing_threshold
    # match +-10% size random in the pool
    parcels_size_dict = {}
    for parcel_size, number_parcels in zip(parcel_size_list, num_parcel_list):
        if output_subfolder_suffix == 'parcel_size':
            output_subfolder = Path(args.output, f'{output_subfolder_suffix}_{parcel_size}')
        else:
            output_subfolder =  Path(args.output, f'{output_subfolder_suffix}_{number_parcels}_parcels')
        if parcel_size is not None:
            print(f'Running {args.method} with parcel size = {parcel_size}')
            parcels_img = create_parcel_set(
                coverage_mask, parcel_size=int(parcel_size),
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
            parcel_size = len(np.where(parcel_data)[0])
            parcel_max = int(np.max(parcel_data))
            print(f'Parcel size: {parcel_size}, Max value in parcel: {parcel_max}')

            file_name = f'parcel_{parcel_size}_cluster{parcel_max}.nii.gz'
            file_path = Path(output_subfolder, file_name)
            if parcel_size in parcels_size_dict:
                parcels_size_dict[parcel_size].append(str(file_path))
            else:
                parcels_size_dict[parcel_size] = [str(file_path)]

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

    """
    main()
