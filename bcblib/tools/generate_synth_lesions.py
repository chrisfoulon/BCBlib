"""
Generate matched synthetic lesions dataset

Authors: Chris Foulon & Michel Thiebaut de Scotten

DEPRECATION WARNING:
This script is deprecated and maintained only for reproducibility purposes as referenced in the scientific article.
Please use the updated tool 'parcitron' for generating synthetic lesions.
"""
import os
import argparse
import random
import numpy as np
import json
import csv

import nibabel as nib
import nilearn
from nilearn.masking import compute_multi_background_mask, intersect_masks
from nilearn.image import threshold_img
from sklearn.cluster import KMeans

from bcblib.tools.general_utils import file_to_list

# Set the filter to show the warning only once
warnings.simplefilter('once', DeprecationWarning)

# input: /data/Chris/lesionsFormated
def create_coverage_mask(image_path_list):
    warnings.warn("create_coverage_mask is deprecated, please use the updated tool 'parcitron'.", DeprecationWarning)

    nii_list = []

    for f in image_path_list:
        if not os.path.isfile(f):
            raise ValueError('{} is not an existing file'.format(f))
        if not nii_list:
            nii_list = [nib.load(f)]
        else:
            nii_list.append(nib.load(f))
    return compute_multi_background_mask(nii_list, threshold=0, connected=False, n_jobs=-1)


def create_parcel_set(coverage_mask, roi_size=None, num_parcels=None, output_path=None):
    warnings.warn("create_parcel_set is deprecated, please use the updated tool 'parcitron'.", DeprecationWarning)

    mask_coord = np.where(coverage_mask.get_fdata())
    mask_coord = [(mask_coord[0][i], mask_coord[1][i], mask_coord[2][i]) for i, _ in enumerate(mask_coord[0])]

    if num_parcels is not None:
        k = num_parcels
    elif roi_size is not None:
        k = int(np.floor(len(mask_coord) / roi_size))
    else:
        raise ValueError("Either roi_size or num_parcels must be provided.")
    if k == 0:
        return None
    print('Running KMeans with k = {}'.format(k))
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
    warnings.warn("split_labels is deprecated, please use the updated tool 'parcitron'.", DeprecationWarning)

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
    sizes = []
    for img in list_img:
        sizes.append(len(np.where(img.get_fdata())[0]))
    print('Mean size of the images: {}'.format(np.mean(sizes)))


def main():
    """

    Returns
    -------

    Examples:
    roi_size_list = ['300000', '200000', '120000', '110000', '100000', '90000', '80000', '70000', '60000', '50000',
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

    # New arguments for roi_size_list
    # Create a mutually exclusive group for roi_size_list and num_parcels
    roi_group = parser.add_mutually_exclusive_group(required=True)
    roi_group.add_argument('-rsl', '--roi_size_list', type=str,
                           help='Comma-separated list of parcel sizes, '
                                'or path to file containing the list of parcel sizes')
    roi_group.add_argument('-np', '--num_parcels', type=int,
                           help='Number of parcels to generate')

    args = parser.parse_args()
    args.output = os.path.abspath(args.output)
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)

    # Process the roi_size_list or num_parcels argument
    if args.roi_size_list:
        if os.path.exists(args.roi_size_list):
            roi_size_list = file_to_list(args.roi_size_list)
        else:
            roi_size_list = [s.strip() for s in args.roi_size_list.split(',')]
        roi_size_list = [int(s) for s in roi_size_list]
    elif args.num_parcels:
        roi_size_list = [None]  # roi_size will be None, and num_parcels will be used instead

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
    synth_lesion_size_dict = {}
    for s in roi_size_list:
        if s is not None:
            print('Running the KMeans with ROIsize = {}'.format(s))
            parcels_img = create_parcel_set(coverage_mask, roi_size=int(s),
                                            output_path=os.path.join(args.output, 'parcels_{}.nii.gz'.format(s)))
        else:
            print('Running the KMeans with num_parcels = {}'.format(args.num_parcels))
            parcels_img = create_parcel_set(coverage_mask, num_parcels=args.num_parcels,
                                            output_path=os.path.join(args.output, 'parcels_{}_parcels.nii.gz'.format(
                                                args.num_parcels)))
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
        else:
            smoothed_thr_binarized_label_list = parcel_img_list
        smoothed_thr_binarized_masked_label_list = [
            intersect_masks(
                [nii, coverage_mask], 1, True) for nii in smoothed_thr_binarized_label_list]
        print_imgs_avg_size(smoothed_thr_binarized_masked_label_list)
        for lesion in smoothed_thr_binarized_masked_label_list:
            lesion_size = len(np.where(lesion.get_fdata())[0])

            if lesion_size not in synth_lesion_size_dict:
                file_name = 'synth_les_{}.nii.gz'.format(lesion_size)
                file_path = os.path.join(args.output, file_name)
                synth_lesion_size_dict[lesion_size] = [file_path]
            else:
                file_name = 'synth_les_{}_{}.nii.gz'.format(lesion_size, len(synth_lesion_size_dict[lesion_size]))
                file_path = os.path.join(args.output, file_name)
                synth_lesion_size_dict[lesion_size].append(file_path)
            nib.save(lesion, file_path)
    with open(os.path.join(args.output, '__lesion_dict.json'), 'w+') as out_file:
        json.dump(synth_lesion_size_dict, out_file, indent=4)


if __name__ == '__main__':
    main()
