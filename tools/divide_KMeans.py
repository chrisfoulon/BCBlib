# -*- coding: utf-8 -*-

import os
import math

import numpy as np

import nibabel as nib
from nilearn.image import math_img, threshold_img
from nilearn.masking import intersect_masks
from sklearn.cluster import KMeans

""" People will never want to ROIze only the seed and look at the connectivity
voxel-wise in the target ? """

def divide(img, ROIs_size):#, res_path):
    """
    """
    data = img.get_data()
    affine = img.affine
    pre_coords = np.array(np.where(data))
    coords = np.array([[pre_coords[0][i], pre_coords[1][i],
               pre_coords[2][i]] for i in np.arange(0, len(pre_coords[0]))])

    # number of clusters to create
    k = int(math.floor(len(coords) / ROIs_size))

    print('There are ' + str(np.shape(coords)[0]) + ' voxel')
    print('With the chosen ROI size of ' + str(ROIs_size) +
          ' voxels there will be ' + str(k) + ' ROIs')
    print(' ')
    print('I need to create the seed ROIs.')
    print('This might take some time for large seed regions...')

    ROIlabels = KMeans(n_clusters=k, n_init=10).fit_predict(coords)

    mask = np.zeros(data.shape)
    for i in np.arange(k):
        ind = np.where(ROIlabels==i)
        mask[coords[ind,0],
             coords[ind,1], coords[ind,2]] = i + 1

    img_ROIs = nib.Nifti1Image(mask, affine)
    # nib.save(img_ROIs, res_path)

    return img_ROIs#, ROIlabels

def increase_numbers(img, o_max):
    # WARNING, the image has to contain some non-zero integer voxels
    # We add it to the img
    img = math_img("img1 +" + str(o_max), img1 = img)
    # Now we need to threshold the image to come back to zeros
    # outside of the mask
    img = threshold_img(img, o_max + 1)
    return img


def binarise(path, res_path):
    img = nib.load(path)
    data = img.get_data()
    affine = img.affine
    coords = np.array(np.where(data))
    res_data = data + 0
    res_data[res_data != 0] = 1
    img_res = nib.Nifti1Image(res_data, affine)
    nib.save(img_res, res_path)

def nifti_bin(nii):
    data = nii.get_data() + 0
    data[data != 0] = 1
    return nib.Nifti1Image(data, nii.affine)

def roization(seed_path, target_path, ROIs_size, res_folder):
    """ We binarise the seed and the target, then we will ROIze the seed and
    the target-minus-seed images. The result will be a ROIzed seed image and
    the addition of those image with the ROIzation of the target-minus-seed
    image which will be the ROIzed target image
    """
    bn_seed = 'ROIs_' + os.path.basename(seed_path)
    bn_target = 'ROIs_' + os.path.basename(target_path)
    seed_img = nib.load(seed_path)
    target_img = nib.load(target_path)
    # We binarise
    seed_bin = nifti_bin(seed_img)
    target_bin = nifti_bin(target_img)
    roiz_seed_path = os.path.join(res_folder, bn_seed)
    roiz_tar_path = os.path.join(res_folder, bn_target)

    # Build the overlap between seed and target
    overlap = intersect_masks(seed_bin, target_bin, threshold=1,
                              connected=False)
    is_overlapping = np.unique(overlap.get_data())
    ffffffffuuuuuuuuuuuuuuu i need to change the labels because seeds and target
    labels start at 1 but if I do the addition I will have 2 clusters with 1 and
    ....
    # "Seed != Target && Overlap == {0}"
    if len(is_overlapping) == 1:
        # then we can divide seed and target separatly
        roized_seed = divide(seed_bin, ROIs_size)
        roized_target = divide(t_m_s, ROIs_size)
    else:
        seed_eq_overlap = math_img("i1 - i2", i1=overlap, i2=seed_bin)
        # Seed == Overlap
        if len(np.unique(seed_eq_overlap)) == 1:
            target_eq_overlap = math_img("i1 - i2", i1=overlap, i2=target_bin)
            # "Seed == Target"
            if len(np.unique(target_eq_overlap)) == 1:
                roized_seed = divide(seed_bin, ROIs_size)
                roized_target = roized_seed
            # "Seed == Overlap && Target > Seed"
            else:
                roized_seed = divide(seed_bin, ROIs_size)
                # Target without the seed region
                t_m_s = math_img("img1 - img2", img1 = target_bin,
                                 img2 = seed_bin)
                roiz_t_m_s = divide(t_m_s, ROIs_size)
                # We calculate the max of the roized_seed
                o_max = np.amax(roized_seed.get_data())
                roiz_t_m_s = increase_numbers(roiz_t_m_s, o_max)
                roized_target = math_img("img1 + img2", img1 = roized_seed,
                                         img2 = roiz_t_m_s)
        # "Target != Seed && Overlap != Seed && Overlap != Target"
        else:
            # Target without the overlap
            t_m_o = math_img("img1 - img2", img1 = target_bin,
                             img2 = overlap)
            # Seed without the overlap
            s_m_o = math_img("img1 - img2", img1 = seed_bin,
                             img2 = overlap)
            roiz_s_m_o = divide(s_m_o, ROIs_size)
            roiz_t_m_o = divide(t_m_o, ROIs_size)
            roiz_overlap = divide(overlap, ROIs_size)
            o_max = np.amax(roiz_overlap.get_data())
            roiz_s_m_o = increase_numbers(roiz_s_m_o, o_max)
            roiz_t_m_o = increase_numbers(roiz_t_m_o, o_max)
            roized_seed = math_img("img1 + img2", img1 = roiz_overlap,
                                     img2 = roiz_s_m_o)
            roized_target = math_img("img1 + img2", img1 = roiz_overlap,
                                     img2 = roiz_t_m_o)

    nib.save(roized_seed, roiz_seed_path)
    nib.save(roized_target, roiz_tar_path)

    return roized_seed, roized_target


target = '/data/BCBlab/Data/tests/targetMask.nii.gz'
seed = '/data/BCBlab/Data/tests/pref_seedMask.nii.gz'
roization(seed, target, 128, '/data/BCBlab/Data/tests')

"""Tests sur le cerveau de Leonardo:
ATTENTION: il faut d'abord masquer avec HemiL
Whiteribbon : on le masque avec HEmiL == > target
Et ensuite on le masque avec FrontalL ==> seed
ROIs_size = 3
"""
