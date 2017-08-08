# -*- coding: utf-8 -*-

import os
import math
import numpy as np

import nibabel as nib
from nilearn.image import math_img
from sklearn.cluster import KMeans

""" People will never want to ROIze only the seed and look at the connectivity
voxel-wise in the target ? """

def divide(img, ROIs_size, res_path):
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
    nib.save(img_ROIs, res_path)

    return img_ROIs#, ROIlabels


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

target = '/data/experiments_BCBlab/test_COBRA/S1/Tracto_4D/s01M_LH_targetMask.nii.gz'
seed = '/data/experiments_BCBlab/test_COBRA/S1/Tracto_4D/s01M_LH_seedROIs.nii.gz'
res_path = '/data/BCBlab/Data/test_divide.nii.gz'
bin_path = '/data/BCBlab/Data/test_bin.nii.gz'
bin_seed = '/data/BCBlab/Data/target_bin.nii.gz'
bin_tar = '/data/BCBlab/Data/seed_bin.nii.gz'
# labels = divide(target, 128, res_path)
"""IT WILL BE BETTER TO USE ONLY NIIFTI1IMAGES !!! Instead of save nifti files
"""
binarise(seed, bin_seed)
binarise(target, bin_tar)
tar_sub = math_img("img1 - img2", img1 = bin_tar, img2 = bin_seed)
tar_sub

def roization(seed_path, target_path, ROIs_size, res_folder):
    """ We binarise the seed and the target, then we will ROIze the seed and
    the target-minus-seed images. The result will be a ROIzed seed image and
    the addition of those image with the ROIzation of the target-minus-seed
    image which will be the ROIzed target image
    """
    bn_seed = os.path.basename(seed_path)
    bn_target = os.path.basename(target_path)
    seed_img = nib.load(seed_path)
    target_img = nib.load(target_path)
    # We binarise
    seed_bin = nifti_bin(seed_img)
    target_bin = nifti_bin(target_img)
    t_m_s = math_img("img1 - img2", img1 = target_bin,
                               img2 = seed_bin)

    roized_seed = divide(seed_bin, ROIs_size,
                         os.path.join(bn_seed, res_path))
    roiz_tar_path = os.path.join(bn_target, res_path)
    roized_t_m_s = divide(t_m_s, ROIs_size, roiz_tar_path)

    # We create the roization of the target
    roized_target = math_img("img1 + img2", img1 = roized_t_m_s,
                               img2 = roized_seed)
    nib.save(roized_target, roiz_tar_path)

    return roized_seed, roized_target

roization(seed, target, 128, '/data/BCBlab/Data/')

""" WHAT DO WE DO IF TARGET == SEED ?!?!?!?!?!?!"""
