# -*- coding: utf-8 -*-

import math
import numpy as np

import nibabel as nib
import nilearn as nil
from sklearn.cluster import KMeans
# %%

def divide(im_path, ROIs_size, res_path):
    """
    """
    img = nib.load(im_path)
    data = img.get_data()
    affine = img.affine
    pre_coords = np.array(np.where(data))
    coords = np.array([[pre_coords[0][i], pre_coords[1][i],
               pre_coords[2][i]] for i in np.arange(0, len(pre_coords[0]))])

    print(str(coords.shape))
    # print(str(data[list(coords[2])]))
    # for c in coords:
    #     print(data[c])

    # number of clusters to create
    k = int(math.floor(len(coords) / ROIs_size))

    print('There are ' + str(np.shape(coords)[0]) + ' voxel')
    print('With the chosen ROI size of ' + str(ROIs_size) +
          ' voxels there will be ' + str(k) + ' ROIs')
    print(' ')
    print('I need to create the seed ROIs.')
    print('This might take some time for large seed regions...')

    ROIlabels = KMeans(n_clusters=k, n_init=10).fit_predict(coords)
    print(str(ROIlabels))
    print(str(ROIlabels.shape))

    mask = np.zeros(data.shape)
    for i in np.arange(k):
        ind = np.where(ROIlabels==i)
        mask[coords[ind,0],
             coords[ind,1], coords[ind,2]] = i + 1

    img_ROIs = nib.Nifti1Image(mask, affine)
    nib.save(img_ROIs, res_path)

    return ROIlabels

# %%
def binarise(path, res_path):
    img = nib.load(path)
    data = img.get_data()
    affine = img.affine
    coords = np.array(np.where(data))
    res_data = data + 0
    res_data[res_data != 0] = 1
    img_res = nib.Nifti1Image(res_data, affine)
    nib.save(img_res, res_path)

target = '/data/experiments_BCBlab/test_COBRA/S1/Tracto_4D/s01M_LH_targetMask.nii.gz'
seed = '/data/experiments_BCBlab/test_COBRA/S1/Tracto_4D/s01M_LH_seedROIs.nii.gz'
res_path = '/data/BCBlab/Data/test_divide.nii.gz'
bin_path = '/data/BCBlab/Data/test_bin.nii.gz'
bin_seed = '/data/BCBlab/Data/target_bin.nii.gz'
bin_tar = '/data/BCBlab/Data/seed_bin.nii.gz'
labels = divide(target, 128, res_path)
"""IT WILL BE BETTER TO USE ONLY NIIFTI1IMAGES !!! Instead of save nifti files
"""
binarise(seed, bin_seed)
binarise(target, bin_tar)
tar_sub = nil.image.math_img("img1 - img2", img1 = bin_tar, img2 = bin_seed)
tar_sub
