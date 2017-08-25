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

def divide(img, ROIs_size, nb_run=10):
    #, res_path):
    """
    """
    data = img.get_data()
    affine = img.affine
    pre_coords = np.array(np.where(data))
    coords = np.array([[pre_coords[0][i], pre_coords[1][i],
               pre_coords[2][i]] for i in np.arange(0, len(pre_coords[0]))])

    # number of clusters to create
    k = int(math.floor(len(coords) / ROIs_size))

    print('There are ' + str(len(coords)) + ' voxel')
    print('With the chosen ROI size of ' + str(ROIs_size) +
          ' voxels there will be ' + str(k) + ' ROIs')
    print(' ')
    print('I need to create the seed ROIs.')
    print('This might take some time for large seed regions...')
    # , n_jobs=-2 (to use every proc)
    km = KMeans(n_clusters=k, n_init=nb_run)
    ROIlabels = km.fit_predict(coords)

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
    overlap = intersect_masks([seed_bin, target_bin], threshold=1.0,
                              connected=False)
    # nib.save(overlap, os.path.join(res_folder, "overlap.nii.gz"))
    is_overlapping = np.unique(overlap.get_data())

    # "Seed != Target && Overlap == {0}"
    if len(is_overlapping) == 1:
        # then we can divide seed and target separatly
        roized_seed = divide(seed_bin, ROIs_size)
        roized_target = divide(t_m_s, ROIs_size)
        print("Case 1")
    else:
        seed_eq_overlap = math_img("i1 - i2", i1=overlap, i2=seed_bin)
        # nib.save(seed_eq_overlap, os.path.join(res_folder, "o_s.nii.gz"))
        # Seed == Overlap
        if len(np.unique(seed_eq_overlap.get_data())) == 1:
            target_eq_overlap = math_img("i1 - i2", i1=overlap, i2=target_bin)
            # nib.save(target_eq_overlap, os.path.join(res_folder, "o_t.nii.gz"))
            # "Seed == Target"
            if len(np.unique(target_eq_overlap.get_data())) == 1:
                roized_seed = divide(seed_bin, ROIs_size)
                roized_target = roized_seed
                print("Case 2")
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
                print("Case 3")
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
            print("Case 4")

    nib.save(roized_seed, roiz_seed_path)
    nib.save(roized_target, roiz_tar_path)

    print("Roization done!")

    return roized_seed, roized_target


# seed = '/data/BCBlab/Data/tests/seed_thr300.nii.gz'
# target = '/data/BCBlab/Data/tests/s01M_LH_targetMask.nii.gz'
# roization(seed, target, 128, '/data/BCBlab/Data/tests')
# seed = '/data/BCBlab/Data/Parcellotron3000/res_divide/FrontalL_seed.nii.gz'
# target = '/data/BCBlab/Data/Parcellotron3000/res_divide/WhiteRibbon_masked.nii.gz'
# roization(seed, target, 5, '/data/BCBlab/Data/Parcellotron3000/res_divide')

"""Tests sur le cerveau de Leonardo:
ATTENTION: il faut d'abord masquer avec HemiL
Whiteribbon : on le masque avec HEmiL == > target
Et ensuite on le masque avec FrontalL ==> seed
ROIs_size = 3
"""


def seperate_ROIs(nii, res_folder, name):
    # if not os.path.exists(res_folder):
    #     os.mkdir(res_folder)
    # nii = nib.load(img)
    # affine = nii.affine
    data = nii.get_data()
    affine = nii.affine
    o_max = np.amax(data)

    # tt = []
    folder = os.path.join(res_folder, name)
    os.mkdir(folder)
    for i in np.arange(1, o_max + 1):
        clu = np.array(np.where(data == i))
        mask = np.zeros(data.shape)
        mask[clu[0,],
             clu[1,], clu[2,]] = i
        # tt.append(len(clu))
        img_ROIs = nib.Nifti1Image(mask, affine)
        path = os.path.join(folder, "clu" + str(i) + "_" + base)
        nib.save(mask, path)


    # print(np.unique(tt))
    # print("Length: " + str(len(tt)))
    # return tt

def testing():
    nii = nib.load("/data/experiments_BCBlab/555555_LH/Tracto_mat/30.625_seedROIs.nii.gz")
    os.path.basename(nii.get_filename())
    wd = "/data/BCBlab/Data/Parcellotron3000/res_divide/"
    aa = np.array(np.where(nii.get_data())).T
    type(aa[0])
    folders = [d for d in os.listdir(wd) if os.path.isdir(os.path.join(wd, d))]
    seperate_ROIs(nii, "/data/BCBlab/Data/Parcellotron3000/res_divide/tt")
    folders = sorted(folders)
    for d in folders:
        nii = nib.load(os.path.join(wd, os.path.join(d, "ROIs_seed_FrontalL.nii.gz")))
        res_f = os.path.join(wd, os.path.join(d, "tt"))
        seperate_ROIs(nii, res_f)

    p_s = os.path.join(wd, "tt/FrontalL_seed.nii.gz")
    p_t = os.path.join(wd, "tt/WhiteRibbon_masked.nii.gz")

    roization(p_s, p_t, 5, os.path.join(wd, 'tt'))
    ts = nib.load(os.path.join(wd, "tt/ROIs_FrontalL_seed.nii.gz"))
    tt = nib.load(os.path.join(wd, "ROIs_WhiteRibbon_masked.nii.gz"))
    t_s = seperate_ROIs(ts)
    t_t = seperate_ROIs(tt)
    for i in np.arange(1, len(np.unique(t_s))):
        w = np.where(t_s == i)
        arr = np.array(w).T
        print(len(arr))
    for i in np.arange(1, len(np.unique(t_t))):
        w = np.where(t_t == i)
        arr = np.array(w).T
        print(len(arr))


import sys

res = sys.argv[3]
seed = nib.load(os.path.join(res, sys.argv[1]))
target = nib.load(os.path.join(res, sys.argv[2]))

filename = seed.get_filename()
base = os.path.basename(filename)
ind = base.index('.')
seed_name = base[:ind]
filename = target.get_filename()
base = os.path.basename(filename)
ind = base.index('.')
target_name = base[:ind]

roized_seed = divide(seed, 3)
roized_target = divide(target, 3)
# roization(seed, target, 5, res)
seperate_ROIs(roized_seed, res, seed_name)
seperate_ROIs(roized_target, res, target_name)

def test_KMeans(img_name):
    wd = "/data/BCBlab/Data/Parcellotron3000/res_divide/"
    nii = nib.load(os.path.join(wd, "tt/" + img_name))
    for i in [5, 10, 15, 20, 50, 100]:
        roized = divide(nii, 5, i)
        print("##########TRIAL WITH " + str(i), " REPETITIONS ############")
        tt = seperate_ROIs(roized)
        for i in np.arange(1, len(np.unique(tt))):
            w = np.where(tt == i)
            arr = np.array(w).T
            print("[" + str(i) + "]" + str(len(arr)))
        print("################""TRIAL'S END##############")

    test_KMeans("FrontalL_seed.nii.gz")
    print("WHIIIIIIIIIIIITE RIBBBBBBBBBBBOOOOOOOONNNN !!!!!")
    print("WHIIIIIIIIIIIITE RIBBBBBBBBBBBOOOOOOOONNNN !!!!!")
    print("WHIIIIIIIIIIIITE RIBBBBBBBBBBBOOOOOOOONNNN !!!!!")
    test_KMeans("ROIs_WhiteRibbon_masked.nii.gz")

    from scipy.spatial import distance_matrix
    m1 = [[4,4,1],[3,5,1]]
    m1[0:2]
    m2 = [[3,7,1],[3,5,1]]
    distance_matrix(m1,m2)
    import math
    math.sqrt(10)
    math.sqrt(2)




def find_seed(coords, dir):
    pass

def gather_round(seed, coords, size):
    """ Find the nearest voxels from the seed and return an array of their
    coordinates
    Parameters
    ----------
    seed: np.array
        array([x,y,z]) an array of the coordinates of the seed voxel
        of the cluster
    coords: np.array (coordinates of each voxel on lines)
        coordinates of voxels in the mask
    size: int
        size of the cluster. If there isn't enough coordinates, the function
        will still return a cluster but with less voxels
    Returns
    -------
    np.array
        array with the coordinates of the voxels in the cluster of neighbors.
        So it will contain seed
    """
    dist_mat = distance_matrix(np.array(seed), coords)
    dist_mat = sorted(dist_mat)
    return dist_mat[0:size]

def divide_compactor(img, size):
    pass
