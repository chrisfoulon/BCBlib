# -*- coding: utf-8 -*-

import numpy as np
import random

import nibabel as nib
from scipy.spatial import KDTree


def find_seed(coords, direc):
    """ Find (one of) the most distant voxels in a given direction and
    return its coordinates
    Parameters
    ----------
    coords: np.array (coordinates of each voxel on lines)
        coordinates of voxels in the mask
    direc: int
        0: x
        1: -x
        2: y
        3: -y
        4: z
        5: -z
    Returns
    -------
    ext: [(int)x, (int)y, (int)z]
        the coordinates of the most extreme voxels in the direction direc
    """
    side = direc % 2
    axis = direc // 2
    # print("The direction is: " + ["+", "-"][side] + ["x", "y", "z"][axis])
    side_func = np.argmin if side == 0 else np.argmax
    ext = side_func(coords[:, axis])
    return coords[ext]


def find_next_seed(coords, tree):
    """
    Find the nearest unclustered voxel to the current cluster.

    Parameters
    ----------
    coords: np.array
        Array of coordinates of the remaining voxels in the mask.
    tree: KDTree
        KDTree built from the coordinates of the voxels.

    Returns
    -------
    np.array
        Coordinates of the next seed voxel.
    """
    # Find the nearest unclustered voxel
    dist, ind_sort = tree.query(coords, k=1)
    return coords[ind_sort[0]]


def gather_round(seed, coords, size, tree):
    """ Find the nearest voxels from the seed using KDTree and return an array of their indices.
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
        array with the indixes of the nearest voxels(in coords) from seed
        (the array contains seed)
    """
    # Adjust the size to the number of available points
    size = min(size, len(coords))
    print(f'Gathering {size} voxels around {seed}')
    dist, ind_sort = tree.query(seed, k=size)
    print(f'Closest voxels: {len(coords[ind_sort])}')
    return ind_sort, coords[ind_sort]


def divide_compactor(img, sizes, random_labels=False):
    # TODO add an option to force clusters contiguity
    #  (testing that the input mask is fully contiguous).
    # TODO Also add an option to allow for a random first seed.
    """ Cluster img in groups of a given number of neighbour voxels (Be sure to use a skull stripped image as the
    function divides all non-zero voxel)
    Parameters
    ----------
    img: Nifti1Image
        The nifti mask of non-zero voxels to cluster
    size: int
        The size of each cluster (The last cluster can have a lower number
        of voxels)
    random_labels: bool
        (default: False) generate the labels randomly or not
    Returns
    -------
    res_img: Nifti1Image
        An image with the same dimension than img and its voxels labelled with
        their cluster number
    """
    coords = np.asarray(np.where(img.get_fdata())).T
    res_data = np.zeros(img.shape)

    label_list = list(range(1, len(sizes) + 1))
    if random_labels:
        random.shuffle(label_list)

    tree = KDTree(coords)

    for clu_ind, (label, size) in enumerate(zip(label_list, sizes)):
        if len(coords) == 0:
            break

        if clu_ind == 0:
            # Start with a random point
            seed_index = np.random.choice(len(coords))
            seed = coords[seed_index]
        else:
            # Find the nearest unclustered seed
            seed = find_next_seed(coords, tree)

        # Get both indices and coordinates, including the seed
        tmp_clu_indices, tmp_clu_coords = gather_round(seed, coords, size, tree)

        # Use numpy indexing to assign labels
        res_data[tuple(tmp_clu_coords.T)] = label
        print(f'Number of voxels with label {label}: {len(res_data[res_data == label])}')

        # Remove the clustered points from coords
        coords = np.delete(coords, tmp_clu_indices, axis=0)

        # Rebuild the KDTree with the remaining points
        if len(coords) > 0:
            tree = KDTree(coords)

    # Handling remaining voxels if there are any left over
    if len(coords) > 0:
        last_label = max(label_list) + 1
        res_data[tuple(coords.T)] = last_label

    res_img = nib.Nifti1Image(res_data, img.affine)
    return res_img
