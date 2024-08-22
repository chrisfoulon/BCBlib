# -*- coding: utf-8 -*-
"""
This script provides a method to cluster non-zero voxels in a 3D image (Nifti format) into groups of specified sizes. The clusters can be made contiguous, ensuring that the voxels in each cluster are spatially connected, or non-contiguous, where clusters consist of the nearest neighbors in the voxel space without enforcing spatial connectivity.

Key Features:
- **Random Seed Selection:** The initial seed voxel for clustering can be selected randomly or based on a fixed criterion.
- **Contiguous Clustering:** The algorithm can enforce contiguity in clusters, ensuring all voxels within a cluster are spatially connected.
- **Random Label Assignment:** Cluster labels can be assigned randomly or sequentially.
- **Custom Cluster Sizes:** The user can specify the exact size of each cluster.

Functions:
- `find_seed(coords, direc)`: Identifies the most distant voxel in a given direction from the list of coordinates.
- `find_next_seed(coords, tree)`: Randomly selects the next seed voxel from the remaining unclustered voxels.
- `gather_round(seed, coords, size, tree, contiguous_clusters)`: Collects the nearest voxels to a seed, optionally ensuring that the collected voxels form a contiguous cluster.
- `divide_compactor(img, sizes, random_labels=False, random_first_seed=False, contiguous_clusters=False)`: Divides the input image into clusters based on specified sizes, with options for random seed selection, contiguous clustering, and random label assignment.

Potential Edge Cases:
1. **Insufficient Contiguous Voxels:** If there are fewer contiguous voxels available than the specified cluster size, the script will issue a warning and create a smaller cluster. This can happen near the edges of the mask or in regions with sparse voxel density.
2. **Exact Cluster Sizes:** If the total number of voxels does not divide evenly by the specified sizes, the last cluster may have more or fewer voxels than intended.
3. **Random Seed Selection:** While random seed selection helps avoid biases, it may lead to less predictable cluster shapes, especially in the first few clusters.
4. **KDTree Limitations:** The script assumes isotropic voxel spacing; if the voxel grid is anisotropic, the KDTree distance calculations may not accurately reflect true spatial distances, potentially leading to elongated clusters.
5. **Remaining Voxels:** The script handles any leftover voxels after clustering by assigning them to a final cluster. If the number of remaining voxels is small, this cluster may be disproportionately small compared to the others.
"""
import heapq

import numpy as np
import random

import nibabel as nib
from scipy.ndimage import generate_binary_structure
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
    # dist, ind_sort = tree.query(coords, k=1)
    # return coords[ind_sort[0]]
    return coords[np.random.choice(len(coords))]


def gather_round(seed, coords, size, tree, contiguous_clusters=False):
    """
    Find the nearest voxels from the seed using KDTree and return an array of their indices.

    Parameters
    ----------
    seed : np.array
        An array representing the coordinates of the seed voxel for the current cluster.
    coords : np.array
        An array of coordinates of voxels in the mask.
    size : int
        The desired size of the cluster. If not enough coordinates are available, the function will return a smaller cluster.
    tree : KDTree
        KDTree built from the coordinates of the voxels.
    contiguous_clusters : bool, optional
        If True, enforce that the clusters are contiguous, meaning all voxels in a cluster must be spatially connected.

    Returns
    -------
    np.array
        An array of indices corresponding to the selected cluster voxels.
    np.array
        An array of coordinates corresponding to the selected cluster voxels.

    Notes
    -----
    When `contiguous_clusters` is True, the function uses a priority queue to ensure that voxels are added to the cluster
    based on their spatial proximity, starting from the seed and expanding outwards. If not enough contiguous voxels are
    found to meet the requested cluster size, a smaller cluster is returned.
    """
    if contiguous_clusters:
        cluster_coords = []
        visited = set()
        heap = []
        seed_index = tree.query(seed, k=1)[1][0]  # Get the seed index
        seed_distance = 0  # Distance of the seed to itself is 0
        heapq.heappush(heap, (seed_distance, seed_index, seed))
        visited.add(tuple(seed))

        while len(cluster_coords) < size and heap:
            dist, idx, current_point = heapq.heappop(heap)
            cluster_coords.append(current_point)

            neighbors = np.array(np.where(generate_binary_structure(3, 1))).T - 1 + current_point
            random.shuffle(neighbors)  # Randomize the order of the neighbors to ovoid a bias in the cluster shape
            for neighbor in neighbors:
                neighbor_tuple = tuple(neighbor)
                if (neighbor_tuple not in visited and
                        tuple(neighbor) in set(
                            map(tuple, coords))):  # Ensure the neighbor is within the original coordinates
                    neighbor_idx, neighbor_dist = tree.query(neighbor, k=1)
                    heapq.heappush(heap, (neighbor_dist[0], neighbor_idx[0], neighbor))
                    visited.add(neighbor_tuple)

        if len(cluster_coords) < size:
            print("Warning: Could not find enough contiguous voxels.")

        return np.array([coords.tolist().index(list(p)) for p in cluster_coords]), np.array(cluster_coords)
    else:
        size = min(size, len(coords))
        dist, ind_sort = tree.query(seed, k=size)
        return ind_sort, coords[ind_sort]


def divide_compactor(img, sizes, random_labels=False, random_first_seed=False, contiguous_clusters=False):
    """
    Cluster the non-zero voxels in a Nifti image into groups of specified sizes.

    Parameters
    ----------
    img : Nifti1Image
        The Nifti mask image containing non-zero voxels to cluster.
    sizes : list of int
        A list of integers specifying the size of each cluster.
    random_labels : bool, optional
        If True, assign cluster labels randomly. Otherwise, labels are assigned sequentially.
    random_first_seed : bool, optional
        If True, select the initial seed voxel randomly. Otherwise, start with the first voxel in the list.
    contiguous_clusters : bool, optional
        If True, enforce that the clusters are contiguous, meaning all voxels in a cluster must be spatially connected.

    Returns
    -------
    Nifti1Image
        A Nifti image with the same dimensions as `img`, where the voxels are labeled according to their cluster.

    Notes
    -----
    - If `random_labels` is True, the cluster labels are assigned in a random order.
    - If `random_first_seed` is True, the first seed voxel is chosen randomly from the available voxels.
    - If `contiguous_clusters` is True, the function will attempt to create clusters where all voxels are spatially
    connected. In cases where there are fewer contiguous voxels than required, the cluster will be smaller than the
    specified size.
    - The function handles any leftover voxels by assigning them to a final cluster.
    """
    # Extract coordinates of non-zero voxels
    coords = np.asarray(np.where(img.get_fdata())).T

    # Check if the mask is empty or too small
    if len(coords) == 0:
        raise ValueError("The mask is empty. No voxels to cluster.")

    if len(coords) < min(sizes):
        raise ValueError("The mask has fewer voxels than the smallest requested cluster size.")

    # Proceed with clustering if the mask is valid
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
            if random_first_seed:
                seed_index = np.random.choice(len(coords))
            else:
                seed_index = 0
            seed = coords[seed_index]
        else:
            # Find the nearest unclustered seed
            seed = find_next_seed(coords, tree)

        # Get both indices and coordinates, including the seed
        tmp_clu_indices, tmp_clu_coords = gather_round(seed, coords, size, tree, contiguous_clusters)

        # Use numpy indexing to assign labels
        res_data[tuple(tmp_clu_coords.T)] = label

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
