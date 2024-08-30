import random
from pathlib import Path
from functools import partial

import numpy as np
from scipy import ndimage
from scipy.ndimage import label, generate_binary_structure
from scipy.spatial.distance import cdist

from bcblib.tools.arrays_utils import coord_in_array
from bcblib.tools.blob import Blob
from bcblib.tools.cell import Cell
from bcblib.tools.general_utils import partition_values_to_sizes_with_margin
from bcblib.tools.misc import calculate_hypersphere_radius

"""
https://softologyblog.wordpress.com/2019/12/28/3d-cellular-automata-3/
Has good ideas and some rules (+ explanation on the rules)
The ways to colour the cells:
"RGB Cube. Convert the XYZ coordinates to RGB color values.

Color Palette. Map the distance of each cube from the center to a color palette.

White Only. Color all cubes white. This can be useful when you have multiple colored lights.

State Shading. Color cells based on which state they are in. Shaded between yellow and red for the example movie.

Neighborhood Density. Color based on how dense each cell and its nearest neighboring cells are."

We could add a colour scheme for multi states CA where the colour depends on the state of the cell
"""


def initialize_cell_array_with_mask(shape, mask, seed_coords, seed_values, seed_shape=None, spawnable_value=0,
                                    non_spawnable_value=-1):
    """
    Initialize the cell array with a mask and seed values. Each seed can take a specified shape.

    Parameters
    ----------
    shape : tuple
        The shape of the cell array.
    mask : np.ndarray
        A binary mask indicating spawnable areas (1) and non-spawnable areas (0).
    seed_coords : list of tuples
        The coordinates of the seeds.
    seed_values : list of ints
        The values (cluster indices) for each seed.
    seed_shape : np.ndarray, optional
        A binary array defining the shape of each seed. The shape must fit within the cell array.
    spawnable_value : int, optional
        The value for spawnable cells.
    non_spawnable_value : int, optional
        The value for non-spawnable cells.

    Returns
    -------
    np.ndarray
        The initialized cell array.
    """
    def create_cell(x, y, z):
        if mask[x, y, z] == 1:
            return Cell(spawnable_value)
        else:
            return Cell(non_spawnable_value)

    vectorized_create_cell = np.vectorize(create_cell, otypes=[object])
    cell_array = vectorized_create_cell(*np.indices(shape))

    # Handle the placement of seeds with the specified shape
    for seed_coord, seed_value in zip(seed_coords, seed_values):
        if seed_shape is not None:
            seed_shape_coords = np.argwhere(seed_shape)
            for offset in seed_shape_coords:
                target_coord = tuple(np.array(seed_coord) + offset - np.array(seed_shape.shape) // 2)
                if coord_in_array(target_coord, cell_array) and mask[target_coord] == 1:
                    current_cell = cell_array[target_coord]
                    if current_cell.get_state() == spawnable_value:
                        current_cell.set_next_state(seed_value, it_time=0)
                        current_cell.update_state()
        else:
            # If no shape is specified, just place a single cell
            cell_array[seed_coord].set_next_state(seed_value, it_time=0)
            cell_array[seed_coord].update_state()

    return cell_array


def create_neighbours_array(state_array, footprint=None, neighbourhood='moore', out_of_bound_values=0):
    """
    Create an array of the number of neighbours for each cell in the state_array.
    The neighbourhood can be either 'moore' or 'von_neumann'.

    Parameters
    ----------
    bin_state_array: np.ndarray
    footprint: np.ndarray (optional)
    neighbourhood: str (optional)
    out_of_bound_values: int (optional)

    Returns
    -------
    np.ndarray
        An array with the same shape as `state_array`, where each element
        contains the number of neighbors for the corresponding cell in the input array.
    """
    # Replace -1 (non-spawnable cells) with 0 in state_array to avoid counting them as neighbors in the convolution
    bin_state_array = np.where(state_array > 0, 1, 0)

    if footprint is None:
        if neighbourhood == 'moore':
            footprint = ndimage.generate_binary_structure(rank=bin_state_array.ndim, connectivity=3)
        elif neighbourhood in ['von_neumann', 'von neumann', 'vn', 'n']:
            footprint = ndimage.generate_binary_structure(rank=bin_state_array.ndim, connectivity=1)

    # Ensure the cell itself is not counted as a neighbor
    footprint[tuple([1] * bin_state_array.ndim)] = 0

    neighbours_array = ndimage.convolve(bin_state_array, footprint, mode='constant', cval=out_of_bound_values)

    return neighbours_array


def create_spawnable_neighbours_array(state_array, footprint=None, neighbourhood='moore', out_of_bound_values=0):
    """
    Create an array of the number of spawnable (state 0) neighbours for each cell in the state_array.
    The neighbourhood can be either 'moore' or 'von_neumann'.

    Parameters
    ----------
    state_array: np.ndarray
        The state array where 0 represents spawnable cells, -1 represents non-spawnable cells,
        and positive values represent cells belonging to blobs.
    footprint: np.ndarray (optional)
        The structuring element used to define the neighbourhood.
    neighbourhood: str (optional)
        Defines the neighbourhood type: 'moore' (default) or 'von_neumann'.
    out_of_bound_values: int (optional)
        The value to fill in the out-of-bounds cells.

    Returns
    -------
    np.ndarray
        An array with the same shape as `state_array`, where each element
        contains the number of spawnable (state 0) neighbors for the corresponding cell.
    """
    # Binary array where spawnable cells (state == 0) are 1, and all others are 0
    spawnable_bin_array = np.where(state_array == 0, 1, 0)

    if footprint is None:
        if neighbourhood == 'moore':
            footprint = ndimage.generate_binary_structure(rank=spawnable_bin_array.ndim, connectivity=3)
        elif neighbourhood in ['von_neumann', 'von neumann', 'vn', 'n']:
            footprint = ndimage.generate_binary_structure(rank=spawnable_bin_array.ndim, connectivity=1)

    # Ensure the cell itself is not counted as a neighbor
    footprint[tuple([1] * spawnable_bin_array.ndim)] = 0

    # Convolve to count the number of spawnable neighbors
    spawnable_neighbours_array = ndimage.convolve(spawnable_bin_array, footprint, mode='constant', cval=out_of_bound_values)

    return spawnable_neighbours_array


def check_rule(rule, neighbours_count):
    """
    Check if the rule is satisfied by the number of neighbours.
    Parameters
    ----------
    rule : int or list
    neighbours_count : int

    Returns
    -------
    bool : True if the rule is satisfied, False otherwise
    """
    if isinstance(rule, int):
        return neighbours_count == rule
    else:
        for value in rule:
            if isinstance(value, range):
                if value.start <= neighbours_count < value.stop:
                    return True
            else:
                if neighbours_count == value:
                    return True
    return False


def cell_array_to_state_array(cell_array, value_mode='state'):
    """
    Convert a cell array to a state array.

    Parameters
    ----------
    cell_array: np.ndarray
    value_mode: str

    Returns
    -------

    """
    get_state_vectorized = np.vectorize(lambda cell: cell.get_state())
    get_spawn_vectorized = np.vectorize(lambda cell: cell.get_state() + cell.get_spawn())

    if value_mode == 'state':
        return get_state_vectorized(cell_array)
    elif value_mode == 'spawn':
        return get_spawn_vectorized(cell_array)


def create_cell_array(shape, init_state=0):
    """
    Create a cell array of the given shape and initialize all the cells to the given state.

    Parameters
    ----------
    shape: tuple
    init_state: int

    Returns
    -------

    """

    def create_cell(x, y, z):
        return Cell(init_state)

    vectorized_create_cell = np.vectorize(create_cell, otypes=[object])
    return vectorized_create_cell(*np.indices(shape))

# TODO Make a 2D slice growth mode where the 3D automata would simply be the different states of a 2D CA evolution
# TODO Have a sliced growth (neighbourhood only on the slice) starting from a couple of random cells on each slice


def select_seed_coords_in_component(component_mask, max_size, num_seeds, n_dim):
    """
    Select seed coordinates within a connected component such that each seed is
    distant from another seed by about the radius of a hypersphere of the size
    of the blob's max_size.

    Parameters
    ----------
    component_mask : np.ndarray
        A binary mask of the connected component where seeds are to be placed.
    max_size : int
        The maximum size of the blob.
    num_seeds : int
        The number of seed coordinates to select.
    n_dim : int
        The number of dimensions (N) of the space.

    Returns
    -------
    seed_coords : list of tuples
        A list of coordinates for the seeds within the connected component.
    """
    # Calculate the radius of a hypersphere corresponding to the blob's max size
    radius = calculate_hypersphere_radius(max_size, n_dim)

    # Find all the coordinates within the connected component
    component_coords = np.argwhere(component_mask)

    seed_coords = []

    while len(seed_coords) < num_seeds and len(component_coords) > 0:
        # Randomly select a potential seed coordinate
        seed = random.choice(component_coords)

        if len(seed_coords) == 0:
            seed_coords.append(tuple(seed))
        else:
            # Check if the selected seed is distant enough from all other seeds
            distances = cdist([seed], seed_coords)
            if np.all(distances >= radius):
                seed_coords.append(tuple(seed))

        # Remove the selected coordinate from the pool to avoid reselection
        component_coords = np.array([coord for coord in component_coords if not np.array_equal(coord, seed)])

    return seed_coords


def initialize_random_blobs_with_mask(shape, mask, max_size=None, seed_shape=None, margin=0.1,
                                      neighbourhood='moore', min_volume_threshold=4, verbose=False):
    """
    Initialize blobs in a cell array based on a binary mask and seed values, with random seed selection within
    connected components. The function partitions the connected components of the mask, assigns seeds, and
    initializes blobs accordingly.

    Parameters
    ----------
    shape : tuple
        The shape of the cell array (e.g., (x, y, z) for a 3D array).
    mask : np.ndarray
        A binary mask indicating spawnable areas (1) and non-spawnable areas (0).
    max_size : int or list of ints, optional
        The maximum size of each blob. If an int is provided, all blobs will have the same max size.
        If a list of ints is provided, each blob will have a corresponding max size.
        If None, blobs can grow indefinitely.
    seed_shape : np.ndarray, optional
        A binary array defining the shape of each seed. The shape must fit within the cell array.
        The volume of the seed shape should not exceed the max_size of the blob.
    margin : float, optional
        The margin allowed when partitioning connected components relative to the max_size.
        For example, a margin of 0.1 allows a blob's size to be within 10% above or below the max_size.
    neighbourhood : str or np.ndarray, optional
        Defines the neighborhood structure used for connected components labeling.
        Options are:
        - "moore" (default): Uses a Moore neighborhood (all adjacent cells).
        - "von neumann": Uses a Von Neumann neighborhood (only face-adjacent cells).
        - Custom: A binary structure defining the neighborhood.
    min_volume_threshold : int, optional
        The minimum volume threshold for connected components. Components smaller than this volume will be ignored.
    verbose : bool, optional
        If True, prints detailed information about the partitioning process, seed selection, and blob initialization.

    Returns
    -------
    blobs : list
        A list of Blob instances initialized with the seed values and corresponding sizes.
    cell_array : np.ndarray
        The initialized cell array with blobs placed according to the mask and partitioning.

    Notes
    -----
    - The function first extracts connected components from the mask.
    - Connected components smaller than the `min_volume_threshold` are ignored.
    - The remaining connected components are then partitioned based on the max_size and margin, ensuring each blob's size
      approximately matches one of the partitioned sizes.
    - Seed coordinates within each connected component are selected randomly, ensuring that they are sufficiently
      distant from each other based on the calculated radius of a hypersphere with the given max_size.
    - The blobs are initialized and updated incrementally within the partitioning loop.
    - If no valid partition can be found within the specified margin and max_size constraints, an error is raised.
    - If `seed_shape` is provided, its volume must not exceed the `max_size` of the blob.
    """

    # Step 1: Determine the neighborhood structure for labeling
    if isinstance(neighbourhood, str):
        if neighbourhood.lower() == 'moore':
            structure = generate_binary_structure(rank=len(shape), connectivity=len(shape))
        elif neighbourhood.lower() == 'von neumann':
            structure = generate_binary_structure(rank=len(shape), connectivity=1)
        else:
            raise ValueError("Invalid neighborhood option. Choose 'moore', 'von neumann', or "
                             "provide a custom binary structure.")
    elif isinstance(neighbourhood, np.ndarray):
        structure = neighbourhood
    else:
        raise ValueError("neighbourhood must be either a string ('moore' or 'von neumann') "
                         "or a binary structure array.")

    # Step 2: Extract connected components from the mask
    labeled_mask, num_features = label(mask, structure=structure)
    component_sizes = np.bincount(labeled_mask.ravel())[1:]  # Exclude background

    if verbose:
        print(f"Number of connected components found: {num_features}")
        print(f"Sizes of connected components before filtering: {component_sizes}")

    # Step 3: Filter out components smaller than the min_volume_threshold
    valid_components = component_sizes >= min_volume_threshold
    component_sizes = component_sizes[valid_components]

    if verbose:
        print(f"Sizes of connected components after filtering: {component_sizes}")

    if len(component_sizes) == 0:
        raise ValueError("No connected components meet the minimum volume threshold.")

    # Step 4: Generate the partition using the connected components
    if isinstance(max_size, int):
        total_cells = mask.sum()
        base_size = max_size
        num_full_blobs = total_cells // base_size
        remainder = total_cells % base_size
        max_sizes = [base_size] * num_full_blobs + ([remainder] if remainder > 0 else [])
    elif isinstance(max_size, list):
        max_sizes = max_size
    else:
        raise ValueError("max_size must be an integer or a list of integers.")

    seed_values = list(range(1, len(max_sizes) + 1))

    partitions = partition_values_to_sizes_with_margin(max_sizes, component_sizes.tolist(), margin)

    print(f"Partitions: {partitions}")

    if partitions is None:
        raise ValueError("No valid partition found with the given margin and max sizes. You may need to adjust the "
                         f"margin or the max_size parameter. Current max sizes: {max_sizes}")

    if verbose:
        print("Partitioning result:")
        for i, partition in enumerate(partitions):
            print(f"Partition {i} sizes: {[max_sizes[p] for p in partition]}")
            print(f'Indices: {partition}')

    # Step 5: Initialize the cell array using vectorization
    def create_cell(x, y, z):
        if mask[x, y, z] == 1:
            return Cell(0)
        else:
            return Cell(-1)

    vectorized_create_cell = np.vectorize(create_cell, otypes=[object])
    cell_array = vectorized_create_cell(*np.indices(shape))

    blobs = []
    used_seeds = set()

    # Step 6: Iterate over the partitioned connected components and select seeds
    for i, partition in enumerate(partitions[0]):  # Use the first valid partition
        component_mask = (labeled_mask == (np.where(valid_components)[0][partition] + 1))
        seed_coord = select_seed_coords_in_component(component_mask, max_sizes[partition],
                                                     1, n_dim=len(shape))[0]
        seed_value = seed_values[partition]
        blob_max_size = max_sizes[partition]

        if partition in used_seeds:
            continue

        used_seeds.add(partition)
        if verbose:
            print(f'Initializing blob {seed_value} at {seed_coord} with max size {blob_max_size}')

        if seed_shape is not None:
            seed_volume = np.sum(seed_shape)
            if seed_volume > blob_max_size:
                raise ValueError(f"The seed shape volume ({seed_volume}) exceeds the max_size "
                                 f"({blob_max_size}) for the blob.")

        blob = Blob(seed_coord, seed_value, cell_array, blob_max_size)
        blobs.append(blob)

        if seed_shape is not None:
            seed_shape_coords = np.argwhere(seed_shape)
            for offset in seed_shape_coords:
                target_coord = tuple(np.array(seed_coord) + offset - np.array(seed_shape.shape) // 2)
                if coord_in_array(target_coord, cell_array) and cell_array[target_coord].get_state() == 0:
                    cell_array[target_coord].set_next_state(seed_value, it_time=0)
                    blob.add_new_cells(target_coord)
        else:
            cell_array[seed_coord].set_next_state(seed_value, it_time=0)

        # Update the blob immediately after initialization and seeding
        blob.update_blob(cell_array)

    return blobs, cell_array


def initialize_blobs_with_mask(shape, mask, seed_coords, seed_values, max_size=None, seed_shape=None):
    """
    Initialize the cell array with a mask and seed values, and create Blob instances with optional shapes.

    Parameters
    ----------
    shape : tuple
        The shape of the cell array.
    mask : np.ndarray
        A binary mask indicating spawnable areas (1) and non-spawnable areas (0).
    seed_coords : list of tuples
        The coordinates of the seeds.
    seed_values : list of ints
        The values (cluster indices) for each seed.
    max_size : int or list of ints, optional
        The maximum size of each blob. If an int is provided, all blobs will have the same max size.
        If a list of ints is provided, each blob will have a corresponding max size.
        If None, blobs can grow indefinitely.
    seed_shape : np.ndarray, optional
        A binary array defining the shape of each seed. The shape must fit within the cell array.

    Returns
    -------
    blobs : list
        A list of Blob instances.
    cell_array : np.ndarray
        The initialized cell array.

    Notes:
    - The blobs are initialized with the seed values and the edge cells.
    - If the seed coord of a blob falls into another blob's initial shape, the seed will replace the other blob's cell.
        This will prevent the new blob from growing. This needs to be handled in the seeds coordinate generation.
    """
    # Initialize the cell array using vectorization
    def create_cell(x, y, z):
        if mask[x, y, z] == 1:
            return Cell(0)  # Spawnable cells
        else:
            return Cell(-1)  # Non-spawnable cells

    vectorized_create_cell = np.vectorize(create_cell, otypes=[object])
    cell_array = vectorized_create_cell(*np.indices(shape))

    # Ensure max_size is a list of appropriate length
    if isinstance(max_size, int) or max_size is None:
        max_size = [max_size] * len(seed_coords)  # Apply the same max_size to all blobs
    elif isinstance(max_size, list) and len(max_size) != len(seed_coords):
        raise ValueError("If max_size is a list, it must have the same length as seed_coords.")

    # Create Blob instances and initialize the seeds with optional shapes
    blobs = []
    for seed_coord, seed_value, blob_max_size in zip(seed_coords, seed_values, max_size):
        print(f'Initializing blob {seed_value} at {seed_coord}')
        blob = Blob(seed_coord, seed_value, cell_array, blob_max_size)
        print(f'Blob init values: {blob.seed_value} | {blob.max_size} | {blob.new_cells} | {blob.size}')
        blobs.append(blob)

        if seed_shape is not None:
            # Apply the seed shape around the seed_coord
            seed_shape_coords = np.argwhere(seed_shape)
            for offset in seed_shape_coords:
                target_coord = tuple(np.array(seed_coord) + offset - np.array(seed_shape.shape) // 2)
                if coord_in_array(target_coord, cell_array) and cell_array[target_coord].get_state() == 0:
                    cell_array[target_coord].set_next_state(seed_value, it_time=0)
                    blob.add_new_cells(target_coord)
        else:
            # If no shape is specified, just place a single cell
            cell_array[seed_coord].set_next_state(seed_value, it_time=0)
            # the seed is already in the edge_cells from the Blob initialization and will be updated in the next step
        print(f'Number of new cells for seed {seed_value}: {len(blob.new_cells)}')
        blob.update_blob(cell_array)

    return blobs, cell_array
