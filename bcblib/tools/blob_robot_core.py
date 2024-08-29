from pathlib import Path
from functools import partial

import numpy as np
from scipy import ndimage
from scipy.spatial.distance import cdist

from bcblib.tools.arrays_utils import coord_in_array
from bcblib.tools.blob import Blob
from bcblib.tools.cell import Cell

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


def assign_blobs_to_components(labeled_mask, blob_sizes, margin=0.1):
    """
    Assign blobs to connected components based on their sizes.

    Parameters
    ----------
    labeled_mask : np.ndarray
        A labeled array where each unique label corresponds to a connected component.
    blob_sizes : list of ints
        The maximum sizes of the blobs to be assigned.
    margin : float
        The allowed margin for the sum of blob sizes compared to component size.

    Returns
    -------
    dict
        A dictionary where keys are component labels and values are lists of blob indices assigned to each component.
    """
    # Calculate the size of each component
    component_sizes = [(label, np.sum(labeled_mask == label)) for label in np.unique(labeled_mask) if label != 0]

    # Sort components by size (largest to smallest)
    component_sizes.sort(key=lambda x: x[1], reverse=True)

    # Sort blobs by size (largest to smallest)
    blob_sizes = sorted(enumerate(blob_sizes), key=lambda x: x[1], reverse=True)

    assignments = {label: [] for label, _ in component_sizes}

    for component_label, component_size in component_sizes:
        available_size = component_size
        for blob_index, blob_size in blob_sizes[:]:  # Iterate over a copy of the list
            if (available_size * (1 - margin)) <= blob_size <= (available_size * (1 + margin)):
                assignments[component_label].append(blob_index)
                available_size -= blob_size
                blob_sizes.remove((blob_index, blob_size))  # Remove assigned blob
            elif blob_size < available_size * (1 - margin):
                assignments[component_label].append(blob_index)
                available_size -= blob_size
                blob_sizes.remove((blob_index, blob_size))  # Remove assigned blob
            if available_size <= 0:
                break

    return assignments


def determine_seed_positions(labeled_mask, assignments, shape, seed_shape=None):
    """
    Determine seed positions for blobs within each connected component.

    Parameters
    ----------
    labeled_mask : np.ndarray
        A labeled array where each unique label corresponds to a connected component.
    assignments : dict
        A dictionary where keys are component labels and values are lists of blob indices assigned to each component.
    shape : tuple
        The overall shape of the cell array.
    seed_shape : np.ndarray, optional
        A binary array defining the shape of each seed. The shape must fit within the cell array.

    Returns
    -------
    list of tuples
        Each tuple contains the seed coordinates, blob index, and max_size.
    """
    seed_positions = []

    for component_label, blob_indices in assignments.items():
        # Get all positions in the component
        component_coords = np.argwhere(labeled_mask == component_label)

        # Calculate the minimum distance for each blob based on its max_size
        min_distances = []
        for blob_index in blob_indices:
            max_size = shape[blob_index]  # Assuming blob_sizes correspond to some dimension in `shape`
            radius = (max_size / np.pi) ** (1 / 2)  # Adjust for higher dims if needed
            min_distances.append(radius)

        # Place blobs within the component
        for blob_index, min_distance in zip(blob_indices, min_distances):
            valid_positions = component_coords

            # Filter out positions that are too close to already placed seeds
            if seed_positions:
                existing_coords = np.array([pos[0] for pos in seed_positions if pos[1] in blob_indices])
                dists = cdist(valid_positions, existing_coords)
                valid_positions = valid_positions[np.all(dists >= min_distance, axis=1)]

            # Randomly select a valid seed position
            if len(valid_positions) == 0:
                raise ValueError("No valid positions found. Consider adjusting the minimum distance or component size.")

            seed_position = valid_positions[np.random.choice(len(valid_positions))]
            seed_positions.append((tuple(seed_position), blob_index, shape[blob_index]))

    return seed_positions


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


