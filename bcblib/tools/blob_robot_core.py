from pathlib import Path
from functools import partial

import numpy as np
from joblib import Parallel, delayed
from scipy import ndimage

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
    cell_array = np.empty(shape, dtype=object)

    # Initialize the cell array based on the mask
    with np.nditer(cell_array, flags=['refs_ok', 'multi_index'], op_flags=['readwrite']) as it:
        for c in it:
            coord = it.multi_index
            if mask[coord] == 1:
                c[...] = Cell(spawnable_value)
            else:
                c[...] = Cell(non_spawnable_value)

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
    state_array: np.ndarray
    footprint: np.ndarray (optional)
    neighbourhood: str (optional)
    out_of_bound_values: int (optional)

    Returns
    -------
    np.ndarray
        An array with the same shape as `state_array`, where each element
        contains the number of neighbors for the corresponding cell in the input array.
    """
    if footprint is None:
        if neighbourhood == 'moore':
            footprint = ndimage.generate_binary_structure(rank=state_array.ndim, connectivity=3)
        elif neighbourhood in ['von_neumann', 'von neumann', 'vn', 'n']:
            footprint = ndimage.generate_binary_structure(rank=state_array.ndim, connectivity=1)

    # Ensure the cell itself is not counted as a neighbor
    footprint[tuple([1] * state_array.ndim)] = 0

    # Using convolve might be faster than generic_filter if the operation is simple
    neighbours_array = ndimage.convolve(state_array, footprint, mode='constant', cval=out_of_bound_values)
    # return ndimage.generic_filter(state_array, np.count_nonzero, footprint=footprint, mode='constant',
    #                               cval=out_of_bound_values)
    return neighbours_array


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
    state_array = np.zeros_like(cell_array, dtype=int)
    with np.nditer(cell_array, flags=['refs_ok', 'multi_index']) as it:
        for c in it:
            if value_mode == 'state':
                state_array[it.multi_index] = c.item().get_state()
            if value_mode == 'spawn':
                state_array[it.multi_index] = c.item().get_state() + c.item().get_spawn()
    return state_array


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
    cell_array = np.empty(shape, dtype=object)
    with np.nditer(cell_array, flags=['refs_ok', 'multi_index'], op_flags=['readwrite']) as it:
        for c in it:
            c[...] = Cell(init_state)
    return cell_array

# TODO Make a 2D slice growth mode where the 3D automata would simply be the different states of a 2D CA evolution
# TODO Have a sliced growth (neighbourhood only on the slice) starting from a couple of random cells on each slice


def initialize_blobs_with_mask(shape, mask, seed_coords, seed_values, max_size=None):
    blobs = []
    cell_array = np.empty(shape, dtype=object)

    with np.nditer(cell_array, flags=['refs_ok', 'multi_index'], op_flags=['readwrite']) as it:
        for c in it:
            coord = it.multi_index
            if mask[coord] == 1:
                c[...] = Cell()
            else:
                c[...] = Cell(non_spawnable_value=-1)

    for seed_coord, seed_value in zip(seed_coords, seed_values):
        blob = Blob(seed_coord, seed_value, max_size)
        blobs.append(blob)
        cell_array[seed_coord].set_next_state(seed_value, it_time=0)
        cell_array[seed_coord].update_state()

    return blobs, cell_array
