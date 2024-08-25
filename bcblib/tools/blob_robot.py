from pathlib import Path
from functools import partial

import numpy as np
from joblib import Parallel, delayed
from scipy import ndimage

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


class Cell:
    """
    A cell is a point in space that can have a state and a spawn time.
    The spawn time is the iteration time at which the cell was created.
    The state is the current state of the cell.
    The next state is the state the cell will have at the next iteration.
    """
    def __init__(self, state=0, spawn_default_value=-1):
        self.spawn_default_value = spawn_default_value
        self.state = 0
        self.next_state = 0
        self.spawn_time = spawn_default_value
        self.set_next_state(state, it_time=0)
        self.update_state()

    def get_state(self):
        return self.state

    def get_next_state(self):
        return self.next_state

    def get_spawn(self):
        return self.spawn_time

    def set_next_state(self, new_state, it_time):
        self.next_state = new_state
        # So, if new_state is True or > 0, but other types might have weird interactions
        if new_state and self.get_spawn() == -1:
            self.spawn_time = it_time
        if not new_state:
            self.spawn_time = self.spawn_default_value

    def update_state(self):
        self.state = self.get_next_state()


# TODO make a helper function module in the bcblib for generic functions like this one.
def coord_in_array(coord, array):
    """

    Parameters
    ----------
    coord: tuple
    array: np.ndarray

    Returns
    -------
    bool

    """
    if len(coord) != len(array.shape):
        raise ValueError('Coord must have the same dimension as array.shape')
    in_arr = True
    for i, c in enumerate(coord):
        if not 0 <= c < array.shape[i]:
            in_arr = False
    return in_arr


def create_spherical_shape(radius):
    """
    Create a 3D binary array with a spherical shape.

    Parameters
    ----------
    radius : int
        The radius of the sphere.

    Returns
    -------
    np.ndarray
        A 3D binary array with a spherical shape.
    """
    # Define the grid
    diameter = 2 * radius + 1
    grid = np.ogrid[-radius:radius + 1, -radius:radius + 1, -radius:radius + 1]

    # Create a binary sphere
    sphere = grid[0] ** 2 + grid[1] ** 2 + grid[2] ** 2 <= radius ** 2

    return sphere.astype(int)


def create_shapes_in_arr(cell_array, coords=1, structure=None, connectivity=None, dimensions=3, value=1, it_time=0):
    """
    Create shapes in an array of cells.
    The shapes are created by dilating a random number of points in the array.
    The number of points is given by the 'coords' parameter.
    The dilation is defined with a structure or a connectivity.
    The structure is a binary array
    Parameters
    ----------
    cell_array: np.ndarray
    coords: int or list of tuples
    structure: np.ndarray
    connectivity: int
    dimensions: int
    value: int
    it_time: int

    Returns
    -------
    None
    Examples:
    >>> import numpy as np
    >>> from bcblib.tools.blob_robot import create_shapes_in_arr
    >>> arr = np.zeros((10, 10, 10))
    >>> create_shapes_in_arr(arr, coords=10, value=1)
    >>> np.count_nonzero(arr)
    10
    """
    if structure is None and connectivity is None:
        raise ValueError('Either a structure or a connectivity must be provided')
    if connectivity is not None:
        structure = ndimage.generate_binary_structure(dimensions, connectivity)
    array = np.zeros_like(cell_array).astype(int)
    # TODO add the other 'coords' options (one coord or a list of coordinates)
    # Then we want coords structures randomly located in the array
    if isinstance(coords, int):
        arr_coords = np.argwhere(array == 0)
        for i in range(coords):
            array[tuple(arr_coords[np.random.randint(len(arr_coords))])] = 1
        dilated_array = ndimage.binary_dilation(array, structure=structure).astype(array.dtype)
        for c in np.argwhere(dilated_array):
            cell_array[tuple(c)].set_next_state(value, it_time)
            cell_array[tuple(c)].update_state()


# Deprecated (slow AF)
def get_neighbours(array, cell_coord, out_of_bound_values=0):
    neighbours_arr = np.zeros((3,) * 3)
    offset_array = [-1, 0, 1]
    for x in offset_array:
        for y in offset_array:
            for z in offset_array:
                if coord_in_array([x + cell_coord[0], y + cell_coord[1], z + cell_coord[2]], array):
                    neighbours_arr[x + 1, y + 1, z + 1] = array[(x + cell_coord[0],
                                                                 y + cell_coord[1],
                                                                 z + cell_coord[2])].get_state()
                else:
                    # TODO might be removed for optimisation
                    neighbours_arr[x + 1, y + 1, z + 1] = out_of_bound_values
    return neighbours_arr


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


# TODO there could be a bunch of other strategies for the spawn and states
# e.g. If the spawn condition is on a living cell it could go up a state
def apply_rule_to_cell(array, cell_coord, neighbour_array, it_time, rule=(4, 2, 1, 'M')):
    """
    Still from https://softologyblog.wordpress.com/2019/12/28/3d-cellular-automata-3/:
    "Rule 445 is the first rule in the video and shown as 4/4/5/M. This is fairly standard survival/birth CA syntax.
    -The first 4 indicates that a state 1 cell survives if it has 4 neighbor cells.
    -The second 4 indicates that a cell is born in an empty location if it has 4 neighbors.
    -The 5 means each cell has 5 total states it can be in (state 4 for newly born which then fades to state 1 and
    then state 0 for no cell)
    -M means a Moore neighborhood.

    Another rule is Clouds 1 shown as 13-26/13-14,17-19/2/M
    Alive cells with 13,14,15,16,17,18,19,20,21,22,23,24,25 or 26 neighbors survive.
    Empty cells with 13,14,17,18 or 19 neighbors have a new cell born at that location.
    2 states. Cells are either dead or alive. No refractory period they fade from birth to death.
    M means a Moore neighborhood.

    More than 2 states can be confusing at first. In a 2 state CA when a cell dies it goes immediately from living
    (state 1) to dead (state 0). In more than 2 states, when a cell dies it does not immediately go to state 0.
    Instead, it fades out to state 0. If there are 5 total states then a live cell with state 4
    (4 not 5 as the possible state values are 0,1,2,3 and 4) fades to state 3, then 2,
    then 1 and finally disappears at state 0."

    Parameters
    ----------
    neighbour_array
    cell_coord
    array
    rule :
        survive rule and spawn rule can be lists but in that case, all the values must be explicit
        e.g. [2, 3, 4, 6, 7, 8]
    it_time
    """
    cell = array[cell_coord]
    neighbours_count = neighbour_array[cell_coord]
    survive_rule = rule[0]
    spawn_rule = rule[1]
    num_states = rule[2]

    cell_state = cell.get_state()
    new_state = cell_state

    if cell_state:  # The cell is alive
        fades = check_rule(survive_rule, neighbours_count)
        if fades:
            new_state -= 1
    else:  # The cell is dead
        spawns = check_rule(spawn_rule, neighbours_count)
        if spawns:
            new_state = num_states - 1

    # Apply the new state to the cell
    cell.set_next_state(new_state, it_time)
    

def evolve_automaton(cell_array, it_time, rule=(4, 2, 1, 'M'), fading='hp'):
    """
    Evolve the automaton one step forward.
    Parameters
    ----------
    cell_array
    it_time
    rule
    fading

    Returns
    -------

    """
    neighbour_array = create_neighbours_array(cell_array_to_state_array(cell_array))
    with np.nditer(cell_array, flags=['refs_ok', 'multi_index']) as it:
        acc = 0
        for c in it:
            if c.item().get_state():
                acc += 1
            apply_rule_to_cell(cell_array, it.multi_index, neighbour_array, it_time, rule=rule)
    with np.nditer(cell_array, flags=['refs_ok', 'multi_index']) as it:
        for c in it:
            c.item().update_state()


def apply_growth_rule(array, cell_coord, neighbour_array, it_time):
    """
    Apply the growth rule to a cell, ensuring that it spreads its value only to spawnable cells.

    Parameters
    ----------
    array : np.ndarray
        The array representing the cellular automaton grid.
    cell_coord : tuple
        The coordinates of the cell in the array.
    neighbour_array : np.ndarray
        An array of the same shape as `array`, containing the number of neighbors for each cell.
    it_time : int
        The current iteration time.

    Returns
    -------
    None
    """
    cell = array[cell_coord]
    cell_state = cell.get_state()

    if cell_state <= 0:
        return  # Skip non-spawnable cells and empty cells

    neighbours_count = neighbour_array[cell_coord]

    # Spread the cell's value to its neighbors if they are spawnable (i.e., state == 0)
    spreadable_cells = []
    for offset in np.argwhere(ndimage.generate_binary_structure(rank=3, connectivity=1)):
        neighbour_coord = tuple(np.array(cell_coord) + offset - 1)
        if neighbour_coord == cell_coord:
            continue  # Skip if the neighbor coordinate is the same as the current cell coordinate
        if coord_in_array(neighbour_coord, array) and array[neighbour_coord].get_state() == 0:
            spreadable_cells.append(neighbour_coord)

    for coord in spreadable_cells:
        array[coord].set_next_state(cell_state, it_time)

    cell.update_state()


def apply_viscous_growth_rule(array, cell_coord, neighbour_array, it_time):
    """
    Apply a viscous growth rule to simulate a liquid-like spreading.

    Parameters
    ----------
    array : np.ndarray
        The array representing the cellular automaton grid.
    cell_coord : tuple
        The coordinates of the cell in the array.
    neighbour_array : np.ndarray
        An array of the same shape as `array`, containing the number of neighbors for each cell.
    it_time : int
        The current iteration time.

    Returns
    -------
    None
    """
    cell = array[cell_coord]
    cell_state = cell.get_state()

    if cell_state <= 0:
        return  # Skip non-spawnable cells and empty cells

    # Viscosity factor: decrease the chance of spreading in dense areas
    viscosity = 1 / (1 + neighbour_array[cell_coord])

    neighbour_offsets = np.argwhere(ndimage.generate_binary_structure(rank=3, connectivity=1))
    np.random.shuffle(neighbour_offsets)

    for offset in neighbour_offsets:
        neighbour_coord = tuple(np.array(cell_coord) + offset - 1)
        if coord_in_array(neighbour_coord, array) and array[neighbour_coord].get_state() == 0:
            if np.random.rand() < viscosity:
                array[neighbour_coord].set_next_state(cell_state, it_time)
                break  # Spread to only one neighbor per iteration to simulate fluid, controlled growth

    cell.update_state()



def evolve_parcellation(cell_array, it_time):
    """
    Evolve the automaton one step forward for parcellation.

    Parameters
    ----------
    cell_array : np.ndarray
        The array representing the cellular automaton grid.
    it_time : int
        The current iteration time.

    Returns
    -------
    None
    """
    neighbour_array = create_neighbours_array(cell_array_to_state_array(cell_array))

    with np.nditer(cell_array, flags=['refs_ok', 'multi_index']) as it:
        for c in it:
            apply_growth_rule(cell_array, it.multi_index, neighbour_array, it_time)

    with np.nditer(cell_array, flags=['refs_ok', 'multi_index']) as it:
        for c in it:
            c.item().update_state()


def evolve_parcellation_opti(cell_array, it_time, growth_rule='default'):
    """
    Optimized function to evolve the cellular automaton for parcellation using parallel processing.

    Parameters
    ----------
    cell_array : np.ndarray
        The array representing the cellular automaton grid.
    it_time : int
        The current iteration time.

    Returns
    -------
    None
    """
    state_array = cell_array_to_state_array(cell_array)
    neighbour_array = create_neighbours_array(state_array)

    # Iterate only over cells that are active (i.e., not in state 0)
    active_cells = np.argwhere(state_array > 0)

    for coord in active_cells:
        coord = tuple(coord)  # Convert from numpy array to tuple
        if growth_rule == 'viscous':
            apply_viscous_growth_rule(cell_array, coord, neighbour_array, it_time)
        elif growth_rule == 'default':
            apply_growth_rule(cell_array, coord, neighbour_array, it_time)

    # Update the state of all cells in one go
    with np.nditer(cell_array, flags=['refs_ok', 'multi_index']) as it:
        for c in it:
            c.item().update_state()


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
