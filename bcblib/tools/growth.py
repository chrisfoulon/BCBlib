import numpy as np
from scipy import ndimage

from bcblib.tools.arrays_utils import coord_in_array
from bcblib.tools.blob_robot_core import (check_rule, create_neighbours_array,
                                          create_spawnable_neighbours_array, cell_array_to_state_array)


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

def apply_growth_rule(array, cell_coord, neighbour_array, it_time, n_dim=3):
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
    n_dim : int, optional
        The number of dimensions of the array.

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
    # The cell is included in the neighbour as its state needs to be updated with the it_time
    for offset in np.argwhere(ndimage.generate_binary_structure(rank=n_dim, connectivity=1)):
        neighbour_coord = tuple(np.array(cell_coord) + offset - 1)
        if coord_in_array(neighbour_coord, array) and array[neighbour_coord].get_state() == 0:
            spreadable_cells.append(neighbour_coord)

    for coord in spreadable_cells:
        array[coord].set_next_state(cell_state, it_time)

    # This should not be necessary if we call evolve_autonmaton
    # cell.update_state()
    # TODO review the logic for a classic CA


def apply_viscous_growth_rule(array, cell_coord, neighbour_array, it_time, n_dim=3):
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
    n_dim : int, optional
        The number of dimensions of the array.

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

    neighbour_offsets = np.argwhere(ndimage.generate_binary_structure(rank=n_dim, connectivity=1))
    np.random.shuffle(neighbour_offsets)

    for offset in neighbour_offsets:
        neighbour_coord = tuple(np.array(cell_coord) + offset - 1)
        if coord_in_array(neighbour_coord, array) and array[neighbour_coord].get_state() == 0:
            if np.random.rand() < viscosity:
                array[neighbour_coord].set_next_state(cell_state, it_time)
                break  # Spread to only one neighbor per iteration to simulate fluid, controlled growth
                # TODO that might be a bit too viscous, maybe we should spread to more than one neighbour

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
    Optimized function to evolve the cellular automaton for parcellation

    Parameters
    ----------
    cell_array : np.ndarray
        The array representing the cellular automaton grid.
    it_time : int
        The current iteration time.
    growth_rule : str
        The growth rule to apply. Either 'default' or 'viscous'.

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


def evolve_parcellation_opti_subset(cell_array, edge_cells, neighbour_array, it_time, growth_rule='default'):
    """
    Optimized function to evolve the cellular automaton for parcellation on a subset of cells.

    Parameters
    ----------
    cell_array : np.ndarray
        The array representing the cellular automaton grid.
    edge_cells : list of tuples
        List of coordinates to be processed.
    neighbour_array : np.ndarray
        The neighbor array representing the number of neighbors for each cell.
    it_time : int
        The current iteration time.
    growth_rule : str, optional
        The growth rule to apply, either 'default' or 'viscous'.

    Returns
    -------
    None
    """
    for coord in edge_cells:
        if growth_rule == 'viscous':
            apply_viscous_growth_rule(cell_array, coord, neighbour_array, it_time)
        elif growth_rule == 'default':
            apply_growth_rule(cell_array, coord, neighbour_array, it_time)

    # Update the state of all cells in the edge subset
    for coord in edge_cells:
        cell_array[coord].update_state()


def evolve_blobs(blobs, cell_array, it_time):
    # Convert cell array to state array
    state_array = cell_array_to_state_array(cell_array)
    bin_state_array = state_array > 0
    # Create the neighbour array
    spawnable_neighbours_array = create_spawnable_neighbours_array(state_array, neighbourhood='von_neumann')

    # Check if any blob can still grow
    blobs_can_still_grow = any(blob.can_grow() for blob in blobs)
    if not blobs_can_still_grow:
        return

    # Track all live cells as seen by blobs
    tracked_live_cells_set = set()

    # Process each blob
    for blob in blobs:
        if not blob.growth_complete:
            blob.grow(cell_array, spawnable_neighbours_array, it_time)
        # Track live cells for each blob
        tracked_live_cells_set.update(blob.new_cells)