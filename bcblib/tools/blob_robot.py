from pathlib import Path

import numpy as np

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
    def __init__(self, state=0):
        self.state = 0
        self.next_state = 0
        self.spawn_time = -1
        self.set_next_state(state, it_time=0)
        self.update_state()

    def get_state(self):
        return self.state

    def get_spawn(self):
        return self.spawn_time

    def set_next_state(self, new_state, it_time):
        self.next_state = new_state
        # So, if new_state is True or > 0, but other types might have weird interactions
        if new_state and self.spawn_time == -1:
            self.spawn_time = it_time
        if not new_state:
            self.spawn_time = -1

    def update_state(self):
        self.state = self.next_state


def coord_in_array(coord, array):
    if len(coord) != len(array.shape):
        raise ValueError('Coord must have the same dimension as array.shape')
    in_arr = True
    for i, c in enumerate(coord):
        if not 0 <= c < array.shape[i]:
            in_arr = False
    return in_arr


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


# TODO there could be a bunch of other strategies for the spawn and states
# e.g. If the spawn condition is on a living cell it could go up a state
def apply_rule_to_cell(array, cell_coord, it_time, rule=(4, 2, 1, 'M'), fading='hp'):
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
    cell_coord
    array
    rule
    fading
    it_time
    """
    cell = array[cell_coord]
    neighbours = get_neighbours(array, cell_coord)
    survive_rule = rule[0]
    spawn_rule = rule[1]
    num_states = rule[2]
    neighbourhood_method = 'moore'
    if len(rule) == 4:
        if rule[3].lower() in ['n', 'vn', 'von_neumann', 'von neumann']:
            neighbourhood_method = 'von_neumann'
    if neighbourhood_method == 'moore':
        cell_state = cell.get_state()
        neighbours_count = np.count_nonzero(neighbours)
        new_state = cell_state
        if cell_state:
            # Here the cell either survives or fades/dies
            if neighbours_count < survive_rule or fading == 'final_countdown':
                new_state -= 1
        else:
            # Here the cell can spawn
            if neighbours_count >= spawn_rule:
                new_state = num_states - 1
    else:
        # TODO
        new_state = cell.get_state()
    cell.set_next_state(new_state, it_time)
    

def evolve_automata(cell_array, it_time, rule=(4, 2, 1, 'M'), fading='hp'):
    with np.nditer(cell_array, flags=['refs_ok', 'multi_index']) as it:
        acc = 0
        for c in it:
            if c.item().get_state():
                acc += 1
            apply_rule_to_cell(cell_array, it.multi_index, it_time, rule=rule, fading=fading)
    with np.nditer(cell_array, flags=['refs_ok', 'multi_index']) as it:
        for c in it:
            c.item().update_state()


def cell_array_to_state_array(cell_array):
    state_array = np.zeros_like(cell_array, dtype=int)
    with np.nditer(cell_array, flags=['refs_ok', 'multi_index']) as it:
        for c in it:
            state_array[it.multi_index] = c.item().get_state()
    return state_array


def create_cell_array(shape, init_state=0):
    cell_array = np.empty(shape, dtype=object)
    with np.nditer(cell_array, flags=['refs_ok', 'multi_index'], op_flags=['readwrite']) as it:
        for c in it:
            c[...] = Cell(init_state)
    return cell_array
