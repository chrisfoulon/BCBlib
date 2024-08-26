import numpy as np
from scipy import ndimage


def get_neighbors(coord, shape):
    """
    Get valid neighbors of a given coordinate.

    Parameters
    ----------
    coord : tuple
        The coordinate to find neighbors for.
    shape : tuple
        The shape of the grid.

    Returns
    -------
    list
        A list of neighboring coordinates.
    """
    neighbour_offsets = np.argwhere(ndimage.generate_binary_structure(rank=3, connectivity=1))

    neighbors = []
    for offset in neighbour_offsets:
        neighbor = tuple(np.array(coord) + offset - 1)
        if all(0 <= n < s for n, s in zip(neighbor, shape)):
            neighbors.append(neighbor)

    return neighbors


class Blob:
    def __init__(self, seed_coord, seed_value, max_size=None):
        self.seed_value = seed_value
        self.max_size = max_size
        # TODO we need to update the size somewhere
        self.size = 1
        self.edge_cells = {seed_coord}  # Start with the seed as the edge
        self.growth_complete = False

    def can_grow(self):
        """
        Check if the blob can continue to grow.

        The blob can grow if it has not reached its maximum size and still has edge cells.

        Returns
        -------
        bool
            True if the blob can grow, False otherwise.
        """
        if self.max_size is not None and self.size >= self.max_size:
            self.growth_complete = True
            return False
        if not self.edge_cells:  # Check if the edge set is empty
            self.growth_complete = True
            return False
        return True

    def update_edge(self, cell_array, neighbour_array):
        """
        Update the edge cells of the blob using the neighbour array.

        Parameters
        ----------
        cell_array : np.ndarray
            The array representing the cellular automaton grid.
        neighbour_array : np.ndarray
            The array representing the number of neighbors for each cell.
        """
        new_edge_cells = set()
        # TODO we're still checking all the cells, we could optimize this
        # and we are not removing the edge cells that are not edge cells anymore
        for coord in self.edge_cells:
            if neighbour_array[coord] < 6:  # Blob-specific edge condition
                neighbors = get_neighbors(coord, cell_array.shape)
                for neighbor in neighbors:
                    if cell_array[neighbor].get_state() == 0:  # Check if unoccupied
                        new_edge_cells.add(neighbor)

        self.edge_cells = new_edge_cells

