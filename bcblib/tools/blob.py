import numpy as np


class Blob:
    def __init__(self, seed_coord, seed_value, max_size=None):
        self.seed_value = seed_value
        self.max_size = max_size
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

    def update_edge(self, cell_array, mask, neighbour_array):
        """
        Update the edge cells of the blob.

        Parameters
        ----------
        cell_array : np.ndarray
            The array representing the cellular automaton grid.
        mask : np.ndarray
            The binary mask indicating spawnable areas.
        neighbour_array : np.ndarray
            The array representing the number of neighbors for each cell.
        """
        new_edge_cells = set()

        for coord in self.edge_cells:
            if neighbour_array[coord] < 6:  # Blob-specific edge condition
                neighbors = self.get_neighbors(coord, cell_array.shape)
                for neighbor in neighbors:
                    if mask[neighbor] and cell_array[neighbor].get_state() == 0:  # Check spawnable and unoccupied
                        new_edge_cells.add(neighbor)

        self.edge_cells = new_edge_cells

    def grow(self, cell_array, it_time, neighbour_array, mask):
        """
        Perform one growth iteration for the blob.

        Parameters
        ----------
        cell_array : np.ndarray
            The array representing the cellular automaton grid.
        it_time : int
            The current iteration time.
        neighbour_array : np.ndarray
            The array representing the number of neighbors for each cell.
        mask : np.ndarray
            The binary mask indicating spawnable areas.
        """
        if not self.can_grow():
            return

        new_growth = set()
        for coord in list(self.edge_cells):  # Iterate over a copy to modify the set safely
            if mask[coord] and cell_array[coord].get_state() == 0:
                cell_array[coord].set_next_state(self.seed_value, it_time)
                new_growth.add(coord)
                self.size += 1

                if not self.can_grow():
                    break

        self.edge_cells = new_growth
        self.update_edge(cell_array, mask, neighbour_array)

    def get_neighbors(self, coord, shape):
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
        neighbors = []
        for offset in np.ndindex(3, 3, 3):
            if offset == (1, 1, 1):  # Skip the center (itself)
                continue
            neighbor = tuple(np.array(coord) + np.array(offset) - 1)
            if all(0 <= n < s for n, s in zip(neighbor, shape)):
                neighbors.append(neighbor)
        return neighbors