import numpy as np
from scipy import ndimage
import random

from bcblib.tools.arrays_utils import coord_in_array


def get_neighbours(coord, shape, n_dim=3):
    """
    Get valid neighbors of a given coordinate.

    Parameters
    ----------
    coord : tuple
        The coordinate to find neighbors for.
    shape : tuple
        The shape of the grid.
    n_dim : int, optional
        The number of dimensions of the grid. Default is 3.

    Returns
    -------
    list
        A list of neighboring coordinates.
    """
    neighbour_offsets = np.argwhere(ndimage.generate_binary_structure(rank=n_dim, connectivity=1))

    neighbors = []
    for offset in neighbour_offsets:
        neighbor = tuple(np.array(coord) + offset - 1)
        if all(0 <= n < s for n, s in zip(neighbor, shape)):
            neighbors.append(neighbor)

    return neighbors


class Blob:
    def __init__(self, seed_coord, seed_value, cell_array, max_size=None, n_dim=3):
        """
        Initialize a blob with a seed cell. The seed cell needs to be updated before the blob can grow. This is
        usually done by the initializing function.

        Parameters
        ----------
        seed_coord : tuple
            The initial coordinate of the seed.
        seed_value : int
            The value associated with the seed and the blob.
        seed_cell : Cell
            The cell object corresponding to the seed.
        max_size : int, optional
            The maximum size the blob can grow to.
        """
        self.n_dim = n_dim
        self.seed_value = seed_value
        self.max_size = max_size
        self.size = 0  # Start size at 0; it will be updated when adding cells.
        self.edge_cells = {seed_coord}  # Initialize an empty edge cells set
        self.new_cells = set()  # Initialize an empty new cells set
        self.growth_complete = False

        # Use the blob's own methods to manage the initial state
        self.add_new_cells(seed_coord)  # Adds the seed as a new cell and updates the size
        self.update_blob(cell_array)  # Initial edge is set based on the seed cell

    def can_grow(self):
        """
        Check if the blob can continue to grow.

        The blob can grow if it has not reached its maximum size and still has edge cells.

        Returns
        -------
        bool
            True if the blob can grow, False otherwise.
        """
        # TODO just ">" give me the right size at the end but I don't know why ...
        if self.max_size is not None and self.size > self.max_size:
            self.growth_complete = True
            return False
        if not self.edge_cells:  # Check if the edge set is empty
            self.growth_complete = True
            return False
        return True

    def add_new_cells(self, new_cells):
        """
        Add new cells to the blob.

        Parameters
        ----------
        new_cells : set
            A set of new cells to add to the blob.
        """
        if not isinstance(new_cells, set):
            new_cells = {new_cells}
        self.new_cells.update(new_cells)

    def update_edge(self, cell_array):
        """
        Update the edge cells of the blob using the neighbour array.
        Not made to be called directly, but rather handled internally.
        Parameters
        ----------
        cell_array : np.ndarray
            The array representing the cellular automaton grid.
        """
        for coords in self.new_cells.copy():
            neighbours = get_neighbours(coords, cell_array.shape)
            for neighbour in neighbours:
                if cell_array[neighbour].get_state() == 0:
                    self.edge_cells.add(coords)
                    break
        # flush the new cells
        self.new_cells.clear()

    def apply_cell_growth(self, cell_coord, cell_array, spawnable_neighbour_array, it_time, viscosity_factor=1):
        if not self.can_grow():
            return

        cell = cell_array[cell_coord]
        cell_state = cell.get_state()

        if cell_state <= 0 or cell_state != self.seed_value:
            return  # Skip non-spawnable cells, empty cells, and cells with different values because a blob can only grow with its seed value

        neighbour_offsets = np.argwhere(ndimage.generate_binary_structure(rank=self.n_dim, connectivity=1))
        np.random.shuffle(neighbour_offsets)

        # List to store candidates for growth
        candidate_neighbours = []
        weights = []

        for offset in neighbour_offsets:
            neighbour_coord = tuple(np.array(cell_coord) + offset - 1)
            if coord_in_array(neighbour_coord, cell_array) and cell_array[neighbour_coord].get_state() == 0:
                # Calculate the weight inversely proportional to the number of spawnable neighbors (i.e., less crowded = higher weight)
                weight = (1 / (1 + spawnable_neighbour_array[neighbour_coord])) * viscosity_factor
                candidate_neighbours.append(neighbour_coord)
                weights.append(weight)

        if candidate_neighbours:
            # Check if sum(weights) is zero or if any weight is non-finite
            weight_sum = sum(weights)
            if weight_sum == 0 or not np.all(np.isfinite(weights)):
                print(f"Invalid weights: {weights}")
                print(f"Sum of weights: {weight_sum}")
                return  # Skip this iteration to avoid errors

            # Normalize weights to sum to 1
            normalized_weights = [w / weight_sum for w in weights]

            # Use Python's built-in random.choices to select a neighbor based on the weights
            chosen_neighbour = random.choices(candidate_neighbours, weights=normalized_weights, k=1)[0]

            # Update the state of the chosen neighbor
            cell_array[chosen_neighbour].set_next_state(self.seed_value, it_time)
            self.add_new_cells(chosen_neighbour)

            # If the cell still has spawnable neighbors, keep it as an edge cell
            if spawnable_neighbour_array[cell_coord] > 0:
                self.edge_cells.add(cell_coord)
            else:
                self.edge_cells.discard(cell_coord)
        else:
            # If no neighbors are available, remove this cell from edge cells
            self.edge_cells.discard(cell_coord)

    def grow(self, cell_array, neighbour_array, it_time):
        """
        Grow the blob by applying the growth rule to the edge cells.

        Parameters
        ----------
        cell_array : np.ndarray
            The array representing the cellular automaton grid.
        neighbour_array : np.ndarray
            The array representing the number of neighbors for each cell.
        it_time : int
            The current iteration time.
        """
        for cell_coord in self.edge_cells.copy():
            self.apply_cell_growth(cell_coord, cell_array, neighbour_array, it_time)
        self.update_blob(cell_array)

    def update_blob(self, cell_array):
        if not self.new_cells:
            print("No new cells to update.")
            return

        for coord in self.new_cells:
            # Assuming `coord` is always valid if it reaches this point
            # we need to check that the blob can still grow
            if not self.can_grow():
                break
            cell_array[coord].update_state()
            self.size += 1
        self.update_edge(cell_array)

