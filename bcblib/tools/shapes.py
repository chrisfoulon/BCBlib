import numpy as np
from scipy import ndimage


def create_hypersphere_shape(radius, n_dim=3):
    """
    Create an ND binary array with a hyperspherical shape.

    Parameters
    ----------
    radius : int
        The radius of the hypersphere.
    n_dim : int
        The number of dimensions. Default is 3.

    Returns
    -------
    np.ndarray
        An ND binary array with a hyperspherical shape.
    """
    # Define the grid for each dimension
    grid = np.ogrid[[slice(-radius, radius + 1)] * n_dim]

    # Create a binary hypersphere
    distance_squared = sum((g ** 2 for g in grid))
    hypersphere = distance_squared <= radius ** 2

    return hypersphere.astype(int)


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
    >>> from bcblib.tools.shapes import create_shapes_in_arr
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