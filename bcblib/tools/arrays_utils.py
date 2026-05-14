import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import label
from scipy.stats import mannwhitneyu, ttest_ind
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm


def coord_in_array(coord, array):
    """
    Check if a coordinate is within the bounds of a numpy array.

    Parameters
    ----------
    coord: tuple
    array: np.ndarray

    Returns
    -------
    bool
    """
    return all(0 <= c < array.shape[i] for i, c in enumerate(coord))


def find_centroid_and_check(array, check_inside=True):
    """
    Computes the centroid of a contiguous cluster of continuous values in a numpy array.
    Optionally checks if the centroid falls within the cluster and, if not, returns the closest coordinate in the cluster.

    Parameters:
    array (np.ndarray): Numpy array with the cluster of continuous values.
    check_inside (bool): Whether to check if the centroid falls within the cluster. Default is True.

    Returns:
    tuple: Centroid coordinates (x, y) or the closest coordinate in the cluster if the centroid is not within the cluster.
    """

    # Ensure input is a numpy array
    if not isinstance(array, np.ndarray):
        raise ValueError("Input must be a numpy array")

    # Get the coordinates of the non-zero (cluster) points
    cluster_points = np.transpose(np.nonzero(array))

    # Calculate the centroid
    centroid = np.mean(cluster_points, axis=0)

    if check_inside:
        # Check if the centroid is within the cluster
        centroid_int = np.round(centroid).astype(int)
        if array[centroid_int[0], centroid_int[1]] != 0:
            return tuple(centroid)  # Centroid is within the cluster

        # If not, find the closest coordinate in the cluster
        distances = np.linalg.norm(cluster_points - centroid, axis=1)
        closest_index = np.argmin(distances)
        closest_point = cluster_points[closest_index]
        return tuple(closest_point)
    else:
        return tuple(centroid)


def separate_clusters_and_extract_coords(array, coords_array):
    """
    Separates the clusters in the input array and extracts the coordinates of the elements in each cluster.

    Parameters
    ----------
    array : np.ndarray
        The input array containing the clusters.
    coords_array : np.ndarray
        An array containing some coordinates of points in the input array.

    Returns
    -------
    clusters : list of np.ndarray
        List of numpy arrays of the same size as the input array, each containing only the values of a given cluster.
    indices_lists : list of list of tuple
        List of lists of coordinates for each cluster.
    """
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    binary_array = np.where(np.isfinite(array) & (array != 0), 1, 0)
    # Label the clusters in the input array
    labeled_array, num_clusters = label(binary_array)

    # Initialize lists to store the separated clusters and the corresponding indices
    clusters = []
    indices_lists = []

    # For each cluster, find the corresponding indices in the coords_array
    for i in range(1, num_clusters + 1):
        # Create a mask for the current cluster
        cluster_mask = (labeled_array == i)

        # Extract the cluster values from the input array
        cluster_data = np.zeros_like(array)
        cluster_data[cluster_mask] = array[cluster_mask]
        clusters.append(cluster_data)

        # Find the coordinates in coords_array that are within the current cluster
        cluster_coords_indices = []
        for ind, coord in enumerate(coords_array):
            if cluster_mask[tuple(coord)]:
                cluster_coords_indices.append(ind)

        indices_lists.append(cluster_coords_indices)

    return clusters, indices_lists


def normalize_data(data, method='min-max', feature_range=(0, 1)):
    """
    Normalize data with different methods.
    
    Parameters:
    -----------
    data : numpy.ndarray
        The data to normalize.
    method : str
        The normalization method: 'min-max', 'z-score', 'robust', or 'none'.
    feature_range : tuple, default=(0, 1)
        Range for the normalized data for min-max scaling.
        
    Returns:
    --------
    numpy.ndarray
        Normalized data.
    dict
        Parameters used for normalization.
    """
    if method == 'none':
        return data, {'method': 'none'}
    
    if method == 'min-max':
        # Min-max normalization
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        range_vals = max_vals - min_vals
        # Avoid division by zero
        range_vals[range_vals == 0] = 1.0
        
        normalized_data = (data - min_vals) / range_vals
        
        # Scale to the desired feature range
        a, b = feature_range
        normalized_data = a + normalized_data * (b - a)
        
        params = {
            'method': 'min-max',
            'min_vals': min_vals,
            'max_vals': max_vals,
            'feature_range': feature_range
        }
        
    elif method == 'z-score':
        # Z-score normalization
        mean_vals = np.mean(data, axis=0)
        std_vals = np.std(data, axis=0)
        # Avoid division by zero
        std_vals[std_vals == 0] = 1.0
        
        normalized_data = (data - mean_vals) / std_vals
        
        params = {
            'method': 'z-score',
            'mean_vals': mean_vals,
            'std_vals': std_vals
        }
    elif method == 'robust':
        # Robust scaling (less sensitive to outliers)
        scaler = RobustScaler()
        normalized_data = scaler.fit_transform(data)
        params = {
            'method': 'robust',
            'scaler': scaler
        }
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized_data, params


def inverse_normalize_data(normalized_data, params):
    """
    Reverse the normalization applied by normalize_data.
    
    Parameters:
    -----------
    normalized_data : numpy.ndarray
        The normalized data to inverse transform.
    params : dict
        The parameters dictionary returned by normalize_data.
        
    Returns:
    --------
    numpy.ndarray
        The original data (before normalization).
    """
    method = params.get('method', 'none')
    
    if method == 'none':
        return normalized_data
    
    elif method == 'min-max':
        min_vals = params['min_vals']
        max_vals = params['max_vals']
        a, b = params['feature_range']
        
        # Rescale from [a,b] to [0,1]
        data_01 = (normalized_data - a) / (b - a)
        
        # Rescale from [0,1] to original range
        original_data = min_vals + data_01 * (max_vals - min_vals)
        
    elif method == 'z-score':
        mean_vals = params['mean_vals']
        std_vals = params['std_vals']
        
        # Reverse z-score normalization
        original_data = normalized_data * std_vals + mean_vals
        
    elif method == 'robust':
        # Use the scaler's inverse_transform method
        scaler = params['scaler']
        original_data = scaler.inverse_transform(normalized_data)
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return original_data