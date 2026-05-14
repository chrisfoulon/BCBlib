import csv
import json
import random
import uuid
from argparse import ArgumentTypeError
from pathlib import Path

import numpy as np
import pandas as pd
from bcblib.tools.spreadsheet_io_utils import import_spreadsheet


def str_to_lower(value):
    """
    Convert a string to lowercase. Mostly useful for argparse arguments.
    Parameters
    ----------
    value

    Returns
    -------
    str : str
        The input string converted to lowercase.
    """
    return value.lower()


class EnhancedEncoder(json.JSONEncoder):
    """Enhanced JSON encoder with support for numpy types and UUIDs.
    
    Provides serialization support for:
    - UUID objects (converted to string representation)
    - NumPy numeric types and arrays  
    - Pandas Series and custom objects with to_dict() method
    """
    def default(self, obj):
        if isinstance(obj, uuid.UUID):
            return str(obj)  # Convert UUID objects to string representation
        elif isinstance(obj, np.number):
            return obj.item()  # Handles ALL numpy numeric types
        elif isinstance(obj, (np.ndarray, pd.Series)):
            return obj.tolist()  # Preserves array structure as lists
        elif hasattr(obj, 'to_dict'):  # Handle pandas/custom objects with to_dict method
            return obj.to_dict()
        try:
            # Let the base class handle it or throw an exception
            return super(EnhancedEncoder, self).default(obj)
        except TypeError:
            # Last resort: convert to string
            return str(obj)


class EnhancedNumpyEncoder(EnhancedEncoder):
    """Deprecated: Use EnhancedEncoder instead.
    
    This class is kept for backward compatibility but will be removed in future versions.
    Please use EnhancedEncoder which provides the same functionality with a more accurate name.
    """
    def __init__(self, *args, **kwargs):
        import warnings
        warnings.warn(
            "EnhancedNumpyEncoder is deprecated and will be removed in a future version. "
            "Use EnhancedEncoder instead, which provides the same functionality.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)


def uuid_object_hook(dct, uuid_keys=None, uuid_patterns=None):
    """Convert string UUIDs back to UUID objects where appropriate.
    
    This function automatically detects dictionary keys that contain UUID values
    and attempts to convert string values back to UUID objects. Non-UUID strings
    are left unchanged for backward compatibility.
    
    Parameters
    ----------
    dct : dict
        Dictionary from JSON deserialization
    uuid_keys : set or list, optional
        Specific key names that should be treated as UUIDs.
        Default: {'user_id', 'owner_id', 'workspace_id', 'model_id', 'granted_by'}
    uuid_patterns : list, optional
        Patterns for key names (e.g., ['_id', '_uuid']).
        Default: ['_id'] (matches keys ending with '_id')
        
    Returns
    -------
    dict
        Dictionary with UUID strings converted to UUID objects where appropriate
    """
    # Conservative defaults - exclude generic 'id' to avoid breaking common usage
    if uuid_keys is None:
        uuid_keys = {'user_id', 'owner_id', 'workspace_id', 'model_id', 'granted_by'}
    
    if uuid_patterns is None:
        uuid_patterns = ['_id']  # Only match *_id patterns, not bare 'id'
    
    # Convert to set for faster lookups
    if not isinstance(uuid_keys, set):
        uuid_keys = set(uuid_keys)
    
    for key, value in dct.items():
        should_convert = False
        
        # Check explicit key names
        if key in uuid_keys:
            should_convert = True
        
        # Check patterns
        for pattern in uuid_patterns:
            if key.endswith(pattern):
                should_convert = True
                break
        
        # Convert if criteria met and value is string
        if should_convert and isinstance(value, str):
            try:
                dct[key] = uuid.UUID(value)
            except (ValueError, TypeError):
                # Not a valid UUID string, keep as string for backward compatibility
                pass
    
    return dct


def open_json(path, convert_uuids=True, uuid_keys=None, uuid_patterns=None):
    """Load JSON from file with optional UUID conversion.
    
    Parameters
    ----------
    path : str or Path
        Path to JSON file
    convert_uuids : bool, default=True
        Whether to automatically convert UUID strings to UUID objects
    uuid_keys : set or list, optional
        Specific key names that should be treated as UUIDs.
        Only used if convert_uuids=True.
    uuid_patterns : list, optional
        Patterns for key names (e.g., ['_id', '_uuid']).
        Only used if convert_uuids=True.
        
    Returns
    -------
    dict or list
        Parsed JSON data with optional UUID conversion
    """
    with open(path, 'r') as j:
        if convert_uuids:
            # Create a partial function with the custom parameters
            def custom_uuid_hook(dct):
                return uuid_object_hook(dct, uuid_keys=uuid_keys, uuid_patterns=uuid_patterns)
            return json.load(j, object_hook=custom_uuid_hook)
        else:
            return json.load(j)


def save_json(path, d):
    """Save data to JSON file with enhanced encoding support.
    
    Parameters
    ----------
    path : str or Path
        Path to save JSON file
    d : dict or list
        Data to serialize to JSON
        
    Returns
    -------
    None
    """
    with open(path, 'w+') as j:
        return json.dump(d, j, indent=4, cls=EnhancedEncoder)


def save_list(path, li):
    np.savetxt(path, li, delimiter='\n', fmt='%s')


def file_to_list(file_path, delimiter=' '):
    file_path = Path(file_path)
    if not file_path.is_file():
        raise ValueError(f'{file_path} does not exist.')
    if file_path.name.endswith('.csv'):
        spreadsheet = pd.read_csv(file_path)
        str_list = spreadsheet.values.flatten()
    elif file_path.name.endswith('.xlsx') or file_path.name.endswith('.xls'):
        spreadsheet = pd.read_excel(file_path)
        str_list = spreadsheet.values.flatten()
    else:
        # default delimiter is ' ', it might need to be changed
        str_list = np.loadtxt(str(file_path), dtype=str, delimiter=delimiter)
    return str_list


def is_index(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def list_of_file_paths_from_spreadsheet(spreadsheet_path, column_identifier, root_folder=None):
    if not spreadsheet_path.is_file():
        raise ValueError(f'{spreadsheet_path} does not exist.')
    spreadsheet = import_spreadsheet(spreadsheet_path, header=None if is_index(column_identifier) else 0)
    if is_index(column_identifier):
        column_index = int(column_identifier)
        file_paths = spreadsheet.iloc[:, column_index].values
    else:
        column_name = column_identifier
        file_paths = spreadsheet[column_name].values
    if root_folder is not None:
        root_folder = Path(root_folder)
        file_paths = [str(root_folder / Path(f)) for f in file_paths]
    else:
        file_paths = [str(Path(f)) for f in file_paths]
    return file_paths


def parse_file_list_argument(argument, recursive_file_search=False, file_types=None, arg_separator=','):
    input_list = str(argument).split(arg_separator)
    if file_types is None:
        file_types = [['.nii', '.nii.gz'], ['.jpg', '.png']]
    if len(input_list) == 1:
        # Only folder path
        folder_path = Path(input_list[0]).resolve()
        flat_file_types = [item for sublist in file_types for item in sublist]
        if recursive_file_search:
            # find the files recursively that correspond to the file types
            paths_list = [str(f) for f in folder_path.rglob('*') if ''.join(f.suffixes) in flat_file_types]
            # if there are different file type (.nii and .nii.gz are together and .jpg and .png are together) raise
            # an error
        else:
            paths_list = [str(f) for f in folder_path.iterdir() if ''.join(f.suffixes) in flat_file_types]
        unique_extensions = set([''.join(f.suffixes) for f in folder_path.rglob('*')])
        if len(unique_extensions) > 1:
            # if file_types is a list, there must only be one unique extension, otherwise,
            # there should be extensions from only one list in file_types
            if len(file_types) == 1:
                raise ArgumentTypeError(f"Multiple file types found in {folder_path}: {unique_extensions}")
            else:
                file_class_found = 0
                for file_class in file_types:
                    if len(set(file_class).intersection(unique_extensions)) > 0:
                        file_class_found += 1
                if file_class_found > 1:
                    raise ArgumentTypeError(f"Conflicting file types found in {folder_path}: {unique_extensions}")
        if len(paths_list) == 0:
            raise ArgumentTypeError(f"No files found in {folder_path}")
        print(f'Found {len(paths_list)} files in {folder_path}')
        # TODO add BIDS dataset detection and handling

    elif len(input_list) == 2:
        first, second = input_list
        if Path(first).is_dir():
            # Folder path and file
            folder_path = Path(first).resolve()
            paths_list = file_to_list(second)
            paths_list = [str(folder_path.joinpath(p)) for p in paths_list]
            # check if the paths exist
            for p in paths_list:
                if not Path(p).exists():
                    raise ArgumentTypeError(f"Path {p} does not exist")
        elif Path(second).is_dir():
            # File and folder path
            folder_path = Path(second).resolve()
            paths_list = file_to_list(first)
            paths_list = [str(folder_path.joinpath(p)) for p in paths_list]
            # check if the paths exist
            for p in paths_list:
                if not Path(p).exists():
                    raise ArgumentTypeError(f"Path {p} does not exist")
        else:
            if Path(first).is_file() and Path(second).is_file():
                # Two files
                raise ArgumentTypeError(f"Two files provided: {first}, {second}")
            else:
                if Path(first).is_file():
                    # Spreadsheet and column
                    spreadsheet_path = Path(first).resolve()
                    column_identifier = second
                elif Path(second).is_file():
                    # Spreadsheet and column
                    spreadsheet_path = Path(second).resolve()
                    column_identifier = first
                else:
                    raise ArgumentTypeError(f"Invalid input: {first}, {second}")
            paths_list = list_of_file_paths_from_spreadsheet(spreadsheet_path, column_identifier)

    elif len(input_list) == 3:
        # Identify the folder path, spreadsheet, and column identifier
        folder_path = spreadsheet_path = column_identifier = None
        for item in input_list:
            if Path(item).is_dir():
                folder_path = Path(item).resolve()
            elif Path(item).is_file():
                spreadsheet_path = Path(item).resolve()
            else:
                column_identifier = item

        if folder_path is None or spreadsheet_path is None or column_identifier is None:
            raise ArgumentTypeError(f"Invalid input: {input_list}")
        paths_list = list_of_file_paths_from_spreadsheet(spreadsheet_path, column_identifier, folder_path)

    else:
        raise ArgumentTypeError(f"Invalid number of sub-arguments: {len(input_list)}, expected 1, 2, or 3")
    # TODO create a test set to test each of the cases

    return paths_list


def split_dict(d, chunk_size, output_dir=None, output_pref=None):
    """
    Split a dictionary into chunks of a given size and save them to disk.
    Parameters
    ----------
    d
    chunk_size
    output_dir
    output_pref

    Returns
    -------

    """
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        if not output_pref:
            output_pref = 'chunk'
        # Check whether the output_dir does not contain files with the same prefix followed by _ and a number and .json
        for f in output_dir.iterdir():
            if f.name.startswith(output_pref) and f.name.endswith('.json'):
                try:
                    int(f.name.split('_')[-1].split('.')[0])
                    print(f'ERROR: Output directory {output_dir} already contains files matching the output filenames')
                    return
                except ValueError:
                    pass

    print(f'Length of the input dictionary: {len(d)}')
    keys_list = list(d.keys())
    i = -1
    end_index = 0
    chunk_list = []
    for i in range(len(keys_list) // chunk_size):
        begin_index = chunk_size * i
        end_index = chunk_size * (1 + i)
        chunk = {k: d[k] for k in keys_list[begin_index:end_index]}
        if output_dir is not None:
            print(f'Saving chunk {i} to {output_dir.joinpath(f"{output_pref}_{i}.json")}')
            save_json(output_dir.joinpath(f'{output_pref}_{i}.json'), chunk)
        print(f'Chunk {i} length: {len(chunk)}')
        chunk_list.append(chunk)
    last_chunk = {k: d[k] for k in keys_list[end_index:]}
    print(f'Last chunk length: {len(last_chunk)}')
    if output_dir is not None:
        save_json(output_dir.joinpath(f'{output_pref}_{i + 1}.json'),
                  last_chunk)
    chunk_list.append(last_chunk)
    return chunk_list


def partition_values_to_sizes_with_margin(values, sizes, margin):
    def backtrack(index, current_partition, used):
        if index == len(values):
            # Check if all partitions are valid according to the margin
            for i in range(len(sizes)):
                current_sum = sum(values[j] for j in current_partition[i])
                if not (lower_bounds[i] <= current_sum <= upper_bounds[i]):
                    return False
            return all(used)  # Ensure all values are used

        # Shuffle the indices to introduce randomness
        size_indices = list(range(len(sizes)))
        random.shuffle(size_indices)

        for i in size_indices:
            current_sum = sum(values[j] for j in current_partition[i])
            if current_sum + values[index] <= upper_bounds[i]:
                # Add the index of the value to the current bin
                current_partition[i].append(index)
                used[index] = True

                # Recur to assign the next value
                if backtrack(index + 1, current_partition, used):
                    return True

                # If it doesn't work, backtrack
                current_partition[i].pop()
                used[index] = False

        return False  # No valid partition was found

    # Calculate the lower and upper bounds for each size based on the margin
    lower_bounds = [size * (1 - margin) for size in sizes]
    print(f'Lower bounds: {lower_bounds}')
    upper_bounds = [size * (1 + margin) for size in sizes]
    print(f'Upper bounds: {upper_bounds}')

    # Sort indices of values in descending order based on value
    value_indices = sorted(range(len(values)), key=lambda i: values[i], reverse=True)

    # Initialize partitions (one list for each size)
    current_partition = [[] for _ in sizes]

    # Track which values have been used
    used = [False] * len(values)

    # Start the backtracking process
    if backtrack(0, current_partition, used):
        # Return the partition as a list of indices
        return current_partition
    else:
        return None  # No valid partition found
