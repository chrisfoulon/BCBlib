import csv
import json
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


def open_json(path):
    with open(path, 'r') as j:
        return json.load(j)


def save_json(path, d):
    with open(path, 'w+') as j:
        return json.dump(d, j, indent=4)


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
