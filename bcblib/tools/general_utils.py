import json
from pathlib import Path

import numpy as np


def open_json(path):
    with open(path, 'r') as j:
        return json.load(j)


def save_json(path, d):
    with open(path, 'w+') as j:
        return json.dump(d, j, indent=4)


def save_list(path, li):
    np.savetxt(path, li, delimiter='\n', fmt='%s')


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
