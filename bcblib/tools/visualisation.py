from pathlib import Path
import subprocess
from typing import Collection, Union
import os
import re
import shutil

import numpy as np
from bcblib.tools.nifti_utils import is_nifti


def mricron_display_old(paths: Union[Union[str, bytes, os.PathLike], Collection[Union[str, bytes, os.PathLike]]], *args):
    if isinstance(paths, str):
        paths = [paths]

    if len(paths) == 1:
        mricron_command = ['mricron', paths[0], *args]
    else:
        overlays = []
        for path in paths[1:]:
            overlays.append('-o')
            overlays.append(path)
        mricron_command = ['mricron', paths[0], *overlays, *args]
    print('Mricron command: "{}"'.format(mricron_command))
    process = subprocess.run(mricron_command,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             universal_newlines=True)
    return process


def mricron_display(path: Union[str, bytes, os.PathLike],
                    options: Collection[Union[str, bytes, os.PathLike]] = None):
    opt = []
    if options is not None:
        opt = list(options)
    mricron_command = ['mricron', path] + opt
    print('Mricron command: "{}"'.format(mricron_command))
    process = subprocess.run(mricron_command,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             universal_newlines=True)
    return process


def list_folder_patterns(folder: Union[str, bytes, os.PathLike], fname_patterns: Union[str, Collection[str]] = None):
    patterns = fname_patterns
    if fname_patterns is None:
        patterns = ['*']
    if isinstance(fname_patterns, str):
        patterns = [fname_patterns]

    f_list = [f for f in Path(folder).iterdir() if is_nifti(f)]
    f_lists = np.array([[f for f in f_list if re.search(p, f.name)] for p in patterns])
    for el in f_lists:
        if len(el) != f_lists[0]:
            raise ValueError('Impossible to match the images and overlays. '
                             'Found a different number of files for each pattern.')
    return f_lists


def loop_display_folder(folder: Union[str, bytes, os.PathLike], fname_patterns: Union[str, Collection[str]] = None)\
        -> None:
    f_lists = list_folder_patterns(folder, fname_patterns)
    for i in range(len(f_lists[0])):
        images = f_lists[:, i]
        mricron_display(images)
        resp = input('Next one [enter]. Quit? [quit/exit/e/q]')
        if resp in ['q', 'e', 'exit', 'quit']:
            return


def loop_display_sort_folder(folder: Union[str, bytes, os.PathLike],
                             keep_folder: Union[str, bytes, os.PathLike],
                             reject_folder: Union[str, bytes, os.PathLike],
                             fname_patterns: Union[str, Collection[str]] = None,
                             check_output_folder: bool = True):
    keep_keys = ['', 'y', 'yes', 'k', 'keep', 'a', 'accept']
    reject_keys = ['r', 'reject', 'n', 'no']
    quit_keys = ['q', 'e', 'exit', 'quit']
    os.makedirs(keep_folder, exist_ok=True)
    os.makedirs(reject_folder, exist_ok=True)
    f_lists = list_folder_patterns(folder, fname_patterns)
    for i in range(len(f_lists[0])):
        images = f_lists[:, i]
        mricron_display(images)
        resp = input('Keep the image and the overlays? [Y(yes), n(no), k(keep), r(reject), e(exit), q(quit)]')
        for f in images:
            if check_output_folder and (Path(keep_folder, f.name).is_file() or Path(reject_folder, f.name)):
                continue
            while resp is not None:
                if resp.lower() in keep_keys:
                    shutil.copyfile(f, Path(keep_folder, f.name))
                    resp = None
                if resp.lower() in reject_keys:
                    shutil.copyfile(f, Path(reject_folder, f.name))
                    resp = None
                if resp.lower() in quit_keys:
                    return
                else:
                    print('Error: Wrong key entered')
                    resp = input('Keep the image and the overlays? [Y(yes), n(no), k(keep), r(reject)]')
