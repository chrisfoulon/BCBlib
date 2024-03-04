from pathlib import Path
import subprocess
from typing import Collection, Union
import os
import re
import shutil
from pprint import pprint
import random

import numpy as np
import pandas as pd
import nibabel as nib
from matplotlib import pyplot as plt
import seaborn as sns

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from bcblib.tools.nifti_utils import is_nifti, get_centre_of_mass
from bcblib.tools.general_utils import open_json, save_json


def mricron_display_old(paths: Union[Union[str, bytes, os.PathLike], Collection[Union[str, bytes, os.PathLike]]],
                        *args):
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


def display_img(img, over1=None, over2=None, display='mricron', coord=None, print_cmd=True):
    # TODO change img so it can be a list of images
    data = nib.load(img).get_fdata()
    if display == 'mricron':
        img_opt = ['-x', '-c', '-0',
                   '-l', '{:.4f}'.format(np.min(data)), '-h', '{:.4f}'.format(np.max(data)), '-b', '60']
    elif display == 'fsleyes':
        # img_opt = ['-cm', 'red', '-a', '40', ]
        if coord is not None:
            coord_list = ['-vl'] + [str(c) for c in coord]
            fsleyes_command = ['fsleyes'] + coord_list + [str(img)]
        else:
            fsleyes_command = ['fsleyes', str(img)]
        if over1 is not None:
            fsleyes_command += [str(over1), '-cm', 'red', '-a', '40']
        if over2 is not None:
            fsleyes_command += [str(over2), '-cm', 'green', '-a', '40']
        fsleyes_command = fsleyes_command  # + img_opt
        if print_cmd:
            print('Fsleyes command: "{}"'.format(' '.join(fsleyes_command)))
        # if "DISPLAY" not in os.environ:
        #     os.environ["DISPLAY"] = ':1'
        process = subprocess.run(fsleyes_command,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 universal_newlines=True
                                 )
        if process.stderr != '':
            print('Error/Warning during fsleyes execution')
            print('exit status:', process.returncode)
            print('stderr:', process.stderr)
        return process
    else:
        raise ValueError(f'{display} display tool unknown')
    label_opt = []
    change_dir = True
    current_dir = os.getcwd()
    relative_dir = Path(img).parent
    if over1:
        change_dir = change_dir and Path(over1).parent == Path(img).parent
        print(f'Non zero voxels in the label: {np.count_nonzero(nib.load(over1).get_fdata())})')
    if over2:
        change_dir = change_dir and Path(over2).parent == Path(img).parent
    if change_dir:
        img = Path(img).name
        over1 = Path(over1).name if over1 else None
        over2 = Path(over2).name if over2 else None
    if over1:
        # -c -1 means red
        label_opt = ['-o', str(over1), '-c', '-1', '-t', '-1']
    seg_opt = []
    if over2:
        # -c -3 means green
        seg_opt = ['-o', str(over2), '-c', '-3', '-t', '-1']
    mricron_options = img_opt + label_opt + seg_opt
    mricron_command = ['mricron', str(img)] + mricron_options
    if print_cmd:
        print('Mricron command: "{}"'.format(mricron_command))
    os.chdir(relative_dir)
    process = subprocess.run(mricron_command,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             universal_newlines=True)
    os.chdir(current_dir)
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


def loop_display_folder(folder: Union[str, bytes, os.PathLike], fname_patterns: Union[str, Collection[str]] = None) \
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


def check_and_annotate_segmentation(seg_dict, output_path, images_root='', label_dict_path=None, spreadsheets=None,
                                    matching_columns=None, info_columns=None, display='fsleyes', seg_coord=False,
                                    zfill_matching_col=True, highlight_terms_list=None, checking_key='manual_check',
                                    checking_value=None, randomise_lbl_seg=False, no_coord=False, lock_labels=False):
    """
    Check and annotate a segmentation dictionary.
    Parameters
    ----------
    seg_dict : dict or str or Path
        Segmentation dictionary or path to a json file containing the segmentation dictionary.
    output_path : str or Path
        Path to the output json file.
    images_root : str or Path
        Path to the root folder containing the images.
    label_dict_path : str or Path
        Path to the json file containing the label dictionary.
    spreadsheets : list of str or Path
        List of paths to the spreadsheets containing the information about the images.
    matching_columns : list of str
        List of columns in the spreadsheets that will be used to match the images.
    info_columns : list of str
        List of columns in the spreadsheets that will be used to display information about the images.
    display : str
        Display software to use. Can be 'fsleyes' or 'mricron'.
    seg_coord : bool
        If True, the coordinates of the segmentation will be displayed.
    zfill_matching_col : bool
        If True, the matching columns will be zfilled to 4 digits.
    highlight_terms_list : list of str
        List of terms to highlight in the spreadsheets.
    checking_key : str
        Key in the seg_dict that will be used to check the segmentation.
    checking_value : str
        Value of the checking_key that will be used to check the segmentation.
    randomise_lbl_seg : bool
        If True, the labels and the segmentations will be randomly shuffled.
    no_coord : bool
        If True, the coordinates will not be displayed.
    lock_labels : bool
        If True, the labels will not be shuffled.

    Returns
    -------
    output_dict : dict
        Segmentation dictionary.
    """
    # TODO CLEAN THAT UP
    pd.set_option('display.max_colwidth', None)
    # Test for the input dict
    seg_dict_path = None
    if not isinstance(seg_dict, dict):
        seg_dict_path = seg_dict
        seg_dict = open_json(seg_dict)
    else:
        if checking_value is not None:
            raise ValueError('You cannot check a value in seg_dict if seg_dict is not a json file')
    # Check output path if we are not in checking_value mode
    if not Path(output_path).parent.is_dir() and checking_value is None:
        raise ValueError(f'Parent folder of {output_path} must be an existing directory')
    # Making sure the user is aware of the overwriting
    if checking_value is not None:
        resp = input(f'WARNING: You selected a value to check, the output_path is then ignored and the input seg_dict '
                     f'will be modified and saved. Are you sure you want to continue? [y/n]')
        if resp.lower() not in ['y', 'yes']:
            print('Abort mission! (ノಠ益ಠ)ノ彡┻━┻')
            return
    # Check if label_dict_path exists, if not, we create it at the given path
    if Path(label_dict_path).exists():
        label_dict = open_json(label_dict_path)
    else:
        print("Creating dictionary containing the labels. It will be filled up by the user's answers")
        label_dict = {}
    # Now we open the spreadsheet(s)
    if spreadsheets is not None:
        if not isinstance(spreadsheets, list):
            spreadsheets = [spreadsheets]
        for ind, spreadsheet in enumerate(spreadsheets):
            if not isinstance(spreadsheet, pd.DataFrame):
                if Path(spreadsheet).name.endswith('.csv'):
                    spreadsheets[ind] = pd.read_csv(spreadsheet, header=0)
                else:
                    spreadsheets[ind] = pd.read_excel(spreadsheet, header=0)
    else:
        spreadsheets = []
    # We test that the columns in matching_columns are in the spreadsheets
    if matching_columns is not None:
        if not isinstance(matching_columns, list):
            matching_columns = [matching_columns]
        for ind, matching_column in enumerate(matching_columns):
            if matching_column not in spreadsheets[ind].columns:
                raise ValueError(f'{matching_column} not in spreadsheet number {ind}')
            if zfill_matching_col:
                spreadsheets[ind][matching_column] = [
                    str(value).zfill(8) for value in spreadsheets[ind][matching_column]]
    else:
        matching_columns = []
    # Same for info_columns
    if info_columns is not None:
        if not isinstance(info_columns, list):
            info_columns = [info_columns]
        for ind, info_column in enumerate(info_columns):
            if info_column not in spreadsheets[ind].columns:
                raise ValueError(f'{info_column} not in spreadsheet number {ind}')
    else:
        info_columns = []
    if len(spreadsheets) != len(matching_columns) != len(info_columns):
        raise ValueError('There must be the same number of spreadsheets, matching_columns and info_columns!')
    if Path(output_path).is_file():
        output_dict = open_json(output_path)
    else:
        output_dict = {}
    try:
        save_json(output_path, output_dict)
    except Exception as e:
        print(f'Exception caught when trying to save {output_path}')
        raise e
    # Filtering the values
    if checking_value is None:
        # Here we want to avoid having to check images already labelled
        to_check_keys = [k for k in seg_dict if k not in output_dict]
    else:
        # Here we want to select the images with the given checking_value
        to_check_keys = [k for k in seg_dict if seg_dict[k][checking_key] == checking_value]

    # Try catch clause to avoid losing anything when a ctr+c is hit
    try:
        for counter, k in enumerate(to_check_keys):
            """
            Check if the paths in seg_dict exist
            """
            pid = seg_dict[k]['PatientID']
            if not Path(seg_dict[k]['b1000']).exists():
                b1000 = Path(images_root, seg_dict[k]['b1000'])
            else:
                b1000 = seg_dict[k]['b1000']
            label = None
            if 'label' in seg_dict[k]:
                if not Path(seg_dict[k]['label']).exists():
                    label = Path(images_root, seg_dict[k]['label'])
                else:
                    label = seg_dict[k]['label']
            seg = None
            if 'segmentation' in seg_dict[k]:
                if not Path(seg_dict[k]['segmentation']).exists():
                    seg = Path(images_root, seg_dict[k]['segmentation'])
                else:
                    seg = seg_dict[k]['segmentation']

            # TODO if randomise_lbl_seg we need the coordinates of each lesion centre
            if display == 'fsleyes' and not no_coord:
                if (seg is not None and seg_coord) or label is None:
                    coord = get_centre_of_mass(seg, round_coord=True).astype(int)
                else:
                    coord = get_centre_of_mass(label, round_coord=True).astype(int)
            else:
                coord = None
            if randomise_lbl_seg:
                if label is not None and seg is not None:
                    if random.random() > .5:
                        label, seg = seg, label
            show_image = True
            show_report = True
            print(f'############### IMAGE NUMBER {counter}/{len(to_check_keys)} #################')
            while show_image or show_report:
                if show_report:
                    for ind, spreadsheet in enumerate(spreadsheets):
                        matched_entries = spreadsheet[spreadsheet[matching_columns[ind]] == pid][info_columns[ind]]
                        print(f'Spreadsheet number {ind} ###############')
                        for entry_ind, entry in enumerate(matched_entries):
                            if highlight_terms_list is not None:
                                for term in highlight_terms_list:
                                    # TODO
                                    break
                            print(f'###### {entry_ind}: {entry}')
                if show_image:
                    display_img(b1000, label, seg, display, coord, print_cmd=not randomise_lbl_seg)
                # if randomise_lbl_seg we need two different ratings for each image, so we count them.
                count_answer = 0
                pprint(label_dict)
                print('Select a label from the list above using either the number or the label itself or ')
                print('quit [exit]: to quit and save')
                print('image [display]: to display the image again and ask for an answer again')
                print('report: to show the report(s) information again and ask for an answer again')
                resp = input('Answer: ')
                show_image = False
                show_report = False
                if resp.lower() == 'report':
                    show_report = True
                elif resp.lower() in ['display', 'image']:
                    show_image = True
                elif resp.lower() in ['quit', 'exit']:
                    save_json(output_path, output_dict)
                    return output_dict
                else:
                    # TODO: if randomise_lbl_seg, there should be 2 ratings for each image
                    value = None
                    if resp in label_dict.values():
                        value = resp
                        output_dict[k] = value
                        save_json(output_path, output_dict)
                    elif resp in label_dict.keys():
                        value = label_dict[resp]
                        output_dict[k] = value
                        save_json(output_path, output_dict)
                    else:
                        yn = input(f'{resp} is neither a label nor a label code.'
                                   f'\nDo you want to add this as a new label [Y/n]')
                        if yn.lower() == 'n':
                            print('Alright! Showing the image again!')
                            show_image = True
                        else:
                            value = resp
                            output_dict[k] = value
                            save_json(output_path, output_dict)
                            label_dict.update({str(len(label_dict)): resp})
                            save_json(label_dict_path, label_dict)
                    if value is not None and checking_value is not None:
                        seg_dict[k][checking_key] = value
                        save_json(seg_dict_path, seg_dict)
        save_json(output_path, output_dict)
    except KeyboardInterrupt as e:
        print('Script interrupted using keyboard interruption. Saving the output dictionary.')
        save_json(output_path, output_dict)
        raise e
    return output_dict


def rate_two_segmentations(seg_dict, output_path, images_root='', label_dict_path=None, spreadsheets=None,
                           matching_columns=None, info_columns=None, display='fsleyes',
                           zfill_matching_col=True, randomise_lbl_seg=False, no_coord=False):
    """
    Check and annotate a segmentation dictionary.
    Parameters
    ----------
    seg_dict : dict or str or Path
        Segmentation dictionary or path to a json file containing the segmentation dictionary.
    output_path : str or Path
        Path to the output json file.
    images_root : str or Path
        Path to the root folder containing the images.
    label_dict_path : str or Path
        Path to the json file containing the label dictionary.
    spreadsheets : list of str or Path
        List of paths to the spreadsheets containing the information about the images.
    matching_columns : list of str
        List of columns in the spreadsheets that will be used to match the images.
    info_columns : list of str
        List of columns in the spreadsheets that will be used to display information about the images.
    display : str
        Display software to use. Can be 'fsleyes' or 'mricron'.
    seg_coord : bool
        If True, the coordinates of the segmentation will be displayed.
    zfill_matching_col : bool
        If True, the matching columns will be zfilled to 4 digits.
    highlight_terms_list : list of str
        List of terms to highlight in the spreadsheets.
    checking_key : str
        Key in the seg_dict that will be used to check the segmentation.
    checking_value : str
        Value of the checking_key that will be used to check the segmentation.
    randomise_lbl_seg : bool
        If True, the labels and the segmentations will be randomly shuffled.
    no_coord : bool
        If True, the coordinates will not be displayed.
    lock_labels : bool
        If True, the labels will not be shuffled.

    Returns
    -------
    output_dict : dict
        Segmentation dictionary.
    """
    # TODO CLEAN THAT UP
    pd.set_option('display.max_colwidth', None)
    # Test for the input dict
    seg_dict_path = None
    if not isinstance(seg_dict, dict):
        seg_dict_path = seg_dict
        seg_dict = open_json(seg_dict)
    # Check if label_dict_path exists, if not, we create it at the given path
    if Path(label_dict_path).exists():
        label_dict = open_json(label_dict_path)
    else:
        print("Creating dictionary containing the labels. It will be filled up by the user's answers")
        label_dict = {}
    # Now we open the spreadsheet(s)
    if spreadsheets is not None:
        if not isinstance(spreadsheets, list):
            spreadsheets = [spreadsheets]
        for ind, spreadsheet in enumerate(spreadsheets):
            if not isinstance(spreadsheet, pd.DataFrame):
                if Path(spreadsheet).name.endswith('.csv'):
                    spreadsheets[ind] = pd.read_csv(spreadsheet, header=0)
                else:
                    spreadsheets[ind] = pd.read_excel(spreadsheet, header=0)
    else:
        spreadsheets = []
    # We test that the columns in matching_columns are in the spreadsheets
    if matching_columns is not None:
        if not isinstance(matching_columns, list):
            matching_columns = [matching_columns]
        for ind, matching_column in enumerate(matching_columns):
            if matching_column not in spreadsheets[ind].columns:
                raise ValueError(f'{matching_column} not in spreadsheet number {ind}')
            if zfill_matching_col:
                spreadsheets[ind][matching_column] = [
                    str(value).zfill(8) for value in spreadsheets[ind][matching_column]]
    else:
        matching_columns = []
    # Same for info_columns
    if info_columns is not None:
        if not isinstance(info_columns, list):
            info_columns = [info_columns]
        for ind, info_column in enumerate(info_columns):
            if info_column not in spreadsheets[ind].columns:
                raise ValueError(f'{info_column} not in spreadsheet number {ind}')
    else:
        info_columns = []
    if len(spreadsheets) != len(matching_columns) != len(info_columns):
        raise ValueError('There must be the same number of spreadsheets, matching_columns and info_columns!')
    if Path(output_path).is_file():
        output_dict = open_json(output_path)
    else:
        output_dict = {}
    try:
        save_json(output_path, output_dict)
    except Exception as e:
        print(f'Exception caught when trying to save {output_path}')
        raise e

    if randomise_lbl_seg:
        first_rating_keyname = 'first_rating'
        second_rating_keyname = 'second_rating'
    else:
        first_rating_keyname = 'label_rating'
        second_rating_keyname = 'seg_rating'
    # Filtering the values
    # Here we want to avoid having to check images already labelled
    to_check_keys = [k for k in seg_dict if k not in output_dict or first_rating_keyname not in seg_dict[k] or
                     second_rating_keyname not in seg_dict[k]]

    # Try catch clause to avoid losing anything when a ctr+c is hit
    try:
        for counter, k in enumerate(to_check_keys):
            """
            Check if the paths in seg_dict exist
            """
            pid = seg_dict[k]['PatientID']
            if not Path(seg_dict[k]['b1000']).exists():
                b1000 = Path(images_root, seg_dict[k]['b1000'])
            else:
                b1000 = seg_dict[k]['b1000']
            label = None
            if 'label' in seg_dict[k]:
                if not Path(seg_dict[k]['label']).exists():
                    label = Path(images_root, seg_dict[k]['label'])
                else:
                    label = seg_dict[k]['label']
            seg = None
            if 'segmentation' in seg_dict[k]:
                if not Path(seg_dict[k]['segmentation']).exists():
                    seg = Path(images_root, seg_dict[k]['segmentation'])
                else:
                    seg = seg_dict[k]['segmentation']

            # TODO if randomise_lbl_seg we need the coordinates of each lesion centre
            output_dict[k] = {}
            if randomise_lbl_seg:
                if label is not None and seg is not None:
                    if random.random() > .5:
                        label, seg = seg, label
                output_dict[k] = {'first': label, 'second': seg}
            else:
                output_dict[k] = {'label': label, 'segmentation': seg}
            show_image = True
            show_report = True
            if display == 'fsleyes' and not no_coord:
                coord = get_centre_of_mass(label, round_coord=True).astype(int)
            else:
                coord = None
            print(f'############### IMAGE NUMBER {counter}/{len(to_check_keys)} #################')
            while show_image or show_report:
                if show_report:
                    for ind, spreadsheet in enumerate(spreadsheets):
                        matched_entries = spreadsheet[spreadsheet[matching_columns[ind]] == pid][info_columns[ind]]
                        print(f'Spreadsheet number {ind} ###############')
                        for entry_ind, entry in enumerate(matched_entries):
                            print(f'###### {entry_ind}: {entry}')
                if show_image:
                    display_img(b1000, label, coord=coord, display=display)
                pprint(label_dict)
                print('Please, select a label from the list above using either the number or the label itself or ')
                print('quit [exit]: to quit and save')
                print('image [display]: to display the image again and ask for an answer again')
                print('report: to show the report(s) information again and ask for an answer again')
                resp = input('Answer: ')
                show_image = False
                show_report = False
                if resp.lower() == 'report':
                    show_report = True
                elif resp.lower() in ['display', 'image']:
                    show_image = True
                elif resp.lower() in ['quit', 'exit']:
                    save_json(output_path, output_dict)
                    return output_dict
                else:
                    if resp in label_dict.values():
                        value = resp
                        output_dict[k][first_rating_keyname] = value
                        save_json(output_path, output_dict)
                    elif resp in label_dict.keys():
                        value = label_dict[resp]
                        output_dict[k][first_rating_keyname] = value
                        save_json(output_path, output_dict)
                    else:
                        yn = input(f'{resp} is neither a label nor a label code.'
                                   f'\nDo you want to add this as a new label [Y/n]')
                        if yn.lower() == 'n':
                            print('Alright! Showing the image again!')
                            show_image = True
                        else:
                            value = resp
                            output_dict[k][first_rating_keyname] = value
                            save_json(output_path, output_dict)
                            label_dict.update({str(len(label_dict)): resp})
                            save_json(label_dict_path, label_dict)
            show_image = True
            show_report = True
            if display == 'fsleyes' and not no_coord:
                coord = get_centre_of_mass(seg, round_coord=True).astype(int)
            else:
                coord = None
            print(f'############### IMAGE NUMBER {counter}/{len(to_check_keys)} #################')
            while show_image or show_report:
                if show_report:
                    for ind, spreadsheet in enumerate(spreadsheets):
                        matched_entries = spreadsheet[spreadsheet[matching_columns[ind]] == pid][info_columns[ind]]
                        print(f'Spreadsheet number {ind} ###############')
                        for entry_ind, entry in enumerate(matched_entries):
                            print(f'###### {entry_ind}: {entry}')
                if show_image:
                    display_img(b1000, seg, coord=coord, display=display)
                pprint(label_dict)
                print('Please, select a label from the list above using either the number or the label itself or ')
                print('quit [exit]: to quit and save')
                print('image [display]: to display the image again and ask for an answer again')
                print('report: to show the report(s) information again and ask for an answer again')
                resp = input('Answer: ')
                show_image = False
                show_report = False
                if resp.lower() == 'report':
                    show_report = True
                elif resp.lower() in ['display', 'image']:
                    show_image = True
                elif resp.lower() in ['quit', 'exit']:
                    save_json(output_path, output_dict)
                    return output_dict
                else:
                    if resp in label_dict.values():
                        value = resp
                        output_dict[k][second_rating_keyname] = value
                        save_json(output_path, output_dict)
                    elif resp in label_dict.keys():
                        value = label_dict[resp]
                        output_dict[k][second_rating_keyname] = value
                        save_json(output_path, output_dict)
                    else:
                        yn = input(f'{resp} is neither a label nor a label code.'
                                   f'\nDo you want to add this as a new label [Y/n]')
                        if yn.lower() == 'n':
                            print('Alright! Showing the image again!')
                            show_image = True
                        else:
                            value = resp
                            output_dict[k][second_rating_keyname] = value
                            save_json(output_path, output_dict)
                            label_dict.update({str(len(label_dict)): resp})
                            save_json(label_dict_path, label_dict)
        save_json(output_path, output_dict)
    except KeyboardInterrupt as e:
        print('Script interrupted using keyboard interruption. Saving the output dictionary.')
        save_json(output_path, output_dict)
        raise e
    return output_dict


def load_tensorboard_file(event_file, verbose=False):
    # Load the TensorBoard file
    event_acc = EventAccumulator(event_file)
    event_acc.Reload()

    # If verbose is True, print all the tags from the file
    if verbose:
        print("Tags:")
        for tag in event_acc.Tags()['scalars']:
            print(tag)

    return event_acc


def create_and_save_plots(event_acc, best_func_dict=None, output_folder=None, display_plot=True):
    # Get the list of scalar tags
    scalar_tags = event_acc.Tags()['scalars']

    # If best_func_dict is not provided, use a default function (min) for all tags
    if best_func_dict is None:
        best_func_dict = {tag: min for tag in scalar_tags}

    # Create and save a plot for each tag
    for tag in scalar_tags:
        fig, ax = plt.subplots(figsize=(10, 5))
        x, y = zip(*[(s.step, s.value) for s in event_acc.Scalars(tag)])
        ax.plot(x, y)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(tag)
        # Find the best value
        best_value_func = best_func_dict.get(tag, min)  # Use min if the tag is not in best_func_dict
        best_value = best_value_func(y)
        best_epoch = x[y.index(best_value)]
        ax.plot(best_epoch, best_value, 'ro')
        ax.set_title(f'Best value: {best_value:.4f}')
        plt.tight_layout()
        # Save the plot if output_folder is provided
        if output_folder is not None:
            plt.savefig(output_folder / f'{tag}.png')
        # Display the plot if display_plot is True
        if display_plot:
            plt.show()
        plt.close(fig)  # Close the figure


def load_all_event_files(model_folder):
    """
    Load the latest event file from each subfolder in the given model folder.

    This function iterates over each subfolder in the model folder, finds the latest event file in each subfolder,
    and loads the data from these event files using the `load_tensorboard_file` function. The data from each event file
    is stored in a dictionary where the keys are the fold names (subfolder names) and the values are the EventAccumulator
    objects returned by `load_tensorboard_file`. This dictionary is returned by the function.

    Parameters:
    model_folder (str): The path to the model folder containing the subfolders with event files.

    Returns:
    dict: A dictionary where the keys are the fold names and the values are the EventAccumulator objects containing the
    data from the latest event file in each subfolder.
    """
    model_folder = Path(model_folder)
    fold_data = {}

    for fold_folder in model_folder.iterdir():
        if fold_folder.is_dir():
            event_files = list(fold_folder.rglob('events.out.tfevents.*'))
            if event_files:
                latest_event_file = max(event_files, key=os.path.getmtime)
                event_acc = load_tensorboard_file(str(latest_event_file))
                fold_data[fold_folder.name] = event_acc

    return fold_data


def plot_scalar_per_fold(fold_data, scalar_name, best_epochs=None, output_path=None,
                         display_plot=True, best_func=None, display_names_dict=None, output_prefix=None):
    """
    Plot the values of a scalar per fold and optionally save the plot to a file.

    This function iterates over the items in fold_data. For each item, the key is the fold name and the value is the
    EventAccumulator object. It checks if the EventAccumulator object contains the scalar. If it does, it gets the
    values of the scalar and plots these values. The fold name is used as the label for the plot.

    If output_path is provided, the function will save the plot to the specified path. If output_path is a directory,
    the function will create the directory if it doesn't exist and save the plot as scalar_name_folds_plot.png in that
    directory.

    If best_func is provided, the function will compute the best value for each scalar, signal it with a red dot on the
    plot, and print the best value and the epoch number of the best value next to the fold name in the legend. The best
    values for each fold are stored in a dictionary.

    If best_epochs is provided, the function will plot a red dot at the epoch number provided for each fold. The scalar
    values at the best epochs for each fold are stored in a dictionary.

    The function returns a dictionary containing the best values for each fold if best_func is used or the scalar values
    at the best epochs for each fold if best_epochs is used. If neither best_func nor best_epochs is used, the function
    returns None.

    Parameters:
    fold_data (dict): A dictionary where the keys are the fold names and the values are the EventAccumulator objects.
    scalar_name (str): The name of a scalar.
    best_epochs (dict, optional): A dictionary where the keys are the fold names and the values are the epoch numbers
        where the best values occur. Defaults to None.
    output_path (str, optional): The path where the plot will be saved. If this is a directory, the plot will be saved
        as scalar_name_folds_plot.png in this directory. Defaults to None.
    display_plot (bool, optional): Whether to display the plot. Defaults to True.
    best_func (function or dict, optional): A function to compute the best value for each scalar, or a dictionary
    mapping scalar names to such functions. Defaults to None.
    display_names_dict (dict, optional): A dictionary mapping scalar names to names to be displayed on the plot.
    output_prefix (str, optional): A reference to be added to the output path. Defaults to None.

    Returns:
    dict: A dictionary containing the best values for each fold if best_func is used or the scalar values at the
    best epochs for each fold if best_epochs is used. If neither best_func nor best_epochs is used, the function returns
    {}.

    Raises:
    ValueError: If an EventAccumulator object does not contain the scalar.
    """
    if display_names_dict is None:
        display_names_dict = {}
    sorted_fold_names = sorted(fold_data.keys(), key=lambda x: int(x.split('_')[1]))
    best_epochs_dict = {}
    plot_params_dict = {}
    marker_params_dict = {}
    cmap = sns.color_palette("muted")
    for fold_index, fold_name in enumerate(sorted_fold_names):
        event_acc = fold_data[fold_name]
        if scalar_name not in event_acc.Tags()['scalars']:
            raise ValueError(f"The scalar '{scalar_name}' is not found in the fold '{fold_name}'.")

        x, y = zip(*[(s.step, s.value) for s in event_acc.Scalars(scalar_name)])

        fold_label = fold_name
        markers_created = False
        marker_colour = cmap[fold_index % len(cmap)]
        marker_type = 'o'
        marker_size = 5
        plot_marker_param = None
        if best_epochs is not None and fold_name in best_epochs:
            best_epoch_provided = best_epochs[fold_name]
            if best_epoch_provided in x:
                best_value_provided = y[x.index(best_epoch_provided)]
                # store marker parameters for later use
                plot_marker_param = (best_epoch_provided, best_value_provided, marker_colour, marker_type,
                                     marker_size)
                fold_label += f', best model at {best_epoch_provided}: {best_value_provided:.4f}'
                markers_created = True
                best_epochs_dict[fold_name] = best_epoch_provided
            else:
                print(f"Warning: Best epoch {best_epoch_provided} not found in epochs for fold {fold_name}. Skipping.")

        if best_func is not None and not markers_created:
            best_func_scalar = best_func[scalar_name] if isinstance(best_func, dict) else best_func
            best_value = best_func_scalar(y)
            best_epoch = x[y.index(best_value)]
            # store marker parameters for later use
            plot_marker_param = (best_epoch, best_value, marker_colour, marker_type, marker_size)
            fold_label += f', best at {best_epoch}: {best_value:.4f}'
            best_epochs_dict[fold_name] = best_epoch

        plot_params_dict[fold_name] = (x, y, fold_label, marker_colour)
        if plot_marker_param is not None:
            marker_params_dict[fold_name] = plot_marker_param

    for fold_name in sorted_fold_names:
        x, y, fold_label, colour = plot_params_dict[fold_name]
        plt.plot(x, y, label=fold_label, color=colour)

    for fold_name in sorted_fold_names:
        if fold_name in marker_params_dict:
            plt.plot(*marker_params_dict[fold_name][:2], color=marker_params_dict[fold_name][2],
                     marker=marker_params_dict[fold_name][3],
                     markersize=marker_params_dict[fold_name][4], markeredgewidth=1, markeredgecolor='black')
    plt.xlabel('Epoch')
    # Use the display name for the y-axis label if it is provided, otherwise use the scalar name
    ylabel = display_names_dict[
        scalar_name] if display_names_dict and scalar_name in display_names_dict else scalar_name
    plt.ylabel(ylabel)
    plt.legend()
    if output_prefix is not None:
        output_filename = f'{output_prefix}_{scalar_name}_folds_plot.png'
    else:
        output_filename = f'{scalar_name}_folds_plot.png'
    if output_path is not None:
        output_path = Path(output_path)
        if output_path.is_dir():
            output_path.mkdir(parents=True, exist_ok=True)
            output_path = output_path / output_filename
        plt.savefig(output_path)

    if display_plot:
        plt.show()

    plt.close()

    return best_epochs_dict


def plot_scalar_per_fold_return_subplot(fold_data, scalar_name, best_epochs=None, best_func=None, display_names_dict=None):
    """
    Plot the values of a scalar per fold and return the subplot.

    This function iterates over the items in fold_data. For each item, the key is the fold name and the value is the
    EventAccumulator object. It checks if the EventAccumulator object contains the scalar. If it does, it gets the
    values of the scalar and plots these values. The fold name is used as the label for the plot.

    If best_func is provided, the function will compute the best value for each scalar, signal it with a red dot on the
    plot, and print the best value and the epoch number of the best value next to the fold name in the legend. The best
    values for each fold are stored in a dictionary.

    If best_epochs is provided, the function will plot a red dot at the epoch number provided for each fold. The scalar
    values at the best epochs for each fold are stored in a dictionary.

    The function returns a subplot and a dictionary containing the best values for each fold if best_func is used or the scalar values
    at the best epochs for each fold if best_epochs is used. If neither best_func nor best_epochs is used, the function
    returns None.

    Parameters:
    fold_data (dict): A dictionary where the keys are the fold names and the values are the EventAccumulator objects.
    scalar_name (str): The name of a scalar.
    best_epochs (dict, optional): A dictionary where the keys are the fold names and the values are the epoch numbers
        where the best values occur. Defaults to None.
    best_func (function or dict, optional): A function to compute the best value for each scalar, or a dictionary
    mapping scalar names to such functions. Defaults to None.
    display_names_dict (dict, optional): A dictionary mapping scalar names to names to be displayed on the plot.

    Returns:
    fig, ax, dict: A matplotlib figure and axes objects, and a dictionary containing the best values for each fold if best_func is used or the scalar values at the
    best epochs for each fold if best_epochs is used. If neither best_func nor best_epochs is used, the function returns
    {}.

    Raises:
    ValueError: If an EventAccumulator object does not contain the scalar.
    """
    if display_names_dict is None:
        display_names_dict = {}
    sorted_fold_names = sorted(fold_data.keys(), key=lambda x: int(x.split('_')[1]))
    best_epochs_dict = {}
    plot_params_dict = {}
    marker_params_dict = {}
    cmap = sns.color_palette("muted")
    fig, ax = plt.subplots(figsize=(10, 5))
    best_values = []  # List to store the best values for each fold
    for fold_index, fold_name in enumerate(sorted_fold_names):
        event_acc = fold_data[fold_name]
        if scalar_name not in event_acc.Tags()['scalars']:
            raise ValueError(f"The scalar '{scalar_name}' is not found in the fold '{fold_name}'.")
        x, y = zip(*[(s.step, s.value) for s in event_acc.Scalars(scalar_name)])
        fold_label = fold_name
        markers_created = False
        marker_colour = cmap[fold_index % len(cmap)]
        marker_type = 'o'
        marker_size = 5
        plot_marker_param = None
        if best_epochs is not None and fold_name in best_epochs:
            best_epoch_provided = best_epochs[fold_name]
            if best_epoch_provided in x:
                best_value_provided = y[x.index(best_epoch_provided)]
                plot_marker_param = (best_epoch_provided, best_value_provided, marker_colour, marker_type, marker_size)
                fold_label += f', best model at {best_epoch_provided}: {best_value_provided:.4f}'
                markers_created = True
                best_epochs_dict[fold_name] = best_epoch_provided
                best_values.append(best_value_provided)  # Add the best value to the list
            else:
                print(f"Warning: Best epoch {best_epoch_provided} not found in epochs for fold {fold_name}. Skipping.")
        if best_func is not None and not markers_created:
            best_func_scalar = best_func[scalar_name] if isinstance(best_func, dict) else best_func
            best_value = best_func_scalar(y)
            best_epoch = x[y.index(best_value)]
            plot_marker_param = (best_epoch, best_value, marker_colour, marker_type, marker_size)
            fold_label += f', best at {best_epoch}: {best_value:.4f}'
            best_epochs_dict[fold_name] = best_epoch
            best_values.append(best_value)  # Add the best value to the list
        plot_params_dict[fold_name] = (x, y, fold_label, marker_colour)
        if plot_marker_param is not None:
            marker_params_dict[fold_name] = plot_marker_param
    for fold_name in sorted_fold_names:
        x, y, fold_label, colour = plot_params_dict[fold_name]
        plt.plot(x, y, label=fold_label, color=colour)
    for fold_name in sorted_fold_names:
        if fold_name in marker_params_dict:
            plt.plot(*marker_params_dict[fold_name][:2], color=marker_params_dict[fold_name][2],
                     marker=marker_params_dict[fold_name][3],
                     markersize=marker_params_dict[fold_name][4], markeredgewidth=1, markeredgecolor='black')
    ax.set_xlabel('Epoch')
    ylabel = display_names_dict[
        scalar_name] if display_names_dict and scalar_name in display_names_dict else scalar_name
    ax.set_ylabel(ylabel)
    ax.legend()
    # Calculate mean and standard deviation of best scalar values for all folds
    mean_value = np.mean(best_values)
    std_value = np.std(best_values)
    # Set the title of the subplot to include the mean and standard deviation of the best values
    ax.set_title(f'(mean: {mean_value:.4f}, std: {std_value:.4f})')
    return fig, ax, best_epochs_dict
