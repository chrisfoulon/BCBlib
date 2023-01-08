from pathlib import Path
import os
import shutil
from typing import Union, List

import pandas as pd
import numpy as np
import nibabel as nib
from nilearn.image import concat_imgs
from bcblib.tools.spreadsheet_io_utils import import_spreadsheet
import argparse

from bcblib.tools.nifti_utils import is_nifti, file_to_list


def spreadsheet_to_mat_and_file_list(spreadsheet: Union[str, pd.DataFrame],
                                     columns: Union[str, int, List[str], List[int]],
                                     output_dir: Union[str, Path], pref: str = '', header: int = 0,
                                     filenames_column: Union[str, int] = 0):
    """Convert a spreadsheet to a .mat file and a .csv file with the list of filenames.

    The .mat file contains the selected columns from the input spreadsheet, and the .csv file contains the
    filenames for each row in the .mat file. The .mat file is formatted for use as an input for the FSL randomise
    tool.

    Parameters
    ----------
    spreadsheet: Union[str, pandas.DataFrame]
        DataFrame or path to the spreadsheet containing the data to input in the .mat file.
    columns: Union[str, int, List[str], List[int]]
        Columns names or indices (if no header) to include in the .mat file.
    output_dir: str
        Output directory (created if not existing) for the .mat and .csv files.
    pref: str, optional
        Prefix to add before the filenames of the output files (default is '').
    header: int, optional
        Header option for pandas.read_csv or pandas.read_excel (default is 0).
    filenames_column: Union[str, int], optional
        Column name or index where to find the filenames for each row in the input spreadsheet (default is 0).

    Returns
    -------
    filtered_filenames: List[str]
        List of filenames for each row in the .mat file.

    Examples
    --------
    # Import data from an Excel spreadsheet
    df = pd.read_excel('/home/user/Downloads/spreadsheet.xlsx', header=0)
    # Convert the data to a .mat file and a .csv file
    spreadsheet_to_mat_and_file_list(df, ['SubID', 'IDP'], '/home/user/Downloads/')
    """

    df = import_spreadsheet(spreadsheet, header)
    if isinstance(columns, str):
        if columns not in df.columns:
            try:
                columns = int(columns)
            except ValueError:
                raise ValueError(f'{columns} must either be column names or integers')
        columns = [columns]
    for i, c in enumerate(columns):
        if c not in df.columns:
            try:
                columns[i] = int(c)
            except ValueError:
                raise ValueError(f'{c} must either be column names or integers')
    # Drop the rows missing a value
    if isinstance(columns[0], int):
        columns = [df.columns[i] for i in columns]
    filtered_df = df.iloc[df[columns][pd.notnull(df[columns])].dropna().index][columns]
    np_mat = filtered_df.values
    os.makedirs(output_dir, exist_ok=True)
    output_path = Path(output_dir, pref + 'design.mat')
    np.savetxt(output_path, np_mat, delimiter=' ', fmt='%s')

    if filenames_column not in df.columns:
        try:
            filenames_column = int(filenames_column)
        except ValueError:
            raise ValueError(f'{filenames_column} must either be column names or integers')
        filenames_column = df.columns[filenames_column]
    filtered_filenames = df.iloc[filtered_df.index][filenames_column].tolist()
    filtered_filenames = [str(f) for f in filtered_filenames]
    prepend_str = f'/NumWaves {len(columns)}'
    prepend_str += f'\n/NumPoints {len(filtered_filenames)}'
    prepend_str += f'\n/PPheights 1 1'
    prepend_str += f'\n/Matrix\n'
    with open(output_path, 'r') as original:
        data = original.read()
    with open(output_path, 'w') as modified:
        modified.write(prepend_str + data)
    np.savetxt(Path(output_dir, pref + '4D_file_list.csv'), filtered_filenames, delimiter=',', fmt='%s')
    return filtered_filenames


def filtered_images_to_4d(images, filtered_filenames, output_dir, pref=''):
    if Path(images).is_file():
        images = file_to_list(images)
    if isinstance(images, list):
        for img in images:
            if not is_nifti(img):
                raise ValueError(f'{img} is not an existing nifti file')
    if Path(images).is_dir():
        images = [str(p) for p in Path(images).iterdir() if is_nifti(p)]
    fname_list = [Path(f).name for f in images]
    filtered_file_list = []
    for f in filtered_filenames:
        # the order in the folder and in the spreadsheet might not be the same
        found = False
        only_f_name = Path(f).name.split('.nii')[0]
        for ind, fname in enumerate(fname_list):
            if only_f_name == fname.split('.nii')[0]:
                filtered_file_list.append(images[ind])
                found = True
        if not found:
            raise ValueError(f'{f} was not found in the provided list of files')
    nii_list = [nib.load(f) for f in filtered_file_list]
    voxel_dtype = nii_list[0].get_data_dtype()
    new_4d_images = concat_imgs(nii_list, dtype=voxel_dtype)
    output_4d = Path(output_dir, pref + 'filtered_4D.nii.gz')
    nib.save(new_4d_images, output_4d)


def randomise_helper():
    desc = "Helper function to create a design matrix and 4D image to input to fsl randomise"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('output_dir', type=str, help='Path to the output directory (created if not existing)')
    parser.add_argument('spreadsheet', type=str, help='Path to the spreadsheet')
    parser.add_argument('images', type=str, help='Folder containing the masks or file containing the list of the paths')
    parser.add_argument('-sav', '--split_all_var', action="store_true",
                        help='if selected, split all the columns from the spreadsheet (that are not in the '
                             'common columns or the filename column) into different experiment folders with the '
                             'name of the column used as variable of interest')
    parser.add_argument('-cc', '--common_columns', metavar='N', type=str, nargs='+',
                        help='Names of the columns to be used as co-variables')
    parser.add_argument('-pref', type=str, default='', help='Prefix to the filename of the output files')
    parser.add_argument('-fc', '--filename_col', default=0, help='Columns name/index where to find the filenames')
    parser.add_argument('-no_hdr', action="store_true",
                        help='If the spreadsheet does not a row corresponding to the column names')
    parser.add_argument('-con', '--copy_contrast', help='If provided, the file will be copied in every '
                                                        'output subfolders')
    args = parser.parse_args()
    if not args.pref.endswith('_'):
        args.pref = args.pref + '_'
    if args.common_columns is None:
        args.common_columns = []
    if args.split_all_var:
        df = import_spreadsheet(args.spreadsheet, header=0)
        variables_of_interest = [c for c in df.columns if c != args.filename_col and c not in args.common_columns]
        # voi for variable of interest
        for voi in variables_of_interest:
            image_list = spreadsheet_to_mat_and_file_list(args.spreadsheet, [voi] + args.common_columns,
                                                          Path(args.output_dir, voi), pref=args.pref, header=0,
                                                          filenames_column=args.filename_col)
            if len(image_list) == 0:
                raise ValueError(f'No patient could be associated with values in {voi}')
            filtered_images_to_4d(args.images, image_list, Path(args.output_dir, voi), pref=args.pref)
            if args.copy_contrast is not None:
                shutil.copy(args.copy_contrast, Path(args.output_dir, voi, args.pref + Path(args.copy_contrast).name))
    else:
        image_list = spreadsheet_to_mat_and_file_list(args.spreadsheet, args.common_columns, args.output_dir,
                                                      pref=args.pref, header=0, filenames_column=args.filename_col)
        filtered_images_to_4d(args.images, image_list, args.output_dir, pref=args.pref)
