import argparse
import os
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
from bcblib.tools.spreadsheet_io_utils import import_spreadsheet, str_to_column_id
from bcblib.tools.nifti_utils import is_nifti, file_to_list, load_nifti


def anacom2():
    """
    Main function of the anacom2 script

    """
    desc = "AnaCOM2: cluster based lesion symptom analysis"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('output_dir', type=str, required=True,
                        help='Path to the output directory (created if not existing)')
    parser.add_argument('patient_scores', type=str, required=True,
                        help='Path to the spreadsheet containing the patient scroes')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-cs', '--control_scores', type=str,
                       help='Path to the spreadsheet containing the control scroes')
    group.add_argument('-cm', '--control_mean', type=float,
                       help='The mean of control scores')
    parser.add_argument('images', type=str, required=True,
                        help='Folder containing the masks or file containing the list of the paths')
    parser.add_argument('-ph', '--patient_header', action='store_true', help='Does the spreadsheet contain a header')
    parser.add_argument('-sc', '--score_col', default=1, help='Columns name/index where to find the scores')
    parser.add_argument('-fc', '--filename_col', default=0, help='Columns name/index where to find the filenames')
    parser.add_argument('-ch', '--control_header', action='store_true',
                        help='Does the control spreadsheet contain a header')
    parser.add_argument('-csc', '--control_score_col', default=None,
                        help='Columns name/index where to find the scores of the controls')
    args = parser.parse_args()

    output_dir = args.output_dir
    images = args.images
    if Path(images).is_dir():
        images = [str(p) for p in Path(images).iterdir() if is_nifti(p)]
    elif Path(images).is_file():
        images = file_to_list(images)
    else:
        raise ValueError(f'{images} should either be the path to a directory containing niftis or a file containing the'
                         f' list of image paths')
    os.makedirs(output_dir, exist_ok=True)
    spreadsheet = args.spreadsheet
    if args.patient_header:
        header = 0
    else:
        header = None
    patient_df = import_spreadsheet(spreadsheet, header)

    if isinstance(args.score_col, str):
        score_col = str_to_column_id(args.score_col, patient_df)
    else:
        score_col = patient_df.columns[args.score_col]
    score_series = patient_df[score_col].dropna()
    print(f'{len(score_series)} usable patient scores found in column {score_col}')

    if isinstance(args.filename_col, str):
        filename_col = str_to_column_id(args.filename_col, patient_df)
    else:
        filename_col = patient_df.columns[args.filename_col]
    filename_series = patient_df[filename_col].iloc[score_series].dropna()
    print(f'{len(filename_series)} usable patient filenames corresponding to the scores found in column {filename_col}')
    filename_series_fnames = [Path(p).name for p in filename_series]
    for fname, filename in zip(filename_series_fnames, filename_series):
        found = False
        for path in images:
            if fname == Path(path).name:
                found = True
                if not Path(path).is_file():
                    raise ValueError(f'{path} is not an existing file')
        if not found:
            raise ValueError(f'{filename} could not be matched with any file in the images list')

    # get all the file paths associated with the filenames in filename_series
    # check that all the files exist
    # check that all the files have the same resolution and orientation
    for path in images:
        if not Path(path).is_file():
            raise ValueError(f'{path} is not an existing file')
    first_affine = None
    # The affine can be obtained with nibabel.load(path).affine
    for path in images:
        if first_affine is None:
            first_affine = load_nifti(path).affine
        else:
            if not np.allclose(first_affine, load_nifti(path).affine):
                raise ValueError(f'{path} does not have the same affine as the first file')

    if args.control_scores is not None:
        if args.control_header:
            control_header = 0
        else:
            control_header = None
        control_df = import_spreadsheet(args.control_scores, control_header)
        if args.control_score_col is None:
            if score_col in control_df.columns:
                control_score_col = score_col
            else:
                control_score_col = 0
        elif isinstance(args.control_score_col, str):
            control_score_col = str_to_column_id(args.control_score_col, control_df)
        else:
            control_score_col = control_df.columns[args.score_col]
        control_scores = control_df[control_score_col].dropna()
        print(f'{len(control_scores)} usable patient scores found in column {control_score_col}')
    else:
        control_scores = args.control_mean
        print(f'Mean score for the controls: {control_scores}')
    """
    Summarise what is done here and explain the content of each variable (write it in comments)
    
    1. Create the output directory if it does not exist
    2. Check that the images are either a folder containing niftis or a file containing the list of image paths
    3. Import the patient scores
    4. Check that the patient scores are usable
    5. Check that the patient filenames are usable
    6. Import the control scores
    7. Check that the control scores are usable
    8. Summarise what is done    
    """
    # Make a patient img : patient score dictionary
    patient_dict = dict(zip(filename_series_fnames, score_series))
    print(f'{len(patient_dict)} patient scores found in the spreadsheet')
    # print first key and value of the dictionary
    print(f'First key: {list(patient_dict.keys())[0]}')
    coord_dict = defaultdict(list)

    for ind, path in enumerate(patient_dict):
        img_path = Path(path)
        hdr = load_nifti(img_path)
        data = hdr.get_fdata()
        coord_data = np.argwhere(data)
        for coord in coord_data:
            coord_dict[coord].append(patient_dict[path])
        # Now we create the clusters
        score_clusters_coord_dict = defaultdict(list)
        #TODO Make the clusters uniquer (they might not be with just the scrores)
        for coord, scores in coord_dict.items():
            score_cluster = []
