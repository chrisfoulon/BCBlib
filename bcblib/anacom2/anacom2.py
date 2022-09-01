import argparse
import os
from pathlib import Path

from bcblib.tools.spreadsheet_io_utils import import_spreadsheet, str_to_column_id
from bcblib.tools.nifti_utils import is_nifti, file_to_list


def anacom2():
    desc = "Helper function to create a design matrix and 4D image to input to fsl randomise"
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
        #     TODO verify all the patients are in the folder
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
