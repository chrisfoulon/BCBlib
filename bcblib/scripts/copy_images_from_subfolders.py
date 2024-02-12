import os
import shutil
import argparse
from pathlib import Path


def copy_nifti_files(imaging_root_folder: str, subfolder_name: str, output_folder: str):
    # create the output folder
    os.makedirs(output_folder, exist_ok=True)

    root_path = Path(imaging_root_folder)
    subfolders = list(root_path.glob(f'**/{subfolder_name}'))

    found_files_counter = 0
    for subfolder in subfolders:
        if '__MACOSX' in subfolder.parts:
            continue  # don't process MACOSX directories

        nifti_files = [f for f in subfolder.glob('*') if (f.name.endswith('.nii.gz') or f.name.endswith('.nii')) and
                       not f.name.startswith('._')]
        if len(nifti_files) > 1:
            raise ValueError(f'More than one nifti file found in {subfolder}')

        for nifti_file in nifti_files:
            shutil.copy(nifti_file, output_folder)
            found_files_counter += 1

    print(f'Found {found_files_counter} files in {imaging_root_folder} with subfolder {subfolder_name}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Copy nifti files from a specific subfolder.')
    parser.add_argument('imaging_root_folder', type=str, help='Path to the imaging root folder')
    parser.add_argument('subfolder_name', type=str, help='Name of the subfolder to copy files from')
    parser.add_argument('output_folder', type=str, help='Path to the output folder')

    args = parser.parse_args()

    copy_nifti_files(args.imaging_root_folder, args.subfolder_name, args.output_folder)