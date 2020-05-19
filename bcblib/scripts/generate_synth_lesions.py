import os
import sys
import argparse

import nibabel as nib
import nilearn


def create_coverage_mask(image_path_list):
    coverage_mask = None
    for f in image_path_list:
        if not os.path.isfile(f):
            raise ValueError('{} is not an existing file'.format(f))
        if coverage_mask = None:
            coverage_mask = nib.load(f)


def main():
    parser = argparse.ArgumentParser(description='Generate matched synthetic lesions dataset')
    paths_group = parser.add_mutually_exclusive_group(required=True)
    paths_group.add_argument('-p', '--input_path', type=str, help='Root folder of the lesion dataset')
    paths_group.add_argument('-li-', '--input_list', type=str, help='Text file containing the list of lesion files')
    parser.add_argument('-o', '--output', default='./', type=str, help='output folder')

    # parser.add_argument('-v', '--verbose', default='info', choices=['none', 'info', 'debug'], nargs='?', const='info',
    #                     type=str, help='print info or debugging messages [default is "info"] ')
    args = parser.parse_args()
