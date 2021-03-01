import os
from pathlib import Path
import csv
import numpy as np

import nibabel as nib


def is_nifti(filename):
    return str(filename)[-4:] == '.nii' or str(filename)[-7:] == '.nii.gz'


def file_to_list(file_path, delimiter=' '):
    if not os.path.exists(file_path):
        raise ValueError(file_path + ' does not exist.')
    if file_path.endswith('.csv'):
        with open(file_path, 'r') as csv_file:
            dir_list = []
            for row in csv.reader(csv_file):
                if len(row) > 1:
                    dir_list += [r for r in row]
                else:
                    dir_list.append(row[0])
    else:
        # default delimiter is ' ', it might need to be changed
        dir_list = np.loadtxt(file_path, dtype=str, delimiter=delimiter)
    return dir_list


def load_nifti(img):
    """

    Parameters
    ----------
    img : nibabel.Nifti1Image or str or pathlike object
        already loaded image or path to a nifti image

    Returns
    -------
        nibabel loaded Nifti1Image
    """
    if isinstance(img, (str, bytes, os.PathLike)):
        if Path(img).is_file():
            img = nib.load(img, mmap=False)
    if not isinstance(img, nib.Nifti1Image):
        raise ValueError('img must either be a valid path or a nibabel.Nifti1Image')
    return img


def get_nifti_orientation(img):
    img = load_nifti(img)
    return nib.aff2axcodes(img.affine)


def reorient_to_canonical(img, save=False):
    """
    WARNING: if save == True, the input image is modified
    Try to reorient the input images to the canonical RAS+ orientation
    Parameters
    ----------
    img : nibabel.Nifti1Image or path-like object
    save : bool
        save the reoriented image in place of the input image
    Returns
    -------
    img : nibabel.Nifti1Image
        return the loaded reoriented image
    """
    img = load_nifti(img)
    if not img.get_filename():
        raise ValueError('ERROR: the nifti image is not an already existing file and thus will not be reoriented')
    else:
        nii = nib.as_closest_canonical(img)
        if save:
            nii.set_filename(img.get_filename())
            nib.save(nii, img.get_filename())
    return nii


def reorient_nifti_list(nifti_list, output_dir=None, save_in_place=False, discard_errors=False):
    """
    WARNING: the input images may be modified
    Try to reorient the input images to the canonical RAS+ orientation
    Parameters
    ----------
    nifti_list : list of nibabel.Nifti1Image or list of path-like object
    output_dir : path-like object
        optional is save_in_place is True. If given, the reoriented images will be saved in the output directory
    save_in_place : bool
        save the reoriented images in place of the input images
    discard_errors : bool
        If True : if an image cannot be reoriented because of an error due to corrupted data (preventing nibabel from
        reading properly) this option will remove it from the list
        WARNING: the output list might be shorter than the input list
        if False, in case of such error, the function will raise an error and fail.
    Returns
    -------
    reoriented_list : list of nibabel.Nifti1Image
    """
    if (output_dir is None or not Path(output_dir).is_dir()) and not save_in_place:
        raise Exception('output_dir does not exist or is missing. output_dir MUST be given IF save_in_place is False')
    reoriented_list = []
    failed_list = []
    for nii in nifti_list:
        try:
            nii = load_nifti(nii)
            img = reorient_to_canonical(nii)
            if output_dir is not None and Path(output_dir).is_dir():
                img.set_filename(Path(output_dir, Path(nii.get_filename()).name))
            if save_in_place:
                img.set_filename(nii.get_filename())
            nib.save(img, img.get_filename())
            reoriented_list.append(img.get_filename())
        except Exception as e:
            if isinstance(nii, nib.Nifti1Image):
                fname = nii.get_filename()
            else:
                fname = str(nii)
            failed_list.append(fname)
            if discard_errors:
                print('Error in file {}: {}\n the image will then be discarded from the list'
                      '\n WARNING: the output list will be shorter than the input list'.format(fname, e))
            else:
                raise TypeError('Error in file {}: {}'.format(fname, e))
    return reoriented_list, failed_list


# for img in orientation_list:
#     ...:     if nifti_utils.get_nifti_orientation(img) != ('R', 'A', 'S'):
#     ...:         print(img)
#     ...:         print(nib.load(img).affine)
#     ...:         print(nib.as_closest_canonical(nib.load(img)).affine)
#     ...:         print(nib.load(img).affine == nib.as_closest_canonical(nib.load(img)).affine)
#     ...:         print(orientation_list[img])
#     ...:         print('######################### END IMAGE ##############################')


def resave_nifti(nifti, output=None):
    output_file = None
    output_dir = None
    if output is not None:
        if Path(output).is_dir():
            output_dir = output
        else:
            if Path(output).parent.is_dir():
                output_dir = Path(output).parent
                output_file = output
            else:
                raise ValueError('{} is not an existing directory'.format(Path(output).parent))

    img = load_nifti(nifti)
    if not output_dir:
        output_file = img.get_filename()
    if output_dir and not output_file:
        output_file = Path(output_dir, Path(img.get_filename()).name)
    if not output_dir and not output_file:
        raise ValueError('The given image does not have a defined filename and no output has been given')
    nib.save(nib.Nifti1Image(img.get_fdata(), img.affine), output_file)
    return output_file


def resave_nifti_list(nifti_list, output_dir=None, save_in_place=False, discard_errors=False):
    """
    WARNING: the input images may be modified
    Try to reorient the input images to the canonical RAS+ orientation
    Parameters
    ----------
    nifti_list : list of nibabel.Nifti1Image or list of path-like object
    output_dir : path-like object
        optional is save_in_place is True. If given, the reoriented images will be saved in the output directory
    save_in_place : bool
        save the reoriented images in place of the input images
    discard_errors : bool
        If True : if an image cannot be reoriented because of an error due to corrupted data (preventing nibabel from
        reading properly) this option will remove it from the list
        WARNING: the output list might be shorter than the input list
        if False, in case of such error, the function will raise an error and fail.
    Returns
    -------
    reoriented_list : list of nibabel.Nifti1Image
    """
    if (output_dir is None or not Path(output_dir).is_dir()) and not save_in_place:
        raise Exception('output_dir does not exist or is missing. output_dir MUST be given IF save_in_place is False')
    resaved_list = []
    failed_list = []
    for nii in nifti_list:
        try:
            fname = resave_nifti(nii, output=output_dir)
            resaved_list.append(fname)
        except Exception as e:
            if isinstance(nii, nib.Nifti1Image):
                fname = nii.get_filename()
            else:
                fname = str(nii)
            failed_list.append(fname)
            if discard_errors:
                print('Error in file {}: {}\n the image will then be discarded from the list'
                      '\n WARNING: the output list will be shorter than the input list'.format(fname, e))
            else:
                raise TypeError('Error in file {}: {}'.format(fname, e))
    return resaved_list, failed_list
