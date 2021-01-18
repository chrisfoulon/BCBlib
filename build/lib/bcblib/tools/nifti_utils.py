import os
from pathlib import Path

import nibabel as nib


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


def reorient_nifti_list(nifti_list, save=False, discard_errors=False):
    """
    WARNING: the input images are modified
    Try to reorient the input images to the canonical RAS+ orientation
    Parameters
    ----------
    nifti_list : list of nibabel.Nifti1Image or list of path-like object
    save : bool
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
    reoriented_list = []
    for nii in nifti_list:
        try:
            nii = load_nifti(nii)
            img = reorient_to_canonical(nii)
            if save:
                img.set_filename(nii.get_filename())
                if discard_errors:
                    nib.save(img, img.get_filename())
            reoriented_list.append(img)
        except TypeError as e:
            if isinstance(nii, nib.Nifti1Image):
                fname = nii.get_filename()
            else:
                fname = str(nii)
            if discard_errors:
                print('Error in file {}: {}\n the image will then be discarded from the list'
                      '\n WARNING: the output list will be shorter than the input list'.format(fname, e))
            else:
                raise TypeError('Error in file {}: {}'.format(fname, e))
        except ValueError as e:
            raise e
    if save and not discard_errors:
        for nii in reoriented_list:
            nib.save(nii, nii.get_filename())
    return reoriented_list


# for img in orientation_list:
#     ...:     if nifti_utils.get_nifti_orientation(img) != ('R', 'A', 'S'):
#     ...:         print(img)
#     ...:         print(nib.load(img).affine)
#     ...:         print(nib.as_closest_canonical(nib.load(img)).affine)
#     ...:         print(nib.load(img).affine == nib.as_closest_canonical(nib.load(img)).affine)
#     ...:         print(orientation_list[img])
#     ...:         print('######################### END IMAGE ##############################')
