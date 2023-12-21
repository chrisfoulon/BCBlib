import os
from pathlib import Path
import csv
from typing import Union

from nilearn.regions import connected_regions
from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.ndimage.measurements import center_of_mass

import nibabel as nib


def is_nifti(filename):
    return str(filename)[-4:] == '.nii' or str(filename)[-7:] == '.nii.gz'


def file_to_list(file_path, delimiter=' '):
    file_path = Path(file_path)
    if not file_path.is_file():
        raise ValueError(f'{file_path} does not exist.')
    if file_path.name.endswith('.csv'):
        with open(file_path, 'r') as csv_file:
            dir_list = []
            for row in csv.reader(csv_file):
                if len(row) > 1:
                    dir_list += [r for r in row]
                else:
                    dir_list.append(row[0])
    else:
        # default delimiter is ' ', it might need to be changed
        dir_list = np.loadtxt(str(file_path), dtype=str, delimiter=delimiter)
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
        raise ValueError(
            'bcblib.tools.load_nifti error: {} must either be a valid path or a nibabel.Nifti1Image'.format(img))
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
    for nii in tqdm(nifti_list, desc='Reorient Nifti list'):
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
        optional if save_in_place is True. If given, the reoriented images will be saved in the output directory
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


def get_centre_of_mass(nifti, round_coord=False):
    nii = load_nifti(nifti)
    if not nii.get_fdata().any():
        return tuple(np.zeros(len(nii.shape)))
    if round_coord:
        return np.round(center_of_mass(np.nan_to_num(np.abs(nii.get_fdata()))))
    else:
        return center_of_mass(np.nan_to_num(np.abs(nii.get_fdata())))


def centre_of_mass_difference(nifti, reference, round_centres=False):
    if not (isinstance(reference, list) or isinstance(reference, tuple) or isinstance(reference, set)):
        reference = get_centre_of_mass(reference, round_centres)
    nii_centre = get_centre_of_mass(nifti, round_centres)
    if len(nii_centre) != len(reference):
        raise ValueError('Nifti image ({}) and reference ({}) must have the same number of dimensions'.format(
            nii_centre, reference))
    return euclidean(nii_centre, reference)


def centre_of_mass_difference_list(nifti_list, reference, fname_filter=None, round_centres=False):
    if fname_filter is not None:
        nifti_list = [f for f in nifti_list if fname_filter in Path(f).name]
    distance_dict = {}
    for f in nifti_list:
        distance_dict[str(f)] = centre_of_mass_difference(f, reference, round_centres)
    return distance_dict


def nifti_overlap_images(input_images, filter_pref='', recursive=False, mean=False, mean_for_std=None):
    if not isinstance(input_images, list):
        if Path(input_images).is_file():
            input_images = [str(p) for p in file_to_list(input_images) if is_nifti(p)]
        elif Path(input_images).is_dir():
            if recursive:
                input_images = [str(p) for p in Path(input_images).rglob('*') if is_nifti(p)]
            else:
                input_images = [str(p) for p in Path(input_images).iterdir() if is_nifti(p)]
        else:
            raise ValueError('Wrong input (must be a file/directory path of a list of paths)')
    if mean_for_std is not None and mean:
        raise ValueError('mean_for_std can only be used if mean is False')
    if mean_for_std is not None:
        # if mean_for_std is pathlike, load the image
        if isinstance(mean_for_std, str) or isinstance(mean_for_std, Path) or is_nifti(mean_for_std):
            mean_for_std = load_nifti(mean_for_std).get_fdata()
        elif not isinstance(mean_for_std, np.ndarray):
            raise ValueError('mean_for_std must be a pathlike object, a nifti image or a numpy array')
    if filter_pref:
        input_images = [p for p in input_images if Path(p).name.startswith(filter_pref)]
    if not input_images:
        print(' The image list is empty')
        return None
    temp_overlap = None
    temp_overlap_data = None
    for img in tqdm(input_images):
        nii = nib.load(img)
        if temp_overlap is None:
            temp_overlap = nii
            temp_overlap_data = nii.get_fdata()
        else:
            if mean:
                temp_overlap_data = temp_overlap_data + nii.get_fdata()
            if mean_for_std is not None:
                # imaging_stats['std'][modality] += np.square(current_data - imaging_stats['mean'][modality])
                temp_overlap_data += np.square(nii.get_fdata() - mean_for_std)
            else:
                temp_overlap_data += nii.get_fdata()
    if mean_for_std is not None:
        # imaging_stats['std'][key] = np.sqrt(imaging_stats['std'][key] / counters[key])
        temp_overlap_data = np.sqrt(temp_overlap_data / len(input_images))
    if mean:
        temp_overlap_data = temp_overlap_data / len(input_images)
    temp_overlap = nib.Nifti1Image(temp_overlap_data, temp_overlap.affine)
    return temp_overlap


def overlaps_subfolders(root_folder, filter_pref='', subfolders_recursive=True,
                        subfolders_overlap=False, output_pref='overlap_',
                        save_in_root=True, mean=False, std_pref=None):
    """
    Compute the overlap of all the images in the subfolders of a root folder
    Notes: subfolders_overlap and subfolders_recursive are interacting with each other.
    If both are True, each subfolder overlap will be the overlap of all the subfolders of the subfolder
    Parameters
    ----------
    root_folder : str
    filter_pref : str
    subfolders_recursive : bool
    subfolders_overlap : bool
    output_pref : str
    save_in_root : bool
    mean : bool
    std_pref : str

    Returns
    -------

    """
    if not Path(root_folder).is_dir():
        raise ValueError('root_folder must be a directory')
    if subfolders_recursive and subfolders_overlap:
        raise ValueError('subfolders_recursive and subfolders_overlap cannot be True together')
    if subfolders_overlap:
        folder_list = [p for p in Path(root_folder).rglob('*') if p.is_dir()]
    else:
        folder_list = [p for p in Path(root_folder).iterdir() if p.is_dir()]
    if not mean and std_pref is not None:
        print('WARNING: std_pref is set but mean is False, mean will be computed anyway (if not already present)')
        mean = True

    for subfolder in folder_list:
        if save_in_root:
            output_folder = Path(root_folder)
        else:
            output_folder = Path(root_folder, subfolder.relative_to(root_folder).parent)
        overlap_path = output_folder / (output_pref + subfolder.name + '.nii.gz')
        if std_pref is not None:
            std_output_path = Path(root_folder, subfolder.relative_to(root_folder).parent,
                                   std_pref + subfolder.name + '.nii.gz')
            if overlap_path.is_file():
                print(f'Mean image found for [{subfolder.name}], computing std')
                overlap_nifti = load_nifti(overlap_path)
            else:
                print(f'Computing mean image for [{subfolder.name}]')
                overlap_nifti = nifti_overlap_images(subfolder, filter_pref, recursive=subfolders_recursive, mean=True)
                if overlap_nifti is not None:
                    nib.save(overlap_nifti, overlap_path)
            if overlap_nifti is not None:
                print(f'Computing std image for [{subfolder.name}]')
                std_nifti = nifti_overlap_images(subfolder, filter_pref, recursive=subfolders_recursive,
                                                 mean_for_std=overlap_nifti.get_fdata())
                if std_nifti is not None:
                    nib.save(std_nifti, std_output_path)
        else:
            output_type = 'Overlap'
            if mean:
                output_type = 'Mean'
            print(f'{output_type} of [{subfolder.name}]')
            overlap_nifti = nifti_overlap_images(subfolder, filter_pref, recursive=subfolders_recursive, mean=mean)
            if overlap_nifti is not None:
                nib.save(overlap_nifti, overlap_path)


def binarize_nii(nii: Union[os.PathLike, nib.Nifti1Image], thr: Union[float, int] = None):
    """
    Binarize the input image by setting everything >= thr to 1 and everything < thr to 0
    Parameters
    ----------
    nii : Union[os.PathLike, nib.Nifti1Image]
    thr : Union[float, int]

    Returns
    -------
    nib.Nifti1Image

    Note:
    -------
    Warning: It is assumed that background intensity is smaller than anything else.
    Warning 2: if the image only contains one value, it is returned without changed as it is impossible to know
    whether this is background or foreground
    """
    hdr = load_nifti(nii)
    data = hdr.get_fdata()
    # The is already binary (or only contain 1 value)
    unique_val = set(np.unique(data))
    if len(unique_val) == 1 or unique_val == {0, 1}:
        return hdr
    thr_data = np.zeros(data.shape)
    if thr is not None:
        thr_data[data >= thr] = 1
        thr_data[data < thr] = 0
    else:
        background_intensity = np.min(data)
        if background_intensity != 1:
            thr_data[data > background_intensity] = 1
            thr_data[data == background_intensity] = 0
        else:
            thr_data[data == background_intensity] = 0
            thr_data[data > background_intensity] = 1
    return nib.Nifti1Image(thr_data, hdr.affine)


def reorient_image(img, orientation):
    # TODO
    img = nib.load(img)
    ornt = np.array([[0, 1],
                     [1, -1],
                     [2, 1]])
    img = nib.load(img)
    img = img.as_reoriented(ornt)
    return img


def laterality_ratio(image):
    """
    Computes the ratio of volume
    Parameters
    ----------
    image

    Returns
    -------

    """
    hdr = load_nifti(image)
    ori = nib.aff2axcodes(hdr.affine)
    data = hdr.get_fdata()
    mid_x = int(data.shape[0] / 2)
    # So, if the lesioned voxels have x <= mid_x they are on the RIGHT (in LAS orientation)
    coord = np.where(data)
    if ((ori[0] == 'L' and ori[1] == 'A') or (ori[0] == 'R' and ori[1] == 'P')) and ori[2] == 'S':
        right_les = len(coord[0][coord[0] <= mid_x])
        left_les = len(coord[0][coord[0] > mid_x])
    else:
        right_les = len(coord[0][coord[0] > mid_x])
        left_les = len(coord[0][coord[0] <= mid_x])
    total_les_vol = len(coord[0])
    right_ratio = right_les / total_les_vol
    left_ratio = left_les / total_les_vol
    return left_ratio - right_ratio


def has_big_enough_cluster(img, min_cluster_size=4):
    hdr = load_nifti(img)
    data, _ = connected_regions(hdr, min_region_size=1, extract_type='connected_components', smoothing_fwhm=0)
    data = data.get_fdata()
    max_cluster_size = np.max([np.count_nonzero(data[..., i]) for i in range(data.shape[-1])])
    return max_cluster_size >= min_cluster_size


def get_volume(img, ratio=False, threshold=0):
    if not ratio:
        # volume of voxels with > threshold
        data = load_nifti(img).get_fdata()
        return np.count_nonzero(data > threshold)
    else:
        # ratio of voxels with > threshold by the total number of voxels
        data = load_nifti(img).get_fdata()
        return np.count_nonzero(data > threshold) / np.prod(data.shape)


def get_dispersion(img):
    nii = load_nifti(img)
    # zeros to non-zero values ratios
    data = nii.get_fdata()
    zero_nonzero_ratio = np.count_nonzero(data) / np.prod(data.shape)
    mean_to_median_ratio = np.mean(data) / np.median(data)
    """
    Returns:

    regions_extracted_imgnibabel.nifti1.Nifti1Image

        Gives the image in 4D of extracted brain regions. Each 3D image consists of only one separated region.
    index_of_each_mapnumpy.ndarray

        An array of list of indices where each index denotes the identity of each extracted region to their family of brain maps.

    """
    clusters_4d, cluster_indices = connected_regions(nii, min_region_size=1, extract_type='connected_components', fwhm=0)
    clusters_4d = clusters_4d.get_fdata()
    mean_cluster_size = np.mean([np.count_nonzero(clusters_4d[..., i]) for i in range(clusters_4d.shape[-1])])
    # distance between the centre of mass of every cluster to the centre (coordinate) of the image
    centre_of_image = np.array(nii.shape) / 2
    # TODO
    # mean_distance_to_center =
    # return zero_nonzero_ratio, mean_to_median_ratio, mean_cluster_size, distance_to_center
