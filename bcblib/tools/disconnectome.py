#!/usr/bin/env python3
# -*-coding:Utf-8 -*

import nibabel as nib
import numpy as np
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.utils import is_header_compatible
from dipy.tracking.vox2track import _streamlines_in_mask
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.tracking.utils import density_map
from bcblib.tools.nifti_utils import load_nifti
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.segment.streamlines import filter_grid_roi


o_dict = {}


sft = load_tractogram(args.in_tractogram, 'same', bbox_valid_check=True)
print(f'Input streamlines: {len(sft.streamlines)}')
roi = load_nifti(roi_path)
roi_data = roi.get_fdata()
if not all(np.unique(roi_data) == [0, 1]):
    raise ValueError('ROI mask does not contain only 0s and 1s')
if not is_header_compatible(roi, sft):
    raise ValueError('Headers from the tractogram and the mask are not compatible.')
# mask = get_data_as_mask(img)

sft.to_vox()
sft.to_corner()
target_mask = np.array(roi_data, dtype=np.uint8, copy=True)
streamlines_case = _streamlines_in_mask(list(sft.streamlines),
                                        target_mask,
                                        np.eye(3), [0, 0, 0])
line_based_indices = np.where(streamlines_case == [0, 1][True])[0].tolist()
line_based_indices = np.asarray(line_based_indices).astype(np.int32)

# From indices to sft
streamlines = sft.streamlines[line_based_indices]
data_per_streamline = sft.data_per_streamline[line_based_indices]
data_per_point = sft.data_per_point[line_based_indices]

new_sft = StatefulTractogram.from_sft(streamlines, sft,
                                      data_per_streamline=data_per_streamline,
                                      data_per_point=data_per_point)

filtered_sft, _ = new_sft, line_based_indices
save_tractogram(filtered_sft, args.out_tractogram)
max_ = np.iinfo(np.int16).max
# if args.binary is not None and (args.binary <= 0 or args.binary > max_):
#     parser.error('The value of --binary ({}) '
#                  'must be greater than 0 and smaller or equal to {}'
#                  .format(args.binary, max_))
#
# sft = load_tractogram_with_reference(parser, args, args.in_bundle)
sft.to_vox()
sft.to_corner()
streamlines = sft.streamlines
transformation, dimensions, _, _ = sft.space_attributes

density_data = density_map(streamlines, transformation, dimensions)
density_data[density_data > 0] = args.binary
out_nii = nib.Nifti1Image(density_data, transformation)
nib.save(out_nii, args.out_img)
# streamline_count = compute_tract_counts_map(streamlines, dimensions)
#
# if args.binary is not None:
#     streamline_count[streamline_count > 0] = args.binary
#
# nib.save(nib.Nifti1Image(streamline_count.astype(np.int16), transformation),
#          args.out_img)

import nibabel as nib
from dipy.io.streamline import load_trk
# from dipy.tracking.utils import partial_tractogram

# def compute_partial_tractography(tractography_path, nifti_mask_path):
#     # load the tractography file
#     tractography, hdr = load_trk(tractography_path, lazy_load=True)
#
#     # load the NIFTI mask
#     nifti_mask_img = nib.load(nifti_mask_path)
#     nifti_mask_data = nifti_mask_img.get_data()
#
#     # compute the partial tractography
#     partial_tract = partial_tractogram(tractography, nifti_mask_data)
#
#     return partial_tract

# test the function
# partial_tract = compute_partial_tractography('path/to/tractography.trk', 'path/to/nifti/mask.nii.gz')


def apply_mask(tracts, mask, min_len=0, max_len=np.inf):
    """ Applies a binary mask to a set of tracts.

    Parameters
    ----------
    tracts : sequence of ndarrays or generator
        The tracts to be masked.
    mask : ndarray
        A binary mask with the same shape as the tracts.
    min_len : int, optional
        The minimum length of tracts to keep.
    max_len : int, optional
        The maximum length of tracts to keep.

    Returns
    -------
    generator
        A generator object containing the masked tracts.
    """
    mask = np.asarray(mask)
    for tract in tracts:
        # Compute the length of the tract
        tract_len = len(tract)
        # Skip the tract if it is outside the length range
        if tract_len < min_len or tract_len > max_len:
            continue
        # Apply the mask to the tract
        masked_tract = tract[mask[tuple(tract.T)]]
        # Return the masked tract
        yield masked_tract


# import nibabel as nib
# from dipy.tracking.utils import apply_mask

# def partial_tractography(tract_path, mask_path):
#     # Load the tractography file
#     tracts = nib.streamlines.load(tract_path)
#     # Load the NIFTI mask
#     mask_img = nib.load(mask_path)
#     # Apply the mask to the tractography data
#     tracts = apply_mask(tracts, mask_img.get_data())
#     # Return the partial tractography
#     return tracts


def partial_tractography(tract_path, mask_path, output_path):
    # Load the tractography file
    tracts = nib.streamlines.load(tract_path)
    # Load the NIFTI mask
    mask_img = nib.load(mask_path)
    # Apply the mask to the tractography data
    tracts = apply_mask(tracts, mask_img.get_data())
    # Create a NIFTI image with the same shape as the mask
    output_img = nib.Nifti1Image(mask_img.shape, mask_img.affine)
    # Set the voxels in the output image to 1 where there are tracts
    for tract in tracts:
        output_img.get_data()[tuple(tract.T)] = 1
    # Save the output image
    nib.save(output_img, output_path)

