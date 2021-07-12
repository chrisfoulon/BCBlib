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
