"""NIfTI structural manipulation: ROI extraction, merging, splitting, geometry copying."""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import nibabel as nib

from bcblib.imaging._types import NiftiLike
from bcblib.imaging.io import load_nifti


# ---------------------------------------------------------------------------
# extract_roi  (fslroi equivalent)
# ---------------------------------------------------------------------------

def extract_roi(
    img: NiftiLike,
    x_min: int = 0,
    x_size: int = -1,
    y_min: int = 0,
    y_size: int = -1,
    z_min: int = 0,
    z_size: int = -1,
    t_min: int = 0,
    t_size: int = -1,
) -> nib.Nifti1Image:
    """Extract a region of interest from a NIfTI image (like ``fslroi``).

    A size of ``-1`` means "take all remaining voxels along that axis".

    Parameters
    ----------
    img : NiftiLike
    x_min, x_size, y_min, y_size, z_min, z_size, t_min, t_size : int

    Returns
    -------
    nibabel.Nifti1Image
    """
    nii = load_nifti(img)
    data = nii.get_fdata()

    def _end(start, size, dim_len):
        if size == -1:
            return dim_len
        return min(start + size, dim_len)

    x_end = _end(x_min, x_size, data.shape[0])
    y_end = _end(y_min, y_size, data.shape[1])
    z_end = _end(z_min, z_size, data.shape[2])

    if data.ndim >= 4:
        t_end = _end(t_min, t_size, data.shape[3])
        roi_data = data[x_min:x_end, y_min:y_end, z_min:z_end, t_min:t_end]
    else:
        roi_data = data[x_min:x_end, y_min:y_end, z_min:z_end]

    # Update the affine origin to reflect the spatial crop
    new_affine = nii.affine.copy()
    voxel_offset = np.array([x_min, y_min, z_min, 1], dtype=float)
    new_affine[:3, 3] = (nii.affine @ voxel_offset)[:3]

    return nib.Nifti1Image(roi_data, new_affine, nii.header)


# ---------------------------------------------------------------------------
# merge_images  (fslmerge equivalent)
# ---------------------------------------------------------------------------

def merge_images(
    images: List[NiftiLike],
    axis: int = 3,
) -> nib.Nifti1Image:
    """Concatenate images along *axis* (like ``fslmerge``).

    Parameters
    ----------
    images : list of NiftiLike
        Must all share the same spatial dimensions.
    axis : int
        Axis to concatenate along (default 3 = time).

    Returns
    -------
    nibabel.Nifti1Image
    """
    if not images:
        raise ValueError("images list is empty")

    loaded = [load_nifti(im) for im in images]
    arrays = []
    for nii in loaded:
        d = nii.get_fdata()
        # Promote 3-D to 4-D if merging along t
        if axis >= d.ndim:
            d = d[..., np.newaxis]
        arrays.append(d)

    merged = np.concatenate(arrays, axis=axis)
    return nib.Nifti1Image(merged, loaded[0].affine, loaded[0].header)


# ---------------------------------------------------------------------------
# split_image  (fslsplit equivalent)
# ---------------------------------------------------------------------------

def split_image(
    img: NiftiLike,
    axis: int = 3,
) -> List[nib.Nifti1Image]:
    """Split an image along *axis* into a list of volumes (like ``fslsplit``).

    Parameters
    ----------
    img : NiftiLike
    axis : int
        Default 3 (time).

    Returns
    -------
    list of nibabel.Nifti1Image
    """
    nii = load_nifti(img)
    data = nii.get_fdata()
    if axis >= data.ndim:
        raise ValueError(
            f"Cannot split along axis {axis}; image has {data.ndim} dimensions"
        )

    slices = np.split(data, data.shape[axis], axis=axis)
    volumes = []
    for s in slices:
        vol = np.squeeze(s, axis=axis)
        volumes.append(nib.Nifti1Image(vol, nii.affine, nii.header))
    return volumes


# ---------------------------------------------------------------------------
# copy_geometry  (fslcpgeom equivalent)
# ---------------------------------------------------------------------------

def copy_geometry(
    source: NiftiLike,
    target: NiftiLike,
) -> nib.Nifti1Image:
    """Copy the affine and header geometry from *source* onto *target*.

    The voxel data of *target* is preserved; only the spatial metadata
    (affine, pixdim, sform/qform codes) is taken from *source*.

    Parameters
    ----------
    source : NiftiLike
        Image whose geometry to copy.
    target : NiftiLike
        Image whose data to keep.

    Returns
    -------
    nibabel.Nifti1Image
    """
    src = load_nifti(source)
    tgt = load_nifti(target)

    new_header = src.header.copy()
    # Preserve the data-shape fields from the target
    tgt_hdr = tgt.header
    new_header["dim"] = tgt_hdr["dim"]
    new_header.set_data_dtype(tgt_hdr.get_data_dtype())

    return nib.Nifti1Image(tgt.get_fdata(), src.affine, new_header)
