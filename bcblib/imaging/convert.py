"""NIfTI format conversion and image creation utilities."""

import os
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import nibabel as nib

from bcblib.imaging._types import NiftiLike
from bcblib.imaging.io import load_nifti


# ---------------------------------------------------------------------------
# convert_format
# ---------------------------------------------------------------------------

def convert_format(
    img: NiftiLike,
    output: Union[str, os.PathLike],
) -> Path:
    """Save an image in a different NIfTI format (e.g. ``.nii`` <-> ``.nii.gz``).

    The format is determined by the *output* file extension.

    Parameters
    ----------
    img : NiftiLike
    output : path-like
        Destination path (must end with ``.nii`` or ``.nii.gz``).

    Returns
    -------
    pathlib.Path
        The path to the saved file.
    """
    nii = load_nifti(img)
    output = Path(output)
    if not (str(output).endswith(".nii") or str(output).endswith(".nii.gz")):
        raise ValueError(
            f"output must end with .nii or .nii.gz, got: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nii, str(output))
    return output


# ---------------------------------------------------------------------------
# create_image
# ---------------------------------------------------------------------------

def create_image(
    shape: Tuple[int, ...],
    affine: Optional[np.ndarray] = None,
    dtype: Union[str, np.dtype] = np.float64,
    fill: float = 0.0,
    like: Optional[NiftiLike] = None,
) -> nib.Nifti1Image:
    """Create a new NIfTI image filled with a constant value.

    Parameters
    ----------
    shape : tuple of int
        Voxel dimensions.
    affine : ndarray, optional
        4x4 affine matrix.  Defaults to identity if *like* is not given.
    dtype : str or numpy dtype
    fill : float
        Value to fill every voxel with (default 0).
    like : NiftiLike, optional
        If provided, ``affine`` and ``shape`` default to those of *like*.

    Returns
    -------
    nibabel.Nifti1Image
    """
    if like is not None:
        ref = load_nifti(like)
        if affine is None:
            affine = ref.affine.copy()
        if shape is None:
            shape = ref.shape

    if affine is None:
        affine = np.eye(4)

    data = np.full(shape, fill, dtype=dtype)
    return nib.Nifti1Image(data, affine)
