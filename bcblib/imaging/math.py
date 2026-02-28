"""Voxel-wise maths on NIfTI images: binarize, threshold, morphology, masking, arithmetic."""

import os
from typing import Optional, Union

import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure

from bcblib.imaging._types import NiftiLike
from bcblib.imaging.io import load_nifti


# =========================================================================
# Private array-level helpers  (preserve old images_utils.py contract)
# =========================================================================

def _dilate_array(array: np.ndarray, connectivity: int, dimensions: int = 3) -> np.ndarray:
    """Dilate a binary array using a structure with the given connectivity."""
    struct = generate_binary_structure(dimensions, connectivity)
    return binary_dilation(array, struct).astype(array.dtype)


def _erode_array(array: np.ndarray, connectivity: int, dimensions: int = 3) -> np.ndarray:
    """Erode a binary array using a structure with the given connectivity."""
    struct = generate_binary_structure(dimensions, connectivity)
    return binary_erosion(array, struct).astype(array.dtype)


def _mask_in_array(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero out voxels where *mask* is zero (keep only inside-mask values)."""
    out = np.copy(array)
    out[mask == 0] = 0
    return out


def _mask_out_array(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero out voxels where *mask* is non-zero (keep only outside-mask values)."""
    out = np.copy(array)
    out[mask == 1] = 0
    return out


# =========================================================================
# NIfTI-level public API
# =========================================================================

# ---------------------------------------------------------------------------
# Binarize / Threshold (moved from nifti_utils.binarize_nii)
# ---------------------------------------------------------------------------

def binarize(
    img: NiftiLike,
    thr: Optional[Union[float, int]] = None,
) -> nib.Nifti1Image:
    """Binarize an image: voxels >= *thr* become 1, the rest 0.

    When *thr* is ``None`` the minimum intensity is treated as background.

    Parameters
    ----------
    img : NiftiLike
    thr : float or int, optional
    """
    nii = load_nifti(img)
    data = nii.get_fdata()
    unique_val = set(np.unique(data))
    if len(unique_val) == 1 or unique_val == {0, 1}:
        return nii  # already binary or single-value
    out = np.zeros(data.shape)
    if thr is not None:
        out[data >= thr] = 1
    else:
        bg = np.min(data)
        if bg != 1:
            out[data > bg] = 1
        else:
            out[data > bg] = 1
    return nib.Nifti1Image(out, nii.affine)


def threshold(
    img: NiftiLike,
    low: Optional[float] = None,
    high: Optional[float] = None,
) -> nib.Nifti1Image:
    """Zero out voxels outside ``[low, high]``.

    Parameters
    ----------
    img : NiftiLike
    low : float, optional
        Values < *low* are set to 0.
    high : float, optional
        Values > *high* are set to 0.
    """
    nii = load_nifti(img)
    data = nii.get_fdata().copy()
    if low is not None:
        data[data < low] = 0
    if high is not None:
        data[data > high] = 0
    return nib.Nifti1Image(data, nii.affine)


# ---------------------------------------------------------------------------
# Morphology
# ---------------------------------------------------------------------------

def dilate(
    img: NiftiLike,
    connectivity: int = 1,
    dimensions: int = 3,
) -> nib.Nifti1Image:
    """Dilate a binary mask image.

    Parameters
    ----------
    img : NiftiLike
    connectivity : int
        Connectivity of the structuring element (1–3).
    dimensions : int
    """
    nii = load_nifti(img)
    out = _dilate_array(nii.get_fdata(), connectivity, dimensions)
    return nib.Nifti1Image(out, nii.affine)


def erode(
    img: NiftiLike,
    connectivity: int = 1,
    dimensions: int = 3,
) -> nib.Nifti1Image:
    """Erode a binary mask image.

    Parameters
    ----------
    img : NiftiLike
    connectivity : int
    dimensions : int
    """
    nii = load_nifti(img)
    out = _erode_array(nii.get_fdata(), connectivity, dimensions)
    return nib.Nifti1Image(out, nii.affine)


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

def apply_mask(
    img: NiftiLike,
    mask: NiftiLike,
) -> nib.Nifti1Image:
    """Apply a mask: zero out voxels where *mask* is zero.

    Parameters
    ----------
    img : NiftiLike
    mask : NiftiLike
    """
    nii = load_nifti(img)
    mask_data = load_nifti(mask).get_fdata()
    out = _mask_in_array(nii.get_fdata(), mask_data)
    return nib.Nifti1Image(out, nii.affine)


def apply_inverse_mask(
    img: NiftiLike,
    mask: NiftiLike,
) -> nib.Nifti1Image:
    """Apply an inverse mask: zero out voxels where *mask* is non-zero.

    Parameters
    ----------
    img : NiftiLike
    mask : NiftiLike
    """
    nii = load_nifti(img)
    mask_data = load_nifti(mask).get_fdata()
    out = _mask_out_array(nii.get_fdata(), mask_data)
    return nib.Nifti1Image(out, nii.affine)


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------

def add(
    img1: NiftiLike,
    img2: NiftiLike,
) -> nib.Nifti1Image:
    """Element-wise addition of two images.

    Parameters
    ----------
    img1, img2 : NiftiLike
        Must have the same shape.
    """
    a = load_nifti(img1)
    b = load_nifti(img2)
    return nib.Nifti1Image(a.get_fdata() + b.get_fdata(), a.affine)


def subtract(
    img1: NiftiLike,
    img2: NiftiLike,
) -> nib.Nifti1Image:
    """Element-wise subtraction (*img1* - *img2*).

    Parameters
    ----------
    img1, img2 : NiftiLike
    """
    a = load_nifti(img1)
    b = load_nifti(img2)
    return nib.Nifti1Image(a.get_fdata() - b.get_fdata(), a.affine)


def multiply(
    img1: NiftiLike,
    img2: NiftiLike,
) -> nib.Nifti1Image:
    """Element-wise multiplication of two images.

    Parameters
    ----------
    img1, img2 : NiftiLike
    """
    a = load_nifti(img1)
    b = load_nifti(img2)
    return nib.Nifti1Image(a.get_fdata() * b.get_fdata(), a.affine)
