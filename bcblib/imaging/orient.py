"""NIfTI orientation utilities: query, reorient, and swap axes."""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import nibabel as nib
from tqdm import tqdm

from bcblib.imaging._types import NiftiLike
from bcblib.imaging.io import load_nifti


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

def get_orientation(img: NiftiLike) -> Tuple[str, ...]:
    """Return the 3-letter axis-code tuple, e.g. ``('R', 'A', 'S')``.

    Parameters
    ----------
    img : NiftiLike
    """
    nii = load_nifti(img)
    return nib.aff2axcodes(nii.affine)


# ---------------------------------------------------------------------------
# Reorientation
# ---------------------------------------------------------------------------

def reorient_to_standard(img: NiftiLike, save: bool = False) -> nib.Nifti1Image:
    """Reorient an image to the closest canonical (RAS+) orientation.

    Parameters
    ----------
    img : NiftiLike
    save : bool
        If *True*, overwrite the source file on disk.

    Returns
    -------
    nibabel.Nifti1Image
    """
    nii = load_nifti(img)
    if not nii.get_filename():
        raise ValueError(
            "The NIfTI image has no associated filename and cannot be reoriented"
        )
    canonical = nib.as_closest_canonical(nii)
    if save:
        canonical.set_filename(nii.get_filename())
        nib.save(canonical, nii.get_filename())
    return canonical


def reorient_list(
    nifti_list: List[NiftiLike],
    output_dir: Optional[Union[str, os.PathLike]] = None,
    save_in_place: bool = False,
    discard_errors: bool = False,
) -> Tuple[list, list]:
    """Reorient a list of images to canonical orientation.

    Parameters
    ----------
    nifti_list : list
        Paths or loaded images.
    output_dir : path-like, optional
        Destination directory.
    save_in_place : bool
        Overwrite originals.
    discard_errors : bool
        Skip failures instead of raising.

    Returns
    -------
    (reoriented, failed) : tuple of lists
    """
    if (output_dir is None or not Path(output_dir).is_dir()) and not save_in_place:
        raise Exception(
            "output_dir does not exist or is missing. "
            "output_dir MUST be given IF save_in_place is False"
        )
    reoriented: list = []
    failed: list = []
    for item in tqdm(nifti_list, desc="Reorient NIfTI list"):
        try:
            nii = load_nifti(item)
            canonical = reorient_to_standard(nii)
            if output_dir is not None and Path(output_dir).is_dir():
                canonical.set_filename(
                    Path(output_dir, Path(nii.get_filename()).name)
                )
            if save_in_place:
                canonical.set_filename(nii.get_filename())
            nib.save(canonical, canonical.get_filename())
            reoriented.append(canonical.get_filename())
        except Exception as e:
            if isinstance(item, nib.Nifti1Image):
                fname = item.get_filename()
            else:
                fname = str(item)
            failed.append(fname)
            if discard_errors:
                print(
                    f"Error in file {fname}: {e}\n"
                    " the image will be discarded from the list"
                )
            else:
                raise TypeError(f"Error in file {fname}: {e}") from e
    return reoriented, failed


# ---------------------------------------------------------------------------
# New helpers
# ---------------------------------------------------------------------------

def set_orientation(
    img: NiftiLike,
    target: str = "RAS",
) -> nib.Nifti1Image:
    """Reorient an image to an arbitrary *target* orientation code.

    Parameters
    ----------
    img : NiftiLike
    target : str
        Three-character orientation string, e.g. ``"RAS"``, ``"LPI"``.

    Returns
    -------
    nibabel.Nifti1Image
    """
    nii = load_nifti(img)
    current = nib.aff2axcodes(nii.affine)
    target_codes = tuple(target.upper())

    if current == target_codes:
        return nii

    # Build orientation transform
    src_ornt = nib.io_orientation(nii.affine)
    tgt_ornt = nib.orientations.axcodes2ornt(target_codes)
    transform = nib.orientations.ornt_transform(src_ornt, tgt_ornt)
    reoriented = nii.as_reoriented(transform)
    return reoriented


def swap_dimensions(
    img: NiftiLike,
    new_axes: Tuple[str, str, str],
) -> nib.Nifti1Image:
    """Reorder image dimensions according to *new_axes* (e.g. ``('x','z','y')``).

    This is a convenience wrapper around :func:`set_orientation` that
    accepts FSL-style ``x/y/z`` or ``-x/-y/-z`` axis names.

    Parameters
    ----------
    img : NiftiLike
    new_axes : tuple of str
        Three axis descriptors.  Positive axes are ``'x'``, ``'y'``, ``'z'``;
        negated axes are ``'-x'``, ``'-y'``, ``'-z'``.

    Returns
    -------
    nibabel.Nifti1Image
    """
    nii = load_nifti(img)

    # Map x/y/z to current orientation axis codes
    current = list(nib.aff2axcodes(nii.affine))
    axis_map = {
        "x": current[0],
        "y": current[1],
        "z": current[2],
    }
    # Flip map: negation inverts the axis label
    flip_map = {
        "R": "L", "L": "R",
        "A": "P", "P": "A",
        "S": "I", "I": "S",
    }

    target_codes = []
    for ax in new_axes:
        ax = ax.strip()
        if ax.startswith("-"):
            base = axis_map[ax[1:]]
            target_codes.append(flip_map[base])
        else:
            target_codes.append(axis_map[ax])

    return set_orientation(nii, "".join(target_codes))
