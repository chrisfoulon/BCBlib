"""Low-level NIfTI I/O helpers: loading, saving, format detection."""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import nibabel as nib

from bcblib.imaging._types import NiftiLike


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def is_nifti(filename) -> bool:
    """Return *True* if *filename* ends with ``.nii`` or ``.nii.gz``."""
    s = str(filename)
    return s.endswith('.nii') or s.endswith('.nii.gz')


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_nifti(img: NiftiLike) -> nib.Nifti1Image:
    """Load a NIfTI image, accepting a path or an already-loaded image.

    Parameters
    ----------
    img : str, bytes, os.PathLike, or nibabel.Nifti1Image
        Path to a NIfTI file **or** an already-loaded image.

    Returns
    -------
    nibabel.Nifti1Image

    Raises
    ------
    ValueError
        If *img* is not a valid path or ``Nifti1Image``.
    """
    if isinstance(img, (str, bytes, os.PathLike)):
        p = Path(img)
        if p.is_file():
            img = nib.load(str(p), mmap=False)
    if not isinstance(img, nib.Nifti1Image):
        raise ValueError(
            f"bcblib.imaging.io.load_nifti: {img!r} must be a valid "
            "file path or a nibabel.Nifti1Image"
        )
    return img


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def resave_nifti(
    nifti: NiftiLike,
    output: Optional[Union[str, os.PathLike]] = None,
) -> Path:
    """Re-save a NIfTI image (strip mmap, refresh header).

    Parameters
    ----------
    nifti : NiftiLike
        Source image (path or loaded image).
    output : str or path-like, optional
        Destination path **or** directory.  When *None* the image is
        overwritten in place.

    Returns
    -------
    pathlib.Path
        Path to the saved file.
    """
    output_file = None
    output_dir = None
    if output is not None:
        output = Path(output)
        if output.is_dir():
            output_dir = output
        elif output.parent.is_dir():
            output_dir = output.parent
            output_file = output
        else:
            raise ValueError(f"{output.parent} is not an existing directory")

    img = load_nifti(nifti)
    if not output_dir:
        output_file = img.get_filename()
    if output_dir and not output_file:
        fname = img.get_filename()
        if fname is None:
            raise ValueError(
                "The image has no filename and no output path was given"
            )
        output_file = Path(output_dir, Path(fname).name)
    if not output_file:
        raise ValueError(
            "The image has no filename and no output path was given"
        )

    nib.save(nib.Nifti1Image(img.get_fdata(), img.affine), str(output_file))
    return Path(output_file)


def resave_nifti_list(
    nifti_list: List[NiftiLike],
    output_dir: Optional[Union[str, os.PathLike]] = None,
    save_in_place: bool = False,
    discard_errors: bool = False,
) -> Tuple[List[Path], List[str]]:
    """Re-save a list of NIfTI images.

    Parameters
    ----------
    nifti_list : list
        Paths or loaded images.
    output_dir : path-like, optional
        Destination directory (required unless *save_in_place* is True).
    save_in_place : bool
        Overwrite original files.
    discard_errors : bool
        If True, skip images that fail instead of raising.

    Returns
    -------
    (resaved, failed) : tuple of lists
    """
    if (output_dir is None or not Path(output_dir).is_dir()) and not save_in_place:
        raise Exception(
            "output_dir does not exist or is missing. "
            "output_dir MUST be given IF save_in_place is False"
        )
    resaved: List[Path] = []
    failed: List[str] = []
    for nii in nifti_list:
        try:
            fname = resave_nifti(nii, output=output_dir)
            resaved.append(fname)
        except Exception as e:
            if isinstance(nii, nib.Nifti1Image):
                fname_str = nii.get_filename() or "<unnamed>"
            else:
                fname_str = str(nii)
            failed.append(fname_str)
            if discard_errors:
                print(
                    f"Error in file {fname_str}: {e}\n"
                    " the image will be discarded from the list\n"
                    " WARNING: the output list will be shorter than the input"
                )
            else:
                raise TypeError(f"Error in file {fname_str}: {e}") from e
    return resaved, failed
