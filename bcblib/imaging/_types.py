"""Shared type aliases for the imaging subpackage."""

import os
from typing import Union

import nibabel as nib

# Anything that ``io.load_nifti`` can accept.
NiftiLike = Union[str, bytes, os.PathLike, nib.Nifti1Image]
