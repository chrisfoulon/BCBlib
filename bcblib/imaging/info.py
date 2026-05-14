"""NIfTI header inspection utilities (``fslinfo`` / ``fslhd`` / ``fslval`` equivalents)."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import nibabel as nib

from bcblib.imaging._types import NiftiLike
from bcblib.imaging.io import load_nifti


# ---------------------------------------------------------------------------
# header_field  (fslval equivalent)
# ---------------------------------------------------------------------------

def header_field(img: NiftiLike, field: str) -> Any:
    """Return a single header field value (like ``fslval``).

    Parameters
    ----------
    img : NiftiLike
        Path or loaded image.
    field : str
        Header field name, e.g. ``"dim1"``, ``"pixdim1"``, ``"datatype"``.
        Shorthand ``dimN`` / ``pixdimN`` is supported so that
        ``header_field(img, "dim1")`` returns the first spatial dimension.

    Returns
    -------
    scalar or array
        The value stored in the header for *field*.

    Raises
    ------
    KeyError
        If *field* is not found in the header.
    """
    hdr = load_nifti(img).header

    # Convenience aliases: dim1..dim7, pixdim1..pixdim7
    for prefix, hdr_key in (("dim", "dim"), ("pixdim", "pixdim")):
        if field.startswith(prefix) and field[len(prefix):].isdigit():
            idx = int(field[len(prefix):])
            arr = hdr[hdr_key]
            if idx < 0 or idx >= len(arr):
                raise KeyError(f"{field}: index {idx} out of range")
            return arr[idx]

    if field in hdr:
        return hdr[field]
    raise KeyError(f"Unknown header field: {field!r}")


# ---------------------------------------------------------------------------
# header_summary  (fslinfo equivalent)
# ---------------------------------------------------------------------------

def header_summary(img: NiftiLike) -> Dict[str, Any]:
    """Return a concise dictionary of key image properties (like ``fslinfo``).

    Keys returned: ``filename``, ``data_type``, ``dim1``..``dim4``,
    ``pixdim1``..``pixdim4``, ``cal_min``, ``cal_max``, ``orientation``.

    Parameters
    ----------
    img : NiftiLike

    Returns
    -------
    dict
    """
    nii = load_nifti(img)
    hdr = nii.header
    dims = hdr["dim"]
    pixdims = hdr["pixdim"]
    ndim = int(dims[0])

    info: Dict[str, Any] = {}
    info["filename"] = nii.get_filename() or "<in-memory>"
    info["data_type"] = hdr.get_data_dtype().name
    for i in range(1, min(ndim + 1, 8)):
        info[f"dim{i}"] = int(dims[i])
        info[f"pixdim{i}"] = float(pixdims[i])
    info["cal_min"] = float(hdr["cal_min"])
    info["cal_max"] = float(hdr["cal_max"])
    info["orientation"] = "".join(nib.aff2axcodes(nii.affine))

    # Extended keys for grouped display
    info["ndim"] = ndim
    info["dimensions"] = tuple(int(dims[i]) for i in range(1, ndim + 1))
    info["voxel_size"] = tuple(float(pixdims[i]) for i in range(1, ndim + 1))
    spatial_units = hdr.get_xyzt_units()[0]
    info["vox_units"] = spatial_units if spatial_units != "unknown" else "mm"
    fname = nii.get_filename()
    info["file_size"] = Path(fname).stat().st_size if fname else None

    return info


def _bold(text: str, styled: bool) -> str:
    """Wrap *text* in ANSI bold if *styled* is True."""
    if styled:
        return f"\033[1m{text}\033[0m"
    return text


def format_summary(info: Dict[str, Any], styled: bool = True) -> str:
    """Pretty-print the dict returned by :func:`header_summary`.

    Parameters
    ----------
    info : dict
        As returned by :func:`header_summary`.
    styled : bool
        When True, section headers are rendered in ANSI bold.
        Set to False for piped / ``NO_COLOR`` output.
    """
    filename = info.get("filename", "<in-memory>")
    basename = Path(filename).name if filename != "<in-memory>" else filename

    dims = info.get("dimensions", ())
    vox = info.get("voxel_size", ())
    units = info.get("vox_units", "mm")

    dim_str = " x ".join(str(d) for d in dims) if dims else "?"
    vox_str = (
        " x ".join(f"{v:.2f}" for v in vox) + f" {units}" if vox else "?"
    )

    sections: List[Tuple[str, List[Tuple[str, str]]]] = [
        ("File", [
            ("path", basename),
            ("data type", info.get("data_type", "?")),
        ]),
        ("Dimensions", [
            ("size", dim_str),
            ("voxel size", vox_str),
            ("orientation", info.get("orientation", "?")),
        ]),
        ("Intensity", [
            ("cal_min", str(info.get("cal_min", "?"))),
            ("cal_max", str(info.get("cal_max", "?"))),
        ]),
    ]

    lines: List[str] = []
    for i, (header, rows) in enumerate(sections):
        if i > 0:
            lines.append("")
        lines.append(_bold(header, styled))
        for label, value in rows:
            lines.append(f"  {label:<14s}  {value}")

    return "\n".join(lines)


def format_summary_short(info: Dict[str, Any]) -> str:
    """One-liner ``nib-ls``-style summary.

    Example output::

        brain.nii.gz  float32  [91, 109, 91]  2.00x2.00x2.00mm  RAS
    """
    filename = info.get("filename", "<in-memory>")
    basename = Path(filename).name if filename != "<in-memory>" else filename
    dtype = info.get("data_type", "?")
    dims = info.get("dimensions", ())
    vox = info.get("voxel_size", ())
    units = info.get("vox_units", "mm")
    ori = info.get("orientation", "?")

    dim_str = "[" + ", ".join(str(d) for d in dims) + "]" if dims else "?"
    vox_str = "x".join(f"{v:.2f}" for v in vox) + units if vox else "?"

    return f"{basename}  {dtype}  {dim_str}  {vox_str}  {ori}"


# ---------------------------------------------------------------------------
# header_dump  (fslhd equivalent)
# ---------------------------------------------------------------------------

def header_dump(img: NiftiLike) -> Dict[str, Any]:
    """Return *all* NIfTI header fields as a dictionary (like ``fslhd``).

    Parameters
    ----------
    img : NiftiLike

    Returns
    -------
    dict
        Every field stored in the NIfTI-1 header.
    """
    nii = load_nifti(img)
    hdr = nii.header
    dump: Dict[str, Any] = {}
    for field in hdr.keys():
        val = hdr[field]
        # Convert numpy scalars / arrays to native Python types for display
        if isinstance(val, np.ndarray):
            if val.ndim == 0:
                val = val.item()
            else:
                val = val.tolist()
        elif isinstance(val, (np.integer, np.floating)):
            val = val.item()
        elif isinstance(val, bytes):
            val = val.decode("latin-1").rstrip("\x00")
        dump[field] = val
    return dump


def format_dump(dump: Dict[str, Any]) -> str:
    """Pretty-print the dict returned by :func:`header_dump`.

    Uses a tab-aligned fixed-width layout matching fslhd convention.
    """
    lines = []
    for k, v in dump.items():
        lines.append(f"{k:<20s}\t{v}")
    return "\n".join(lines)
