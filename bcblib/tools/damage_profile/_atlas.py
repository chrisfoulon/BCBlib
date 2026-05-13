"""Atlas loading: format detection and region weight extraction."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import nibabel as nib

from bcblib.imaging.io import is_nifti


@dataclass
class AtlasSpec:
    """Specification for a single atlas to profile against.

    Parameters
    ----------
    source : str or Path
        Path to atlas: a directory of NIfTI files, a 4D NIfTI, or a label NIfTI.
    name : str
        Short identifier used as the key in output dicts and CSV filenames.
    threshold : float
        Minimum weight value retained when loading probabilistic atlases.
        Values strictly below this are set to zero.
    label_file : str or Path, optional
        Path to a text or TSV file mapping region indices to names.
    space : str, optional
        Template space identifier, e.g. ``'MNI152NLin6Asym'``.
        Used by space handling to decide whether resampling is needed.
    """

    source: str
    name: str
    threshold: float = 0.0
    label_file: Optional[str] = None
    space: Optional[str] = None


def detect_atlas_format(source) -> str:
    """Identify the atlas format from *source*.

    Returns
    -------
    str
        One of ``'directory'``, ``'4d_nifti'``, ``'label_nifti'``.

    Raises
    ------
    ValueError
        If the source is neither a directory nor a recognised NIfTI file.
    """
    p = Path(source)
    if p.is_dir():
        return "directory"
    if is_nifti(str(p)):
        img = nib.load(str(p))
        return "4d_nifti" if img.ndim == 4 else "label_nifti"
    raise ValueError(f"Cannot determine atlas format for: {source}")


def _load_directory(spec: AtlasSpec) -> Dict[str, np.ndarray]:
    """Load a directory of NIfTI files into a name→weight dict.

    Parameters
    ----------
    spec : AtlasSpec

    Returns
    -------
    dict[str, np.ndarray]
        All-zero volumes are excluded.
    """
    files = sorted(Path(spec.source).glob("*.nii*"))
    result: Dict[str, np.ndarray] = {}
    for f in files:
        name = f.name.replace(".nii.gz", "").replace(".nii", "")
        data = nib.load(str(f)).get_fdata()
        if spec.threshold > 0:
            data = np.where(data >= spec.threshold, data, 0.0)
        if data.any():
            result[name] = data
    return result


def _parse_fsl_xml_labels(text: str) -> Dict[int, str]:
    """Parse an FSL XML atlas label file.

    Parameters
    ----------
    text : str
        Raw XML content with ``<label index="N">Name</label>`` elements.

    Returns
    -------
    dict[int, str]
        Integer index → region name.
    """
    import xml.etree.ElementTree as ET
    root = ET.fromstring(text)
    labels: Dict[int, str] = {}
    for el in root.iter("label"):
        idx_str = el.get("index")
        name = (el.text or "").strip()
        if idx_str is not None and name:
            try:
                labels[int(idx_str)] = name
            except ValueError:
                pass
    return labels


def _parse_text_labels(text: str) -> Dict[int, str]:
    """Parse a plain-text or TSV atlas label file.

    Plain text: one label per line, 1-indexed, blank lines skipped.
    TSV: first column is integer index, second column is region name.

    Parameters
    ----------
    text : str
        Raw file content.

    Returns
    -------
    dict[int, str]
        Integer index → region name.
    """
    labels: Dict[int, str] = {}
    for i, line in enumerate(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        if "\t" in line:
            parts = line.split("\t")
            try:
                labels[int(parts[0])] = parts[1].strip()
            except (ValueError, IndexError):
                pass
        else:
            labels[i + 1] = line
    return labels


def _parse_label_file(label_file) -> Dict[int, str]:
    """Parse a label file into index→name dict.

    Supported formats:
    - Plain text: one name per line, 1-indexed.
    - TSV: first column is integer index, second column is name.
    - FSL XML: ``<label index="N" ...>Name</label>`` elements.
    """
    text = Path(label_file).read_text().strip()
    if text.lstrip().startswith("<"):
        return _parse_fsl_xml_labels(text)
    return _parse_text_labels(text)


def _load_4d_nifti(spec: AtlasSpec) -> Dict[str, np.ndarray]:
    """Load a 4D NIfTI (one volume per region) into a name→weight dict.

    Parameters
    ----------
    spec : AtlasSpec

    Returns
    -------
    dict[str, np.ndarray]
        All-zero volumes are excluded; region names come from *spec.label_file*
        or are auto-numbered as ``region_XXXX``.
    """
    img = nib.load(str(spec.source))
    data = img.get_fdata()
    n_regions = data.shape[3]

    labels: Dict[int, str] = {}
    if spec.label_file is not None:
        labels = _parse_label_file(spec.label_file)

    result: Dict[str, np.ndarray] = {}
    for i in range(n_regions):
        vol = data[..., i]
        if spec.threshold > 0:
            vol = np.where(vol >= spec.threshold, vol, 0.0)
        if not vol.any():
            continue
        region_name = labels.get(i + 1, f"region_{i + 1:04d}")
        result[region_name] = vol
    return result


def _load_label_nifti(spec: AtlasSpec) -> Dict[str, np.ndarray]:
    """Load an integer-label NIfTI into a name→binary-weight dict.

    Parameters
    ----------
    spec : AtlasSpec

    Returns
    -------
    dict[str, np.ndarray]
        Each unique non-zero integer label becomes a binary float32 mask.
        Names come from *spec.label_file* or are auto-numbered as
        ``region_XXXX``.
    """
    img = nib.load(str(spec.source))
    data = img.get_fdata()

    labels: Dict[int, str] = {}
    if spec.label_file is not None:
        labels = _parse_label_file(spec.label_file)

    unique_vals = sorted(v for v in np.unique(data).tolist() if v != 0)
    result: Dict[str, np.ndarray] = {}
    for val in unique_vals:
        idx = int(val)
        mask = (data == val).astype(np.float32)
        region_name = labels.get(idx, f"region_{idx:04d}")
        result[region_name] = mask
    return result


def load_atlas(spec: AtlasSpec) -> Dict[str, np.ndarray]:
    """Load an atlas into a region-name → weight array mapping.

    Parameters
    ----------
    spec : AtlasSpec

    Returns
    -------
    dict[str, np.ndarray]
        Keys are region names; values are 3D float arrays (weights or binary masks).
    """
    fmt = detect_atlas_format(spec.source)
    if fmt == "directory":
        return _load_directory(spec)
    if fmt == "4d_nifti":
        return _load_4d_nifti(spec)
    return _load_label_nifti(spec)
