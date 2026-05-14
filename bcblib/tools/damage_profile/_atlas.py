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
    Header lines whose first field is not an integer are skipped.

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
    plain_idx = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "\t" in line:
            parts = line.split("\t")
            try:
                labels[int(parts[0])] = parts[1].strip()
            except (ValueError, IndexError):
                pass  # header or malformed line
        else:
            plain_idx += 1
            labels[plain_idx] = line
    return labels


def _parse_csv_labels(text: str) -> Dict[int, str]:
    """Parse a CSV label file with format ``index,name,...`` (neuroparc style).

    Lines where the index column is not a positive integer are skipped
    (handles null/header rows).

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
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 2:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        name = parts[1].strip()
        if idx > 0 and name and name.lower() != "null":
            labels[idx] = name
    return labels


def _parse_alternating_labels(text: str) -> Dict[int, str]:
    """Parse an alternating name/index-colour label file (Tian atlas style).

    Format (repeating pairs)::

        RegionName
        index R G B A

    The index on the second line determines the key.

    Parameters
    ----------
    text : str
        Raw file content.

    Returns
    -------
    dict[int, str]
        Integer index → region name.
    """
    import re
    labels: Dict[int, str] = {}
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    i = 0
    while i < len(lines) - 1:
        name_line = lines[i]
        idx_line = lines[i + 1]
        m = re.match(r'^(\d+)\s', idx_line)
        if m:
            labels[int(m.group(1))] = name_line
            i += 2
        else:
            i += 1
    return labels


def _parse_label_file(label_file) -> Dict[int, str]:
    """Parse a label file into index→name dict.

    Supported formats:

    - FSL XML: ``<label index="N" ...>Name</label>`` elements.
    - TSV: tab-separated; first column integer index, second column name.
    - CSV: comma-separated ``index,name,...`` (neuroparc style).
    - Alternating: name line followed by ``index R G B A`` line (Tian style).
    - Plain text: one name per line, 1-indexed, blank lines skipped.
    """
    text = Path(label_file).read_text().strip()
    if text.lstrip().startswith("<"):
        return _parse_fsl_xml_labels(text)
    # CSV: first non-empty data line starts with a digit followed by a comma
    data_lines = [l.strip() for l in text.splitlines() if l.strip()]
    if data_lines and "," in data_lines[0] and data_lines[0].split(",")[0].lstrip("-").isdigit():
        return _parse_csv_labels(text)
    # Alternating (Tian style): second non-empty line is "index R G B A" — five
    # whitespace-separated integers.  A simple "\d+ \d+" would also match TSV
    # lines whose region name starts with a digit, so require 5 numeric fields.
    import re
    if len(data_lines) >= 2 and re.match(r'^\d+(\s+\d+){4}\s*$', data_lines[1]):
        return _parse_alternating_labels(text)
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


# Sentinel keys used by the compact label-atlas representation returned by
# _load_label_nifti.  Callers that need to dispatch on atlas type check for
# the presence of LABEL_DATA_KEY in the dict.
LABEL_DATA_KEY = "__label_array__"
LABEL_NAMES_KEY = "__label_names__"


def _load_label_nifti(spec: AtlasSpec) -> Dict:
    """Load an integer-label NIfTI in compact form (single array + name map).

    Instead of expanding into N binary masks (which is O(N × voxels) memory),
    the raw int32 label array and the index→name mapping are stored under
    sentinel keys.  Callers detect this via ``LABEL_DATA_KEY``.

    Parameters
    ----------
    spec : AtlasSpec

    Returns
    -------
    dict
        ``{LABEL_DATA_KEY: int32 ndarray, LABEL_NAMES_KEY: {idx: name}}``.
    """
    img = nib.load(str(spec.source))
    data = img.get_fdata().astype(np.int32)

    labels: Dict[int, str] = {}
    if spec.label_file is not None:
        labels = _parse_label_file(spec.label_file)

    unique_vals = sorted(int(v) for v in np.unique(data).tolist() if v != 0)
    label_names = {idx: labels.get(idx, f"region_{idx:04d}") for idx in unique_vals}

    return {LABEL_DATA_KEY: data, LABEL_NAMES_KEY: label_names}


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
