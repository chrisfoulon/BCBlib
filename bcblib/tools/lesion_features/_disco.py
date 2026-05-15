"""BCBToolKit run_disco.sh wrapper."""

import os
import re
import subprocess
import warnings
from pathlib import Path
from typing import Dict, Optional

from bcblib.tools.lesion_features._constants import DEFAULT_BCBTOOLKIT


def find_bcbtoolkit(path_hint: Optional[str] = None) -> Path:
    """Locate the BCBToolKit directory containing run_disco.sh.

    Search order: *path_hint* → ``BCBTOOLKIT_PATH`` env var → DEFAULT_BCBTOOLKIT.

    Raises
    ------
    FileNotFoundError
        If ``run_disco.sh`` is not found.
    """
    candidates = [
        path_hint,
        os.environ.get("BCBTOOLKIT_PATH"),
        DEFAULT_BCBTOOLKIT,
    ]
    for c in candidates:
        if c is None:
            continue
        p = Path(c)
        script = p / "run_disco.sh"
        if script.exists():
            return p
    raise FileNotFoundError(
        "BCBToolKit run_disco.sh not found. "
        "Set BCBTOOLKIT_PATH or pass --bcbtoolkit to the CLI."
    )


def predict_disco_output(input_path, disco_dir) -> Path:
    """Return the expected disconnectome output path for a given lesion input.

    Applies the rename ``_label-lesion_mask`` → ``_desc-disconnectome``.
    """
    stem = Path(input_path).name
    # strip .nii or .nii.gz
    if stem.endswith(".nii.gz"):
        stem = stem[:-7]
    elif stem.endswith(".nii"):
        stem = stem[:-4]
    stem = stem.replace("_label-lesion_mask", "_desc-disconnectome")
    return Path(disco_dir) / (stem + ".nii.gz")


def run_disco_batch(
    lesion_dir,
    disco_dir,
    bcbtoolkit: Path,
    ncores: Optional[int] = None,
    tracks_dir: Optional[str] = None,
    tmpdir: Optional[str] = None,
) -> Dict[str, Path]:
    """Run run_disco.sh in folder mode on a directory of lesion NIfTIs.

    Parameters
    ----------
    lesion_dir : str or Path
    disco_dir : str or Path
    bcbtoolkit : Path
        Directory containing run_disco.sh.
    ncores : int or None
        Number of parallel cores.  If None, BCBToolKit uses its default.
    tracks_dir : str or None
        Path to the tractography atlas directory (-T flag).  Required when the
        tracts are not in BCBToolKit's default location.
    tmpdir : str or None
        Directory for intermediate per-subject working files (-w flag).
        Defaults to ``$TMPDIR/bcb_disco_<PID>`` (or ``/tmp`` if ``$TMPDIR``
        is unset).  Set this on systems where ``/tmp`` is restricted or too
        small (e.g. some HPC/JupyterHub environments).

    Returns
    -------
    dict[str, Path]
        sub_id → expected disconnectome path (not verified to exist yet).

    Raises
    ------
    RuntimeError
        If run_disco.sh exits non-zero.
    """
    lesion_dir = Path(lesion_dir)
    disco_dir = Path(disco_dir)
    disco_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "bash", str(bcbtoolkit / "run_disco.sh"),
        "-l", str(lesion_dir),
        "-o", str(disco_dir),
        "-r", "_label-lesion_mask:_desc-disconnectome",
    ]
    if tracks_dir is not None:
        cmd += ["-T", str(tracks_dir)]
    if ncores is not None:
        cmd += ["-n", str(ncores)]
    if tmpdir is not None:
        cmd += ["-w", str(tmpdir)]

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"run_disco.sh failed with exit code {result.returncode}."
        )

    outputs: Dict[str, Path] = {}
    for f in sorted(lesion_dir.glob("*_label-lesion_mask.nii.gz")):
        m = re.search(r'sub-([^_]+)', f.name)
        if m:
            sub_id = m.group(1)
            outputs[sub_id] = predict_disco_output(f, disco_dir)
    return outputs


def collect_disco_outputs(
    disco_dir, expected: Dict[str, Path]
) -> Dict[str, Path]:
    """Validate expected disconnectome files and return only those present.

    Missing files produce a warning; no exception is raised.
    """
    valid: Dict[str, Path] = {}
    for sub_id, path in expected.items():
        if path.exists():
            valid[sub_id] = path
        else:
            warnings.warn(
                f"Disconnectome not found for sub-{sub_id}: {path}",
                RuntimeWarning,
                stacklevel=2,
            )
    return valid
