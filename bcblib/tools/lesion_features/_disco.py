"""Disconnectome computation: BCBToolKit run_disco.sh and disconnectome2 CLI wrappers."""

import importlib.util
import os
import re
import subprocess
import sys
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

    For plain lesions: ``_label-lesion_mask`` → ``_desc-disconnectome``.
    For desc-labelled lesions (e.g. glioma): ``_desc-core_label-lesion_mask``
    → ``_desc-core-disconnectome`` (merged into one desc entity).
    """
    stem = Path(input_path).name
    if stem.endswith(".nii.gz"):
        stem = stem[:-7]
    elif stem.endswith(".nii"):
        stem = stem[:-4]
    m = re.search(r'_desc-([^_]+)_label-lesion_mask$', stem)
    if m:
        stem = re.sub(
            r'_desc-[^_]+_label-lesion_mask$',
            f'_desc-{m.group(1)}-disconnectome',
            stem,
        )
    else:
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

    # Derive a default tracks directory from the toolkit path when not explicit.
    if tracks_dir is None:
        default_tracks = bcbtoolkit / "Tools" / "extraFiles" / "tracks_1mm"
        if default_tracks.is_dir():
            tracks_dir = str(default_tracks)

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

    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
    returncode = proc.wait()
    if returncode != 0:
        raise RuntimeError(
            f"run_disco.sh failed with exit code {returncode}."
        )

    outputs: Dict[str, Path] = {}
    for f in sorted(lesion_dir.glob("*_label-lesion_mask.nii.gz")):
        m = re.search(r'sub-([^_]+)', f.name)
        if m:
            sub_id = m.group(1)
            outputs[sub_id] = predict_disco_output(f, disco_dir)
    return outputs


def disco2_ready(index_path_hint: Optional[str] = None) -> Optional[Path]:
    """Check whether disconnectome2 is installed and its index dir is resolvable.

    Both conditions must hold: the ``disconnectome2`` package must be
    importable, and an index directory must resolve via
    *index_path_hint* → ``DISCO2_INDEX_PATH`` env var (no default — there is
    no sane global default for a per-site fetched atlas).

    Returns
    -------
    Path or None
        The resolved index directory if disco2 is ready to run, else None.
        Does not warn or raise; callers decide how to react to None.
    """
    if importlib.util.find_spec("disconnectome2") is None:
        return None
    index_path = index_path_hint or os.environ.get("DISCO2_INDEX_PATH")
    if index_path is None:
        return None
    p = Path(index_path)
    return p if p.is_dir() else None


def require_disco2(index_path_hint: Optional[str] = None) -> Path:
    """Resolve the disco2 index dir or raise with an actionable message.

    Same resolution logic as :func:`disco2_ready`, but raises instead of
    returning None. Intended for callers that have explicitly forced the
    disco2 engine and want a clear, specific failure rather than a silent
    fallback.

    Raises
    ------
    ModuleNotFoundError
        If the ``disconnectome2`` package is not installed.
    FileNotFoundError
        If the package is installed but no usable index directory is found.
    """
    if importlib.util.find_spec("disconnectome2") is None:
        raise ModuleNotFoundError(
            "disconnectome2 is not installed but --engine disco2 was "
            "requested. Install disconnectome2 or use --engine auto/bcbtoolkit."
        )
    index_path = index_path_hint or os.environ.get("DISCO2_INDEX_PATH")
    if index_path is None:
        raise FileNotFoundError(
            "disconnectome2 index directory not set. Set DISCO2_INDEX_PATH "
            "or pass --disco2-index to the CLI."
        )
    p = Path(index_path)
    if not p.is_dir():
        raise FileNotFoundError(f"disconnectome2 index directory not found: {p}")
    return p


def run_disco2_batch(
    lesion_dir,
    disco_dir,
    index_dir: Path,
    n_jobs: Optional[int] = None,
    skip_existing: bool = True,
    fiber_class: Optional[str] = None,
    out_voxel_size: Optional[float] = None,
    len_min: Optional[float] = None,
    len_max: Optional[float] = None,
) -> Dict[str, Path]:
    """Run `disco2 batch` on a directory of lesion NIfTIs.

    Drop-in disco2-backed sibling of :func:`run_disco_batch`: same
    flat-directory input layout (``*_label-lesion_mask.nii.gz``) and the
    same ``_desc-disconnectome`` output naming — :func:`predict_disco_output`
    applies unchanged (confirmed identical to disco2's own naming logic).

    Parameters
    ----------
    lesion_dir : str or Path
    disco_dir : str or Path
    index_dir : Path
        disco2 atlas/tract index directory (see :func:`require_disco2` /
        :func:`disco2_ready`).
    n_jobs : int or None
        Parallel worker count. If None, disco2 picks its own default.
    skip_existing : bool
        Pass ``--skip-existing`` to the disco2 CLI (skip subjects whose
        output already exists).
    fiber_class : str or None
        disco2-only: restrict to a macro fiber class (e.g. "association").
        No BCBToolKit equivalent; disco2's own CLI validates the value.
    out_voxel_size : float or None
        disco2-only: coarsen output voxel size (mm). No BCBToolKit
        equivalent.
    len_min, len_max : float or None
        disco2-only: runtime streamline length filter (mm), needs stored
        lengths in the index. No BCBToolKit equivalent.

    Returns
    -------
    dict[str, Path]
        sub_id → expected disconnectome path (not verified to exist yet).

    Raises
    ------
    RuntimeError
        If `disco2 batch` exits non-zero.
    """
    lesion_dir = Path(lesion_dir)
    disco_dir = Path(disco_dir)
    disco_dir.mkdir(parents=True, exist_ok=True)

    # Note positional order: lesion_dir, index_dir, out_dir.
    cmd = ["disco2", "batch", str(lesion_dir), str(index_dir), str(disco_dir)]
    if skip_existing:
        cmd.append("--skip-existing")
    if n_jobs is not None:
        cmd += ["--n-jobs", str(n_jobs)]
    if fiber_class is not None:
        cmd += ["--fiber-class", fiber_class]
    if out_voxel_size is not None:
        cmd += ["--out-voxel-size", str(out_voxel_size)]
    if len_min is not None:
        cmd += ["--len-min", str(len_min)]
    if len_max is not None:
        cmd += ["--len-max", str(len_max)]

    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
    returncode = proc.wait()
    if returncode != 0:
        raise RuntimeError(
            f"disco2 batch failed with exit code {returncode}."
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
