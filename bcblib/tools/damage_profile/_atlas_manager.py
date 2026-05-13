"""Atlas download manager: preset registry, cache, and consent flow."""

import zipfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from bcblib.tools.damage_profile._atlas import AtlasSpec, load_atlas


@dataclass
class AtlasInfo:
    """Metadata for a preset atlas.

    Parameters
    ----------
    full_name : str
        Human-readable name.
    url : str or None
        Download URL.  ``None`` for FSL-bundled atlases.
    size_mb : float
        Approximate download size in MB.
    fmt : str
        Atlas format: ``'directory'``, ``'4d_nifti'``, or ``'label_nifti'``.
    space : str
        Template space identifier, e.g. ``'MNI152NLin6Asym'``.
    citation : str
        Reference string for the atlas.
    label_file : str or None
        Relative path within the extracted archive to the label file, if any.
    nifti_path : str or None
        Relative path within the extracted archive to the NIfTI file (for
        single-file atlases).  ``None`` for directory-format atlases.
    """

    full_name: str
    url: Optional[str]
    size_mb: float
    fmt: str
    space: str
    citation: str
    label_file: Optional[str] = None
    nifti_path: Optional[str] = None


PRESET_ATLASES: Dict[str, AtlasInfo] = {
    "jhu_wm_prob": AtlasInfo(
        full_name="JHU White Matter Probabilistic Tractography Atlas",
        url=None,
        size_mb=0.0,
        fmt="4d_nifti",
        space="MNI152NLin6Asym",
        citation="Hua et al. (2008) NeuroImage 40:570-582",
        nifti_path="JHU-ICBM-tracts-prob-2mm.nii.gz",
        label_file="JHU-tracts.xml",
    ),
    "jhu_wm_labels": AtlasInfo(
        full_name="JHU White Matter Labels Atlas",
        url=None,
        size_mb=0.0,
        fmt="label_nifti",
        space="MNI152NLin6Asym",
        citation="Mori et al. (2008) MRI Atlas of Human White Matter",
        nifti_path="JHU-ICBM-labels-2mm.nii.gz",
        label_file="JHU-labels.xml",
    ),
    "rojkova": AtlasInfo(
        full_name="Rojkova White Matter Tractography Atlas",
        url="https://www.dropbox.com/s/dnbt3gdm1iledkv/Atlas_Rojkova.zip?dl=1",
        size_mb=45.0,
        fmt="directory",
        space="MNI152NLin6Asym",
        citation="Rojkova et al. (2016) Brain Struct Funct 221:4006-4021",
    ),
    "tian_s1": AtlasInfo(
        full_name="Tian Subcortex Atlas Scale I (16 regions)",
        url="https://www.nitrc.org/frs/download.php/13364/Tian2020MSA_v1.4.zip",
        size_mb=18.0,
        fmt="label_nifti",
        space="MNI152NLin6Asym",
        citation="Tian et al. (2020) Science 369:eabb7547",
        nifti_path="Tian2020MSA/3T/Subcortex-Only/Tian_Subcortex_S1_3T_1mm.nii.gz",
        label_file="Tian2020MSA/3T/Subcortex-Only/Tian_Subcortex_S1_3T_1mm_label.txt",
    ),
    "tian_s2": AtlasInfo(
        full_name="Tian Subcortex Atlas Scale II (32 regions)",
        url="https://www.nitrc.org/frs/download.php/13364/Tian2020MSA_v1.4.zip",
        size_mb=18.0,
        fmt="label_nifti",
        space="MNI152NLin6Asym",
        citation="Tian et al. (2020) Science 369:eabb7547",
        nifti_path="Tian2020MSA/3T/Subcortex-Only/Tian_Subcortex_S2_3T_1mm.nii.gz",
        label_file="Tian2020MSA/3T/Subcortex-Only/Tian_Subcortex_S2_3T_1mm_label.txt",
    ),
    "buckner_7n": AtlasInfo(
        full_name="Buckner Cerebellar Atlas (7 networks)",
        url=(
            "https://raw.githubusercontent.com/DiedrichsenLab/"
            "cerebellar_atlases/master/Buckner_2011/atl-Buckner7_space-MNI_dseg.nii"
        ),
        size_mb=0.5,
        fmt="label_nifti",
        space="MNI152NLin6Asym",
        citation="Buckner et al. (2011) J Neurophysiol 106:2329-2344",
        label_file="atl-Buckner7.tsv",
    ),
    "yeh_hcp1065": AtlasInfo(
        full_name="Yeh HCP1065 White Matter Atlas (64 tracts)",
        url=(
            "https://github.com/data-others/atlas/releases/download/"
            "hcp1065/hcp1065_prob_coverage_nifti.zip"
        ),
        size_mb=17.0,
        fmt="directory",
        space="MNI152NLin2009cAsym",
        citation="Yeh et al. (2022) NeuroImage 249:118931",
    ),
}


def get_atlas_dir() -> Path:
    """Return the root cache directory for downloaded atlases.

    Returns
    -------
    Path
        ``~/.bcblib/atlases/``
    """
    return Path.home() / ".bcblib" / "atlases"


def list_preset_atlases() -> List[str]:
    """Return the keys of all registered preset atlases.

    Returns
    -------
    list of str
    """
    return list(PRESET_ATLASES.keys())


def _download_atlas(info: AtlasInfo, dest: Path) -> None:
    """Download and extract an atlas archive to *dest*.

    Parameters
    ----------
    info : AtlasInfo
    dest : Path
        Target directory (created if absent).
    """
    dest.mkdir(parents=True, exist_ok=True)

    if info.url is None:
        raise ValueError(f"Atlas '{info.full_name}' has no download URL.")

    url_path = info.url.split("?")[0]
    if url_path.endswith(".zip"):
        tmp = dest / "_download.zip"
        print(f"Downloading {info.full_name} ({info.size_mb:.0f} MB) …")
        urllib.request.urlretrieve(info.url, tmp)
        print("Extracting …")
        with zipfile.ZipFile(tmp) as zf:
            zf.extractall(dest)
        tmp.unlink()
    else:
        # Single file download
        filename = url_path.split("/")[-1]
        out = dest / filename
        print(f"Downloading {info.full_name} …")
        urllib.request.urlretrieve(info.url, out)

    # Buckner TSV label file is a separate URL
    if info.full_name.startswith("Buckner") and info.label_file:
        tsv_url = (
            "https://raw.githubusercontent.com/DiedrichsenLab/"
            "cerebellar_atlases/master/Buckner_2011/atl-Buckner7.tsv"
        )
        urllib.request.urlretrieve(tsv_url, dest / "atl-Buckner7.tsv")


def get_preset_atlas(
    name: str,
    path: Optional[str] = None,
    assume_yes: bool = False,
) -> Dict:
    """Resolve a named preset atlas to a loaded region-weight dict.

    Resolution order:
    1. ``path`` argument if provided.
    2. Local cache at ``~/.bcblib/atlases/<name>/``.
    3. For JHU: ``$FSLDIR/data/atlases/JHU/`` environment variable.
    4. Prompt user to download (or skip prompt if ``assume_yes``).

    Parameters
    ----------
    name : str
        Key from :data:`PRESET_ATLASES`.
    path : str, optional
        Explicit path to the atlas directory or file.
    assume_yes : bool
        Skip the download consent prompt.

    Returns
    -------
    dict[str, np.ndarray]

    Raises
    ------
    KeyError
        If *name* is not in :data:`PRESET_ATLASES`.
    RuntimeError
        If the atlas cannot be resolved and the user declines the download.
    """
    import os

    if name not in PRESET_ATLASES:
        available = ", ".join(PRESET_ATLASES.keys())
        raise KeyError(
            f"Unknown preset atlas: {name!r}. Available: {available}"
        )

    info = PRESET_ATLASES[name]

    # 1. Explicit path
    if path is not None:
        spec = AtlasSpec(source=path, name=name, space=info.space)
        return load_atlas(spec)

    # 2. Local cache
    cache = get_atlas_dir() / name
    if cache.exists() and any(cache.iterdir()):
        nifti_path = (cache / info.nifti_path) if info.nifti_path else cache
        _lc = (cache / info.label_file) if info.label_file else None
        label = str(_lc) if _lc and _lc.exists() else None
        spec = AtlasSpec(
            source=str(nifti_path),
            name=name,
            label_file=label,
            space=info.space,
        )
        return load_atlas(spec)

    # 3. FSL for JHU atlases
    if name.startswith("jhu"):
        fsldir = os.environ.get("FSLDIR")
        if fsldir:
            jhu_dir = Path(fsldir) / "data" / "atlases" / "JHU"
            if jhu_dir.exists():
                nifti_path = jhu_dir / info.nifti_path if info.nifti_path else jhu_dir
                _lc = jhu_dir.parent / info.label_file if info.label_file else None
                label = str(_lc) if _lc and _lc.exists() else None
                spec = AtlasSpec(
                    source=str(nifti_path),
                    name=name,
                    label_file=label,
                    space=info.space,
                )
                return load_atlas(spec)
        raise RuntimeError(
            "JHU atlas not found. Set $FSLDIR or provide the path with the "
            "`path` argument."
        )

    # 4. Download
    if info.url is None:
        raise RuntimeError(
            f"Atlas '{name}' has no download URL and was not found locally. "
            f"Provide the path with the `path` argument."
        )

    if not assume_yes:
        answer = input(
            f"\nAtlas '{info.full_name}' (~{info.size_mb:.0f} MB) is not in the "
            f"local cache.\nDownload to {cache}? [y/N] "
        )
        if answer.strip().lower() not in ("y", "yes"):
            raise RuntimeError(
                "Download declined. Provide the atlas path with the `path` argument."
            )

    _download_atlas(info, cache)

    nifti_path = (cache / info.nifti_path) if info.nifti_path else cache
    _lc2 = (cache / info.label_file) if info.label_file else None
    label = str(_lc2) if _lc2 and _lc2.exists() else None
    spec = AtlasSpec(
        source=str(nifti_path),
        name=name,
        label_file=label,
        space=info.space,
    )
    return load_atlas(spec)
