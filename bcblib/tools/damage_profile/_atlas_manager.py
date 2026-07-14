"""Atlas download manager: preset registry, cache, and consent flow."""

import os
import shutil
import warnings
import zipfile
import urllib.error
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
        Relative path within the cache directory to the label file, if any.
    nifti_path : str or None
        Relative path within the cache directory to the NIfTI file (for
        single-file atlases).  ``None`` for directory-format atlases.
    label_url : str or None
        Secondary download URL for the label file.  When set, the label file
        is downloaded from this URL and saved as ``label_file`` inside the
        cache directory after the main atlas file is downloaded.
    """

    full_name: str
    url: Optional[str]
    size_mb: float
    fmt: str
    space: str
    citation: str
    label_file: Optional[str] = None
    nifti_path: Optional[str] = None
    label_url: Optional[str] = None
    trk_url: Optional[str] = None
    trk_size_mb: float = 0.0


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
        nifti_path="Tracts",
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
        label_url=(
            "https://raw.githubusercontent.com/DiedrichsenLab/"
            "cerebellar_atlases/master/Buckner_2011/atl-Buckner7.tsv"
        ),
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
        nifti_path="prob",
        trk_url=(
            "https://github.com/data-others/atlas/releases/download/"
            "hcp1065/hcp1065_avg_tracts_trk.zip"
        ),
        trk_size_mb=588.0,
    ),
    "aal": AtlasInfo(
        full_name="Automated Anatomical Labeling Atlas (AAL, 116 regions)",
        url=(
            "https://raw.githubusercontent.com/neurodata/neuroparc/master/"
            "atlases/label/Human/AAL_space-MNI152NLin6_res-1x1x1.nii.gz"
        ),
        size_mb=0.3,
        fmt="label_nifti",
        space="MNI152NLin6Asym",
        citation="Tzourio-Mazoyer et al. (2002) NeuroImage 15:273-289",
        nifti_path="AAL_space-MNI152NLin6_res-1x1x1.nii.gz",
        label_file="AAL.csv",
        label_url=(
            "https://raw.githubusercontent.com/neurodata/neuroparc/master/"
            "atlases/label/Human/Anatomical-labels-csv/AAL.csv"
        ),
    ),
    "schaefer_200_7n": AtlasInfo(
        full_name="Schaefer 2018 Cortical Parcellation (200 parcels, 7 networks)",
        url=(
            "https://raw.githubusercontent.com/neurodata/neuroparc/master/"
            "atlases/label/Human/Schaefer200_space-MNI152NLin6_res-1x1x1.nii.gz"
        ),
        size_mb=0.3,
        fmt="label_nifti",
        space="MNI152NLin6Asym",
        citation="Schaefer et al. (2018) Cereb Cortex 28:3095-3114",
        nifti_path="Schaefer200_space-MNI152NLin6_res-1x1x1.nii.gz",
        label_file="Schaefer200_7N.tsv",
        label_url=(
            "https://raw.githubusercontent.com/templateflow/tpl-MNI152NLin6Asym/"
            "master/tpl-MNI152NLin6Asym_atlas-Schaefer2018_desc-200Parcels7Networks_dseg.tsv"
        ),
    ),
    "schaefer_400_7n": AtlasInfo(
        full_name="Schaefer 2018 Cortical Parcellation (400 parcels, 7 networks)",
        url=(
            "https://raw.githubusercontent.com/neurodata/neuroparc/master/"
            "atlases/label/Human/Schaefer400_space-MNI152NLin6_res-1x1x1.nii.gz"
        ),
        size_mb=0.3,
        fmt="label_nifti",
        space="MNI152NLin6Asym",
        citation="Schaefer et al. (2018) Cereb Cortex 28:3095-3114",
        nifti_path="Schaefer400_space-MNI152NLin6_res-1x1x1.nii.gz",
        label_file="Schaefer400_7N.tsv",
        label_url=(
            "https://raw.githubusercontent.com/templateflow/tpl-MNI152NLin6Asym/"
            "master/tpl-MNI152NLin6Asym_atlas-Schaefer2018_desc-400Parcels7Networks_dseg.tsv"
        ),
    ),
    "schaefer_200_tian_s1": AtlasInfo(
        full_name="Schaefer 200 + Tian Subcortex S1 (216 regions)",
        url=(
            "https://raw.githubusercontent.com/yetianmed/subcortex/master/"
            "Group-Parcellation/3T/Cortex-Subcortex/MNIvolumetric/"
            "Schaefer2018_200Parcels_7Networks_order_Tian_Subcortex_S1_MNI152NLin6Asym_1mm.nii.gz"
        ),
        size_mb=0.3,
        fmt="label_nifti",
        space="MNI152NLin6Asym",
        citation=(
            "Tian et al. (2020) Science 369:eabb7547; "
            "Schaefer et al. (2018) Cereb Cortex 28:3095-3114"
        ),
        nifti_path=(
            "Schaefer2018_200Parcels_7Networks_order_Tian_Subcortex_S1"
            "_MNI152NLin6Asym_1mm.nii.gz"
        ),
        label_file="Schaefer2018_200Parcels_7Networks_order_Tian_Subcortex_S1_label.txt",
        label_url=(
            "https://raw.githubusercontent.com/yetianmed/subcortex/master/"
            "Group-Parcellation/3T/Cortex-Subcortex/"
            "Schaefer2018_200Parcels_7Networks_order_Tian_Subcortex_S1_label.txt"
        ),
    ),
    "schaefer_400_tian_s1": AtlasInfo(
        full_name="Schaefer 400 + Tian Subcortex S1 (416 regions)",
        url=(
            "https://raw.githubusercontent.com/yetianmed/subcortex/master/"
            "Group-Parcellation/3T/Cortex-Subcortex/MNIvolumetric/"
            "Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S1_MNI152NLin6Asym_1mm.nii.gz"
        ),
        size_mb=0.4,
        fmt="label_nifti",
        space="MNI152NLin6Asym",
        citation=(
            "Tian et al. (2020) Science 369:eabb7547; "
            "Schaefer et al. (2018) Cereb Cortex 28:3095-3114"
        ),
        nifti_path=(
            "Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S1"
            "_MNI152NLin6Asym_1mm.nii.gz"
        ),
        label_file="Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S1_label.txt",
        label_url=(
            "https://raw.githubusercontent.com/yetianmed/subcortex/master/"
            "Group-Parcellation/3T/Cortex-Subcortex/"
            "Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S1_label.txt"
        ),
    ),
    "schaefer_300_7n": AtlasInfo(
        full_name="Schaefer 2018 Cortical Parcellation (300 parcels, 7 networks)",
        url=(
            "https://raw.githubusercontent.com/neurodata/neuroparc/master/"
            "atlases/label/Human/Schaefer300_space-MNI152NLin6_res-1x1x1.nii.gz"
        ),
        size_mb=0.3,
        fmt="label_nifti",
        space="MNI152NLin6Asym",
        citation="Schaefer et al. (2018) Cereb Cortex 28:3095-3114",
        nifti_path="Schaefer300_space-MNI152NLin6_res-1x1x1.nii.gz",
        label_file="Schaefer300_7N.tsv",
        label_url=(
            "https://raw.githubusercontent.com/templateflow/tpl-MNI152NLin6Asym/"
            "master/tpl-MNI152NLin6Asym_atlas-Schaefer2018_desc-300Parcels7Networks_dseg.tsv"
        ),
    ),
    "schaefer_300_tian_s1": AtlasInfo(
        full_name="Schaefer 300 + Tian Subcortex S1 (316 regions)",
        url=(
            "https://raw.githubusercontent.com/yetianmed/subcortex/master/"
            "Group-Parcellation/3T/Cortex-Subcortex/MNIvolumetric/"
            "Schaefer2018_300Parcels_7Networks_order_Tian_Subcortex_S1_MNI152NLin6Asym_1mm.nii.gz"
        ),
        size_mb=0.3,
        fmt="label_nifti",
        space="MNI152NLin6Asym",
        citation=(
            "Tian et al. (2020) Science 369:eabb7547; "
            "Schaefer et al. (2018) Cereb Cortex 28:3095-3114"
        ),
        nifti_path=(
            "Schaefer2018_300Parcels_7Networks_order_Tian_Subcortex_S1"
            "_MNI152NLin6Asym_1mm.nii.gz"
        ),
        label_file="Schaefer2018_300Parcels_7Networks_order_Tian_Subcortex_S1_label.txt",
        label_url=(
            "https://raw.githubusercontent.com/yetianmed/subcortex/master/"
            "Group-Parcellation/3T/Cortex-Subcortex/"
            "Schaefer2018_300Parcels_7Networks_order_Tian_Subcortex_S1_label.txt"
        ),
    ),
    "schaefer_200_tian_s2": AtlasInfo(
        full_name="Schaefer 200 + Tian Subcortex S2 (232 regions)",
        url=(
            "https://raw.githubusercontent.com/yetianmed/subcortex/master/"
            "Group-Parcellation/3T/Cortex-Subcortex/MNIvolumetric/"
            "Schaefer2018_200Parcels_7Networks_order_Tian_Subcortex_S2_MNI152NLin6Asym_1mm.nii.gz"
        ),
        size_mb=0.3,
        fmt="label_nifti",
        space="MNI152NLin6Asym",
        citation=(
            "Tian et al. (2020) Science 369:eabb7547; "
            "Schaefer et al. (2018) Cereb Cortex 28:3095-3114"
        ),
        nifti_path=(
            "Schaefer2018_200Parcels_7Networks_order_Tian_Subcortex_S2"
            "_MNI152NLin6Asym_1mm.nii.gz"
        ),
        label_file="Schaefer2018_200Parcels_7Networks_order_Tian_Subcortex_S2_label.txt",
        label_url=(
            "https://raw.githubusercontent.com/yetianmed/subcortex/master/"
            "Group-Parcellation/3T/Cortex-Subcortex/"
            "Schaefer2018_200Parcels_7Networks_order_Tian_Subcortex_S2_label.txt"
        ),
    ),
    "schaefer_300_tian_s2": AtlasInfo(
        full_name="Schaefer 300 + Tian Subcortex S2 (332 regions)",
        url=(
            "https://raw.githubusercontent.com/yetianmed/subcortex/master/"
            "Group-Parcellation/3T/Cortex-Subcortex/MNIvolumetric/"
            "Schaefer2018_300Parcels_7Networks_order_Tian_Subcortex_S2_MNI152NLin6Asym_1mm.nii.gz"
        ),
        size_mb=0.3,
        fmt="label_nifti",
        space="MNI152NLin6Asym",
        citation=(
            "Tian et al. (2020) Science 369:eabb7547; "
            "Schaefer et al. (2018) Cereb Cortex 28:3095-3114"
        ),
        nifti_path=(
            "Schaefer2018_300Parcels_7Networks_order_Tian_Subcortex_S2"
            "_MNI152NLin6Asym_1mm.nii.gz"
        ),
        label_file="Schaefer2018_300Parcels_7Networks_order_Tian_Subcortex_S2_label.txt",
        label_url=(
            "https://raw.githubusercontent.com/yetianmed/subcortex/master/"
            "Group-Parcellation/3T/Cortex-Subcortex/"
            "Schaefer2018_300Parcels_7Networks_order_Tian_Subcortex_S2_label.txt"
        ),
    ),
    "schaefer_400_tian_s2": AtlasInfo(
        full_name="Schaefer 400 + Tian Subcortex S2 (432 regions)",
        url=(
            "https://raw.githubusercontent.com/yetianmed/subcortex/master/"
            "Group-Parcellation/3T/Cortex-Subcortex/MNIvolumetric/"
            "Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S2_MNI152NLin6Asym_1mm.nii.gz"
        ),
        size_mb=0.4,
        fmt="label_nifti",
        space="MNI152NLin6Asym",
        citation=(
            "Tian et al. (2020) Science 369:eabb7547; "
            "Schaefer et al. (2018) Cereb Cortex 28:3095-3114"
        ),
        nifti_path=(
            "Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S2"
            "_MNI152NLin6Asym_1mm.nii.gz"
        ),
        label_file="Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S2_label.txt",
        label_url=(
            "https://raw.githubusercontent.com/yetianmed/subcortex/master/"
            "Group-Parcellation/3T/Cortex-Subcortex/"
            "Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S2_label.txt"
        ),
    ),
}


def get_atlas_dir() -> Path:
    """Return the root cache directory for downloaded atlases.

    Respects the ``BCBLIB_ATLAS_DIR`` environment variable so that a
    system-wide shared cache can be used on multi-user servers.

    Returns
    -------
    Path
        ``$BCBLIB_ATLAS_DIR`` if set, otherwise ``~/.bcblib/atlases/``.
    """
    env = os.environ.get("BCBLIB_ATLAS_DIR")
    if env:
        return Path(env)
    return Path.home() / ".bcblib" / "atlases"


def list_preset_atlases() -> List[str]:
    """Return the keys of all registered preset atlases.

    Returns
    -------
    list of str
    """
    return list(PRESET_ATLASES.keys())


_MNI6_SPACE = "MNI152NLin6Asym"
_MNI6_READY_MARKER = ".mni6_ready"


def _warp_directory_to_mni6(cache_dir: Path, source_space: str) -> None:
    """Warp all NIfTI files in *cache_dir* from *source_space* to MNI152NLin6Asym.

    Files are overwritten in-place.  A ``.mni6_ready`` marker is written on
    success so that subsequent calls skip the warp.
    """
    from bcblib.tools.damage_profile._space import _apply_templateflow_warp
    import nibabel as nib

    niftis = sorted(cache_dir.glob("*.nii*"))
    if not niftis:
        return
    print(f"Warping {len(niftis)} file(s) from {source_space} → MNI152NLin6Asym …")
    for f in niftis:
        img = nib.load(str(f))
        warped = _apply_templateflow_warp(img, source_space, _MNI6_SPACE, order=1)
        nib.save(warped, str(f))
    (cache_dir / _MNI6_READY_MARKER).touch()


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

    # Download secondary label file if a separate URL is provided
    if info.label_url and info.label_file:
        try:
            urllib.request.urlretrieve(info.label_url, dest / info.label_file)
        except urllib.error.HTTPError as exc:
            bundled = Path(__file__).parent.parent.parent / "data" / info.label_file
            if bundled.exists():
                shutil.copy2(bundled, dest / info.label_file)
            else:
                warnings.warn(
                    f"Could not download label file for '{info.full_name}' "
                    f"(HTTP {exc.code}); atlas will be used without region names."
                )


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
        nifti_dir = (cache / info.nifti_path) if info.nifti_path else cache
        if (
            info.fmt == "directory"
            and info.space != _MNI6_SPACE
            and not (nifti_dir / _MNI6_READY_MARKER).exists()
        ):
            _warp_directory_to_mni6(nifti_dir, info.space)
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

    if info.fmt == "directory" and info.space != _MNI6_SPACE:
        nifti_dir = (cache / info.nifti_path) if info.nifti_path else cache
        _warp_directory_to_mni6(nifti_dir, info.space)

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


def _find_trk_dir(root: Path) -> Optional[Path]:
    """Return the first directory under *root* that directly contains .trk files."""
    if any(root.glob("*.trk")):
        return root
    for sub in sorted(root.iterdir()):
        if sub.is_dir() and any(sub.glob("*.trk")):
            return sub
    return None


def get_preset_trk_dir(name: str, assume_yes: bool = False) -> Optional[Path]:
    """Return the directory of TRK files for a preset atlas, downloading if needed.

    Parameters
    ----------
    name : str
        Key from :data:`PRESET_ATLASES` (only ``'yeh_hcp1065'`` currently has TRKs).
    assume_yes : bool
        Skip the download consent prompt.

    Returns
    -------
    Path or None
        Directory containing ``.trk`` files, or ``None`` if unavailable.
    """
    if name not in PRESET_ATLASES:
        raise KeyError(f"Unknown preset atlas: {name!r}")

    info = PRESET_ATLASES[name]
    if not info.trk_url:
        return None

    trk_root = get_atlas_dir() / name / "trk"

    if trk_root.exists():
        found = _find_trk_dir(trk_root)
        if found is not None:
            return found

    if not assume_yes:
        answer = input(
            f"\nTractography files for '{info.full_name}' (~{info.trk_size_mb:.0f} MB) "
            f"are not in the local cache.\nDownload to {trk_root}? [y/N] "
        )
        if answer.strip().lower() not in ("y", "yes"):
            warnings.warn(
                f"TRK download declined for '{name}'; streamline ratio will be skipped.",
                RuntimeWarning,
            )
            return None

    trk_root.mkdir(parents=True, exist_ok=True)
    tmp = trk_root / "_download.zip"
    print(f"Downloading TRK files for {info.full_name} ({info.trk_size_mb:.0f} MB) …")
    urllib.request.urlretrieve(info.trk_url, tmp)
    print("Extracting …")
    with zipfile.ZipFile(tmp) as zf:
        zf.extractall(trk_root)
    tmp.unlink()

    found = _find_trk_dir(trk_root)
    if found is None:
        warnings.warn(
            f"No .trk files found after extracting to {trk_root}; "
            "streamline ratio will be skipped.",
            RuntimeWarning,
        )
    return found
