"""Constants for the lesion_features pipeline."""

TARGET_SPACE = "MNI152NLin6Asym"
TARGET_RES = 1
DEFAULT_BCBTOOLKIT = "/opt/BCBToolkit"
LF_SUBDIR = "lesion"

EBRAINS_ATLAS_SPECS = [
    "aal",
    "buckner_7n",
    "tian_s1",
    "tian_s2",
    "rojkova",
    "yeh_hcp1065",
    "schaefer_200_7n",
    "schaefer_300_7n",
    "schaefer_400_7n",
    "schaefer_200_tian_s1",
    "schaefer_300_tian_s1",
    "schaefer_400_tian_s1",
    "schaefer_200_tian_s2",
    "schaefer_300_tian_s2",
    "schaefer_400_tian_s2",
]

# FSL-bundled atlases — only available when $FSLDIR is set
EBRAINS_FSL_ATLAS_SPECS = [
    "jhu_wm_prob",
    "jhu_wm_labels",
]
