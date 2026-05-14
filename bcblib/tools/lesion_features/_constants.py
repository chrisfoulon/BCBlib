"""Constants for the lesion_features pipeline."""

TARGET_SPACE = "MNI152NLin6Asym"
TARGET_RES = 1
DEFAULT_BCBTOOLKIT = "/home/chrisfoulon/neuro_apps/BCBToolKitLINUX/BCBToolKit"

EBRAINS_ATLAS_SPECS = [
    "aal",
    "schaefer_200_7n",
    "schaefer_400_7n",
    "schaefer_200_tian_s1",
    "schaefer_400_tian_s1",
]

# FSL-bundled atlases — only available when $FSLDIR is set
EBRAINS_FSL_ATLAS_SPECS = [
    "jhu_wm_prob",
    "jhu_wm_labels",
]
