"""lesion_features — BIDS-compatible lesion and disconnectome feature extraction."""

from bcblib.tools.lesion_features._bids import (
    parse_bids_entities,
    build_lf_csv_path,
    build_lf_tsv_path,
    build_prep_path,
    iter_bids_lesions,
)
from bcblib.tools.lesion_features._constants import EBRAINS_ATLAS_SPECS
from bcblib.tools.lesion_features._pipeline import (
    preprocess_batch,
    extract_features_batch,
    get_ebrains_atlas_specs,
)

__all__ = [
    "parse_bids_entities",
    "build_lf_csv_path",
    "build_lf_tsv_path",
    "build_prep_path",
    "iter_bids_lesions",
    "EBRAINS_ATLAS_SPECS",
    "preprocess_batch",
    "extract_features_batch",
    "get_ebrains_atlas_specs",
]
