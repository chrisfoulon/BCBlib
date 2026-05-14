"""damage_profile — overlap statistics between subject maps and brain atlases."""

from bcblib.tools.damage_profile._atlas import AtlasSpec, load_atlas, detect_atlas_format
from bcblib.tools.damage_profile._atlas_manager import (
    get_preset_atlas,
    list_preset_atlases,
)
from bcblib.tools.damage_profile._core import damage_profile

__all__ = [
    "damage_profile",
    "AtlasSpec",
    "load_atlas",
    "detect_atlas_format",
    "get_preset_atlas",
    "list_preset_atlases",
]
