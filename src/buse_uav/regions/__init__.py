"""Region construction, cropping, and selection."""

from buse_uav.regions.coordinates import (
    LetterboxTransform,
    crop_letterbox_to_original,
    letterbox_image,
    original_to_crop_letterbox,
)
from buse_uav.regions.grid import crop_region, make_grid
from buse_uav.regions.selection import SelectionResult, combine_scores, select_regions

__all__ = [
    "LetterboxTransform",
    "SelectionResult",
    "combine_scores",
    "crop_letterbox_to_original",
    "crop_region",
    "letterbox_image",
    "make_grid",
    "original_to_crop_letterbox",
    "select_regions",
]
