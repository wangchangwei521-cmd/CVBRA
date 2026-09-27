from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np

from buse_uav.schemas import Region


def make_grid(
    image_shape: Sequence[int],
    *,
    rows: int,
    cols: int,
    context_padding: float = 0.0,
) -> tuple[Region, ...]:
    """Partition an image into non-overlapping cores with clipped context crops."""
    if len(image_shape) < 2:
        raise ValueError("image_shape must contain height and width")
    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("image height and width must be positive")
    if rows <= 0 or cols <= 0:
        raise ValueError("grid rows and columns must be positive")
    if height < rows or width < cols:
        raise ValueError("grid cannot contain zero-area core regions")
    if not 0.0 <= context_padding <= 0.5:
        raise ValueError("context_padding must be in [0, 0.5]")

    x_edges = (*((index * width) // cols for index in range(cols)), width)
    y_edges = (*((index * height) // rows for index in range(rows)), height)
    regions: list[Region] = []
    image_area = float(width * height)
    for row in range(rows):
        for col in range(cols):
            x1, x2 = x_edges[col], x_edges[col + 1]
            y1, y2 = y_edges[row], y_edges[row + 1]
            core_width, core_height = x2 - x1, y2 - y1
            pad_x = round(core_width * context_padding)
            pad_y = round(core_height * context_padding)
            crop = (
                max(0, x1 - pad_x),
                max(0, y1 - pad_y),
                min(width, x2 + pad_x),
                min(height, y2 + pad_y),
            )
            regions.append(
                Region(
                    id=row * cols + col,
                    core_xyxy=(x1, y1, x2, y2),
                    crop_xyxy=crop,
                    area_ratio=(core_width * core_height) / image_area,
                )
            )
    return tuple(regions)


def crop_region(image: np.ndarray, region: Region) -> np.ndarray:
    """Return an owned context crop for one region."""
    if image.ndim not in {2, 3}:
        raise ValueError("image must be HW or HWC")
    x1, y1, x2, y2 = region.crop_xyxy
    height, width = image.shape[:2]
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("region crop is outside the image")
    return cast(np.ndarray, image[y1:y2, x1:x2].copy())
