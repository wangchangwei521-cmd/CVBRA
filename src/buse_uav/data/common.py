from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


class DataError(RuntimeError):
    """Dataset error safe to display to a CLI user without a traceback."""


@dataclass(frozen=True)
class ParsedObject:
    class_id: int
    bbox_xywh: tuple[float, float, float, float]
    truncation: int | None = None
    occlusion: int | None = None


def image_size(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            image.load()
            return image.size
    except (OSError, ValueError) as exc:
        raise DataError(f"cannot read image {path}: {exc}") from exc


def is_finite_positive_bbox(bbox: tuple[float, float, float, float]) -> bool:
    x, y, width, height = bbox
    return (
        all(math.isfinite(value) for value in bbox)
        and width > 0
        and height > 0
        and x >= 0
        and y >= 0
    )


def object_size_bucket(width: float, height: float) -> str:
    area = width * height
    if area < 32**2:
        return "small"
    if area < 96**2:
        return "medium"
    return "large"


def validation_result(
    *,
    dataset: str,
    split: str,
    errors: list[str],
    warnings: list[str],
    stats: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "split": split,
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "stats": stats,
    }
