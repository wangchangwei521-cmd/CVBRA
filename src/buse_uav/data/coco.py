from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from buse_uav.data.common import DataError


@dataclass(frozen=True)
class CocoDocument:
    raw: dict[str, Any]
    images: tuple[dict[str, Any], ...]
    annotations: tuple[dict[str, Any], ...]
    categories: tuple[dict[str, Any], ...]

    @property
    def category_names_by_id(self) -> dict[int, str]:
        return {
            int(category["id"]): str(category["name"])
            for category in self.categories
            if "id" in category and "name" in category
        }

    @property
    def images_by_id(self) -> dict[int | str, dict[str, Any]]:
        return {image["id"]: image for image in self.images if "id" in image}


def load_coco(path: Path) -> CocoDocument:
    if not path.is_file():
        raise DataError(f"COCO annotation file does not exist: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DataError(f"cannot parse COCO JSON {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise DataError(f"COCO JSON root must be an object: {path}")
    for key in ("images", "annotations", "categories"):
        if not isinstance(raw.get(key), list):
            raise DataError(f"COCO JSON field `{key}` must be a list: {path}")
    return CocoDocument(
        raw=raw,
        images=tuple(raw["images"]),
        annotations=tuple(raw["annotations"]),
        categories=tuple(raw["categories"]),
    )
