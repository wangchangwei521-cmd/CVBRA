from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from buse_uav.data.coco import CocoDocument, load_coco
from buse_uav.data.common import (
    DataError,
    image_size,
    object_size_bucket,
    validation_result,
)
from buse_uav.data.manifest import build_file_manifest
from buse_uav.schemas import ImageRecord

HAZYDET_CLASS_NAMES = ("car", "truck", "bus")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class HazyDetAdapter:
    def __init__(self, root: Path, *, split: str, image_variant: str = "hazy") -> None:
        self.root = root
        self.split = split
        self.image_variant = image_variant

    @property
    def class_names(self) -> Sequence[str]:
        return HAZYDET_CLASS_NAMES

    def split_dir(self) -> Path:
        candidate = self.root / self.split
        if candidate.is_dir():
            return candidate
        matched = _casefold_child(self.root, self.split)
        if matched is not None and matched.is_dir():
            return matched
        raise DataError(
            f"HazyDet split directory is unavailable: {candidate}. "
            "Set dataset.root to the downloaded HazyDet root."
        )

    def image_dir(self, variant: str | None = None) -> Path:
        actual_variant = variant or self.image_variant
        split_dir = self.split_dir()
        names = [
            f"{actual_variant}_images",
            f"{actual_variant} images",
            f"{actual_variant}-images",
        ]
        if (actual_variant.casefold() == "clean" and self.split.casefold() != "rddts") or (
            actual_variant.casefold() == "hazy" and self.split.casefold() == "rddts"
        ):
            names.append("images")
        for name in names:
            direct = split_dir / name
            if direct.is_dir():
                return direct
            matched = _casefold_child(split_dir, name)
            if matched is not None and matched.is_dir():
                return matched
        raise DataError(
            f"HazyDet {actual_variant} image directory was not found under {split_dir}; "
            f"checked: {', '.join(names)}"
        )

    def annotation_file(self) -> Path:
        split_dir = self.split_dir()
        preferred = split_dir / f"{self.split}_coco.json"
        if preferred.is_file():
            return preferred
        json_files = sorted(split_dir.glob("*.json"))
        if len(json_files) == 1:
            return json_files[0]
        if not json_files:
            raise DataError(f"no COCO annotation JSON found under {split_dir}")
        raise DataError(
            f"multiple annotation JSON files found under {split_dir}; "
            "pass dataset.annotations explicitly rather than guessing"
        )

    def document(self) -> CocoDocument:
        return load_coco(self.annotation_file())

    def image_path(self, file_name: str) -> Path:
        candidate = self.image_dir() / Path(file_name).name
        if candidate.is_file():
            return candidate
        nested = self.image_dir() / file_name
        if nested.is_file() and nested.resolve().is_relative_to(self.image_dir().resolve()):
            return nested
        return candidate

    def records(self) -> Sequence[ImageRecord]:
        document = self.document()
        records: list[ImageRecord] = []
        for image in document.images:
            path = self.image_path(str(image["file_name"]))
            width, height = image_size(path)
            records.append(
                ImageRecord(
                    image_id=image["id"],
                    path=str(path),
                    width=width,
                    height=height,
                )
            )
        return tuple(records)

    def validate(self) -> dict[str, Any]:
        errors: list[str] = []
        warnings: list[str] = []
        try:
            document = self.document()
            image_dir = self.image_dir()
        except DataError as exc:
            return validation_result(
                dataset="hazydet",
                split=self.split,
                errors=[str(exc)],
                warnings=[],
                stats={},
            )

        category_names = document.category_names_by_id
        if set(category_names.values()) != set(HAZYDET_CLASS_NAMES):
            errors.append(
                "HazyDet categories must be exactly car/truck/bus; "
                f"found {sorted(category_names.values())}"
            )

        image_ids = [image.get("id") for image in document.images]
        if len(image_ids) != len(set(image_ids)):
            errors.append("COCO image IDs are not unique")

        actual_sizes: dict[int | str, tuple[int, int]] = {}
        missing_images = 0
        corrected_swapped_image_dimensions = 0
        for image in document.images:
            image_id = image.get("id")
            file_name = image.get("file_name")
            if image_id is None or not isinstance(file_name, str):
                errors.append(f"image entry is missing id/file_name: {image}")
                continue
            path = self.image_path(file_name)
            if not path.is_file():
                missing_images += 1
                errors.append(f"image file does not exist: {path}")
                continue
            try:
                width, height = image_size(path)
            except DataError as exc:
                errors.append(str(exc))
                continue
            actual_sizes[image_id] = (width, height)
            declared = (int(image.get("width", -1)), int(image.get("height", -1)))
            if declared != (width, height):
                if self.split.casefold() == "rddts" and declared == (height, width):
                    corrected_swapped_image_dimensions += 1
                    warnings.append(
                        f"image {image_id} uses swapped JSON dimensions "
                        f"{declared}; actual={(width, height)}"
                    )
                else:
                    errors.append(
                        f"image {image_id} size mismatch: JSON={declared}, actual={(width, height)}"
                    )

        class_counts: Counter[str] = Counter()
        size_counts: Counter[str] = Counter()
        image_by_id = document.images_by_id
        annotation_ids: list[int | str] = []
        instance_count = 0
        dropped_non_positive_bboxes = 0
        clipped_out_of_bounds_bboxes = 0
        dropped_fully_outside_bboxes = 0
        dropped_duplicate_bboxes = 0
        seen_boxes: dict[
            tuple[int | str, int, float, float, float, float],
            int | str | None,
        ] = {}
        for annotation in document.annotations:
            annotation_id = annotation.get("id")
            if annotation_id is not None:
                annotation_ids.append(annotation_id)
            image_id = annotation.get("image_id")
            if not isinstance(image_id, (int, str)):
                errors.append(f"annotation {annotation_id} has invalid image_id {image_id}")
                continue
            image_record = image_by_id.get(image_id)
            if image_record is None:
                errors.append(f"annotation {annotation_id} references unknown image {image_id}")
                continue
            category_id = annotation.get("category_id")
            if category_id not in category_names:
                errors.append(
                    f"annotation {annotation_id} references unknown category {category_id}"
                )
                continue
            bbox_raw = annotation.get("bbox")
            if not isinstance(bbox_raw, list) or len(bbox_raw) != 4:
                errors.append(f"annotation {annotation_id} has invalid bbox: {bbox_raw}")
                continue
            x, y, box_width, box_height = (float(value) for value in bbox_raw)
            bbox = (x, y, box_width, box_height)
            if not all(math.isfinite(value) for value in bbox):
                errors.append(f"annotation {annotation_id} has non-finite bbox")
                continue
            if x < 0 or y < 0:
                errors.append(f"annotation {annotation_id} has negative bbox origin")
                continue
            if box_width <= 0 or box_height <= 0:
                dropped_non_positive_bboxes += 1
                warnings.append(f"dropped annotation {annotation_id} with non-positive bbox")
                continue
            actual_size = actual_sizes.get(image_id)
            if actual_size is None:
                continue
            image_width, image_height = (float(value) for value in actual_size)
            if x + box_width > image_width + 1e-6 or y + box_height > image_height + 1e-6:
                if self.split.casefold() != "rddts":
                    errors.append(f"annotation {annotation_id} bbox exceeds image bounds")
                elif x >= image_width or y >= image_height:
                    dropped_fully_outside_bboxes += 1
                    warnings.append(
                        f"dropped annotation {annotation_id} fully outside image bounds"
                    )
                    continue
                else:
                    clipped_width = min(x + box_width, image_width) - x
                    clipped_height = min(y + box_height, image_height) - y
                    warnings.append(f"clipped annotation {annotation_id} to image bounds")
                    box_width, box_height = clipped_width, clipped_height
                    clipped_out_of_bounds_bboxes += 1
            area = annotation.get("area", box_width * box_height)
            if not isinstance(area, (int, float)) or float(area) <= 0:
                errors.append(f"annotation {annotation_id} has non-positive area")
            normalized_key = (
                image_id,
                int(category_id),
                x,
                y,
                box_width,
                box_height,
            )
            if normalized_key in seen_boxes:
                dropped_duplicate_bboxes += 1
                warnings.append(
                    f"dropped duplicate annotation {annotation_id}; "
                    f"matches {seen_boxes[normalized_key]}"
                )
                continue
            seen_boxes[normalized_key] = annotation_id
            class_counts[category_names[int(category_id)]] += 1
            size_counts[object_size_bucket(box_width, box_height)] += 1
            instance_count += 1

        if annotation_ids and len(annotation_ids) != len(set(annotation_ids)):
            errors.append("COCO annotation IDs are not unique")

        pairing = self._pairing_stats(warnings)
        return validation_result(
            dataset="hazydet",
            split=self.split,
            errors=errors,
            warnings=warnings,
            stats={
                "images": len(document.images),
                "instances": instance_count,
                "classes": dict(sorted(class_counts.items())),
                "object_sizes": {
                    key: size_counts.get(key, 0) for key in ("small", "medium", "large")
                },
                "missing_images": missing_images,
                "corrected_swapped_image_dimensions": (corrected_swapped_image_dimensions),
                "dropped_non_positive_bboxes": dropped_non_positive_bboxes,
                "clipped_out_of_bounds_bboxes": clipped_out_of_bounds_bboxes,
                "dropped_fully_outside_bboxes": dropped_fully_outside_bboxes,
                "dropped_duplicate_bboxes": dropped_duplicate_bboxes,
                "image_directory": str(image_dir),
                "annotation_file": str(self.annotation_file()),
                "clean_hazy_pairing": pairing,
            },
        )

    def manifest(self) -> dict[str, Any]:
        document = self.document()
        image_paths = [
            self.image_path(str(image["file_name"]))
            for image in document.images
            if self.image_path(str(image["file_name"])).is_file()
        ]
        paths = [self.annotation_file(), *image_paths]
        return {
            "dataset": "hazydet",
            "split": self.split,
            "image_variant": self.image_variant,
            "root": str(self.root.resolve()),
            "files": build_file_manifest(paths, root=self.root),
        }

    def _pairing_stats(self, warnings: list[str]) -> dict[str, Any]:
        try:
            clean = {
                path.stem
                for path in self.image_dir("clean").iterdir()
                if path.suffix.casefold() in IMAGE_SUFFIXES
            }
            hazy = {
                path.stem
                for path in self.image_dir("hazy").iterdir()
                if path.suffix.casefold() in IMAGE_SUFFIXES
            }
        except DataError:
            warnings.append("clean/hazy pairing not available for this split")
            return {"applicable": False}
        union = clean | hazy
        return {
            "applicable": True,
            "clean": len(clean),
            "hazy": len(hazy),
            "paired": len(clean & hazy),
            "coverage": len(clean & hazy) / len(union) if union else 1.0,
            "clean_only": len(clean - hazy),
            "hazy_only": len(hazy - clean),
        }


def _casefold_child(parent: Path, expected_name: str) -> Path | None:
    if not parent.is_dir():
        return None
    expected = expected_name.casefold()
    for child in parent.iterdir():
        if child.name.casefold() == expected:
            return child
    return None
