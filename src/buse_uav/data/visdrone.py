from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from buse_uav.data.common import (
    DataError,
    ParsedObject,
    image_size,
    is_finite_positive_bbox,
    object_size_bucket,
    validation_result,
)
from buse_uav.data.manifest import build_file_manifest
from buse_uav.schemas import ImageRecord

VISDRONE_CLASS_NAMES = (
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


class VisDroneAdapter:
    def __init__(
        self,
        root: Path,
        *,
        split: str,
        image_directory: Path | None = None,
        annotation_directory: Path | None = None,
    ) -> None:
        self.root = root
        self.split = split
        self._image_directory = image_directory
        self._annotation_directory = annotation_directory

    @property
    def class_names(self) -> Sequence[str]:
        return VISDRONE_CLASS_NAMES

    def split_dir(self) -> Path:
        names = (
            f"VisDrone2019-DET-{self.split}",
            self.split,
        )
        for name in names:
            candidate = self.root / name
            if candidate.is_dir():
                return candidate
        raise DataError(
            f"VisDrone split `{self.split}` was not found under {self.root}; "
            f"checked: {', '.join(names)}"
        )

    def image_dir(self) -> Path:
        path = self._image_directory or self.split_dir() / "images"
        if not path.is_dir():
            raise DataError(f"VisDrone image directory does not exist: {path}")
        return path

    def annotation_dir(self) -> Path:
        path = self._annotation_directory or self.split_dir() / "annotations"
        if not path.is_dir():
            raise DataError(f"VisDrone annotation directory does not exist: {path}")
        return path

    def image_paths(self) -> tuple[Path, ...]:
        return tuple(
            sorted(
                (
                    path
                    for path in self.image_dir().iterdir()
                    if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
                ),
                key=lambda path: path.name.casefold(),
            )
        )

    def annotation_file(self, image_path: Path) -> Path:
        return self.annotation_dir() / f"{image_path.stem}.txt"

    def _objects_with_warnings(
        self,
        image_path: Path,
    ) -> tuple[tuple[ParsedObject, ...], tuple[str, ...]]:
        annotation = self.annotation_file(image_path)
        if not annotation.is_file():
            raise DataError(f"VisDrone annotation file does not exist: {annotation}")
        objects: list[ParsedObject] = []
        warnings: list[str] = []
        for line_number, line in enumerate(
            annotation.read_text(encoding="utf-8-sig").splitlines(), start=1
        ):
            stripped = line.strip().rstrip(",")
            if not stripped:
                continue
            fields = stripped.split(",")
            if len(fields) < 8:
                raise DataError(f"{annotation}:{line_number} must have 8 comma-separated fields")
            try:
                x, y, width, height = (float(value) for value in fields[:4])
                score = int(float(fields[4]))
                category = int(float(fields[5]))
                truncation = int(float(fields[6]))
                occlusion = int(float(fields[7]))
            except ValueError as exc:
                raise DataError(f"{annotation}:{line_number} contains non-numeric data") from exc
            if score == 0:
                continue
            if width <= 0 or height <= 0:
                warnings.append(
                    f"{annotation}:{line_number} dropped non-positive bbox "
                    f"({x:g},{y:g},{width:g},{height:g})"
                )
                continue
            objects.append(
                ParsedObject(
                    class_id=category - 1,
                    bbox_xywh=(x, y, width, height),
                    truncation=truncation,
                    occlusion=occlusion,
                )
            )
        return tuple(objects), tuple(warnings)

    def objects(self, image_path: Path) -> tuple[ParsedObject, ...]:
        objects, _ = self._objects_with_warnings(image_path)
        return objects

    def records(self) -> Sequence[ImageRecord]:
        records: list[ImageRecord] = []
        for image_path in self.image_paths():
            width, height = image_size(image_path)
            records.append(
                ImageRecord(
                    image_id=image_path.stem,
                    path=str(image_path),
                    width=width,
                    height=height,
                )
            )
        return tuple(records)

    def validate(self) -> dict[str, Any]:
        errors: list[str] = []
        warnings: list[str] = []
        try:
            image_paths = self.image_paths()
            annotation_dir = self.annotation_dir()
        except DataError as exc:
            return validation_result(
                dataset="visdrone",
                split=self.split,
                errors=[str(exc)],
                warnings=[],
                stats={},
            )

        class_counts: Counter[str] = Counter()
        size_counts: Counter[str] = Counter()
        instance_count = 0
        dropped_non_positive_bboxes = 0
        for image_path in image_paths:
            try:
                width, height = image_size(image_path)
                objects, object_warnings = self._objects_with_warnings(image_path)
            except DataError as exc:
                errors.append(str(exc))
                continue
            warnings.extend(object_warnings)
            dropped_non_positive_bboxes += len(object_warnings)
            for index, obj in enumerate(objects):
                if not 0 <= obj.class_id < len(VISDRONE_CLASS_NAMES):
                    errors.append(
                        f"{self.annotation_file(image_path)} object {index} "
                        f"has category {obj.class_id + 1}, expected 1-10"
                    )
                    continue
                if not is_finite_positive_bbox(obj.bbox_xywh):
                    errors.append(
                        f"{self.annotation_file(image_path)} object {index} has invalid bbox"
                    )
                    continue
                x, y, box_width, box_height = obj.bbox_xywh
                if x + box_width > width + 1e-6 or y + box_height > height + 1e-6:
                    errors.append(
                        f"{self.annotation_file(image_path)} object {index} exceeds image bounds"
                    )
                class_counts[VISDRONE_CLASS_NAMES[obj.class_id]] += 1
                size_counts[object_size_bucket(box_width, box_height)] += 1
                instance_count += 1

        image_stems = {path.stem for path in image_paths}
        annotation_stems = {path.stem for path in annotation_dir.glob("*.txt")}
        missing_annotations = sorted(image_stems - annotation_stems)
        orphan_annotations = sorted(annotation_stems - image_stems)
        if missing_annotations:
            errors.append(
                f"{len(missing_annotations)} images lack annotations; "
                f"first: {missing_annotations[:3]}"
            )
        if orphan_annotations:
            warnings.append(
                f"{len(orphan_annotations)} annotation files lack images; "
                f"first: {orphan_annotations[:3]}"
            )

        return validation_result(
            dataset="visdrone",
            split=self.split,
            errors=errors,
            warnings=warnings,
            stats={
                "images": len(image_paths),
                "instances": instance_count,
                "classes": dict(sorted(class_counts.items())),
                "object_sizes": {
                    key: size_counts.get(key, 0) for key in ("small", "medium", "large")
                },
                "missing_annotations": len(missing_annotations),
                "orphan_annotations": len(orphan_annotations),
                "dropped_non_positive_bboxes": dropped_non_positive_bboxes,
                "image_directory": str(self.image_dir()),
                "annotation_directory": str(annotation_dir),
            },
        )

    def manifest(self) -> dict[str, Any]:
        images = self.image_paths()
        annotations = [
            self.annotation_file(image_path)
            for image_path in images
            if self.annotation_file(image_path).is_file()
        ]
        return {
            "dataset": "visdrone",
            "split": self.split,
            "root": str(self.root.resolve()),
            "files": build_file_manifest([*images, *annotations], root=self.root),
        }

    def coco_document(self, *, image_ids: Sequence[int | str] | None = None) -> dict[str, Any]:
        """Build deterministic COCO ground truth for evaluation only."""
        allowed = set(image_ids) if image_ids is not None else None
        images: list[dict[str, Any]] = []
        annotations: list[dict[str, Any]] = []
        annotation_id = 1
        for image_path in self.image_paths():
            image_id = image_path.stem
            if allowed is not None and image_id not in allowed:
                continue
            width, height = image_size(image_path)
            images.append(
                {
                    "id": image_id,
                    "file_name": image_path.name,
                    "width": width,
                    "height": height,
                }
            )
            for obj in self.objects(image_path):
                x, y, box_width, box_height = obj.bbox_xywh
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": obj.class_id + 1,
                        "bbox": [x, y, box_width, box_height],
                        "area": box_width * box_height,
                        "iscrowd": 0,
                        "truncation": obj.truncation,
                        "occlusion": obj.occlusion,
                    }
                )
                annotation_id += 1
        return {
            "info": {"description": f"VisDrone2019-DET {self.split} evaluation"},
            "images": images,
            "annotations": annotations,
            "categories": [
                {"id": index + 1, "name": name} for index, name in enumerate(VISDRONE_CLASS_NAMES)
            ],
        }
