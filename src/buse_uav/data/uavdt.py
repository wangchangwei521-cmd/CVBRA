from __future__ import annotations

import csv
import io
import json
import re
import shutil
import uuid
import zipfile
from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from buse_uav.data.coco import CocoDocument, load_coco
from buse_uav.data.common import DataError
from buse_uav.schemas import ImageRecord
from buse_uav.uavdt_source import uavdt_archive_evidence_sha256
from buse_uav.utils.hashing import sha256_file, stable_hash
from buse_uav.utils.io import atomic_write_json

UAVDT_CLASS_NAMES = ("car", "truck", "bus")
UAVDT_CATEGORY_MAPPING = {1: "car", 2: "truck", 3: "bus"}
UAVDT_ATTRIBUTE_NAMES = (
    "daylight",
    "night",
    "fog",
    "low_altitude",
    "medium_altitude",
    "high_altitude",
    "front_view",
    "side_view",
    "bird_view",
    "long_term",
)
_SEQUENCE_RE = re.compile(r"M\d{4}\Z")
_IMAGE_RE = re.compile(r"img(?P<frame>\d{6})\.jpg\Z", re.IGNORECASE)
_ATTRIBUTE_RE = re.compile(r"M_attr/(?P<split>train|test)/(?P<sequence>M\d{4})\s*_attr\.txt\Z")


class UavdtAdapter:
    def __init__(
        self,
        root: Path,
        *,
        split: str,
        image_directory: Path | None = None,
        annotation_file: Path | None = None,
    ) -> None:
        self.root = root
        self.split = split
        self._image_directory = image_directory
        self._annotation_file = annotation_file

    @property
    def class_names(self) -> Sequence[str]:
        return UAVDT_CLASS_NAMES

    def image_dir(self) -> Path:
        path = self._image_directory or self.root / "images"
        if not path.is_dir():
            raise DataError(f"UAVDT image directory does not exist: {path}")
        return path

    def annotation_file(self) -> Path:
        path = self._annotation_file or self.root / "annotations" / f"{self.split}.coco.json"
        if not path.is_file():
            raise DataError(f"UAVDT COCO annotation file does not exist: {path}")
        return path

    def document(self) -> CocoDocument:
        return load_coco(self.annotation_file())

    def image_path(self, file_name: str) -> Path:
        root = self.image_dir().resolve()
        candidate = (root / file_name).resolve()
        if candidate != root and root not in candidate.parents:
            raise DataError(f"UAVDT image path escapes the configured root: {file_name}")
        return candidate

    def records(self) -> Sequence[ImageRecord]:
        records: list[ImageRecord] = []
        for image in self.document().images:
            image_id = image.get("id")
            file_name = image.get("file_name")
            width = image.get("width")
            height = image.get("height")
            if (
                not isinstance(image_id, int)
                or not isinstance(file_name, str)
                or not isinstance(width, int)
                or not isinstance(height, int)
                or width <= 0
                or height <= 0
            ):
                raise DataError(f"invalid UAVDT COCO image entry: {image}")
            path = self.image_path(file_name)
            if not path.is_file():
                raise DataError(f"UAVDT image file does not exist: {path}")
            records.append(
                ImageRecord(
                    image_id=image_id,
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
        except DataError as exc:
            return {
                "dataset": "uavdt",
                "split": self.split,
                "valid": False,
                "errors": [str(exc)],
                "warnings": [],
                "stats": {},
            }
        if document.category_names_by_id != UAVDT_CATEGORY_MAPPING:
            errors.append(
                "UAVDT categories must be exactly car=1, truck=2, bus=3; "
                f"found {document.category_names_by_id}"
            )
        image_ids: set[int] = set()
        sequences: set[str] = set()
        missing_images = 0
        for image in document.images:
            image_id = image.get("id")
            file_name = image.get("file_name")
            sequence = image.get("sequence")
            if not isinstance(image_id, int) or image_id in image_ids:
                errors.append(f"invalid or duplicate UAVDT image id: {image_id}")
            else:
                image_ids.add(image_id)
            if not isinstance(sequence, str) or _SEQUENCE_RE.fullmatch(sequence) is None:
                errors.append(f"invalid UAVDT sequence field: {sequence}")
            else:
                sequences.add(sequence)
            if not isinstance(file_name, str) or not self.image_path(file_name).is_file():
                missing_images += 1
                errors.append(f"missing UAVDT image file: {file_name}")
        annotation_ids: set[int] = set()
        class_counts: Counter[str] = Counter()
        out_of_bounds = 0
        images_by_id = document.images_by_id
        for annotation in document.annotations:
            annotation_id = annotation.get("id")
            image_id = annotation.get("image_id")
            category_id = annotation.get("category_id")
            bbox = annotation.get("bbox")
            if not isinstance(annotation_id, int) or annotation_id in annotation_ids:
                errors.append(f"invalid or duplicate UAVDT annotation id: {annotation_id}")
                continue
            annotation_ids.add(annotation_id)
            image_entry = images_by_id.get(image_id) if isinstance(image_id, int) else None
            if image_entry is None:
                errors.append(f"UAVDT annotation {annotation_id} references unknown image")
                continue
            if category_id not in UAVDT_CATEGORY_MAPPING:
                errors.append(f"UAVDT annotation {annotation_id} has invalid category")
                continue
            if not isinstance(bbox, list) or len(bbox) != 4:
                errors.append(f"UAVDT annotation {annotation_id} has invalid bbox")
                continue
            x, y, width, height = (float(value) for value in bbox)
            if x < 0 or y < 0 or width <= 0 or height <= 0:
                errors.append(f"UAVDT annotation {annotation_id} has invalid bbox geometry")
                continue
            if x + width > float(image_entry["width"]) or y + height > float(image_entry["height"]):
                out_of_bounds += 1
            class_counts[UAVDT_CATEGORY_MAPPING[int(category_id)]] += 1
        if out_of_bounds:
            warnings.append(
                f"preserved {out_of_bounds} official UAVDT boxes extending outside image bounds"
            )
        return {
            "dataset": "uavdt",
            "split": self.split,
            "valid": not errors,
            "errors": errors,
            "warnings": warnings,
            "stats": {
                "images": len(document.images),
                "instances": len(document.annotations),
                "sequences": len(sequences),
                "classes": dict(sorted(class_counts.items())),
                "missing_images": missing_images,
                "preserved_out_of_bounds_boxes": out_of_bounds,
                "annotation_file": str(self.annotation_file()),
                "image_directory": str(self.image_dir()),
            },
        }

    def manifest(self) -> dict[str, Any]:
        return {
            "dataset": "uavdt",
            "split": self.split,
            "root": str(self.root.resolve()),
            "image_root": str(self.image_dir().resolve()),
            "annotation": str(self.annotation_file().resolve()),
            "annotation_sha256": sha256_file(self.annotation_file()),
            "images": len(self.document().images),
        }


def _load_gate(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DataError(f"cannot read UAVDT archive-gate report: {exc}") from exc
    if not isinstance(value, dict) or value.get("status") != "PASS":
        raise DataError("UAVDT archive-gate report is not PASS")
    gates = value.get("gates")
    required = {"download_hashes", "archive_structure", "official_split", "category_mapping"}
    if not isinstance(gates, dict) or any(gates.get(name) != "PASS" for name in required):
        raise DataError("UAVDT ordered source gates are incomplete")
    category = value.get("category_mapping")
    if not isinstance(category, dict) or category.get("mapping") != {
        "1": "car",
        "2": "truck",
        "3": "bus",
    }:
        raise DataError("UAVDT archive gate does not lock car=1, truck=2, bus=3")
    return value


def _load_extraction_marker(source_root: Path) -> dict[str, Any]:
    marker = source_root.parent / ".uavdt_extraction.json"
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DataError(f"UAVDT extraction marker is unavailable or invalid: {exc}") from exc
    if (
        not isinstance(value, dict)
        or value.get("status") != "PASS"
        or value.get("formal_metrics_calculated_or_viewed") is not False
    ):
        raise DataError("UAVDT extraction marker is not a clean PASS")
    return value


def _parse_attributes_archive(
    path: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
) -> tuple[dict[str, tuple[int, ...]], dict[str, str]]:
    if not path.is_file() or path.stat().st_size != expected_bytes:
        raise DataError("official UAVDT Attributes archive is missing or changed size")
    if sha256_file(path) != expected_sha256:
        raise DataError("official UAVDT Attributes archive hash differs from archive gate")
    values: dict[str, tuple[int, ...]] = {}
    splits: dict[str, str] = {}
    try:
        with zipfile.ZipFile(path) as bundle:
            bad = bundle.testzip()
            if bad is not None:
                raise DataError(f"official UAVDT Attributes CRC failed: {bad}")
            for entry in bundle.infolist():
                if entry.is_dir() or entry.filename == "M_attr/readme.txt":
                    continue
                match = _ATTRIBUTE_RE.fullmatch(entry.filename)
                if match is None:
                    raise DataError(
                        f"unexpected official UAVDT Attributes member: {entry.filename}"
                    )
                sequence = match.group("sequence")
                split = match.group("split")
                if sequence in values:
                    raise DataError(f"duplicate official UAVDT Attributes sequence: {sequence}")
                try:
                    rows = list(
                        csv.reader(
                            io.StringIO(bundle.read(entry).decode("ascii")),
                            strict=True,
                        )
                    )
                except (UnicodeError, csv.Error) as exc:
                    raise DataError(
                        f"cannot parse official UAVDT Attributes file {entry.filename}: {exc}"
                    ) from exc
                if len(rows) != 1 or len(rows[0]) != len(UAVDT_ATTRIBUTE_NAMES):
                    raise DataError(
                        f"official UAVDT Attributes row has wrong width: {entry.filename}"
                    )
                try:
                    parsed = tuple(int(item.strip()) for item in rows[0])
                except ValueError as exc:
                    raise DataError(
                        f"official UAVDT Attributes row is not integral: {entry.filename}"
                    ) from exc
                if any(item not in {0, 1} for item in parsed):
                    raise DataError(
                        f"official UAVDT Attributes row is not binary: {entry.filename}"
                    )
                values[sequence] = parsed
                splits[sequence] = split
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise DataError(f"cannot inspect official UAVDT Attributes archive: {exc}") from exc
    return values, splits


def _parse_integer_rows(path: Path, *, allow_empty: bool) -> list[tuple[int, ...]]:
    try:
        rows = csv.reader(path.read_text(encoding="ascii").splitlines(), strict=True)
        parsed: list[tuple[int, ...]] = []
        for line_number, row in enumerate(rows, start=1):
            if len(row) != 9:
                raise DataError(f"UAVDT GT row must have nine columns: {path}:{line_number}")
            try:
                parsed.append(tuple(int(value.strip()) for value in row))
            except ValueError as exc:
                raise DataError(f"UAVDT GT row is not integral: {path}:{line_number}") from exc
    except (OSError, UnicodeError, csv.Error) as exc:
        raise DataError(f"cannot parse UAVDT GT file {path}: {exc}") from exc
    if not parsed and not allow_empty:
        raise DataError(f"UAVDT GT file is unexpectedly empty: {path}")
    return parsed


def _image_dimensions(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            image.verify()
            width, height = image.size
    except (OSError, ValueError) as exc:
        raise DataError(f"cannot validate UAVDT image {path}: {exc}") from exc
    if width <= 0 or height <= 0:
        raise DataError(f"UAVDT image has invalid dimensions: {path}")
    return width, height


def _attribute_mapping(values: tuple[int, ...]) -> dict[str, bool]:
    return {name: bool(value) for name, value in zip(UAVDT_ATTRIBUTE_NAMES, values, strict=True)}


def _sequence_image_id(sequence_index: int, frame_index: int) -> int:
    if sequence_index < 1 or frame_index < 1 or frame_index >= 1_000_000:
        raise DataError("UAVDT sequence/frame identifier is outside the deterministic ID domain")
    return sequence_index * 1_000_000 + frame_index


def _build_coco_split(
    source_root: Path,
    output_root: Path,
    *,
    split: str,
    sequences: list[str],
    sequence_ordinals: dict[str, int],
    frame_counts: dict[str, int],
    attributes: dict[str, tuple[int, ...]],
) -> dict[str, Any]:
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    ignore_regions: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()
    dimensions: Counter[str] = Counter()
    preserved_out_of_bounds: list[dict[str, Any]] = []
    annotation_id = 1
    ignore_id = 1

    for sequence in sequences:
        sequence_dir = source_root / sequence
        image_dir = sequence_dir / "img1"
        whole_path = sequence_dir / "gt" / "gt_whole.txt"
        ignore_path = sequence_dir / "gt" / "gt_ignore.txt"
        if not image_dir.is_dir():
            raise DataError(f"UAVDT sequence image directory is missing: {image_dir}")
        image_paths = sorted(image_dir.glob("*.jpg"), key=lambda path: path.name.casefold())
        observed_frames: list[int] = []
        paths_by_frame: dict[int, Path] = {}
        for image_path in image_paths:
            match = _IMAGE_RE.fullmatch(image_path.name)
            if match is None:
                raise DataError(f"unexpected UAVDT image filename: {image_path}")
            frame = int(match.group("frame"))
            observed_frames.append(frame)
            paths_by_frame[frame] = image_path
        expected_frames = list(range(1, frame_counts[sequence] + 1))
        if observed_frames != expected_frames:
            raise DataError(f"extracted UAVDT frame membership changed for {sequence}")

        whole_by_frame: dict[int, list[tuple[int, ...]]] = {}
        for row in _parse_integer_rows(whole_path, allow_empty=False):
            whole_by_frame.setdefault(row[0], []).append(row)
        ignore_by_frame: dict[int, list[tuple[int, ...]]] = {}
        for row in _parse_integer_rows(ignore_path, allow_empty=True):
            ignore_by_frame.setdefault(row[0], []).append(row)
        if not set(whole_by_frame).issubset(set(expected_frames)):
            raise DataError(f"UAVDT GT references a missing frame in {sequence}")
        if not set(ignore_by_frame).issubset(set(expected_frames)):
            raise DataError(f"UAVDT ignore GT references a missing frame in {sequence}")

        sequence_attributes = _attribute_mapping(attributes[sequence])
        for frame in expected_frames:
            image_path = paths_by_frame[frame]
            width, height = _image_dimensions(image_path)
            dimensions[f"{width}x{height}"] += 1
            image_id = _sequence_image_id(sequence_ordinals[sequence], frame)
            images.append(
                {
                    "id": image_id,
                    "file_name": f"{sequence}/img1/{image_path.name}",
                    "width": width,
                    "height": height,
                    "sequence": sequence,
                    "frame_index": frame,
                    "uavdt_attributes": sequence_attributes,
                }
            )
            for source_line, row in enumerate(whole_by_frame.get(frame, []), start=1):
                _, target_id, x, y, box_width, box_height, out_of_view, occlusion, category = row
                if category not in UAVDT_CATEGORY_MAPPING:
                    raise DataError(
                        f"UAVDT GT category is outside the official mapping: {sequence}/{frame}"
                    )
                if x < 0 or y < 0 or box_width <= 0 or box_height <= 0:
                    raise DataError(f"UAVDT GT bbox is invalid: {sequence}/{frame}")
                if x + box_width > width or y + box_height > height:
                    preserved_out_of_bounds.append(
                        {
                            "sequence": sequence,
                            "frame_index": frame,
                            "target_id": target_id,
                            "bbox": [x, y, box_width, box_height],
                            "image_size": [width, height],
                        }
                    )
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category,
                        "bbox": [x, y, box_width, box_height],
                        "area": box_width * box_height,
                        "iscrowd": 0,
                        "segmentation": [],
                        "uavdt_target_id": target_id,
                        "uavdt_out_of_view": out_of_view,
                        "uavdt_occlusion": occlusion,
                        "uavdt_source_line_within_frame": source_line,
                    }
                )
                class_counts[UAVDT_CATEGORY_MAPPING[category]] += 1
                annotation_id += 1
            for row in ignore_by_frame.get(frame, []):
                _, target_id, x, y, box_width, box_height, score, in_view, occlusion = row
                if x < 0 or y < 0 or box_width <= 0 or box_height <= 0:
                    raise DataError(f"UAVDT ignore bbox is invalid: {sequence}/{frame}")
                ignore_regions.append(
                    {
                        "id": ignore_id,
                        "image_id": image_id,
                        "sequence": sequence,
                        "frame_index": frame,
                        "target_id": target_id,
                        "bbox": [x, y, box_width, box_height],
                        "source_tail": [score, in_view, occlusion],
                    }
                )
                ignore_id += 1

    categories = [
        {"id": category_id, "name": name, "supercategory": "vehicle"}
        for category_id, name in UAVDT_CATEGORY_MAPPING.items()
    ]
    document = {
        "info": {
            "description": "UAVDT-Benchmark-M deterministic COCO conversion",
            "split": split,
            "bbox_policy": "preserve_official_gt_whole_without_clipping",
            "ignore_policy": "preserve_gt_ignore_in_separate_sidecar_without_category_guessing",
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    annotation_path = output_root / "annotations" / f"{split}.coco.json"
    ignore_path = output_root / "ignore_regions" / f"{split}.json"
    atomic_write_json(annotation_path, document)
    atomic_write_json(
        ignore_path,
        {
            "schema_version": 1,
            "dataset": "uavdt",
            "split": split,
            "category_assignment": "unknown_preserved_as_ignore_sidecar",
            "regions": ignore_regions,
        },
    )
    return {
        "split": split,
        "sequences": len(sequences),
        "sequence_names": sequences,
        "images": len(images),
        "annotations": len(annotations),
        "ignore_regions": len(ignore_regions),
        "class_counts": dict(sorted(class_counts.items())),
        "image_dimensions": dict(sorted(dimensions.items())),
        "preserved_out_of_bounds_boxes": preserved_out_of_bounds,
        "annotation_path": annotation_path.relative_to(output_root).as_posix(),
        "annotation_sha256": sha256_file(annotation_path),
        "ignore_path": ignore_path.relative_to(output_root).as_posix(),
        "ignore_sha256": sha256_file(ignore_path),
    }


def _reuse_conversion(
    output_root: Path,
    *,
    archive_gate_sha256: str,
    archive_gate_evidence_sha256: str,
    extraction_marker_sha256: str,
) -> dict[str, Any] | None:
    if not output_root.exists():
        return None
    manifest_path = output_root / "conversion_manifest.json"
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DataError(
            f"UAVDT conversion output exists without a valid manifest: {output_root}: {exc}"
        ) from exc
    if not isinstance(value, dict) or value.get("status") != "PASS":
        raise DataError("existing UAVDT conversion does not have PASS status")
    legacy_report_lock_matches = (
        value.get("archive_gate_evidence_sha256") is None
        and value.get("archive_gate_sha256") == archive_gate_sha256
    )
    evidence_lock_matches = (
        value.get("archive_gate_evidence_sha256") == archive_gate_evidence_sha256
    )
    if not legacy_report_lock_matches and not evidence_lock_matches:
        raise DataError("existing UAVDT conversion does not match the current source evidence")
    if value.get("extraction_marker_sha256") != extraction_marker_sha256:
        raise DataError("existing UAVDT conversion does not match the current source locks")
    for split in value.get("splits", []):
        if not isinstance(split, dict):
            raise DataError("existing UAVDT conversion manifest has an invalid split")
        annotation = output_root / str(split.get("annotation_path", ""))
        ignore = output_root / str(split.get("ignore_path", ""))
        if (
            not annotation.is_file()
            or sha256_file(annotation) != split.get("annotation_sha256")
            or not ignore.is_file()
            or sha256_file(ignore) != split.get("ignore_sha256")
        ):
            raise DataError("existing UAVDT converted output hash verification failed")
    normalized: dict[str, Any] = {
        **value,
        "archive_gate_sha256": archive_gate_sha256,
        "archive_gate_evidence_sha256": archive_gate_evidence_sha256,
        "reused_existing": False,
    }
    normalized.pop("manifest_sha256", None)
    normalized["evidence_sha256"] = uavdt_conversion_evidence_sha256(normalized)
    normalized["manifest_sha256"] = stable_hash(normalized, length=64)
    atomic_write_json(manifest_path, normalized)
    return {**normalized, "reused_existing": True}


def convert_uavdt_to_coco(
    source_root: Path,
    attributes_archive: Path,
    archive_gate_report: Path,
    output_root: Path,
    *,
    report_output: Path | None = None,
) -> dict[str, Any]:
    """Convert frozen UAVDT GT to COCO while preserving official split and raw boxes."""

    source_root = source_root.resolve()
    attributes_archive = attributes_archive.resolve()
    archive_gate_report = archive_gate_report.resolve()
    output_root = output_root.resolve()
    if source_root.name != "UAV-benchmark-M" or not source_root.is_dir():
        raise DataError(f"validated UAVDT extraction root is unavailable: {source_root}")
    gate = _load_gate(archive_gate_report)
    extraction = _load_extraction_marker(source_root)
    archives = gate.get("archives")
    dataset_gate = archives.get("dataset") if isinstance(archives, dict) else None
    attributes_gate = archives.get("attributes") if isinstance(archives, dict) else None
    if not isinstance(dataset_gate, dict) or not isinstance(attributes_gate, dict):
        raise DataError("UAVDT archive gate lacks Dataset or Attributes evidence")
    if extraction.get("archive_sha256") != dataset_gate.get("sha256"):
        raise DataError("UAVDT extraction marker differs from the archive gate")

    archive_gate_sha256 = sha256_file(archive_gate_report)
    archive_gate_evidence_sha256 = uavdt_archive_evidence_sha256(gate)
    extraction_marker_path = source_root.parent / ".uavdt_extraction.json"
    extraction_marker_sha256 = sha256_file(extraction_marker_path)
    reused = _reuse_conversion(
        output_root,
        archive_gate_sha256=archive_gate_sha256,
        archive_gate_evidence_sha256=archive_gate_evidence_sha256,
        extraction_marker_sha256=extraction_marker_sha256,
    )
    if reused is not None:
        if report_output is not None:
            atomic_write_json(report_output, reused)
        return reused

    split_evidence = gate.get("official_split")
    dataset_evidence = gate.get("dataset")
    if not isinstance(split_evidence, dict) or not isinstance(dataset_evidence, dict):
        raise DataError("UAVDT archive gate lacks split or Dataset inventory")
    train_sequences = split_evidence.get("train_sequences")
    test_sequences = split_evidence.get("test_sequences")
    frame_counts_raw = dataset_evidence.get("frame_counts")
    if (
        not isinstance(train_sequences, list)
        or not all(isinstance(value, str) for value in train_sequences)
        or not isinstance(test_sequences, list)
        or not all(isinstance(value, str) for value in test_sequences)
        or not isinstance(frame_counts_raw, dict)
    ):
        raise DataError("UAVDT archive gate split membership is malformed")
    frame_counts: dict[str, int] = {}
    for sequence, count in frame_counts_raw.items():
        if not isinstance(sequence, str) or not isinstance(count, int) or count <= 0:
            raise DataError("UAVDT archive gate frame counts are malformed")
        frame_counts[sequence] = count
    train = list(train_sequences)
    test = list(test_sequences)
    official_sequences = set(train) | set(test)
    if set(train) & set(test) or official_sequences != set(frame_counts):
        raise DataError("UAVDT archive gate train/test union is inconsistent")
    actual_sequences = {
        child.name
        for child in source_root.iterdir()
        if child.is_dir() and _SEQUENCE_RE.fullmatch(child.name) is not None
    }
    if actual_sequences != official_sequences:
        raise DataError("extracted UAVDT sequence set differs from the archive gate")

    expected_attributes_bytes = attributes_gate.get("bytes")
    expected_attributes_sha256 = attributes_gate.get("sha256")
    if not isinstance(expected_attributes_bytes, int) or not isinstance(
        expected_attributes_sha256, str
    ):
        raise DataError("UAVDT archive gate Attributes evidence is malformed")
    attribute_values, attribute_splits = _parse_attributes_archive(
        attributes_archive,
        expected_bytes=expected_attributes_bytes,
        expected_sha256=expected_attributes_sha256,
    )
    if set(attribute_values) != official_sequences:
        raise DataError("official UAVDT Attributes union differs from the archive gate")
    if {sequence for sequence, split in attribute_splits.items() if split == "train"} != set(train):
        raise DataError("official UAVDT Attributes train membership changed")
    if {sequence for sequence, split in attribute_splits.items() if split == "test"} != set(test):
        raise DataError("official UAVDT Attributes test membership changed")

    sequence_ordinals = {
        sequence: index for index, sequence in enumerate(sorted(official_sequences), start=1)
    }
    temporary = output_root.parent / f".{output_root.name}.converting-{uuid.uuid4().hex}"
    output_root.parent.mkdir(parents=True, exist_ok=True)
    if temporary.exists():
        raise DataError(f"refusing to reuse UAVDT conversion temporary directory: {temporary}")
    try:
        temporary.mkdir()
        split_results = [
            _build_coco_split(
                source_root,
                temporary,
                split=split,
                sequences=sequences,
                sequence_ordinals=sequence_ordinals,
                frame_counts=frame_counts,
                attributes=attribute_values,
            )
            for split, sequences in (("train", train), ("test", test))
        ]
        atomic_write_json(
            temporary / "sequence_attributes.json",
            {
                "schema_version": 1,
                "field_order": list(UAVDT_ATTRIBUTE_NAMES),
                "sequences": {
                    sequence: {
                        "split": attribute_splits[sequence],
                        "values": list(attribute_values[sequence]),
                        "attributes": _attribute_mapping(attribute_values[sequence]),
                    }
                    for sequence in sorted(attribute_values)
                },
                "source_files_modified": False,
            },
        )
        sequence_attributes_sha256 = sha256_file(temporary / "sequence_attributes.json")
        core: dict[str, Any] = {
            "schema_version": 1,
            "status": "PASS",
            "dataset": "uavdt",
            "source_root": str(source_root),
            "output_root": str(output_root),
            "archive_gate_report": str(archive_gate_report),
            "archive_gate_sha256": archive_gate_sha256,
            "archive_gate_evidence_sha256": archive_gate_evidence_sha256,
            "extraction_marker_sha256": extraction_marker_sha256,
            "attributes_archive_sha256": expected_attributes_sha256,
            "category_mapping": {str(key): value for key, value in UAVDT_CATEGORY_MAPPING.items()},
            "bbox_policy": "preserve_official_gt_whole_without_clipping",
            "ignore_policy": "separate_sidecar_without_category_guessing",
            "id_policy": "sorted_sequence_ordinal_times_1000000_plus_frame_index",
            "splits": split_results,
            "sequence_attributes_sha256": sequence_attributes_sha256,
            "formal_metrics_calculated_or_viewed": False,
            "training_tuning_or_inference_run": False,
            "source_files_modified": False,
            "reused_existing": False,
        }
        core["evidence_sha256"] = uavdt_conversion_evidence_sha256(core)
        core["manifest_sha256"] = stable_hash(core, length=64)
        atomic_write_json(temporary / "conversion_manifest.json", core)
        temporary.replace(output_root)
    except BaseException:
        if temporary.is_dir():
            shutil.rmtree(temporary)
        raise

    report = {
        **core,
        "converted_at_utc": datetime.now(timezone.utc).isoformat(),
        "conversion_manifest": str(output_root / "conversion_manifest.json"),
        "conversion_manifest_file_sha256": sha256_file(output_root / "conversion_manifest.json"),
    }
    if report_output is not None:
        atomic_write_json(report_output, report)
    return report


def uavdt_conversion_evidence_sha256(report: dict[str, Any]) -> str:
    """Hash converted content and frozen policies while excluding audit timestamps/paths."""

    required = (
        "status",
        "dataset",
        "archive_gate_evidence_sha256",
        "extraction_marker_sha256",
        "attributes_archive_sha256",
        "category_mapping",
        "bbox_policy",
        "ignore_policy",
        "id_policy",
        "splits",
        "sequence_attributes_sha256",
        "formal_metrics_calculated_or_viewed",
        "training_tuning_or_inference_run",
        "source_files_modified",
    )
    missing = [key for key in required if key not in report]
    if missing:
        raise DataError(f"UAVDT conversion evidence is incomplete: {missing}")
    splits = report["splits"]
    if not isinstance(splits, list):
        raise DataError("UAVDT conversion split evidence is malformed")
    stable_splits: list[dict[str, Any]] = []
    for split in splits:
        if not isinstance(split, dict):
            raise DataError("UAVDT conversion split row is malformed")
        stable_splits.append(
            {
                key: value
                for key, value in split.items()
                if key not in {"annotation_path", "ignore_path"}
            }
        )
    payload = {
        "schema_version": 1,
        "status": report["status"],
        "dataset": report["dataset"],
        "archive_gate_evidence_sha256": report["archive_gate_evidence_sha256"],
        "extraction_marker_sha256": report["extraction_marker_sha256"],
        "attributes_archive_sha256": report["attributes_archive_sha256"],
        "category_mapping": report["category_mapping"],
        "bbox_policy": report["bbox_policy"],
        "ignore_policy": report["ignore_policy"],
        "id_policy": report["id_policy"],
        "splits": stable_splits,
        "sequence_attributes_sha256": report["sequence_attributes_sha256"],
        "formal_metrics_calculated_or_viewed": report["formal_metrics_calculated_or_viewed"],
        "training_tuning_or_inference_run": report["training_tuning_or_inference_run"],
        "source_files_modified": report["source_files_modified"],
    }
    return stable_hash(payload, length=64)
