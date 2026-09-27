from __future__ import annotations

import json
import math
import re
import shutil
import uuid
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from buse_uav.data.coco import CocoDocument, load_coco
from buse_uav.data.common import DataError, validation_result
from buse_uav.data.manifest import build_file_manifest
from buse_uav.schemas import ImageRecord
from buse_uav.utils.hashing import sha256_file, stable_hash
from buse_uav.utils.io import atomic_write_json

AUAIR_CLASS_NAMES = ("car", "truck", "bus")
AUAIR_TARGET_CATEGORIES = {1: "car", 2: "truck", 3: "bus"}
AUAIR_SOURCE_CATEGORIES = (
    "Human",
    "Car",
    "Truck",
    "Van",
    "Motorbike",
    "Bicycle",
    "Bus",
    "Trailer",
)
AUAIR_SOURCE_TO_TARGET = {1: 1, 2: 2, 6: 3}
AUAIR_LICENSE_URLS = {
    "http://creativecommons.org/licenses/by-nc-sa/2.0/",
    "http://creativecommons.org/licenses/by-nc/2.0/",
}
AUAIR_ANNOTATION_SHA256 = "3bc6bf049e3430f450ac7bf6933695fe7abc4ce491c31a28c48351c8c86f1ceb"
AUAIR_IMAGE_ARCHIVE_SHA256 = "bd31f8de50fd54182b3569d5671ad70cc3fcfe7e562313319ef9810f6e91e72b"
AUAIR_SEQUENCE_COUNTS = {
    "20190829091111": 2592,
    "20190905091750": 5734,
    "20190905103112": 6840,
    "20190905111947": 771,
    "20190905112522": 5358,
    "20190905142119": 2962,
    "20190905143505": 1580,
    "20190906150731": 6986,
}
AUAIR_1HZ_SEQUENCE_COUNTS = {
    "20190829091111": 520,
    "20190905091750": 1147,
    "20190905103112": 1373,
    "20190905111947": 154,
    "20190905112522": 1074,
    "20190905142119": 592,
    "20190905143505": 313,
    "20190906150731": 1405,
}
_IMAGE_RE = re.compile(
    r"frame_(?P<sequence>\d{14})_x{1,2}_(?P<frame_index>\d{7})\.jpg\Z",
    re.IGNORECASE,
)


class AuAirAdapter:
    """Adapter for the frozen three-class AU-AIR-1Hz COCO conversion."""

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
        return AUAIR_CLASS_NAMES

    def image_dir(self) -> Path:
        path = self._image_directory or self.root / "images"
        if not path.is_dir():
            raise DataError(f"AU-AIR image directory does not exist: {path}")
        return path

    def annotation_file(self) -> Path:
        path = self._annotation_file or self.root / "annotations" / "auair_1hz.coco.json"
        if not path.is_file():
            raise DataError(f"AU-AIR COCO annotation file does not exist: {path}")
        return path

    def document(self) -> CocoDocument:
        return load_coco(self.annotation_file())

    def image_path(self, file_name: str) -> Path:
        root = self.image_dir().resolve()
        candidate = (root / file_name).resolve()
        if candidate != root and root not in candidate.parents:
            raise DataError(f"AU-AIR image path escapes the configured root: {file_name}")
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
                raise DataError(f"invalid AU-AIR COCO image entry: {image}")
            path = self.image_path(file_name)
            if not path.is_file():
                raise DataError(f"AU-AIR image file does not exist: {path}")
            records.append(
                ImageRecord(
                    image_id=image_id,
                    path=str(path),
                    width=width,
                    height=height,
                )
            )
        return tuple(records)

    def sequence_by_image(self) -> dict[int, str]:
        output: dict[int, str] = {}
        for image in self.document().images:
            image_id = image.get("id")
            sequence = image.get("sequence")
            if not isinstance(image_id, int) or not isinstance(sequence, str):
                raise DataError(f"AU-AIR image lacks integer id or sequence: {image}")
            output[image_id] = sequence
        return output

    def validate(self) -> dict[str, Any]:
        errors: list[str] = []
        warnings: list[str] = []
        try:
            document = self.document()
        except DataError as exc:
            return validation_result(
                dataset="auair",
                split=self.split,
                errors=[str(exc)],
                warnings=[],
                stats={},
            )
        if document.category_names_by_id != AUAIR_TARGET_CATEGORIES:
            errors.append(
                "AU-AIR converted categories must be exactly car=1, truck=2, bus=3; "
                f"found {document.category_names_by_id}"
            )
        images_by_id = document.images_by_id
        sequence_counts: Counter[str] = Counter()
        missing_images = 0
        for image in document.images:
            image_id = image.get("id")
            file_name = image.get("file_name")
            sequence = image.get("sequence")
            frame_index = image.get("frame_index")
            if not isinstance(image_id, int):
                errors.append(f"AU-AIR image has invalid ID: {image_id}")
            if not isinstance(file_name, str) or not self.image_path(file_name).is_file():
                missing_images += 1
                errors.append(f"missing AU-AIR image file: {file_name}")
            if not isinstance(sequence, str) or re.fullmatch(r"\d{14}", sequence) is None:
                errors.append(f"invalid AU-AIR sequence: {sequence}")
            else:
                sequence_counts[sequence] += 1
            if not isinstance(frame_index, int) or frame_index % 5 != 0:
                errors.append(f"AU-AIR-1Hz frame violates modulo-five rule: {frame_index}")
        if self.split == "external_1hz" and dict(sorted(sequence_counts.items())) != dict(
            sorted(AUAIR_1HZ_SEQUENCE_COUNTS.items())
        ):
            errors.append("AU-AIR-1Hz sequence membership or counts changed")

        annotation_ids: set[int] = set()
        class_counts: Counter[str] = Counter()
        for annotation in document.annotations:
            annotation_id = annotation.get("id")
            image_id = annotation.get("image_id")
            category_id = annotation.get("category_id")
            bbox = annotation.get("bbox")
            if not isinstance(annotation_id, int) or annotation_id in annotation_ids:
                errors.append(f"invalid or duplicate AU-AIR annotation ID: {annotation_id}")
                continue
            annotation_ids.add(annotation_id)
            image_entry = images_by_id.get(image_id) if isinstance(image_id, int) else None
            if image_entry is None:
                errors.append(f"AU-AIR annotation {annotation_id} references an unknown image")
                continue
            if category_id not in AUAIR_TARGET_CATEGORIES:
                errors.append(f"AU-AIR annotation {annotation_id} has an invalid category")
                continue
            if not isinstance(bbox, list) or len(bbox) != 4:
                errors.append(f"AU-AIR annotation {annotation_id} has an invalid bbox")
                continue
            x, y, width, height = (float(value) for value in bbox)
            if (
                not all(math.isfinite(value) for value in (x, y, width, height))
                or x < 0
                or y < 0
                or width <= 0
                or height <= 0
                or x + width > float(image_entry["width"]) + 1e-6
                or y + height > float(image_entry["height"]) + 1e-6
            ):
                errors.append(f"AU-AIR annotation {annotation_id} has invalid geometry")
                continue
            class_counts[AUAIR_TARGET_CATEGORIES[int(category_id)]] += 1
        return validation_result(
            dataset="auair",
            split=self.split,
            errors=errors,
            warnings=warnings,
            stats={
                "images": len(document.images),
                "instances": len(annotation_ids),
                "classes": dict(sorted(class_counts.items())),
                "sequences": dict(sorted(sequence_counts.items())),
                "missing_images": missing_images,
                "annotation_file": str(self.annotation_file()),
                "image_directory": str(self.image_dir()),
            },
        )

    def manifest(self) -> dict[str, Any]:
        paths = [self.annotation_file(), *(Path(record.path) for record in self.records())]
        return {
            "dataset": "auair",
            "split": self.split,
            "root": str(self.root.resolve()),
            "files": build_file_manifest(paths, root=self.root),
        }


def prepare_auair_1hz(
    annotation_path: Path,
    image_archive: Path,
    output_root: Path,
    *,
    report_output: Path | None = None,
    expected_annotation_sha256: str = AUAIR_ANNOTATION_SHA256,
    expected_archive_sha256: str = AUAIR_IMAGE_ARCHIVE_SHA256,
    expected_sequence_counts: Mapping[str, int] = AUAIR_SEQUENCE_COUNTS,
) -> dict[str, Any]:
    """Validate the official release and extract the frozen modulo-five 1-Hz subset."""
    for path, label in ((annotation_path, "annotation JSON"), (image_archive, "image ZIP")):
        if not path.is_file():
            raise DataError(f"AU-AIR {label} does not exist: {path}")
    annotation_sha256 = sha256_file(annotation_path)
    archive_sha256 = sha256_file(image_archive)
    if annotation_sha256 != expected_annotation_sha256.casefold():
        raise DataError("AU-AIR annotation SHA256 differs from the frozen source lock")
    if archive_sha256 != expected_archive_sha256.casefold():
        raise DataError("AU-AIR image ZIP SHA256 differs from the frozen source lock")

    if output_root.exists():
        return _reuse_prepared_auair(
            output_root,
            annotation_sha256=annotation_sha256,
            archive_sha256=archive_sha256,
            report_output=report_output,
        )
    try:
        raw = json.loads(annotation_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DataError(f"cannot parse official AU-AIR annotations: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("annotations"), list):
        raise DataError("official AU-AIR annotation structure is invalid")
    if tuple(raw.get("categories", ())) != AUAIR_SOURCE_CATEGORIES:
        raise DataError("official AU-AIR source categories changed")
    licenses = raw.get("licenses")
    if not isinstance(licenses, list):
        raise DataError("official AU-AIR JSON has no license declarations")
    license_urls = {
        str(row.get("url")) for row in licenses if isinstance(row, dict) and row.get("url")
    }
    if license_urls != AUAIR_LICENSE_URLS:
        raise DataError(f"official AU-AIR license declarations changed: {sorted(license_urls)}")

    parsed_rows: list[tuple[str, int, str, dict[str, Any]]] = []
    sequence_counts: Counter[str] = Counter()
    for row in raw["annotations"]:
        if not isinstance(row, dict) or not isinstance(row.get("image_name"), str):
            raise DataError("official AU-AIR annotations contain a malformed frame row")
        match = _IMAGE_RE.fullmatch(row["image_name"])
        if match is None:
            raise DataError(f"unexpected AU-AIR image name: {row['image_name']}")
        sequence = match.group("sequence")
        frame_index = int(match.group("frame_index"))
        sequence_counts[sequence] += 1
        parsed_rows.append((sequence, frame_index, row["image_name"], row))
    if dict(sorted(sequence_counts.items())) != dict(sorted(expected_sequence_counts.items())):
        raise DataError("AU-AIR sequence membership or full-frame counts changed")

    try:
        archive = zipfile.ZipFile(image_archive)
    except (OSError, zipfile.BadZipFile) as exc:
        raise DataError(f"cannot open official AU-AIR image ZIP: {exc}") from exc
    with archive:
        members: dict[str, str] = {}
        for info in archive.infolist():
            if info.is_dir() or Path(info.filename).suffix.casefold() != ".jpg":
                continue
            name = Path(info.filename).name
            if name in members:
                raise DataError(f"duplicate AU-AIR image basename in ZIP: {name}")
            members[name] = info.filename
        annotated_names = {name for _, _, name, _ in parsed_rows}
        if set(members) != annotated_names:
            raise DataError("AU-AIR ZIP image membership differs from the annotation JSON")

        selected = sorted(
            (item for item in parsed_rows if item[1] % 5 == 0),
            key=lambda item: (item[0], item[1], item[2]),
        )
        selected_sequence_counts = Counter(item[0] for item in selected)
        if expected_sequence_counts == AUAIR_SEQUENCE_COUNTS and dict(
            sorted(selected_sequence_counts.items())
        ) != dict(sorted(AUAIR_1HZ_SEQUENCE_COUNTS.items())):
            raise DataError("AU-AIR modulo-five subset count differs from the frozen lock")

        temporary = output_root.parent / f".{output_root.name}.preparing-{uuid.uuid4().hex}"
        if temporary.exists():
            raise DataError(f"refusing to reuse AU-AIR temporary directory: {temporary}")
        try:
            images_dir = temporary / "images"
            annotations_dir = temporary / "annotations"
            images_dir.mkdir(parents=True)
            annotations_dir.mkdir(parents=True)
            coco_images: list[dict[str, Any]] = []
            coco_annotations: list[dict[str, Any]] = []
            selection_rows: list[dict[str, Any]] = []
            source_selected_counts: Counter[str] = Counter()
            target_counts: Counter[str] = Counter()
            dropped_non_positive: Counter[str] = Counter()
            annotation_id = 1
            for image_id, (sequence, frame_index, name, source_row) in enumerate(selected, start=1):
                destination = images_dir / name
                with archive.open(members[name]) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target)
                try:
                    with Image.open(destination) as image:
                        image.load()
                        actual_width, actual_height = image.size
                except (OSError, ValueError) as exc:
                    raise DataError(f"cannot decode extracted AU-AIR image {name}: {exc}") from exc
                declared_width = int(source_row.get("image_width:", -1))
                declared_height = int(source_row.get("image_height", -1))
                if (actual_width, actual_height) != (declared_width, declared_height):
                    raise DataError(
                        f"AU-AIR image size mismatch for {name}: "
                        f"JSON={(declared_width, declared_height)} "
                        f"actual={(actual_width, actual_height)}"
                    )
                coco_images.append(
                    {
                        "id": image_id,
                        "file_name": name,
                        "width": actual_width,
                        "height": actual_height,
                        "sequence": sequence,
                        "frame_index": frame_index,
                        "source_image_name": name,
                    }
                )
                selection_rows.append(
                    {
                        "image_id": image_id,
                        "source_image_name": name,
                        "sequence": sequence,
                        "frame_index": frame_index,
                        "selection_rule": "frame_index_mod_5_equals_0",
                        "sha256": sha256_file(destination),
                    }
                )
                raw_boxes = source_row.get("bbox")
                if not isinstance(raw_boxes, list):
                    raise DataError(f"AU-AIR frame {name} has no bbox list")
                for raw_box in raw_boxes:
                    if not isinstance(raw_box, dict) or not isinstance(raw_box.get("class"), int):
                        raise DataError(f"AU-AIR frame {name} contains a malformed bbox")
                    source_class = int(raw_box["class"])
                    if not 0 <= source_class < len(AUAIR_SOURCE_CATEGORIES):
                        raise DataError(f"AU-AIR frame {name} contains an unknown class ID")
                    source_name = AUAIR_SOURCE_CATEGORIES[source_class]
                    source_selected_counts[source_name] += 1
                    if source_class not in AUAIR_SOURCE_TO_TARGET:
                        continue
                    x = float(raw_box.get("left", math.nan))
                    y = float(raw_box.get("top", math.nan))
                    width = float(raw_box.get("width", math.nan))
                    height = float(raw_box.get("height", math.nan))
                    if width <= 0 or height <= 0:
                        dropped_non_positive[source_name] += 1
                        continue
                    if (
                        not all(math.isfinite(value) for value in (x, y, width, height))
                        or x < 0
                        or y < 0
                        or x + width > actual_width + 1e-6
                        or y + height > actual_height + 1e-6
                    ):
                        raise DataError(f"AU-AIR frame {name} contains invalid mapped geometry")
                    target_id = AUAIR_SOURCE_TO_TARGET[source_class]
                    target_name = AUAIR_TARGET_CATEGORIES[target_id]
                    target_counts[target_name] += 1
                    coco_annotations.append(
                        {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": target_id,
                            "bbox": [x, y, width, height],
                            "area": width * height,
                            "iscrowd": 0,
                            "source_category_id": source_class,
                            "source_category_name": source_name,
                        }
                    )
                    annotation_id += 1

            selection_path = temporary / "selection_manifest.json"
            atomic_write_json(
                selection_path,
                {
                    "schema_version": 1,
                    "selection_rule": "released_5fps_frame_index_mod_5_equals_0",
                    "source_frame_count": len(parsed_rows),
                    "selected_frame_count": len(selected),
                    "sequence_counts": dict(sorted(selected_sequence_counts.items())),
                    "images": selection_rows,
                },
            )
            annotation_output = annotations_dir / "auair_1hz.coco.json"
            atomic_write_json(
                annotation_output,
                {
                    "info": {
                        "description": "AU-AIR-1Hz exact car/truck/bus conversion",
                        "source": raw.get("info"),
                        "selection_rule": "frame_index_mod_5_equals_0",
                        "mapping_policy": "Car/Truck/Bus only; no inferred superclass mapping",
                    },
                    "licenses": licenses,
                    "images": coco_images,
                    "annotations": coco_annotations,
                    "categories": [
                        {"id": category_id, "name": name}
                        for category_id, name in AUAIR_TARGET_CATEGORIES.items()
                    ],
                },
            )
            core: dict[str, Any] = {
                "schema_version": 1,
                "status": "PASS",
                "dataset": "auair",
                "split": "external_1hz",
                "source": {
                    "annotation_sha256": annotation_sha256,
                    "image_archive_sha256": archive_sha256,
                    "frames": len(parsed_rows),
                    "sequences": dict(sorted(sequence_counts.items())),
                    "licenses": sorted(license_urls),
                },
                "selection": {
                    "rule": "frame_index_mod_5_equals_0",
                    "frames": len(selected),
                    "sequences": dict(sorted(selected_sequence_counts.items())),
                    "manifest_sha256": sha256_file(selection_path),
                },
                "category_mapping": {
                    AUAIR_SOURCE_CATEGORIES[source_id]: AUAIR_TARGET_CATEGORIES[target_id]
                    for source_id, target_id in AUAIR_SOURCE_TO_TARGET.items()
                },
                "unmapped_source_categories": [
                    name
                    for source_id, name in enumerate(AUAIR_SOURCE_CATEGORIES)
                    if source_id not in AUAIR_SOURCE_TO_TARGET
                ],
                "source_selected_class_counts": dict(sorted(source_selected_counts.items())),
                "target_instance_counts": dict(sorted(target_counts.items())),
                "dropped_non_positive_mapped_boxes": dict(sorted(dropped_non_positive.items())),
                "bbox_policy": "drop_non_positive_with_audit; otherwise preserve_exact_xywh",
                "annotation_sha256": sha256_file(annotation_output),
                "formal_metrics_calculated_or_viewed": False,
                "training_tuning_or_calibration": False,
                "source_files_modified": False,
                "reused_existing": False,
            }
            core["evidence_sha256"] = stable_hash(core, length=64)
            atomic_write_json(temporary / "conversion_manifest.json", core)
            output_root.parent.mkdir(parents=True, exist_ok=True)
            temporary.replace(output_root)
        except BaseException:
            if temporary.is_dir():
                shutil.rmtree(temporary)
            raise

    report = {
        **core,
        "output_root": str(output_root.resolve()),
        "annotation": str((output_root / "annotations" / "auair_1hz.coco.json").resolve()),
        "selection_manifest": str((output_root / "selection_manifest.json").resolve()),
        "conversion_manifest": str((output_root / "conversion_manifest.json").resolve()),
        "conversion_manifest_sha256": sha256_file(output_root / "conversion_manifest.json"),
    }
    if report_output is not None:
        atomic_write_json(report_output, report)
    return report


def _reuse_prepared_auair(
    output_root: Path,
    *,
    annotation_sha256: str,
    archive_sha256: str,
    report_output: Path | None,
) -> dict[str, Any]:
    manifest_path = output_root / "conversion_manifest.json"
    annotation_path = output_root / "annotations" / "auair_1hz.coco.json"
    selection_path = output_root / "selection_manifest.json"
    for path in (manifest_path, annotation_path, selection_path):
        if not path.is_file():
            raise DataError(f"existing AU-AIR output is incomplete: {path}")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("status") != "PASS":
        raise DataError("existing AU-AIR conversion manifest is invalid")
    source = value.get("source")
    if (
        not isinstance(source, dict)
        or source.get("annotation_sha256") != annotation_sha256
        or source.get("image_archive_sha256") != archive_sha256
    ):
        raise DataError("existing AU-AIR conversion uses different source locks")
    selection = value.get("selection")
    if (
        not isinstance(selection, dict)
        or selection.get("manifest_sha256") != sha256_file(selection_path)
        or value.get("annotation_sha256") != sha256_file(annotation_path)
    ):
        raise DataError("existing AU-AIR converted evidence hashes do not verify")
    report = {
        **value,
        "reused_existing": True,
        "output_root": str(output_root.resolve()),
        "annotation": str(annotation_path.resolve()),
        "selection_manifest": str(selection_path.resolve()),
        "conversion_manifest": str(manifest_path.resolve()),
        "conversion_manifest_sha256": sha256_file(manifest_path),
    }
    if report_output is not None:
        atomic_write_json(report_output, report)
    return report
