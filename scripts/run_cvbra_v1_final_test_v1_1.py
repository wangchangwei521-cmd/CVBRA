from __future__ import annotations

import argparse
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from scripts import run_cvbra_v1_final_test_v1 as parent

from buse_uav.utils.hashing import sha256_file
from buse_uav.utils.io import atomic_write_json

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/experiment/cvbra_v1_final_test_v1_1_amendment.yaml"
PROTOCOL_SHA256 = "5295e703ccf394cb28f1449aca91d382fcb079d686fcf3bb44ad8b453ac95424"
PARENT_RUNNER = ROOT / "scripts/run_cvbra_v1_final_test_v1.py"
PARENT_RUNNER_SHA256 = "8b022df921c9eeee621c571cd60a225039650aa181b7ec827e388a63eaa9ee5e"
AMENDMENT = parent.OUTPUT / "UAV_CLASS_MAPPING_FAILURE_AMENDMENT_1.json"
AMENDED = parent.OUTPUT / "UAV_CLASS_MAPPING_AMENDED"


class FinalTestAmendmentError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ordered_names(bundle: zipfile.ZipFile) -> list[str]:
    value = yaml.safe_load(bundle.read("UAV-OBB/data.yaml").decode("utf-8"))
    names = value.get("names") if isinstance(value, dict) else None
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=lambda item: int(item))]
    if isinstance(names, list):
        return [str(item) for item in names]
    raise FinalTestAmendmentError("UAV-OBB data.yaml has no class-name registry")


def _released_to_model_category(names: list[str]) -> dict[int, int]:
    normalized = [name.casefold() for name in names]
    result: dict[int, int] = {}
    for model_class, name in enumerate(parent.CLASS_NAMES):
        matches = [index for index, value in enumerate(normalized) if value == name]
        if len(matches) != 1:
            raise FinalTestAmendmentError(f"shared class mapping is ambiguous: {name}")
        result[matches[0]] = model_class + 1
    expected = {1: 3, 2: 1, 5: 2}
    if result != expected:
        raise FinalTestAmendmentError(f"released class mapping changed: {result}")
    return result


def amend() -> dict[str, Any]:
    parent.register()
    parent.predict()
    if not PROTOCOL.is_file() or sha256_file(PROTOCOL) != PROTOCOL_SHA256:
        raise FinalTestAmendmentError("class-mapping amendment protocol changed")
    if not PARENT_RUNNER.is_file() or sha256_file(PARENT_RUNNER) != PARENT_RUNNER_SHA256:
        raise FinalTestAmendmentError("parent final-test runner changed")
    if AMENDMENT.exists() or AMENDED.exists():
        if not AMENDMENT.is_file() or not AMENDED.is_file():
            raise FinalTestAmendmentError("class-mapping amendment is incomplete")
        value = parent._load_mapping(AMENDMENT)
        marker = parent._load_mapping(AMENDED)
        if (
            value.get("protocol_sha256") != PROTOCOL_SHA256
            or value.get("wrapper_sha256") != sha256_file(Path(__file__))
            or marker.get("amendment_sha256") != sha256_file(AMENDMENT)
        ):
            raise FinalTestAmendmentError("class-mapping amendment changed")
        return value
    if any(
        path.exists()
        for path in (
            parent.UAV_ANNOTATION,
            parent.LABEL_LOCK,
            parent.METRICS,
            parent.STATISTICS,
            parent.REPORT,
        )
    ):
        raise FinalTestAmendmentError("corrected output appeared before amendment lock")
    with zipfile.ZipFile(parent.UAV_ARCHIVE) as bundle:
        names = _ordered_names(bundle)
    mapping = _released_to_model_category(names)
    payload = {
        "schema_version": 1,
        "status": "UAV_CLASS_MAPPING_FAILURE_AMENDED_BEFORE_TEST_BOX_LABEL_ACCESS",
        "recorded_at_utc": _now(),
        "protocol": parent._relative(PROTOCOL),
        "protocol_sha256": PROTOCOL_SHA256,
        "wrapper": parent._relative(Path(__file__)),
        "wrapper_sha256": sha256_file(Path(__file__)),
        "parent_runner_sha256": PARENT_RUNNER_SHA256,
        "prediction_lock_sha256": sha256_file(parent.PREDICTION_LOCK),
        "failed_assumption": "first three released class indices equal car, truck, bus",
        "observed_names": names,
        "corrected_released_index_to_model_category": {
            str(key): value for key, value in sorted(mapping.items())
        },
        "UAV_test_box_label_content_accessed_before_amendment": False,
        "test_metric_accessed_before_amendment": False,
        "prediction_recomputed_or_changed": False,
        "outcome_dependent_branch": False,
    }
    atomic_write_json(AMENDMENT, payload)
    atomic_write_json(
        AMENDED,
        {"status": payload["status"], "amendment_sha256": sha256_file(AMENDMENT)},
    )
    return payload


def corrected_convert_uav_labels() -> dict[str, Any]:
    amendment = amend()
    if parent.LABEL_LOCK.exists():
        lock = parent._load_mapping(parent.LABEL_LOCK)
        if lock.get("annotation_sha256") != sha256_file(parent.UAV_ANNOTATION) or lock.get(
            "class_mapping_amendment_sha256"
        ) != sha256_file(AMENDMENT):
            raise FinalTestAmendmentError("corrected UAV annotation lock changed")
        return lock
    if parent.UAV_ANNOTATION.exists() or parent.METRICS.exists():
        raise FinalTestAmendmentError("partial corrected label conversion requires audit")
    view_lock = parent._load_mapping(parent.VIEW_LOCK)
    raw_rows = view_lock.get("rows")
    if not isinstance(raw_rows, list):
        raise FinalTestAmendmentError("UAV view registry changed")
    original_rows = sorted(
        (row for row in raw_rows if isinstance(row, dict) and row.get("view") == "original"),
        key=lambda row: int(row["image_id"]),
    )
    images = [
        {
            "id": int(row["image_id"]),
            "file_name": PurePosixPath(str(row["member"])).name,
            "width": int(row["width"]),
            "height": int(row["height"]),
        }
        for row in original_rows
    ]
    annotations: list[dict[str, Any]] = []
    class_counts = {name: 0 for name in parent.CLASS_NAMES}
    source_formats: dict[str, int] = {"xywh": 0, "polygon": 0}
    with zipfile.ZipFile(parent.UAV_ARCHIVE) as bundle:
        names = _ordered_names(bundle)
        released_to_category = _released_to_model_category(names)
        category_to_name = {index + 1: name for index, name in enumerate(parent.CLASS_NAMES)}
        archive_names = set(bundle.namelist())
        annotation_id = 1
        for row in original_rows:
            image_id = int(row["image_id"])
            width = int(row["width"])
            height = int(row["height"])
            stem = PurePosixPath(str(row["member"])).stem
            label_member = f"UAV-OBB/test/labels/{stem}.txt"
            if label_member not in archive_names:
                raise FinalTestAmendmentError(f"UAV test label is missing: {label_member}")
            text = bundle.read(label_member).decode("utf-8")
            for line_number, line in enumerate(text.splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    values = [float(value) for value in line.split()]
                except ValueError as exc:
                    raise FinalTestAmendmentError(
                        f"invalid UAV label value: {label_member}:{line_number}"
                    ) from exc
                if len(values) not in {5, 9}:
                    raise FinalTestAmendmentError(
                        f"unsupported UAV label geometry: {label_member}:{line_number}"
                    )
                released_class = int(values[0])
                if float(released_class) != values[0]:
                    raise FinalTestAmendmentError(
                        f"noninteger UAV class: {label_member}:{line_number}"
                    )
                category_id = released_to_category.get(released_class)
                if category_id is None:
                    continue
                coords = values[1:]
                if len(coords) == 4:
                    center_x, center_y, box_w, box_h = coords
                    x1 = center_x - box_w / 2.0
                    y1 = center_y - box_h / 2.0
                    x2 = center_x + box_w / 2.0
                    y2 = center_y + box_h / 2.0
                    source_formats["xywh"] += 1
                else:
                    xs = coords[0::2]
                    ys = coords[1::2]
                    x1, x2 = min(xs), max(xs)
                    y1, y2 = min(ys), max(ys)
                    source_formats["polygon"] += 1
                x1 = min(max(x1, 0.0), 1.0)
                y1 = min(max(y1, 0.0), 1.0)
                x2 = min(max(x2, 0.0), 1.0)
                y2 = min(max(y2, 0.0), 1.0)
                if x2 <= x1 or y2 <= y1:
                    continue
                bbox = [
                    x1 * width,
                    y1 * height,
                    (x2 - x1) * width,
                    (y2 - y1) * height,
                ]
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 0,
                    }
                )
                class_counts[category_to_name[category_id]] += 1
                annotation_id += 1
    atomic_write_json(
        parent.UAV_ANNOTATION,
        {
            "info": {
                "description": "UAV-OBB official test, shared classes, HBB enclosure",
                "conversion": ("released class-name mapping plus xywh or polygon geometry to HBB"),
                "class_mapping_amendment_sha256": sha256_file(AMENDMENT),
            },
            "images": images,
            "annotations": annotations,
            "categories": [
                {"id": index + 1, "name": name} for index, name in enumerate(parent.CLASS_NAMES)
            ],
        },
    )
    payload = {
        "schema_version": 2,
        "status": "UAV_OBB_TEST_LABELS_CORRECTLY_CONVERTED_AFTER_PREDICTION_LOCK",
        "locked_at_utc": _now(),
        "prediction_lock_sha256": sha256_file(parent.PREDICTION_LOCK),
        "class_mapping_amendment_sha256": sha256_file(AMENDMENT),
        "archive_sha256": parent.UAV_ARCHIVE_SHA256,
        "annotation": parent._relative(parent.UAV_ANNOTATION),
        "annotation_sha256": sha256_file(parent.UAV_ANNOTATION),
        "images": len(images),
        "annotations": len(annotations),
        "class_counts": class_counts,
        "source_formats": source_formats,
        "released_names": names,
        "released_index_to_model_category": {
            str(key): value for key, value in sorted(released_to_category.items())
        },
        "prediction_or_metric_recomputed_after_label_access": False,
        "amendment_status": amendment["status"],
    }
    atomic_write_json(parent.LABEL_LOCK, payload)
    return payload


parent.convert_uav_labels = corrected_convert_uav_labels


def main() -> int:
    parser = argparse.ArgumentParser(description="Run corrected CVBRA-v1 final test v1.1")
    parser.add_argument(
        "--stage",
        choices=("amend", "convert", "points", "statistics", "all"),
        default="all",
    )
    args = parser.parse_args()
    if args.stage == "amend":
        result = amend()
    elif args.stage == "convert":
        result = corrected_convert_uav_labels()
    elif args.stage == "points":
        result = parent.points()
    elif args.stage == "statistics":
        result = parent.statistics()
    else:
        result = parent.finalize()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
