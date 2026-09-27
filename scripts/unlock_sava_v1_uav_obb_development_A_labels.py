from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from buse_uav.utils.hashing import sha256_file, stable_hash
from buse_uav.utils.io import atomic_write_json

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "experiment" / "sava_v1_fresh_scope_protocol.yaml"
PROTOCOL_SHA256 = "5332649233d63cb8e0356f2e82bd87b461f2e0d6db5d10b60e9f8808b1aeb9d1"
ARCHIVE = ROOT / "data" / "raw" / "UAV_OBB_v4" / "downloads" / "UAV-OBB-dlaCi7.zip"
ARCHIVE_SHA256 = "bf65b0d6a00bc1a320002012146bf44913e2b3f97b9e6acbd7954775d61f69e2"
PARTITION_LOCK = ROOT / "reports" / "data_metadata" / "uav_obb_v4" / "train_partition_lock.json"
PARTITION_LOCK_SHA256 = "9c4a1052d0030f6b909305cbbb953e96baa5384e2d7b6133200828ff5a767e27"
DEVELOPMENT_ROOT = ROOT / "reports" / "development" / "sava_v1" / "uav_obb_development_A"
MATERIALIZATION_LOCK = DEVELOPMENT_ROOT / "view_materialization_lock.json"
PREDICTION_LOCK = DEVELOPMENT_ROOT / "evaluation" / "joint_prediction_lock.json"
PREDICTION_MARKER = DEVELOPMENT_ROOT / "evaluation" / "PREDICTIONS_LOCKED"
LABEL_ROOT = DEVELOPMENT_ROOT / "annotations"
PREOPEN_LOCK = LABEL_ROOT / "preopen_authorization.json"
PREOPEN_MARKER = LABEL_ROOT / "PREOPEN_AUTHORIZED"
ANNOTATION = LABEL_ROOT / "development_A_exact_car_truck_bus_hbb.coco.json"
CONVERSION_LOCK = LABEL_ROOT / "conversion_lock.json"
CONVERSION_MARKER = LABEL_ROOT / "LABELS_CONVERTED_AND_LOCKED"
ERROR = LABEL_ROOT / "CONVERSION_ERROR.json"
COORDINATE_AMENDMENT = LABEL_ROOT / "LABEL_COORDINATE_DOMAIN_AMENDMENT_1.json"

IMAGES = 900
WIDTH = 1920
HEIGHT = 1080
SCORED_NAMES = ("car", "truck", "bus")
IGNORED_NAMES = ("bike", "other_vehicle", "taxi")
COCO_CATEGORY_ID = {"car": 1, "truck": 2, "bus": 3}


class SAVADevelopmentLabelError(RuntimeError):
    """Raised when the prediction-gated development-label access fails closed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SAVADevelopmentLabelError(f"cannot parse mapping {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SAVADevelopmentLabelError(f"expected mapping: {path}")
    return value


def _assert_hash(path: Path, expected: object, *, label: str) -> None:
    if not path.is_file():
        raise SAVADevelopmentLabelError(f"missing locked {label}: {path}")
    observed = sha256_file(path)
    if observed != str(expected):
        raise SAVADevelopmentLabelError(
            f"locked {label} changed: expected {expected}, observed {observed}"
        )


def _validate_prerequisites() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    _assert_hash(PROTOCOL, PROTOCOL_SHA256, label="SAVA protocol")
    _assert_hash(ARCHIVE, ARCHIVE_SHA256, label="UAV-OBB v4 archive")
    _assert_hash(PARTITION_LOCK, PARTITION_LOCK_SHA256, label="UAV-OBB partition")
    partition = _load_mapping(PARTITION_LOCK)
    materialization = _load_mapping(MATERIALIZATION_LOCK)
    prediction = _load_mapping(PREDICTION_LOCK)
    marker = _load_mapping(PREDICTION_MARKER)
    assignments = partition.get("assignments")
    if (
        partition.get("pass") is not True
        or partition.get("annotation_content_accessed") is not False
        or partition.get("validation_or_test_content_accessed") is not False
        or not isinstance(assignments, list)
        or sum(isinstance(row, dict) and row.get("role") == "development_A" for row in assignments)
        != IMAGES
    ):
        raise SAVADevelopmentLabelError("UAV-OBB partition evidence changed")
    if (
        materialization.get("status") != "SAVA_UAV_OBB_DEVELOPMENT_A_VIEWS_MATERIALIZED_AND_LOCKED"
        or materialization.get("images") != IMAGES
        or materialization.get("annotation_content_accessed") is not False
        or materialization.get("reserve_B_content_accessed") is not False
        or materialization.get("validation_or_test_content_accessed") is not False
    ):
        raise SAVADevelopmentLabelError("SAVA materialization evidence changed")
    artifacts = prediction.get("artifacts")
    if (
        prediction.get("status")
        != "SAVA_V1_UAV_OBB_DEVELOPMENT_A_PREDICTIONS_LOCKED_BEFORE_ANNOTATIONS_OR_METRICS"
        or marker.get("joint_prediction_lock_sha256") != sha256_file(PREDICTION_LOCK)
        or prediction.get("images") != IMAGES
        or prediction.get("annotation_content_accessed") is not False
        or prediction.get("aggregate_metrics_accessed") is not False
        or prediction.get("reserve_B_or_validation_or_test_accessed") is not False
        or not isinstance(artifacts, list)
        or len(artifacts) != 12
    ):
        raise SAVADevelopmentLabelError("joint prediction evidence changed")
    for row in artifacts:
        if not isinstance(row, dict):
            raise SAVADevelopmentLabelError("invalid locked prediction artifact")
        path = ROOT / str(row["prediction"])
        _assert_hash(path, row["prediction_sha256"], label="derived prediction")
    return partition, materialization, prediction


def _create_or_validate_preopen_lock() -> dict[str, Any]:
    _, materialization, _ = _validate_prerequisites()
    if PREOPEN_LOCK.exists() or PREOPEN_MARKER.exists():
        if not PREOPEN_LOCK.is_file() or not PREOPEN_MARKER.is_file():
            raise SAVADevelopmentLabelError("label pre-open lock is incomplete")
        lock = _load_mapping(PREOPEN_LOCK)
        marker = _load_mapping(PREOPEN_MARKER)
        runner_changed = lock.get("runner_sha256") != sha256_file(Path(__file__))
        if runner_changed:
            _validate_coordinate_amendment()
        if (
            lock.get("joint_prediction_lock_sha256") != sha256_file(PREDICTION_LOCK)
            or lock.get("materialization_lock_sha256") != sha256_file(MATERIALIZATION_LOCK)
            or marker.get("preopen_authorization_sha256") != sha256_file(PREOPEN_LOCK)
        ):
            raise SAVADevelopmentLabelError("label pre-open lock changed")
        return lock
    if any(path.exists() for path in (ANNOTATION, CONVERSION_LOCK, CONVERSION_MARKER)):
        raise SAVADevelopmentLabelError("annotation output appeared before label pre-open lock")
    payload = {
        "schema_version": 1,
        "status": "AUTHORIZED_TO_OPEN_ONLY_UAV_OBB_DEVELOPMENT_A_TRAIN_LABELS",
        "authorized_at_utc": _utc_now(),
        "protocol_sha256": PROTOCOL_SHA256,
        "partition_lock_sha256": PARTITION_LOCK_SHA256,
        "materialization_lock_sha256": sha256_file(MATERIALIZATION_LOCK),
        "joint_prediction_lock_sha256": sha256_file(PREDICTION_LOCK),
        "runner": _relative(Path(__file__)),
        "runner_sha256": sha256_file(Path(__file__)),
        "development_A_labels_authorized": IMAGES,
        "reserve_B_labels_authorized": 0,
        "validation_labels_authorized": 0,
        "test_labels_authorized": 0,
        "aggregate_metrics_accessed": False,
        "annotation_content_accessed_before_authorization": False,
        "source_clean_rows_payload_sha256": materialization["clean_rows_payload_sha256"],
    }
    atomic_write_json(PREOPEN_LOCK, payload)
    atomic_write_json(
        PREOPEN_MARKER,
        {
            "status": payload["status"],
            "preopen_authorization_sha256": sha256_file(PREOPEN_LOCK),
        },
    )
    return payload


def _validate_coordinate_amendment() -> dict[str, Any]:
    amendment = _load_mapping(COORDINATE_AMENDMENT)
    prerequisites = amendment.get("locked_prerequisites")
    replacement = amendment.get("replacement_rule")
    scope = amendment.get("scope")
    trigger = amendment.get("trigger")
    if not all(isinstance(value, dict) for value in (prerequisites, replacement, scope, trigger)):
        raise SAVADevelopmentLabelError("coordinate-domain amendment is incomplete")
    assert isinstance(prerequisites, dict)
    assert isinstance(replacement, dict)
    assert isinstance(scope, dict)
    assert isinstance(trigger, dict)
    if (
        amendment.get("status")
        != "REGISTERED_AFTER_FIRST_LABEL_PARSE_ERROR_BEFORE_ANY_AGGREGATE_METRIC"
        or prerequisites.get("joint_prediction_lock_sha256") != sha256_file(PREDICTION_LOCK)
        or prerequisites.get("preopen_authorization_sha256") != sha256_file(PREOPEN_LOCK)
        or trigger.get("conversion_error_sha256") != sha256_file(ERROR)
        or replacement.get("accept_only_finite_vertex_coordinates") is not True
        or replacement.get("require_vertex_coordinates_in_closed_unit_interval") is not False
        or replacement.get("coordinate_clipping") is not False
        or replacement.get("geometry_filtering") is not False
        or replacement.get("box_expansion") is not False
        or scope.get("development_A_labels_only") is not True
        or scope.get("prediction_files_or_settings_changed") is not False
        or scope.get("model_or_method_changed") is not False
        or scope.get("class_ontology_changed") is not False
        or scope.get("reserve_B_labels_authorized") is not False
        or scope.get("validation_or_test_content_authorized") is not False
        or amendment.get("aggregate_metrics_accessed_before_registration") is not False
    ):
        raise SAVADevelopmentLabelError("coordinate-domain amendment changed")
    return amendment


def _class_names(value: object) -> tuple[str, ...]:
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return tuple(value)
    if isinstance(value, dict):
        try:
            ordered = sorted((int(key), str(name)) for key, name in value.items())
        except (TypeError, ValueError) as exc:
            raise SAVADevelopmentLabelError("invalid class-name mapping") from exc
        if [index for index, _ in ordered] != list(range(len(ordered))):
            raise SAVADevelopmentLabelError("class-name indices are not contiguous")
        return tuple(name for _, name in ordered)
    raise SAVADevelopmentLabelError("UAV-OBB data.yaml has invalid class names")


def _label_member(image_member: str) -> str:
    pure = PurePosixPath(image_member)
    parts = list(pure.parts)
    try:
        index = parts.index("images")
    except ValueError as exc:
        raise SAVADevelopmentLabelError(
            f"image member lacks images directory: {image_member}"
        ) from exc
    parts[index] = "labels"
    return str(PurePosixPath(*parts).with_suffix(".txt"))


def _decode_label(
    raw: bytes,
    *,
    image_id: int,
    member: str,
    names: tuple[str, ...],
    annotation_id_start: int,
) -> tuple[list[dict[str, Any]], Counter[str], int]:
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise SAVADevelopmentLabelError(f"label is not UTF-8: {member}") from exc
    annotations: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    annotation_id = annotation_id_start
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        fields = stripped.split()
        if len(fields) != 9:
            raise SAVADevelopmentLabelError(f"invalid OBB field count at {member}:{line_number}")
        try:
            class_value = float(fields[0])
            coordinates = [float(value) for value in fields[1:]]
        except ValueError as exc:
            raise SAVADevelopmentLabelError(
                f"non-numeric OBB row at {member}:{line_number}"
            ) from exc
        class_id = int(class_value)
        if class_value != float(class_id) or not 0 <= class_id < len(names):
            raise SAVADevelopmentLabelError(f"invalid OBB class at {member}:{line_number}")
        if not all(math.isfinite(value) for value in coordinates):
            raise SAVADevelopmentLabelError(f"nonfinite OBB coordinate at {member}:{line_number}")
        class_name = names[class_id]
        if class_name not in SCORED_NAMES + IGNORED_NAMES:
            raise SAVADevelopmentLabelError(f"unregistered UAV-OBB class: {class_name}")
        counts[class_name] += 1
        if class_name in IGNORED_NAMES:
            continue
        xs = [coordinates[index] * WIDTH for index in (0, 2, 4, 6)]
        ys = [coordinates[index] * HEIGHT for index in (1, 3, 5, 7)]
        left, right = min(xs), max(xs)
        top, bottom = min(ys), max(ys)
        width, height = right - left, bottom - top
        if width <= 0.0 or height <= 0.0:
            raise SAVADevelopmentLabelError(f"degenerate OBB enclosure at {member}:{line_number}")
        annotations.append(
            {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": COCO_CATEGORY_ID[class_name],
                "bbox": [left, top, width, height],
                "area": width * height,
                "iscrowd": 0,
            }
        )
        annotation_id += 1
    return annotations, counts, annotation_id


def _validate_conversion_lock() -> dict[str, Any]:
    _create_or_validate_preopen_lock()
    if not CONVERSION_LOCK.is_file() or not CONVERSION_MARKER.is_file():
        raise SAVADevelopmentLabelError("label conversion lock is incomplete")
    lock = _load_mapping(CONVERSION_LOCK)
    marker = _load_mapping(CONVERSION_MARKER)
    if (
        lock.get("status") != "SAVA_UAV_OBB_DEVELOPMENT_A_LABELS_CONVERTED_AND_LOCKED"
        or lock.get("runner_sha256") != sha256_file(Path(__file__))
        or lock.get("preopen_authorization_sha256") != sha256_file(PREOPEN_LOCK)
        or lock.get("joint_prediction_lock_sha256") != sha256_file(PREDICTION_LOCK)
        or lock.get("annotation_sha256") != sha256_file(ANNOTATION)
        or lock.get("coordinate_amendment_sha256") != sha256_file(COORDINATE_AMENDMENT)
        or marker.get("conversion_lock_sha256") != sha256_file(CONVERSION_LOCK)
        or lock.get("reserve_B_labels_accessed") is not False
        or lock.get("validation_or_test_content_accessed") is not False
    ):
        raise SAVADevelopmentLabelError("label conversion lock changed")
    return lock


def convert() -> dict[str, Any]:
    partition, materialization, _ = _validate_prerequisites()
    _create_or_validate_preopen_lock()
    amendment = _validate_coordinate_amendment()
    if CONVERSION_LOCK.exists() or CONVERSION_MARKER.exists():
        return _validate_conversion_lock()
    assignments = partition["assignments"]
    clean_rows = materialization["clean_rows"]
    assert isinstance(assignments, list)
    assert isinstance(clean_rows, list)
    role_by_member = {
        str(row["member"]): str(row["role"]) for row in assignments if isinstance(row, dict)
    }
    rows = sorted(
        (row for row in clean_rows if isinstance(row, dict)),
        key=lambda row: int(row["image_id"]),
    )
    if (
        len(rows) != IMAGES
        or {int(row["image_id"]) for row in rows} != set(range(1, IMAGES + 1))
        or any(role_by_member.get(str(row["member"])) != "development_A" for row in rows)
    ):
        raise SAVADevelopmentLabelError("development-A clean-row identity changed")
    try:
        with zipfile.ZipFile(ARCHIVE) as bundle:
            member_names = set(bundle.namelist())
            yaml_members = [name for name in member_names if name.casefold().endswith("/data.yaml")]
            if yaml_members != ["UAV-OBB/data.yaml"]:
                raise SAVADevelopmentLabelError("UAV-OBB data.yaml membership changed")
            data_yaml_raw = bundle.read(yaml_members[0])
            try:
                data_yaml = yaml.safe_load(data_yaml_raw.decode("utf-8-sig"))
            except (UnicodeDecodeError, yaml.YAMLError) as exc:
                raise SAVADevelopmentLabelError("cannot parse UAV-OBB data.yaml") from exc
            if not isinstance(data_yaml, dict):
                raise SAVADevelopmentLabelError("UAV-OBB data.yaml is not a mapping")
            names = _class_names(data_yaml.get("names"))
            if set(names) != set(SCORED_NAMES + IGNORED_NAMES) or len(names) != 6:
                raise SAVADevelopmentLabelError(f"UAV-OBB ontology changed: {names}")
            annotations: list[dict[str, Any]] = []
            source_counts: Counter[str] = Counter()
            label_hash_rows: list[dict[str, Any]] = []
            annotation_id = 1
            for row in rows:
                image_member = str(row["member"])
                label_member = _label_member(image_member)
                if label_member not in member_names:
                    raise SAVADevelopmentLabelError(f"missing development label: {label_member}")
                raw = bundle.read(label_member)
                converted, counts, annotation_id = _decode_label(
                    raw,
                    image_id=int(row["image_id"]),
                    member=label_member,
                    names=names,
                    annotation_id_start=annotation_id,
                )
                annotations.extend(converted)
                source_counts.update(counts)
                label_hash_rows.append(
                    {
                        "image_id": int(row["image_id"]),
                        "member": label_member,
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "bytes": len(raw),
                    }
                )
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        atomic_write_json(
            ERROR,
            {
                "status": "SAVA_UAV_OBB_DEVELOPMENT_A_LABEL_CONVERSION_ERROR",
                "failed_at_utc": _utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "reserve_B_labels_accessed": False,
                "validation_or_test_content_accessed": False,
            },
        )
        raise
    images = [
        {
            "id": int(row["image_id"]),
            "file_name": PurePosixPath(str(row["member"])).name,
            "width": int(row["width"]),
            "height": int(row["height"]),
        }
        for row in rows
    ]
    coco = {
        "info": {
            "description": "UAV-OBB v4 development-A exact car/truck/bus HBB evaluation",
            "source_doi": "10.17632/6snrjwcpkh.4",
            "conversion": "coordinatewise min/max enclosure of registered YOLOv8 OBB vertices",
        },
        "licenses": [{"id": 1, "name": "CC BY 4.0"}],
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": category_id, "name": name, "supercategory": "vehicle"}
            for name, category_id in COCO_CATEGORY_ID.items()
        ],
    }
    atomic_write_json(ANNOTATION, coco)
    scored_counts = Counter(
        next(
            name
            for name, category_id in COCO_CATEGORY_ID.items()
            if category_id == row["category_id"]
        )
        for row in annotations
    )
    payload = {
        "schema_version": 1,
        "status": "SAVA_UAV_OBB_DEVELOPMENT_A_LABELS_CONVERTED_AND_LOCKED",
        "locked_at_utc": _utc_now(),
        "runner": _relative(Path(__file__)),
        "runner_sha256": sha256_file(Path(__file__)),
        "protocol_sha256": PROTOCOL_SHA256,
        "preopen_authorization_sha256": sha256_file(PREOPEN_LOCK),
        "coordinate_amendment": _relative(COORDINATE_AMENDMENT),
        "coordinate_amendment_sha256": sha256_file(COORDINATE_AMENDMENT),
        "joint_prediction_lock_sha256": sha256_file(PREDICTION_LOCK),
        "archive_sha256": ARCHIVE_SHA256,
        "source_data_yaml_sha256": hashlib.sha256(data_yaml_raw).hexdigest(),
        "source_class_order": list(names),
        "development_A_labels_read": len(label_hash_rows),
        "development_A_label_hash_rows_payload_sha256": stable_hash(label_hash_rows, length=64),
        "source_objects_by_exact_class": dict(sorted(source_counts.items())),
        "scored_objects_by_exact_class": dict(sorted(scored_counts.items())),
        "ignored_objects": sum(source_counts[name] for name in IGNORED_NAMES),
        "annotation": _relative(ANNOTATION),
        "annotation_sha256": sha256_file(ANNOTATION),
        "images": len(images),
        "annotations": len(annotations),
        "ontology": {
            "scored": list(SCORED_NAMES),
            "ignored": list(IGNORED_NAMES),
            "class_merging": False,
        },
        "obb_to_hbb": "coordinatewise_minimum_and_maximum_enclosure",
        "vertex_coordinate_domain": "finite_values_preserved_without_clipping",
        "geometry_filtering_or_box_expansion": False,
        "aggregate_metrics_accessed": False,
        "reserve_B_labels_accessed": False,
        "validation_or_test_content_accessed": False,
        "paper_body_change_authorized": False,
        "aggregate_metrics_accessed_before_coordinate_amendment": amendment[
            "aggregate_metrics_accessed_before_registration"
        ],
    }
    atomic_write_json(CONVERSION_LOCK, payload)
    atomic_write_json(
        CONVERSION_MARKER,
        {
            "status": payload["status"],
            "conversion_lock_sha256": sha256_file(CONVERSION_LOCK),
        },
    )
    return payload


def main() -> int:
    argparse.ArgumentParser(
        description="Unlock and convert only prediction-gated UAV-OBB development-A labels"
    ).parse_args()
    result = convert()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
