from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from scripts.unlock_sava_v1_uav_obb_development_A_labels import (
    COCO_CATEGORY_ID,
    IGNORED_NAMES,
    SCORED_NAMES,
    _class_names,
    _decode_label,
    _label_member,
)

from buse_uav.utils.hashing import sha256_file, stable_hash
from buse_uav.utils.io import atomic_write_json

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "experiment" / "cvbra_v1_backup_protocol.yaml"
PROTOCOL_SHA256 = "65c26c8e96284af4f58c3536d4ef81f5531ccc9c0455d2286cbd84d228eaf6da"
ARCHIVE = ROOT / "data" / "raw" / "UAV_OBB_v4" / "downloads" / "UAV-OBB-dlaCi7.zip"
ARCHIVE_SHA256 = "bf65b0d6a00bc1a320002012146bf44913e2b3f97b9e6acbd7954775d61f69e2"
PARTITION_LOCK = ROOT / "reports" / "data_metadata" / "uav_obb_v4" / "train_partition_lock.json"
PARTITION_LOCK_SHA256 = "9c4a1052d0030f6b909305cbbb953e96baa5384e2d7b6133200828ff5a767e27"
COORDINATE_AMENDMENT = (
    ROOT
    / "reports"
    / "development"
    / "sava_v1"
    / "uav_obb_development_A"
    / "annotations"
    / "LABEL_COORDINATE_DOMAIN_AMENDMENT_1.json"
)
COORDINATE_AMENDMENT_SHA256 = "e6607717490b8882d640aa6d924fc61e9d93222100053704ae463951cd42ba58"
RESERVE_ROOT = ROOT / "reports" / "development" / "cvbra_v1" / "uav_obb_reserve_B"
MATERIALIZATION_LOCK = RESERVE_ROOT / "view_materialization_lock.json"
PREDICTION_LOCK = RESERVE_ROOT / "evaluation" / "joint_prediction_lock.json"
PREDICTION_MARKER = RESERVE_ROOT / "evaluation" / "PREDICTIONS_LOCKED"
LABEL_ROOT = RESERVE_ROOT / "annotations"
PREOPEN_LOCK = LABEL_ROOT / "preopen_authorization.json"
PREOPEN_MARKER = LABEL_ROOT / "PREOPEN_AUTHORIZED"
ANNOTATION = LABEL_ROOT / "reserve_B_exact_car_truck_bus_hbb.coco.json"
CONVERSION_LOCK = LABEL_ROOT / "conversion_lock.json"
CONVERSION_MARKER = LABEL_ROOT / "LABELS_CONVERTED_AND_LOCKED"
ERROR = LABEL_ROOT / "CONVERSION_ERROR.json"
IMAGES = 483


class CVBRAReserveLabelError(RuntimeError):
    """Raised when prediction-gated reserve-B label conversion fails closed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CVBRAReserveLabelError(f"cannot parse mapping {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CVBRAReserveLabelError(f"expected mapping: {path}")
    return value


def _assert_hash(path: Path, expected: object, *, label: str) -> None:
    if not path.is_file():
        raise CVBRAReserveLabelError(f"missing locked {label}: {path}")
    observed = sha256_file(path)
    if observed != str(expected):
        raise CVBRAReserveLabelError(
            f"locked {label} changed: expected {expected}, observed {observed}"
        )


def _validate_prerequisites() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    _assert_hash(PROTOCOL, PROTOCOL_SHA256, label="CVBRA protocol")
    _assert_hash(ARCHIVE, ARCHIVE_SHA256, label="UAV-OBB archive")
    _assert_hash(PARTITION_LOCK, PARTITION_LOCK_SHA256, label="UAV-OBB partition")
    _assert_hash(
        COORDINATE_AMENDMENT,
        COORDINATE_AMENDMENT_SHA256,
        label="finite-coordinate conversion amendment",
    )
    partition = _load_mapping(PARTITION_LOCK)
    materialization = _load_mapping(MATERIALIZATION_LOCK)
    prediction = _load_mapping(PREDICTION_LOCK)
    marker = _load_mapping(PREDICTION_MARKER)
    assignments = partition.get("assignments")
    if (
        partition.get("pass") is not True
        or not isinstance(assignments, list)
        or sum(isinstance(row, dict) and row.get("role") == "reserve_B" for row in assignments)
        != IMAGES
        or partition.get("validation_or_test_content_accessed") is not False
    ):
        raise CVBRAReserveLabelError("reserve-B partition evidence changed")
    if (
        materialization.get("status") != "CVBRA_UAV_OBB_RESERVE_B_VIEWS_MATERIALIZED_AND_LOCKED"
        or materialization.get("images") != IMAGES
        or materialization.get("annotation_content_accessed") is not False
        or materialization.get("official_validation_or_test_content_accessed") is not False
    ):
        raise CVBRAReserveLabelError("reserve-B materialization evidence changed")
    artifacts = prediction.get("artifacts")
    if (
        prediction.get("status")
        != "CVBRA_V1_UAV_OBB_RESERVE_B_PREDICTIONS_LOCKED_BEFORE_ANNOTATIONS_OR_METRICS"
        or marker.get("joint_prediction_lock_sha256") != sha256_file(PREDICTION_LOCK)
        or prediction.get("images") != IMAGES
        or prediction.get("reserve_B_annotation_content_accessed") is not False
        or prediction.get("aggregate_metrics_accessed") is not False
        or prediction.get("official_validation_or_test_accessed") is not False
        or not isinstance(artifacts, list)
        or len(artifacts) != 12
    ):
        raise CVBRAReserveLabelError("reserve-B joint prediction evidence changed")
    for row in artifacts:
        if not isinstance(row, dict):
            raise CVBRAReserveLabelError("invalid reserve-B prediction artifact")
        _assert_hash(ROOT / str(row["prediction"]), row["prediction_sha256"], label="prediction")
    return partition, materialization, prediction


def _preopen_lock() -> dict[str, Any]:
    _, materialization, _ = _validate_prerequisites()
    if PREOPEN_LOCK.exists() or PREOPEN_MARKER.exists():
        if not PREOPEN_LOCK.is_file() or not PREOPEN_MARKER.is_file():
            raise CVBRAReserveLabelError("reserve-B pre-open lock is incomplete")
        lock = _load_mapping(PREOPEN_LOCK)
        marker = _load_mapping(PREOPEN_MARKER)
        if (
            lock.get("runner_sha256") != sha256_file(Path(__file__))
            or lock.get("joint_prediction_lock_sha256") != sha256_file(PREDICTION_LOCK)
            or marker.get("preopen_authorization_sha256") != sha256_file(PREOPEN_LOCK)
        ):
            raise CVBRAReserveLabelError("reserve-B pre-open lock changed")
        return lock
    if any(path.exists() for path in (ANNOTATION, CONVERSION_LOCK, CONVERSION_MARKER)):
        raise CVBRAReserveLabelError("reserve-B annotation appeared before pre-open lock")
    payload = {
        "schema_version": 1,
        "status": "AUTHORIZED_TO_OPEN_ONLY_UAV_OBB_RESERVE_B_TRAIN_LABELS",
        "authorized_at_utc": _utc_now(),
        "protocol_sha256": PROTOCOL_SHA256,
        "partition_lock_sha256": PARTITION_LOCK_SHA256,
        "materialization_lock_sha256": sha256_file(MATERIALIZATION_LOCK),
        "joint_prediction_lock_sha256": sha256_file(PREDICTION_LOCK),
        "coordinate_amendment_sha256": COORDINATE_AMENDMENT_SHA256,
        "runner": _relative(Path(__file__)),
        "runner_sha256": sha256_file(Path(__file__)),
        "reserve_B_labels_authorized": IMAGES,
        "development_A_labels_newly_accessed": 0,
        "validation_or_test_labels_authorized": 0,
        "aggregate_metrics_accessed": False,
        "annotation_content_accessed_before_authorization": False,
        "source_clean_rows_payload_sha256": materialization["clean_rows_payload_sha256"],
    }
    atomic_write_json(PREOPEN_LOCK, payload)
    atomic_write_json(
        PREOPEN_MARKER,
        {"status": payload["status"], "preopen_authorization_sha256": sha256_file(PREOPEN_LOCK)},
    )
    return payload


def _validate_conversion_lock() -> dict[str, Any]:
    _preopen_lock()
    if not CONVERSION_LOCK.is_file() or not CONVERSION_MARKER.is_file():
        raise CVBRAReserveLabelError("reserve-B conversion lock is incomplete")
    lock = _load_mapping(CONVERSION_LOCK)
    marker = _load_mapping(CONVERSION_MARKER)
    if (
        lock.get("status") != "CVBRA_UAV_OBB_RESERVE_B_LABELS_CONVERTED_AND_LOCKED"
        or lock.get("runner_sha256") != sha256_file(Path(__file__))
        or lock.get("joint_prediction_lock_sha256") != sha256_file(PREDICTION_LOCK)
        or lock.get("annotation_sha256") != sha256_file(ANNOTATION)
        or marker.get("conversion_lock_sha256") != sha256_file(CONVERSION_LOCK)
        or lock.get("official_validation_or_test_content_accessed") is not False
    ):
        raise CVBRAReserveLabelError("reserve-B conversion lock changed")
    return lock


def convert() -> dict[str, Any]:
    partition, materialization, _ = _validate_prerequisites()
    _preopen_lock()
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
        or any(role_by_member.get(str(row["member"])) != "reserve_B" for row in rows)
    ):
        raise CVBRAReserveLabelError("reserve-B clean-row identity changed")
    try:
        with zipfile.ZipFile(ARCHIVE) as bundle:
            member_names = set(bundle.namelist())
            if "UAV-OBB/data.yaml" not in member_names:
                raise CVBRAReserveLabelError("UAV-OBB data.yaml membership changed")
            data_yaml_raw = bundle.read("UAV-OBB/data.yaml")
            data_yaml = yaml.safe_load(data_yaml_raw.decode("utf-8-sig"))
            if not isinstance(data_yaml, dict):
                raise CVBRAReserveLabelError("UAV-OBB data.yaml is not a mapping")
            names = _class_names(data_yaml.get("names"))
            if set(names) != set(SCORED_NAMES + IGNORED_NAMES) or len(names) != 6:
                raise CVBRAReserveLabelError(f"UAV-OBB ontology changed: {names}")
            annotations: list[dict[str, Any]] = []
            source_counts: Counter[str] = Counter()
            label_hash_rows: list[dict[str, Any]] = []
            annotation_id = 1
            for row in rows:
                label_member = _label_member(str(row["member"]))
                if label_member not in member_names:
                    raise CVBRAReserveLabelError(f"missing reserve-B label: {label_member}")
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
    except (OSError, UnicodeError, zipfile.BadZipFile, yaml.YAMLError, RuntimeError) as exc:
        atomic_write_json(
            ERROR,
            {
                "status": "CVBRA_UAV_OBB_RESERVE_B_LABEL_CONVERSION_ERROR",
                "failed_at_utc": _utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "official_validation_or_test_content_accessed": False,
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
            "description": "UAV-OBB v4 reserve-B exact car/truck/bus HBB evaluation",
            "source_doi": "10.17632/6snrjwcpkh.4",
            "conversion": "coordinatewise min/max enclosure without coordinate clipping",
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
        "status": "CVBRA_UAV_OBB_RESERVE_B_LABELS_CONVERTED_AND_LOCKED",
        "locked_at_utc": _utc_now(),
        "runner": _relative(Path(__file__)),
        "runner_sha256": sha256_file(Path(__file__)),
        "protocol_sha256": PROTOCOL_SHA256,
        "preopen_authorization_sha256": sha256_file(PREOPEN_LOCK),
        "joint_prediction_lock_sha256": sha256_file(PREDICTION_LOCK),
        "coordinate_amendment_sha256": COORDINATE_AMENDMENT_SHA256,
        "archive_sha256": ARCHIVE_SHA256,
        "source_data_yaml_sha256": hashlib.sha256(data_yaml_raw).hexdigest(),
        "source_class_order": list(names),
        "reserve_B_labels_read": len(label_hash_rows),
        "label_hash_rows_payload_sha256": stable_hash(label_hash_rows, length=64),
        "source_objects_by_exact_class": dict(sorted(source_counts.items())),
        "scored_objects_by_exact_class": dict(sorted(scored_counts.items())),
        "ignored_objects": sum(source_counts[name] for name in IGNORED_NAMES),
        "annotation": _relative(ANNOTATION),
        "annotation_sha256": sha256_file(ANNOTATION),
        "images": len(images),
        "annotations": len(annotations),
        "obb_to_hbb": "coordinatewise_minimum_and_maximum_enclosure",
        "vertex_coordinate_domain": "finite_values_preserved_without_clipping",
        "geometry_filtering_or_box_expansion": False,
        "aggregate_metrics_accessed": False,
        "official_validation_or_test_content_accessed": False,
        "paper_body_change_authorized": False,
    }
    atomic_write_json(CONVERSION_LOCK, payload)
    atomic_write_json(
        CONVERSION_MARKER,
        {"status": payload["status"], "conversion_lock_sha256": sha256_file(CONVERSION_LOCK)},
    )
    return payload


def main() -> int:
    argparse.ArgumentParser(
        description="Unlock only prediction-gated reserve-B labels"
    ).parse_args()
    result = convert()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
