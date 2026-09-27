from __future__ import annotations

import argparse
import csv
import json
import math
import platform
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

from buse_uav.evaluation.bootstrap import paired_coco_ap_bootstrap
from buse_uav.evaluation.coco import evaluate_coco
from buse_uav.evaluation.supplementary import bootstrap_sign_pvalue, holm_adjust
from buse_uav.utils.hashing import sha256_file
from buse_uav.utils.io import atomic_write_json, atomic_write_text

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "experiment" / "cvbra_v1_hazydet_source_retention_v2.yaml"
PARENT_PROTOCOL = ROOT / "configs" / "experiment" / "cvbra_v1_hazydet_source_retention.yaml"
PARENT_RUNNER = ROOT / "scripts" / "run_cvbra_v1_hazydet_source_retention.py"
ANNOTATION = ROOT / "data" / "raw" / "HazyDet" / "val" / "val_coco.json"
PARENT_OUTPUT = (
    ROOT
    / "reports"
    / "development"
    / "cvbra_v1_matched_baselines"
    / "hazydet_source_retention"
)
OUTPUT = (
    ROOT
    / "reports"
    / "development"
    / "cvbra_v1_matched_baselines"
    / "hazydet_source_retention_v2"
)

PARENT_PROTOCOL_SHA256 = "501d1e4605b295faafda78786cb6b807ae68f58c06ebe8352bad2daba7eb723c"
PARENT_RUNNER_SHA256 = "115a1ee35740ec4b071684fefc285307012b4d68d4e99e2f1069a240329ab177"
PARENT_PREDICTION_LOCK_SHA256 = (
    "28e7adfbdf71a2a76d883beaaeaeffdb95df322d54c5f3366d7d44d20f82999c"
)
INVALID_METRICS_SHA256 = "60f626030a908ed64ef99d103bc33c86d3302c97a9f782d38c6265ec9712cfc9"
INVALID_BOOTSTRAP_SHA256 = "d7dbe31c97ff9dfa4bb596d77c8bc1c00cbe229b2296f56e0020845b77ca12f3"
ANNOTATION_SHA256 = "2b2e39f7812631dfb4f3f0fbe1e743b65ca873151ca92d8e24ddbf0e9feacb9a"
HISTORICAL_SOURCE_AP = 0.5169972340948197
SOURCE_AP_MAX_ABS_DRIFT = 0.0025

MODELS = ("source", "CVBRA_v1", "STF", "CVBRA_noCV", "CVBRA_noReplay", "CVBRA_noFreeze")
WEIGHTS: dict[str, Path] = {
    "source": ROOT / "weights" / "hazydet" / "yolo11n_best.pt",
    "CVBRA_v1": ROOT / "runs" / "cvbra_v1" / "yolo11n" / "cvbra_v1.pt",
    **{
        model: ROOT / "runs" / "cvbra_v1_matched_baselines" / model / f"{model}.pt"
        for model in MODELS[2:]
    },
}

AMENDMENT = PARENT_OUTPUT / "MAPPING_FAILURE_AMENDMENT_1.json"
INVALID_MARKER = PARENT_OUTPUT / "INVALID_SCORING_ATTEMPT_1"
REGISTRATION = OUTPUT / "REGISTRATION_LOCK.json"
REGISTRATION_MARKER = OUTPUT / "REGISTRATION_LOCKED"
IMPLEMENTATION_LOCK = OUTPUT / "implementation_lock.json"
IMPLEMENTATION_MARKER = OUTPUT / "IMPLEMENTATION_LOCKED"
PREDICTION_LOCK = OUTPUT / "prediction_lock.json"
PREDICTION_MARKER = OUTPUT / "PREDICTIONS_LOCKED"
ANALYSIS_LOCK = OUTPUT / "analysis_authorization.json"
ANALYSIS_MARKER = OUTPUT / "ANALYSIS_AUTHORIZED"
METRICS = OUTPUT / "metrics.csv"
STATISTICS = OUTPUT / "paired_statistics.json"
REPORT = OUTPUT / "source_retention_report.json"
COMPLETE = OUTPUT / "SOURCE_RETENTION_COMPLETE"

IMAGES = 1000
MAX_DET = 500
RESAMPLES = 2000
SEED = 20260814
SOURCE_CATEGORY_IDS = (1, 2, 3)
TARGET_CATEGORY_IDS = (0, 1, 2)
METRIC_KEYS = ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large")


class SourceRetentionCorrectionError(RuntimeError):
    """Raised when the registered source-retention correction cannot fail closed."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _rooted(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SourceRetentionCorrectionError(f"cannot parse mapping {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SourceRetentionCorrectionError(f"expected mapping: {path}")
    return value


def _load_rows(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SourceRetentionCorrectionError(f"cannot parse rows {path}: {exc}") from exc
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise SourceRetentionCorrectionError(f"expected row list: {path}")
    return value


def _assert_hash(path: Path, expected: str, *, label: str) -> None:
    if not path.is_file() or sha256_file(path) != expected:
        raise SourceRetentionCorrectionError(f"locked {label} changed: {path}")


def _annotation_image_ids() -> tuple[int, ...]:
    annotation = _load_mapping(ANNOTATION)
    raw_images = annotation.get("images")
    raw_categories = annotation.get("categories")
    if not isinstance(raw_images, list) or not isinstance(raw_categories, list):
        raise SourceRetentionCorrectionError("HazyDet annotation structure changed")
    image_ids = tuple(
        int(row["id"])
        for row in raw_images
        if isinstance(row, dict) and isinstance(row.get("id"), int)
    )
    category_ids = tuple(
        int(row["id"])
        for row in raw_categories
        if isinstance(row, dict) and isinstance(row.get("id"), int)
    )
    if len(image_ids) != IMAGES or len(set(image_ids)) != IMAGES:
        raise SourceRetentionCorrectionError("HazyDet annotation image scope changed")
    if category_ids != TARGET_CATEGORY_IDS:
        raise SourceRetentionCorrectionError("HazyDet category IDs changed")
    return image_ids


def correct_prediction_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Apply the registered 1-based to 0-based category correction only."""
    corrected: list[dict[str, Any]] = []
    for index, source in enumerate(rows):
        if not all(key in source for key in ("image_id", "category_id", "bbox", "score")):
            raise SourceRetentionCorrectionError(f"prediction row {index} lacks required fields")
        raw_category = source["category_id"]
        if isinstance(raw_category, bool) or not isinstance(raw_category, int):
            raise SourceRetentionCorrectionError(f"prediction row {index} category is not integer")
        if raw_category not in SOURCE_CATEGORY_IDS:
            raise SourceRetentionCorrectionError(
                f"prediction row {index} has unregistered category {raw_category}"
            )
        target = dict(source)
        target["category_id"] = raw_category - 1
        corrected.append(target)
    if not corrected:
        raise SourceRetentionCorrectionError("prediction correction received no rows")
    return corrected


def _correction_audit(
    source_rows: Sequence[Mapping[str, Any]],
    corrected_rows: Sequence[Mapping[str, Any]],
    *,
    annotation_image_ids: set[int],
) -> dict[str, Any]:
    if len(source_rows) != len(corrected_rows):
        raise SourceRetentionCorrectionError("category correction changed the row count")
    source_counts: Counter[int] = Counter()
    target_counts: Counter[int] = Counter()
    predicted_ids: set[int] = set()
    for index, (source, target) in enumerate(zip(source_rows, corrected_rows, strict=True)):
        if set(source) != set(target):
            raise SourceRetentionCorrectionError(f"category correction changed keys at row {index}")
        source_category = int(source["category_id"])
        target_category = int(target["category_id"])
        if target_category != source_category - 1:
            raise SourceRetentionCorrectionError(f"category correction failed at row {index}")
        for key, value in source.items():
            if key != "category_id" and target[key] != value:
                raise SourceRetentionCorrectionError(
                    f"category correction changed immutable field {key} at row {index}"
                )
        image_id = int(source["image_id"])
        if image_id not in annotation_image_ids:
            raise SourceRetentionCorrectionError(f"prediction references unknown image {image_id}")
        predicted_ids.add(image_id)
        source_counts[source_category] += 1
        target_counts[target_category] += 1
    if set(source_counts) != set(SOURCE_CATEGORY_IDS):
        raise SourceRetentionCorrectionError(
            "parent predictions do not cover all source categories"
        )
    if set(target_counts) != set(TARGET_CATEGORY_IDS):
        raise SourceRetentionCorrectionError(
            "corrected predictions do not cover all target categories"
        )
    return {
        "rows": len(source_rows),
        "predicted_image_ids": len(predicted_ids),
        "annotation_image_ids": len(annotation_image_ids),
        "source_category_counts": {str(key): source_counts[key] for key in SOURCE_CATEGORY_IDS},
        "target_category_counts": {str(key): target_counts[key] for key in TARGET_CATEGORY_IDS},
        "immutable_fields_exact": True,
        "row_order_exact": True,
        "operation": "category_id_minus_one",
    }


def _parent_prediction_lock() -> dict[str, Any]:
    lock_path = PARENT_OUTPUT / "prediction_lock.json"
    _assert_hash(lock_path, PARENT_PREDICTION_LOCK_SHA256, label="parent prediction lock")
    lock = _load_mapping(lock_path)
    artifacts = lock.get("artifacts")
    if (
        lock.get("status") != "ALL_HAZYDET_SOURCE_RETENTION_PREDICTIONS_LOCKED_BEFORE_METRICS"
        or lock.get("models") != list(MODELS)
        or lock.get("metric_or_selection_feedback_used") is not False
        or not isinstance(artifacts, list)
        or len(artifacts) != len(MODELS)
    ):
        raise SourceRetentionCorrectionError("parent prediction lock scope changed")
    observed_models: list[str] = []
    for row in artifacts:
        if not isinstance(row, dict):
            raise SourceRetentionCorrectionError("parent prediction artifact is invalid")
        model = str(row.get("model"))
        observed_models.append(model)
        prediction_path = _rooted(row.get("prediction"))
        _assert_hash(prediction_path, str(row.get("prediction_sha256")), label=f"{model} raw")
        repeat_path = PARENT_OUTPUT / "raw" / model / "rep_2" / "predictions.coco.json"
        _assert_hash(repeat_path, str(row.get("prediction_sha256")), label=f"{model} repeat")
    if observed_models != list(MODELS):
        raise SourceRetentionCorrectionError("parent prediction model order changed")
    return lock


def _validate_parent_failure() -> dict[str, Any]:
    _assert_hash(PARENT_PROTOCOL, PARENT_PROTOCOL_SHA256, label="parent protocol")
    _assert_hash(PARENT_RUNNER, PARENT_RUNNER_SHA256, label="parent runner")
    _assert_hash(ANNOTATION, ANNOTATION_SHA256, label="HazyDet annotation")
    _assert_hash(PARENT_OUTPUT / "metrics.csv", INVALID_METRICS_SHA256, label="invalid metrics")
    _assert_hash(
        PARENT_OUTPUT / "bootstrap" / "CVBRA_v1_minus_source.json",
        INVALID_BOOTSTRAP_SHA256,
        label="invalid bootstrap",
    )
    image_ids = set(_annotation_image_ids())
    lock = _parent_prediction_lock()
    artifacts = lock["artifacts"]
    if not isinstance(artifacts, list):
        raise SourceRetentionCorrectionError("parent prediction artifacts changed")
    for row in artifacts:
        if not isinstance(row, dict):
            raise SourceRetentionCorrectionError("parent prediction row changed")
        source_rows = _load_rows(_rooted(row["prediction"]))
        corrected = correct_prediction_rows(source_rows)
        _correction_audit(source_rows, corrected, annotation_image_ids=image_ids)
    return lock


def _write_or_validate_amendment() -> dict[str, Any]:
    if AMENDMENT.exists() or INVALID_MARKER.exists():
        if not AMENDMENT.is_file() or not INVALID_MARKER.is_file():
            raise SourceRetentionCorrectionError("mapping-failure amendment is incomplete")
        amendment = _load_mapping(AMENDMENT)
        marker = _load_mapping(INVALID_MARKER)
        if (
            amendment.get("parent_prediction_lock_sha256") != PARENT_PREDICTION_LOCK_SHA256
            or amendment.get("invalid_metrics_sha256") != INVALID_METRICS_SHA256
            or amendment.get("invalid_bootstrap_sha256") != INVALID_BOOTSTRAP_SHA256
            or marker.get("amendment_sha256") != sha256_file(AMENDMENT)
        ):
            raise SourceRetentionCorrectionError("mapping-failure amendment changed")
        return amendment
    amendment = {
        "schema_version": 1,
        "status": "INVALIDATED_HAZYDET_SOURCE_RETENTION_SCORING_ATTEMPT_1",
        "recorded_at_utc": _now(),
        "parent_protocol_sha256": PARENT_PROTOCOL_SHA256,
        "parent_runner_sha256": PARENT_RUNNER_SHA256,
        "parent_prediction_lock_sha256": PARENT_PREDICTION_LOCK_SHA256,
        "invalid_metrics": _relative(PARENT_OUTPUT / "metrics.csv"),
        "invalid_metrics_sha256": INVALID_METRICS_SHA256,
        "invalid_bootstrap": _relative(
            PARENT_OUTPUT / "bootstrap" / "CVBRA_v1_minus_source.json"
        ),
        "invalid_bootstrap_sha256": INVALID_BOOTSTRAP_SHA256,
        "root_causes": [
            "predictions were exported with category IDs 1/2/3 while HazyDet uses 0/1/2",
            (
                "point metrics evaluated only image IDs present in predictions "
                "instead of all 1000 images"
            ),
        ],
        "scientific_disposition": "excluded_from_all_method_comparisons_and_claims",
        "preservation": "all invalid outputs and immutable raw predictions retained in place",
        "authorized_correction": {
            "protocol": _relative(PROTOCOL),
            "category_operation": "subtract exactly one; preserve every other field and row order",
            "evaluation_scope": "all annotation image IDs including zero-detection images",
            "raw_inference_rerun": False,
            "method_or_hyperparameter_selection": False,
        },
        "HazyDet_test_access": "prohibited",
        "UAV_OBB_test_access": "prohibited",
    }
    atomic_write_json(AMENDMENT, amendment)
    atomic_write_json(
        INVALID_MARKER,
        {"status": amendment["status"], "amendment_sha256": sha256_file(AMENDMENT)},
    )
    return amendment


def _registration() -> dict[str, Any]:
    protocol_sha256 = sha256_file(PROTOCOL)
    if REGISTRATION.exists() or REGISTRATION_MARKER.exists():
        if not REGISTRATION.is_file() or not REGISTRATION_MARKER.is_file():
            raise SourceRetentionCorrectionError("v2 registration is incomplete")
        lock = _load_mapping(REGISTRATION)
        marker = _load_mapping(REGISTRATION_MARKER)
        if (
            lock.get("protocol_sha256") != protocol_sha256
            or lock.get("parent_prediction_lock_sha256") != PARENT_PREDICTION_LOCK_SHA256
            or lock.get("corrected_metrics_before_registration") is not False
            or marker.get("registration_sha256") != sha256_file(REGISTRATION)
        ):
            raise SourceRetentionCorrectionError("v2 registration changed")
        return lock
    if any(path.exists() for path in (PREDICTION_LOCK, ANALYSIS_LOCK, METRICS, REPORT)):
        raise SourceRetentionCorrectionError("v2 output appeared before registration")
    lock = {
        "schema_version": 1,
        "status": "CVBRA_V1_HAZYDET_SOURCE_RETENTION_V2_REGISTERED",
        "registered_at_utc": _now(),
        "protocol": _relative(PROTOCOL),
        "protocol_sha256": protocol_sha256,
        "amendment_sha256": sha256_file(AMENDMENT),
        "parent_prediction_lock_sha256": PARENT_PREDICTION_LOCK_SHA256,
        "raw_predictions_already_available": True,
        "invalid_metrics_already_observed": True,
        "corrected_predictions_before_registration": False,
        "corrected_metrics_before_registration": False,
        "method_or_hyperparameter_selection": False,
        "category_mapping": {"1": 0, "2": 1, "3": 2},
        "evaluation_image_scope": "all_1000_annotation_image_ids",
        "models": list(MODELS),
        "HazyDet_test_access": "prohibited",
        "UAV_OBB_test_access": "prohibited",
    }
    atomic_write_json(REGISTRATION, lock)
    atomic_write_json(
        REGISTRATION_MARKER,
        {"status": lock["status"], "registration_sha256": sha256_file(REGISTRATION)},
    )
    return lock


def _implementation_lock() -> dict[str, Any]:
    _validate_parent_failure()
    _write_or_validate_amendment()
    registration = _registration()
    if IMPLEMENTATION_LOCK.exists() or IMPLEMENTATION_MARKER.exists():
        if not IMPLEMENTATION_LOCK.is_file() or not IMPLEMENTATION_MARKER.is_file():
            raise SourceRetentionCorrectionError("v2 implementation lock is incomplete")
        lock = _load_mapping(IMPLEMENTATION_LOCK)
        marker = _load_mapping(IMPLEMENTATION_MARKER)
        if (
            lock.get("runner_sha256") != sha256_file(Path(__file__))
            or lock.get("protocol_sha256") != sha256_file(PROTOCOL)
            or lock.get("registration_sha256") != sha256_file(REGISTRATION)
            or marker.get("implementation_lock_sha256") != sha256_file(IMPLEMENTATION_LOCK)
        ):
            raise SourceRetentionCorrectionError("v2 implementation lock changed")
        return lock
    if any(path.exists() for path in (PREDICTION_LOCK, ANALYSIS_LOCK, METRICS, REPORT)):
        raise SourceRetentionCorrectionError(
            "v2 downstream output appeared before implementation lock"
        )
    lock = {
        "schema_version": 1,
        "status": "CVBRA_V1_HAZYDET_SOURCE_RETENTION_V2_IMPLEMENTATION_LOCKED",
        "locked_at_utc": _now(),
        "runner": _relative(Path(__file__)),
        "runner_sha256": sha256_file(Path(__file__)),
        "protocol_sha256": sha256_file(PROTOCOL),
        "registration_sha256": sha256_file(REGISTRATION),
        "amendment_sha256": sha256_file(AMENDMENT),
        "parent_prediction_lock_sha256": PARENT_PREDICTION_LOCK_SHA256,
        "annotation_sha256": ANNOTATION_SHA256,
        "python": platform.python_version(),
        "models": list(MODELS),
        "images": len(_annotation_image_ids()),
        "category_correction": {"source": [1, 2, 3], "target": [0, 1, 2]},
        "immutable_fields": ["image_id", "bbox", "score", "row_order"],
        "evaluation_image_scope": "all_annotation_image_ids",
        "corrected_metric_or_selection_feedback_allowed": False,
        "validation_labels_previously_accessed": True,
        "HazyDet_test_access": "prohibited",
        "UAV_OBB_test_access": "prohibited",
        "registration_status": registration["status"],
    }
    atomic_write_json(IMPLEMENTATION_LOCK, lock)
    atomic_write_json(
        IMPLEMENTATION_MARKER,
        {
            "status": lock["status"],
            "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
        },
    )
    return lock


def preflight() -> dict[str, Any]:
    lock = _implementation_lock()
    return {
        "status": "PASS_CVBRA_V1_HAZYDET_SOURCE_RETENTION_V2_PREFLIGHT",
        "models": list(MODELS),
        "images": IMAGES,
        "runner_sha256": lock["runner_sha256"],
        "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
        "invalid_attempt_preserved": True,
        "corrected_metrics_accessed": False,
        "HazyDet_test_access": "prohibited",
    }


def _validate_prediction_lock() -> dict[str, Any]:
    _implementation_lock()
    if not PREDICTION_LOCK.is_file() or not PREDICTION_MARKER.is_file():
        raise SourceRetentionCorrectionError("v2 prediction lock is incomplete")
    lock = _load_mapping(PREDICTION_LOCK)
    marker = _load_mapping(PREDICTION_MARKER)
    artifacts = lock.get("artifacts")
    if (
        lock.get("status") != "ALL_CORRECTED_HAZYDET_SOURCE_RETENTION_PREDICTIONS_LOCKED"
        or lock.get("implementation_lock_sha256") != sha256_file(IMPLEMENTATION_LOCK)
        or lock.get("models") != list(MODELS)
        or not isinstance(artifacts, list)
        or len(artifacts) != len(MODELS)
        or marker.get("prediction_lock_sha256") != sha256_file(PREDICTION_LOCK)
    ):
        raise SourceRetentionCorrectionError("v2 prediction lock changed")
    for row in artifacts:
        if not isinstance(row, dict):
            raise SourceRetentionCorrectionError("v2 prediction artifact is invalid")
        _assert_hash(
            _rooted(row["corrected_prediction"]),
            str(row["corrected_prediction_sha256"]),
            label="corrected prediction",
        )
        _assert_hash(
            _rooted(row["transformation_audit"]),
            str(row["transformation_audit_sha256"]),
            label="transformation audit",
        )
    return lock


def materialize() -> dict[str, Any]:
    _implementation_lock()
    if PREDICTION_LOCK.exists() or PREDICTION_MARKER.exists():
        return _validate_prediction_lock()
    corrected_root = OUTPUT / "corrected_predictions"
    audit_root = OUTPUT / "transformation_audits"
    if corrected_root.exists() or audit_root.exists():
        raise SourceRetentionCorrectionError("partial v2 correction requires audit")
    parent = _parent_prediction_lock()
    raw_artifacts = parent.get("artifacts")
    timing = parent.get("timing_median_mean_ms")
    if not isinstance(raw_artifacts, list) or not isinstance(timing, dict):
        raise SourceRetentionCorrectionError("parent prediction metadata changed")
    annotation_ids = set(_annotation_image_ids())
    artifacts: list[dict[str, Any]] = []
    for row in raw_artifacts:
        if not isinstance(row, dict):
            raise SourceRetentionCorrectionError("parent prediction artifact changed")
        model = str(row["model"])
        source_path = _rooted(row["prediction"])
        source_rows = _load_rows(source_path)
        corrected_rows = correct_prediction_rows(source_rows)
        audit = _correction_audit(
            source_rows,
            corrected_rows,
            annotation_image_ids=annotation_ids,
        )
        corrected_path = corrected_root / f"{model}.coco.json"
        audit_path = audit_root / f"{model}.json"
        atomic_write_json(corrected_path, corrected_rows)
        atomic_write_json(
            audit_path,
            {
                "schema_version": 1,
                "status": "EXACT_CATEGORY_MAPPING_CORRECTION_VERIFIED",
                "model": model,
                "source_prediction": _relative(source_path),
                "source_prediction_sha256": sha256_file(source_path),
                "corrected_prediction": _relative(corrected_path),
                "corrected_prediction_sha256": sha256_file(corrected_path),
                **audit,
            },
        )
        artifacts.append(
            {
                "model": model,
                "source_prediction": _relative(source_path),
                "source_prediction_sha256": sha256_file(source_path),
                "corrected_prediction": _relative(corrected_path),
                "corrected_prediction_sha256": sha256_file(corrected_path),
                "transformation_audit": _relative(audit_path),
                "transformation_audit_sha256": sha256_file(audit_path),
                "mean_ms": float(timing[model]),
            }
        )
    lock = {
        "schema_version": 1,
        "status": "ALL_CORRECTED_HAZYDET_SOURCE_RETENTION_PREDICTIONS_LOCKED",
        "locked_at_utc": _now(),
        "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
        "parent_prediction_lock_sha256": PARENT_PREDICTION_LOCK_SHA256,
        "models": list(MODELS),
        "images": IMAGES,
        "artifacts": artifacts,
        "timing_median_mean_ms": timing,
        "correction": "category_id_minus_one_only",
        "row_order_and_other_fields_exact": True,
        "evaluation_image_scope": "all_1000_annotation_image_ids",
        "corrected_metrics_accessed": False,
        "method_or_hyperparameter_selection": False,
        "HazyDet_test_access": "prohibited",
        "UAV_OBB_test_access": "prohibited",
    }
    atomic_write_json(PREDICTION_LOCK, lock)
    atomic_write_json(
        PREDICTION_MARKER,
        {"status": lock["status"], "prediction_lock_sha256": sha256_file(PREDICTION_LOCK)},
    )
    return lock


def _analysis_authorization() -> dict[str, Any]:
    materialize()
    if ANALYSIS_LOCK.exists() or ANALYSIS_MARKER.exists():
        if not ANALYSIS_LOCK.is_file() or not ANALYSIS_MARKER.is_file():
            raise SourceRetentionCorrectionError("v2 analysis authorization is incomplete")
        lock = _load_mapping(ANALYSIS_LOCK)
        marker = _load_mapping(ANALYSIS_MARKER)
        if (
            lock.get("runner_sha256") != sha256_file(Path(__file__))
            or lock.get("prediction_lock_sha256") != sha256_file(PREDICTION_LOCK)
            or lock.get("annotation_sha256") != ANNOTATION_SHA256
            or marker.get("analysis_lock_sha256") != sha256_file(ANALYSIS_LOCK)
        ):
            raise SourceRetentionCorrectionError("v2 analysis authorization changed")
        return lock
    if any(path.exists() for path in (METRICS, STATISTICS, REPORT)):
        raise SourceRetentionCorrectionError("v2 metrics appeared before analysis authorization")
    lock = {
        "schema_version": 1,
        "status": "HAZYDET_SOURCE_RETENTION_V2_ANALYSIS_AUTHORIZED",
        "authorized_at_utc": _now(),
        "runner_sha256": sha256_file(Path(__file__)),
        "protocol_sha256": sha256_file(PROTOCOL),
        "prediction_lock_sha256": sha256_file(PREDICTION_LOCK),
        "annotation_sha256": ANNOTATION_SHA256,
        "evaluation_image_scope": "all_1000_annotation_image_ids",
        "corrected_metric_accessed_before_authorization": False,
        "validation_annotation_previously_accessed": True,
        "method_or_hyperparameter_selection": False,
        "HazyDet_test_access": "prohibited",
    }
    atomic_write_json(ANALYSIS_LOCK, lock)
    atomic_write_json(
        ANALYSIS_MARKER,
        {"status": lock["status"], "analysis_lock_sha256": sha256_file(ANALYSIS_LOCK)},
    )
    return lock


def _prediction_paths(lock: Mapping[str, Any]) -> dict[str, Path]:
    artifacts = lock.get("artifacts")
    if not isinstance(artifacts, list):
        raise SourceRetentionCorrectionError("v2 artifacts are invalid")
    paths = {
        str(row["model"]): _rooted(row["corrected_prediction"])
        for row in artifacts
        if isinstance(row, dict)
    }
    if tuple(paths) != MODELS:
        raise SourceRetentionCorrectionError("v2 prediction coverage changed")
    return paths


def _metrics_text(rows: Sequence[Mapping[str, Any]]) -> str:
    fields = ("model", *METRIC_KEYS, "images_evaluated", "mean_ms", "checkpoint_size_bytes")
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row[key] for key in fields})
    return buffer.getvalue()


def _checkpoint_deltas(path: Path) -> list[float]:
    values = _load_mapping(path).get("deltas")
    if not isinstance(values, list) or len(values) != RESAMPLES:
        raise SourceRetentionCorrectionError(f"bootstrap checkpoint is incomplete: {path}")
    return [float(value) for value in values]


def score() -> dict[str, Any]:
    prediction_lock = materialize()
    _analysis_authorization()
    if REPORT.exists():
        report = _load_mapping(REPORT)
        marker = _load_mapping(COMPLETE)
        if marker.get("report_sha256") != sha256_file(REPORT):
            raise SourceRetentionCorrectionError("existing v2 report is not locked")
        return report
    predictions = _prediction_paths(prediction_lock)
    timing = prediction_lock.get("timing_median_mean_ms")
    if not isinstance(timing, dict):
        raise SourceRetentionCorrectionError("v2 timing metadata changed")
    image_ids = _annotation_image_ids()
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        metrics = evaluate_coco(
            ANNOTATION,
            predictions[model],
            max_det=MAX_DET,
            image_ids=image_ids,
        )
        if int(metrics["images_evaluated"]) != IMAGES:
            raise SourceRetentionCorrectionError(f"{model} did not evaluate the full split")
        rows.append(
            {
                "model": model,
                **{key: float(metrics[key]) for key in METRIC_KEYS},
                "images_evaluated": int(metrics["images_evaluated"]),
                "mean_ms": float(timing[model]),
                "checkpoint_size_bytes": WEIGHTS[model].stat().st_size,
            }
        )
    expected_metrics = _metrics_text(rows)
    if METRICS.exists():
        if METRICS.read_text(encoding="utf-8") != expected_metrics:
            raise SourceRetentionCorrectionError("resumable v2 metrics changed")
    else:
        atomic_write_text(METRICS, expected_metrics)
    by_model = {str(row["model"]): row for row in rows}
    source_ap_drift = float(by_model["source"]["AP"]) - HISTORICAL_SOURCE_AP
    if abs(source_ap_drift) > SOURCE_AP_MAX_ABS_DRIFT:
        raise SourceRetentionCorrectionError(
            f"source AP anchor failed: drift={source_ap_drift:+.9f}"
        )
    point_deltas = {
        "CVBRA_v1_minus_source": float(by_model["CVBRA_v1"]["AP"])
        - float(by_model["source"]["AP"]),
        **{
            f"CVBRA_v1_minus_{model}": float(by_model["CVBRA_v1"]["AP"])
            - float(by_model[model]["AP"])
            for model in MODELS[2:]
        },
    }
    comparisons = (("source", "CVBRA_v1"), *((model, "CVBRA_v1") for model in MODELS[2:]))
    statistics_rows: list[dict[str, Any]] = []
    for baseline, method in comparisons:
        name = f"{method}_minus_{baseline}"
        checkpoint = OUTPUT / "bootstrap" / f"{name}.json"
        result = paired_coco_ap_bootstrap(
            ANNOTATION,
            predictions[baseline],
            predictions[method],
            resamples=RESAMPLES,
            seed=SEED,
            max_det=MAX_DET,
            image_ids=image_ids,
            workers=4,
            checkpoint_path=checkpoint,
            checkpoint_identity={
                "study": "cvbra_v1_hazydet_source_retention_v2",
                "comparison": name,
                "prediction_lock_sha256": sha256_file(PREDICTION_LOCK),
                "mapping_correction_only": True,
                "descriptive_post_freeze": True,
            },
            chunk_resamples=50,
        )
        expected = point_deltas[name]
        if abs(float(result["delta"]) - expected) > 1e-10:
            raise SourceRetentionCorrectionError(f"bootstrap point estimate drifted: {name}")
        deltas = _checkpoint_deltas(checkpoint)
        standard_deviation = stdev(deltas)
        if standard_deviation <= 0.0 or not math.isfinite(standard_deviation):
            raise SourceRetentionCorrectionError(f"bootstrap variance is invalid: {name}")
        statistics_rows.append(
            {
                "comparison": name,
                **result,
                "bootstrap_mean_delta": fmean(deltas),
                "bootstrap_standard_deviation": standard_deviation,
                "standardized_effect": expected / standard_deviation,
                "p_two_sided": bootstrap_sign_pvalue(deltas),
                "checkpoint": _relative(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
            }
        )
        print(json.dumps({"bootstrap_complete": name}), flush=True)
    adjusted = holm_adjust([float(row["p_two_sided"]) for row in statistics_rows])
    for row, value in zip(statistics_rows, adjusted, strict=True):
        row["holm_adjusted_p"] = value
        row["holm_significant_0p05"] = value < 0.05
    statistics = {
        "schema_version": 1,
        "status": "HAZYDET_SOURCE_RETENTION_V2_PAIRED_STATISTICS_COMPLETE",
        "completed_at_utc": _now(),
        "resamples": RESAMPLES,
        "seed": SEED,
        "unit": "image",
        "images": IMAGES,
        "multiple_testing": "Holm across five descriptive AP contrasts",
        "rows": statistics_rows,
    }
    atomic_write_json(STATISTICS, statistics)
    report = {
        "schema_version": 1,
        "status": "COMPLETE_CVBRA_V1_HAZYDET_SOURCE_RETENTION_V2_AUDIT",
        "completed_at_utc": _now(),
        "protocol_sha256": sha256_file(PROTOCOL),
        "runner": _relative(Path(__file__)),
        "runner_sha256": sha256_file(Path(__file__)),
        "amendment_sha256": sha256_file(AMENDMENT),
        "prediction_lock_sha256": sha256_file(PREDICTION_LOCK),
        "analysis_lock_sha256": sha256_file(ANALYSIS_LOCK),
        "metrics": _relative(METRICS),
        "metrics_sha256": sha256_file(METRICS),
        "statistics": _relative(STATISTICS),
        "statistics_sha256": sha256_file(STATISTICS),
        "rows": rows,
        "AP_deltas": point_deltas,
        "paired_statistics": statistics_rows,
        "source_anchor": {
            "historical_AP": HISTORICAL_SOURCE_AP,
            "observed_AP": float(by_model["source"]["AP"]),
            "drift": source_ap_drift,
            "maximum_absolute_drift": SOURCE_AP_MAX_ABS_DRIFT,
            "pass": True,
        },
        "evidence_boundary": {
            "scope": "spent full 1000-image HazyDet validation; descriptive source retention",
            "mapping_correction_only": True,
            "all_annotation_images_evaluated": True,
            "validation_labels_previously_accessed": True,
            "method_or_hyperparameter_selection": False,
            "independent_confirmation_claim": False,
            "HazyDet_test_access": "prohibited",
            "UAV_OBB_test_access": "prohibited",
        },
        "paper_body_change_authorized": False,
    }
    atomic_write_json(REPORT, report)
    atomic_write_json(
        COMPLETE,
        {
            "status": report["status"],
            "report_sha256": sha256_file(REPORT),
            "metrics_sha256": sha256_file(METRICS),
            "statistics_sha256": sha256_file(STATISTICS),
        },
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Correct and score the frozen CVBRA-v1 HazyDet source-retention outputs"
    )
    parser.add_argument(
        "--stage",
        choices=("preflight", "materialize", "score"),
        default="score",
    )
    args = parser.parse_args()
    if args.stage == "preflight":
        result = preflight()
        artifact = IMPLEMENTATION_LOCK
    elif args.stage == "materialize":
        result = materialize()
        artifact = PREDICTION_LOCK
    else:
        result = score()
        artifact = REPORT
    print(
        json.dumps(
            {"status": result["status"], "artifact_sha256": sha256_file(artifact)},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
