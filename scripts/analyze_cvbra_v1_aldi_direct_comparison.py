from __future__ import annotations

import argparse
import csv
import json
import math
import platform
from collections.abc import Mapping, Sequence
from io import StringIO
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

from scripts import analyze_cvbra_v1_uav_obb_official_validation as target_analysis
from scripts import run_cvbra_v1_aldi_direct_baseline as training
from scripts import run_cvbra_v1_hazydet_source_retention_v3 as hazydet_reference
from scripts import run_cvbra_v1_hazydet_source_retention_v4 as hazydet_recenter

from buse_uav.evaluation.bootstrap import (
    ClusterBootstrapScope,
    paired_coco_ap_cluster_bootstrap_scopes,
)
from buse_uav.evaluation.coco import evaluate_coco
from buse_uav.evaluation.supplementary import bootstrap_sign_pvalue, holm_adjust
from buse_uav.utils.hashing import sha256_file
from buse_uav.utils.io import atomic_write_json, atomic_write_text

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/experiment/cvbra_v1_aldi_direct_baseline_v1.yaml"
PROTOCOL_SHA256 = "3e8f28f2d474d4c0caf971cdd2259baadaeac364c9ff168aed694d45481ec3e2"
REGISTRATION = (
    ROOT / "reports/development/cvbra_v1_aldi_direct_baseline_v1/REGISTRATION_LOCK.json"
)
REGISTRATION_SHA256 = "ad7046de7a913321eaa9e528719a309358962cc1d517273db8672fb53413facf"
TRAINING_IMPLEMENTATION = (
    ROOT / "reports/development/cvbra_v1_aldi_direct_baseline_v1/implementation_lock_v4.json"
)
TRAINING_IMPLEMENTATION_SHA256 = (
    "786cf19e1502914bf7c65aa22f4b052bd047ca0c1d98e5af1c3456cfd24be8ac"
)

TARGET_MATERIALIZATION = (
    ROOT
    / "reports/development/cvbra_v1/uav_obb_official_validation/view_materialization_lock.json"
)
TARGET_MATERIALIZATION_SHA256 = (
    "9f6c9db26806679b6853316156785a9b475afe3a4a24fbda1c8ef2725ebbc07d"
)
TARGET_ANNOTATION = (
    ROOT
    / "reports/development/cvbra_v1/uav_obb_official_validation/annotations"
    / "official_validation_exact_car_truck_bus_hbb.coco.json"
)
TARGET_ANNOTATION_SHA256 = (
    "21d2431e0c6d70b21dc9bf9a3da03ae19a2d6e1cba96530c8e4be44e8bfe945e"
)
TARGET_REFERENCE_LOCK = (
    ROOT
    / "reports/development/cvbra_v1/uav_obb_official_validation/evaluation"
    / "joint_prediction_lock.json"
)
TARGET_REFERENCE_LOCK_SHA256 = (
    "5014b03170c7223faa3228686db44593c31135f5a7bd81e9c2fc57a59e3eef97"
)
TARGET_REFERENCE_MARKER = TARGET_REFERENCE_LOCK.parent / "PREDICTIONS_LOCKED"
TARGET_ALDI_LOCK = (
    ROOT
    / "reports/development/cvbra_v1_aldi_direct_baseline_v1"
    / "uav_obb_official_validation/evaluation/joint_prediction_lock.json"
)
TARGET_ALDI_MARKER = TARGET_ALDI_LOCK.parent / "PREDICTIONS_LOCKED"

HAZYDET_ANNOTATION = ROOT / "data/raw/HazyDet/val/val_coco.json"
HAZYDET_ANNOTATION_SHA256 = (
    "2b2e39f7812631dfb4f3f0fbe1e743b65ca873151ca92d8e24ddbf0e9feacb9a"
)
HAZYDET_REFERENCE_LOCK = (
    ROOT
    / "reports/development/cvbra_v1_matched_baselines/hazydet_source_retention_v2"
    / "prediction_lock.json"
)
HAZYDET_REFERENCE_LOCK_SHA256 = (
    "8733301dc19faa2b65acd668200fd2842a6223a785e6879ea50a9d209e5637e6"
)
HAZYDET_ALDI_LOCK = (
    ROOT
    / "reports/development/cvbra_v1_aldi_direct_baseline_v1"
    / "hazydet_source_retention/evaluation/prediction_lock.json"
)
HAZYDET_ALDI_MARKER = HAZYDET_ALDI_LOCK.parent / "PREDICTIONS_LOCKED"

OUTPUT = ROOT / "reports/development/cvbra_v1_aldi_direct_baseline_v1/direct_comparison_analysis"
AUTHORIZATION = OUTPUT / "analysis_authorization.json"
AUTHORIZATION_MARKER = OUTPUT / "ANALYSIS_AUTHORIZED"
TARGET_METRICS = OUTPUT / "target_metrics.csv"
HAZYDET_METRICS = OUTPUT / "hazydet_metrics.csv"
POINT_REPORT = OUTPUT / "point_report.json"
POINT_MARKER = OUTPUT / "POINTS_COMPLETE"
STATISTICS = OUTPUT / "paired_statistics.json"
STATISTICS_MARKER = OUTPUT / "STATISTICS_COMPLETE"
REPORT = OUTPUT / "direct_comparison_report.json"
COMPLETE = OUTPUT / "DIRECT_COMPARISON_COMPLETE"

SOURCE = "source"
CVBRA = "CVBRA_v1"
NATIVE = training.PAPER_LABELS["native_uda"]
EQUAL = training.PAPER_LABELS["equal_supervision"]
MODELS = (SOURCE, CVBRA, NATIVE, EQUAL)
ALDI_MODELS = (NATIVE, EQUAL)
VIEWS = ("original", "fog_0p6", "fog_1p0")
METRIC_KEYS = ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large")
TARGET_PRIMARY_SCOPE = "decontaminated_primary"
TARGET_SECONDARY_SCOPE = "official_full_secondary"
TARGET_PRIMARY_IMAGES = 167
TARGET_PRIMARY_GROUPS = 141
TARGET_IMAGES = 218
HAZYDET_IMAGES = 1000
MAX_DET = 500
TARGET_RESAMPLES = 10000
HAZYDET_RESAMPLES = 2000
SEED = 20260822


class ALDIDirectComparisonAnalysisError(RuntimeError):
    """Raised when the registered direct-comparison analysis cannot fail closed."""


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ALDIDirectComparisonAnalysisError(f"cannot parse mapping {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ALDIDirectComparisonAnalysisError(f"expected mapping: {path}")
    return value


def _assert_hash(path: Path, expected: object, *, label: str) -> None:
    if not path.is_file() or sha256_file(path) != str(expected):
        raise ALDIDirectComparisonAnalysisError(f"locked {label} changed: {path}")


def _rooted(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _artifacts(lock: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = lock.get("artifacts")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ALDIDirectComparisonAnalysisError("prediction artifacts are invalid")
    return [dict(row) for row in rows]


def _validate_target_lock(
    path: Path,
    marker_path: Path,
    *,
    expected_models: Sequence[str],
    expected_sha256: str | None,
) -> dict[str, Any]:
    if expected_sha256 is not None:
        _assert_hash(path, expected_sha256, label="target prediction lock")
    if not path.is_file() or not marker_path.is_file():
        raise ALDIDirectComparisonAnalysisError("target prediction lock is incomplete")
    lock = _load_mapping(path)
    marker = _load_mapping(marker_path)
    rows = _artifacts(lock)
    if (
        lock.get("status")
        != "CVBRA_V1_UAV_OBB_OFFICIAL_VALIDATION_PREDICTIONS_LOCKED_BEFORE_LABELS_OR_METRICS"
        or marker.get("joint_prediction_lock_sha256") != sha256_file(path)
        or len(rows) != len(expected_models) * len(VIEWS) * 2
        or {str(row.get("model")) for row in rows} != set(expected_models)
        or lock.get("official_test_content_accessed") is not False
    ):
        raise ALDIDirectComparisonAnalysisError("target prediction scope changed")
    for row in rows:
        _assert_hash(
            _rooted(row["prediction"]),
            row["prediction_sha256"],
            label="target prediction",
        )
    return lock


def _validate_hazydet_aldi_lock() -> dict[str, Any]:
    if not HAZYDET_ALDI_LOCK.is_file() or not HAZYDET_ALDI_MARKER.is_file():
        raise ALDIDirectComparisonAnalysisError("ALDI HazyDet prediction lock is incomplete")
    lock = _load_mapping(HAZYDET_ALDI_LOCK)
    marker = _load_mapping(HAZYDET_ALDI_MARKER)
    rows = _artifacts(lock)
    if (
        lock.get("status") != "ALL_HAZYDET_SOURCE_RETENTION_PREDICTIONS_LOCKED_BEFORE_METRICS"
        or marker.get("prediction_lock_sha256") != sha256_file(HAZYDET_ALDI_LOCK)
        or lock.get("models") != list(ALDI_MODELS)
        or len(rows) != len(ALDI_MODELS)
        or lock.get("metric_or_selection_feedback_used") is not False
        or lock.get("HazyDet_test_access") != "prohibited"
    ):
        raise ALDIDirectComparisonAnalysisError("ALDI HazyDet prediction scope changed")
    for row in rows:
        _assert_hash(
            _rooted(row["prediction"]),
            row["prediction_sha256"],
            label="HazyDet prediction",
        )
    return lock


def _validate_evidence() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    for path, digest, label in (
        (PROTOCOL, PROTOCOL_SHA256, "ALDI protocol"),
        (REGISTRATION, REGISTRATION_SHA256, "ALDI registration"),
        (
            TRAINING_IMPLEMENTATION,
            TRAINING_IMPLEMENTATION_SHA256,
            "ALDI training implementation",
        ),
        (TARGET_MATERIALIZATION, TARGET_MATERIALIZATION_SHA256, "target materialization"),
        (TARGET_ANNOTATION, TARGET_ANNOTATION_SHA256, "target annotation"),
        (HAZYDET_ANNOTATION, HAZYDET_ANNOTATION_SHA256, "HazyDet annotation"),
        (HAZYDET_REFERENCE_LOCK, HAZYDET_REFERENCE_LOCK_SHA256, "HazyDet reference lock"),
    ):
        _assert_hash(path, digest, label=label)
    materialization = _load_mapping(TARGET_MATERIALIZATION)
    if (
        materialization.get("images") != TARGET_IMAGES
        or materialization.get("primary_images") != TARGET_PRIMARY_IMAGES
        or materialization.get("primary_source_groups") != TARGET_PRIMARY_GROUPS
        or materialization.get("official_test_content_accessed") is not False
    ):
        raise ALDIDirectComparisonAnalysisError("target materialization scope changed")
    for variant in training.VARIANTS:
        training._validate_checkpoint_lock(variant)
    target_reference = _validate_target_lock(
        TARGET_REFERENCE_LOCK,
        TARGET_REFERENCE_MARKER,
        expected_models=(SOURCE, CVBRA),
        expected_sha256=TARGET_REFERENCE_LOCK_SHA256,
    )
    target_aldi = _validate_target_lock(
        TARGET_ALDI_LOCK,
        TARGET_ALDI_MARKER,
        expected_models=ALDI_MODELS,
        expected_sha256=None,
    )
    if (
        target_aldi.get("official_validation_labels_previously_accessed") is not True
        or target_aldi.get("labels_read_by_inference_runner") is not False
        or target_aldi.get("method_or_hyperparameter_selection") is not False
    ):
        raise ALDIDirectComparisonAnalysisError("ALDI target inference provenance changed")
    hazydet_reference_lock = hazydet_reference._v2_prediction_lock()
    hazydet_aldi = _validate_hazydet_aldi_lock()
    return target_reference, target_aldi, hazydet_reference_lock, hazydet_aldi


def _analysis_authorization() -> dict[str, Any]:
    _validate_evidence()
    if AUTHORIZATION.exists() or AUTHORIZATION_MARKER.exists():
        if not AUTHORIZATION.is_file() or not AUTHORIZATION_MARKER.is_file():
            raise ALDIDirectComparisonAnalysisError("analysis authorization is incomplete")
        lock = _load_mapping(AUTHORIZATION)
        marker = _load_mapping(AUTHORIZATION_MARKER)
        if (
            lock.get("runner_sha256") != sha256_file(Path(__file__))
            or lock.get("target_aldi_prediction_lock_sha256") != sha256_file(TARGET_ALDI_LOCK)
            or lock.get("hazydet_aldi_prediction_lock_sha256")
            != sha256_file(HAZYDET_ALDI_LOCK)
            or marker.get("analysis_authorization_sha256") != sha256_file(AUTHORIZATION)
        ):
            raise ALDIDirectComparisonAnalysisError("analysis authorization changed")
        return lock
    later_outputs = (TARGET_METRICS, HAZYDET_METRICS, POINT_REPORT, STATISTICS, REPORT)
    if any(path.exists() for path in later_outputs):
        raise ALDIDirectComparisonAnalysisError("metric output appeared before analysis lock")
    payload = {
        "schema_version": 1,
        "status": "ALDI_DIRECT_COMPARISON_ANALYSIS_AUTHORIZED_AFTER_ALL_PREDICTIONS_LOCKED",
        "authorized_at_utc": target_analysis._utc_now(),
        "protocol_sha256": PROTOCOL_SHA256,
        "registration_sha256": REGISTRATION_SHA256,
        "training_implementation_sha256": TRAINING_IMPLEMENTATION_SHA256,
        "runner": _relative(Path(__file__)),
        "runner_sha256": sha256_file(Path(__file__)),
        "python": platform.python_version(),
        "target_reference_prediction_lock_sha256": TARGET_REFERENCE_LOCK_SHA256,
        "target_aldi_prediction_lock_sha256": sha256_file(TARGET_ALDI_LOCK),
        "hazydet_reference_prediction_lock_sha256": HAZYDET_REFERENCE_LOCK_SHA256,
        "hazydet_aldi_prediction_lock_sha256": sha256_file(HAZYDET_ALDI_LOCK),
        "target_annotation_sha256": TARGET_ANNOTATION_SHA256,
        "hazydet_annotation_sha256": HAZYDET_ANNOTATION_SHA256,
        "target_resamples": TARGET_RESAMPLES,
        "hazydet_resamples": HAZYDET_RESAMPLES,
        "seed": SEED,
        "primary_method": "Identity_one_pass",
        "validation_labels_previously_accessed": True,
        "metric_accessed_before_authorization": False,
        "method_or_hyperparameter_selection": False,
        "independent_confirmation_claim": False,
        "UAV_OBB_official_test_access": "prohibited",
        "HazyDet_test_or_RDDTS_access": "prohibited",
        "paper_body_change_authorized": False,
    }
    atomic_write_json(AUTHORIZATION, payload)
    atomic_write_json(
        AUTHORIZATION_MARKER,
        {
            "status": payload["status"],
            "analysis_authorization_sha256": sha256_file(AUTHORIZATION),
        },
    )
    return payload


def _target_lookup(*locks: Mapping[str, Any]) -> dict[tuple[str, str], Path]:
    result: dict[tuple[str, str], Path] = {}
    for lock in locks:
        for row in _artifacts(lock):
            if row.get("method") == "Identity":
                key = (str(row["model"]), str(row["view"]))
                result[key] = _rooted(row["prediction"])
    if set(result) != {(model, view) for model in MODELS for view in VIEWS}:
        raise ALDIDirectComparisonAnalysisError("target Identity prediction coverage changed")
    return result


def _target_timing(*locks: Mapping[str, Any]) -> dict[tuple[str, str], float]:
    result: dict[tuple[str, str], float] = {}
    for lock in locks:
        timing = lock.get("timing")
        if not isinstance(timing, dict):
            raise ALDIDirectComparisonAnalysisError("target timing payload is invalid")
        for model, by_view in timing.items():
            if not isinstance(by_view, dict):
                raise ALDIDirectComparisonAnalysisError("target view timing is invalid")
            for view, values in by_view.items():
                if not isinstance(values, dict):
                    raise ALDIDirectComparisonAnalysisError("target method timing is invalid")
                result[(str(model), str(view))] = float(values["Identity_mean_ms"])
    if set(result) != {(model, view) for model in MODELS for view in VIEWS}:
        raise ALDIDirectComparisonAnalysisError("target timing coverage changed")
    return result


def _hazydet_lookup(
    reference: Mapping[str, Any], aldi: Mapping[str, Any]
) -> tuple[dict[str, Path], dict[str, float]]:
    predictions: dict[str, Path] = {}
    timing: dict[str, float] = {}
    reference_timing = reference.get("timing_median_mean_ms")
    if not isinstance(reference_timing, dict):
        raise ALDIDirectComparisonAnalysisError("HazyDet reference timing changed")
    for row in _artifacts(reference):
        model = str(row["model"])
        if model in (SOURCE, CVBRA):
            predictions[model] = _rooted(row["corrected_prediction"])
            timing[model] = float(reference_timing[model])
    aldi_timing = aldi.get("timing_median_mean_ms")
    if not isinstance(aldi_timing, dict):
        raise ALDIDirectComparisonAnalysisError("ALDI HazyDet timing changed")
    for row in _artifacts(aldi):
        model = str(row["model"])
        predictions[model] = _rooted(row["prediction"])
        timing[model] = float(aldi_timing[model])
    if set(predictions) != set(MODELS) or set(timing) != set(MODELS):
        raise ALDIDirectComparisonAnalysisError("HazyDet prediction coverage changed")
    return predictions, timing


def _checkpoint_sizes() -> dict[str, int]:
    locks = {variant: training._validate_checkpoint_lock(variant) for variant in training.VARIANTS}
    sizes = {
        SOURCE: (ROOT / "weights/hazydet/yolo11n_best.pt").stat().st_size,
        CVBRA: (ROOT / "runs/cvbra_v1/yolo11n/cvbra_v1.pt").stat().st_size,
    }
    for variant, lock in locks.items():
        sizes[training.PAPER_LABELS[variant]] = int(lock["checkpoint_size_bytes"])
    return sizes


def _csv_text(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row[field] for field in fields})
    return buffer.getvalue()


def points() -> dict[str, Any]:
    target_reference, target_aldi, hazydet_reference_lock, hazydet_aldi = _validate_evidence()
    _analysis_authorization()
    if POINT_REPORT.exists():
        report = _load_mapping(POINT_REPORT)
        if not POINT_MARKER.is_file() or _load_mapping(POINT_MARKER).get(
            "point_report_sha256"
        ) != sha256_file(POINT_REPORT):
            raise ALDIDirectComparisonAnalysisError("existing point report is not locked")
        return report
    later_outputs = (TARGET_METRICS, HAZYDET_METRICS, POINT_MARKER, STATISTICS, REPORT)
    if any(path.exists() for path in later_outputs):
        raise ALDIDirectComparisonAnalysisError("partial point analysis requires audit")
    materialization = _load_mapping(TARGET_MATERIALIZATION)
    primary_ids_raw = materialization.get("primary_image_ids")
    if not isinstance(primary_ids_raw, list):
        raise ALDIDirectComparisonAnalysisError("target primary image registry changed")
    target_scopes = {
        TARGET_PRIMARY_SCOPE: [int(value) for value in primary_ids_raw],
        TARGET_SECONDARY_SCOPE: list(range(1, TARGET_IMAGES + 1)),
    }
    target_predictions = _target_lookup(target_reference, target_aldi)
    target_timing = _target_timing(target_reference, target_aldi)
    sizes = _checkpoint_sizes()
    target_rows: list[dict[str, Any]] = []
    for scope, image_ids in target_scopes.items():
        for model in MODELS:
            for view in VIEWS:
                prediction = target_predictions[(model, view)]
                metrics = evaluate_coco(
                    TARGET_ANNOTATION,
                    prediction,
                    max_det=MAX_DET,
                    image_ids=image_ids,
                )
                target_rows.append(
                    {
                        "scope": scope,
                        "model": model,
                        "view": view,
                        **{key: float(metrics[key]) for key in METRIC_KEYS},
                        "images_evaluated": int(metrics["images_evaluated"]),
                        "mean_ms": target_timing[(model, view)],
                        "checkpoint_size_bytes": sizes[model],
                        "prediction_sha256": sha256_file(prediction),
                    }
                )
    target_fields = (
        "scope",
        "model",
        "view",
        *METRIC_KEYS,
        "images_evaluated",
        "mean_ms",
        "checkpoint_size_bytes",
        "prediction_sha256",
    )
    atomic_write_text(TARGET_METRICS, _csv_text(target_rows, target_fields))

    hazydet_predictions, hazydet_timing = _hazydet_lookup(
        hazydet_reference_lock, hazydet_aldi
    )
    hazydet_ids = hazydet_reference.ordered_image_ids()
    hazydet_rows: list[dict[str, Any]] = []
    for model in MODELS:
        prediction = hazydet_predictions[model]
        metrics = evaluate_coco(
            HAZYDET_ANNOTATION,
            prediction,
            max_det=MAX_DET,
            image_ids=hazydet_ids,
        )
        hazydet_rows.append(
            {
                "model": model,
                **{key: float(metrics[key]) for key in METRIC_KEYS},
                "images_evaluated": int(metrics["images_evaluated"]),
                "mean_ms": hazydet_timing[model],
                "checkpoint_size_bytes": sizes[model],
                "prediction_sha256": sha256_file(prediction),
            }
        )
    hazydet_fields = (
        "model",
        *METRIC_KEYS,
        "images_evaluated",
        "mean_ms",
        "checkpoint_size_bytes",
        "prediction_sha256",
    )
    atomic_write_text(HAZYDET_METRICS, _csv_text(hazydet_rows, hazydet_fields))

    target_lookup = {
        (str(row["scope"]), str(row["model"]), str(row["view"])): row
        for row in target_rows
    }
    hazydet_by_model = {str(row["model"]): row for row in hazydet_rows}

    def target_delta(left: str, right: str, view: str, metric: str = "AP") -> float:
        return float(target_lookup[(TARGET_PRIMARY_SCOPE, left, view)][metric]) - float(
            target_lookup[(TARGET_PRIMARY_SCOPE, right, view)][metric]
        )

    report = {
        "schema_version": 1,
        "status": "ALDI_DIRECT_COMPARISON_POINT_ESTIMATES_LOCKED",
        "completed_at_utc": target_analysis._utc_now(),
        "analysis_authorization_sha256": sha256_file(AUTHORIZATION),
        "target_metrics": _relative(TARGET_METRICS),
        "target_metrics_sha256": sha256_file(TARGET_METRICS),
        "hazydet_metrics": _relative(HAZYDET_METRICS),
        "hazydet_metrics_sha256": sha256_file(HAZYDET_METRICS),
        "target_rows": target_rows,
        "hazydet_rows": hazydet_rows,
        "target_primary_AP_deltas": {
            "CVBRA_minus_native_UDA": {
                view: target_delta(CVBRA, NATIVE, view) for view in VIEWS
            },
            "CVBRA_minus_equal_supervision": {
                view: target_delta(CVBRA, EQUAL, view) for view in VIEWS
            },
            "equal_supervision_minus_native_UDA": {
                view: target_delta(EQUAL, NATIVE, view) for view in VIEWS
            },
        },
        "hazydet_AP_deltas": {
            "CVBRA_minus_source": float(hazydet_by_model[CVBRA]["AP"])
            - float(hazydet_by_model[SOURCE]["AP"]),
            "native_UDA_minus_source": float(hazydet_by_model[NATIVE]["AP"])
            - float(hazydet_by_model[SOURCE]["AP"]),
            "equal_supervision_minus_source": float(hazydet_by_model[EQUAL]["AP"])
            - float(hazydet_by_model[SOURCE]["AP"]),
        },
        "information_contracts": {
            SOURCE: "labeled_source_only",
            CVBRA: "labeled_source_plus_labeled_target",
            NATIVE: "labeled_source_plus_unlabeled_target",
            EQUAL: "labeled_source_plus_labeled_target_plus_registered_teacher_distillation",
        },
        "direct_same_information_pair": [CVBRA, EQUAL],
        "official_ALDI_authors_code": False,
        "translation": "registered_anchor_free_YOLO11n_translation_of_official_ALDIpp_roles",
        "evidence_boundary": {
            "target_primary": "167 validation images in 141 train-disjoint source-key groups",
            "target_secondary": "all 218 official validation images",
            "source_retention": "all 1000 HazyDet validation images",
            "primary_method": "Identity one-pass inference",
            "validation_labels_previously_accessed": True,
            "post_freeze_descriptive": True,
            "method_or_hyperparameter_selection": False,
            "independent_confirmation_claim": False,
            "UAV_OBB_official_test_access": "prohibited",
            "HazyDet_test_or_RDDTS_access": "prohibited",
        },
        "statistics_required_next": True,
        "paper_body_change_authorized": False,
    }
    atomic_write_json(POINT_REPORT, report)
    atomic_write_json(
        POINT_MARKER,
        {"status": report["status"], "point_report_sha256": sha256_file(POINT_REPORT)},
    )
    return report


def _checkpoint_deltas(path: Path, expected: int) -> list[float]:
    document = _load_mapping(path)
    values = document.get("deltas")
    if not isinstance(values, list) or len(values) != expected:
        raise ALDIDirectComparisonAnalysisError(f"bootstrap checkpoint is incomplete: {path}")
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise ALDIDirectComparisonAnalysisError("bootstrap checkpoint has non-finite values")
    return result


def statistics() -> dict[str, Any]:
    point = points()
    target_reference, target_aldi, hazydet_reference_lock, hazydet_aldi = _validate_evidence()
    if STATISTICS.exists():
        report = _load_mapping(STATISTICS)
        if not STATISTICS_MARKER.is_file() or _load_mapping(STATISTICS_MARKER).get(
            "statistics_sha256"
        ) != sha256_file(STATISTICS):
            raise ALDIDirectComparisonAnalysisError("existing statistics are not locked")
        return report
    target_predictions = _target_lookup(target_reference, target_aldi)
    target_clusters = target_analysis._clusters(TARGET_ANNOTATION)
    point_deltas = point["target_primary_AP_deltas"]
    target_rows: list[dict[str, Any]] = []
    target_comparisons = (
        (NATIVE, CVBRA, "CVBRA_minus_native_UDA"),
        (EQUAL, CVBRA, "CVBRA_minus_equal_supervision"),
    )
    for baseline, method, family in target_comparisons:
        for view in VIEWS:
            name = f"{family}_{view}"
            checkpoint = OUTPUT / "target_bootstrap" / f"{name}.json"
            result = paired_coco_ap_cluster_bootstrap_scopes(
                TARGET_ANNOTATION,
                target_predictions[(baseline, view)],
                target_predictions[(method, view)],
                {
                    "primary": ClusterBootstrapScope(
                        clusters=target_clusters,
                        checkpoint_path=checkpoint,
                        checkpoint_identity={
                            "study": "cvbra_v1_aldi_direct_comparison",
                            "comparison": name,
                            "point_report_sha256": sha256_file(POINT_REPORT),
                            "post_freeze_descriptive": True,
                        },
                    )
                },
                resamples=TARGET_RESAMPLES,
                seed=SEED,
                max_det=MAX_DET,
                workers=4,
                chunk_resamples=100,
                accelerate_ap_only=True,
            )["primary"]
            expected = float(point_deltas[family][view])
            if abs(float(result["delta"]) - expected) > 1e-10:
                raise ALDIDirectComparisonAnalysisError(f"target bootstrap point drifted: {name}")
            deltas = _checkpoint_deltas(checkpoint, TARGET_RESAMPLES)
            standard_deviation = stdev(deltas)
            if standard_deviation <= 0.0 or not math.isfinite(standard_deviation):
                raise ALDIDirectComparisonAnalysisError(f"target bootstrap variance failed: {name}")
            target_rows.append(
                {
                    "comparison": name,
                    "family": family,
                    "view": view,
                    "baseline": baseline,
                    "method": method,
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
    adjusted_target = holm_adjust([float(row["p_two_sided"]) for row in target_rows])
    for row, value in zip(target_rows, adjusted_target, strict=True):
        row["holm_adjusted_p"] = value
        row["holm_significant_0p05"] = value < 0.05

    hazydet_predictions, _ = _hazydet_lookup(hazydet_reference_lock, hazydet_aldi)
    hazydet_ids = hazydet_reference.ordered_image_ids()
    hazydet_clusters = hazydet_reference.singleton_image_clusters(hazydet_ids)
    source_deltas = point["hazydet_AP_deltas"]
    hazydet_rows: list[dict[str, Any]] = []
    for model, key in (
        (NATIVE, "native_UDA_minus_source"),
        (EQUAL, "equal_supervision_minus_source"),
    ):
        checkpoint = OUTPUT / "hazydet_bootstrap" / f"{key}.json"
        raw = paired_coco_ap_cluster_bootstrap_scopes(
            HAZYDET_ANNOTATION,
            hazydet_predictions[SOURCE],
            hazydet_predictions[model],
            {
                "all_images": ClusterBootstrapScope(
                    clusters=hazydet_clusters,
                    checkpoint_path=checkpoint,
                    checkpoint_identity={
                        "study": "cvbra_v1_aldi_direct_comparison",
                        "comparison": key,
                        "point_report_sha256": sha256_file(POINT_REPORT),
                        "recenter_rule": "raw_plus_direct_minus_raw_observed",
                    },
                )
            },
            resamples=HAZYDET_RESAMPLES,
            seed=SEED,
            max_det=MAX_DET,
            workers=4,
            chunk_resamples=100,
            accelerate_ap_only=True,
        )["all_images"]
        direct_delta = float(source_deltas[key])
        raw_delta = float(raw["delta"])
        raw_deltas = _checkpoint_deltas(checkpoint, HAZYDET_RESAMPLES)
        adjusted, offset = hazydet_recenter.recenter_bootstrap_deltas(
            raw_deltas,
            direct_delta=direct_delta,
            raw_observed_delta=raw_delta,
        )
        standard_deviation = stdev(adjusted)
        if standard_deviation <= 0.0 or not math.isfinite(standard_deviation):
            raise ALDIDirectComparisonAnalysisError(f"HazyDet bootstrap variance failed: {key}")
        hazydet_rows.append(
            {
                "comparison": key,
                "model": model,
                "direct_delta": direct_delta,
                "raw_observed_delta": raw_delta,
                "recenter_offset": offset,
                "adjusted_delta": raw_delta + offset,
                "raw_ci_low": float(raw["ci_low"]),
                "raw_ci_high": float(raw["ci_high"]),
                "ci_low": float(raw["ci_low"]) + offset,
                "ci_high": float(raw["ci_high"]) + offset,
                "resamples": int(raw["resamples"]),
                "seed": int(raw["seed"]),
                "images": int(raw["images"]),
                "clusters": int(raw["clusters"]),
                "raw_bootstrap_mean_delta": fmean(raw_deltas),
                "bootstrap_mean_delta": fmean(adjusted),
                "bootstrap_standard_deviation": standard_deviation,
                "standardized_effect": direct_delta / standard_deviation,
                "raw_p_two_sided": bootstrap_sign_pvalue(raw_deltas),
                "p_two_sided": bootstrap_sign_pvalue(adjusted),
                "checkpoint": _relative(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
            }
        )
        print(json.dumps({"bootstrap_complete": key}), flush=True)
    adjusted_hazydet = holm_adjust([float(row["p_two_sided"]) for row in hazydet_rows])
    for row, value in zip(hazydet_rows, adjusted_hazydet, strict=True):
        row["holm_adjusted_p"] = value
        row["holm_significant_0p05"] = value < 0.05

    report = {
        "schema_version": 1,
        "status": "ALDI_DIRECT_COMPARISON_PAIRED_STATISTICS_COMPLETE",
        "completed_at_utc": target_analysis._utc_now(),
        "point_report_sha256": sha256_file(POINT_REPORT),
        "target": {
            "unit": "canonical_source_key_group",
            "images": TARGET_PRIMARY_IMAGES,
            "groups": TARGET_PRIMARY_GROUPS,
            "resamples": TARGET_RESAMPLES,
            "seed": SEED,
            "multiple_testing": "Holm across six registered target AP contrasts",
            "rows": target_rows,
        },
        "hazydet": {
            "unit": "image_via_singleton_cluster_encoding",
            "images": HAZYDET_IMAGES,
            "resamples": HAZYDET_RESAMPLES,
            "seed": SEED,
            "recenter_rule": "raw_delta_plus_direct_delta_minus_raw_observed_delta",
            "multiple_testing": "Holm across two registered source-retention AP contrasts",
            "rows": hazydet_rows,
        },
        "method_or_hyperparameter_selection": False,
        "independent_confirmation_claim": False,
        "paper_body_change_authorized": False,
    }
    atomic_write_json(STATISTICS, report)
    atomic_write_json(
        STATISTICS_MARKER,
        {"status": report["status"], "statistics_sha256": sha256_file(STATISTICS)},
    )
    return report


def finalize() -> dict[str, Any]:
    point = points()
    stats = statistics()
    if REPORT.exists():
        report = _load_mapping(REPORT)
        if not COMPLETE.is_file() or _load_mapping(COMPLETE).get(
            "report_sha256"
        ) != sha256_file(REPORT):
            raise ALDIDirectComparisonAnalysisError("existing direct-comparison report changed")
        return report
    target_stats = stats.get("target")
    hazydet_stats = stats.get("hazydet")
    if not isinstance(target_stats, dict) or not isinstance(hazydet_stats, dict):
        raise ALDIDirectComparisonAnalysisError("direct-comparison statistics are incomplete")
    target_rows = target_stats.get("rows")
    hazydet_rows = hazydet_stats.get("rows")
    if not isinstance(target_rows, list) or len(target_rows) != 6:
        raise ALDIDirectComparisonAnalysisError("target direct-comparison coverage is incomplete")
    if not isinstance(hazydet_rows, list) or len(hazydet_rows) != 2:
        raise ALDIDirectComparisonAnalysisError("HazyDet direct-comparison coverage is incomplete")
    report = {
        "schema_version": 1,
        "status": "COMPLETE_CVBRA_V1_ALDI_DIRECT_COMPARISON_EVIDENCE",
        "completed_at_utc": target_analysis._utc_now(),
        "protocol_sha256": PROTOCOL_SHA256,
        "runner": _relative(Path(__file__)),
        "runner_sha256": sha256_file(Path(__file__)),
        "point_report": _relative(POINT_REPORT),
        "point_report_sha256": sha256_file(POINT_REPORT),
        "statistics": _relative(STATISTICS),
        "statistics_sha256": sha256_file(STATISTICS),
        "information_contracts": point["information_contracts"],
        "direct_same_information_pair": point["direct_same_information_pair"],
        "target_primary_AP_deltas": point["target_primary_AP_deltas"],
        "hazydet_AP_deltas": point["hazydet_AP_deltas"],
        "target_paired_statistics": target_rows,
        "hazydet_paired_statistics": hazydet_rows,
        "interpretation_rule": {
            "native_UDA": "recent-method native information contract",
            "equal_supervision": "primary direct same-information comparison to CVBRA",
            "result_reporting": "report all locked outcomes without method reselection",
            "architecture_claim": "anchor-free translation, not official authors' YOLO11 code",
        },
        "evidence_boundary": point["evidence_boundary"],
        "decision": {
            "recent_DAOD_direct_comparison_complete": True,
            "same_information_direct_comparison_complete": True,
            "main_method_reselection_authorized": False,
            "official_test_access_authorized": False,
            "paper_body_change_authorized": True,
        },
    }
    atomic_write_json(REPORT, report)
    atomic_write_json(
        COMPLETE,
        {
            "status": report["status"],
            "report_sha256": sha256_file(REPORT),
            "point_report_sha256": sha256_file(POINT_REPORT),
            "statistics_sha256": sha256_file(STATISTICS),
        },
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze the registered ALDI direct comparison")
    parser.add_argument("--stage", choices=("points", "statistics", "final"), default="final")
    args = parser.parse_args()
    if args.stage == "points":
        result = points()
        artifact = POINT_REPORT
    elif args.stage == "statistics":
        result = statistics()
        artifact = STATISTICS
    else:
        result = finalize()
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
