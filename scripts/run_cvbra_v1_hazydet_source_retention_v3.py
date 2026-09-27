from __future__ import annotations

import argparse
import csv
import json
import math
import platform
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

from buse_uav.evaluation.bootstrap import (
    ClusterBootstrapScope,
    paired_coco_ap_cluster_bootstrap_scopes,
)
from buse_uav.evaluation.coco import evaluate_coco
from buse_uav.evaluation.supplementary import bootstrap_sign_pvalue, holm_adjust
from buse_uav.utils.hashing import sha256_file
from buse_uav.utils.io import atomic_write_json, atomic_write_text

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "experiment" / "cvbra_v1_hazydet_source_retention_v3.yaml"
ANNOTATION = ROOT / "data" / "raw" / "HazyDet" / "val" / "val_coco.json"
V2_OUTPUT = (
    ROOT
    / "reports"
    / "development"
    / "cvbra_v1_matched_baselines"
    / "hazydet_source_retention_v2"
)
OUTPUT = (
    ROOT
    / "reports"
    / "development"
    / "cvbra_v1_matched_baselines"
    / "hazydet_source_retention_v3"
)

ANNOTATION_SHA256 = "2b2e39f7812631dfb4f3f0fbe1e743b65ca873151ca92d8e24ddbf0e9feacb9a"
V2_PROTOCOL_SHA256 = "029525c278359d1f61adc0215381b071a3d813deb650488ccd1ac39bb411f2fa"
V2_RUNNER_SHA256 = "02b8c7e9c8215c5d7015abe647db857d13ed6ae83c02adb89153337653271c08"
V2_PREDICTION_LOCK_SHA256 = "8733301dc19faa2b65acd668200fd2842a6223a785e6879ea50a9d209e5637e6"
V2_METRICS_SHA256 = "6910e8ff0aa215956f7356728ac45ec93a45d8096cb7e437152658bed3b6d3fe"
V2_INVALID_BOOTSTRAP_SHA256 = (
    "2632ceadbf71e7d9dded62f989246c5e9f5788b8670c9abeb611127fdd3ef170"
)
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

AMENDMENT = V2_OUTPUT / "BOOTSTRAP_ORDER_FAILURE_AMENDMENT_1.json"
INVALID_MARKER = V2_OUTPUT / "INVALID_BOOTSTRAP_ATTEMPT_1"
REGISTRATION = OUTPUT / "REGISTRATION_LOCK.json"
REGISTRATION_MARKER = OUTPUT / "REGISTERED"
IMPLEMENTATION_LOCK = OUTPUT / "implementation_lock.json"
IMPLEMENTATION_MARKER = OUTPUT / "IMPLEMENTATION_LOCKED"
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
METRIC_KEYS = ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large")


class SourceRetentionV3Error(RuntimeError):
    """Raised when the registered order-parity analysis cannot fail closed."""


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
        raise SourceRetentionV3Error(f"cannot parse mapping {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SourceRetentionV3Error(f"expected mapping: {path}")
    return value


def _assert_hash(path: Path, expected: str, *, label: str) -> None:
    if not path.is_file() or sha256_file(path) != expected:
        raise SourceRetentionV3Error(f"locked {label} changed: {path}")


def ordered_image_ids() -> tuple[int, ...]:
    """Match evaluate_coco's deterministic string-key image ordering exactly."""
    annotation = _load_mapping(ANNOTATION)
    raw_images = annotation.get("images")
    raw_categories = annotation.get("categories")
    if not isinstance(raw_images, list) or not isinstance(raw_categories, list):
        raise SourceRetentionV3Error("HazyDet annotation structure changed")
    image_ids = [
        int(row["id"])
        for row in raw_images
        if isinstance(row, dict) and isinstance(row.get("id"), int)
    ]
    category_ids = [
        int(row["id"])
        for row in raw_categories
        if isinstance(row, dict) and isinstance(row.get("id"), int)
    ]
    if len(image_ids) != IMAGES or len(set(image_ids)) != IMAGES or category_ids != [0, 1, 2]:
        raise SourceRetentionV3Error("HazyDet image or category scope changed")
    return tuple(sorted(image_ids, key=str))


def singleton_image_clusters(image_ids: Sequence[int]) -> dict[str, tuple[int]]:
    """Encode an ordinary image bootstrap as one singleton cluster per image."""
    if len(image_ids) != IMAGES or len(set(image_ids)) != IMAGES:
        raise SourceRetentionV3Error("singleton image bootstrap scope changed")
    return {f"image_{index:04d}": (image_id,) for index, image_id in enumerate(image_ids)}


def _v2_prediction_lock() -> dict[str, Any]:
    _assert_hash(
        V2_OUTPUT / "prediction_lock.json",
        V2_PREDICTION_LOCK_SHA256,
        label="v2 prediction lock",
    )
    lock = _load_mapping(V2_OUTPUT / "prediction_lock.json")
    artifacts = lock.get("artifacts")
    if (
        lock.get("status") != "ALL_CORRECTED_HAZYDET_SOURCE_RETENTION_PREDICTIONS_LOCKED"
        or lock.get("models") != list(MODELS)
        or not isinstance(artifacts, list)
        or len(artifacts) != len(MODELS)
    ):
        raise SourceRetentionV3Error("v2 prediction lock scope changed")
    for row in artifacts:
        if not isinstance(row, dict):
            raise SourceRetentionV3Error("v2 prediction artifact changed")
        _assert_hash(
            _rooted(row["corrected_prediction"]),
            str(row["corrected_prediction_sha256"]),
            label=f"{row['model']} corrected prediction",
        )
        _assert_hash(
            _rooted(row["transformation_audit"]),
            str(row["transformation_audit_sha256"]),
            label=f"{row['model']} transformation audit",
        )
    return lock


def _validate_v2_failure() -> dict[str, Any]:
    _assert_hash(ANNOTATION, ANNOTATION_SHA256, label="HazyDet annotation")
    _assert_hash(
        ROOT / "configs" / "experiment" / "cvbra_v1_hazydet_source_retention_v2.yaml",
        V2_PROTOCOL_SHA256,
        label="v2 protocol",
    )
    _assert_hash(
        ROOT / "scripts" / "run_cvbra_v1_hazydet_source_retention_v2.py",
        V2_RUNNER_SHA256,
        label="v2 runner",
    )
    _assert_hash(V2_OUTPUT / "metrics.csv", V2_METRICS_SHA256, label="v2 point metrics")
    _assert_hash(
        V2_OUTPUT / "bootstrap" / "CVBRA_v1_minus_source.json",
        V2_INVALID_BOOTSTRAP_SHA256,
        label="v2 invalid bootstrap",
    )
    ordered_image_ids()
    return _v2_prediction_lock()


def _write_or_validate_amendment() -> dict[str, Any]:
    if AMENDMENT.exists() or INVALID_MARKER.exists():
        if not AMENDMENT.is_file() or not INVALID_MARKER.is_file():
            raise SourceRetentionV3Error("v2 order-failure amendment is incomplete")
        amendment = _load_mapping(AMENDMENT)
        marker = _load_mapping(INVALID_MARKER)
        if (
            amendment.get("v2_prediction_lock_sha256") != V2_PREDICTION_LOCK_SHA256
            or amendment.get("invalid_bootstrap_sha256") != V2_INVALID_BOOTSTRAP_SHA256
            or marker.get("amendment_sha256") != sha256_file(AMENDMENT)
        ):
            raise SourceRetentionV3Error("v2 order-failure amendment changed")
        return amendment
    amendment = {
        "schema_version": 1,
        "status": "INVALIDATED_HAZYDET_SOURCE_RETENTION_V2_BOOTSTRAP_ATTEMPT_1",
        "recorded_at_utc": _now(),
        "v2_prediction_lock_sha256": V2_PREDICTION_LOCK_SHA256,
        "v2_point_metrics_sha256": V2_METRICS_SHA256,
        "invalid_bootstrap": _relative(
            V2_OUTPUT / "bootstrap" / "CVBRA_v1_minus_source.json"
        ),
        "invalid_bootstrap_sha256": V2_INVALID_BOOTSTRAP_SHA256,
        "root_cause": (
            "direct evaluation used lexicographic string-key image order while v2 bootstrap "
            "used annotation order; stable sorting of tied FP16 scores made AP order-sensitive"
        ),
        "direct_delta": -0.0351853933253464,
        "v2_bootstrap_delta": -0.0351661729459501,
        "absolute_difference": 0.0000192203793962231,
        "v2_direct_point_metrics_disposition": "valid; recompute under v3 for one locked report",
        "v2_bootstrap_disposition": "preserved and excluded from evidence",
        "authorized_v3": {
            "protocol": _relative(PROTOCOL),
            "image_order": "lexicographic_string_sort_for_direct_and_bootstrap",
            "bootstrap": "singleton cluster per image; exact image bootstrap",
            "prediction_change": False,
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
    _validate_v2_failure()
    _write_or_validate_amendment()
    protocol_sha256 = sha256_file(PROTOCOL)
    if REGISTRATION.exists() or REGISTRATION_MARKER.exists():
        if not REGISTRATION.is_file() or not REGISTRATION_MARKER.is_file():
            raise SourceRetentionV3Error("v3 registration is incomplete")
        lock = _load_mapping(REGISTRATION)
        marker = _load_mapping(REGISTRATION_MARKER)
        if (
            lock.get("protocol_sha256") != protocol_sha256
            or lock.get("v3_bootstrap_before_registration") is not False
            or marker.get("registration_sha256") != sha256_file(REGISTRATION)
        ):
            raise SourceRetentionV3Error("v3 registration changed")
        return lock
    if any(path.exists() for path in (ANALYSIS_LOCK, METRICS, STATISTICS, REPORT)):
        raise SourceRetentionV3Error("v3 output appeared before registration")
    lock = {
        "schema_version": 1,
        "status": "CVBRA_V1_HAZYDET_SOURCE_RETENTION_V3_REGISTERED",
        "registered_at_utc": _now(),
        "protocol": _relative(PROTOCOL),
        "protocol_sha256": protocol_sha256,
        "amendment_sha256": sha256_file(AMENDMENT),
        "v2_prediction_lock_sha256": V2_PREDICTION_LOCK_SHA256,
        "corrected_point_metrics_already_observed": True,
        "v2_bootstrap_already_observed_and_invalidated": True,
        "v3_bootstrap_before_registration": False,
        "method_or_hyperparameter_selection": False,
        "image_order": "lexicographic_string_sort",
        "bootstrap_encoding": "singleton_cluster_per_image",
        "models": list(MODELS),
    }
    atomic_write_json(REGISTRATION, lock)
    atomic_write_json(
        REGISTRATION_MARKER,
        {"status": lock["status"], "registration_sha256": sha256_file(REGISTRATION)},
    )
    return lock


def _implementation_lock() -> dict[str, Any]:
    registration = _registration()
    if IMPLEMENTATION_LOCK.exists() or IMPLEMENTATION_MARKER.exists():
        if not IMPLEMENTATION_LOCK.is_file() or not IMPLEMENTATION_MARKER.is_file():
            raise SourceRetentionV3Error("v3 implementation lock is incomplete")
        lock = _load_mapping(IMPLEMENTATION_LOCK)
        marker = _load_mapping(IMPLEMENTATION_MARKER)
        if (
            lock.get("runner_sha256") != sha256_file(Path(__file__))
            or lock.get("protocol_sha256") != sha256_file(PROTOCOL)
            or lock.get("registration_sha256") != sha256_file(REGISTRATION)
            or marker.get("implementation_lock_sha256") != sha256_file(IMPLEMENTATION_LOCK)
        ):
            raise SourceRetentionV3Error("v3 implementation lock changed")
        return lock
    if any(path.exists() for path in (ANALYSIS_LOCK, METRICS, STATISTICS, REPORT)):
        raise SourceRetentionV3Error("v3 output appeared before implementation lock")
    image_ids = ordered_image_ids()
    lock = {
        "schema_version": 1,
        "status": "CVBRA_V1_HAZYDET_SOURCE_RETENTION_V3_IMPLEMENTATION_LOCKED",
        "locked_at_utc": _now(),
        "runner": _relative(Path(__file__)),
        "runner_sha256": sha256_file(Path(__file__)),
        "protocol_sha256": sha256_file(PROTOCOL),
        "registration_sha256": sha256_file(REGISTRATION),
        "amendment_sha256": sha256_file(AMENDMENT),
        "v2_prediction_lock_sha256": V2_PREDICTION_LOCK_SHA256,
        "annotation_sha256": ANNOTATION_SHA256,
        "images": len(image_ids),
        "image_order": "lexicographic_string_sort",
        "first_image_ids": list(image_ids[:10]),
        "bootstrap_encoding": "singleton_cluster_per_image",
        "accelerated_AP_only_cache": True,
        "point_parity_tolerance": 1e-12,
        "resamples": RESAMPLES,
        "seed": SEED,
        "python": platform.python_version(),
        "registration_status": registration["status"],
        "method_or_hyperparameter_selection": False,
        "HazyDet_test_access": "prohibited",
        "UAV_OBB_test_access": "prohibited",
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
        "status": "PASS_CVBRA_V1_HAZYDET_SOURCE_RETENTION_V3_PREFLIGHT",
        "models": list(MODELS),
        "images": IMAGES,
        "resamples": RESAMPLES,
        "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
        "runner_sha256": lock["runner_sha256"],
    }


def _analysis_authorization() -> dict[str, Any]:
    _implementation_lock()
    if ANALYSIS_LOCK.exists() or ANALYSIS_MARKER.exists():
        if not ANALYSIS_LOCK.is_file() or not ANALYSIS_MARKER.is_file():
            raise SourceRetentionV3Error("v3 analysis authorization is incomplete")
        lock = _load_mapping(ANALYSIS_LOCK)
        marker = _load_mapping(ANALYSIS_MARKER)
        if (
            lock.get("runner_sha256") != sha256_file(Path(__file__))
            or lock.get("v2_prediction_lock_sha256") != V2_PREDICTION_LOCK_SHA256
            or lock.get("annotation_sha256") != ANNOTATION_SHA256
            or marker.get("analysis_lock_sha256") != sha256_file(ANALYSIS_LOCK)
        ):
            raise SourceRetentionV3Error("v3 analysis authorization changed")
        return lock
    if any(path.exists() for path in (METRICS, STATISTICS, REPORT)):
        raise SourceRetentionV3Error("v3 metric output appeared before analysis authorization")
    lock = {
        "schema_version": 1,
        "status": "HAZYDET_SOURCE_RETENTION_V3_ANALYSIS_AUTHORIZED",
        "authorized_at_utc": _now(),
        "runner_sha256": sha256_file(Path(__file__)),
        "protocol_sha256": sha256_file(PROTOCOL),
        "v2_prediction_lock_sha256": V2_PREDICTION_LOCK_SHA256,
        "annotation_sha256": ANNOTATION_SHA256,
        "image_order": "lexicographic_string_sort",
        "v3_metric_or_bootstrap_accessed_before_authorization": False,
        "validation_annotation_previously_accessed": True,
        "method_or_hyperparameter_selection": False,
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
        raise SourceRetentionV3Error("v2 prediction artifacts changed")
    paths = {
        str(row["model"]): _rooted(row["corrected_prediction"])
        for row in artifacts
        if isinstance(row, dict)
    }
    if tuple(paths) != MODELS:
        raise SourceRetentionV3Error("v2 prediction coverage changed")
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
        raise SourceRetentionV3Error(f"v3 bootstrap checkpoint is incomplete: {path}")
    return [float(value) for value in values]


def score() -> dict[str, Any]:
    v2_lock = _v2_prediction_lock()
    _analysis_authorization()
    if REPORT.exists():
        report = _load_mapping(REPORT)
        marker = _load_mapping(COMPLETE)
        if marker.get("report_sha256") != sha256_file(REPORT):
            raise SourceRetentionV3Error("existing v3 report is not locked")
        return report
    paths = _prediction_paths(v2_lock)
    timing = v2_lock.get("timing_median_mean_ms")
    if not isinstance(timing, dict):
        raise SourceRetentionV3Error("v2 timing metadata changed")
    image_ids = ordered_image_ids()
    clusters = singleton_image_clusters(image_ids)
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        metrics = evaluate_coco(ANNOTATION, paths[model], max_det=MAX_DET, image_ids=image_ids)
        if int(metrics["images_evaluated"]) != IMAGES:
            raise SourceRetentionV3Error(f"{model} did not evaluate the full split")
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
    if METRICS.exists() and METRICS.read_text(encoding="utf-8") != expected_metrics:
        raise SourceRetentionV3Error("resumable v3 metrics changed")
    if not METRICS.exists():
        atomic_write_text(METRICS, expected_metrics)
    by_model = {str(row["model"]): row for row in rows}
    source_ap_drift = float(by_model["source"]["AP"]) - HISTORICAL_SOURCE_AP
    if abs(source_ap_drift) > SOURCE_AP_MAX_ABS_DRIFT:
        raise SourceRetentionV3Error(f"source AP anchor failed: {source_ap_drift:+.9f}")
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
        result = paired_coco_ap_cluster_bootstrap_scopes(
            ANNOTATION,
            paths[baseline],
            paths[method],
            {
                "all_images": ClusterBootstrapScope(
                    clusters=clusters,
                    checkpoint_path=checkpoint,
                    checkpoint_identity={
                        "study": "cvbra_v1_hazydet_source_retention_v3",
                        "comparison": name,
                        "v2_prediction_lock_sha256": V2_PREDICTION_LOCK_SHA256,
                        "image_order": "lexicographic_string_sort",
                        "singleton_cluster_is_image_bootstrap": True,
                    },
                )
            },
            resamples=RESAMPLES,
            seed=SEED,
            max_det=MAX_DET,
            workers=4,
            chunk_resamples=100,
            accelerate_ap_only=True,
        )["all_images"]
        expected = point_deltas[name]
        if abs(float(result["delta"]) - expected) > 1e-12:
            raise SourceRetentionV3Error(f"v3 point parity failed: {name}")
        deltas = _checkpoint_deltas(checkpoint)
        standard_deviation = stdev(deltas)
        if standard_deviation <= 0.0 or not math.isfinite(standard_deviation):
            raise SourceRetentionV3Error(f"v3 bootstrap variance is invalid: {name}")
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
        "status": "HAZYDET_SOURCE_RETENTION_V3_PAIRED_STATISTICS_COMPLETE",
        "completed_at_utc": _now(),
        "resamples": RESAMPLES,
        "seed": SEED,
        "unit": "image_via_singleton_cluster_encoding",
        "images": IMAGES,
        "image_order": "lexicographic_string_sort",
        "multiple_testing": "Holm across five descriptive AP contrasts",
        "rows": statistics_rows,
    }
    atomic_write_json(STATISTICS, statistics)
    report = {
        "schema_version": 1,
        "status": "COMPLETE_CVBRA_V1_HAZYDET_SOURCE_RETENTION_V3_AUDIT",
        "completed_at_utc": _now(),
        "protocol_sha256": sha256_file(PROTOCOL),
        "runner": _relative(Path(__file__)),
        "runner_sha256": sha256_file(Path(__file__)),
        "amendment_sha256": sha256_file(AMENDMENT),
        "v2_prediction_lock_sha256": V2_PREDICTION_LOCK_SHA256,
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
            "order_parity_correction_only": True,
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
        description="Run order-parity-correct HazyDet source-retention statistics"
    )
    parser.add_argument("--stage", choices=("preflight", "score"), default="score")
    args = parser.parse_args()
    if args.stage == "preflight":
        result = preflight()
        artifact = IMPLEMENTATION_LOCK
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
