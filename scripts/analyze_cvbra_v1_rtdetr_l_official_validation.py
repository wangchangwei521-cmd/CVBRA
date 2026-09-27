from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

from scripts import analyze_cvbra_v1_uav_obb_official_validation as engine

from buse_uav.evaluation.bootstrap import (
    ClusterBootstrapScope,
    paired_coco_ap_cluster_bootstrap_scopes,
)
from buse_uav.evaluation.supplementary import bootstrap_sign_pvalue, holm_adjust
from buse_uav.utils.hashing import sha256_file
from buse_uav.utils.io import atomic_write_json

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "experiment" / "cvbra_v1_rtdetr_l_confirmation.yaml"
PROTOCOL_SHA256 = "6f11519c837a7b68679707cd0025631a90c073d8c37b856a9e0eb40b98fc7f9c"
MATERIALIZATION_LOCK = (
    ROOT
    / "reports"
    / "development"
    / "cvbra_v1"
    / "uav_obb_official_validation"
    / "view_materialization_lock.json"
)
CONVERSION_LOCK = (
    ROOT
    / "reports"
    / "development"
    / "cvbra_v1"
    / "uav_obb_official_validation"
    / "annotations"
    / "conversion_lock.json"
)
CONVERSION_MARKER = CONVERSION_LOCK.parent / "LABELS_CONVERTED_AND_LOCKED"
CONVERSION_LOCK_SHA256 = "aee05f6635fd4d9f56cce016a8cfd79f9a91da966a6284278865ffcbcf59e179"
ANNOTATION = CONVERSION_LOCK.parent / "official_validation_exact_car_truck_bus_hbb.coco.json"
ANNOTATION_SHA256 = "21d2431e0c6d70b21dc9bf9a3da03ae19a2d6e1cba96530c8e4be44e8bfe945e"
EVALUATION_ROOT = (
    ROOT
    / "reports"
    / "development"
    / "cvbra_v1_rtdetr_l"
    / "uav_obb_official_validation"
    / "evaluation"
)
PREDICTION_LOCK = EVALUATION_ROOT / "joint_prediction_lock.json"
PREDICTION_MARKER = EVALUATION_ROOT / "PREDICTIONS_LOCKED"
RAW_LOCK = EVALUATION_ROOT / "raw_observation_lock.json"
METRICS = EVALUATION_ROOT / "metrics_full_and_decontaminated.csv"
POINT_REPORT = EVALUATION_ROOT / "point_report.json"
POINT_MARKER = EVALUATION_ROOT / "POINTS_COMPLETE"
STATISTICS = EVALUATION_ROOT / "confirmatory_statistics.json"
STATISTICS_MARKER = EVALUATION_ROOT / "STATISTICS_COMPLETE"
REPORT = EVALUATION_ROOT / "validation_report.json"
COMPLETE = EVALUATION_ROOT / "VALIDATION_COMPLETE"
SUCCESS = EVALUATION_ROOT / "SUCCESS"
FAILURE = EVALUATION_ROOT / "FAILED_DIRECTION_CHECK"
ANNOTATION_REUSE_LOCK = EVALUATION_ROOT / "annotation_reuse_authorization.json"
ANNOTATION_REUSE_MARKER = EVALUATION_ROOT / "ANNOTATION_REUSE_AUTHORIZED"
CANDIDATE = "CVBRA_v1_RTDETR_L"
MODELS = ("source", CANDIDATE)
RESAMPLES = 10000
SEED = 20260814
MAX_DET = 500
PRIMARY_VIEW = "fog_1p0"
REUSED_ENGINE = ROOT / "scripts" / "analyze_cvbra_v1_uav_obb_official_validation.py"


class CVBRARTDETRAnalysisError(RuntimeError):
    """Raised when the fixed RT-DETR-L direction check cannot fail closed."""


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CVBRARTDETRAnalysisError(f"cannot parse mapping {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CVBRARTDETRAnalysisError(f"expected mapping: {path}")
    return value


def _delta(
    by_key: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
    *,
    scope: str,
    view: str,
    method: str,
    metric: str,
) -> float:
    return float(by_key[(scope, CANDIDATE, view, method)][metric]) - float(
        by_key[(scope, "source", view, method)][metric]
    )


def _maximum_latency_ratio(
    timing: Mapping[str, Any], *, method: str
) -> tuple[float, dict[str, float]]:
    key = f"{method}_mean_ms"
    ratios: dict[str, float] = {}
    for view in engine.VIEWS:
        source = timing["source"]
        candidate = timing[CANDIDATE]
        if not isinstance(source, dict) or not isinstance(candidate, dict):
            raise CVBRARTDETRAnalysisError("timing model mapping is invalid")
        source_view = source[view]
        candidate_view = candidate[view]
        if not isinstance(source_view, dict) or not isinstance(candidate_view, dict):
            raise CVBRARTDETRAnalysisError("timing view mapping is invalid")
        ratios[view] = float(candidate_view[key]) / float(source_view[key])
    return max(ratios.values()), ratios


def _annotation_reuse_lock() -> dict[str, Any]:
    if ANNOTATION_REUSE_LOCK.exists() or ANNOTATION_REUSE_MARKER.exists():
        if not ANNOTATION_REUSE_LOCK.is_file() or not ANNOTATION_REUSE_MARKER.is_file():
            raise CVBRARTDETRAnalysisError("annotation-reuse lock is incomplete")
        lock = _load_mapping(ANNOTATION_REUSE_LOCK)
        marker = _load_mapping(ANNOTATION_REUSE_MARKER)
        if (
            lock.get("runner_sha256") != sha256_file(Path(__file__))
            or lock.get("joint_prediction_lock_sha256") != sha256_file(PREDICTION_LOCK)
            or lock.get("annotation_sha256") != ANNOTATION_SHA256
            or marker.get("annotation_reuse_lock_sha256") != sha256_file(ANNOTATION_REUSE_LOCK)
        ):
            raise CVBRARTDETRAnalysisError("annotation-reuse lock changed")
        return lock
    if any(path.exists() for path in (METRICS, POINT_REPORT, STATISTICS, REPORT)):
        raise CVBRARTDETRAnalysisError("RT-DETR metric output appeared before annotation lock")
    payload = {
        "schema_version": 1,
        "status": (
            "AUTHORIZED_TO_REUSE_FROZEN_UAV_OBB_VALIDATION_ANNOTATION_"
            "AFTER_RTDETR_PREDICTION_LOCK"
        ),
        "authorized_at_utc": engine._utc_now(),
        "protocol_sha256": PROTOCOL_SHA256,
        "runner": engine._relative(Path(__file__)),
        "runner_sha256": sha256_file(Path(__file__)),
        "joint_prediction_lock_sha256": sha256_file(PREDICTION_LOCK),
        "historical_conversion_lock_sha256": CONVERSION_LOCK_SHA256,
        "annotation_sha256": ANNOTATION_SHA256,
        "annotation_newly_opened": False,
        "annotation_previously_accessed_for_YOLO11n": True,
        "RTDETR_metric_accessed_before_authorization": False,
        "RTDETR_endpoint_selected_by_annotation_or_metric": False,
        "official_test_accessed": False,
        "paper_body_change_authorized": False,
    }
    atomic_write_json(ANNOTATION_REUSE_LOCK, payload)
    atomic_write_json(
        ANNOTATION_REUSE_MARKER,
        {
            "status": payload["status"],
            "annotation_reuse_lock_sha256": sha256_file(ANNOTATION_REUSE_LOCK),
        },
    )
    return payload


def _validate_evidence() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    if not PROTOCOL.is_file() or sha256_file(PROTOCOL) != PROTOCOL_SHA256:
        raise CVBRARTDETRAnalysisError("RT-DETR protocol changed")
    if not CONVERSION_LOCK.is_file() or sha256_file(CONVERSION_LOCK) != CONVERSION_LOCK_SHA256:
        raise CVBRARTDETRAnalysisError("historical conversion lock changed")
    if not ANNOTATION.is_file() or sha256_file(ANNOTATION) != ANNOTATION_SHA256:
        raise CVBRARTDETRAnalysisError("frozen validation annotation changed")
    materialization = _load_mapping(MATERIALIZATION_LOCK)
    prediction = _load_mapping(PREDICTION_LOCK)
    prediction_marker = _load_mapping(PREDICTION_MARKER)
    raw = _load_mapping(RAW_LOCK)
    conversion = _load_mapping(CONVERSION_LOCK)
    artifacts = prediction.get("artifacts")
    if (
        materialization.get("images") != engine.IMAGES
        or materialization.get("primary_images") != engine.PRIMARY_IMAGES
        or materialization.get("primary_source_groups") != engine.PRIMARY_GROUPS
        or materialization.get("official_test_content_accessed") is not False
    ):
        raise CVBRARTDETRAnalysisError("materialization evidence changed")
    if (
        prediction.get("status")
        != "CVBRA_V1_UAV_OBB_OFFICIAL_VALIDATION_PREDICTIONS_LOCKED_BEFORE_LABELS_OR_METRICS"
        or prediction_marker.get("joint_prediction_lock_sha256") != sha256_file(PREDICTION_LOCK)
        or prediction.get("raw_observation_lock_sha256") != sha256_file(RAW_LOCK)
        or prediction.get("aggregate_metrics_accessed") is not False
        or prediction.get("official_validation_labels_previously_accessed") is not True
        or prediction.get("labels_read_by_inference_runner") is not False
        or prediction.get("official_test_content_accessed") is not False
        or not isinstance(artifacts, list)
        or len(artifacts) != 12
    ):
        raise CVBRARTDETRAnalysisError("RT-DETR prediction evidence changed")
    if (
        raw.get("status") != "ALL_CVBRA_V1_UAV_OBB_OFFICIAL_VALIDATION_RAW_OBSERVATIONS_LOCKED"
        or raw.get("prediction_determinism") != "exact_sha256_match_across_two_repetitions"
        or raw.get("aggregate_metrics_accessed") is not False
    ):
        raise CVBRARTDETRAnalysisError("RT-DETR raw evidence changed")
    if (
        conversion.get("status") != "CVBRA_UAV_OBB_OFFICIAL_VALIDATION_LABELS_CONVERTED_AND_LOCKED"
        or conversion.get("annotation_sha256") != ANNOTATION_SHA256
        or conversion.get("official_test_content_accessed") is not False
    ):
        raise CVBRARTDETRAnalysisError("historical annotation evidence changed")
    for row in artifacts:
        if not isinstance(row, dict):
            raise CVBRARTDETRAnalysisError("invalid RT-DETR prediction artifact")
        path = engine._rooted(row["prediction"])
        if not path.is_file() or sha256_file(path) != str(row["prediction_sha256"]):
            raise CVBRARTDETRAnalysisError(f"RT-DETR prediction changed: {path}")
    _annotation_reuse_lock()
    return materialization, prediction, conversion, ANNOTATION


def _configure_engine() -> None:
    assignments = {
        "PROTOCOL": PROTOCOL,
        "PROTOCOL_SHA256": PROTOCOL_SHA256,
        "MATERIALIZATION_LOCK": MATERIALIZATION_LOCK,
        "EVALUATION_ROOT": EVALUATION_ROOT,
        "PREDICTION_LOCK": PREDICTION_LOCK,
        "PREDICTION_MARKER": PREDICTION_MARKER,
        "RAW_LOCK": RAW_LOCK,
        "CONVERSION_LOCK": CONVERSION_LOCK,
        "CONVERSION_MARKER": CONVERSION_MARKER,
        "METRICS": METRICS,
        "POINT_REPORT": POINT_REPORT,
        "POINT_MARKER": POINT_MARKER,
        "STATISTICS": STATISTICS,
        "STATISTICS_MARKER": STATISTICS_MARKER,
        "REPORT": REPORT,
        "COMPLETE": COMPLETE,
        "SUCCESS": SUCCESS,
        "FAILURE": FAILURE,
        "MODELS": MODELS,
        "_delta": _delta,
        "_maximum_latency_ratio": _maximum_latency_ratio,
        "_validate_evidence": _validate_evidence,
    }
    for name, value in assignments.items():
        setattr(engine, name, value)


def points() -> dict[str, Any]:
    _configure_engine()
    result = engine.points()
    result["detector"] = "RT-DETR-L"
    result["candidate"] = CANDIDATE
    result["reused_analysis_engine_sha256"] = sha256_file(REUSED_ENGINE)
    atomic_write_json(POINT_REPORT, result)
    atomic_write_json(
        POINT_MARKER,
        {"status": result["status"], "point_report_sha256": sha256_file(POINT_REPORT)},
    )
    return result


def statistics() -> dict[str, Any]:
    point = points()
    _configure_engine()
    _, prediction, _, annotation = engine._validate_evidence()
    if STATISTICS.exists():
        report = _load_mapping(STATISTICS)
        if not STATISTICS_MARKER.is_file() or _load_mapping(STATISTICS_MARKER).get(
            "statistics_sha256"
        ) != sha256_file(STATISTICS):
            raise CVBRARTDETRAnalysisError("existing statistics report is not locked")
        return report
    artifacts = prediction.get("artifacts")
    if not isinstance(artifacts, list):
        raise CVBRARTDETRAnalysisError("prediction artifacts are invalid")
    lookup = {
        (str(row["model"]), str(row["view"]), str(row["method"])): engine._rooted(row["prediction"])
        for row in artifacts
        if isinstance(row, dict)
    }
    clusters = engine._clusters(annotation)
    specs = (
        ("Identity_original", "original", "Identity"),
        ("Identity_primary_fog", PRIMARY_VIEW, "Identity"),
        ("Flip_original", "original", "Flip_HardNMS"),
        ("Flip_primary_fog", PRIMARY_VIEW, "Flip_HardNMS"),
    )
    rows: list[dict[str, Any]] = []
    for name, view, method in specs:
        checkpoint = EVALUATION_ROOT / "bootstrap" / f"{name}.json"
        result = paired_coco_ap_cluster_bootstrap_scopes(
            annotation,
            lookup[("source", view, method)],
            lookup[(CANDIDATE, view, method)],
            {
                "primary": ClusterBootstrapScope(
                    clusters=clusters,
                    checkpoint_path=checkpoint,
                    checkpoint_identity={
                        "study": "cvbra_v1_rtdetr_l_uav_obb_official_validation",
                        "comparison": name,
                        "point_report_sha256": sha256_file(POINT_REPORT),
                        "cross_detector_confirmation_only": True,
                    },
                )
            },
            resamples=RESAMPLES,
            seed=SEED,
            max_det=MAX_DET,
            workers=4,
            chunk_resamples=100,
            accelerate_ap_only=True,
        )["primary"]
        expected_delta = float(point["deltas"][f"{name}_AP"])
        if abs(float(result["delta"]) - expected_delta) > 1e-10:
            raise CVBRARTDETRAnalysisError(f"bootstrap point estimate drifted: {name}")
        deltas = engine._checkpoint_deltas(checkpoint)
        standard_deviation = stdev(deltas)
        if standard_deviation <= 0.0 or not math.isfinite(standard_deviation):
            raise CVBRARTDETRAnalysisError(f"bootstrap variance is invalid: {name}")
        rows.append(
            {
                "comparison": name,
                "view": view,
                "method": method,
                **result,
                "bootstrap_mean_delta": fmean(deltas),
                "bootstrap_standard_deviation": standard_deviation,
                "standardized_effect": expected_delta / standard_deviation,
                "p_two_sided": bootstrap_sign_pvalue(deltas),
                "checkpoint": engine._relative(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
            }
        )
        print(json.dumps({"bootstrap_complete": name}), flush=True)
    adjusted = holm_adjust([float(row["p_two_sided"]) for row in rows])
    for row, value in zip(rows, adjusted, strict=True):
        row["holm_adjusted_p"] = value
        row["holm_significant_0p05"] = value < 0.05
    report = {
        "schema_version": 1,
        "status": "CVBRA_V1_RTDETR_L_CONFIRMATORY_STATISTICS_COMPLETE",
        "completed_at_utc": engine._utc_now(),
        "protocol_sha256": PROTOCOL_SHA256,
        "point_report_sha256": sha256_file(POINT_REPORT),
        "detector": "RT-DETR-L",
        "candidate": CANDIDATE,
        "unit": "canonical source-key group",
        "images": engine.PRIMARY_IMAGES,
        "groups": engine.PRIMARY_GROUPS,
        "resamples": RESAMPLES,
        "seed": SEED,
        "interval": "paired percentile 95 percent",
        "multiple_testing": "Holm across four registered AP direction checks",
        "effect_size": "point AP delta divided by bootstrap delta standard deviation",
        "rows": rows,
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
            "validation_report_sha256"
        ) != sha256_file(REPORT):
            raise CVBRARTDETRAnalysisError("existing validation report is not locked")
        return report
    stats_rows = stats.get("rows")
    if not isinstance(stats_rows, list):
        raise CVBRARTDETRAnalysisError("statistics rows are invalid")
    direction_keys = (
        "Identity_original_AP",
        "Identity_primary_fog_AP",
        "Identity_primary_fog_AP75",
        "Flip_original_AP",
        "Flip_primary_fog_AP",
    )
    checks = {
        **{f"{key}_nonnegative": float(point["deltas"][key]) >= 0.0 for key in direction_keys},
        "Identity_latency_ratio_at_most_1p05": (
            float(point["latency"]["Identity_maximum_CVBRA_to_source_ratio"]) <= 1.05
        ),
        "Flip_latency_ratio_at_most_1p05": (
            float(point["latency"]["Flip_maximum_CVBRA_to_source_ratio"]) <= 1.05
        ),
        "prediction_reproducibility_exact": True,
        "four_registered_bootstraps_complete": len(stats_rows) == 4,
    }
    passed = all(checks.values())
    report = {
        "schema_version": 1,
        "status": (
            "PASS_CVBRA_V1_RTDETR_L_CROSS_DETECTOR_DIRECTION_CHECK"
            if passed
            else "FAIL_CVBRA_V1_RTDETR_L_CROSS_DETECTOR_DIRECTION_CHECK"
        ),
        "completed_at_utc": engine._utc_now(),
        "protocol_sha256": PROTOCOL_SHA256,
        "runner": engine._relative(Path(__file__)),
        "runner_sha256": sha256_file(Path(__file__)),
        "reused_analysis_engine_sha256": sha256_file(REUSED_ENGINE),
        "point_report": engine._relative(POINT_REPORT),
        "point_report_sha256": sha256_file(POINT_REPORT),
        "statistics": engine._relative(STATISTICS),
        "statistics_sha256": sha256_file(STATISTICS),
        "checks": checks,
        "all_checks_pass": passed,
        "primary_deltas": point["deltas"],
        "primary_statistics": stats_rows,
        "latency": point["latency"],
        "evidence_boundary": {
            **point["evidence_boundary"],
            "official_validation_labels_previously_accessed_for_YOLO11n": True,
            "RTDETR_endpoint_selected_by_validation_metric": False,
            "universal_stability_claim": False,
        },
        "decision": {
            "cross_detector_direction_confirmed": passed,
            "official_test_access_authorized": False,
            "paper_body_change_authorized": False,
        },
    }
    atomic_write_json(REPORT, report)
    marker = {
        "status": report["status"],
        "validation_report_sha256": sha256_file(REPORT),
        "point_report_sha256": sha256_file(POINT_REPORT),
        "statistics_sha256": sha256_file(STATISTICS),
    }
    atomic_write_json(COMPLETE, marker)
    atomic_write_json(SUCCESS if passed else FAILURE, marker)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze fixed CVBRA RT-DETR-L evidence")
    parser.add_argument("--stage", choices=("points", "statistics", "final"), default="final")
    args = parser.parse_args()
    if args.stage == "points":
        result = points()
        report_path = POINT_REPORT
    elif args.stage == "statistics":
        result = statistics()
        report_path = STATISTICS
    else:
        result = finalize()
        report_path = REPORT
    print(
        json.dumps(
            {"status": result["status"], "report_sha256": sha256_file(report_path)},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
