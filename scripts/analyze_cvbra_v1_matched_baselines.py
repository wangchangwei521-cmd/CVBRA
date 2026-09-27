from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Mapping, Sequence
from io import StringIO
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

from scripts import analyze_cvbra_v1_uav_obb_official_validation as primary_engine

from buse_uav.evaluation.bootstrap import (
    ClusterBootstrapScope,
    paired_coco_ap_cluster_bootstrap_scopes,
)
from buse_uav.evaluation.coco import evaluate_coco
from buse_uav.evaluation.supplementary import bootstrap_sign_pvalue, holm_adjust
from buse_uav.utils.hashing import sha256_file
from buse_uav.utils.io import atomic_write_json, atomic_write_text

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "experiment" / "cvbra_v1_matched_baselines.yaml"
PROTOCOL_SHA256 = "1671b6f13b9f1ffad0af6b2bcb20c5ac2c680e5368174505be8c884bf3dd7b9b"
REGISTRATION = (
    ROOT
    / "reports"
    / "development"
    / "cvbra_v1_matched_baselines"
    / "REGISTRATION_LOCK.json"
)
REGISTRATION_SHA256 = "aaf2e18153b10605926c4c93593db071ee7f0e503f54ef5da89a13d8f7a26278"
MATERIALIZATION_LOCK = (
    ROOT
    / "reports"
    / "development"
    / "cvbra_v1"
    / "uav_obb_official_validation"
    / "view_materialization_lock.json"
)
MATERIALIZATION_LOCK_SHA256 = "9f6c9db26806679b6853316156785a9b475afe3a4a24fbda1c8ef2725ebbc07d"
CONVERSION_LOCK = MATERIALIZATION_LOCK.parent / "annotations" / "conversion_lock.json"
CONVERSION_LOCK_SHA256 = "aee05f6635fd4d9f56cce016a8cfd79f9a91da966a6284278865ffcbcf59e179"
ANNOTATION = CONVERSION_LOCK.parent / "official_validation_exact_car_truck_bus_hbb.coco.json"
ANNOTATION_SHA256 = "21d2431e0c6d70b21dc9bf9a3da03ae19a2d6e1cba96530c8e4be44e8bfe945e"
PRIMARY_EVALUATION = (
    ROOT
    / "reports"
    / "development"
    / "cvbra_v1"
    / "uav_obb_official_validation"
    / "evaluation"
)
PRIMARY_REPORT = PRIMARY_EVALUATION / "validation_report.json"
PRIMARY_REPORT_SHA256 = "dbe2d7efb5f06d0251cde76fa6b8b9d3f4bf6f61e68652f0f9c9ed3bda81c29b"
PRIMARY_PREDICTION_LOCK = PRIMARY_EVALUATION / "joint_prediction_lock.json"
PRIMARY_PREDICTION_LOCK_SHA256 = "5014b03170c7223faa3228686db44593c31135f5a7bd81e9c2fc57a59e3eef97"
PRIMARY_PREDICTION_MARKER = PRIMARY_EVALUATION / "PREDICTIONS_LOCKED"

EVALUATION_ROOT = REGISTRATION.parent / "uav_obb_official_validation" / "evaluation"
BASELINE_PREDICTION_LOCK = EVALUATION_ROOT / "joint_prediction_lock.json"
BASELINE_PREDICTION_MARKER = EVALUATION_ROOT / "PREDICTIONS_LOCKED"
BASELINE_RAW_LOCK = EVALUATION_ROOT / "raw_observation_lock.json"
AUTHORIZATION = EVALUATION_ROOT / "analysis_authorization.json"
AUTHORIZATION_MARKER = EVALUATION_ROOT / "ANALYSIS_AUTHORIZED"
METRICS = EVALUATION_ROOT / "matched_metrics_full_and_decontaminated.csv"
POINT_REPORT = EVALUATION_ROOT / "matched_point_report.json"
POINT_MARKER = EVALUATION_ROOT / "MATCHED_POINTS_COMPLETE"
STATISTICS = EVALUATION_ROOT / "matched_statistics.json"
STATISTICS_MARKER = EVALUATION_ROOT / "MATCHED_STATISTICS_COMPLETE"
REPORT = EVALUATION_ROOT / "matched_baseline_report.json"
COMPLETE = EVALUATION_ROOT / "MATCHED_BASELINES_COMPLETE"

REFERENCE_MODELS = ("source", "CVBRA_v1")
BASELINES = ("STF", "CVBRA_noCV", "CVBRA_noReplay", "CVBRA_noFreeze")
MODELS = (*REFERENCE_MODELS, *BASELINES)
VIEWS = ("original", "fog_0p6", "fog_1p0")
METHODS = ("Identity", "Flip_HardNMS")
METRIC_KEYS = ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large")
PRIMARY_SCOPE = "decontaminated_primary"
SECONDARY_SCOPE = "official_full_secondary"
PRIMARY_VIEW = "fog_1p0"
IMAGES = 218
PRIMARY_IMAGES = 167
PRIMARY_GROUPS = 141
MAX_DET = 500
RESAMPLES = 10000
SEED = 20260814


class MatchedBaselineAnalysisError(RuntimeError):
    """Raised when the frozen matched-baseline analysis cannot fail closed."""


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MatchedBaselineAnalysisError(f"cannot parse mapping {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MatchedBaselineAnalysisError(f"expected mapping: {path}")
    return value


def _rooted(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _assert_hash(path: Path, expected: object, *, label: str) -> None:
    if not path.is_file() or sha256_file(path) != str(expected):
        raise MatchedBaselineAnalysisError(f"locked {label} changed: {path}")


def _validate_prediction_lock(
    path: Path,
    marker_path: Path,
    *,
    expected_models: Sequence[str],
    expected_sha256: str | None,
) -> dict[str, Any]:
    if expected_sha256 is not None:
        _assert_hash(path, expected_sha256, label="primary prediction lock")
    if not path.is_file() or not marker_path.is_file():
        raise MatchedBaselineAnalysisError(f"prediction lock is incomplete: {path}")
    lock = _load_mapping(path)
    marker = _load_mapping(marker_path)
    artifacts = lock.get("artifacts")
    expected_artifacts = len(expected_models) * len(VIEWS) * len(METHODS)
    if (
        lock.get("status")
        != "CVBRA_V1_UAV_OBB_OFFICIAL_VALIDATION_PREDICTIONS_LOCKED_BEFORE_LABELS_OR_METRICS"
        or marker.get("joint_prediction_lock_sha256") != sha256_file(path)
        or not isinstance(artifacts, list)
        or len(artifacts) != expected_artifacts
        or lock.get("official_test_content_accessed") is not False
    ):
        raise MatchedBaselineAnalysisError(f"prediction lock changed: {path}")
    observed = {
        str(row.get("model")) for row in artifacts if isinstance(row, dict)
    }
    if observed != set(expected_models):
        raise MatchedBaselineAnalysisError(f"prediction model coverage changed: {path}")
    for row in artifacts:
        if not isinstance(row, dict):
            raise MatchedBaselineAnalysisError("invalid prediction artifact")
        prediction = _rooted(row["prediction"])
        _assert_hash(prediction, row["prediction_sha256"], label="prediction artifact")
    return lock


def _validate_evidence() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    for path, digest, label in (
        (PROTOCOL, PROTOCOL_SHA256, "matched-baseline protocol"),
        (REGISTRATION, REGISTRATION_SHA256, "registration lock"),
        (MATERIALIZATION_LOCK, MATERIALIZATION_LOCK_SHA256, "materialization lock"),
        (CONVERSION_LOCK, CONVERSION_LOCK_SHA256, "conversion lock"),
        (ANNOTATION, ANNOTATION_SHA256, "validation annotation"),
        (PRIMARY_REPORT, PRIMARY_REPORT_SHA256, "primary validation report"),
        (PRIMARY_PREDICTION_LOCK, PRIMARY_PREDICTION_LOCK_SHA256, "primary prediction lock"),
    ):
        _assert_hash(path, digest, label=label)
    registration = _load_mapping(REGISTRATION)
    materialization = _load_mapping(MATERIALIZATION_LOCK)
    conversion = _load_mapping(CONVERSION_LOCK)
    if (
        registration.get("registered_baselines") != list(BASELINES)
        or registration.get("method_or_hyperparameter_selection_from_this_study") is not False
        or materialization.get("images") != IMAGES
        or materialization.get("primary_images") != PRIMARY_IMAGES
        or materialization.get("primary_source_groups") != PRIMARY_GROUPS
        or materialization.get("official_test_content_accessed") is not False
        or conversion.get("annotation_sha256") != ANNOTATION_SHA256
        or conversion.get("official_test_content_accessed") is not False
    ):
        raise MatchedBaselineAnalysisError("registered analysis evidence changed")
    primary = _validate_prediction_lock(
        PRIMARY_PREDICTION_LOCK,
        PRIMARY_PREDICTION_MARKER,
        expected_models=REFERENCE_MODELS,
        expected_sha256=PRIMARY_PREDICTION_LOCK_SHA256,
    )
    baseline = _validate_prediction_lock(
        BASELINE_PREDICTION_LOCK,
        BASELINE_PREDICTION_MARKER,
        expected_models=BASELINES,
        expected_sha256=None,
    )
    if (
        baseline.get("raw_observation_lock_sha256") != sha256_file(BASELINE_RAW_LOCK)
        or baseline.get("official_validation_labels_previously_accessed") is not True
        or baseline.get("labels_read_by_inference_runner") is not False
        or baseline.get("method_or_hyperparameter_selection") is not False
    ):
        raise MatchedBaselineAnalysisError("baseline prediction provenance changed")
    return materialization, primary, baseline


def _analysis_authorization() -> dict[str, Any]:
    _validate_evidence()
    if AUTHORIZATION.exists() or AUTHORIZATION_MARKER.exists():
        if not AUTHORIZATION.is_file() or not AUTHORIZATION_MARKER.is_file():
            raise MatchedBaselineAnalysisError("analysis authorization is incomplete")
        lock = _load_mapping(AUTHORIZATION)
        marker = _load_mapping(AUTHORIZATION_MARKER)
        if (
            lock.get("runner_sha256") != sha256_file(Path(__file__))
            or lock.get("baseline_prediction_lock_sha256")
            != sha256_file(BASELINE_PREDICTION_LOCK)
            or lock.get("primary_prediction_lock_sha256") != PRIMARY_PREDICTION_LOCK_SHA256
            or lock.get("annotation_sha256") != ANNOTATION_SHA256
            or marker.get("analysis_authorization_sha256") != sha256_file(AUTHORIZATION)
        ):
            raise MatchedBaselineAnalysisError("analysis authorization changed")
        return lock
    if any(path.exists() for path in (METRICS, POINT_REPORT, STATISTICS, REPORT)):
        raise MatchedBaselineAnalysisError("metric output appeared before analysis authorization")
    payload = {
        "schema_version": 1,
        "status": "MATCHED_BASELINE_ANALYSIS_AUTHORIZED_AFTER_ALL_PREDICTIONS_LOCKED",
        "authorized_at_utc": primary_engine._utc_now(),
        "protocol_sha256": PROTOCOL_SHA256,
        "registration_sha256": REGISTRATION_SHA256,
        "runner": _relative(Path(__file__)),
        "runner_sha256": sha256_file(Path(__file__)),
        "primary_prediction_lock_sha256": PRIMARY_PREDICTION_LOCK_SHA256,
        "baseline_prediction_lock_sha256": sha256_file(BASELINE_PREDICTION_LOCK),
        "annotation_sha256": ANNOTATION_SHA256,
        "validation_annotation_previously_accessed": True,
        "baseline_metric_accessed_before_authorization": False,
        "method_or_hyperparameter_selection": False,
        "independent_confirmation_claim": False,
        "official_test_access": "prohibited",
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


def _artifact_rows(lock: Mapping[str, Any]) -> list[dict[str, Any]]:
    artifacts = lock.get("artifacts")
    if not isinstance(artifacts, list):
        raise MatchedBaselineAnalysisError("prediction artifacts are invalid")
    return [dict(row) for row in artifacts if isinstance(row, dict)]


def _combined_predictions(
    primary: Mapping[str, Any], baseline: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    artifacts = [*_artifact_rows(primary), *_artifact_rows(baseline)]
    timing: dict[str, Any] = {}
    for lock in (primary, baseline):
        lock_timing = lock.get("timing")
        if not isinstance(lock_timing, dict):
            raise MatchedBaselineAnalysisError("prediction timing is invalid")
        for model, values in lock_timing.items():
            if str(model) in timing:
                raise MatchedBaselineAnalysisError(f"duplicate timing model: {model}")
            timing[str(model)] = values
    if len(artifacts) != len(MODELS) * len(VIEWS) * len(METHODS) or set(timing) != set(MODELS):
        raise MatchedBaselineAnalysisError("combined prediction coverage changed")
    return artifacts, timing


def _write_metrics(rows: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "scope",
        "model",
        "view",
        "method",
        *METRIC_KEYS,
        "images_evaluated",
        "mean_ms",
        "prediction_sha256",
    )
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row[key] for key in fields})
    atomic_write_text(METRICS, buffer.getvalue())


def _row_lookup(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str, str], Mapping[str, Any]]:
    return {
        (
            str(row["scope"]),
            str(row["model"]),
            str(row["view"]),
            str(row["method"]),
        ): row
        for row in rows
    }


def _metric_delta(
    lookup: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
    *,
    left: str,
    right: str,
    view: str,
    method: str,
    metric: str,
) -> float:
    return float(lookup[(PRIMARY_SCOPE, left, view, method)][metric]) - float(
        lookup[(PRIMARY_SCOPE, right, view, method)][metric]
    )


def points() -> dict[str, Any]:
    materialization, primary, baseline = _validate_evidence()
    _analysis_authorization()
    if POINT_REPORT.exists():
        report = _load_mapping(POINT_REPORT)
        if not POINT_MARKER.is_file() or _load_mapping(POINT_MARKER).get(
            "point_report_sha256"
        ) != sha256_file(POINT_REPORT):
            raise MatchedBaselineAnalysisError("existing point report is not locked")
        return report
    if any(path.exists() for path in (METRICS, POINT_MARKER, STATISTICS, REPORT)):
        raise MatchedBaselineAnalysisError("partial point analysis requires audit")
    artifacts, timing = _combined_predictions(primary, baseline)
    primary_ids_raw = materialization.get("primary_image_ids")
    if not isinstance(primary_ids_raw, list):
        raise MatchedBaselineAnalysisError("primary image registry is invalid")
    scope_ids = {
        PRIMARY_SCOPE: [int(value) for value in primary_ids_raw],
        SECONDARY_SCOPE: list(range(1, IMAGES + 1)),
    }
    rows: list[dict[str, Any]] = []
    for scope, image_ids in scope_ids.items():
        for artifact in artifacts:
            metrics = evaluate_coco(
                ANNOTATION,
                _rooted(artifact["prediction"]),
                max_det=MAX_DET,
                image_ids=image_ids,
            )
            model = str(artifact["model"])
            view = str(artifact["view"])
            method = str(artifact["method"])
            model_timing = timing[model]
            if not isinstance(model_timing, dict) or not isinstance(model_timing[view], dict):
                raise MatchedBaselineAnalysisError("timing payload is invalid")
            rows.append(
                {
                    "scope": scope,
                    "model": model,
                    "view": view,
                    "method": method,
                    **{key: float(metrics[key]) for key in METRIC_KEYS},
                    "images_evaluated": int(metrics["images_evaluated"]),
                    "mean_ms": float(model_timing[view][f"{method}_mean_ms"]),
                    "prediction_sha256": artifact["prediction_sha256"],
                }
            )
    expected_rows = 2 * len(MODELS) * len(VIEWS) * len(METHODS)
    if len(rows) != expected_rows:
        raise MatchedBaselineAnalysisError("matched metric coverage changed")
    _write_metrics(rows)
    lookup = _row_lookup(rows)
    cvbra_minus_baseline: dict[str, Any] = {}
    baseline_minus_source: dict[str, Any] = {}
    for candidate in BASELINES:
        cvbra_minus_baseline[candidate] = {}
        baseline_minus_source[candidate] = {}
        for method in METHODS:
            cvbra_minus_baseline[candidate][method] = {}
            baseline_minus_source[candidate][method] = {}
            for view in VIEWS:
                cvbra_minus_baseline[candidate][method][view] = {
                    metric: _metric_delta(
                        lookup,
                        left="CVBRA_v1",
                        right=candidate,
                        view=view,
                        method=method,
                        metric=metric,
                    )
                    for metric in ("AP", "AP50", "AP75")
                }
                baseline_minus_source[candidate][method][view] = {
                    metric: _metric_delta(
                        lookup,
                        left=candidate,
                        right="source",
                        view=view,
                        method=method,
                        metric=metric,
                    )
                    for metric in ("AP", "AP50", "AP75")
                }
    latency_ratios: dict[str, Any] = {}
    for candidate in BASELINES:
        latency_ratios[candidate] = {}
        for method in METHODS:
            key = f"{method}_mean_ms"
            latency_ratios[candidate][method] = {
                view: float(timing[candidate][view][key])
                / float(timing["CVBRA_v1"][view][key])
                for view in VIEWS
            }
    report = {
        "schema_version": 1,
        "status": "CVBRA_V1_MATCHED_BASELINE_POINT_ESTIMATES_LOCKED",
        "completed_at_utc": primary_engine._utc_now(),
        "protocol_sha256": PROTOCOL_SHA256,
        "analysis_authorization_sha256": sha256_file(AUTHORIZATION),
        "primary_prediction_lock_sha256": PRIMARY_PREDICTION_LOCK_SHA256,
        "baseline_prediction_lock_sha256": sha256_file(BASELINE_PREDICTION_LOCK),
        "annotation_sha256": ANNOTATION_SHA256,
        "metrics": _relative(METRICS),
        "metrics_sha256": sha256_file(METRICS),
        "models": list(MODELS),
        "rows": rows,
        "CVBRA_minus_baseline": cvbra_minus_baseline,
        "baseline_minus_source": baseline_minus_source,
        "baseline_to_CVBRA_latency_ratio": latency_ratios,
        "evidence_boundary": {
            "primary_scope": "167 validation images in 141 train-disjoint source-key groups",
            "secondary_scope": "all 218 official validation images",
            "validation_labels_previously_accessed": True,
            "configuration_frozen_before_baseline_training_and_prediction": True,
            "method_or_hyperparameter_selection": False,
            "independent_confirmation_claim": False,
            "official_test_access": "prohibited",
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


def _prediction_lookup(
    primary: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[tuple[str, str, str], Path]:
    rows = [*_artifact_rows(primary), *_artifact_rows(baseline)]
    return {
        (str(row["model"]), str(row["view"]), str(row["method"])): _rooted(
            row["prediction"]
        )
        for row in rows
    }


def _checkpoint_deltas(path: Path) -> list[float]:
    document = _load_mapping(path)
    values = document.get("deltas")
    if not isinstance(values, list) or len(values) != RESAMPLES:
        raise MatchedBaselineAnalysisError(f"bootstrap checkpoint is incomplete: {path}")
    return [float(value) for value in values]


def statistics() -> dict[str, Any]:
    point = points()
    _, primary, baseline = _validate_evidence()
    if STATISTICS.exists():
        report = _load_mapping(STATISTICS)
        if not STATISTICS_MARKER.is_file() or _load_mapping(STATISTICS_MARKER).get(
            "statistics_sha256"
        ) != sha256_file(STATISTICS):
            raise MatchedBaselineAnalysisError("existing statistics report is not locked")
        return report
    lookup = _prediction_lookup(primary, baseline)
    clusters = primary_engine._clusters(ANNOTATION)
    rows: list[dict[str, Any]] = []
    for candidate in BASELINES:
        for view in ("original", PRIMARY_VIEW):
            name = f"CVBRA_minus_{candidate}_Identity_{view}"
            checkpoint = EVALUATION_ROOT / "matched_bootstrap" / f"{name}.json"
            result = paired_coco_ap_cluster_bootstrap_scopes(
                ANNOTATION,
                lookup[(candidate, view, "Identity")],
                lookup[("CVBRA_v1", view, "Identity")],
                {
                    "primary": ClusterBootstrapScope(
                        clusters=clusters,
                        checkpoint_path=checkpoint,
                        checkpoint_identity={
                            "study": "cvbra_v1_matched_baselines",
                            "comparison": name,
                            "point_report_sha256": sha256_file(POINT_REPORT),
                            "post_freeze_descriptive_mechanism_evidence": True,
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
            expected = float(point["CVBRA_minus_baseline"][candidate]["Identity"][view]["AP"])
            if abs(float(result["delta"]) - expected) > 1e-10:
                raise MatchedBaselineAnalysisError(f"bootstrap point estimate drifted: {name}")
            deltas = _checkpoint_deltas(checkpoint)
            standard_deviation = stdev(deltas)
            if standard_deviation <= 0.0 or not math.isfinite(standard_deviation):
                raise MatchedBaselineAnalysisError(f"bootstrap variance is invalid: {name}")
            rows.append(
                {
                    "comparison": name,
                    "baseline": candidate,
                    "view": view,
                    "method": "Identity",
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
    adjusted = holm_adjust([float(row["p_two_sided"]) for row in rows])
    for row, value in zip(rows, adjusted, strict=True):
        row["holm_adjusted_p"] = value
        row["holm_significant_0p05"] = value < 0.05
    report = {
        "schema_version": 1,
        "status": "CVBRA_V1_MATCHED_BASELINE_STATISTICS_COMPLETE",
        "completed_at_utc": primary_engine._utc_now(),
        "protocol_sha256": PROTOCOL_SHA256,
        "point_report_sha256": sha256_file(POINT_REPORT),
        "unit": "canonical source-key group",
        "images": PRIMARY_IMAGES,
        "groups": PRIMARY_GROUPS,
        "resamples": RESAMPLES,
        "seed": SEED,
        "interval": "paired percentile 95 percent",
        "multiple_testing": "Holm across eight descriptive Identity AP contrasts",
        "effect_size": "point AP delta divided by bootstrap delta standard deviation",
        "rows": rows,
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
            raise MatchedBaselineAnalysisError("existing final report is not locked")
        return report
    stats_rows = stats.get("rows")
    if not isinstance(stats_rows, list) or len(stats_rows) != 8:
        raise MatchedBaselineAnalysisError("matched statistics coverage is incomplete")
    deltas = point["CVBRA_minus_baseline"]
    findings = {
        "CVBRA_above_STF_original_Identity_AP": (
            float(deltas["STF"]["Identity"]["original"]["AP"]) > 0.0
        ),
        "CVBRA_above_STF_fog1_Identity_AP": (
            float(deltas["STF"]["Identity"][PRIMARY_VIEW]["AP"]) > 0.0
        ),
        "cross_visibility_control_supports_CVBRA_on_fog1": (
            float(deltas["CVBRA_noCV"]["Identity"][PRIMARY_VIEW]["AP"]) > 0.0
        ),
        "source_replay_control_supports_CVBRA_on_original": (
            float(deltas["CVBRA_noReplay"]["Identity"]["original"]["AP"]) > 0.0
        ),
        "frozen_backbone_control_supports_CVBRA_on_original": (
            float(deltas["CVBRA_noFreeze"]["Identity"]["original"]["AP"]) > 0.0
        ),
    }
    report = {
        "schema_version": 1,
        "status": "COMPLETE_CVBRA_V1_MATCHED_BASELINE_EVIDENCE",
        "completed_at_utc": primary_engine._utc_now(),
        "protocol_sha256": PROTOCOL_SHA256,
        "runner": _relative(Path(__file__)),
        "runner_sha256": sha256_file(Path(__file__)),
        "point_report": _relative(POINT_REPORT),
        "point_report_sha256": sha256_file(POINT_REPORT),
        "statistics": _relative(STATISTICS),
        "statistics_sha256": sha256_file(STATISTICS),
        "findings": findings,
        "CVBRA_minus_baseline": deltas,
        "primary_statistics": stats_rows,
        "evidence_boundary": point["evidence_boundary"],
        "decision": {
            "descriptive_mechanism_study_complete": True,
            "main_method_reselection_authorized": False,
            "official_test_access_authorized": False,
            "paper_body_change_authorized": False,
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
    parser = argparse.ArgumentParser(description="Analyze CVBRA-v1 matched baselines")
    parser.add_argument("--stage", choices=("points", "statistics", "final"), default="final")
    args = parser.parse_args()
    if args.stage == "points":
        result = points()
        path = POINT_REPORT
    elif args.stage == "statistics":
        result = statistics()
        path = STATISTICS
    else:
        result = finalize()
        path = REPORT
    print(
        json.dumps(
            {"status": result["status"], "report_sha256": sha256_file(path)},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
