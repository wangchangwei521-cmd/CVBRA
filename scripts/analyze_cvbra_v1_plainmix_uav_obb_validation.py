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
PROTOCOL = ROOT / "configs/experiment/cvbra_v1_acceptance_upgrade_v1.yaml"
PROTOCOL_SHA256 = "2b0dd70e4c67c2afcca2dafea3a4d5e885ce233e45681db691ce9422ca78e73b"
REGISTRATION = (
    ROOT / "reports/development/cvbra_v1_acceptance_upgrade/PlainMix/REGISTRATION_LOCK.json"
)
REGISTRATION_SHA256 = "9bd0e3d383d23a8bfb14d18c4298fca8e445d82ae5a7a749acfe22405886fc85"
MATERIALIZATION_LOCK = (
    ROOT
    / "reports/development/cvbra_v1/uav_obb_official_validation/view_materialization_lock.json"
)
MATERIALIZATION_LOCK_SHA256 = (
    "9f6c9db26806679b6853316156785a9b475afe3a4a24fbda1c8ef2725ebbc07d"
)
CONVERSION_LOCK = MATERIALIZATION_LOCK.parent / "annotations/conversion_lock.json"
CONVERSION_LOCK_SHA256 = (
    "aee05f6635fd4d9f56cce016a8cfd79f9a91da966a6284278865ffcbcf59e179"
)
ANNOTATION = CONVERSION_LOCK.parent / "official_validation_exact_car_truck_bus_hbb.coco.json"
ANNOTATION_SHA256 = "21d2431e0c6d70b21dc9bf9a3da03ae19a2d6e1cba96530c8e4be44e8bfe945e"

PRIMARY_EVALUATION = (
    ROOT / "reports/development/cvbra_v1/uav_obb_official_validation/evaluation"
)
PRIMARY_PREDICTION_LOCK = PRIMARY_EVALUATION / "joint_prediction_lock.json"
PRIMARY_PREDICTION_LOCK_SHA256 = (
    "5014b03170c7223faa3228686db44593c31135f5a7bd81e9c2fc57a59e3eef97"
)
PRIMARY_PREDICTION_MARKER = PRIMARY_EVALUATION / "PREDICTIONS_LOCKED"

MATCHED_EVALUATION = (
    ROOT
    / "reports/development/cvbra_v1_matched_baselines/uav_obb_official_validation/evaluation"
)
MATCHED_PREDICTION_LOCK = MATCHED_EVALUATION / "joint_prediction_lock.json"
MATCHED_PREDICTION_LOCK_SHA256 = (
    "40e426b49314820904e84bbf9ee7c8b461bdab42f833e421327c0d756d960025"
)
MATCHED_PREDICTION_MARKER = MATCHED_EVALUATION / "PREDICTIONS_LOCKED"

OUTPUT = (
    ROOT
    / "reports/development/cvbra_v1_acceptance_upgrade/PlainMix"
    / "uav_obb_official_validation/evaluation"
)
PLAINMIX_PREDICTION_LOCK = OUTPUT / "joint_prediction_lock.json"
PLAINMIX_PREDICTION_MARKER = OUTPUT / "PREDICTIONS_LOCKED"
ANALYSIS_LOCK = OUTPUT / "analysis_authorization.json"
ANALYSIS_MARKER = OUTPUT / "ANALYSIS_AUTHORIZED"
METRICS = OUTPUT / "plainmix_metrics_full_and_decontaminated.csv"
POINT_REPORT = OUTPUT / "plainmix_point_report.json"
POINT_MARKER = OUTPUT / "PLAINMIX_POINTS_COMPLETE"
STATISTICS = OUTPUT / "plainmix_statistics.json"
STATISTICS_MARKER = OUTPUT / "PLAINMIX_STATISTICS_COMPLETE"
REPORT = OUTPUT / "plainmix_validation_report.json"
COMPLETE = OUTPUT / "PLAINMIX_VALIDATION_COMPLETE"

MODELS = ("source", "CVBRA_v1", "CVBRA_noCV", "PlainMix")
VIEWS = ("original", "fog_0p6", "fog_1p0")
METHODS = ("Identity", "Flip_HardNMS")
COMPARISONS = (
    ("CVBRA_noCV", "PlainMix", "PlainMix_minus_CVBRA_noCV"),
    ("PlainMix", "CVBRA_v1", "CVBRA_v1_minus_PlainMix"),
)
METRIC_KEYS = ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large")
PRIMARY_SCOPE = "decontaminated_primary"
SECONDARY_SCOPE = "official_full_secondary"
PRIMARY_VIEWS = ("original", "fog_1p0")
IMAGES = 218
PRIMARY_IMAGES = 167
PRIMARY_GROUPS = 141
MAX_DET = 500
RESAMPLES = 10000
SEED = 20260821


class PlainMixUAVOBBAnalysisError(RuntimeError):
    """Raised when the registered PlainMix UAV-OBB analysis cannot fail closed."""


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PlainMixUAVOBBAnalysisError(f"cannot parse mapping {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PlainMixUAVOBBAnalysisError(f"expected mapping: {path}")
    return value


def _rooted(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _assert_hash(path: Path, expected: object, *, label: str) -> None:
    if not path.is_file() or sha256_file(path) != str(expected):
        raise PlainMixUAVOBBAnalysisError(f"locked {label} changed: {path}")


def _validate_prediction_lock(
    path: Path,
    marker_path: Path,
    *,
    expected_models: Sequence[str],
    expected_sha256: str | None,
) -> dict[str, Any]:
    if expected_sha256 is not None:
        _assert_hash(path, expected_sha256, label="prediction lock")
    if not path.is_file() or not marker_path.is_file():
        raise PlainMixUAVOBBAnalysisError(f"prediction lock is incomplete: {path}")
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
        raise PlainMixUAVOBBAnalysisError(f"prediction lock changed: {path}")
    observed = {str(row.get("model")) for row in artifacts if isinstance(row, dict)}
    if observed != set(expected_models):
        raise PlainMixUAVOBBAnalysisError(f"prediction model coverage changed: {path}")
    for row in artifacts:
        if not isinstance(row, dict):
            raise PlainMixUAVOBBAnalysisError("invalid prediction artifact")
        _assert_hash(
            _rooted(row["prediction"]), row["prediction_sha256"], label="prediction artifact"
        )
    return lock


def _validate_evidence() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    for path, digest, label in (
        (PROTOCOL, PROTOCOL_SHA256, "acceptance-upgrade protocol"),
        (REGISTRATION, REGISTRATION_SHA256, "PlainMix registration"),
        (MATERIALIZATION_LOCK, MATERIALIZATION_LOCK_SHA256, "materialization lock"),
        (CONVERSION_LOCK, CONVERSION_LOCK_SHA256, "annotation conversion lock"),
        (ANNOTATION, ANNOTATION_SHA256, "UAV-OBB validation annotation"),
    ):
        _assert_hash(path, digest, label=label)
    registration = _load_mapping(REGISTRATION)
    materialization = _load_mapping(MATERIALIZATION_LOCK)
    conversion = _load_mapping(CONVERSION_LOCK)
    if (
        registration.get("baseline") != "PlainMix"
        or registration.get("evaluation_contract", {}).get("official_test_access")
        != "prohibited"
        or materialization.get("images") != IMAGES
        or materialization.get("primary_images") != PRIMARY_IMAGES
        or materialization.get("primary_source_groups") != PRIMARY_GROUPS
        or conversion.get("annotation_sha256") != ANNOTATION_SHA256
    ):
        raise PlainMixUAVOBBAnalysisError("registered UAV-OBB analysis scope changed")
    primary = _validate_prediction_lock(
        PRIMARY_PREDICTION_LOCK,
        PRIMARY_PREDICTION_MARKER,
        expected_models=("source", "CVBRA_v1"),
        expected_sha256=PRIMARY_PREDICTION_LOCK_SHA256,
    )
    matched = _validate_prediction_lock(
        MATCHED_PREDICTION_LOCK,
        MATCHED_PREDICTION_MARKER,
        expected_models=("STF", "CVBRA_noCV", "CVBRA_noReplay", "CVBRA_noFreeze"),
        expected_sha256=MATCHED_PREDICTION_LOCK_SHA256,
    )
    plainmix = _validate_prediction_lock(
        PLAINMIX_PREDICTION_LOCK,
        PLAINMIX_PREDICTION_MARKER,
        expected_models=("PlainMix",),
        expected_sha256=None,
    )
    if (
        plainmix.get("official_validation_labels_previously_accessed") is not True
        or plainmix.get("labels_read_by_inference_runner") is not False
        or plainmix.get("metric_or_selection_feedback_used") is not False
    ):
        raise PlainMixUAVOBBAnalysisError("PlainMix prediction provenance changed")
    return materialization, primary, matched, plainmix


def _analysis_authorization() -> dict[str, Any]:
    _validate_evidence()
    if ANALYSIS_LOCK.exists() or ANALYSIS_MARKER.exists():
        if not ANALYSIS_LOCK.is_file() or not ANALYSIS_MARKER.is_file():
            raise PlainMixUAVOBBAnalysisError("PlainMix analysis authorization is incomplete")
        lock = _load_mapping(ANALYSIS_LOCK)
        marker = _load_mapping(ANALYSIS_MARKER)
        if (
            lock.get("runner_sha256") != sha256_file(Path(__file__))
            or lock.get("plainmix_prediction_lock_sha256")
            != sha256_file(PLAINMIX_PREDICTION_LOCK)
            or lock.get("annotation_sha256") != ANNOTATION_SHA256
            or marker.get("analysis_authorization_sha256") != sha256_file(ANALYSIS_LOCK)
        ):
            raise PlainMixUAVOBBAnalysisError("PlainMix analysis authorization changed")
        return lock
    if any(path.exists() for path in (METRICS, POINT_REPORT, STATISTICS, REPORT)):
        raise PlainMixUAVOBBAnalysisError("metric output appeared before analysis authorization")
    payload = {
        "schema_version": 1,
        "status": "PLAINMIX_UAV_OBB_ANALYSIS_AUTHORIZED_AFTER_ALL_PREDICTIONS_LOCKED",
        "authorized_at_utc": primary_engine._utc_now(),
        "protocol_sha256": PROTOCOL_SHA256,
        "registration_sha256": REGISTRATION_SHA256,
        "runner": _relative(Path(__file__)),
        "runner_sha256": sha256_file(Path(__file__)),
        "primary_prediction_lock_sha256": PRIMARY_PREDICTION_LOCK_SHA256,
        "matched_prediction_lock_sha256": MATCHED_PREDICTION_LOCK_SHA256,
        "plainmix_prediction_lock_sha256": sha256_file(PLAINMIX_PREDICTION_LOCK),
        "annotation_sha256": ANNOTATION_SHA256,
        "validation_annotation_previously_accessed": True,
        "metric_accessed_before_authorization": False,
        "method_or_hyperparameter_selection": False,
        "independent_confirmation_claim": False,
        "official_test_access": "prohibited",
    }
    atomic_write_json(ANALYSIS_LOCK, payload)
    atomic_write_json(
        ANALYSIS_MARKER,
        {
            "status": payload["status"],
            "analysis_authorization_sha256": sha256_file(ANALYSIS_LOCK),
        },
    )
    return payload


def _artifact_rows(lock: Mapping[str, Any]) -> list[dict[str, Any]]:
    artifacts = lock.get("artifacts")
    if not isinstance(artifacts, list):
        raise PlainMixUAVOBBAnalysisError("prediction artifacts are invalid")
    return [dict(row) for row in artifacts if isinstance(row, dict)]


def _combined_predictions(
    primary: Mapping[str, Any],
    matched: Mapping[str, Any],
    plainmix: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_rows = [*_artifact_rows(primary), *_artifact_rows(matched), *_artifact_rows(plainmix)]
    artifacts = [row for row in all_rows if str(row["model"]) in MODELS]
    timing: dict[str, Any] = {}
    for lock in (primary, matched, plainmix):
        lock_timing = lock.get("timing")
        if not isinstance(lock_timing, dict):
            raise PlainMixUAVOBBAnalysisError("prediction timing is invalid")
        for model, values in lock_timing.items():
            if str(model) in MODELS:
                if str(model) in timing:
                    raise PlainMixUAVOBBAnalysisError(f"duplicate timing model: {model}")
                timing[str(model)] = values
    if len(artifacts) != len(MODELS) * len(VIEWS) * len(METHODS) or set(timing) != set(MODELS):
        raise PlainMixUAVOBBAnalysisError("combined prediction coverage changed")
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
        (str(row["scope"]), str(row["model"]), str(row["view"]), str(row["method"])): row
        for row in rows
    }


def points() -> dict[str, Any]:
    materialization, primary, matched, plainmix = _validate_evidence()
    _analysis_authorization()
    if POINT_REPORT.exists():
        report = _load_mapping(POINT_REPORT)
        if not POINT_MARKER.is_file() or _load_mapping(POINT_MARKER).get(
            "point_report_sha256"
        ) != sha256_file(POINT_REPORT):
            raise PlainMixUAVOBBAnalysisError("existing point report is not locked")
        return report
    if any(path.exists() for path in (METRICS, POINT_MARKER, STATISTICS, REPORT)):
        raise PlainMixUAVOBBAnalysisError("partial point analysis requires audit")
    artifacts, timing = _combined_predictions(primary, matched, plainmix)
    primary_ids_raw = materialization.get("primary_image_ids")
    if not isinstance(primary_ids_raw, list):
        raise PlainMixUAVOBBAnalysisError("primary image registry is invalid")
    scope_ids = {
        PRIMARY_SCOPE: [int(value) for value in primary_ids_raw],
        SECONDARY_SCOPE: list(range(1, IMAGES + 1)),
    }
    rows: list[dict[str, Any]] = []
    for scope, image_ids in scope_ids.items():
        for artifact in artifacts:
            result = evaluate_coco(
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
                raise PlainMixUAVOBBAnalysisError("timing payload is invalid")
            rows.append(
                {
                    "scope": scope,
                    "model": model,
                    "view": view,
                    "method": method,
                    **{key: float(result[key]) for key in METRIC_KEYS},
                    "images_evaluated": int(result["images_evaluated"]),
                    "mean_ms": float(model_timing[view][f"{method}_mean_ms"]),
                    "prediction_sha256": artifact["prediction_sha256"],
                }
            )
    expected_rows = 2 * len(MODELS) * len(VIEWS) * len(METHODS)
    if len(rows) != expected_rows:
        raise PlainMixUAVOBBAnalysisError("PlainMix metric coverage changed")
    _write_metrics(rows)
    lookup = _row_lookup(rows)
    deltas: dict[str, dict[str, float]] = {}
    for baseline, method, name in COMPARISONS:
        deltas[name] = {
            view: float(lookup[(PRIMARY_SCOPE, method, view, "Identity")]["AP"])
            - float(lookup[(PRIMARY_SCOPE, baseline, view, "Identity")]["AP"])
            for view in PRIMARY_VIEWS
        }
    plainmix_minus_source = {
        view: float(lookup[(PRIMARY_SCOPE, "PlainMix", view, "Identity")]["AP"])
        - float(lookup[(PRIMARY_SCOPE, "source", view, "Identity")]["AP"])
        for view in PRIMARY_VIEWS
    }
    report = {
        "schema_version": 1,
        "status": "PLAINMIX_UAV_OBB_POINT_ESTIMATES_LOCKED",
        "completed_at_utc": primary_engine._utc_now(),
        "protocol_sha256": PROTOCOL_SHA256,
        "analysis_authorization_sha256": sha256_file(ANALYSIS_LOCK),
        "plainmix_prediction_lock_sha256": sha256_file(PLAINMIX_PREDICTION_LOCK),
        "annotation_sha256": ANNOTATION_SHA256,
        "metrics": _relative(METRICS),
        "metrics_sha256": sha256_file(METRICS),
        "models": list(MODELS),
        "rows": rows,
        "registered_AP_deltas": deltas,
        "PlainMix_minus_source_AP": plainmix_minus_source,
        "evidence_boundary": {
            "primary_scope": "167 validation images in 141 train-disjoint source-key groups",
            "secondary_scope": "all 218 official validation images",
            "validation_labels_previously_accessed": True,
            "configuration_frozen_before_prediction": True,
            "method_or_hyperparameter_selection": False,
            "independent_confirmation_claim": False,
            "official_test_access": "prohibited",
        },
        "statistics_required_next": True,
        "paper_result_integration_authorized": False,
    }
    atomic_write_json(POINT_REPORT, report)
    atomic_write_json(
        POINT_MARKER,
        {"status": report["status"], "point_report_sha256": sha256_file(POINT_REPORT)},
    )
    return report


def _prediction_lookup(
    primary: Mapping[str, Any], matched: Mapping[str, Any], plainmix: Mapping[str, Any]
) -> dict[tuple[str, str, str], Path]:
    rows = [*_artifact_rows(primary), *_artifact_rows(matched), *_artifact_rows(plainmix)]
    return {
        (str(row["model"]), str(row["view"]), str(row["method"])): _rooted(
            row["prediction"]
        )
        for row in rows
        if str(row["model"]) in MODELS
    }


def _checkpoint_deltas(path: Path) -> list[float]:
    document = _load_mapping(path)
    values = document.get("deltas")
    if (
        not isinstance(values, list)
        or len(values) != RESAMPLES
        or document.get("completed_resamples") != RESAMPLES
    ):
        raise PlainMixUAVOBBAnalysisError(f"bootstrap checkpoint is incomplete: {path}")
    deltas = [float(value) for value in values]
    if not all(math.isfinite(value) for value in deltas):
        raise PlainMixUAVOBBAnalysisError(f"bootstrap checkpoint is non-finite: {path}")
    return deltas


def statistics() -> dict[str, Any]:
    point = points()
    _, primary, matched, plainmix = _validate_evidence()
    if STATISTICS.exists():
        report = _load_mapping(STATISTICS)
        if not STATISTICS_MARKER.is_file() or _load_mapping(STATISTICS_MARKER).get(
            "statistics_sha256"
        ) != sha256_file(STATISTICS):
            raise PlainMixUAVOBBAnalysisError("existing statistics report is not locked")
        return report
    lookup = _prediction_lookup(primary, matched, plainmix)
    clusters = primary_engine._clusters(ANNOTATION)
    rows: list[dict[str, Any]] = []
    for baseline, method, name in COMPARISONS:
        for view in PRIMARY_VIEWS:
            comparison = f"{name}_Identity_{view}"
            checkpoint = OUTPUT / "plainmix_bootstrap" / f"{comparison}.json"
            result = paired_coco_ap_cluster_bootstrap_scopes(
                ANNOTATION,
                lookup[(baseline, view, "Identity")],
                lookup[(method, view, "Identity")],
                {
                    "primary": ClusterBootstrapScope(
                        clusters=clusters,
                        checkpoint_path=checkpoint,
                        checkpoint_identity={
                            "study": "cvbra_v1_plainmix_uav_obb_validation",
                            "comparison": comparison,
                            "point_report_sha256": sha256_file(POINT_REPORT),
                            "post_freeze_descriptive_control": True,
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
            expected = float(point["registered_AP_deltas"][name][view])
            if abs(float(result["delta"]) - expected) > 1e-10:
                raise PlainMixUAVOBBAnalysisError(f"bootstrap point estimate drifted: {comparison}")
            deltas = _checkpoint_deltas(checkpoint)
            standard_deviation = stdev(deltas)
            if standard_deviation <= 0.0 or not math.isfinite(standard_deviation):
                raise PlainMixUAVOBBAnalysisError(f"bootstrap variance is invalid: {comparison}")
            rows.append(
                {
                    "comparison": comparison,
                    "registered_contrast": name,
                    "baseline": baseline,
                    "method": method,
                    "view": view,
                    **result,
                    "bootstrap_mean_delta": fmean(deltas),
                    "bootstrap_standard_deviation": standard_deviation,
                    "standardized_effect": expected / standard_deviation,
                    "p_two_sided": bootstrap_sign_pvalue(deltas),
                    "checkpoint": _relative(checkpoint),
                    "checkpoint_sha256": sha256_file(checkpoint),
                }
            )
            print(json.dumps({"bootstrap_complete": comparison}), flush=True)
    adjusted = holm_adjust([float(row["p_two_sided"]) for row in rows])
    for row, value in zip(rows, adjusted, strict=True):
        row["holm_adjusted_p"] = value
        row["holm_significant_0p05"] = value < 0.05
    report = {
        "schema_version": 1,
        "status": "PLAINMIX_UAV_OBB_PAIRED_STATISTICS_COMPLETE",
        "completed_at_utc": primary_engine._utc_now(),
        "protocol_sha256": PROTOCOL_SHA256,
        "point_report_sha256": sha256_file(POINT_REPORT),
        "unit": "canonical source-key group",
        "images": PRIMARY_IMAGES,
        "groups": PRIMARY_GROUPS,
        "resamples": RESAMPLES,
        "seed": SEED,
        "interval": "paired percentile 95 percent",
        "multiple_testing": "Holm across four registered Identity AP contrasts",
        "rows": rows,
        "method_or_hyperparameter_selection": False,
        "independent_confirmation_claim": False,
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
        if not COMPLETE.is_file() or _load_mapping(COMPLETE).get("report_sha256") != sha256_file(
            REPORT
        ):
            raise PlainMixUAVOBBAnalysisError("existing PlainMix report is not locked")
        return report
    stats_rows = stats.get("rows")
    if not isinstance(stats_rows, list) or len(stats_rows) != 4:
        raise PlainMixUAVOBBAnalysisError("PlainMix statistics coverage is incomplete")
    report = {
        "schema_version": 1,
        "status": "COMPLETE_PLAINMIX_UAV_OBB_VALIDATION",
        "completed_at_utc": primary_engine._utc_now(),
        "protocol_sha256": PROTOCOL_SHA256,
        "runner": _relative(Path(__file__)),
        "runner_sha256": sha256_file(Path(__file__)),
        "point_report": _relative(POINT_REPORT),
        "point_report_sha256": sha256_file(POINT_REPORT),
        "statistics": _relative(STATISTICS),
        "statistics_sha256": sha256_file(STATISTICS),
        "registered_AP_deltas": point["registered_AP_deltas"],
        "PlainMix_minus_source_AP": point["PlainMix_minus_source_AP"],
        "paired_statistics": stats_rows,
        "evidence_boundary": point["evidence_boundary"],
        "decision": {
            "descriptive_strong_control_complete": True,
            "main_method_reselection_authorized": False,
            "official_test_access_authorized": False,
            "paper_result_integration_authorized": True,
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
    parser = argparse.ArgumentParser(description="Analyze registered PlainMix UAV-OBB validation")
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
