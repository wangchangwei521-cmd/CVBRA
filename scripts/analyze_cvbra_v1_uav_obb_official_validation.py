from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
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
PROTOCOL = ROOT / "configs" / "experiment" / "cvbra_v1_official_validation_protocol.yaml"
PROTOCOL_SHA256 = "820984d022cdb5583484ce1edaab718f428e0001ead5f320653cf1199dcbe4a7"
VALIDATION_ROOT = (
    ROOT / "reports" / "development" / "cvbra_v1" / "uav_obb_official_validation"
)
MATERIALIZATION_LOCK = VALIDATION_ROOT / "view_materialization_lock.json"
EVALUATION_ROOT = VALIDATION_ROOT / "evaluation"
PREDICTION_LOCK = EVALUATION_ROOT / "joint_prediction_lock.json"
PREDICTION_MARKER = EVALUATION_ROOT / "PREDICTIONS_LOCKED"
RAW_LOCK = EVALUATION_ROOT / "raw_observation_lock.json"
CONVERSION_LOCK = VALIDATION_ROOT / "annotations" / "conversion_lock.json"
CONVERSION_MARKER = VALIDATION_ROOT / "annotations" / "LABELS_CONVERTED_AND_LOCKED"
METRICS = EVALUATION_ROOT / "metrics_full_and_decontaminated.csv"
POINT_REPORT = EVALUATION_ROOT / "point_report.json"
POINT_MARKER = EVALUATION_ROOT / "POINTS_COMPLETE"
STATISTICS = EVALUATION_ROOT / "confirmatory_statistics.json"
STATISTICS_MARKER = EVALUATION_ROOT / "STATISTICS_COMPLETE"
REPORT = EVALUATION_ROOT / "validation_report.json"
COMPLETE = EVALUATION_ROOT / "VALIDATION_COMPLETE"
SUCCESS = EVALUATION_ROOT / "SUCCESS"
FAILURE = EVALUATION_ROOT / "FAILED_CONFIRMATION"

MODELS = ("source", "CVBRA_v1")
VIEWS = ("original", "fog_0p6", "fog_1p0")
PRIMARY_VIEW = "fog_1p0"
METHODS = ("Identity", "Flip_HardNMS")
SCOPES = ("decontaminated_primary", "official_full_secondary")
METRIC_KEYS = ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large")
IMAGES = 218
PRIMARY_IMAGES = 167
PRIMARY_GROUPS = 141
MAX_DET = 500
RESAMPLES = 10000
SEED = 20260814
WORKERS = 4


class CVBRAValidationAnalysisError(RuntimeError):
    """Raised when frozen official-validation evidence cannot be analyzed."""


def _utc_now() -> str:
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
        raise CVBRAValidationAnalysisError(f"cannot parse mapping {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CVBRAValidationAnalysisError(f"expected mapping: {path}")
    return value


def _assert_hash(path: Path, expected: object, *, label: str) -> None:
    if not path.is_file() or sha256_file(path) != str(expected):
        raise CVBRAValidationAnalysisError(f"locked {label} changed: {path}")


def _validate_evidence() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    _assert_hash(PROTOCOL, PROTOCOL_SHA256, label="official-validation protocol")
    materialization = _load_mapping(MATERIALIZATION_LOCK)
    prediction = _load_mapping(PREDICTION_LOCK)
    prediction_marker = _load_mapping(PREDICTION_MARKER)
    raw = _load_mapping(RAW_LOCK)
    conversion = _load_mapping(CONVERSION_LOCK)
    conversion_marker = _load_mapping(CONVERSION_MARKER)
    artifacts = prediction.get("artifacts")
    if (
        materialization.get("status")
        != "CVBRA_UAV_OBB_OFFICIAL_VALIDATION_VIEWS_MATERIALIZED_BEFORE_LABEL_ACCESS"
        or materialization.get("images") != IMAGES
        or materialization.get("primary_images") != PRIMARY_IMAGES
        or materialization.get("primary_source_groups") != PRIMARY_GROUPS
        or materialization.get("official_test_content_accessed") is not False
    ):
        raise CVBRAValidationAnalysisError("validation materialization changed")
    if (
        prediction.get("status")
        != "CVBRA_V1_UAV_OBB_OFFICIAL_VALIDATION_PREDICTIONS_LOCKED_BEFORE_LABELS_OR_METRICS"
        or prediction_marker.get("joint_prediction_lock_sha256")
        != sha256_file(PREDICTION_LOCK)
        or prediction.get("raw_observation_lock_sha256") != sha256_file(RAW_LOCK)
        or prediction.get("aggregate_metrics_accessed") is not False
        or prediction.get("official_test_content_accessed") is not False
        or not isinstance(artifacts, list)
        or len(artifacts) != 12
    ):
        raise CVBRAValidationAnalysisError("validation prediction evidence changed")
    if (
        raw.get("status")
        != "ALL_CVBRA_V1_UAV_OBB_OFFICIAL_VALIDATION_RAW_OBSERVATIONS_LOCKED"
        or raw.get("prediction_determinism")
        != "exact_sha256_match_across_two_repetitions"
        or raw.get("aggregate_metrics_accessed") is not False
    ):
        raise CVBRAValidationAnalysisError("validation raw evidence changed")
    if (
        conversion.get("status")
        != "CVBRA_UAV_OBB_OFFICIAL_VALIDATION_LABELS_CONVERTED_AND_LOCKED"
        or conversion_marker.get("conversion_lock_sha256") != sha256_file(CONVERSION_LOCK)
        or conversion.get("joint_prediction_lock_sha256") != sha256_file(PREDICTION_LOCK)
        or conversion.get("aggregate_metrics_accessed") is not False
        or conversion.get("official_test_content_accessed") is not False
        or conversion.get("images") != IMAGES
        or conversion.get("primary_images") != PRIMARY_IMAGES
    ):
        raise CVBRAValidationAnalysisError("validation conversion evidence changed")
    annotation = _rooted(conversion["annotation"])
    _assert_hash(annotation, conversion["annotation_sha256"], label="validation annotation")
    for row in artifacts:
        if not isinstance(row, dict):
            raise CVBRAValidationAnalysisError("invalid prediction artifact")
        _assert_hash(_rooted(row["prediction"]), row["prediction_sha256"], label="prediction")
    return materialization, prediction, conversion, annotation


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


def _delta(
    by_key: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
    *,
    scope: str,
    view: str,
    method: str,
    metric: str,
) -> float:
    return float(by_key[(scope, "CVBRA_v1", view, method)][metric]) - float(
        by_key[(scope, "source", view, method)][metric]
    )


def _maximum_latency_ratio(
    timing: Mapping[str, Any], *, method: str
) -> tuple[float, dict[str, float]]:
    key = f"{method}_mean_ms"
    ratios: dict[str, float] = {}
    for view in VIEWS:
        source = timing["source"]
        candidate = timing["CVBRA_v1"]
        if not isinstance(source, dict) or not isinstance(candidate, dict):
            raise CVBRAValidationAnalysisError("timing model mapping is invalid")
        source_view = source[view]
        candidate_view = candidate[view]
        if not isinstance(source_view, dict) or not isinstance(candidate_view, dict):
            raise CVBRAValidationAnalysisError("timing view mapping is invalid")
        ratios[view] = float(candidate_view[key]) / float(source_view[key])
    return max(ratios.values()), ratios


def points() -> dict[str, Any]:
    materialization, prediction, conversion, annotation = _validate_evidence()
    if POINT_REPORT.exists():
        report = _load_mapping(POINT_REPORT)
        if not POINT_MARKER.is_file() or _load_mapping(POINT_MARKER).get(
            "point_report_sha256"
        ) != sha256_file(POINT_REPORT):
            raise CVBRAValidationAnalysisError("existing point report is not locked")
        return report
    if any(path.exists() for path in (METRICS, POINT_MARKER, STATISTICS, REPORT)):
        raise CVBRAValidationAnalysisError("partial point analysis requires audit")
    artifacts = prediction["artifacts"]
    timing = prediction["timing"]
    if not isinstance(artifacts, list) or not isinstance(timing, dict):
        raise CVBRAValidationAnalysisError("prediction payload is invalid")
    primary_ids_raw = materialization.get("primary_image_ids")
    if not isinstance(primary_ids_raw, list):
        raise CVBRAValidationAnalysisError("primary image registry is invalid")
    primary_ids = [int(value) for value in primary_ids_raw]
    full_ids = list(range(1, IMAGES + 1))
    scope_ids = {
        "decontaminated_primary": primary_ids,
        "official_full_secondary": full_ids,
    }
    rows: list[dict[str, Any]] = []
    for scope, image_ids in scope_ids.items():
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                raise CVBRAValidationAnalysisError("invalid prediction artifact")
            metrics = evaluate_coco(
                annotation,
                _rooted(artifact["prediction"]),
                max_det=MAX_DET,
                image_ids=image_ids,
            )
            model = str(artifact["model"])
            view = str(artifact["view"])
            method = str(artifact["method"])
            model_timing = timing[model]
            if not isinstance(model_timing, dict) or not isinstance(model_timing[view], dict):
                raise CVBRAValidationAnalysisError("timing payload is invalid")
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
    if len(rows) != 24:
        raise CVBRAValidationAnalysisError("validation metric coverage changed")
    _write_metrics(rows)
    by_key = {
        (
            str(row["scope"]),
            str(row["model"]),
            str(row["view"]),
            str(row["method"]),
        ): row
        for row in rows
    }
    primary = "decontaminated_primary"
    deltas = {
        "Identity_original_AP": _delta(
            by_key, scope=primary, view="original", method="Identity", metric="AP"
        ),
        "Identity_primary_fog_AP": _delta(
            by_key, scope=primary, view=PRIMARY_VIEW, method="Identity", metric="AP"
        ),
        "Identity_primary_fog_AP75": _delta(
            by_key, scope=primary, view=PRIMARY_VIEW, method="Identity", metric="AP75"
        ),
        "Flip_original_AP": _delta(
            by_key, scope=primary, view="original", method="Flip_HardNMS", metric="AP"
        ),
        "Flip_primary_fog_AP": _delta(
            by_key, scope=primary, view=PRIMARY_VIEW, method="Flip_HardNMS", metric="AP"
        ),
    }
    identity_latency_ratio, identity_latency_by_view = _maximum_latency_ratio(
        timing, method="Identity"
    )
    flip_latency_ratio, flip_latency_by_view = _maximum_latency_ratio(
        timing, method="Flip_HardNMS"
    )
    checks = {
        "CVBRA_Identity_original_AP_delta_positive": deltas["Identity_original_AP"] > 0.0,
        "CVBRA_Identity_primary_fog_AP_delta_positive": (
            deltas["Identity_primary_fog_AP"] > 0.0
        ),
        "CVBRA_Flip_original_AP_delta_positive": deltas["Flip_original_AP"] > 0.0,
        "CVBRA_Flip_primary_fog_AP_delta_positive": deltas["Flip_primary_fog_AP"] > 0.0,
        "CVBRA_Identity_primary_fog_AP75_delta_nonnegative": (
            deltas["Identity_primary_fog_AP75"] >= 0.0
        ),
        "CVBRA_Identity_latency_ratio_at_most_1p05": identity_latency_ratio <= 1.05,
        "CVBRA_Flip_latency_ratio_at_most_1p05": flip_latency_ratio <= 1.05,
        "prediction_reproducibility_exact": True,
    }
    report = {
        "schema_version": 1,
        "status": "CVBRA_V1_OFFICIAL_VALIDATION_POINT_ESTIMATES_LOCKED",
        "completed_at_utc": _utc_now(),
        "protocol_sha256": PROTOCOL_SHA256,
        "joint_prediction_lock_sha256": sha256_file(PREDICTION_LOCK),
        "conversion_lock_sha256": sha256_file(CONVERSION_LOCK),
        "annotation_sha256": conversion["annotation_sha256"],
        "metrics": _relative(METRICS),
        "metrics_sha256": sha256_file(METRICS),
        "checks": checks,
        "all_point_checks_pass": all(checks.values()),
        "deltas": deltas,
        "latency": {
            "Identity_maximum_CVBRA_to_source_ratio": identity_latency_ratio,
            "Identity_ratio_by_view": identity_latency_by_view,
            "Flip_maximum_CVBRA_to_source_ratio": flip_latency_ratio,
            "Flip_ratio_by_view": flip_latency_by_view,
        },
        "rows": rows,
        "evidence_boundary": {
            "primary_scope": "167 validation images in 141 train-disjoint source-key groups",
            "secondary_scope": "all 218 official validation images",
            "full_scope_exact_train_image_duplicates": materialization[
                "exact_train_image_sha256_overlap_full"
            ],
            "primary_scope_exact_train_image_duplicates": 0,
            "official_test_accessed": False,
        },
        "confirmatory_statistics_required_next": True,
        "paper_body_change_authorized": False,
    }
    atomic_write_json(POINT_REPORT, report)
    atomic_write_json(
        POINT_MARKER,
        {"status": report["status"], "point_report_sha256": sha256_file(POINT_REPORT)},
    )
    return report


def _clusters(annotation: Path) -> dict[str, list[int]]:
    document = _load_mapping(annotation)
    images = document.get("images")
    if not isinstance(images, list):
        raise CVBRAValidationAnalysisError("annotation image registry is invalid")
    clusters: dict[str, list[int]] = defaultdict(list)
    for row in sorted(
        (row for row in images if isinstance(row, dict)), key=lambda item: int(item["id"])
    ):
        if bool(row.get("primary_eligible")):
            clusters[str(row["source_key"])].append(int(row["id"]))
    flattened = [image_id for values in clusters.values() for image_id in values]
    expected = sorted(flattened)
    if (
        len(flattened) != PRIMARY_IMAGES
        or len(clusters) != PRIMARY_GROUPS
        or flattened != expected
    ):
        raise CVBRAValidationAnalysisError("primary source-key clusters changed")
    return dict(clusters)


def _checkpoint_deltas(path: Path) -> list[float]:
    document = _load_mapping(path)
    raw = document.get("deltas")
    if (
        not isinstance(raw, list)
        or len(raw) != RESAMPLES
        or document.get("completed_resamples") != RESAMPLES
    ):
        raise CVBRAValidationAnalysisError("bootstrap checkpoint is incomplete")
    values = [float(value) for value in raw]
    if not all(math.isfinite(value) for value in values):
        raise CVBRAValidationAnalysisError("bootstrap checkpoint is non-finite")
    return values


def statistics() -> dict[str, Any]:
    point = points()
    _, prediction, _, annotation = _validate_evidence()
    if STATISTICS.exists():
        report = _load_mapping(STATISTICS)
        if not STATISTICS_MARKER.is_file() or _load_mapping(STATISTICS_MARKER).get(
            "statistics_sha256"
        ) != sha256_file(STATISTICS):
            raise CVBRAValidationAnalysisError("existing statistics report is not locked")
        return report
    artifacts = prediction.get("artifacts")
    if not isinstance(artifacts, list):
        raise CVBRAValidationAnalysisError("prediction artifacts are invalid")
    lookup = {
        (str(row["model"]), str(row["view"]), str(row["method"])): _rooted(
            row["prediction"]
        )
        for row in artifacts
        if isinstance(row, dict)
    }
    clusters = _clusters(annotation)
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
            lookup[("CVBRA_v1", view, method)],
            {
                "primary": ClusterBootstrapScope(
                    clusters=clusters,
                    checkpoint_path=checkpoint,
                    checkpoint_identity={
                        "study": "cvbra_v1_uav_obb_official_validation",
                        "comparison": name,
                        "point_report_sha256": sha256_file(POINT_REPORT),
                        "decontamination": "train_disjoint_canonical_source_key",
                    },
                )
            },
            resamples=RESAMPLES,
            seed=SEED,
            max_det=MAX_DET,
            workers=WORKERS,
            chunk_resamples=100,
            accelerate_ap_only=True,
        )["primary"]
        expected_delta = float(point["deltas"][f"{name}_AP"])
        if abs(float(result["delta"]) - expected_delta) > 1e-10:
            raise CVBRAValidationAnalysisError(f"bootstrap point estimate drifted: {name}")
        deltas = _checkpoint_deltas(checkpoint)
        standard_deviation = stdev(deltas)
        if standard_deviation <= 0.0:
            raise CVBRAValidationAnalysisError(f"bootstrap variance vanished: {name}")
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
        "status": "CVBRA_V1_OFFICIAL_VALIDATION_CONFIRMATORY_STATISTICS_COMPLETE",
        "completed_at_utc": _utc_now(),
        "protocol_sha256": PROTOCOL_SHA256,
        "point_report_sha256": sha256_file(POINT_REPORT),
        "unit": "canonical source-key group",
        "images": PRIMARY_IMAGES,
        "groups": PRIMARY_GROUPS,
        "resamples": RESAMPLES,
        "seed": SEED,
        "interval": "paired percentile 95 percent",
        "multiple_testing": "Holm across four registered primary AP comparisons",
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
            raise CVBRAValidationAnalysisError("existing validation report is not locked")
        return report
    stats_rows = stats.get("rows")
    if not isinstance(stats_rows, list):
        raise CVBRAValidationAnalysisError("statistics rows are invalid")
    identity_primary = next(
        row
        for row in stats_rows
        if isinstance(row, dict) and row.get("comparison") == "Identity_primary_fog"
    )
    checks = {
        **point["checks"],
        "Identity_primary_fog_95pct_CI_above_zero": float(identity_primary["ci_low"]) > 0.0,
        "four_registered_bootstraps_complete": len(stats_rows) == 4,
    }
    passed = all(bool(value) for value in checks.values())
    report = {
        "schema_version": 1,
        "status": (
            "PASS_CVBRA_V1_OFFICIAL_VALIDATION_CONFIRMATION"
            if passed
            else "FAIL_CVBRA_V1_OFFICIAL_VALIDATION_CONFIRMATION"
        ),
        "completed_at_utc": _utc_now(),
        "protocol_sha256": PROTOCOL_SHA256,
        "point_report": _relative(POINT_REPORT),
        "point_report_sha256": sha256_file(POINT_REPORT),
        "statistics": _relative(STATISTICS),
        "statistics_sha256": sha256_file(STATISTICS),
        "checks": checks,
        "all_checks_pass": passed,
        "primary_deltas": point["deltas"],
        "primary_statistics": stats_rows,
        "latency": point["latency"],
        "evidence_boundary": point["evidence_boundary"],
        "decision": {
            "CVBRA_v1_official_validation_confirmed": passed,
            "second_detector_execution_authorized_next": passed,
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
    parser = argparse.ArgumentParser(description="Analyze frozen CVBRA official validation")
    parser.add_argument("--stage", choices=("points", "statistics", "final"), default="final")
    args = parser.parse_args()
    if args.stage == "points":
        result = points()
    elif args.stage == "statistics":
        result = statistics()
    else:
        result = finalize()
    report_path = {
        "points": POINT_REPORT,
        "statistics": STATISTICS,
        "final": REPORT,
    }[args.stage]
    print(
        json.dumps(
            {
                "status": result["status"],
                "report_sha256": sha256_file(report_path),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
