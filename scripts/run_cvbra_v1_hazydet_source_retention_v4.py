from __future__ import annotations

import argparse
import json
import math
import platform
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

from scripts import run_cvbra_v1_hazydet_source_retention_v3 as parent

from buse_uav.evaluation.bootstrap import (
    ClusterBootstrapScope,
    paired_coco_ap_cluster_bootstrap_scopes,
)
from buse_uav.evaluation.coco import evaluate_coco
from buse_uav.evaluation.supplementary import bootstrap_sign_pvalue, holm_adjust
from buse_uav.utils.hashing import sha256_file
from buse_uav.utils.io import atomic_write_json, atomic_write_text

ROOT = parent.ROOT
PROTOCOL = ROOT / "configs" / "experiment" / "cvbra_v1_hazydet_source_retention_v4.yaml"
OUTPUT = (
    ROOT
    / "reports"
    / "development"
    / "cvbra_v1_matched_baselines"
    / "hazydet_source_retention_v4"
)
V3_OUTPUT = parent.OUTPUT

V3_PROTOCOL_SHA256 = "cd334b967d850add1d47622eb4e53f2fbd3fbd2324292af30cff44279c633ad0"
V3_RUNNER_SHA256 = "52c8c21f7aa30a6091e84377e73c11c751c8b6379bfbae6e238dfe40e9e677b3"
V3_IMPLEMENTATION_LOCK_SHA256 = (
    "8b975f7e3480c52147b5001c96de8fa534a07390784a8a9ff42d3a4b48041c96"
)
V3_METRICS_SHA256 = "6910e8ff0aa215956f7356728ac45ec93a45d8096cb7e437152658bed3b6d3fe"
V3_INCOMPLETE_BOOTSTRAP_SHA256 = (
    "33cf684c057e8a174de53602927134553148e681421606540e5aaa662711d677"
)

AMENDMENT = V3_OUTPUT / "RENUMBERING_POINT_FAILURE_AMENDMENT_1.json"
INVALID_MARKER = V3_OUTPUT / "INVALID_STATISTICS_ATTEMPT_1"
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


class SourceRetentionV4Error(RuntimeError):
    """Raised when the registered recentered analysis cannot fail closed."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SourceRetentionV4Error(f"cannot parse mapping {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SourceRetentionV4Error(f"expected mapping: {path}")
    return value


def _assert_hash(path: Path, expected: str, *, label: str) -> None:
    if not path.is_file() or sha256_file(path) != expected:
        raise SourceRetentionV4Error(f"locked {label} changed: {path}")


def recenter_bootstrap_deltas(
    raw_deltas: list[float],
    *,
    direct_delta: float,
    raw_observed_delta: float,
) -> tuple[list[float], float]:
    """Shift the bootstrap distribution to the standard direct COCO point estimate."""
    if len(raw_deltas) != parent.RESAMPLES or not all(math.isfinite(v) for v in raw_deltas):
        raise SourceRetentionV4Error("raw bootstrap delta coverage changed")
    if not math.isfinite(direct_delta) or not math.isfinite(raw_observed_delta):
        raise SourceRetentionV4Error("bootstrap recentering point is nonfinite")
    offset = direct_delta - raw_observed_delta
    adjusted = [value + offset for value in raw_deltas]
    if not math.isclose(
        raw_observed_delta + offset,
        direct_delta,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise SourceRetentionV4Error("bootstrap recentering did not preserve the direct point")
    return adjusted, offset


def _validate_v3_failure() -> None:
    _assert_hash(
        ROOT / "configs" / "experiment" / "cvbra_v1_hazydet_source_retention_v3.yaml",
        V3_PROTOCOL_SHA256,
        label="v3 protocol",
    )
    _assert_hash(
        ROOT / "scripts" / "run_cvbra_v1_hazydet_source_retention_v3.py",
        V3_RUNNER_SHA256,
        label="v3 runner",
    )
    _assert_hash(
        V3_OUTPUT / "implementation_lock.json",
        V3_IMPLEMENTATION_LOCK_SHA256,
        label="v3 implementation lock",
    )
    _assert_hash(V3_OUTPUT / "metrics.csv", V3_METRICS_SHA256, label="v3 point metrics")
    _assert_hash(
        V3_OUTPUT / "bootstrap" / "CVBRA_v1_minus_source.json",
        V3_INCOMPLETE_BOOTSTRAP_SHA256,
        label="v3 incomplete bootstrap",
    )
    parent._v2_prediction_lock()


def _write_or_validate_amendment() -> dict[str, Any]:
    if AMENDMENT.exists() or INVALID_MARKER.exists():
        if not AMENDMENT.is_file() or not INVALID_MARKER.is_file():
            raise SourceRetentionV4Error("v3 failure amendment is incomplete")
        amendment = _load_mapping(AMENDMENT)
        marker = _load_mapping(INVALID_MARKER)
        if (
            amendment.get("v3_incomplete_bootstrap_sha256")
            != V3_INCOMPLETE_BOOTSTRAP_SHA256
            or amendment.get("v2_prediction_lock_sha256")
            != parent.V2_PREDICTION_LOCK_SHA256
            or marker.get("amendment_sha256") != sha256_file(AMENDMENT)
        ):
            raise SourceRetentionV4Error("v3 failure amendment changed")
        return amendment
    amendment = {
        "schema_version": 1,
        "status": "INVALIDATED_HAZYDET_SOURCE_RETENTION_V3_STATISTICS_ATTEMPT_1",
        "recorded_at_utc": _now(),
        "v3_protocol_sha256": V3_PROTOCOL_SHA256,
        "v3_runner_sha256": V3_RUNNER_SHA256,
        "v3_implementation_lock_sha256": V3_IMPLEMENTATION_LOCK_SHA256,
        "v3_point_metrics_sha256": V3_METRICS_SHA256,
        "v3_incomplete_bootstrap": _relative(
            V3_OUTPUT / "bootstrap" / "CVBRA_v1_minus_source.json"
        ),
        "v3_incomplete_bootstrap_sha256": V3_INCOMPLETE_BOOTSTRAP_SHA256,
        "v2_prediction_lock_sha256": parent.V2_PREDICTION_LOCK_SHA256,
        "diagnostic_non_cached_AP": {
            "source": 0.5177009146075794,
            "CVBRA_v1": 0.4824848292829973,
            "delta": -0.0352160853245821,
        },
        "standard_direct_AP": {
            "source": 0.5176728478165689,
            "CVBRA_v1": 0.4824874544912225,
            "delta": -0.0351853933253464,
        },
        "root_cause": (
            "bootstrap resamples require duplicate image renumbering; tied FP16 scores make "
            "the renumbered point differ slightly from standard COCO AP on original IDs"
        ),
        "scientific_disposition": (
            "v3 point metrics remain valid; incomplete v3 statistics are preserved and excluded"
        ),
        "authorized_v4": {
            "protocol": _relative(PROTOCOL),
            "prediction_change": False,
            "recenter_rule": (
                "adjusted_delta_i = raw_delta_i + direct_delta - raw_observed_delta"
            ),
            "rule_applies_to_all_comparisons_without_outcome_branching": True,
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
    _validate_v3_failure()
    _write_or_validate_amendment()
    protocol_sha256 = sha256_file(PROTOCOL)
    if REGISTRATION.exists() or REGISTRATION_MARKER.exists():
        if not REGISTRATION.is_file() or not REGISTRATION_MARKER.is_file():
            raise SourceRetentionV4Error("v4 registration is incomplete")
        lock = _load_mapping(REGISTRATION)
        marker = _load_mapping(REGISTRATION_MARKER)
        if (
            lock.get("protocol_sha256") != protocol_sha256
            or lock.get("v4_bootstrap_before_registration") is not False
            or marker.get("registration_sha256") != sha256_file(REGISTRATION)
        ):
            raise SourceRetentionV4Error("v4 registration changed")
        return lock
    if any(path.exists() for path in (ANALYSIS_LOCK, METRICS, STATISTICS, REPORT)):
        raise SourceRetentionV4Error("v4 output appeared before registration")
    lock = {
        "schema_version": 1,
        "status": "CVBRA_V1_HAZYDET_SOURCE_RETENTION_V4_REGISTERED",
        "registered_at_utc": _now(),
        "protocol": _relative(PROTOCOL),
        "protocol_sha256": protocol_sha256,
        "amendment_sha256": sha256_file(AMENDMENT),
        "v2_prediction_lock_sha256": parent.V2_PREDICTION_LOCK_SHA256,
        "direct_point_metrics_already_observed": True,
        "v3_raw_bootstrap_already_observed": True,
        "v4_bootstrap_before_registration": False,
        "recenter_rule": "raw_delta_i_plus_direct_delta_minus_raw_observed_delta",
        "rule_applies_to_all_comparisons": True,
        "method_or_hyperparameter_selection": False,
        "models": list(parent.MODELS),
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
            raise SourceRetentionV4Error("v4 implementation lock is incomplete")
        lock = _load_mapping(IMPLEMENTATION_LOCK)
        marker = _load_mapping(IMPLEMENTATION_MARKER)
        if (
            lock.get("runner_sha256") != sha256_file(Path(__file__))
            or lock.get("protocol_sha256") != sha256_file(PROTOCOL)
            or lock.get("registration_sha256") != sha256_file(REGISTRATION)
            or marker.get("implementation_lock_sha256") != sha256_file(IMPLEMENTATION_LOCK)
        ):
            raise SourceRetentionV4Error("v4 implementation lock changed")
        return lock
    if any(path.exists() for path in (ANALYSIS_LOCK, METRICS, STATISTICS, REPORT)):
        raise SourceRetentionV4Error("v4 output appeared before implementation lock")
    image_ids = parent.ordered_image_ids()
    lock = {
        "schema_version": 1,
        "status": "CVBRA_V1_HAZYDET_SOURCE_RETENTION_V4_IMPLEMENTATION_LOCKED",
        "locked_at_utc": _now(),
        "runner": _relative(Path(__file__)),
        "runner_sha256": sha256_file(Path(__file__)),
        "protocol_sha256": sha256_file(PROTOCOL),
        "registration_sha256": sha256_file(REGISTRATION),
        "amendment_sha256": sha256_file(AMENDMENT),
        "v2_prediction_lock_sha256": parent.V2_PREDICTION_LOCK_SHA256,
        "annotation_sha256": parent.ANNOTATION_SHA256,
        "images": len(image_ids),
        "image_order": "lexicographic_string_sort",
        "bootstrap_encoding": "singleton_cluster_per_image",
        "recenter_rule": "raw_delta_i_plus_direct_delta_minus_raw_observed_delta",
        "raw_and_adjusted_statistics_retained": True,
        "resamples": parent.RESAMPLES,
        "seed": parent.SEED,
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
        "status": "PASS_CVBRA_V1_HAZYDET_SOURCE_RETENTION_V4_PREFLIGHT",
        "models": list(parent.MODELS),
        "images": parent.IMAGES,
        "resamples": parent.RESAMPLES,
        "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
        "runner_sha256": lock["runner_sha256"],
    }


def _analysis_authorization() -> dict[str, Any]:
    _implementation_lock()
    if ANALYSIS_LOCK.exists() or ANALYSIS_MARKER.exists():
        if not ANALYSIS_LOCK.is_file() or not ANALYSIS_MARKER.is_file():
            raise SourceRetentionV4Error("v4 analysis authorization is incomplete")
        lock = _load_mapping(ANALYSIS_LOCK)
        marker = _load_mapping(ANALYSIS_MARKER)
        if (
            lock.get("runner_sha256") != sha256_file(Path(__file__))
            or lock.get("v2_prediction_lock_sha256") != parent.V2_PREDICTION_LOCK_SHA256
            or lock.get("annotation_sha256") != parent.ANNOTATION_SHA256
            or marker.get("analysis_lock_sha256") != sha256_file(ANALYSIS_LOCK)
        ):
            raise SourceRetentionV4Error("v4 analysis authorization changed")
        return lock
    if any(path.exists() for path in (METRICS, STATISTICS, REPORT)):
        raise SourceRetentionV4Error("v4 metric output appeared before analysis authorization")
    lock = {
        "schema_version": 1,
        "status": "HAZYDET_SOURCE_RETENTION_V4_ANALYSIS_AUTHORIZED",
        "authorized_at_utc": _now(),
        "runner_sha256": sha256_file(Path(__file__)),
        "protocol_sha256": sha256_file(PROTOCOL),
        "v2_prediction_lock_sha256": parent.V2_PREDICTION_LOCK_SHA256,
        "annotation_sha256": parent.ANNOTATION_SHA256,
        "recenter_rule": "raw_delta_i_plus_direct_delta_minus_raw_observed_delta",
        "v4_metric_or_bootstrap_accessed_before_authorization": False,
        "validation_annotation_previously_accessed": True,
        "method_or_hyperparameter_selection": False,
    }
    atomic_write_json(ANALYSIS_LOCK, lock)
    atomic_write_json(
        ANALYSIS_MARKER,
        {"status": lock["status"], "analysis_lock_sha256": sha256_file(ANALYSIS_LOCK)},
    )
    return lock


def _checkpoint_deltas(path: Path) -> list[float]:
    values = _load_mapping(path).get("deltas")
    if not isinstance(values, list) or len(values) != parent.RESAMPLES:
        raise SourceRetentionV4Error(f"v4 bootstrap checkpoint is incomplete: {path}")
    return [float(value) for value in values]


def _prediction_paths(lock: Mapping[str, Any]) -> dict[str, Path]:
    return parent._prediction_paths(lock)


def score() -> dict[str, Any]:
    v2_lock = parent._v2_prediction_lock()
    _analysis_authorization()
    if REPORT.exists():
        report = _load_mapping(REPORT)
        marker = _load_mapping(COMPLETE)
        if marker.get("report_sha256") != sha256_file(REPORT):
            raise SourceRetentionV4Error("existing v4 report is not locked")
        return report
    paths = _prediction_paths(v2_lock)
    timing = v2_lock.get("timing_median_mean_ms")
    if not isinstance(timing, dict):
        raise SourceRetentionV4Error("v2 timing metadata changed")
    image_ids = parent.ordered_image_ids()
    clusters = parent.singleton_image_clusters(image_ids)
    rows: list[dict[str, Any]] = []
    for model in parent.MODELS:
        metrics = evaluate_coco(
            parent.ANNOTATION,
            paths[model],
            max_det=parent.MAX_DET,
            image_ids=image_ids,
        )
        if int(metrics["images_evaluated"]) != parent.IMAGES:
            raise SourceRetentionV4Error(f"{model} did not evaluate the full split")
        rows.append(
            {
                "model": model,
                **{key: float(metrics[key]) for key in parent.METRIC_KEYS},
                "images_evaluated": int(metrics["images_evaluated"]),
                "mean_ms": float(timing[model]),
                "checkpoint_size_bytes": parent.WEIGHTS[model].stat().st_size,
            }
        )
    expected_metrics = parent._metrics_text(rows)
    if METRICS.exists() and METRICS.read_text(encoding="utf-8") != expected_metrics:
        raise SourceRetentionV4Error("resumable v4 metrics changed")
    if not METRICS.exists():
        atomic_write_text(METRICS, expected_metrics)
    by_model = {str(row["model"]): row for row in rows}
    source_ap_drift = float(by_model["source"]["AP"]) - parent.HISTORICAL_SOURCE_AP
    if abs(source_ap_drift) > parent.SOURCE_AP_MAX_ABS_DRIFT:
        raise SourceRetentionV4Error(f"source AP anchor failed: {source_ap_drift:+.9f}")
    direct_deltas = {
        "CVBRA_v1_minus_source": float(by_model["CVBRA_v1"]["AP"])
        - float(by_model["source"]["AP"]),
        **{
            f"CVBRA_v1_minus_{model}": float(by_model["CVBRA_v1"]["AP"])
            - float(by_model[model]["AP"])
            for model in parent.MODELS[2:]
        },
    }
    comparisons = (
        ("source", "CVBRA_v1"),
        *((model, "CVBRA_v1") for model in parent.MODELS[2:]),
    )
    statistics_rows: list[dict[str, Any]] = []
    for baseline, method in comparisons:
        name = f"{method}_minus_{baseline}"
        checkpoint = OUTPUT / "bootstrap" / f"{name}.json"
        raw_result = paired_coco_ap_cluster_bootstrap_scopes(
            parent.ANNOTATION,
            paths[baseline],
            paths[method],
            {
                "all_images": ClusterBootstrapScope(
                    clusters=clusters,
                    checkpoint_path=checkpoint,
                    checkpoint_identity={
                        "study": "cvbra_v1_hazydet_source_retention_v4",
                        "comparison": name,
                        "v2_prediction_lock_sha256": parent.V2_PREDICTION_LOCK_SHA256,
                        "recenter_rule": (
                            "raw_delta_i_plus_direct_delta_minus_raw_observed_delta"
                        ),
                    },
                )
            },
            resamples=parent.RESAMPLES,
            seed=parent.SEED,
            max_det=parent.MAX_DET,
            workers=4,
            chunk_resamples=100,
            accelerate_ap_only=True,
        )["all_images"]
        direct_delta = direct_deltas[name]
        raw_observed_delta = float(raw_result["delta"])
        raw_deltas = _checkpoint_deltas(checkpoint)
        adjusted_deltas, offset = recenter_bootstrap_deltas(
            raw_deltas,
            direct_delta=direct_delta,
            raw_observed_delta=raw_observed_delta,
        )
        adjusted_point = raw_observed_delta + offset
        if not math.isclose(adjusted_point, direct_delta, rel_tol=0.0, abs_tol=1e-15):
            raise SourceRetentionV4Error(f"v4 recentered point parity failed: {name}")
        standard_deviation = stdev(adjusted_deltas)
        if standard_deviation <= 0.0 or not math.isfinite(standard_deviation):
            raise SourceRetentionV4Error(f"v4 bootstrap variance is invalid: {name}")
        raw_ci_low = float(raw_result["ci_low"])
        raw_ci_high = float(raw_result["ci_high"])
        statistics_rows.append(
            {
                "comparison": name,
                "direct_delta": direct_delta,
                "raw_observed_delta": raw_observed_delta,
                "recenter_offset": offset,
                "adjusted_delta": adjusted_point,
                "raw_ci_low": raw_ci_low,
                "raw_ci_high": raw_ci_high,
                "ci_low": raw_ci_low + offset,
                "ci_high": raw_ci_high + offset,
                "resamples": int(raw_result["resamples"]),
                "seed": int(raw_result["seed"]),
                "images": int(raw_result["images"]),
                "clusters": int(raw_result["clusters"]),
                "raw_bootstrap_mean_delta": fmean(raw_deltas),
                "bootstrap_mean_delta": fmean(adjusted_deltas),
                "bootstrap_standard_deviation": standard_deviation,
                "standardized_effect": direct_delta / standard_deviation,
                "raw_p_two_sided": bootstrap_sign_pvalue(raw_deltas),
                "p_two_sided": bootstrap_sign_pvalue(adjusted_deltas),
                "checkpoint": _relative(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
            }
        )
        print(json.dumps({"bootstrap_complete": name}), flush=True)
    adjusted_p = holm_adjust([float(row["p_two_sided"]) for row in statistics_rows])
    for row, value in zip(statistics_rows, adjusted_p, strict=True):
        row["holm_adjusted_p"] = value
        row["holm_significant_0p05"] = value < 0.05
    statistics = {
        "schema_version": 1,
        "status": "HAZYDET_SOURCE_RETENTION_V4_RECENTERED_STATISTICS_COMPLETE",
        "completed_at_utc": _now(),
        "resamples": parent.RESAMPLES,
        "seed": parent.SEED,
        "unit": "image_via_singleton_cluster_encoding",
        "images": parent.IMAGES,
        "recenter_rule": "raw_delta_i_plus_direct_delta_minus_raw_observed_delta",
        "raw_and_adjusted_statistics_retained": True,
        "multiple_testing": "Holm across five adjusted descriptive AP contrasts",
        "rows": statistics_rows,
    }
    atomic_write_json(STATISTICS, statistics)
    report = {
        "schema_version": 1,
        "status": "COMPLETE_CVBRA_V1_HAZYDET_SOURCE_RETENTION_V4_AUDIT",
        "completed_at_utc": _now(),
        "protocol_sha256": sha256_file(PROTOCOL),
        "runner": _relative(Path(__file__)),
        "runner_sha256": sha256_file(Path(__file__)),
        "amendment_sha256": sha256_file(AMENDMENT),
        "v2_prediction_lock_sha256": parent.V2_PREDICTION_LOCK_SHA256,
        "analysis_lock_sha256": sha256_file(ANALYSIS_LOCK),
        "metrics": _relative(METRICS),
        "metrics_sha256": sha256_file(METRICS),
        "statistics": _relative(STATISTICS),
        "statistics_sha256": sha256_file(STATISTICS),
        "rows": rows,
        "AP_deltas": direct_deltas,
        "paired_statistics": statistics_rows,
        "source_anchor": {
            "historical_AP": parent.HISTORICAL_SOURCE_AP,
            "observed_AP": float(by_model["source"]["AP"]),
            "drift": source_ap_drift,
            "maximum_absolute_drift": parent.SOURCE_AP_MAX_ABS_DRIFT,
            "pass": True,
        },
        "evidence_boundary": {
            "scope": "spent full 1000-image HazyDet validation; descriptive source retention",
            "mapping_correction_only": True,
            "direct_point_recentered_bootstrap": True,
            "raw_and_adjusted_statistics_retained": True,
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
        description="Run direct-point-recentered HazyDet source-retention statistics"
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
