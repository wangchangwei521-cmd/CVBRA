from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from statistics import fmean
from typing import Any

from buse_uav.evaluation.coco import evaluate_coco
from buse_uav.utils.hashing import sha256_file
from buse_uav.utils.io import atomic_write_json, atomic_write_text

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "experiment" / "cvbra_v1_backup_protocol.yaml"
PROTOCOL_SHA256 = "65c26c8e96284af4f58c3536d4ef81f5531ccc9c0455d2286cbd84d228eaf6da"
RESERVE_ROOT = ROOT / "reports" / "development" / "cvbra_v1" / "uav_obb_reserve_B"
EVALUATION_ROOT = RESERVE_ROOT / "evaluation"
PREDICTION_LOCK = EVALUATION_ROOT / "joint_prediction_lock.json"
PREDICTION_MARKER = EVALUATION_ROOT / "PREDICTIONS_LOCKED"
RAW_LOCK = EVALUATION_ROOT / "raw_observation_lock.json"
CONVERSION_LOCK = RESERVE_ROOT / "annotations" / "conversion_lock.json"
CONVERSION_MARKER = RESERVE_ROOT / "annotations" / "LABELS_CONVERTED_AND_LOCKED"
METRICS = EVALUATION_ROOT / "metrics.csv"
REPORT = EVALUATION_ROOT / "selection_report.json"
COMPLETE = EVALUATION_ROOT / "SELECTION_COMPLETE"
SUCCESS = EVALUATION_ROOT / "SUCCESS"
FAILURE = EVALUATION_ROOT / "FAILED_SELECTION"

MODELS = ("source", "CVBRA_v1")
VIEWS = ("original", "fog_0p6", "fog_1p0")
FOG_VIEWS = ("fog_0p6", "fog_1p0")
PRIMARY_VIEW = "fog_1p0"
METHODS = ("Identity", "Flip_HardNMS")
METRIC_KEYS = ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large")
IMAGES = 483
MAX_DET = 500


class CVBRAReserveAnalysisError(RuntimeError):
    """Raised when frozen CVBRA reserve-B evidence cannot be analyzed."""


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
        raise CVBRAReserveAnalysisError(f"cannot parse mapping {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CVBRAReserveAnalysisError(f"expected mapping: {path}")
    return value


def _assert_hash(path: Path, expected: object, *, label: str) -> None:
    if not path.is_file():
        raise CVBRAReserveAnalysisError(f"missing locked {label}: {path}")
    observed = sha256_file(path)
    if observed != str(expected):
        raise CVBRAReserveAnalysisError(
            f"locked {label} changed: expected {expected}, observed {observed}"
        )


def _validate_evidence() -> tuple[dict[str, Any], dict[str, Any], Path]:
    _assert_hash(PROTOCOL, PROTOCOL_SHA256, label="CVBRA protocol")
    prediction = _load_mapping(PREDICTION_LOCK)
    prediction_marker = _load_mapping(PREDICTION_MARKER)
    raw = _load_mapping(RAW_LOCK)
    conversion = _load_mapping(CONVERSION_LOCK)
    conversion_marker = _load_mapping(CONVERSION_MARKER)
    artifacts = prediction.get("artifacts")
    if (
        prediction.get("status")
        != "CVBRA_V1_UAV_OBB_RESERVE_B_PREDICTIONS_LOCKED_BEFORE_ANNOTATIONS_OR_METRICS"
        or prediction_marker.get("joint_prediction_lock_sha256") != sha256_file(PREDICTION_LOCK)
        or prediction.get("raw_observation_lock_sha256") != sha256_file(RAW_LOCK)
        or prediction.get("aggregate_metrics_accessed") is not False
        or prediction.get("official_validation_or_test_accessed") is not False
        or not isinstance(artifacts, list)
        or len(artifacts) != 12
    ):
        raise CVBRAReserveAnalysisError("reserve-B prediction evidence changed")
    if (
        raw.get("status") != "ALL_CVBRA_V1_UAV_OBB_RESERVE_B_RAW_OBSERVATIONS_LOCKED"
        or raw.get("prediction_determinism") != "exact_sha256_match_across_two_repetitions"
        or raw.get("aggregate_metrics_accessed") is not False
    ):
        raise CVBRAReserveAnalysisError("reserve-B raw evidence changed")
    if (
        conversion.get("status") != "CVBRA_UAV_OBB_RESERVE_B_LABELS_CONVERTED_AND_LOCKED"
        or conversion_marker.get("conversion_lock_sha256") != sha256_file(CONVERSION_LOCK)
        or conversion.get("joint_prediction_lock_sha256") != sha256_file(PREDICTION_LOCK)
        or conversion.get("aggregate_metrics_accessed") is not False
        or conversion.get("official_validation_or_test_content_accessed") is not False
        or conversion.get("images") != IMAGES
    ):
        raise CVBRAReserveAnalysisError("reserve-B conversion evidence changed")
    annotation = _rooted(conversion["annotation"])
    _assert_hash(annotation, conversion["annotation_sha256"], label="reserve-B annotation")
    for row in artifacts:
        if not isinstance(row, dict):
            raise CVBRAReserveAnalysisError("invalid prediction artifact")
        _assert_hash(_rooted(row["prediction"]), row["prediction_sha256"], label="prediction")
    return prediction, conversion, annotation


def _write_metrics(rows: Sequence[Mapping[str, Any]]) -> None:
    fields = (
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
    by_key: Mapping[tuple[str, str, str], Mapping[str, Any]],
    *,
    view: str,
    method: str,
    metric: str,
) -> float:
    return float(by_key[("CVBRA_v1", view, method)][metric]) - float(
        by_key[("source", view, method)][metric]
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
            raise CVBRAReserveAnalysisError("timing model mapping is invalid")
        source_view = source[view]
        candidate_view = candidate[view]
        if not isinstance(source_view, dict) or not isinstance(candidate_view, dict):
            raise CVBRAReserveAnalysisError("timing view mapping is invalid")
        ratios[view] = float(candidate_view[key]) / float(source_view[key])
    return max(ratios.values()), ratios


def analyze() -> dict[str, Any]:
    prediction, conversion, annotation = _validate_evidence()
    if REPORT.exists():
        report = _load_mapping(REPORT)
        if not COMPLETE.is_file() or _load_mapping(COMPLETE).get(
            "selection_report_sha256"
        ) != sha256_file(REPORT):
            raise CVBRAReserveAnalysisError("existing reserve-B report is not locked")
        return report
    if any(path.exists() for path in (METRICS, COMPLETE, SUCCESS, FAILURE)):
        raise CVBRAReserveAnalysisError("partial reserve-B analysis requires audit")
    artifacts = prediction["artifacts"]
    timing = prediction["timing"]
    if not isinstance(artifacts, list) or not isinstance(timing, dict):
        raise CVBRAReserveAnalysisError("reserve-B prediction payload is invalid")
    image_ids = list(range(1, IMAGES + 1))
    rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise CVBRAReserveAnalysisError("invalid reserve-B artifact")
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
            raise CVBRAReserveAnalysisError("reserve-B timing payload is invalid")
        rows.append(
            {
                "model": model,
                "view": view,
                "method": method,
                **{key: float(metrics[key]) for key in METRIC_KEYS},
                "images_evaluated": int(metrics["images_evaluated"]),
                "mean_ms": float(model_timing[view][f"{method}_mean_ms"]),
                "prediction_sha256": artifact["prediction_sha256"],
            }
        )
    if len(rows) != 12 or any(row["images_evaluated"] != IMAGES for row in rows):
        raise CVBRAReserveAnalysisError("reserve-B metric coverage changed")
    _write_metrics(rows)
    by_key = {(str(row["model"]), str(row["view"]), str(row["method"])): row for row in rows}
    identity_original_ap = _delta(by_key, view="original", method="Identity", metric="AP")
    identity_original_ap75 = _delta(by_key, view="original", method="Identity", metric="AP75")
    identity_primary_ap = _delta(by_key, view=PRIMARY_VIEW, method="Identity", metric="AP")
    identity_mean_fog_ap = fmean(
        _delta(by_key, view=view, method="Identity", metric="AP") for view in FOG_VIEWS
    )
    flip_original_ap = _delta(by_key, view="original", method="Flip_HardNMS", metric="AP")
    flip_primary_ap = _delta(by_key, view=PRIMARY_VIEW, method="Flip_HardNMS", metric="AP")
    flip_primary_ap75 = _delta(by_key, view=PRIMARY_VIEW, method="Flip_HardNMS", metric="AP75")
    identity_latency_ratio, identity_latency_by_view = _maximum_latency_ratio(
        timing, method="Identity"
    )
    flip_latency_ratio, flip_latency_by_view = _maximum_latency_ratio(timing, method="Flip_HardNMS")
    checks = {
        "CVBRA_Identity_original_AP_delta_at_least_0p010": identity_original_ap >= 0.010,
        "CVBRA_Identity_original_AP75_delta_nonnegative": identity_original_ap75 >= 0.0,
        "CVBRA_Identity_primary_fog_AP_delta_at_least_0p010": identity_primary_ap >= 0.010,
        "CVBRA_Identity_mean_two_fog_AP_delta_at_least_0p010": identity_mean_fog_ap >= 0.010,
        "CVBRA_Flip_original_AP_delta_at_least_0p005": flip_original_ap >= 0.005,
        "CVBRA_Flip_primary_fog_AP_delta_at_least_0p005": flip_primary_ap >= 0.005,
        "CVBRA_Flip_primary_fog_AP75_delta_nonnegative": flip_primary_ap75 >= 0.0,
        "CVBRA_Identity_latency_ratio_at_most_1p05": identity_latency_ratio <= 1.05,
        "CVBRA_Flip_latency_ratio_at_most_1p05": flip_latency_ratio <= 1.05,
        "prediction_reproducibility_exact": True,
    }
    passed = all(checks.values())
    deltas = {
        "Identity_original_AP": identity_original_ap,
        "Identity_original_AP75": identity_original_ap75,
        "Identity_fog_0p6_AP": _delta(by_key, view="fog_0p6", method="Identity", metric="AP"),
        "Identity_primary_fog_AP": identity_primary_ap,
        "Identity_primary_fog_AP75": _delta(
            by_key, view=PRIMARY_VIEW, method="Identity", metric="AP75"
        ),
        "Identity_mean_two_fog_AP": identity_mean_fog_ap,
        "Flip_original_AP": flip_original_ap,
        "Flip_original_AP75": _delta(by_key, view="original", method="Flip_HardNMS", metric="AP75"),
        "Flip_fog_0p6_AP": _delta(by_key, view="fog_0p6", method="Flip_HardNMS", metric="AP"),
        "Flip_primary_fog_AP": flip_primary_ap,
        "Flip_primary_fog_AP75": flip_primary_ap75,
        "Flip_mean_two_fog_AP": fmean(
            _delta(by_key, view=view, method="Flip_HardNMS", metric="AP") for view in FOG_VIEWS
        ),
    }
    report = {
        "schema_version": 1,
        "status": (
            "PASS_CVBRA_V1_RESERVE_B_SELECTION" if passed else "FAIL_CVBRA_V1_RESERVE_B_SELECTION"
        ),
        "completed_at_utc": _utc_now(),
        "protocol_sha256": PROTOCOL_SHA256,
        "joint_prediction_lock_sha256": sha256_file(PREDICTION_LOCK),
        "conversion_lock_sha256": sha256_file(CONVERSION_LOCK),
        "annotation_sha256": conversion["annotation_sha256"],
        "metrics": _relative(METRICS),
        "metrics_sha256": sha256_file(METRICS),
        "checks": checks,
        "all_checks_pass": passed,
        "deltas": deltas,
        "latency": {
            "Identity_maximum_CVBRA_to_source_ratio": identity_latency_ratio,
            "Identity_ratio_by_view": identity_latency_by_view,
            "Flip_maximum_CVBRA_to_source_ratio": flip_latency_ratio,
            "Flip_ratio_by_view": flip_latency_by_view,
        },
        "rows": rows,
        "decision": {
            "CVBRA_v1_frozen_as_primary_method": passed,
            "official_validation_access_authorized_next": passed,
            "official_test_access_authorized": False,
            "paper_body_change_authorized": False,
        },
        "evidence_boundary": {
            "split": "UAV-OBB official train reserve_B",
            "reserve_B_images": IMAGES,
            "development_A_role": "training_only",
            "official_validation_accessed": False,
            "official_test_accessed": False,
        },
    }
    atomic_write_json(REPORT, report)
    marker = {
        "status": report["status"],
        "selection_report_sha256": sha256_file(REPORT),
        "metrics_sha256": sha256_file(METRICS),
    }
    atomic_write_json(COMPLETE, marker)
    atomic_write_json(SUCCESS if passed else FAILURE, marker)
    return report


def main() -> int:
    argparse.ArgumentParser(description="Analyze frozen CVBRA reserve-B evidence").parse_args()
    result = analyze()
    print(
        json.dumps(
            {
                "status": result["status"],
                "checks": result["checks"],
                "deltas": result["deltas"],
                "latency": result["latency"],
                "selection_report_sha256": sha256_file(REPORT),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
