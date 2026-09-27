from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from scripts import analyze_cvbra_v1_quality_upgrade_v2 as engine

from buse_uav.utils.hashing import sha256_file
from buse_uav.utils.io import atomic_write_json

ROOT = Path(__file__).resolve().parents[1]
RUNNER = Path(__file__).resolve()
AMENDMENT = (
    ROOT
    / "reports/development/cvbra_v1_quality_upgrade_v2/"
    "ANALYSIS_AMENDMENT_V3.json"
)
AMENDMENT_MARKER = AMENDMENT.with_name("ANALYSIS_V3_REGISTERED")
OUTPUT = ROOT / "reports/development/cvbra_v1_quality_upgrade_v2/analysis_v3"
IMPLEMENTATION_LOCK = OUTPUT / "implementation_lock.json"
IMPLEMENTATION_MARKER = OUTPUT / "IMPLEMENTATION_LOCKED"
POINTS = OUTPUT / "point_estimates.json"
POINTS_MARKER = OUTPUT / "POINTS_LOCKED"
FACTORIAL = OUTPUT / "factorial_statistics.json"
FACTORIAL_MARKER = OUTPUT / "FACTORIAL_LOCKED"
RETENTION = OUTPUT / "ewc_retention_statistics.json"
RETENTION_MARKER = OUTPUT / "RETENTION_LOCKED"
THRESHOLDS = OUTPUT / "threshold_robustness.json"
THRESHOLDS_MARKER = OUTPUT / "THRESHOLDS_LOCKED"
INFLUENCE = OUTPUT / "group_influence.json"
INFLUENCE_MARKER = OUTPUT / "INFLUENCE_LOCKED"
REPORT = OUTPUT / "quality_upgrade_report.json"
COMPLETE = OUTPUT / "QUALITY_UPGRADE_COMPLETE"

V1_RUNNER = engine.RUNNER
V1_IMPLEMENTATION_LOCK = engine.IMPLEMENTATION_LOCK
V2_RUNNER = ROOT / "scripts/analyze_cvbra_v1_quality_upgrade_v2_fixed.py"
V2_IMPLEMENTATION_LOCK = (
    ROOT
    / "reports/development/cvbra_v1_quality_upgrade_v2/analysis_v2/"
    "implementation_lock.json"
)
_original_implementation_lock = engine._implementation_lock
_original_evaluate_coco = engine.evaluate_coco
_original_paired_result = engine._paired_result
_original_factorial = engine.factorial
_original_retention = engine.retention

_HAZY_AP_CACHE: dict[str, float] = {}
_RECENTER_ROWS: dict[str, dict[str, Any]] = {}


class QualityUpgradeAnalysisV3Error(RuntimeError):
    """Raised when the registered HazyDet recentering correction cannot fail closed."""


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise QualityUpgradeAnalysisV3Error(f"cannot parse mapping {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise QualityUpgradeAnalysisV3Error(f"expected mapping: {path}")
    return value


def _validate_amendment() -> dict[str, Any]:
    if not AMENDMENT.is_file() or not AMENDMENT_MARKER.is_file():
        raise QualityUpgradeAnalysisV3Error("analysis v3 amendment is not registered")
    amendment = _load_mapping(AMENDMENT)
    marker = _load_mapping(AMENDMENT_MARKER)
    if (
        amendment.get("status")
        != "REGISTERED_HAZYDET_BOOTSTRAP_RECENTERING_BEFORE_ANALYSIS_V3"
        or amendment.get("v1_runner_sha256") != sha256_file(V1_RUNNER)
        or amendment.get("v1_implementation_lock_sha256")
        != sha256_file(V1_IMPLEMENTATION_LOCK)
        or amendment.get("failed_v2_runner_sha256") != sha256_file(V2_RUNNER)
        or amendment.get("failed_v2_implementation_lock_sha256")
        != sha256_file(V2_IMPLEMENTATION_LOCK)
        or amendment.get("v2_factorial_report_existed_at_amendment") is not False
        or amendment.get("v2_partial_bootstrap_reused_by_v3") is not False
        or amendment.get("v3_runner_sha256") != sha256_file(RUNNER)
        or amendment.get("only_change")
        != "HazyDet_raw_bootstrap_delta_plus_direct_minus_raw_observed"
        or marker.get("amendment_sha256") != sha256_file(AMENDMENT)
    ):
        raise QualityUpgradeAnalysisV3Error("analysis v3 amendment changed")
    return amendment


def _implementation_lock() -> dict[str, Any]:
    amendment = _validate_amendment()
    existed = IMPLEMENTATION_LOCK.exists() or IMPLEMENTATION_MARKER.exists()
    lock = cast(dict[str, Any], _original_implementation_lock())
    if not existed:
        lock["amendment"] = engine._relative(AMENDMENT)
        lock["amendment_sha256"] = sha256_file(AMENDMENT)
        lock["failed_v2_implementation_lock_sha256"] = amendment[
            "failed_v2_implementation_lock_sha256"
        ]
        lock["HazyDet_point_image_scope"] = "all_1000_annotation_image_ids"
        lock["HazyDet_bootstrap_recenter_rule"] = (
            "raw_delta_plus_direct_1000_image_delta_minus_raw_observed_delta"
        )
        atomic_write_json(IMPLEMENTATION_LOCK, lock)
        atomic_write_json(
            IMPLEMENTATION_MARKER,
            {
                "status": lock["status"],
                "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
            },
        )
    elif (
        lock.get("amendment_sha256") != sha256_file(AMENDMENT)
        or lock.get("HazyDet_point_image_scope") != "all_1000_annotation_image_ids"
        or lock.get("HazyDet_bootstrap_recenter_rule")
        != "raw_delta_plus_direct_1000_image_delta_minus_raw_observed_delta"
    ):
        raise QualityUpgradeAnalysisV3Error("analysis v3 implementation lock changed")
    return lock


def _hazy_image_ids() -> list[int]:
    return [
        image_id
        for group in engine._hazy_clusters().values()
        for image_id in group
    ]


def _evaluate_coco_with_registered_hazy_scope(
    annotation_path: Path,
    prediction_path: Path,
    **kwargs: Any,
) -> dict[str, Any]:
    if annotation_path.resolve() == engine.HAZY_ANNOTATION.resolve() and kwargs.get(
        "image_ids"
    ) is None:
        kwargs["image_ids"] = _hazy_image_ids()
    return cast(
        dict[str, Any],
        _original_evaluate_coco(annotation_path, prediction_path, **kwargs),
    )


def _hazy_ap(prediction_path: Path) -> float:
    digest = sha256_file(prediction_path)
    if digest not in _HAZY_AP_CACHE:
        result = _evaluate_coco_with_registered_hazy_scope(
            engine.HAZY_ANNOTATION,
            prediction_path,
            max_det=engine.MAX_DET,
        )
        if int(result["images_evaluated"]) != engine.HAZY_IMAGES:
            raise QualityUpgradeAnalysisV3Error("HazyDet direct AP did not use all 1000 images")
        _HAZY_AP_CACHE[digest] = float(result["AP"])
    return _HAZY_AP_CACHE[digest]


def _paired_result(**kwargs: Any) -> tuple[dict[str, float | int], list[float], Path]:
    result, raw_deltas, checkpoint = _original_paired_result(**kwargs)
    annotation = cast(Path, kwargs["annotation"])
    if annotation.resolve() != engine.HAZY_ANNOTATION.resolve():
        return result, raw_deltas, checkpoint
    baseline = cast(Path, kwargs["baseline_prediction"])
    method = cast(Path, kwargs["method_prediction"])
    direct_delta = _hazy_ap(method) - _hazy_ap(baseline)
    raw_observed = float(result["delta"])
    offset = direct_delta - raw_observed
    adjusted = [value + offset for value in raw_deltas]
    corrected = dict(result)
    corrected["raw_observed_delta"] = raw_observed
    corrected["recenter_offset"] = offset
    corrected["delta"] = direct_delta
    corrected["ci_low"] = float(result["ci_low"]) + offset
    corrected["ci_high"] = float(result["ci_high"]) + offset
    key = f"{kwargs['family']}/{kwargs['domain']}/{kwargs['comparison']}"
    _RECENTER_ROWS[key] = {
        "raw_observed_delta": raw_observed,
        "direct_1000_image_delta": direct_delta,
        "recenter_offset": offset,
        "raw_checkpoint": engine._relative(checkpoint),
        "raw_checkpoint_sha256": sha256_file(checkpoint),
    }
    return corrected, adjusted, checkpoint


def _attach_recenter_metadata(
    report: dict[str, Any],
    *,
    path: Path,
    marker_path: Path,
    marker_key: str,
    family: str,
) -> dict[str, Any]:
    if "HazyDet_bootstrap_recenter" not in report:
        rows = {
            key: value
            for key, value in _RECENTER_ROWS.items()
            if key.startswith(f"{family}/HazyDet/")
        }
        if not rows:
            raise QualityUpgradeAnalysisV3Error(
                f"missing HazyDet recenter provenance for {family}"
            )
        report["HazyDet_bootstrap_recenter"] = {
            "rule": "raw_delta_plus_direct_1000_image_delta_minus_raw_observed_delta",
            "rows": rows,
        }
        atomic_write_json(path, report)
        marker = _load_mapping(marker_path)
        marker[marker_key] = sha256_file(path)
        atomic_write_json(marker_path, marker)
    return report


def _factorial() -> dict[str, Any]:
    report = cast(dict[str, Any], _original_factorial())
    return _attach_recenter_metadata(
        report,
        path=FACTORIAL,
        marker_path=FACTORIAL_MARKER,
        marker_key="factorial_statistics_sha256",
        family="factorial",
    )


def _retention() -> dict[str, Any]:
    report = cast(dict[str, Any], _original_retention())
    return _attach_recenter_metadata(
        report,
        path=RETENTION,
        marker_path=RETENTION_MARKER,
        marker_key="retention_statistics_sha256",
        family="retention",
    )


def _configure_engine() -> None:
    _validate_amendment()
    assignments: dict[str, object] = {
        "RUNNER": RUNNER,
        "OUTPUT": OUTPUT,
        "IMPLEMENTATION_LOCK": IMPLEMENTATION_LOCK,
        "IMPLEMENTATION_MARKER": IMPLEMENTATION_MARKER,
        "POINTS": POINTS,
        "POINTS_MARKER": POINTS_MARKER,
        "FACTORIAL": FACTORIAL,
        "FACTORIAL_MARKER": FACTORIAL_MARKER,
        "RETENTION": RETENTION,
        "RETENTION_MARKER": RETENTION_MARKER,
        "THRESHOLDS": THRESHOLDS,
        "THRESHOLDS_MARKER": THRESHOLDS_MARKER,
        "INFLUENCE": INFLUENCE,
        "INFLUENCE_MARKER": INFLUENCE_MARKER,
        "REPORT": REPORT,
        "COMPLETE": COMPLETE,
        "_implementation_lock": _implementation_lock,
        "evaluate_coco": _evaluate_coco_with_registered_hazy_scope,
        "_paired_result": _paired_result,
        "factorial": _factorial,
        "retention": _retention,
    }
    for name, value in assignments.items():
        setattr(engine, name, value)


def _call(stage: str) -> tuple[dict[str, Any], Path]:
    _configure_engine()
    functions = {
        "preflight": (engine.preflight, IMPLEMENTATION_LOCK),
        "points": (engine.points, POINTS),
        "factorial": (_factorial, FACTORIAL),
        "retention": (_retention, RETENTION),
        "thresholds": (engine.thresholds, THRESHOLDS),
        "influence": (engine.influence, INFLUENCE),
        "final": (engine.finalize, REPORT),
    }
    function, artifact = functions[stage]
    return cast(dict[str, Any], function()), artifact


def main() -> int:
    parser = argparse.ArgumentParser(description="Run registered quality analysis v3")
    parser.add_argument(
        "--stage",
        choices=(
            "preflight",
            "points",
            "factorial",
            "retention",
            "thresholds",
            "influence",
            "final",
        ),
        default="final",
    )
    args = parser.parse_args()
    result, artifact = _call(str(args.stage))
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
