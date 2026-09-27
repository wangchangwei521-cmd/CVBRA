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
    "ANALYSIS_AMENDMENT_V2.json"
)
AMENDMENT_MARKER = AMENDMENT.with_name("ANALYSIS_V2_REGISTERED")
OUTPUT = ROOT / "reports/development/cvbra_v1_quality_upgrade_v2/analysis_v2"
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
_original_implementation_lock = engine._implementation_lock
_original_evaluate_coco = engine.evaluate_coco


class QualityUpgradeAnalysisV2Error(RuntimeError):
    """Raised when the registered image-coverage correction cannot fail closed."""


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise QualityUpgradeAnalysisV2Error(f"cannot parse mapping {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise QualityUpgradeAnalysisV2Error(f"expected mapping: {path}")
    return value


def _validate_amendment() -> dict[str, Any]:
    if not AMENDMENT.is_file() or not AMENDMENT_MARKER.is_file():
        raise QualityUpgradeAnalysisV2Error("analysis v2 amendment is not registered")
    amendment = _load_mapping(AMENDMENT)
    marker = _load_mapping(AMENDMENT_MARKER)
    if (
        amendment.get("status")
        != "REGISTERED_HAZYDET_IMAGE_SCOPE_CORRECTION_BEFORE_ANALYSIS_V2"
        or amendment.get("failed_v1_runner_sha256") != sha256_file(V1_RUNNER)
        or amendment.get("failed_v1_implementation_lock_sha256")
        != sha256_file(V1_IMPLEMENTATION_LOCK)
        or amendment.get("v1_derived_artifacts_existed_at_amendment") is not False
        or amendment.get("v2_runner_sha256") != sha256_file(RUNNER)
        or amendment.get("only_change")
        != "explicit_all_1000_HazyDet_annotation_image_ids_for_point_evaluation"
        or marker.get("amendment_sha256") != sha256_file(AMENDMENT)
    ):
        raise QualityUpgradeAnalysisV2Error("analysis v2 amendment changed")
    return amendment


def _implementation_lock() -> dict[str, Any]:
    amendment = _validate_amendment()
    existed = IMPLEMENTATION_LOCK.exists() or IMPLEMENTATION_MARKER.exists()
    lock = cast(dict[str, Any], _original_implementation_lock())
    if not existed:
        lock["amendment"] = engine._relative(AMENDMENT)
        lock["amendment_sha256"] = sha256_file(AMENDMENT)
        lock["failed_v1_implementation_lock_sha256"] = amendment[
            "failed_v1_implementation_lock_sha256"
        ]
        lock["HazyDet_point_image_scope"] = "all_1000_annotation_image_ids"
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
    ):
        raise QualityUpgradeAnalysisV2Error("analysis v2 implementation lock changed")
    return lock


def _evaluate_coco_with_registered_hazy_scope(
    annotation_path: Path,
    prediction_path: Path,
    **kwargs: Any,
) -> dict[str, Any]:
    if annotation_path.resolve() == engine.HAZY_ANNOTATION.resolve() and kwargs.get(
        "image_ids"
    ) is None:
        kwargs["image_ids"] = [
            image_id
            for group in engine._hazy_clusters().values()
            for image_id in group
        ]
    return cast(
        dict[str, Any],
        _original_evaluate_coco(annotation_path, prediction_path, **kwargs),
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
    }
    for name, value in assignments.items():
        setattr(engine, name, value)


def _call(stage: str) -> tuple[dict[str, Any], Path]:
    _configure_engine()
    functions = {
        "preflight": (engine.preflight, IMPLEMENTATION_LOCK),
        "points": (engine.points, POINTS),
        "factorial": (engine.factorial, FACTORIAL),
        "retention": (engine.retention, RETENTION),
        "thresholds": (engine.thresholds, THRESHOLDS),
        "influence": (engine.influence, INFLUENCE),
        "final": (engine.finalize, REPORT),
    }
    function, artifact = functions[stage]
    return cast(dict[str, Any], function()), artifact


def main() -> int:
    parser = argparse.ArgumentParser(description="Run corrected registered quality analysis")
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
