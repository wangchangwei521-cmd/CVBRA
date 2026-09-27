from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from scripts import run_cvbra_v1_ewc_hazydet_validation as engine

from buse_uav.utils.hashing import sha256_file
from buse_uav.utils.io import atomic_write_json

ROOT = Path(__file__).resolve().parents[1]
RUNNER = Path(__file__).resolve()
AMENDMENT = (
    ROOT
    / "reports/development/cvbra_v1_quality_upgrade_v2/"
    "HAZYDET_INFERENCE_AMENDMENT_V2.json"
)
AMENDMENT_MARKER = AMENDMENT.with_name("HAZYDET_INFERENCE_V2_REGISTERED")
OUTPUT = (
    ROOT
    / "reports/development/cvbra_v1_quality_upgrade_v2/CVBRA_EWC/"
    "hazydet_validation_v2"
)
IMPLEMENTATION_LOCK = OUTPUT / "implementation_lock.json"
IMPLEMENTATION_MARKER = OUTPUT / "IMPLEMENTATION_LOCKED"
PREDICTION_LOCK = OUTPUT / "prediction_lock.json"
PREDICTION_MARKER = OUTPUT / "PREDICTIONS_LOCKED"
CLASS_NAMES = ("car", "truck", "bus")
V1_RUNNER = engine.RUNNER
V1_IMPLEMENTATION_LOCK = engine.IMPLEMENTATION_LOCK


class EWCHazyDetValidationV2Error(RuntimeError):
    """Raised when the registered class-order correction cannot fail closed."""


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EWCHazyDetValidationV2Error(f"cannot parse mapping {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EWCHazyDetValidationV2Error(f"expected mapping: {path}")
    return value


def _validate_amendment() -> dict[str, Any]:
    if not AMENDMENT.is_file() or not AMENDMENT_MARKER.is_file():
        raise EWCHazyDetValidationV2Error("HazyDet v2 amendment is not registered")
    amendment = _load_mapping(AMENDMENT)
    marker = _load_mapping(AMENDMENT_MARKER)
    if (
        amendment.get("status")
        != "REGISTERED_CLASS_ORDER_CORRECTION_BEFORE_HAZYDET_PREDICTION_V2"
        or amendment.get("failed_v1_runner_sha256") != sha256_file(V1_RUNNER)
        or amendment.get("failed_v1_implementation_lock_sha256")
        != sha256_file(V1_IMPLEMENTATION_LOCK)
        or amendment.get("v1_prediction_artifacts_existed_at_amendment") is not False
        or amendment.get("v2_runner_sha256") != sha256_file(RUNNER)
        or amendment.get("only_change") != "expected_class_order_car_truck_bus"
        or marker.get("amendment_sha256") != sha256_file(AMENDMENT)
    ):
        raise EWCHazyDetValidationV2Error("HazyDet v2 amendment changed")
    return amendment


_original_implementation_lock = engine._implementation_lock


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
        lock["corrected_expected_class_names"] = list(CLASS_NAMES)
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
        or lock.get("corrected_expected_class_names") != list(CLASS_NAMES)
    ):
        raise EWCHazyDetValidationV2Error("HazyDet v2 implementation lock changed")
    return lock


def _configure_engine() -> None:
    _validate_amendment()
    engine.RUNNER = RUNNER
    engine.OUTPUT = OUTPUT
    engine.IMPLEMENTATION_LOCK = IMPLEMENTATION_LOCK
    engine.IMPLEMENTATION_MARKER = IMPLEMENTATION_MARKER
    engine.PREDICTION_LOCK = PREDICTION_LOCK
    engine.PREDICTION_MARKER = PREDICTION_MARKER
    engine.CLASS_NAMES = CLASS_NAMES
    engine._implementation_lock = _implementation_lock


def preflight() -> dict[str, Any]:
    _configure_engine()
    result = cast(dict[str, Any], engine.preflight())
    result["amendment_sha256"] = sha256_file(AMENDMENT)
    result["corrected_expected_class_names"] = list(CLASS_NAMES)
    return result


def infer() -> dict[str, Any]:
    _configure_engine()
    return cast(dict[str, Any], engine.infer())


def main() -> int:
    parser = argparse.ArgumentParser(description="Run corrected registered EWC HazyDet prediction")
    parser.add_argument("--stage", choices=("preflight", "infer"), default="infer")
    args = parser.parse_args()
    result = preflight() if args.stage == "preflight" else infer()
    artifact = IMPLEMENTATION_LOCK if args.stage == "preflight" else PREDICTION_LOCK
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
