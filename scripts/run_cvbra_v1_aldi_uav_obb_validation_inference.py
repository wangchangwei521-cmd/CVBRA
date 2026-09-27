from __future__ import annotations

import argparse
import json
import platform
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts import run_cvbra_v1_aldi_direct_baseline as training
from scripts import run_cvbra_v1_uav_obb_official_validation_inference as engine

from buse_uav.detectors.ultralytics_adapter import UltralyticsDetector
from buse_uav.schemas import ImageRecord
from buse_uav.utils.hashing import sha256_file
from buse_uav.utils.io import atomic_write_json

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/experiment/cvbra_v1_aldi_direct_baseline_v1.yaml"
PROTOCOL_SHA256 = "3e8f28f2d474d4c0caf971cdd2259baadaeac364c9ff168aed694d45481ec3e2"
REGISTRATION = (
    ROOT / "reports/development/cvbra_v1_aldi_direct_baseline_v1/REGISTRATION_LOCK.json"
)
REGISTRATION_SHA256 = "ad7046de7a913321eaa9e528719a309358962cc1d517273db8672fb53413facf"
TRAINING_IMPLEMENTATION = (
    ROOT / "reports/development/cvbra_v1_aldi_direct_baseline_v1/implementation_lock_v4.json"
)
TRAINING_IMPLEMENTATION_SHA256 = (
    "786cf19e1502914bf7c65aa22f4b052bd047ca0c1d98e5af1c3456cfd24be8ac"
)
TRAINING_RUNNER = ROOT / "scripts/run_cvbra_v1_aldi_direct_baseline.py"
TRAINING_RUNNER_SHA256 = "abbeda18ded85f6f5b2fee6d3623d28587f597bb58597e8e8c47623f5d1645e5"
REUSED_ENGINE = ROOT / "scripts/run_cvbra_v1_uav_obb_official_validation_inference.py"
REUSED_ENGINE_SHA256 = "23e630e2c8a508e86efdf2ba0a69563d797df846debb2d0ef5df97f5cf73c79a"

PRIMARY_REPORT = (
    ROOT
    / "reports/development/cvbra_v1/uav_obb_official_validation/evaluation/validation_report.json"
)
PRIMARY_REPORT_SHA256 = "dbe2d7efb5f06d0251cde76fa6b8b9d3f4bf6f61e68652f0f9c9ed3bda81c29b"
MATERIALIZATION_LOCK = (
    ROOT
    / "reports/development/cvbra_v1/uav_obb_official_validation/view_materialization_lock.json"
)
MATERIALIZATION_MARKER = MATERIALIZATION_LOCK.parent / "VIEWS_MATERIALIZED"

OUTPUT = (
    ROOT
    / "reports/development/cvbra_v1_aldi_direct_baseline_v1"
    / "uav_obb_official_validation/evaluation"
)
IMPLEMENTATION_LOCK = OUTPUT / "implementation_lock.json"
IMPLEMENTATION_MARKER = OUTPUT / "IMPLEMENTATION_LOCKED"
RAW_LOCK = OUTPUT / "raw_observation_lock.json"
RAW_MARKER = OUTPUT / "RAW_OBSERVATIONS_LOCKED"
PREDICTION_LOCK = OUTPUT / "joint_prediction_lock.json"
PREDICTION_MARKER = OUTPUT / "PREDICTIONS_LOCKED"

MODELS = tuple(training.PAPER_LABELS[variant] for variant in training.VARIANTS)
WEIGHTS: dict[str, tuple[Path, str]] = {}


class ALDIValidationInferenceError(RuntimeError):
    """Raised when the registered ALDI validation inference cannot fail closed."""


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ALDIValidationInferenceError(f"cannot parse mapping {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ALDIValidationInferenceError(f"expected mapping: {path}")
    return value


def _assert_hash(path: Path, expected: object, *, label: str) -> None:
    if not path.is_file() or sha256_file(path) != str(expected):
        raise ALDIValidationInferenceError(f"locked {label} changed: {path}")


def _validate_protocol_and_checkpoints() -> None:
    global WEIGHTS
    for path, digest, label in (
        (PROTOCOL, PROTOCOL_SHA256, "ALDI direct-comparison protocol"),
        (REGISTRATION, REGISTRATION_SHA256, "ALDI registration"),
        (
            TRAINING_IMPLEMENTATION,
            TRAINING_IMPLEMENTATION_SHA256,
            "ALDI training implementation",
        ),
        (TRAINING_RUNNER, TRAINING_RUNNER_SHA256, "ALDI training runner"),
        (REUSED_ENGINE, REUSED_ENGINE_SHA256, "reused inference engine"),
        (PRIMARY_REPORT, PRIMARY_REPORT_SHA256, "primary validation report"),
    ):
        _assert_hash(path, digest, label=label)
    registration = _load_mapping(REGISTRATION)
    implementation = _load_mapping(TRAINING_IMPLEMENTATION)
    primary = _load_mapping(PRIMARY_REPORT)
    if (
        registration.get("variants") != list(training.VARIANTS)
        or registration.get("paper_body_change_before_result_lock") is not False
        or implementation.get("endpoint") != "fixed_last_epoch_EMA_teacher"
        or implementation.get("validation_metric_used_for_method_or_checkpoint_selection")
        is not False
        or primary.get("status") != "PASS_CVBRA_V1_OFFICIAL_VALIDATION_CONFIRMATION"
    ):
        raise ALDIValidationInferenceError("registered ALDI or primary scope changed")
    weights: dict[str, tuple[Path, str]] = {}
    for variant in training.VARIANTS:
        lock = training._validate_checkpoint_lock(variant)
        label = training.PAPER_LABELS[variant]
        checkpoint = ROOT / str(lock["checkpoint"])
        digest = str(lock["checkpoint_sha256"])
        if (
            lock.get("paper_label") != label
            or lock.get("validation_metric_used_for_training_or_selection") is not False
            or lock.get("official_test_access") != "prohibited"
        ):
            raise ALDIValidationInferenceError(f"ALDI checkpoint scope changed: {variant}")
        _assert_hash(checkpoint, digest, label=f"{label} checkpoint")
        weights[label] = (checkpoint, digest)
    WEIGHTS = weights
    engine.WEIGHTS = weights


def _implementation_lock() -> dict[str, Any]:
    _validate_protocol_and_checkpoints()
    materialization = engine._validate_materialization()
    if IMPLEMENTATION_LOCK.exists() or IMPLEMENTATION_MARKER.exists():
        if not IMPLEMENTATION_LOCK.is_file() or not IMPLEMENTATION_MARKER.is_file():
            raise ALDIValidationInferenceError("ALDI validation implementation lock is incomplete")
        lock = _load_mapping(IMPLEMENTATION_LOCK)
        marker = _load_mapping(IMPLEMENTATION_MARKER)
        if (
            lock.get("runner_sha256") != sha256_file(Path(__file__))
            or lock.get("training_implementation_sha256")
            != TRAINING_IMPLEMENTATION_SHA256
            or lock.get("materialization_lock_sha256") != sha256_file(MATERIALIZATION_LOCK)
            or lock.get("cuda_identity") != engine._cuda_identity()
            or marker.get("implementation_lock_sha256")
            != sha256_file(IMPLEMENTATION_LOCK)
        ):
            raise ALDIValidationInferenceError("ALDI validation implementation changed")
        return lock
    if any(path.exists() for path in (RAW_LOCK, PREDICTION_LOCK)):
        raise ALDIValidationInferenceError("ALDI prediction appeared before implementation lock")
    payload = {
        "schema_version": 1,
        "status": "ALDI_DIRECT_COMPARISON_VALIDATION_IMPLEMENTATION_LOCKED",
        "locked_at_utc": engine._utc_now(),
        "protocol_sha256": PROTOCOL_SHA256,
        "registration_sha256": REGISTRATION_SHA256,
        "training_implementation_sha256": TRAINING_IMPLEMENTATION_SHA256,
        "primary_validation_report_sha256": PRIMARY_REPORT_SHA256,
        "materialization_lock_sha256": sha256_file(MATERIALIZATION_LOCK),
        "runner": engine._relative(Path(__file__)),
        "runner_sha256": sha256_file(Path(__file__)),
        "reused_engine": engine._relative(REUSED_ENGINE),
        "reused_engine_sha256": REUSED_ENGINE_SHA256,
        "python": platform.python_version(),
        "host": platform.node(),
        "cuda_identity": engine._cuda_identity(),
        "weights": {
            model: {"path": engine._relative(path), "sha256": digest}
            for model, (path, digest) in WEIGHTS.items()
        },
        "models": list(MODELS),
        "views": list(engine.VIEWS),
        "orientations": list(engine.ORIENTATIONS),
        "methods": list(engine.METHODS),
        "images": engine.IMAGES,
        "primary_reporting_method": "Identity_one_pass",
        "ruff": "PASS",
        "strict_mypy": "PASS",
        "official_validation_labels_previously_accessed": True,
        "labels_read_by_inference_runner": False,
        "method_or_hyperparameter_selection": False,
        "official_test_content_accessed": False,
        "paper_body_change_authorized": False,
        "materialization_status": materialization["status"],
    }
    atomic_write_json(IMPLEMENTATION_LOCK, payload)
    atomic_write_json(
        IMPLEMENTATION_MARKER,
        {
            "status": payload["status"],
            "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
        },
    )
    return payload


def _validate_raw_lock() -> dict[str, Any]:
    _implementation_lock()
    if not RAW_LOCK.is_file() or not RAW_MARKER.is_file():
        raise ALDIValidationInferenceError("ALDI raw lock is incomplete")
    lock = _load_mapping(RAW_LOCK)
    marker = _load_mapping(RAW_MARKER)
    cells = lock.get("cells")
    expected = len(MODELS) * len(engine.VIEWS) * len(engine.ORIENTATIONS) * engine.REPETITIONS
    if (
        lock.get("status")
        != "ALL_CVBRA_V1_UAV_OBB_OFFICIAL_VALIDATION_RAW_OBSERVATIONS_LOCKED"
        or lock.get("implementation_lock_sha256") != sha256_file(IMPLEMENTATION_LOCK)
        or marker.get("raw_observation_lock_sha256") != sha256_file(RAW_LOCK)
        or not isinstance(cells, list)
        or len(cells) != expected
    ):
        raise ALDIValidationInferenceError("ALDI raw lock changed")
    for row in cells:
        if not isinstance(row, dict):
            raise ALDIValidationInferenceError("invalid ALDI raw observation")
        _assert_hash(
            engine._rooted(row["prediction"]),
            row["prediction_sha256"],
            label="prediction",
        )
        _assert_hash(engine._rooted(row["timing"]), row["timing_sha256"], label="timing")
    return lock


def _validate_prediction_lock() -> dict[str, Any]:
    _validate_raw_lock()
    if not PREDICTION_LOCK.is_file() or not PREDICTION_MARKER.is_file():
        raise ALDIValidationInferenceError("ALDI prediction lock is incomplete")
    lock = _load_mapping(PREDICTION_LOCK)
    marker = _load_mapping(PREDICTION_MARKER)
    artifacts = lock.get("artifacts")
    expected = len(MODELS) * len(engine.VIEWS) * len(engine.METHODS)
    if (
        lock.get("status")
        != "CVBRA_V1_UAV_OBB_OFFICIAL_VALIDATION_PREDICTIONS_LOCKED_BEFORE_LABELS_OR_METRICS"
        or lock.get("raw_observation_lock_sha256") != sha256_file(RAW_LOCK)
        or marker.get("joint_prediction_lock_sha256") != sha256_file(PREDICTION_LOCK)
        or not isinstance(artifacts, list)
        or len(artifacts) != expected
    ):
        raise ALDIValidationInferenceError("ALDI prediction lock changed")
    for row in artifacts:
        if not isinstance(row, dict):
            raise ALDIValidationInferenceError("invalid ALDI prediction artifact")
        _assert_hash(
            engine._rooted(row["prediction"]), row["prediction_sha256"], label="prediction"
        )
    return lock


_original_predict_cell = engine._predict_cell
_original_infer = engine.infer
_original_derive = engine.derive


def _predict_cell(
    model: str,
    view: str,
    orientation: str,
    repetition: int,
    detector: UltralyticsDetector,
    records: Sequence[ImageRecord],
) -> None:
    _original_predict_cell(model, view, orientation, repetition, detector, records)
    marker_path = engine._raw_root(model, view, orientation, repetition) / "SUCCESS.json"
    marker = _load_mapping(marker_path)
    marker["official_validation_labels_previously_accessed"] = True
    marker["labels_read_by_inference_runner"] = False
    marker["method_or_hyperparameter_selection"] = False
    atomic_write_json(marker_path, marker)


def _configure_engine() -> None:
    assignments: Mapping[str, object] = {
        "__file__": str(Path(__file__).resolve()),
        "PROTOCOL": PROTOCOL,
        "PROTOCOL_SHA256": PROTOCOL_SHA256,
        "MATERIALIZATION_LOCK": MATERIALIZATION_LOCK,
        "MATERIALIZATION_MARKER": MATERIALIZATION_MARKER,
        "OUTPUT": OUTPUT,
        "IMPLEMENTATION_LOCK": IMPLEMENTATION_LOCK,
        "IMPLEMENTATION_MARKER": IMPLEMENTATION_MARKER,
        "RAW_LOCK": RAW_LOCK,
        "RAW_MARKER": RAW_MARKER,
        "PREDICTION_LOCK": PREDICTION_LOCK,
        "PREDICTION_MARKER": PREDICTION_MARKER,
        "MODELS": MODELS,
        "WEIGHTS": WEIGHTS,
        "WARMUP_IMAGES": 16,
        "CHUNK_SIZE": 8,
        "_validate_protocol_and_checkpoint": _validate_protocol_and_checkpoints,
        "_implementation_lock": _implementation_lock,
        "_validate_raw_lock": _validate_raw_lock,
        "_validate_prediction_lock": _validate_prediction_lock,
        "_predict_cell": _predict_cell,
    }
    for name, value in assignments.items():
        setattr(engine, name, value)


def _mark_prior_label_access(
    payload: dict[str, Any], path: Path, marker_path: Path, marker_key: str
) -> dict[str, Any]:
    payload["official_validation_labels_previously_accessed"] = True
    payload["labels_read_by_inference_runner"] = False
    payload["method_or_hyperparameter_selection"] = False
    payload["official_test_content_accessed"] = False
    atomic_write_json(path, payload)
    marker = _load_mapping(marker_path)
    marker[marker_key] = sha256_file(path)
    atomic_write_json(marker_path, marker)
    return payload


def preflight() -> dict[str, Any]:
    _configure_engine()
    result = engine.preflight()
    result["official_validation_labels_previously_accessed"] = True
    result["labels_read_by_inference_runner"] = False
    result["method_or_hyperparameter_selection"] = False
    return result


def infer() -> dict[str, Any]:
    _configure_engine()
    result = _original_infer()
    return _mark_prior_label_access(result, RAW_LOCK, RAW_MARKER, "raw_observation_lock_sha256")


def derive() -> dict[str, Any]:
    _configure_engine()
    engine.infer = infer
    try:
        result = _original_derive()
    finally:
        engine.infer = _original_infer
    return _mark_prior_label_access(
        result, PREDICTION_LOCK, PREDICTION_MARKER, "joint_prediction_lock_sha256"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run registered ALDI direct-comparison inference")
    parser.add_argument("--stage", choices=("preflight", "infer", "derive"), default="derive")
    args = parser.parse_args()
    if args.stage == "preflight":
        result = preflight()
    elif args.stage == "infer":
        result = infer()
    else:
        result = derive()
    summary = {key: value for key, value in result.items() if key not in {"cells", "artifacts"}}
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
