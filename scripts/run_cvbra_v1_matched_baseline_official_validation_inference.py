from __future__ import annotations

import argparse
import json
import platform
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts import run_cvbra_v1_uav_obb_official_validation_inference as engine

from buse_uav.detectors.ultralytics_adapter import UltralyticsDetector
from buse_uav.schemas import ImageRecord
from buse_uav.utils.hashing import sha256_file
from buse_uav.utils.io import atomic_write_json

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "experiment" / "cvbra_v1_matched_baselines.yaml"
PROTOCOL_SHA256 = "1671b6f13b9f1ffad0af6b2bcb20c5ac2c680e5368174505be8c884bf3dd7b9b"
REGISTRATION = (
    ROOT / "reports" / "development" / "cvbra_v1_matched_baselines" / "REGISTRATION_LOCK.json"
)
REGISTRATION_SHA256 = "aaf2e18153b10605926c4c93593db071ee7f0e503f54ef5da89a13d8f7a26278"
TRAINING_IMPLEMENTATION = REGISTRATION.parent / "implementation_lock.json"
TRAINING_IMPLEMENTATION_SHA256 = "e38621bfdd58f40764782f9dc588ffc028519fdaab6fab12501b2c2b3ca11167"
PRIMARY_REPORT = (
    ROOT
    / "reports"
    / "development"
    / "cvbra_v1"
    / "uav_obb_official_validation"
    / "evaluation"
    / "validation_report.json"
)
PRIMARY_REPORT_SHA256 = "dbe2d7efb5f06d0251cde76fa6b8b9d3f4bf6f61e68652f0f9c9ed3bda81c29b"
MATERIALIZATION_LOCK = (
    ROOT
    / "reports"
    / "development"
    / "cvbra_v1"
    / "uav_obb_official_validation"
    / "view_materialization_lock.json"
)
MATERIALIZATION_MARKER = MATERIALIZATION_LOCK.parent / "VIEWS_MATERIALIZED"
OUTPUT = REGISTRATION.parent / "uav_obb_official_validation" / "evaluation"
IMPLEMENTATION_LOCK = OUTPUT / "implementation_lock.json"
IMPLEMENTATION_MARKER = OUTPUT / "IMPLEMENTATION_LOCKED"
RAW_LOCK = OUTPUT / "raw_observation_lock.json"
RAW_MARKER = OUTPUT / "RAW_OBSERVATIONS_LOCKED"
PREDICTION_LOCK = OUTPUT / "joint_prediction_lock.json"
PREDICTION_MARKER = OUTPUT / "PREDICTIONS_LOCKED"

MODELS = ("STF", "CVBRA_noCV", "CVBRA_noReplay", "CVBRA_noFreeze")
WEIGHTS: dict[str, tuple[Path, str]] = {
    model: (ROOT / "runs" / "cvbra_v1_matched_baselines" / model / f"{model}.pt", "pending")
    for model in MODELS
}
REUSED_ENGINE = ROOT / "scripts" / "run_cvbra_v1_uav_obb_official_validation_inference.py"
REUSED_ENGINE_SHA256 = "23e630e2c8a508e86efdf2ba0a69563d797df846debb2d0ef5df97f5cf73c79a"


class MatchedBaselineInferenceError(RuntimeError):
    """Raised when fixed matched-baseline inference cannot fail closed."""


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MatchedBaselineInferenceError(f"cannot parse mapping {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MatchedBaselineInferenceError(f"expected mapping: {path}")
    return value


def _assert_hash(path: Path, expected: object, *, label: str) -> None:
    if not path.is_file() or sha256_file(path) != str(expected):
        raise MatchedBaselineInferenceError(f"locked {label} changed: {path}")


def _checkpoint_lock_path(model: str) -> Path:
    return REGISTRATION.parent / "checkpoint_locks" / f"{model}.json"


def _checkpoint_marker_path(model: str) -> Path:
    return REGISTRATION.parent / "checkpoint_locks" / f"{model}.LOCKED"


def _validate_protocol_and_checkpoints() -> None:
    for path, digest, label in (
        (PROTOCOL, PROTOCOL_SHA256, "matched-baseline protocol"),
        (REGISTRATION, REGISTRATION_SHA256, "registration lock"),
        (TRAINING_IMPLEMENTATION, TRAINING_IMPLEMENTATION_SHA256, "training implementation"),
        (PRIMARY_REPORT, PRIMARY_REPORT_SHA256, "primary validation report"),
        (REUSED_ENGINE, REUSED_ENGINE_SHA256, "reused inference engine"),
    ):
        _assert_hash(path, digest, label=label)
    registration = _load_mapping(REGISTRATION)
    primary = _load_mapping(PRIMARY_REPORT)
    if (
        registration.get("registered_baselines") != list(MODELS)
        or registration.get("method_or_hyperparameter_selection_from_this_study") is not False
        or primary.get("status") != "PASS_CVBRA_V1_OFFICIAL_VALIDATION_CONFIRMATION"
    ):
        raise MatchedBaselineInferenceError("baseline scope or primary evidence changed")
    for model in MODELS:
        lock_path = _checkpoint_lock_path(model)
        marker_path = _checkpoint_marker_path(model)
        lock = _load_mapping(lock_path)
        marker = _load_mapping(marker_path)
        weight = WEIGHTS[model][0]
        if (
            lock.get("status") != "CVBRA_V1_MATCHED_BASELINE_CHECKPOINT_VERIFIED_AND_LOCKED"
            or lock.get("baseline") != model
            or lock.get("validation_metric_used_for_training_or_selection") is not False
            or lock.get("official_test_access") != "prohibited"
            or marker.get("checkpoint_lock_sha256") != sha256_file(lock_path)
        ):
            raise MatchedBaselineInferenceError(f"checkpoint lock changed for {model}")
        digest = str(lock["checkpoint_sha256"])
        _assert_hash(weight, digest, label=f"{model} checkpoint")
        WEIGHTS[model] = (weight, digest)
    engine.WEIGHTS = WEIGHTS


def _implementation_lock() -> dict[str, Any]:
    _validate_protocol_and_checkpoints()
    materialization = engine._validate_materialization()
    if IMPLEMENTATION_LOCK.exists() or IMPLEMENTATION_MARKER.exists():
        if not IMPLEMENTATION_LOCK.is_file() or not IMPLEMENTATION_MARKER.is_file():
            raise MatchedBaselineInferenceError("baseline inference lock is incomplete")
        lock = _load_mapping(IMPLEMENTATION_LOCK)
        marker = _load_mapping(IMPLEMENTATION_MARKER)
        if (
            lock.get("runner_sha256") != sha256_file(Path(__file__))
            or lock.get("reused_engine_sha256") != REUSED_ENGINE_SHA256
            or lock.get("training_implementation_sha256") != TRAINING_IMPLEMENTATION_SHA256
            or lock.get("materialization_lock_sha256") != sha256_file(MATERIALIZATION_LOCK)
            or lock.get("cuda_identity") != engine._cuda_identity()
            or marker.get("implementation_lock_sha256") != sha256_file(IMPLEMENTATION_LOCK)
        ):
            raise MatchedBaselineInferenceError("baseline inference implementation changed")
        return lock
    if any(path.exists() for path in (RAW_LOCK, PREDICTION_LOCK)):
        raise MatchedBaselineInferenceError("baseline prediction appeared before inference lock")
    payload = {
        "schema_version": 1,
        "status": "CVBRA_V1_MATCHED_BASELINE_INFERENCE_IMPLEMENTATION_LOCKED",
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
        "ruff": "PASS",
        "strict_mypy": "PASS",
        "official_validation_labels_previously_accessed": True,
        "labels_read_by_inference_runner": False,
        "baseline_metrics_accessed": False,
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
        raise MatchedBaselineInferenceError("baseline raw lock is incomplete")
    lock = _load_mapping(RAW_LOCK)
    marker = _load_mapping(RAW_MARKER)
    cells = lock.get("cells")
    expected_cells = len(MODELS) * len(engine.VIEWS) * len(engine.ORIENTATIONS) * engine.REPETITIONS
    if (
        lock.get("status") != "ALL_CVBRA_V1_UAV_OBB_OFFICIAL_VALIDATION_RAW_OBSERVATIONS_LOCKED"
        or lock.get("implementation_lock_sha256") != sha256_file(IMPLEMENTATION_LOCK)
        or marker.get("raw_observation_lock_sha256") != sha256_file(RAW_LOCK)
        or not isinstance(cells, list)
        or len(cells) != expected_cells
    ):
        raise MatchedBaselineInferenceError("baseline raw lock changed")
    for row in cells:
        if not isinstance(row, dict):
            raise MatchedBaselineInferenceError("invalid baseline raw row")
        _assert_hash(
            engine._rooted(row["prediction"]), row["prediction_sha256"], label="prediction"
        )
        _assert_hash(engine._rooted(row["timing"]), row["timing_sha256"], label="timing")
    return lock


def _validate_prediction_lock() -> dict[str, Any]:
    _validate_raw_lock()
    if not PREDICTION_LOCK.is_file() or not PREDICTION_MARKER.is_file():
        raise MatchedBaselineInferenceError("baseline prediction lock is incomplete")
    lock = _load_mapping(PREDICTION_LOCK)
    marker = _load_mapping(PREDICTION_MARKER)
    artifacts = lock.get("artifacts")
    expected_artifacts = len(MODELS) * len(engine.VIEWS) * len(engine.METHODS)
    if (
        lock.get("status")
        != "CVBRA_V1_UAV_OBB_OFFICIAL_VALIDATION_PREDICTIONS_LOCKED_BEFORE_LABELS_OR_METRICS"
        or lock.get("raw_observation_lock_sha256") != sha256_file(RAW_LOCK)
        or marker.get("joint_prediction_lock_sha256") != sha256_file(PREDICTION_LOCK)
        or not isinstance(artifacts, list)
        or len(artifacts) != expected_artifacts
    ):
        raise MatchedBaselineInferenceError("baseline prediction lock changed")
    for row in artifacts:
        if not isinstance(row, dict):
            raise MatchedBaselineInferenceError("invalid baseline prediction row")
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
    marker["official_validation_labels_accessed"] = True
    marker["labels_read_by_inference_runner"] = False
    marker["official_test_content_accessed"] = False
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


def _mark_labels_previously_accessed(
    payload: dict[str, Any], path: Path, marker_path: Path, marker_key: str
) -> dict[str, Any]:
    payload["official_validation_labels_previously_accessed"] = True
    payload["official_validation_labels_accessed"] = True
    payload["labels_read_by_inference_runner"] = False
    payload["official_test_content_accessed"] = False
    payload["method_or_hyperparameter_selection"] = False
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
    return _mark_labels_previously_accessed(
        result, RAW_LOCK, RAW_MARKER, "raw_observation_lock_sha256"
    )


def derive() -> dict[str, Any]:
    _configure_engine()
    engine.infer = infer
    try:
        result = _original_derive()
    finally:
        engine.infer = _original_infer
    return _mark_labels_previously_accessed(
        result, PREDICTION_LOCK, PREDICTION_MARKER, "joint_prediction_lock_sha256"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run frozen CVBRA matched-baseline official-validation inference"
    )
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
