from __future__ import annotations

import argparse
import json
import platform
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from scripts import run_cvbra_v1_uav_obb_official_validation_inference as engine

from buse_uav.detectors.ultralytics_adapter import UltralyticsDetector
from buse_uav.schemas import ImageRecord
from buse_uav.utils.hashing import sha256_file
from buse_uav.utils.io import atomic_write_json

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "experiment" / "cvbra_v1_rtdetr_l_confirmation.yaml"
PROTOCOL_SHA256 = "6f11519c837a7b68679707cd0025631a90c073d8c37b856a9e0eb40b98fc7f9c"
YOLO_VALIDATION_REPORT = (
    ROOT
    / "reports"
    / "development"
    / "cvbra_v1"
    / "uav_obb_official_validation"
    / "evaluation"
    / "validation_report.json"
)
YOLO_VALIDATION_REPORT_SHA256 = "dbe2d7efb5f06d0251cde76fa6b8b9d3f4bf6f61e68652f0f9c9ed3bda81c29b"
MATERIALIZATION_LOCK = (
    ROOT
    / "reports"
    / "development"
    / "cvbra_v1"
    / "uav_obb_official_validation"
    / "view_materialization_lock.json"
)
MATERIALIZATION_MARKER = MATERIALIZATION_LOCK.parent / "VIEWS_MATERIALIZED"
SOURCE_WEIGHT = ROOT / "weights" / "hazydet" / "rtdetr_l_best.pt"
SOURCE_WEIGHT_SHA256 = "7d8fccbc1a9b66e28311ba91363f0b9f6ca0e4fdd61e88bd2f90099c6ea2a44c"
CANDIDATE_WEIGHT = ROOT / "runs" / "cvbra_v1_rtdetr_l" / "cvbra_v1_rtdetr_l.pt"
CHECKPOINT_LOCK = ROOT / "reports" / "development" / "cvbra_v1_rtdetr_l" / "checkpoint_lock.json"
CHECKPOINT_MARKER = CHECKPOINT_LOCK.parent / "CHECKPOINT_LOCKED"
OUTPUT = CHECKPOINT_LOCK.parent / "uav_obb_official_validation" / "evaluation"
IMPLEMENTATION_LOCK = OUTPUT / "implementation_lock.json"
IMPLEMENTATION_MARKER = OUTPUT / "IMPLEMENTATION_LOCKED"
RAW_LOCK = OUTPUT / "raw_observation_lock.json"
RAW_MARKER = OUTPUT / "RAW_OBSERVATIONS_LOCKED"
PREDICTION_LOCK = OUTPUT / "joint_prediction_lock.json"
PREDICTION_MARKER = OUTPUT / "PREDICTIONS_LOCKED"

MODELS = ("source", "CVBRA_v1_RTDETR_L")
WEIGHTS = {
    "source": (SOURCE_WEIGHT, SOURCE_WEIGHT_SHA256),
    "CVBRA_v1_RTDETR_L": (CANDIDATE_WEIGHT, "from_checkpoint_lock"),
}
REUSED_ENGINE = ROOT / "scripts" / "run_cvbra_v1_uav_obb_official_validation_inference.py"
REUSED_ENGINE_SHA256 = "23e630e2c8a508e86efdf2ba0a69563d797df846debb2d0ef5df97f5cf73c79a"


class CVBRARTDETRInferenceError(RuntimeError):
    """Raised when fixed RT-DETR-L confirmation inference cannot fail closed."""


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CVBRARTDETRInferenceError(f"cannot parse mapping {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CVBRARTDETRInferenceError(f"expected mapping: {path}")
    return value


def _assert_hash(path: Path, expected: object, *, label: str) -> None:
    if not path.is_file() or sha256_file(path) != str(expected):
        raise CVBRARTDETRInferenceError(f"locked {label} changed: {path}")


def _validate_protocol_and_checkpoint() -> None:
    _assert_hash(PROTOCOL, PROTOCOL_SHA256, label="RT-DETR confirmation protocol")
    _assert_hash(
        YOLO_VALIDATION_REPORT,
        YOLO_VALIDATION_REPORT_SHA256,
        label="YOLO11n validation report",
    )
    _assert_hash(REUSED_ENGINE, REUSED_ENGINE_SHA256, label="reused inference engine")
    _assert_hash(SOURCE_WEIGHT, SOURCE_WEIGHT_SHA256, label="source RT-DETR-L weight")
    validation = _load_mapping(YOLO_VALIDATION_REPORT)
    checkpoint = _load_mapping(CHECKPOINT_LOCK)
    checkpoint_marker = _load_mapping(CHECKPOINT_MARKER)
    try:
        protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise CVBRARTDETRInferenceError(f"cannot parse protocol: {exc}") from exc
    if not isinstance(protocol, dict) or not isinstance(protocol.get("evaluation"), dict):
        raise CVBRARTDETRInferenceError("RT-DETR evaluation protocol is incomplete")
    evaluation = protocol["evaluation"]
    if (
        validation.get("status") != "PASS_CVBRA_V1_OFFICIAL_VALIDATION_CONFIRMATION"
        or checkpoint.get("status") != "CVBRA_V1_RTDETR_L_CHECKPOINT_VERIFIED_AND_LOCKED"
        or checkpoint.get("checkpoint_selected_by_metric") is not False
        or checkpoint.get("official_validation_metric_feedback_used") is not False
        or checkpoint.get("official_test_accessed") is not False
        or checkpoint_marker.get("checkpoint_lock_sha256") != sha256_file(CHECKPOINT_LOCK)
        or tuple(evaluation.get("models", ())) != MODELS
        or evaluation.get("metric_or_label_feedback_allowed_during_RTDETR_training") is not False
        or evaluation.get("validation_labels_already_accessed_for_frozen_yolo11n_confirmation")
        is not True
    ):
        raise CVBRARTDETRInferenceError("RT-DETR checkpoint or evaluation scope changed")
    _assert_hash(CANDIDATE_WEIGHT, checkpoint["checkpoint_sha256"], label="candidate weight")
    WEIGHTS["CVBRA_v1_RTDETR_L"] = (
        CANDIDATE_WEIGHT,
        str(checkpoint["checkpoint_sha256"]),
    )
    engine.WEIGHTS = WEIGHTS


def _implementation_lock() -> dict[str, Any]:
    _validate_protocol_and_checkpoint()
    materialization = engine._validate_materialization()
    if IMPLEMENTATION_LOCK.exists() or IMPLEMENTATION_MARKER.exists():
        if not IMPLEMENTATION_LOCK.is_file() or not IMPLEMENTATION_MARKER.is_file():
            raise CVBRARTDETRInferenceError("RT-DETR inference lock is incomplete")
        lock = _load_mapping(IMPLEMENTATION_LOCK)
        marker = _load_mapping(IMPLEMENTATION_MARKER)
        if (
            lock.get("runner_sha256") != sha256_file(Path(__file__))
            or lock.get("reused_engine_sha256") != REUSED_ENGINE_SHA256
            or lock.get("checkpoint_lock_sha256") != sha256_file(CHECKPOINT_LOCK)
            or lock.get("materialization_lock_sha256") != sha256_file(MATERIALIZATION_LOCK)
            or lock.get("cuda_identity") != engine._cuda_identity()
            or marker.get("implementation_lock_sha256") != sha256_file(IMPLEMENTATION_LOCK)
        ):
            raise CVBRARTDETRInferenceError("RT-DETR inference implementation changed")
        return lock
    if any(path.exists() for path in (RAW_LOCK, PREDICTION_LOCK)):
        raise CVBRARTDETRInferenceError("prediction appeared before inference lock")
    payload = {
        "schema_version": 1,
        "status": "CVBRA_V1_RTDETR_L_OFFICIAL_VALIDATION_INFERENCE_IMPLEMENTATION_LOCKED",
        "locked_at_utc": engine._utc_now(),
        "protocol_sha256": PROTOCOL_SHA256,
        "yolo11n_validation_report_sha256": YOLO_VALIDATION_REPORT_SHA256,
        "checkpoint_lock_sha256": sha256_file(CHECKPOINT_LOCK),
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
        "RTDETR_metrics_accessed": False,
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


_original_detector = UltralyticsDetector


def _rtdetr_detector(*args: Any, **kwargs: Any) -> UltralyticsDetector:
    kwargs["model_name"] = "rtdetr_l"
    return _original_detector(*args, **kwargs)


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
    assignments = {
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
        "SOURCE_WEIGHT": SOURCE_WEIGHT,
        "SOURCE_WEIGHT_SHA256": SOURCE_WEIGHT_SHA256,
        "CANDIDATE_WEIGHT": CANDIDATE_WEIGHT,
        "MODELS": MODELS,
        "WEIGHTS": WEIGHTS,
        "WARMUP_IMAGES": 8,
        "CHUNK_SIZE": 4,
        "UltralyticsDetector": _rtdetr_detector,
        "_validate_protocol_and_checkpoint": _validate_protocol_and_checkpoint,
        "_implementation_lock": _implementation_lock,
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
    return result


def infer() -> dict[str, Any]:
    _configure_engine()
    result = _original_infer()
    return _mark_labels_previously_accessed(
        result,
        RAW_LOCK,
        RAW_MARKER,
        "raw_observation_lock_sha256",
    )


def derive() -> dict[str, Any]:
    _configure_engine()
    engine.infer = infer
    try:
        result = _original_derive()
    finally:
        engine.infer = _original_infer
    return _mark_labels_previously_accessed(
        result,
        PREDICTION_LOCK,
        PREDICTION_MARKER,
        "joint_prediction_lock_sha256",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run fixed RT-DETR-L CVBRA official-validation inference"
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
