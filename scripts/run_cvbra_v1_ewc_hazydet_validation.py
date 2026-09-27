from __future__ import annotations

import argparse
import gc
import json
import platform
from pathlib import Path
from statistics import fmean, median
from typing import Any, cast

from scripts import run_cvbra_v1_plainmix_hazydet_validation as reference_engine

from buse_uav.detectors.ultralytics_adapter import UltralyticsDetector
from buse_uav.evaluation.coco import write_coco_predictions
from buse_uav.schemas import DetectionBatch, ImageRecord
from buse_uav.utils.hashing import sha256_file
from buse_uav.utils.io import atomic_write_json

ROOT = Path(__file__).resolve().parents[1]
RUNNER = Path(__file__).resolve()
PROTOCOL = ROOT / "configs/experiment/cvbra_v1_quality_upgrade_v2.yaml"
PROTOCOL_SHA256 = "14ce7f1edec4248080e545c5a4832a7f20959157c7ef8f84ae692f758aa980d7"
REGISTRATION = ROOT / "reports/development/cvbra_v1_quality_upgrade_v2/REGISTRATION_LOCK.json"
REGISTRATION_SHA256 = "1b218d9a585d82d01222be7990f749a6aa615887a8080e7b17f7994a26063dd7"
REGISTRATION_MARKER = REGISTRATION.parent / "REGISTERED"
REGISTRATION_MARKER_SHA256 = (
    "6ef505f3a5431ba9a84886f5cb951e78eb9e3be3080a391aad27fe4c508b8343"
)
CHECKPOINT_LOCK = REGISTRATION.parent / "CVBRA_EWC/checkpoint_lock.json"
CHECKPOINT_MARKER = REGISTRATION.parent / "CVBRA_EWC/CHECKPOINT_LOCKED"
CHECKPOINT = ROOT / "runs/cvbra_v1_quality_upgrade_v2/CVBRA_EWC/CVBRA_EWC.pt"

MANIFEST = ROOT / "data/manifests/hazydet_val_manifest.json"
MANIFEST_SHA256 = "e8e5ea234dd098348a1906c8aff93d0d773d6ed1c828c09ee98d7cca6b60f916"
VALIDATION_AUDIT = ROOT / "data/manifests/hazydet_val_validation.json"
VALIDATION_AUDIT_SHA256 = "3b14a7f9181ea4820b7eca2837a7baaa52eb01d4a08055f8e1b2d96fd6aed83f"

OUTPUT = REGISTRATION.parent / "CVBRA_EWC/hazydet_validation"
IMPLEMENTATION_LOCK = OUTPUT / "implementation_lock.json"
IMPLEMENTATION_MARKER = OUTPUT / "IMPLEMENTATION_LOCKED"
PREDICTION_LOCK = OUTPUT / "prediction_lock.json"
PREDICTION_MARKER = OUTPUT / "PREDICTIONS_LOCKED"

IMAGES = 1000
REPETITIONS = 2
IMAGE_SIZE = 1280
PROBE_CONF = 0.08
PUBLISH_CONF = 0.25
NMS_IOU = 0.70
MAX_DET = 500
WARMUP_IMAGES = 16
CHUNK_SIZE = 8
CLASS_NAMES = ("car", "bus", "truck")
CATEGORY_ID_BY_CLASS = {0: 0, 1: 1, 2: 2}


class EWCHazyDetValidationError(RuntimeError):
    """Raised when registered EWC HazyDet prediction cannot fail closed."""


def _now() -> str:
    from datetime import datetime, timezone

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
        raise EWCHazyDetValidationError(f"cannot parse mapping {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EWCHazyDetValidationError(f"expected mapping: {path}")
    return value


def _load_rows(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EWCHazyDetValidationError(f"cannot parse rows {path}: {exc}") from exc
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise EWCHazyDetValidationError(f"expected row list: {path}")
    return value


def _assert_hash(path: Path, expected: object, *, label: str) -> None:
    if not path.is_file() or sha256_file(path) != str(expected):
        raise EWCHazyDetValidationError(f"locked {label} changed: {path}")


def _validate_checkpoint() -> tuple[str, str]:
    for path, digest, label in (
        (PROTOCOL, PROTOCOL_SHA256, "quality-upgrade protocol"),
        (REGISTRATION, REGISTRATION_SHA256, "quality-upgrade registration"),
        (REGISTRATION_MARKER, REGISTRATION_MARKER_SHA256, "registration marker"),
        (MANIFEST, MANIFEST_SHA256, "HazyDet validation manifest"),
        (VALIDATION_AUDIT, VALIDATION_AUDIT_SHA256, "HazyDet validation audit"),
    ):
        _assert_hash(path, digest, label=label)
    registration = _load_mapping(REGISTRATION)
    checkpoint = _load_mapping(CHECKPOINT_LOCK)
    marker = _load_mapping(CHECKPOINT_MARKER)
    audit = _load_mapping(VALIDATION_AUDIT)
    if (
        registration.get("retention_baseline") != "CVBRA_EWC"
        or checkpoint.get("status") != "CVBRA_EWC_CHECKPOINT_VERIFIED_AND_LOCKED"
        or checkpoint.get("validation_metric_used_for_training_or_selection") is not False
        or checkpoint.get("official_test_access") != "prohibited"
        or marker.get("checkpoint_lock_sha256") != sha256_file(CHECKPOINT_LOCK)
        or audit.get("valid") is not True
        or not isinstance(audit.get("stats"), dict)
        or audit["stats"].get("images") != IMAGES
    ):
        raise EWCHazyDetValidationError("EWC checkpoint or HazyDet scope changed")
    digest = str(checkpoint["checkpoint_sha256"])
    _assert_hash(CHECKPOINT, digest, label="CVBRA_EWC checkpoint")
    return digest, sha256_file(CHECKPOINT_LOCK)


def _records(*, verify_hashes: bool) -> tuple[ImageRecord, ...]:
    records = cast(
        tuple[ImageRecord, ...],
        reference_engine._records(verify_hashes=verify_hashes),
    )
    if len(records) != IMAGES:
        raise EWCHazyDetValidationError("HazyDet image coverage changed")
    return records


def _cuda_identity() -> dict[str, Any]:
    try:
        import torch
    except (ImportError, OSError) as exc:
        raise EWCHazyDetValidationError(f"cannot inspect CUDA: {exc}") from exc
    if not torch.cuda.is_available():
        raise EWCHazyDetValidationError("EWC HazyDet inference requires CUDA")
    index = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(index)
    return {
        "device": "cuda:0",
        "index": index,
        "name": torch.cuda.get_device_name(index),
        "compute_capability": [int(capability[0]), int(capability[1])],
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
    }


def _implementation_lock() -> dict[str, Any]:
    checkpoint_sha256, checkpoint_lock_sha256 = _validate_checkpoint()
    if IMPLEMENTATION_LOCK.exists() or IMPLEMENTATION_MARKER.exists():
        if not IMPLEMENTATION_LOCK.is_file() or not IMPLEMENTATION_MARKER.is_file():
            raise EWCHazyDetValidationError("EWC HazyDet implementation lock is incomplete")
        lock = _load_mapping(IMPLEMENTATION_LOCK)
        marker = _load_mapping(IMPLEMENTATION_MARKER)
        if (
            lock.get("runner_sha256") != sha256_file(RUNNER)
            or lock.get("checkpoint_lock_sha256") != checkpoint_lock_sha256
            or lock.get("cuda_identity") != _cuda_identity()
            or marker.get("implementation_lock_sha256") != sha256_file(IMPLEMENTATION_LOCK)
        ):
            raise EWCHazyDetValidationError("EWC HazyDet implementation changed")
        return lock
    if PREDICTION_LOCK.exists() or PREDICTION_MARKER.exists():
        raise EWCHazyDetValidationError(
            "EWC HazyDet predictions appeared before implementation lock"
        )
    records = _records(verify_hashes=True)
    payload = {
        "schema_version": 1,
        "status": "CVBRA_EWC_HAZYDET_VALIDATION_IMPLEMENTATION_LOCKED",
        "locked_at_utc": _now(),
        "protocol_sha256": PROTOCOL_SHA256,
        "registration_sha256": REGISTRATION_SHA256,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_lock_sha256": checkpoint_lock_sha256,
        "manifest_sha256": MANIFEST_SHA256,
        "validation_audit_sha256": VALIDATION_AUDIT_SHA256,
        "runner": _relative(RUNNER),
        "runner_sha256": sha256_file(RUNNER),
        "python": platform.python_version(),
        "host": platform.node(),
        "cuda_identity": _cuda_identity(),
        "images": len(records),
        "models": ["CVBRA_EWC"],
        "repetitions": REPETITIONS,
        "probe_confidence": PROBE_CONF,
        "publish_confidence": PUBLISH_CONF,
        "ruff": "PASS",
        "strict_mypy": "PASS",
        "validation_labels_previously_accessed": True,
        "labels_or_annotation_read_by_inference_runner": False,
        "metric_or_selection_feedback_allowed": False,
        "HazyDet_test_access": "prohibited",
        "UAV_OBB_test_access": "prohibited",
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


def preflight() -> dict[str, Any]:
    lock = _implementation_lock()
    return {
        "status": "PASS_CVBRA_EWC_HAZYDET_VALIDATION_PREFLIGHT",
        "images": IMAGES,
        "models": ["CVBRA_EWC"],
        "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
        "runner_sha256": lock["runner_sha256"],
        "metric_accessed": False,
        "HazyDet_test_access": "prohibited",
    }


def _filter_batches(batches: tuple[DetectionBatch, ...]) -> tuple[DetectionBatch, ...]:
    return tuple(
        DetectionBatch(
            image_id=batch.image_id,
            boxes=tuple(box for box in batch.boxes if box.score >= PUBLISH_CONF),
            latency_ms=batch.latency_ms,
            meta={**batch.meta, "publish_conf": PUBLISH_CONF},
        )
        for batch in batches
    )


def _raw_root(repetition: int) -> Path:
    return OUTPUT / "raw/CVBRA_EWC" / f"rep_{repetition}"


def _release_detector() -> None:
    gc.collect()
    try:
        import torch
    except (ImportError, OSError):
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _predict(
    repetition: int,
    detector: UltralyticsDetector,
    records: tuple[ImageRecord, ...],
) -> dict[str, Any]:
    root = _raw_root(repetition)
    marker_path = root / "SUCCESS.json"
    prediction_path = root / "predictions.coco.json"
    timing_path = root / "timings.json"
    if marker_path.exists():
        marker = _load_mapping(marker_path)
        _assert_hash(prediction_path, marker["prediction_sha256"], label="raw prediction")
        _assert_hash(timing_path, marker["timing_sha256"], label="raw timing")
        return marker
    if root.exists():
        raise EWCHazyDetValidationError(f"partial EWC HazyDet inference requires audit: {root}")
    batches = detector.predict(
        records,
        imgsz=IMAGE_SIZE,
        conf=PROBE_CONF,
        iou=NMS_IOU,
        max_det=MAX_DET,
        fp16=True,
    )
    filtered = _filter_batches(tuple(batches))
    write_coco_predictions(
        prediction_path,
        filtered,
        category_id_by_class=CATEGORY_ID_BY_CLASS,
    )
    rows = _load_rows(prediction_path)
    if not rows or {int(row["category_id"]) for row in rows} - {0, 1, 2}:
        raise EWCHazyDetValidationError("EWC HazyDet category export changed")
    atomic_write_json(
        timing_path,
        {
            "images": len(filtered),
            "mean_end_to_end_ms": fmean(float(batch.latency_ms) for batch in filtered),
            "latencies_ms": [float(batch.latency_ms) for batch in filtered],
        },
    )
    marker = {
        "schema_version": 1,
        "status": "CVBRA_EWC_HAZYDET_RAW_CELL_COMPLETE",
        "model": "CVBRA_EWC",
        "repetition": repetition,
        "images": len(filtered),
        "prediction": _relative(prediction_path),
        "prediction_sha256": sha256_file(prediction_path),
        "timing": _relative(timing_path),
        "timing_sha256": sha256_file(timing_path),
        "validation_labels_previously_accessed": True,
        "labels_or_annotation_read_for_prediction": False,
        "metric_or_selection_feedback_used": False,
        "HazyDet_test_access": "prohibited",
    }
    atomic_write_json(marker_path, marker)
    return marker


def _validate_prediction_lock() -> dict[str, Any]:
    _implementation_lock()
    if not PREDICTION_LOCK.is_file() or not PREDICTION_MARKER.is_file():
        raise EWCHazyDetValidationError("EWC HazyDet prediction lock is incomplete")
    lock = _load_mapping(PREDICTION_LOCK)
    marker = _load_mapping(PREDICTION_MARKER)
    if (
        lock.get("status") != "CVBRA_EWC_HAZYDET_PREDICTIONS_LOCKED_BEFORE_METRICS"
        or lock.get("implementation_lock_sha256") != sha256_file(IMPLEMENTATION_LOCK)
        or marker.get("prediction_lock_sha256") != sha256_file(PREDICTION_LOCK)
        or lock.get("metric_or_selection_feedback_used") is not False
    ):
        raise EWCHazyDetValidationError("EWC HazyDet prediction lock changed")
    _assert_hash(_rooted(lock["prediction"]), lock["prediction_sha256"], label="EWC prediction")
    return lock


def infer() -> dict[str, Any]:
    _implementation_lock()
    if PREDICTION_LOCK.exists() or PREDICTION_MARKER.exists():
        return _validate_prediction_lock()
    records = _records(verify_hashes=False)
    detector = UltralyticsDetector(
        CHECKPOINT,
        model_name="yolo11n",
        device="cuda:0",
        expected_class_names=CLASS_NAMES,
        project_root=ROOT,
        stream_chunk_records=CHUNK_SIZE,
        release_cuda_cache_between_chunks=False,
    )
    detector.predict(
        records[:WARMUP_IMAGES],
        imgsz=IMAGE_SIZE,
        conf=PROBE_CONF,
        iou=NMS_IOU,
        max_det=MAX_DET,
        fp16=True,
    )
    markers = [_predict(repetition, detector, records) for repetition in range(1, 3)]
    del detector
    _release_detector()
    hashes = {str(marker["prediction_sha256"]) for marker in markers}
    if len(hashes) != 1:
        raise EWCHazyDetValidationError("EWC HazyDet repetitions differ")
    means = [
        float(_load_mapping(_rooted(marker["timing"]))["mean_end_to_end_ms"])
        for marker in markers
    ]
    payload = {
        "schema_version": 1,
        "status": "CVBRA_EWC_HAZYDET_PREDICTIONS_LOCKED_BEFORE_METRICS",
        "locked_at_utc": _now(),
        "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
        "model": "CVBRA_EWC",
        "images": IMAGES,
        "prediction": markers[0]["prediction"],
        "prediction_sha256": markers[0]["prediction_sha256"],
        "timing_median_mean_ms": median(means),
        "repetitions": REPETITIONS,
        "prediction_determinism": "exact_sha256_match_across_two_repetitions",
        "validation_labels_previously_accessed": True,
        "labels_or_annotation_read_for_prediction": False,
        "metric_or_selection_feedback_used": False,
        "HazyDet_test_access": "prohibited",
        "UAV_OBB_test_access": "prohibited",
    }
    atomic_write_json(PREDICTION_LOCK, payload)
    atomic_write_json(
        PREDICTION_MARKER,
        {"status": payload["status"], "prediction_lock_sha256": sha256_file(PREDICTION_LOCK)},
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run registered EWC HazyDet validation prediction")
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
