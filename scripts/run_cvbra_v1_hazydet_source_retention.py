from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import platform
from collections.abc import Mapping, Sequence
from io import StringIO
from pathlib import Path
from statistics import fmean, median, stdev
from typing import Any

from PIL import Image

from buse_uav.detectors.ultralytics_adapter import UltralyticsDetector
from buse_uav.evaluation.bootstrap import paired_coco_ap_bootstrap
from buse_uav.evaluation.coco import evaluate_coco, write_coco_predictions
from buse_uav.evaluation.supplementary import bootstrap_sign_pvalue, holm_adjust
from buse_uav.schemas import DetectionBatch, ImageRecord
from buse_uav.utils.hashing import sha256_file
from buse_uav.utils.io import atomic_write_json, atomic_write_text

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "experiment" / "cvbra_v1_hazydet_source_retention.yaml"
PROTOCOL_SHA256 = "501d1e4605b295faafda78786cb6b807ae68f58c06ebe8352bad2daba7eb723c"
OUTPUT = (
    ROOT
    / "reports"
    / "development"
    / "cvbra_v1_matched_baselines"
    / "hazydet_source_retention"
)
REGISTRATION = OUTPUT / "REGISTRATION_LOCK.json"
REGISTRATION_SHA256 = "5a8ba5bec5d80b43292de9b422dc7b8ded98f96c0de396dffdb527488f4ba8a9"
MANIFEST = ROOT / "data" / "manifests" / "hazydet_val_manifest.json"
MANIFEST_SHA256 = "e8e5ea234dd098348a1906c8aff93d0d773d6ed1c828c09ee98d7cca6b60f916"
VALIDATION_AUDIT = ROOT / "data" / "manifests" / "hazydet_val_validation.json"
VALIDATION_AUDIT_SHA256 = "3b14a7f9181ea4820b7eca2837a7baaa52eb01d4a08055f8e1b2d96fd6aed83f"
ANNOTATION = ROOT / "data" / "raw" / "HazyDet" / "val" / "val_coco.json"
ANNOTATION_SHA256 = "2b2e39f7812631dfb4f3f0fbe1e743b65ca873151ca92d8e24ddbf0e9feacb9a"
IMAGE_ROOT = ROOT / "data" / "raw" / "HazyDet"

MODELS = ("source", "CVBRA_v1", "STF", "CVBRA_noCV", "CVBRA_noReplay", "CVBRA_noFreeze")
WEIGHTS: dict[str, tuple[Path, str]] = {
    "source": (
        ROOT / "weights" / "hazydet" / "yolo11n_best.pt",
        "16ad9de39a1ff86a1826eb5cda680166196ff33ecc5d3dcb297070a166e42430",
    ),
    "CVBRA_v1": (
        ROOT / "runs" / "cvbra_v1" / "yolo11n" / "cvbra_v1.pt",
        "d44f0926696e93b5f2e0ec5c9201f1e6360c43d4d40fccc68646e1ced633bd42",
    ),
    **{
        model: (
            ROOT / "runs" / "cvbra_v1_matched_baselines" / model / f"{model}.pt",
            "pending",
        )
        for model in MODELS[2:]
    },
}

IMPLEMENTATION_LOCK = OUTPUT / "implementation_lock.json"
IMPLEMENTATION_MARKER = OUTPUT / "IMPLEMENTATION_LOCKED"
PREDICTION_LOCK = OUTPUT / "prediction_lock.json"
PREDICTION_MARKER = OUTPUT / "PREDICTIONS_LOCKED"
ANALYSIS_LOCK = OUTPUT / "analysis_authorization.json"
ANALYSIS_MARKER = OUTPUT / "ANALYSIS_AUTHORIZED"
METRICS = OUTPUT / "metrics.csv"
STATISTICS = OUTPUT / "paired_statistics.json"
REPORT = OUTPUT / "source_retention_report.json"
COMPLETE = OUTPUT / "SOURCE_RETENTION_COMPLETE"

IMAGES = 1000
REPETITIONS = 2
IMAGE_SIZE = 1280
PROBE_CONF = 0.08
PUBLISH_CONF = 0.25
NMS_IOU = 0.70
MAX_DET = 500
WARMUP_IMAGES = 16
CHUNK_SIZE = 8
RESAMPLES = 2000
SEED = 20260814
CLASS_NAMES = ("car", "truck", "bus")
CATEGORY_ID_BY_CLASS = {0: 1, 1: 2, 2: 3}
METRIC_KEYS = ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large")


class SourceRetentionError(RuntimeError):
    """Raised when the frozen HazyDet source-retention audit cannot fail closed."""


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SourceRetentionError(f"cannot parse mapping {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SourceRetentionError(f"expected mapping: {path}")
    return value


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _rooted(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def _assert_hash(path: Path, expected: object, *, label: str) -> None:
    if not path.is_file() or sha256_file(path) != str(expected):
        raise SourceRetentionError(f"locked {label} changed: {path}")


def _baseline_checkpoint_lock(model: str) -> Path:
    return (
        ROOT
        / "reports"
        / "development"
        / "cvbra_v1_matched_baselines"
        / "checkpoint_locks"
        / f"{model}.json"
    )


def _validate_scope_and_weights() -> None:
    for path, digest, label in (
        (PROTOCOL, PROTOCOL_SHA256, "source-retention protocol"),
        (REGISTRATION, REGISTRATION_SHA256, "source-retention registration"),
        (MANIFEST, MANIFEST_SHA256, "HazyDet validation manifest"),
        (VALIDATION_AUDIT, VALIDATION_AUDIT_SHA256, "HazyDet validation audit"),
        (ANNOTATION, ANNOTATION_SHA256, "HazyDet validation annotation"),
    ):
        _assert_hash(path, digest, label=label)
    registration = _load_mapping(REGISTRATION)
    audit = _load_mapping(VALIDATION_AUDIT)
    if (
        registration.get("models") != list(MODELS)
        or registration.get("predictions_or_metrics_before_registration") is not False
        or registration.get("method_or_hyperparameter_selection") is not False
        or audit.get("valid") is not True
        or not isinstance(audit.get("stats"), dict)
        or audit["stats"].get("images") != IMAGES
    ):
        raise SourceRetentionError("source-retention scope changed")
    for model in MODELS[2:]:
        lock_path = _baseline_checkpoint_lock(model)
        marker_path = lock_path.with_suffix(".LOCKED")
        lock = _load_mapping(lock_path)
        marker = _load_mapping(marker_path)
        if (
            lock.get("status")
            != "CVBRA_V1_MATCHED_BASELINE_CHECKPOINT_VERIFIED_AND_LOCKED"
            or lock.get("baseline") != model
            or lock.get("validation_metric_used_for_training_or_selection") is not False
            or marker.get("checkpoint_lock_sha256") != sha256_file(lock_path)
        ):
            raise SourceRetentionError(f"baseline checkpoint lock changed: {model}")
        WEIGHTS[model] = (WEIGHTS[model][0], str(lock["checkpoint_sha256"]))
    for model, (path, digest) in WEIGHTS.items():
        _assert_hash(path, digest, label=f"{model} weight")


def _records(*, verify_hashes: bool) -> tuple[ImageRecord, ...]:
    manifest = _load_mapping(MANIFEST)
    all_files = manifest.get("files")
    files = (
        [
            row
            for row in all_files
            if isinstance(row, dict)
            and str(row.get("path", "")).replace("\\", "/").startswith(
                "val/hazy_images/"
            )
        ]
        if isinstance(all_files, list)
        else None
    )
    if (
        manifest.get("dataset") != "hazydet"
        or not isinstance(files, list)
        or len(files) != IMAGES
    ):
        raise SourceRetentionError("HazyDet validation manifest changed")
    records: list[ImageRecord] = []
    for row in files:
        if not isinstance(row, dict):
            raise SourceRetentionError("invalid HazyDet manifest row")
        path = IMAGE_ROOT / str(row["path"])
        if not path.is_file() or (verify_hashes and sha256_file(path) != str(row["sha256"])):
            raise SourceRetentionError(f"HazyDet validation image changed: {path}")
        try:
            image_id = int(path.stem)
            with Image.open(path) as image:
                width, height = image.size
        except (OSError, ValueError) as exc:
            raise SourceRetentionError(f"cannot inspect validation image: {path}") from exc
        records.append(
            ImageRecord(
                image_id=image_id,
                path=str(path.resolve()),
                width=int(width),
                height=int(height),
            )
        )
    records.sort(key=lambda record: int(record.image_id))
    if len({int(record.image_id) for record in records}) != IMAGES:
        raise SourceRetentionError("HazyDet validation image IDs are not unique")
    return tuple(records)


def _cuda_identity() -> dict[str, Any]:
    try:
        import torch
    except (ImportError, OSError) as exc:
        raise SourceRetentionError(f"cannot inspect CUDA: {exc}") from exc
    if not torch.cuda.is_available():
        raise SourceRetentionError("source-retention inference requires CUDA")
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
    _validate_scope_and_weights()
    if IMPLEMENTATION_LOCK.exists() or IMPLEMENTATION_MARKER.exists():
        if not IMPLEMENTATION_LOCK.is_file() or not IMPLEMENTATION_MARKER.is_file():
            raise SourceRetentionError("source-retention implementation lock is incomplete")
        lock = _load_mapping(IMPLEMENTATION_LOCK)
        marker = _load_mapping(IMPLEMENTATION_MARKER)
        if (
            lock.get("runner_sha256") != sha256_file(Path(__file__))
            or lock.get("protocol_sha256") != PROTOCOL_SHA256
            or lock.get("cuda_identity") != _cuda_identity()
            or marker.get("implementation_lock_sha256")
            != sha256_file(IMPLEMENTATION_LOCK)
        ):
            raise SourceRetentionError("source-retention implementation lock changed")
        return lock
    if any(path.exists() for path in (PREDICTION_LOCK, ANALYSIS_LOCK, METRICS, REPORT)):
        raise SourceRetentionError("later source-retention output appeared before lock")
    records = _records(verify_hashes=True)
    payload = {
        "schema_version": 1,
        "status": "CVBRA_V1_HAZYDET_SOURCE_RETENTION_IMPLEMENTATION_LOCKED",
        "locked_at_utc": primary_engine_time(),
        "protocol_sha256": PROTOCOL_SHA256,
        "registration_sha256": REGISTRATION_SHA256,
        "runner": _relative(Path(__file__)),
        "runner_sha256": sha256_file(Path(__file__)),
        "python": platform.python_version(),
        "host": platform.node(),
        "cuda_identity": _cuda_identity(),
        "models": list(MODELS),
        "weights": {
            model: {"path": _relative(path), "sha256": digest}
            for model, (path, digest) in WEIGHTS.items()
        },
        "images": len(records),
        "inference_repetitions": REPETITIONS,
        "ruff": "PASS",
        "strict_mypy": "PASS",
        "validation_labels_previously_accessed": True,
        "metric_or_selection_feedback_allowed": False,
        "HazyDet_test_access": "prohibited",
        "UAV_OBB_test_access": "prohibited",
        "paper_body_change_authorized": False,
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


def primary_engine_time() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _filter_batches(batches: Sequence[DetectionBatch]) -> tuple[DetectionBatch, ...]:
    return tuple(
        DetectionBatch(
            image_id=batch.image_id,
            boxes=tuple(box for box in batch.boxes if box.score >= PUBLISH_CONF),
            latency_ms=batch.latency_ms,
            meta={**batch.meta, "publish_conf": PUBLISH_CONF},
        )
        for batch in batches
    )


def _raw_root(model: str, repetition: int) -> Path:
    return OUTPUT / "raw" / model / f"rep_{repetition}"


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
    model: str,
    repetition: int,
    detector: UltralyticsDetector,
    records: Sequence[ImageRecord],
) -> dict[str, Any]:
    root = _raw_root(model, repetition)
    marker_path = root / "SUCCESS.json"
    prediction_path = root / "predictions.coco.json"
    timing_path = root / "timings.json"
    if marker_path.exists():
        marker = _load_mapping(marker_path)
        _assert_hash(prediction_path, marker["prediction_sha256"], label="raw prediction")
        _assert_hash(timing_path, marker["timing_sha256"], label="raw timing")
        return marker
    if root.exists():
        raise SourceRetentionError(f"partial raw inference requires audit: {root}")
    batches = detector.predict(
        records,
        imgsz=IMAGE_SIZE,
        conf=PROBE_CONF,
        iou=NMS_IOU,
        max_det=MAX_DET,
        fp16=True,
    )
    filtered = _filter_batches(batches)
    write_coco_predictions(
        prediction_path,
        filtered,
        category_id_by_class=CATEGORY_ID_BY_CLASS,
    )
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
        "status": "HAZYDET_SOURCE_RETENTION_RAW_CELL_COMPLETE",
        "model": model,
        "repetition": repetition,
        "images": len(filtered),
        "prediction": _relative(prediction_path),
        "prediction_sha256": sha256_file(prediction_path),
        "timing": _relative(timing_path),
        "timing_sha256": sha256_file(timing_path),
        "validation_labels_previously_accessed": True,
        "labels_read_for_prediction": False,
        "metric_or_selection_feedback_used": False,
        "HazyDet_test_access": "prohibited",
    }
    atomic_write_json(marker_path, marker)
    return marker


def _validate_prediction_lock() -> dict[str, Any]:
    _implementation_lock()
    if not PREDICTION_LOCK.is_file() or not PREDICTION_MARKER.is_file():
        raise SourceRetentionError("source-retention prediction lock is incomplete")
    lock = _load_mapping(PREDICTION_LOCK)
    marker = _load_mapping(PREDICTION_MARKER)
    artifacts = lock.get("artifacts")
    if (
        lock.get("status") != "ALL_HAZYDET_SOURCE_RETENTION_PREDICTIONS_LOCKED_BEFORE_METRICS"
        or lock.get("implementation_lock_sha256") != sha256_file(IMPLEMENTATION_LOCK)
        or marker.get("prediction_lock_sha256") != sha256_file(PREDICTION_LOCK)
        or not isinstance(artifacts, list)
        or len(artifacts) != len(MODELS)
        or lock.get("metric_or_selection_feedback_used") is not False
    ):
        raise SourceRetentionError("source-retention prediction lock changed")
    for row in artifacts:
        if not isinstance(row, dict):
            raise SourceRetentionError("invalid prediction artifact")
        _assert_hash(_rooted(row["prediction"]), row["prediction_sha256"], label="prediction")
    return lock


def infer() -> dict[str, Any]:
    _implementation_lock()
    if PREDICTION_LOCK.exists():
        return _validate_prediction_lock()
    records = _records(verify_hashes=False)
    artifacts: list[dict[str, Any]] = []
    timing: dict[str, float] = {}
    for model in MODELS:
        detector = UltralyticsDetector(
            WEIGHTS[model][0],
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
        markers = [
            _predict(model, repetition, detector, records)
            for repetition in range(1, REPETITIONS + 1)
        ]
        del detector
        _release_detector()
        hashes = {str(marker["prediction_sha256"]) for marker in markers}
        if len(hashes) != 1:
            raise SourceRetentionError(f"prediction repetitions differ for {model}")
        means = [
            float(_load_mapping(_rooted(marker["timing"]))["mean_end_to_end_ms"])
            for marker in markers
        ]
        timing[model] = median(means)
        artifacts.append(
            {
                "model": model,
                "prediction": markers[0]["prediction"],
                "prediction_sha256": markers[0]["prediction_sha256"],
                "mean_ms": timing[model],
                "repetitions": REPETITIONS,
            }
        )
    payload = {
        "schema_version": 1,
        "status": "ALL_HAZYDET_SOURCE_RETENTION_PREDICTIONS_LOCKED_BEFORE_METRICS",
        "locked_at_utc": primary_engine_time(),
        "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
        "models": list(MODELS),
        "images": IMAGES,
        "artifacts": artifacts,
        "timing_median_mean_ms": timing,
        "prediction_determinism": "exact_sha256_match_across_two_repetitions",
        "validation_labels_previously_accessed": True,
        "labels_read_for_prediction": False,
        "metric_or_selection_feedback_used": False,
        "HazyDet_test_access": "prohibited",
        "UAV_OBB_test_access": "prohibited",
        "paper_body_change_authorized": False,
    }
    atomic_write_json(PREDICTION_LOCK, payload)
    atomic_write_json(
        PREDICTION_MARKER,
        {"status": payload["status"], "prediction_lock_sha256": sha256_file(PREDICTION_LOCK)},
    )
    return payload


def _analysis_authorization() -> dict[str, Any]:
    prediction = _validate_prediction_lock()
    if ANALYSIS_LOCK.exists() or ANALYSIS_MARKER.exists():
        if not ANALYSIS_LOCK.is_file() or not ANALYSIS_MARKER.is_file():
            raise SourceRetentionError("source-retention analysis lock is incomplete")
        lock = _load_mapping(ANALYSIS_LOCK)
        marker = _load_mapping(ANALYSIS_MARKER)
        if (
            lock.get("runner_sha256") != sha256_file(Path(__file__))
            or lock.get("prediction_lock_sha256") != sha256_file(PREDICTION_LOCK)
            or lock.get("annotation_sha256") != ANNOTATION_SHA256
            or marker.get("analysis_lock_sha256") != sha256_file(ANALYSIS_LOCK)
        ):
            raise SourceRetentionError("source-retention analysis lock changed")
        return lock
    if any(path.exists() for path in (METRICS, STATISTICS, REPORT)):
        raise SourceRetentionError("metric output appeared before source-retention analysis lock")
    payload = {
        "schema_version": 1,
        "status": "HAZYDET_SOURCE_RETENTION_ANALYSIS_AUTHORIZED_AFTER_PREDICTION_LOCK",
        "authorized_at_utc": primary_engine_time(),
        "protocol_sha256": PROTOCOL_SHA256,
        "runner": _relative(Path(__file__)),
        "runner_sha256": sha256_file(Path(__file__)),
        "prediction_lock_sha256": sha256_file(PREDICTION_LOCK),
        "annotation_sha256": ANNOTATION_SHA256,
        "validation_annotation_previously_accessed": True,
        "metric_accessed_before_authorization": False,
        "method_or_hyperparameter_selection": False,
        "HazyDet_test_access": "prohibited",
        "prediction_status": prediction["status"],
    }
    atomic_write_json(ANALYSIS_LOCK, payload)
    atomic_write_json(
        ANALYSIS_MARKER,
        {"status": payload["status"], "analysis_lock_sha256": sha256_file(ANALYSIS_LOCK)},
    )
    return payload


def _prediction_lookup(lock: Mapping[str, Any]) -> dict[str, Path]:
    artifacts = lock.get("artifacts")
    if not isinstance(artifacts, list):
        raise SourceRetentionError("prediction artifacts are invalid")
    return {
        str(row["model"]): _rooted(row["prediction"])
        for row in artifacts
        if isinstance(row, dict)
    }


def _write_metrics(rows: Sequence[Mapping[str, Any]]) -> None:
    fields = ("model", *METRIC_KEYS, "images_evaluated", "mean_ms", "checkpoint_size_bytes")
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row[key] for key in fields})
    atomic_write_text(METRICS, buffer.getvalue())


def _checkpoint_deltas(path: Path) -> list[float]:
    values = _load_mapping(path).get("deltas")
    if not isinstance(values, list) or len(values) != RESAMPLES:
        raise SourceRetentionError(f"bootstrap checkpoint is incomplete: {path}")
    return [float(value) for value in values]


def score() -> dict[str, Any]:
    prediction = infer()
    _analysis_authorization()
    if REPORT.exists():
        report = _load_mapping(REPORT)
        if not COMPLETE.is_file() or _load_mapping(COMPLETE).get(
            "report_sha256"
        ) != sha256_file(REPORT):
            raise SourceRetentionError("existing source-retention report is not locked")
        return report
    if any(path.exists() for path in (METRICS, STATISTICS, COMPLETE)):
        raise SourceRetentionError("partial source-retention metrics require audit")
    predictions = _prediction_lookup(prediction)
    timing = prediction.get("timing_median_mean_ms")
    if not isinstance(timing, dict) or set(predictions) != set(MODELS):
        raise SourceRetentionError("source-retention prediction coverage changed")
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        metrics = evaluate_coco(ANNOTATION, predictions[model], max_det=MAX_DET)
        rows.append(
            {
                "model": model,
                **{key: float(metrics[key]) for key in METRIC_KEYS},
                "images_evaluated": int(metrics["images_evaluated"]),
                "mean_ms": float(timing[model]),
                "checkpoint_size_bytes": WEIGHTS[model][0].stat().st_size,
            }
        )
    _write_metrics(rows)
    by_model = {str(row["model"]): row for row in rows}
    point_deltas = {
        "CVBRA_v1_minus_source": float(by_model["CVBRA_v1"]["AP"])
        - float(by_model["source"]["AP"]),
        **{
            f"CVBRA_v1_minus_{model}": float(by_model["CVBRA_v1"]["AP"])
            - float(by_model[model]["AP"])
            for model in MODELS[2:]
        },
    }
    comparisons = (
        ("source", "CVBRA_v1"),
        *((model, "CVBRA_v1") for model in MODELS[2:]),
    )
    statistics_rows: list[dict[str, Any]] = []
    for baseline, method in comparisons:
        name = f"{method}_minus_{baseline}"
        checkpoint = OUTPUT / "bootstrap" / f"{name}.json"
        result = paired_coco_ap_bootstrap(
            ANNOTATION,
            predictions[baseline],
            predictions[method],
            resamples=RESAMPLES,
            seed=SEED,
            max_det=MAX_DET,
            workers=4,
            checkpoint_path=checkpoint,
            checkpoint_identity={
                "study": "cvbra_v1_hazydet_source_retention",
                "comparison": name,
                "prediction_lock_sha256": sha256_file(PREDICTION_LOCK),
                "descriptive_post_freeze": True,
            },
            chunk_resamples=50,
        )
        expected = point_deltas[name]
        if abs(float(result["delta"]) - expected) > 1e-10:
            raise SourceRetentionError(f"bootstrap point estimate drifted: {name}")
        deltas = _checkpoint_deltas(checkpoint)
        standard_deviation = stdev(deltas)
        if standard_deviation <= 0.0 or not math.isfinite(standard_deviation):
            raise SourceRetentionError(f"bootstrap variance is invalid: {name}")
        statistics_rows.append(
            {
                "comparison": name,
                **result,
                "bootstrap_mean_delta": fmean(deltas),
                "bootstrap_standard_deviation": standard_deviation,
                "standardized_effect": expected / standard_deviation,
                "p_two_sided": bootstrap_sign_pvalue(deltas),
                "checkpoint": _relative(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
            }
        )
        print(json.dumps({"bootstrap_complete": name}), flush=True)
    adjusted = holm_adjust([float(row["p_two_sided"]) for row in statistics_rows])
    for row, value in zip(statistics_rows, adjusted, strict=True):
        row["holm_adjusted_p"] = value
        row["holm_significant_0p05"] = value < 0.05
    statistics = {
        "schema_version": 1,
        "status": "HAZYDET_SOURCE_RETENTION_PAIRED_STATISTICS_COMPLETE",
        "completed_at_utc": primary_engine_time(),
        "resamples": RESAMPLES,
        "seed": SEED,
        "unit": "image",
        "multiple_testing": "Holm across five descriptive AP contrasts",
        "rows": statistics_rows,
    }
    atomic_write_json(STATISTICS, statistics)
    report = {
        "schema_version": 1,
        "status": "COMPLETE_CVBRA_V1_HAZYDET_SOURCE_RETENTION_AUDIT",
        "completed_at_utc": primary_engine_time(),
        "protocol_sha256": PROTOCOL_SHA256,
        "runner": _relative(Path(__file__)),
        "runner_sha256": sha256_file(Path(__file__)),
        "prediction_lock_sha256": sha256_file(PREDICTION_LOCK),
        "analysis_lock_sha256": sha256_file(ANALYSIS_LOCK),
        "metrics": _relative(METRICS),
        "metrics_sha256": sha256_file(METRICS),
        "statistics": _relative(STATISTICS),
        "statistics_sha256": sha256_file(STATISTICS),
        "rows": rows,
        "AP_deltas": point_deltas,
        "paired_statistics": statistics_rows,
        "evidence_boundary": {
            "scope": "spent full 1000-image HazyDet validation; descriptive source retention",
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


def preflight() -> dict[str, Any]:
    lock = _implementation_lock()
    return {
        "status": "PASS_CVBRA_V1_HAZYDET_SOURCE_RETENTION_PREFLIGHT",
        "models": list(MODELS),
        "images": IMAGES,
        "repetitions": REPETITIONS,
        "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
        "runner_sha256": lock["runner_sha256"],
        "validation_labels_previously_accessed": True,
        "method_or_hyperparameter_selection": False,
        "HazyDet_test_access": "prohibited",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CVBRA-v1 HazyDet source-retention audit")
    parser.add_argument("--stage", choices=("preflight", "infer", "score"), default="score")
    args = parser.parse_args()
    if args.stage == "preflight":
        result = preflight()
        path = IMPLEMENTATION_LOCK
    elif args.stage == "infer":
        result = infer()
        path = PREDICTION_LOCK
    else:
        result = score()
        path = REPORT
    print(
        json.dumps(
            {"status": result["status"], "artifact_sha256": sha256_file(path)},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
