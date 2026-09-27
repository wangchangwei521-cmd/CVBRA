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
from buse_uav.evaluation.bootstrap import (
    ClusterBootstrapScope,
    paired_coco_ap_cluster_bootstrap_scopes,
)
from buse_uav.evaluation.coco import evaluate_coco, write_coco_predictions
from buse_uav.evaluation.supplementary import bootstrap_sign_pvalue, holm_adjust
from buse_uav.schemas import DetectionBatch, ImageRecord
from buse_uav.utils.hashing import sha256_file
from buse_uav.utils.io import atomic_write_json, atomic_write_text

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/experiment/cvbra_v1_acceptance_upgrade_v1.yaml"
PROTOCOL_SHA256 = "2b0dd70e4c67c2afcca2dafea3a4d5e885ce233e45681db691ce9422ca78e73b"
REGISTRATION = (
    ROOT / "reports/development/cvbra_v1_acceptance_upgrade/PlainMix/REGISTRATION_LOCK.json"
)
REGISTRATION_SHA256 = "9bd0e3d383d23a8bfb14d18c4298fca8e445d82ae5a7a749acfe22405886fc85"
CHECKPOINT_LOCK = REGISTRATION.parent / "checkpoint_lock.json"
CHECKPOINT_MARKER = REGISTRATION.parent / "CHECKPOINT_LOCKED"
CHECKPOINT = ROOT / "runs/cvbra_v1_acceptance_upgrade/PlainMix/PlainMix.pt"

MANIFEST = ROOT / "data/manifests/hazydet_val_manifest.json"
MANIFEST_SHA256 = "e8e5ea234dd098348a1906c8aff93d0d773d6ed1c828c09ee98d7cca6b60f916"
VALIDATION_AUDIT = ROOT / "data/manifests/hazydet_val_validation.json"
VALIDATION_AUDIT_SHA256 = "3b14a7f9181ea4820b7eca2837a7baaa52eb01d4a08055f8e1b2d96fd6aed83f"
ANNOTATION = ROOT / "data/raw/HazyDet/val/val_coco.json"
ANNOTATION_SHA256 = "2b2e39f7812631dfb4f3f0fbe1e743b65ca873151ca92d8e24ddbf0e9feacb9a"
IMAGE_ROOT = ROOT / "data/raw/HazyDet"
REFERENCE_PREDICTION_LOCK = (
    ROOT
    / "reports/development/cvbra_v1_matched_baselines"
    / "hazydet_source_retention_v2/prediction_lock.json"
)
REFERENCE_PREDICTION_LOCK_SHA256 = (
    "8733301dc19faa2b65acd668200fd2842a6223a785e6879ea50a9d209e5637e6"
)
REFERENCE_PREDICTION_MARKER = REFERENCE_PREDICTION_LOCK.parent / "PREDICTIONS_LOCKED"

OUTPUT = REGISTRATION.parent / "hazydet_validation"
IMPLEMENTATION_LOCK = OUTPUT / "implementation_lock.json"
IMPLEMENTATION_MARKER = OUTPUT / "IMPLEMENTATION_LOCKED"
PREDICTION_LOCK = OUTPUT / "prediction_lock.json"
PREDICTION_MARKER = OUTPUT / "PREDICTIONS_LOCKED"
ANALYSIS_LOCK = OUTPUT / "analysis_authorization.json"
ANALYSIS_MARKER = OUTPUT / "ANALYSIS_AUTHORIZED"
METRICS = OUTPUT / "metrics.csv"
STATISTICS = OUTPUT / "paired_statistics.json"
REPORT = OUTPUT / "validation_report.json"
COMPLETE = OUTPUT / "VALIDATION_COMPLETE"

REFERENCE_MODELS = ("source", "CVBRA_v1", "CVBRA_noCV")
MODELS = (*REFERENCE_MODELS, "PlainMix")
COMPARISONS = (
    ("CVBRA_noCV", "PlainMix", "PlainMix_minus_CVBRA_noCV"),
    ("PlainMix", "CVBRA_v1", "CVBRA_v1_minus_PlainMix"),
)
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
SEED = 20260821
CLASS_NAMES = ("car", "truck", "bus")
CATEGORY_ID_BY_CLASS = {0: 0, 1: 1, 2: 2}
METRIC_KEYS = ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large")
HISTORICAL_SOURCE_AP = 0.5169972340948197
SOURCE_AP_MAX_ABS_DRIFT = 0.0025


class PlainMixHazyDetError(RuntimeError):
    """Raised when the registered PlainMix HazyDet validation cannot fail closed."""


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
        raise PlainMixHazyDetError(f"cannot parse mapping {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PlainMixHazyDetError(f"expected mapping: {path}")
    return value


def _load_rows(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PlainMixHazyDetError(f"cannot parse rows {path}: {exc}") from exc
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise PlainMixHazyDetError(f"expected row list: {path}")
    return value


def _assert_hash(path: Path, expected: object, *, label: str) -> None:
    if not path.is_file() or sha256_file(path) != str(expected):
        raise PlainMixHazyDetError(f"locked {label} changed: {path}")


def _validate_checkpoint() -> tuple[str, str]:
    for path, digest, label in (
        (PROTOCOL, PROTOCOL_SHA256, "acceptance-upgrade protocol"),
        (REGISTRATION, REGISTRATION_SHA256, "PlainMix registration"),
        (MANIFEST, MANIFEST_SHA256, "HazyDet validation manifest"),
        (VALIDATION_AUDIT, VALIDATION_AUDIT_SHA256, "HazyDet validation audit"),
        (ANNOTATION, ANNOTATION_SHA256, "HazyDet validation annotation"),
    ):
        _assert_hash(path, digest, label=label)
    registration = _load_mapping(REGISTRATION)
    checkpoint = _load_mapping(CHECKPOINT_LOCK)
    marker = _load_mapping(CHECKPOINT_MARKER)
    audit = _load_mapping(VALIDATION_AUDIT)
    if (
        registration.get("baseline") != "PlainMix"
        or checkpoint.get("status") != "PLAINMIX_CHECKPOINT_VERIFIED_AND_LOCKED"
        or checkpoint.get("validation_metric_used_for_training_or_selection") is not False
        or checkpoint.get("official_test_access") != "prohibited"
        or marker.get("checkpoint_lock_sha256") != sha256_file(CHECKPOINT_LOCK)
        or audit.get("valid") is not True
        or not isinstance(audit.get("stats"), dict)
        or audit["stats"].get("images") != IMAGES
    ):
        raise PlainMixHazyDetError("PlainMix checkpoint or HazyDet scope changed")
    digest = str(checkpoint["checkpoint_sha256"])
    _assert_hash(CHECKPOINT, digest, label="PlainMix checkpoint")
    return digest, sha256_file(CHECKPOINT_LOCK)


def _reference_prediction_lock() -> dict[str, Any]:
    _assert_hash(
        REFERENCE_PREDICTION_LOCK,
        REFERENCE_PREDICTION_LOCK_SHA256,
        label="reference HazyDet prediction lock",
    )
    lock = _load_mapping(REFERENCE_PREDICTION_LOCK)
    marker = _load_mapping(REFERENCE_PREDICTION_MARKER)
    artifacts = lock.get("artifacts")
    if (
        lock.get("status") != "ALL_CORRECTED_HAZYDET_SOURCE_RETENTION_PREDICTIONS_LOCKED"
        or marker.get("prediction_lock_sha256") != REFERENCE_PREDICTION_LOCK_SHA256
        or not isinstance(artifacts, list)
        or lock.get("models")
        != ["source", "CVBRA_v1", "STF", "CVBRA_noCV", "CVBRA_noReplay", "CVBRA_noFreeze"]
    ):
        raise PlainMixHazyDetError("reference HazyDet prediction scope changed")
    observed: set[str] = set()
    for row in artifacts:
        if not isinstance(row, dict):
            raise PlainMixHazyDetError("invalid reference prediction artifact")
        model = str(row.get("model"))
        if model in REFERENCE_MODELS:
            prediction = _rooted(row["corrected_prediction"])
            _assert_hash(
                prediction,
                row["corrected_prediction_sha256"],
                label=f"{model} corrected prediction",
            )
            observed.add(model)
    if observed != set(REFERENCE_MODELS):
        raise PlainMixHazyDetError("reference HazyDet models are incomplete")
    return lock


def _records(*, verify_hashes: bool) -> tuple[ImageRecord, ...]:
    manifest = _load_mapping(MANIFEST)
    all_files = manifest.get("files")
    files = (
        [
            row
            for row in all_files
            if isinstance(row, dict)
            and str(row.get("path", "")).replace("\\", "/").startswith("val/hazy_images/")
        ]
        if isinstance(all_files, list)
        else None
    )
    if manifest.get("dataset") != "hazydet" or not isinstance(files, list) or len(files) != IMAGES:
        raise PlainMixHazyDetError("HazyDet manifest scope changed")
    records: list[ImageRecord] = []
    for row in files:
        if not isinstance(row, dict):
            raise PlainMixHazyDetError("invalid HazyDet manifest row")
        path = IMAGE_ROOT / str(row["path"])
        if not path.is_file() or (verify_hashes and sha256_file(path) != str(row["sha256"])):
            raise PlainMixHazyDetError(f"HazyDet image changed: {path}")
        try:
            image_id = int(path.stem)
            with Image.open(path) as image:
                width, height = image.size
        except (OSError, ValueError) as exc:
            raise PlainMixHazyDetError(f"cannot inspect HazyDet image: {path}") from exc
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
        raise PlainMixHazyDetError("HazyDet image IDs are not unique")
    return tuple(records)


def _ordered_annotation_image_ids() -> tuple[int, ...]:
    annotation = _load_mapping(ANNOTATION)
    images = annotation.get("images")
    categories = annotation.get("categories")
    if not isinstance(images, list) or not isinstance(categories, list):
        raise PlainMixHazyDetError("HazyDet annotation structure changed")
    image_ids = [
        int(row["id"])
        for row in images
        if isinstance(row, dict) and isinstance(row.get("id"), int)
    ]
    category_ids = [
        int(row["id"])
        for row in categories
        if isinstance(row, dict) and isinstance(row.get("id"), int)
    ]
    if len(image_ids) != IMAGES or len(set(image_ids)) != IMAGES or category_ids != [0, 1, 2]:
        raise PlainMixHazyDetError("HazyDet annotation IDs changed")
    return tuple(sorted(image_ids, key=str))


def _cuda_identity() -> dict[str, Any]:
    try:
        import torch
    except (ImportError, OSError) as exc:
        raise PlainMixHazyDetError(f"cannot inspect CUDA: {exc}") from exc
    if not torch.cuda.is_available():
        raise PlainMixHazyDetError("PlainMix HazyDet inference requires CUDA")
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
    _reference_prediction_lock()
    if IMPLEMENTATION_LOCK.exists() or IMPLEMENTATION_MARKER.exists():
        if not IMPLEMENTATION_LOCK.is_file() or not IMPLEMENTATION_MARKER.is_file():
            raise PlainMixHazyDetError("HazyDet implementation lock is incomplete")
        lock = _load_mapping(IMPLEMENTATION_LOCK)
        marker = _load_mapping(IMPLEMENTATION_MARKER)
        if (
            lock.get("runner_sha256") != sha256_file(Path(__file__))
            or lock.get("checkpoint_lock_sha256") != checkpoint_lock_sha256
            or lock.get("reference_prediction_lock_sha256")
            != REFERENCE_PREDICTION_LOCK_SHA256
            or lock.get("cuda_identity") != _cuda_identity()
            or marker.get("implementation_lock_sha256") != sha256_file(IMPLEMENTATION_LOCK)
        ):
            raise PlainMixHazyDetError("PlainMix HazyDet implementation changed")
        return lock
    if any(path.exists() for path in (PREDICTION_LOCK, ANALYSIS_LOCK, METRICS, REPORT)):
        raise PlainMixHazyDetError("HazyDet output appeared before implementation lock")
    records = _records(verify_hashes=True)
    payload = {
        "schema_version": 1,
        "status": "PLAINMIX_HAZYDET_VALIDATION_IMPLEMENTATION_LOCKED",
        "locked_at_utc": _now(),
        "protocol_sha256": PROTOCOL_SHA256,
        "registration_sha256": REGISTRATION_SHA256,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_lock_sha256": checkpoint_lock_sha256,
        "reference_prediction_lock_sha256": REFERENCE_PREDICTION_LOCK_SHA256,
        "annotation_sha256": ANNOTATION_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "runner": _relative(Path(__file__)),
        "runner_sha256": sha256_file(Path(__file__)),
        "python": platform.python_version(),
        "host": platform.node(),
        "cuda_identity": _cuda_identity(),
        "images": len(records),
        "models": list(MODELS),
        "comparisons": [name for _, _, name in COMPARISONS],
        "resamples": RESAMPLES,
        "seed": SEED,
        "ruff": "PASS",
        "strict_mypy": "PASS",
        "validation_labels_previously_accessed": True,
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
        "status": "PASS_PLAINMIX_HAZYDET_VALIDATION_PREFLIGHT",
        "images": IMAGES,
        "models": list(MODELS),
        "comparisons": [name for _, _, name in COMPARISONS],
        "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
        "runner_sha256": lock["runner_sha256"],
        "metric_accessed": False,
        "HazyDet_test_access": "prohibited",
    }


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


def _raw_root(repetition: int) -> Path:
    return OUTPUT / "raw/PlainMix" / f"rep_{repetition}"


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
    records: Sequence[ImageRecord],
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
        raise PlainMixHazyDetError(f"partial PlainMix inference requires audit: {root}")
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
    rows = _load_rows(prediction_path)
    if not rows or {int(row["category_id"]) for row in rows} - {0, 1, 2}:
        raise PlainMixHazyDetError("PlainMix HazyDet category export changed")
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
        "status": "PLAINMIX_HAZYDET_RAW_CELL_COMPLETE",
        "model": "PlainMix",
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
        raise PlainMixHazyDetError("PlainMix HazyDet prediction lock is incomplete")
    lock = _load_mapping(PREDICTION_LOCK)
    marker = _load_mapping(PREDICTION_MARKER)
    if (
        lock.get("status") != "PLAINMIX_HAZYDET_PREDICTIONS_LOCKED_BEFORE_METRICS"
        or lock.get("implementation_lock_sha256") != sha256_file(IMPLEMENTATION_LOCK)
        or marker.get("prediction_lock_sha256") != sha256_file(PREDICTION_LOCK)
        or lock.get("metric_or_selection_feedback_used") is not False
    ):
        raise PlainMixHazyDetError("PlainMix HazyDet prediction lock changed")
    _assert_hash(
        _rooted(lock["prediction"]), lock["prediction_sha256"], label="PlainMix prediction"
    )
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
        raise PlainMixHazyDetError("PlainMix HazyDet repetitions differ")
    means = [
        float(_load_mapping(_rooted(marker["timing"]))["mean_end_to_end_ms"])
        for marker in markers
    ]
    payload = {
        "schema_version": 1,
        "status": "PLAINMIX_HAZYDET_PREDICTIONS_LOCKED_BEFORE_METRICS",
        "locked_at_utc": _now(),
        "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
        "model": "PlainMix",
        "images": IMAGES,
        "prediction": markers[0]["prediction"],
        "prediction_sha256": markers[0]["prediction_sha256"],
        "timing_median_mean_ms": median(means),
        "repetitions": REPETITIONS,
        "prediction_determinism": "exact_sha256_match_across_two_repetitions",
        "validation_labels_previously_accessed": True,
        "labels_read_for_prediction": False,
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


def _analysis_authorization() -> dict[str, Any]:
    prediction = _validate_prediction_lock()
    _reference_prediction_lock()
    if ANALYSIS_LOCK.exists() or ANALYSIS_MARKER.exists():
        if not ANALYSIS_LOCK.is_file() or not ANALYSIS_MARKER.is_file():
            raise PlainMixHazyDetError("HazyDet analysis authorization is incomplete")
        lock = _load_mapping(ANALYSIS_LOCK)
        marker = _load_mapping(ANALYSIS_MARKER)
        if (
            lock.get("runner_sha256") != sha256_file(Path(__file__))
            or lock.get("prediction_lock_sha256") != sha256_file(PREDICTION_LOCK)
            or lock.get("annotation_sha256") != ANNOTATION_SHA256
            or marker.get("analysis_lock_sha256") != sha256_file(ANALYSIS_LOCK)
        ):
            raise PlainMixHazyDetError("HazyDet analysis authorization changed")
        return lock
    if any(path.exists() for path in (METRICS, STATISTICS, REPORT)):
        raise PlainMixHazyDetError("HazyDet metric output appeared before analysis authorization")
    payload = {
        "schema_version": 1,
        "status": "PLAINMIX_HAZYDET_ANALYSIS_AUTHORIZED_AFTER_PREDICTION_LOCK",
        "authorized_at_utc": _now(),
        "runner": _relative(Path(__file__)),
        "runner_sha256": sha256_file(Path(__file__)),
        "prediction_lock_sha256": sha256_file(PREDICTION_LOCK),
        "reference_prediction_lock_sha256": REFERENCE_PREDICTION_LOCK_SHA256,
        "annotation_sha256": ANNOTATION_SHA256,
        "metric_accessed_before_authorization": False,
        "method_or_hyperparameter_selection": False,
        "prediction_status": prediction["status"],
        "HazyDet_test_access": "prohibited",
    }
    atomic_write_json(ANALYSIS_LOCK, payload)
    atomic_write_json(
        ANALYSIS_MARKER,
        {"status": payload["status"], "analysis_lock_sha256": sha256_file(ANALYSIS_LOCK)},
    )
    return payload


def _prediction_paths(
    reference: Mapping[str, Any], plainmix: Mapping[str, Any]
) -> dict[str, Path]:
    artifacts = reference.get("artifacts")
    if not isinstance(artifacts, list):
        raise PlainMixHazyDetError("reference prediction artifacts changed")
    paths = {
        str(row["model"]): _rooted(row["corrected_prediction"])
        for row in artifacts
        if isinstance(row, dict) and str(row.get("model")) in REFERENCE_MODELS
    }
    paths["PlainMix"] = _rooted(plainmix["prediction"])
    if set(paths) != set(MODELS):
        raise PlainMixHazyDetError("HazyDet prediction path coverage changed")
    return paths


def _metrics_text(rows: Sequence[Mapping[str, Any]]) -> str:
    fields = ("model", *METRIC_KEYS, "images_evaluated", "mean_ms", "checkpoint_size_bytes")
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row[key] for key in fields})
    return buffer.getvalue()


def _checkpoint_deltas(path: Path) -> list[float]:
    values = _load_mapping(path).get("deltas")
    if not isinstance(values, list) or len(values) != RESAMPLES:
        raise PlainMixHazyDetError(f"bootstrap checkpoint is incomplete: {path}")
    return [float(value) for value in values]


def _singleton_clusters(image_ids: Sequence[int]) -> dict[str, tuple[int]]:
    if len(image_ids) != IMAGES or len(set(image_ids)) != IMAGES:
        raise PlainMixHazyDetError("HazyDet bootstrap image scope changed")
    return {f"image_{index:04d}": (image_id,) for index, image_id in enumerate(image_ids)}


def score() -> dict[str, Any]:
    plainmix = infer()
    reference = _reference_prediction_lock()
    _analysis_authorization()
    if REPORT.exists():
        report = _load_mapping(REPORT)
        if not COMPLETE.is_file() or _load_mapping(COMPLETE).get("report_sha256") != sha256_file(
            REPORT
        ):
            raise PlainMixHazyDetError("existing PlainMix HazyDet report is not locked")
        return report
    if any(path.exists() for path in (METRICS, STATISTICS, COMPLETE)):
        raise PlainMixHazyDetError("partial PlainMix HazyDet metrics require audit")
    paths = _prediction_paths(reference, plainmix)
    reference_timing = reference.get("timing_median_mean_ms")
    if not isinstance(reference_timing, dict):
        raise PlainMixHazyDetError("reference HazyDet timing changed")
    timing = {model: float(reference_timing[model]) for model in REFERENCE_MODELS}
    timing["PlainMix"] = float(plainmix["timing_median_mean_ms"])
    checkpoint_sizes = {
        "source": (ROOT / "weights/hazydet/yolo11n_best.pt").stat().st_size,
        "CVBRA_v1": (ROOT / "runs/cvbra_v1/yolo11n/cvbra_v1.pt").stat().st_size,
        "CVBRA_noCV": (
            ROOT / "runs/cvbra_v1_matched_baselines/CVBRA_noCV/CVBRA_noCV.pt"
        ).stat().st_size,
        "PlainMix": CHECKPOINT.stat().st_size,
    }
    image_ids = _ordered_annotation_image_ids()
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        result = evaluate_coco(ANNOTATION, paths[model], max_det=MAX_DET, image_ids=image_ids)
        if int(result["images_evaluated"]) != IMAGES:
            raise PlainMixHazyDetError(f"{model} did not evaluate all HazyDet images")
        rows.append(
            {
                "model": model,
                **{key: float(result[key]) for key in METRIC_KEYS},
                "images_evaluated": int(result["images_evaluated"]),
                "mean_ms": timing[model],
                "checkpoint_size_bytes": checkpoint_sizes[model],
            }
        )
    atomic_write_text(METRICS, _metrics_text(rows))
    by_model = {str(row["model"]): row for row in rows}
    source_drift = float(by_model["source"]["AP"]) - HISTORICAL_SOURCE_AP
    if abs(source_drift) > SOURCE_AP_MAX_ABS_DRIFT:
        raise PlainMixHazyDetError(f"HazyDet source anchor failed: {source_drift:+.9f}")
    point_deltas = {
        name: float(by_model[method]["AP"]) - float(by_model[baseline]["AP"])
        for baseline, method, name in COMPARISONS
    }
    point_deltas["PlainMix_minus_source"] = float(by_model["PlainMix"]["AP"]) - float(
        by_model["source"]["AP"]
    )
    clusters = _singleton_clusters(image_ids)
    statistics_rows: list[dict[str, Any]] = []
    for baseline, method, name in COMPARISONS:
        checkpoint = OUTPUT / "bootstrap" / f"{name}.json"
        raw_result = paired_coco_ap_cluster_bootstrap_scopes(
            ANNOTATION,
            paths[baseline],
            paths[method],
            {
                "all_images": ClusterBootstrapScope(
                    clusters=clusters,
                    checkpoint_path=checkpoint,
                    checkpoint_identity={
                        "study": "cvbra_v1_plainmix_hazydet_validation",
                        "comparison": name,
                        "prediction_lock_sha256": sha256_file(PREDICTION_LOCK),
                        "reference_prediction_lock_sha256": REFERENCE_PREDICTION_LOCK_SHA256,
                        "recenter_rule": "raw_delta_plus_direct_minus_raw_observed",
                    },
                )
            },
            resamples=RESAMPLES,
            seed=SEED,
            max_det=MAX_DET,
            workers=4,
            chunk_resamples=100,
            accelerate_ap_only=True,
        )["all_images"]
        direct_delta = point_deltas[name]
        raw_observed = float(raw_result["delta"])
        offset = direct_delta - raw_observed
        raw_deltas = _checkpoint_deltas(checkpoint)
        adjusted_deltas = [value + offset for value in raw_deltas]
        if not math.isclose(raw_observed + offset, direct_delta, rel_tol=0.0, abs_tol=1e-15):
            raise PlainMixHazyDetError(f"bootstrap recentering failed: {name}")
        standard_deviation = stdev(adjusted_deltas)
        if standard_deviation <= 0.0 or not math.isfinite(standard_deviation):
            raise PlainMixHazyDetError(f"bootstrap variance is invalid: {name}")
        statistics_rows.append(
            {
                "comparison": name,
                "direct_delta": direct_delta,
                "raw_observed_delta": raw_observed,
                "recenter_offset": offset,
                "ci_low": float(raw_result["ci_low"]) + offset,
                "ci_high": float(raw_result["ci_high"]) + offset,
                "resamples": RESAMPLES,
                "seed": SEED,
                "images": IMAGES,
                "bootstrap_mean_delta": fmean(adjusted_deltas),
                "bootstrap_standard_deviation": standard_deviation,
                "standardized_effect": direct_delta / standard_deviation,
                "p_two_sided": bootstrap_sign_pvalue(adjusted_deltas),
                "checkpoint": _relative(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
            }
        )
    adjusted_p = holm_adjust([float(row["p_two_sided"]) for row in statistics_rows])
    for row, value in zip(statistics_rows, adjusted_p, strict=True):
        row["holm_adjusted_p"] = value
        row["holm_significant_0p05"] = value < 0.05
    statistics = {
        "schema_version": 1,
        "status": "PLAINMIX_HAZYDET_PAIRED_STATISTICS_COMPLETE",
        "completed_at_utc": _now(),
        "resamples": RESAMPLES,
        "seed": SEED,
        "unit": "image_via_singleton_cluster_encoding",
        "recenter_rule": "raw_delta_plus_direct_minus_raw_observed",
        "multiple_testing": "Holm across two registered AP contrasts",
        "rows": statistics_rows,
    }
    atomic_write_json(STATISTICS, statistics)
    report = {
        "schema_version": 1,
        "status": "COMPLETE_PLAINMIX_HAZYDET_VALIDATION",
        "completed_at_utc": _now(),
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
        "source_anchor": {
            "historical_AP": HISTORICAL_SOURCE_AP,
            "observed_AP": float(by_model["source"]["AP"]),
            "drift": source_drift,
            "maximum_absolute_drift": SOURCE_AP_MAX_ABS_DRIFT,
            "pass": True,
        },
        "evidence_boundary": {
            "scope": "spent full 1000-image HazyDet validation",
            "validation_labels_previously_accessed": True,
            "configuration_frozen_before_prediction": True,
            "method_or_hyperparameter_selection": False,
            "independent_confirmation_claim": False,
            "HazyDet_test_access": "prohibited",
            "UAV_OBB_test_access": "prohibited",
        },
        "paper_result_integration_authorized": True,
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
    parser = argparse.ArgumentParser(description="Run registered PlainMix HazyDet validation")
    parser.add_argument("--stage", choices=("preflight", "infer", "score"), default="score")
    args = parser.parse_args()
    if args.stage == "preflight":
        result = preflight()
        artifact = IMPLEMENTATION_LOCK
    elif args.stage == "infer":
        result = infer()
        artifact = PREDICTION_LOCK
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
