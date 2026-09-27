from __future__ import annotations

import argparse
import gc
import json
import platform
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, median
from time import perf_counter
from typing import Any

from buse_uav.detectors.tta import merge_classwise_nms, restore_horizontal_flip
from buse_uav.detectors.ultralytics_adapter import UltralyticsDetector
from buse_uav.evaluation.coco import write_coco_predictions
from buse_uav.pipeline.multifidelity_flip import horizontal_flip_records
from buse_uav.schemas import Box, DetectionBatch, ImageRecord
from buse_uav.utils.hashing import sha256_file
from buse_uav.utils.io import atomic_write_json, atomic_write_text

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "experiment" / "cvbra_v1_backup_protocol.yaml"
PROTOCOL_SHA256 = "65c26c8e96284af4f58c3536d4ef81f5531ccc9c0455d2286cbd84d228eaf6da"
CHECKPOINT_LOCK = ROOT / "reports" / "development" / "cvbra_v1" / "checkpoint_lock.json"
CHECKPOINT_MARKER = ROOT / "reports" / "development" / "cvbra_v1" / "CHECKPOINT_LOCKED"
MATERIALIZATION_LOCK = (
    ROOT
    / "reports"
    / "development"
    / "cvbra_v1"
    / "uav_obb_reserve_B"
    / "view_materialization_lock.json"
)
MATERIALIZATION_MARKER = MATERIALIZATION_LOCK.parent / "VIEWS_MATERIALIZED"
OUTPUT = MATERIALIZATION_LOCK.parent / "evaluation"
IMPLEMENTATION_LOCK = OUTPUT / "implementation_lock.json"
IMPLEMENTATION_MARKER = OUTPUT / "IMPLEMENTATION_LOCKED"
RAW_LOCK = OUTPUT / "raw_observation_lock.json"
RAW_MARKER = OUTPUT / "RAW_OBSERVATIONS_LOCKED"
PREDICTION_LOCK = OUTPUT / "joint_prediction_lock.json"
PREDICTION_MARKER = OUTPUT / "PREDICTIONS_LOCKED"

SOURCE_WEIGHT = ROOT / "weights" / "hazydet" / "yolo11n_best.pt"
SOURCE_WEIGHT_SHA256 = "16ad9de39a1ff86a1826eb5cda680166196ff33ecc5d3dcb297070a166e42430"
MODELS = ("source", "CVBRA_v1")
VIEWS = ("original", "fog_0p6", "fog_1p0")
ORIENTATIONS = ("identity", "flip")
METHODS = ("Identity", "Flip_HardNMS")
CLASS_NAMES = ("car", "truck", "bus")
CATEGORY_ID_BY_CLASS = {0: 1, 1: 2, 2: 3}
CLASS_ID_BY_CATEGORY = {value: key for key, value in CATEGORY_ID_BY_CLASS.items()}
IMAGES = 483
IMAGE_SIZE = 1280
PROBE_CONF = 0.08
PUBLISH_CONF = 0.25
NMS_IOU = 0.70
MAX_DET = 500
REPETITIONS = 2
WARMUP_IMAGES = 16
CHUNK_SIZE = 8
POSTPROCESS_WARMUPS = 3
POSTPROCESS_REPETITIONS = 10


class CVBRAReserveInferenceError(RuntimeError):
    """Raised when label-blind CVBRA reserve-B inference fails closed."""


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
        raise CVBRAReserveInferenceError(f"cannot parse mapping {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CVBRAReserveInferenceError(f"expected mapping: {path}")
    return value


def _load_rows(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CVBRAReserveInferenceError(f"cannot parse rows {path}: {exc}") from exc
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise CVBRAReserveInferenceError(f"expected row list: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CVBRAReserveInferenceError(f"cannot parse JSONL {path}: {exc}") from exc
    if not all(isinstance(row, dict) for row in rows):
        raise CVBRAReserveInferenceError(f"expected JSONL objects: {path}")
    return rows


def _assert_hash(path: Path, expected: object, *, label: str) -> None:
    if not path.is_file():
        raise CVBRAReserveInferenceError(f"missing locked {label}: {path}")
    observed = sha256_file(path)
    if observed != str(expected):
        raise CVBRAReserveInferenceError(
            f"locked {label} changed: expected {expected}, observed {observed}"
        )


def _cuda_identity() -> dict[str, Any]:
    try:
        import torch
    except (ImportError, OSError) as exc:
        raise CVBRAReserveInferenceError(f"cannot inspect CUDA: {exc}") from exc
    if not torch.cuda.is_available():
        raise CVBRAReserveInferenceError("CVBRA reserve-B inference requires CUDA")
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


def _validate_protocol() -> None:
    _assert_hash(PROTOCOL, PROTOCOL_SHA256, label="CVBRA protocol")
    import yaml

    try:
        protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise CVBRAReserveInferenceError(f"cannot parse CVBRA protocol: {exc}") from exc
    if not isinstance(protocol, dict):
        raise CVBRAReserveInferenceError("CVBRA protocol is not a mapping")
    execution = protocol.get("reserve_B_execution")
    selection = protocol.get("selection_rule")
    integrity = protocol.get("integrity")
    if not all(isinstance(value, dict) for value in (execution, selection, integrity)):
        raise CVBRAReserveInferenceError("CVBRA reserve-B protocol sections are incomplete")
    assert isinstance(execution, dict)
    assert isinstance(selection, dict)
    assert isinstance(integrity, dict)
    common = execution.get("same_for_all_cells")
    if not isinstance(common, dict):
        raise CVBRAReserveInferenceError("CVBRA matched-control settings are missing")
    if (
        tuple(execution.get("views", ())) != VIEWS
        or tuple(execution.get("models", ())) != MODELS
        or tuple(execution.get("methods", ())) != METHODS
        or int(execution.get("images", -1)) != IMAGES
        or int(execution.get("inference_repetitions", -1)) != REPETITIONS
        or execution.get("identical_prediction_sha256_required_across_repetitions") is not True
        or int(common.get("imgsz", -1)) != IMAGE_SIZE
        or float(common.get("probe_conf", -1.0)) != PROBE_CONF
        or float(common.get("publish_conf", -1.0)) != PUBLISH_CONF
        or float(common.get("nms_iou", -1.0)) != NMS_IOU
        or int(common.get("max_det", -1)) != MAX_DET
        or common.get("fp16") is not True
        or selection.get("single_candidate") != "CVBRA_v1"
        or selection.get("all_checks_required") is not True
        or integrity.get("reserve_B_labels_accessed_before_protocol") is not False
        or integrity.get("UAV_OBB_official_test_access") != "prohibited"
        or integrity.get("paper_body_change_authorized") is not False
    ):
        raise CVBRAReserveInferenceError("registered CVBRA reserve-B fields changed")


def _validate_checkpoint() -> tuple[dict[str, Any], dict[str, tuple[Path, str]]]:
    if not CHECKPOINT_LOCK.is_file() or not CHECKPOINT_MARKER.is_file():
        raise CVBRAReserveInferenceError("CVBRA checkpoint is not yet locked")
    lock = _load_mapping(CHECKPOINT_LOCK)
    marker = _load_mapping(CHECKPOINT_MARKER)
    if (
        lock.get("status") != "CVBRA_V1_CHECKPOINT_VERIFIED_AND_LOCKED_BEFORE_RESERVE_B_PREDICTION"
        or marker.get("checkpoint_lock_sha256") != sha256_file(CHECKPOINT_LOCK)
        or lock.get("reserve_B_label_or_prediction_accessed") is not False
        or lock.get("official_validation_or_test_accessed") is not False
    ):
        raise CVBRAReserveInferenceError("CVBRA checkpoint evidence changed")
    _assert_hash(SOURCE_WEIGHT, SOURCE_WEIGHT_SHA256, label="source weight")
    candidate = _rooted(lock["checkpoint"])
    _assert_hash(candidate, lock["checkpoint_sha256"], label="CVBRA-v1 weight")
    return lock, {
        "source": (SOURCE_WEIGHT, SOURCE_WEIGHT_SHA256),
        "CVBRA_v1": (candidate, str(lock["checkpoint_sha256"])),
    }


def _validate_materialization() -> dict[str, Any]:
    if not MATERIALIZATION_LOCK.is_file() or not MATERIALIZATION_MARKER.is_file():
        raise CVBRAReserveInferenceError("CVBRA reserve-B views are not yet locked")
    lock = _load_mapping(MATERIALIZATION_LOCK)
    marker = _load_mapping(MATERIALIZATION_MARKER)
    if (
        lock.get("status") != "CVBRA_UAV_OBB_RESERVE_B_VIEWS_MATERIALIZED_AND_LOCKED"
        or marker.get("view_materialization_lock_sha256") != sha256_file(MATERIALIZATION_LOCK)
        or lock.get("images") != IMAGES
        or lock.get("fog_outputs") != IMAGES * 2
        or lock.get("pass") is not True
        or lock.get("annotation_content_accessed") is not False
        or lock.get("prediction_or_metric_accessed") is not False
        or lock.get("official_validation_or_test_content_accessed") is not False
    ):
        raise CVBRAReserveInferenceError("CVBRA reserve-B materialization evidence changed")
    return lock


def _view_records(
    materialization: Mapping[str, Any], *, verify_images: bool
) -> dict[str, tuple[ImageRecord, ...]]:
    clean = materialization.get("clean_rows")
    fog = materialization.get("fog_rows")
    if not isinstance(clean, list) or len(clean) != IMAGES:
        raise CVBRAReserveInferenceError("reserve-B clean registry changed")
    if not isinstance(fog, list) or len(fog) != IMAGES * 2:
        raise CVBRAReserveInferenceError("reserve-B fog registry changed")
    registry: dict[str, list[dict[str, Any]]] = {view: [] for view in VIEWS}
    for row in clean:
        if not isinstance(row, dict):
            raise CVBRAReserveInferenceError("invalid reserve-B clean row")
        registry["original"].append(row)
    for row in fog:
        if not isinstance(row, dict) or str(row.get("view")) not in VIEWS[1:]:
            raise CVBRAReserveInferenceError("invalid reserve-B fog row")
        registry[str(row["view"])].append(row)
    result: dict[str, tuple[ImageRecord, ...]] = {}
    for view in VIEWS:
        rows = sorted(registry[view], key=lambda row: int(row["image_id"]))
        records: list[ImageRecord] = []
        for row in rows:
            path = _rooted(row["path"])
            if not path.is_file():
                raise CVBRAReserveInferenceError(f"reserve-B image is missing: {path}")
            if verify_images and sha256_file(path) != str(row["sha256"]):
                raise CVBRAReserveInferenceError(f"reserve-B image changed: {path}")
            records.append(
                ImageRecord(
                    image_id=int(row["image_id"]),
                    path=str(path),
                    width=int(row["width"]),
                    height=int(row["height"]),
                )
            )
        if len(records) != IMAGES or len({row.image_id for row in records}) != IMAGES:
            raise CVBRAReserveInferenceError(f"reserve-B view coverage changed: {view}")
        result[view] = tuple(records)
    reference = [(row.image_id, row.width, row.height) for row in result["original"]]
    if any(
        [(row.image_id, row.width, row.height) for row in result[view]] != reference
        for view in VIEWS[1:]
    ):
        raise CVBRAReserveInferenceError("reserve-B view identity or geometry changed")
    return result


def _implementation_lock() -> dict[str, Any]:
    checkpoint, weights = _validate_checkpoint()
    materialization = _validate_materialization()
    if IMPLEMENTATION_LOCK.exists() or IMPLEMENTATION_MARKER.exists():
        if not IMPLEMENTATION_LOCK.is_file() or not IMPLEMENTATION_MARKER.is_file():
            raise CVBRAReserveInferenceError("reserve-B inference lock is incomplete")
        lock = _load_mapping(IMPLEMENTATION_LOCK)
        marker = _load_mapping(IMPLEMENTATION_MARKER)
        if (
            lock.get("runner_sha256") != sha256_file(Path(__file__))
            or lock.get("checkpoint_lock_sha256") != sha256_file(CHECKPOINT_LOCK)
            or lock.get("materialization_lock_sha256") != sha256_file(MATERIALIZATION_LOCK)
            or lock.get("cuda_identity") != _cuda_identity()
            or marker.get("implementation_lock_sha256") != sha256_file(IMPLEMENTATION_LOCK)
        ):
            raise CVBRAReserveInferenceError("reserve-B inference implementation changed")
        return lock
    if any(path.exists() for path in (RAW_LOCK, PREDICTION_LOCK)):
        raise CVBRAReserveInferenceError("reserve-B inference appeared before implementation lock")
    payload = {
        "schema_version": 1,
        "status": "CVBRA_V1_UAV_OBB_RESERVE_B_INFERENCE_IMPLEMENTATION_LOCKED",
        "locked_at_utc": _utc_now(),
        "protocol_sha256": PROTOCOL_SHA256,
        "checkpoint_lock_sha256": sha256_file(CHECKPOINT_LOCK),
        "materialization_lock_sha256": sha256_file(MATERIALIZATION_LOCK),
        "runner": _relative(Path(__file__)),
        "runner_sha256": sha256_file(Path(__file__)),
        "python": platform.python_version(),
        "host": platform.node(),
        "cuda_identity": _cuda_identity(),
        "weights": {
            model: {"path": _relative(path), "sha256": digest}
            for model, (path, digest) in weights.items()
        },
        "models": list(MODELS),
        "views": list(VIEWS),
        "orientations": list(ORIENTATIONS),
        "methods": list(METHODS),
        "images": IMAGES,
        "ruff": "PASS",
        "strict_mypy": "PASS",
        "checkpoint_status": checkpoint["status"],
        "materialization_status": materialization["status"],
        "reserve_B_annotation_content_accessed": False,
        "aggregate_metrics_accessed": False,
        "official_validation_or_test_accessed": False,
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


def preflight() -> dict[str, Any]:
    _validate_protocol()
    checkpoint, _ = _validate_checkpoint()
    materialization = _validate_materialization()
    records = _view_records(materialization, verify_images=True)
    lock = _implementation_lock()
    return {
        "status": "PASS_CVBRA_V1_UAV_OBB_RESERVE_B_LABEL_BLIND_INFERENCE_PREFLIGHT",
        "views": {view: len(records[view]) for view in VIEWS},
        "raw_cells_planned": len(MODELS) * len(VIEWS) * len(ORIENTATIONS) * REPETITIONS,
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
        "runner_sha256": lock["runner_sha256"],
        "reserve_B_annotation_content_accessed": False,
        "official_validation_or_test_accessed": False,
    }


def _raw_root(model: str, view: str, orientation: str, repetition: int) -> Path:
    return OUTPUT / "raw" / model / view / orientation / f"rep_{repetition}"


def _raw_complete(model: str, view: str, orientation: str, repetition: int) -> bool:
    root = _raw_root(model, view, orientation, repetition)
    marker_path = root / "SUCCESS.json"
    if not marker_path.is_file():
        return False
    marker = _load_mapping(marker_path)
    prediction = root / "predictions.coco.json"
    timing = root / "timings.jsonl"
    if (
        marker.get("status") != "PASS"
        or marker.get("model") != model
        or marker.get("view") != view
        or marker.get("orientation") != orientation
        or marker.get("repetition") != repetition
        or marker.get("prediction_sha256") != sha256_file(prediction)
        or marker.get("timing_sha256") != sha256_file(timing)
        or marker.get("implementation_lock_sha256") != sha256_file(IMPLEMENTATION_LOCK)
    ):
        raise CVBRAReserveInferenceError(
            f"raw reserve-B cell changed: {model}/{view}/{orientation}/r{repetition}"
        )
    return True


def _predict_cell(
    model: str,
    view: str,
    orientation: str,
    repetition: int,
    detector: UltralyticsDetector,
    records: Sequence[ImageRecord],
) -> dict[str, Any]:
    root = _raw_root(model, view, orientation, repetition)
    if _raw_complete(model, view, orientation, repetition):
        return _load_mapping(root / "SUCCESS.json")
    batches: list[DetectionBatch] = []
    timings: list[dict[str, Any]] = []
    for start in range(0, len(records), CHUNK_SIZE):
        source = tuple(records[start : start + CHUNK_SIZE])
        preparation_started = perf_counter()
        if orientation == "identity":
            prediction_input = source
            preparation_ms = 0.0
        else:
            prediction_input = horizontal_flip_records(
                source, uri_prefix=f"memory://cvbra-v1/reserve-B/{view}/flip"
            )
            preparation_ms = (perf_counter() - preparation_started) * 1000.0 / len(source)
        current = detector.predict(
            prediction_input,
            imgsz=IMAGE_SIZE,
            conf=PROBE_CONF,
            iou=NMS_IOU,
            max_det=MAX_DET,
            fp16=True,
        )
        if orientation == "flip":
            current = restore_horizontal_flip(current, source)
        batches.extend(current)
        timings.extend(
            {
                "image_id": batch.image_id,
                "model": model,
                "view": view,
                "orientation": orientation,
                "repetition": repetition,
                "detector_ms": batch.latency_ms,
                "orientation_preparation_ms": preparation_ms,
                "end_to_end_ms": batch.latency_ms + preparation_ms,
            }
            for batch in current
        )
    if len(batches) != len(records):
        raise CVBRAReserveInferenceError(f"incomplete reserve-B inference: {model}/{view}")
    prediction = root / "predictions.coco.json"
    timing = root / "timings.jsonl"
    write_coco_predictions(prediction, batches, category_id_by_class=CATEGORY_ID_BY_CLASS)
    atomic_write_text(
        timing,
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in timings),
    )
    marker = {
        "schema_version": 1,
        "status": "PASS",
        "completed_at_utc": _utc_now(),
        "model": model,
        "view": view,
        "orientation": orientation,
        "repetition": repetition,
        "images": len(records),
        "prediction_sha256": sha256_file(prediction),
        "timing_sha256": sha256_file(timing),
        "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
        "reserve_B_annotation_content_accessed": False,
        "aggregate_metrics_accessed": False,
        "official_validation_or_test_accessed": False,
    }
    atomic_write_json(root / "SUCCESS.json", marker)
    print(
        json.dumps({"raw_cell_complete": f"{model}/{view}/{orientation}/r{repetition}"}), flush=True
    )
    return marker


def _release_detector() -> None:
    gc.collect()
    try:
        import torch
    except (ImportError, OSError):
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _validate_raw_lock() -> dict[str, Any]:
    _implementation_lock()
    if not RAW_LOCK.is_file() or not RAW_MARKER.is_file():
        raise CVBRAReserveInferenceError("reserve-B raw lock is incomplete")
    lock = _load_mapping(RAW_LOCK)
    marker = _load_mapping(RAW_MARKER)
    cells = lock.get("cells")
    if (
        lock.get("status") != "ALL_CVBRA_V1_UAV_OBB_RESERVE_B_RAW_OBSERVATIONS_LOCKED"
        or lock.get("implementation_lock_sha256") != sha256_file(IMPLEMENTATION_LOCK)
        or marker.get("raw_observation_lock_sha256") != sha256_file(RAW_LOCK)
        or not isinstance(cells, list)
        or len(cells) != 24
    ):
        raise CVBRAReserveInferenceError("reserve-B raw lock changed")
    for row in cells:
        if not isinstance(row, dict):
            raise CVBRAReserveInferenceError("invalid reserve-B raw row")
        _assert_hash(_rooted(row["prediction"]), row["prediction_sha256"], label="prediction")
        _assert_hash(_rooted(row["timing"]), row["timing_sha256"], label="timing")
    return lock


def infer() -> dict[str, Any]:
    preflight()
    records_by_view = _view_records(_validate_materialization(), verify_images=False)
    if RAW_LOCK.exists():
        return _validate_raw_lock()
    _, weights = _validate_checkpoint()
    cells: list[dict[str, Any]] = []
    for model in MODELS:
        if not all(
            _raw_complete(model, view, orientation, repetition)
            for view in VIEWS
            for orientation in ORIENTATIONS
            for repetition in range(1, REPETITIONS + 1)
        ):
            detector = UltralyticsDetector(
                weights[model][0],
                model_name="yolo11n",
                device="cuda:0",
                expected_class_names=CLASS_NAMES,
                project_root=ROOT,
                stream_chunk_records=CHUNK_SIZE,
                release_cuda_cache_between_chunks=False,
            )
            warmup = records_by_view["original"][:WARMUP_IMAGES]
            detector.predict(
                warmup,
                imgsz=IMAGE_SIZE,
                conf=PROBE_CONF,
                iou=NMS_IOU,
                max_det=MAX_DET,
                fp16=True,
            )
            detector.predict(
                horizontal_flip_records(warmup, uri_prefix="memory://cvbra-v1/warmup/flip"),
                imgsz=IMAGE_SIZE,
                conf=PROBE_CONF,
                iou=NMS_IOU,
                max_det=MAX_DET,
                fp16=True,
            )
            for repetition in range(1, REPETITIONS + 1):
                for view in VIEWS:
                    for orientation in ORIENTATIONS:
                        _predict_cell(
                            model,
                            view,
                            orientation,
                            repetition,
                            detector,
                            records_by_view[view],
                        )
            del detector
            _release_detector()
        for view in VIEWS:
            for orientation in ORIENTATIONS:
                hashes: set[str] = set()
                for repetition in range(1, REPETITIONS + 1):
                    root = _raw_root(model, view, orientation, repetition)
                    marker = _load_mapping(root / "SUCCESS.json")
                    hashes.add(str(marker["prediction_sha256"]))
                    cells.append(
                        {
                            "model": model,
                            "view": view,
                            "orientation": orientation,
                            "repetition": repetition,
                            "prediction": _relative(root / "predictions.coco.json"),
                            "prediction_sha256": marker["prediction_sha256"],
                            "timing": _relative(root / "timings.jsonl"),
                            "timing_sha256": marker["timing_sha256"],
                        }
                    )
                if len(hashes) != 1:
                    raise CVBRAReserveInferenceError(
                        f"reserve-B predictions are not exact: {model}/{view}/{orientation}"
                    )
    payload = {
        "schema_version": 1,
        "status": "ALL_CVBRA_V1_UAV_OBB_RESERVE_B_RAW_OBSERVATIONS_LOCKED",
        "locked_at_utc": _utc_now(),
        "images": IMAGES,
        "prediction_determinism": "exact_sha256_match_across_two_repetitions",
        "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
        "cells": cells,
        "reserve_B_annotation_content_accessed": False,
        "aggregate_metrics_accessed": False,
        "official_validation_or_test_accessed": False,
    }
    atomic_write_json(RAW_LOCK, payload)
    atomic_write_json(
        RAW_MARKER,
        {"status": payload["status"], "raw_observation_lock_sha256": sha256_file(RAW_LOCK)},
    )
    return payload


def _rows_to_batches(
    rows: Sequence[Mapping[str, Any]], image_ids: Sequence[int], *, source: str
) -> tuple[DetectionBatch, ...]:
    allowed = set(image_ids)
    grouped: dict[int, list[Box]] = defaultdict(list)
    for row in rows:
        image_id = int(row["image_id"])
        if image_id not in allowed:
            continue
        bbox = row.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise CVBRAReserveInferenceError("raw prediction has invalid box")
        x, y, width, height = (float(value) for value in bbox)
        category_id = int(row["category_id"])
        if category_id not in CLASS_ID_BY_CATEGORY or width <= 0.0 or height <= 0.0:
            raise CVBRAReserveInferenceError("raw prediction geometry or class is invalid")
        grouped[image_id].append(
            Box(
                xyxy=(x, y, x + width, y + height),
                score=float(row["score"]),
                class_id=CLASS_ID_BY_CATEGORY[category_id],
                source=source,
            )
        )
    return tuple(
        DetectionBatch(image_id=image_id, boxes=tuple(grouped[image_id]), latency_ms=0.0)
        for image_id in image_ids
    )


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


def _timing_mean(path: Path) -> float:
    rows = _load_jsonl(path)
    if len(rows) != IMAGES:
        raise CVBRAReserveInferenceError(f"timing coverage is incomplete: {path}")
    return fmean(float(row["end_to_end_ms"]) for row in rows)


def _postprocess_ms(identity: Sequence[DetectionBatch], flip: Sequence[DetectionBatch]) -> float:
    def operation() -> tuple[DetectionBatch, ...]:
        return _filter_batches(
            merge_classwise_nms(
                identity,
                flip,
                iou_threshold=NMS_IOU,
                max_det=MAX_DET,
                method="cvbra_v1_registered_standard_flip",
            )
        )

    for _ in range(POSTPROCESS_WARMUPS):
        operation()
    observations: list[float] = []
    for _ in range(POSTPROCESS_REPETITIONS):
        started = perf_counter()
        operation()
        observations.append((perf_counter() - started) * 1000.0 / IMAGES)
    return fmean(observations)


def _validate_prediction_lock() -> dict[str, Any]:
    _validate_raw_lock()
    if not PREDICTION_LOCK.is_file() or not PREDICTION_MARKER.is_file():
        raise CVBRAReserveInferenceError("reserve-B joint prediction lock is incomplete")
    lock = _load_mapping(PREDICTION_LOCK)
    marker = _load_mapping(PREDICTION_MARKER)
    artifacts = lock.get("artifacts")
    if (
        lock.get("status")
        != "CVBRA_V1_UAV_OBB_RESERVE_B_PREDICTIONS_LOCKED_BEFORE_ANNOTATIONS_OR_METRICS"
        or lock.get("raw_observation_lock_sha256") != sha256_file(RAW_LOCK)
        or marker.get("joint_prediction_lock_sha256") != sha256_file(PREDICTION_LOCK)
        or not isinstance(artifacts, list)
        or len(artifacts) != 12
    ):
        raise CVBRAReserveInferenceError("reserve-B joint prediction lock changed")
    for row in artifacts:
        if not isinstance(row, dict):
            raise CVBRAReserveInferenceError("invalid reserve-B prediction row")
        _assert_hash(_rooted(row["prediction"]), row["prediction_sha256"], label="prediction")
    return lock


def derive() -> dict[str, Any]:
    infer()
    records = _view_records(_validate_materialization(), verify_images=False)
    if PREDICTION_LOCK.exists():
        return _validate_prediction_lock()
    artifacts: list[dict[str, Any]] = []
    timing: dict[str, dict[str, dict[str, float]]] = {}
    for model in MODELS:
        timing[model] = {}
        for view in VIEWS:
            image_ids = [int(record.image_id) for record in records[view]]
            identity = _rows_to_batches(
                _load_rows(_raw_root(model, view, "identity", 1) / "predictions.coco.json"),
                image_ids,
                source=f"{model}/{view}/identity",
            )
            flip = _rows_to_batches(
                _load_rows(_raw_root(model, view, "flip", 1) / "predictions.coco.json"),
                image_ids,
                source=f"{model}/{view}/flip",
            )
            identity_output = _filter_batches(identity)
            flip_output = _filter_batches(
                merge_classwise_nms(
                    identity,
                    flip,
                    iou_threshold=NMS_IOU,
                    max_det=MAX_DET,
                    method="cvbra_v1_registered_standard_flip",
                )
            )
            identity_ms = median(
                _timing_mean(_raw_root(model, view, "identity", rep) / "timings.jsonl")
                for rep in range(1, REPETITIONS + 1)
            )
            flip_ms = median(
                _timing_mean(_raw_root(model, view, "flip", rep) / "timings.jsonl")
                for rep in range(1, REPETITIONS + 1)
            )
            postprocess_ms = _postprocess_ms(identity, flip)
            timing[model][view] = {
                "Identity_mean_ms": identity_ms,
                "Flip_observation_mean_ms": flip_ms,
                "Flip_postprocess_mean_ms": postprocess_ms,
                "Flip_HardNMS_mean_ms": identity_ms + flip_ms + postprocess_ms,
            }
            for method, batches in (("Identity", identity_output), ("Flip_HardNMS", flip_output)):
                prediction = (
                    OUTPUT / "cells" / model / view / method.casefold() / "predictions.coco.json"
                )
                write_coco_predictions(
                    prediction, batches, category_id_by_class=CATEGORY_ID_BY_CLASS
                )
                artifacts.append(
                    {
                        "model": model,
                        "view": view,
                        "method": method,
                        "prediction": _relative(prediction),
                        "prediction_sha256": sha256_file(prediction),
                    }
                )
    payload = {
        "schema_version": 1,
        "status": "CVBRA_V1_UAV_OBB_RESERVE_B_PREDICTIONS_LOCKED_BEFORE_ANNOTATIONS_OR_METRICS",
        "locked_at_utc": _utc_now(),
        "images": IMAGES,
        "raw_observation_lock_sha256": sha256_file(RAW_LOCK),
        "artifacts": artifacts,
        "timing": timing,
        "reserve_B_annotation_content_accessed": False,
        "aggregate_metrics_accessed": False,
        "official_validation_or_test_accessed": False,
        "paper_body_change_authorized": False,
    }
    atomic_write_json(PREDICTION_LOCK, payload)
    atomic_write_json(
        PREDICTION_MARKER,
        {"status": payload["status"], "joint_prediction_lock_sha256": sha256_file(PREDICTION_LOCK)},
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run label-blind CVBRA reserve-B inference")
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
