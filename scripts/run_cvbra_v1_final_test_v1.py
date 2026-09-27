from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import zipfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path, PurePosixPath
from statistics import fmean, stdev
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image

from buse_uav.data.corruptions import apply_corruption, deterministic_corruption_seed
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
PROTOCOL = ROOT / "configs/experiment/cvbra_v1_final_test_v1.yaml"
PROTOCOL_SHA256 = "381417d192508cefa90b945b43621e175e49cde7c2bb083b0cc52ba20d0b7699"

SOURCE = ROOT / "weights/hazydet/yolo11n_best.pt"
PRIMARY = ROOT / "runs/cvbra_v1/yolo11n/cvbra_v1.pt"
ALDI_EQUAL = (
    ROOT
    / "runs/cvbra_v1_aldi_direct_baseline_v1/equal_supervision"
    / "ALDIpp_AF_Y11_equal_supervision.pt"
)
PRIMARY_LOCK = ROOT / "reports/development/cvbra_v1/checkpoint_lock.json"
ALDI_LOCK = (
    ROOT
    / "reports/development/cvbra_v1_aldi_direct_baseline_v1/equal_supervision"
    / "checkpoint_lock.json"
)

UAV_ARCHIVE = ROOT / "data/raw/UAV_OBB_v4/downloads/UAV-OBB-dlaCi7.zip"
UAV_ARCHIVE_SHA256 = "bf65b0d6a00bc1a320002012146bf44913e2b3f97b9e6acbd7954775d61f69e2"
HAZY_ANNOTATION = ROOT / "data/raw/HazyDet/test/test_coco.json"
HAZY_IMAGE_ROOT = ROOT / "data/raw/HazyDet/test/hazy_images"
CORRUPTION_SOURCE = ROOT / "src/buse_uav/data/corruptions.py"

OUTPUT = ROOT / "reports/final/cvbra_v1_final_test_v1"
DATA_ROOT = ROOT / "data/processed/cvbra_v1_final_test_v1"
REGISTRATION = OUTPUT / "REGISTRATION_LOCK.json"
REGISTERED = OUTPUT / "REGISTERED"
INCIDENT = OUTPUT / "HAZYDET_PRE_REGISTRATION_READ_INCIDENT.json"
VIEW_LOCK = OUTPUT / "uav_obb_test_view_lock.json"
VIEWS_LOCKED = OUTPUT / "UAV_OBB_TEST_VIEWS_LOCKED"
PREDICTION_LOCK = OUTPUT / "joint_prediction_lock.json"
PREDICTIONS_LOCKED = OUTPUT / "PREDICTIONS_LOCKED"
UAV_ANNOTATION = OUTPUT / "uav_obb_test_shared_hbb.coco.json"
LABEL_LOCK = OUTPUT / "uav_obb_test_label_conversion_lock.json"
METRICS = OUTPUT / "final_test_metrics.csv"
STATISTICS = OUTPUT / "final_test_statistics.json"
REPORT = OUTPUT / "final_test_report.json"
COMPLETE = OUTPUT / "COMPLETE.json"

MODELS = ("source", "CVBRA_v1", "ALDIpp_AF_equal")
WEIGHTS = {"source": SOURCE, "CVBRA_v1": PRIMARY, "ALDIpp_AF_equal": ALDI_EQUAL}
UAV_VIEWS = ("original", "fog_0p6", "fog_1p0")
CLASS_NAMES = ("car", "truck", "bus")
UAV_CATEGORY_ID_BY_CLASS = {0: 1, 1: 2, 2: 3}
HAZY_CATEGORY_ID_BY_CLASS = {0: 0, 1: 1, 2: 2}
IMGSZ = 1280
PROBE_CONF = 0.08
PUBLISH_CONF = 0.25
NMS_IOU = 0.70
MAX_DET = 500
CHUNK_SIZE = 8
WARMUP_IMAGES = 16
UAV_IMAGES = 16
HAZY_IMAGES = 2000
UAV_RESAMPLES = 10000
HAZY_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260822
METRIC_KEYS = ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large")


class FinalTestError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FinalTestError(f"cannot parse mapping {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FinalTestError(f"expected mapping: {path}")
    return value


def _assert_file(path: Path, *, digest: str | None = None, label: str) -> None:
    if not path.is_file() or (digest is not None and sha256_file(path) != digest):
        raise FinalTestError(f"locked {label} changed or is missing: {path}")


def register() -> dict[str, Any]:
    _assert_file(PROTOCOL, digest=PROTOCOL_SHA256, label="final-test protocol")
    _assert_file(SOURCE, label="source checkpoint")
    _assert_file(PRIMARY_LOCK, label="primary checkpoint lock")
    _assert_file(ALDI_LOCK, label="ALDI checkpoint lock")
    _assert_file(UAV_ARCHIVE, digest=UAV_ARCHIVE_SHA256, label="UAV-OBB archive")
    _assert_file(HAZY_ANNOTATION, label="HazyDet test annotation")
    _assert_file(CORRUPTION_SOURCE, label="fog implementation")
    primary_lock = _load_mapping(PRIMARY_LOCK)
    aldi_lock = _load_mapping(ALDI_LOCK)
    primary_digest = str(primary_lock.get("checkpoint_sha256", ""))
    aldi_digest = str(aldi_lock.get("checkpoint_sha256", ""))
    if not primary_digest or not aldi_digest:
        raise FinalTestError("fixed model locks are incomplete")
    _assert_file(PRIMARY, digest=primary_digest, label="primary checkpoint")
    _assert_file(ALDI_EQUAL, digest=aldi_digest, label="ALDI equal checkpoint")
    if REGISTRATION.exists() or REGISTERED.exists():
        if not REGISTRATION.is_file() or not REGISTERED.is_file() or not INCIDENT.is_file():
            raise FinalTestError("final-test registration is incomplete")
        lock = _load_mapping(REGISTRATION)
        marker = _load_mapping(REGISTERED)
        if (
            lock.get("protocol_sha256") != PROTOCOL_SHA256
            or lock.get("runner_sha256") != sha256_file(Path(__file__))
            or marker.get("registration_sha256") != sha256_file(REGISTRATION)
            or lock.get("HazyDet_incident_sha256") != sha256_file(INCIDENT)
        ):
            raise FinalTestError("final-test registration changed")
        return lock
    later = (DATA_ROOT, VIEW_LOCK, PREDICTION_LOCK, UAV_ANNOTATION, METRICS, REPORT)
    if any(path.exists() for path in later):
        raise FinalTestError("final-test output appeared before registration")
    incident = {
        "schema_version": 1,
        "status": "DISCLOSED_HAZYDET_TEST_ANNOTATION_READ_BEFORE_REGISTRATION",
        "recorded_at_utc": _now(),
        "event": (
            "A read-only Get-Content -Raw inventory command opened test_coco.json before "
            "this protocol was locked. No test prediction, metric, model selection, "
            "checkpoint selection, threshold selection, or manuscript result followed it."
        ),
        "annotation": _relative(HAZY_ANNOTATION),
        "annotation_sha256": sha256_file(HAZY_ANNOTATION),
        "model_checkpoints_frozen_before_event": True,
        "test_metrics_computed_before_registration": False,
        "label_blind_preregistration_claim": False,
        "permitted_claim": (
            "post-freeze held-out test evidence unused for training or any selection"
        ),
        "scientific_disposition": "retain and disclose; never describe as label-blind",
    }
    atomic_write_json(INCIDENT, incident)
    payload = {
        "schema_version": 1,
        "status": "CVBRA_V1_FINAL_TEST_REGISTERED_BEFORE_PREDICTION_OR_UAV_LABEL_ACCESS",
        "registered_at_utc": _now(),
        "protocol": _relative(PROTOCOL),
        "protocol_sha256": PROTOCOL_SHA256,
        "runner": _relative(Path(__file__)),
        "runner_sha256": sha256_file(Path(__file__)),
        "weights": {
            "source": {"path": _relative(SOURCE), "sha256": sha256_file(SOURCE)},
            "CVBRA_v1": {"path": _relative(PRIMARY), "sha256": primary_digest},
            "ALDIpp_AF_equal": {"path": _relative(ALDI_EQUAL), "sha256": aldi_digest},
        },
        "UAV_OBB_archive_sha256": UAV_ARCHIVE_SHA256,
        "UAV_OBB_test_label_content_accessed_before_registration": False,
        "UAV_OBB_test_member_names_listed_before_registration": True,
        "HazyDet_annotation_sha256": sha256_file(HAZY_ANNOTATION),
        "HazyDet_incident": _relative(INCIDENT),
        "HazyDet_incident_sha256": sha256_file(INCIDENT),
        "corruption_source_sha256": sha256_file(CORRUPTION_SOURCE),
        "threshold_or_checkpoint_selection_from_test": False,
        "paper_body_change_authorized": False,
    }
    atomic_write_json(REGISTRATION, payload)
    atomic_write_json(
        REGISTERED,
        {"status": payload["status"], "registration_sha256": sha256_file(REGISTRATION)},
    )
    return payload


def _uav_image_members(bundle: zipfile.ZipFile) -> list[str]:
    members = sorted(
        (
            name
            for name in bundle.namelist()
            if name.startswith("UAV-OBB/test/images/")
            and PurePosixPath(name).suffix.casefold() in {".jpg", ".jpeg", ".png"}
        ),
        key=str.casefold,
    )
    if len(members) != UAV_IMAGES:
        raise FinalTestError(f"UAV-OBB test image coverage changed: {len(members)}")
    return members


def materialize_uav_views() -> dict[str, Any]:
    register()
    if VIEW_LOCK.exists() or VIEWS_LOCKED.exists():
        if not VIEW_LOCK.is_file() or not VIEWS_LOCKED.is_file():
            raise FinalTestError("UAV test-view lock is incomplete")
        lock = _load_mapping(VIEW_LOCK)
        marker = _load_mapping(VIEWS_LOCKED)
        if marker.get("view_lock_sha256") != sha256_file(VIEW_LOCK):
            raise FinalTestError("UAV test-view lock changed")
        return lock
    if PREDICTION_LOCK.exists() or UAV_ANNOTATION.exists():
        raise FinalTestError("later test output appeared before view lock")
    original_root = DATA_ROOT / "uav_obb_test/original"
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(UAV_ARCHIVE) as bundle:
        for image_id, member in enumerate(_uav_image_members(bundle), 1):
            raw = bundle.read(member)
            suffix = PurePosixPath(member).suffix.casefold()
            destination = original_root / f"{image_id:04d}{suffix}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            temporary.write_bytes(raw)
            os.replace(temporary, destination)
            with Image.open(destination) as image:
                image.load()
                width, height = image.size
                rgb = np.asarray(image.convert("RGB"))
            rows.append(
                {
                    "image_id": image_id,
                    "member": member,
                    "view": "original",
                    "path": _relative(destination),
                    "sha256": sha256_file(destination),
                    "width": width,
                    "height": height,
                }
            )
            for severity, beta, view in (
                (1, 0.6, "fog_0p6"),
                (2, 1.0, "fog_1p0"),
            ):
                seed = deterministic_corruption_seed(42, str(image_id), "fog", severity)
                output = apply_corruption(rgb, corruption="fog", parameter=beta, seed=seed)
                fog_path = DATA_ROOT / "uav_obb_test" / view / f"{image_id:04d}.png"
                fog_path.parent.mkdir(parents=True, exist_ok=True)
                fog_tmp = fog_path.with_suffix(".png.tmp")
                Image.fromarray(output, mode="RGB").save(
                    fog_tmp, format="PNG", compress_level=3, optimize=False
                )
                os.replace(fog_tmp, fog_path)
                rows.append(
                    {
                        "image_id": image_id,
                        "member": member,
                        "view": view,
                        "severity": severity,
                        "beta": beta,
                        "seed": seed,
                        "path": _relative(fog_path),
                        "sha256": sha256_file(fog_path),
                        "width": width,
                        "height": height,
                    }
                )
    payload = {
        "schema_version": 1,
        "status": "UAV_OBB_TEST_VIEWS_MATERIALIZED_BEFORE_LABEL_ACCESS",
        "locked_at_utc": _now(),
        "registration_sha256": sha256_file(REGISTRATION),
        "archive_sha256": UAV_ARCHIVE_SHA256,
        "images": UAV_IMAGES,
        "views": list(UAV_VIEWS),
        "rows": rows,
        "UAV_OBB_test_labels_accessed": False,
        "metrics_accessed": False,
    }
    atomic_write_json(VIEW_LOCK, payload)
    atomic_write_json(
        VIEWS_LOCKED,
        {"status": payload["status"], "view_lock_sha256": sha256_file(VIEW_LOCK)},
    )
    return payload


def _uav_records() -> dict[str, tuple[ImageRecord, ...]]:
    lock = materialize_uav_views()
    raw_rows = lock.get("rows")
    if not isinstance(raw_rows, list):
        raise FinalTestError("UAV view registry changed")
    output: dict[str, tuple[ImageRecord, ...]] = {}
    for view in UAV_VIEWS:
        records = []
        for row in sorted(
            (item for item in raw_rows if isinstance(item, dict) and item.get("view") == view),
            key=lambda item: int(item["image_id"]),
        ):
            path = ROOT / str(row["path"])
            records.append(
                ImageRecord(
                    image_id=int(row["image_id"]),
                    path=str(path.resolve()),
                    width=int(row["width"]),
                    height=int(row["height"]),
                )
            )
        if len(records) != UAV_IMAGES:
            raise FinalTestError(f"UAV test view coverage changed: {view}")
        output[view] = tuple(records)
    return output


def _hazy_records() -> tuple[ImageRecord, ...]:
    paths = sorted(
        (
            path
            for path in HAZY_IMAGE_ROOT.iterdir()
            if path.is_file() and path.suffix.casefold() in {".jpg", ".jpeg", ".png"}
        ),
        key=lambda path: path.name.casefold(),
    )
    if len(paths) != HAZY_IMAGES:
        raise FinalTestError(f"HazyDet test image coverage changed: {len(paths)}")
    records: list[ImageRecord] = []
    for path in paths:
        with Image.open(path) as image:
            width, height = image.size
        records.append(
            ImageRecord(
                image_id=int(path.stem),
                path=str(path.resolve()),
                width=int(width),
                height=int(height),
            )
        )
    if len({record.image_id for record in records}) != HAZY_IMAGES:
        raise FinalTestError("HazyDet test image identities changed")
    return tuple(records)


def _filtered(batches: Sequence[DetectionBatch]) -> tuple[DetectionBatch, ...]:
    return tuple(
        DetectionBatch(
            image_id=batch.image_id,
            boxes=tuple(box for box in batch.boxes if box.score >= PUBLISH_CONF),
            latency_ms=batch.latency_ms,
            meta={**batch.meta, "publish_conf": PUBLISH_CONF},
        )
        for batch in batches
    )


def _prediction(model: str, dataset: str, view: str) -> Path:
    return OUTPUT / "predictions" / model / dataset / view / "predictions.coco.json"


def _predict_cell(
    model: str,
    dataset: str,
    view: str,
    detector: UltralyticsDetector,
    records: Sequence[ImageRecord],
    category_map: Mapping[int, int],
) -> dict[str, Any]:
    prediction = _prediction(model, dataset, view)
    marker_path = prediction.parent / "SUCCESS.json"
    if marker_path.exists():
        marker = _load_mapping(marker_path)
        if marker.get("prediction_sha256") != sha256_file(prediction):
            raise FinalTestError(f"test prediction changed: {model}/{dataset}/{view}")
        return marker
    if prediction.parent.exists():
        raise FinalTestError(f"partial test prediction requires audit: {prediction.parent}")
    batches = detector.predict(
        records,
        imgsz=IMGSZ,
        conf=PROBE_CONF,
        iou=NMS_IOU,
        max_det=MAX_DET,
        fp16=True,
    )
    filtered = _filtered(batches)
    write_coco_predictions(prediction, filtered, category_id_by_class=category_map)
    marker = {
        "schema_version": 1,
        "status": "CVBRA_FINAL_TEST_PREDICTION_COMPLETE",
        "completed_at_utc": _now(),
        "model": model,
        "dataset": dataset,
        "view": view,
        "images": len(filtered),
        "checkpoint_sha256": sha256_file(WEIGHTS[model]),
        "prediction": _relative(prediction),
        "prediction_sha256": sha256_file(prediction),
        "test_labels_or_metrics_used_for_prediction": False,
    }
    atomic_write_json(marker_path, marker)
    return marker


def predict() -> dict[str, Any]:
    register()
    if PREDICTION_LOCK.exists() or PREDICTIONS_LOCKED.exists():
        if not PREDICTION_LOCK.is_file() or not PREDICTIONS_LOCKED.is_file():
            raise FinalTestError("final-test prediction lock is incomplete")
        lock = _load_mapping(PREDICTION_LOCK)
        marker = _load_mapping(PREDICTIONS_LOCKED)
        if marker.get("prediction_lock_sha256") != sha256_file(PREDICTION_LOCK):
            raise FinalTestError("final-test prediction lock changed")
        return lock
    if UAV_ANNOTATION.exists() or METRICS.exists() or REPORT.exists():
        raise FinalTestError("test label or metric output appeared before predictions")
    uav = _uav_records()
    hazy = _hazy_records()
    artifacts: list[dict[str, Any]] = []
    for model in MODELS:
        detector = UltralyticsDetector(
            WEIGHTS[model],
            model_name="yolo11n",
            device="cuda:0",
            expected_class_names=CLASS_NAMES,
            project_root=ROOT,
            stream_chunk_records=CHUNK_SIZE,
            release_cuda_cache_between_chunks=False,
        )
        detector.predict(
            hazy[:WARMUP_IMAGES],
            imgsz=IMGSZ,
            conf=PROBE_CONF,
            iou=NMS_IOU,
            max_det=MAX_DET,
            fp16=True,
        )
        for view in UAV_VIEWS:
            artifacts.append(
                _predict_cell(
                    model,
                    "UAV_OBB_test",
                    view,
                    detector,
                    uav[view],
                    UAV_CATEGORY_ID_BY_CLASS,
                )
            )
        artifacts.append(
            _predict_cell(
                model,
                "HazyDet_test",
                "hazy",
                detector,
                hazy,
                HAZY_CATEGORY_ID_BY_CLASS,
            )
        )
        del detector
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        print(json.dumps({"final_test_prediction_complete": model}), flush=True)
    payload = {
        "schema_version": 1,
        "status": "CVBRA_V1_FINAL_TEST_PREDICTIONS_LOCKED_BEFORE_SCORING",
        "locked_at_utc": _now(),
        "registration_sha256": sha256_file(REGISTRATION),
        "view_lock_sha256": sha256_file(VIEW_LOCK),
        "artifacts": artifacts,
        "UAV_OBB_label_content_accessed_before_lock": False,
        "test_metrics_accessed_before_lock": False,
        "test_driven_selection": False,
    }
    atomic_write_json(PREDICTION_LOCK, payload)
    atomic_write_json(
        PREDICTIONS_LOCKED,
        {"status": payload["status"], "prediction_lock_sha256": sha256_file(PREDICTION_LOCK)},
    )
    return payload


def convert_uav_labels() -> dict[str, Any]:
    predict()
    if LABEL_LOCK.exists():
        lock = _load_mapping(LABEL_LOCK)
        if lock.get("annotation_sha256") != sha256_file(UAV_ANNOTATION):
            raise FinalTestError("UAV test annotation lock changed")
        return lock
    if UAV_ANNOTATION.exists() or METRICS.exists():
        raise FinalTestError("partial UAV test label conversion requires audit")
    view_lock = _load_mapping(VIEW_LOCK)
    raw_rows = view_lock.get("rows")
    if not isinstance(raw_rows, list):
        raise FinalTestError("UAV view registry changed")
    original_rows = sorted(
        (row for row in raw_rows if isinstance(row, dict) and row.get("view") == "original"),
        key=lambda row: int(row["image_id"]),
    )
    images = [
        {
            "id": int(row["image_id"]),
            "file_name": PurePosixPath(str(row["member"])).name,
            "width": int(row["width"]),
            "height": int(row["height"]),
        }
        for row in original_rows
    ]
    annotations: list[dict[str, Any]] = []
    class_counts = {name: 0 for name in CLASS_NAMES}
    source_formats: dict[str, int] = {"xywh": 0, "polygon": 0}
    with zipfile.ZipFile(UAV_ARCHIVE) as bundle:
        yaml_value = yaml.safe_load(bundle.read("UAV-OBB/data.yaml").decode("utf-8"))
        names = yaml_value.get("names") if isinstance(yaml_value, dict) else None
        ordered_names = (
            [str(names[key]) for key in sorted(names)]
            if isinstance(names, dict)
            else [str(value) for value in names]
            if isinstance(names, list)
            else []
        )
        if tuple(name.casefold() for name in ordered_names[:3]) != CLASS_NAMES:
            raise FinalTestError(f"UAV-OBB class order changed: {ordered_names}")
        archive_names = set(bundle.namelist())
        annotation_id = 1
        for row in original_rows:
            image_id = int(row["image_id"])
            width = int(row["width"])
            height = int(row["height"])
            stem = PurePosixPath(str(row["member"])).stem
            label_member = f"UAV-OBB/test/labels/{stem}.txt"
            if label_member not in archive_names:
                raise FinalTestError(f"UAV test label is missing: {label_member}")
            text = bundle.read(label_member).decode("utf-8")
            for line_number, line in enumerate(text.splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    values = [float(value) for value in line.split()]
                except ValueError as exc:
                    raise FinalTestError(
                        f"invalid UAV label value: {label_member}:{line_number}"
                    ) from exc
                if len(values) not in {5, 9}:
                    raise FinalTestError(
                        f"unsupported UAV label geometry: {label_member}:{line_number}"
                    )
                class_id = int(values[0])
                if float(class_id) != values[0]:
                    raise FinalTestError(f"noninteger UAV class: {label_member}:{line_number}")
                if class_id not in UAV_CATEGORY_ID_BY_CLASS:
                    continue
                coords = values[1:]
                if len(coords) == 4:
                    center_x, center_y, box_w, box_h = coords
                    x1 = center_x - box_w / 2.0
                    y1 = center_y - box_h / 2.0
                    x2 = center_x + box_w / 2.0
                    y2 = center_y + box_h / 2.0
                    source_formats["xywh"] += 1
                else:
                    xs = coords[0::2]
                    ys = coords[1::2]
                    x1, x2 = min(xs), max(xs)
                    y1, y2 = min(ys), max(ys)
                    source_formats["polygon"] += 1
                x1 = min(max(x1, 0.0), 1.0)
                y1 = min(max(y1, 0.0), 1.0)
                x2 = min(max(x2, 0.0), 1.0)
                y2 = min(max(y2, 0.0), 1.0)
                if x2 <= x1 or y2 <= y1:
                    continue
                bbox = [x1 * width, y1 * height, (x2 - x1) * width, (y2 - y1) * height]
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": class_id + 1,
                        "bbox": bbox,
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 0,
                    }
                )
                class_counts[CLASS_NAMES[class_id]] += 1
                annotation_id += 1
    atomic_write_json(
        UAV_ANNOTATION,
        {
            "info": {
                "description": "UAV-OBB official test, shared classes, HBB enclosure",
                "conversion": "released xywh or polygon normalized geometry to HBB",
            },
            "images": images,
            "annotations": annotations,
            "categories": [
                {"id": index + 1, "name": name} for index, name in enumerate(CLASS_NAMES)
            ],
        },
    )
    payload = {
        "schema_version": 1,
        "status": "UAV_OBB_TEST_LABELS_CONVERTED_AFTER_PREDICTION_LOCK",
        "locked_at_utc": _now(),
        "prediction_lock_sha256": sha256_file(PREDICTION_LOCK),
        "archive_sha256": UAV_ARCHIVE_SHA256,
        "annotation": _relative(UAV_ANNOTATION),
        "annotation_sha256": sha256_file(UAV_ANNOTATION),
        "images": len(images),
        "annotations": len(annotations),
        "class_counts": class_counts,
        "source_formats": source_formats,
        "prediction_or_metric_recomputed_after_label_access": False,
    }
    atomic_write_json(LABEL_LOCK, payload)
    return payload


def _csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    fields = (
        "dataset",
        "model",
        "view",
        *METRIC_KEYS,
        "images_evaluated",
        "prediction_sha256",
    )
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row[key] for key in fields})
    return buffer.getvalue()


def _read_metric_rows() -> list[dict[str, Any]]:
    try:
        with METRICS.open("r", encoding="utf-8", newline="") as stream:
            raw_rows = list(csv.DictReader(stream))
    except OSError as exc:
        raise FinalTestError(f"cannot read final-test metrics: {exc}") from exc
    if len(raw_rows) != len(MODELS) * (len(UAV_VIEWS) + 1):
        raise FinalTestError("final-test metric row coverage changed")
    rows: list[dict[str, Any]] = []
    for row in raw_rows:
        rows.append(
            {
                **row,
                **{key: float(row[key]) for key in METRIC_KEYS},
                "images_evaluated": int(row["images_evaluated"]),
            }
        )
    return rows


def points() -> dict[str, Any]:
    convert_uav_labels()
    if METRICS.exists():
        return {
            "status": "FINAL_TEST_POINT_ESTIMATES_LOCKED",
            "metrics_sha256": sha256_file(METRICS),
            "rows": _read_metric_rows(),
        }
    if STATISTICS.exists() or REPORT.exists():
        raise FinalTestError("later final-test analysis appeared before points")
    hazy_document = _load_mapping(HAZY_ANNOTATION)
    hazy_images = hazy_document.get("images")
    if not isinstance(hazy_images, list):
        raise FinalTestError("HazyDet test annotation is incomplete")
    hazy_ids = [int(row["id"]) for row in hazy_images if isinstance(row, dict)]
    if len(hazy_ids) != HAZY_IMAGES or len(set(hazy_ids)) != HAZY_IMAGES:
        raise FinalTestError("HazyDet test annotation coverage changed")
    if set(hazy_ids) != {int(record.image_id) for record in _hazy_records()}:
        raise FinalTestError("HazyDet image and annotation identities differ")
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        for view in UAV_VIEWS:
            prediction = _prediction(model, "UAV_OBB_test", view)
            result = evaluate_coco(
                UAV_ANNOTATION,
                prediction,
                max_det=MAX_DET,
                image_ids=list(range(1, UAV_IMAGES + 1)),
            )
            rows.append(
                {
                    "dataset": "UAV_OBB_test",
                    "model": model,
                    "view": view,
                    **{key: float(result[key]) for key in METRIC_KEYS},
                    "images_evaluated": int(result["images_evaluated"]),
                    "prediction_sha256": sha256_file(prediction),
                }
            )
        prediction = _prediction(model, "HazyDet_test", "hazy")
        result = evaluate_coco(HAZY_ANNOTATION, prediction, max_det=MAX_DET, image_ids=hazy_ids)
        rows.append(
            {
                "dataset": "HazyDet_test",
                "model": model,
                "view": "hazy",
                **{key: float(result[key]) for key in METRIC_KEYS},
                "images_evaluated": int(result["images_evaluated"]),
                "prediction_sha256": sha256_file(prediction),
            }
        )
    rows.sort(key=lambda row: (str(row["dataset"]), str(row["model"]), str(row["view"])))
    atomic_write_text(METRICS, _csv_text(rows))
    return {
        "status": "FINAL_TEST_POINT_ESTIMATES_LOCKED",
        "metrics_sha256": sha256_file(METRICS),
        "rows": rows,
    }


def _checkpoint_deltas(path: Path, expected: int) -> list[float]:
    document = _load_mapping(path)
    values = document.get("deltas")
    if not isinstance(values, list) or len(values) != expected:
        raise FinalTestError(f"bootstrap checkpoint is incomplete: {path}")
    result = [float(value) for value in values]
    if not all(np.isfinite(value) for value in result):
        raise FinalTestError("bootstrap checkpoint contains nonfinite deltas")
    return result


def _bootstrap_contrast(
    *,
    dataset: str,
    view: str,
    baseline: str,
    annotation: Path,
    image_ids: Sequence[int],
    resamples: int,
    direct_delta: float,
    seed_offset: int,
) -> dict[str, Any]:
    checkpoint = OUTPUT / "bootstrap" / f"{dataset}_{view}_CVBRA_minus_{baseline}.json"
    clusters = {str(image_id): [image_id] for image_id in image_ids}
    raw = paired_coco_ap_cluster_bootstrap_scopes(
        annotation,
        _prediction(baseline, dataset, view),
        _prediction("CVBRA_v1", dataset, view),
        {
            "test": ClusterBootstrapScope(
                clusters=clusters,
                checkpoint_path=checkpoint,
                checkpoint_identity={
                    "study": "cvbra_v1_final_test_v1",
                    "dataset": dataset,
                    "view": view,
                    "baseline": baseline,
                    "prediction_lock_sha256": sha256_file(PREDICTION_LOCK),
                },
            )
        },
        resamples=resamples,
        seed=BOOTSTRAP_SEED + seed_offset,
        max_det=MAX_DET,
        workers=4,
        chunk_resamples=100,
        accelerate_ap_only=True,
    )["test"]
    raw_deltas = _checkpoint_deltas(checkpoint, resamples)
    offset = direct_delta - float(raw["delta"])
    adjusted = [value + offset for value in raw_deltas]
    return {
        "dataset": dataset,
        "view": view,
        "comparison": f"CVBRA_v1_minus_{baseline}",
        "baseline": baseline,
        "method": "CVBRA_v1",
        "direct_delta": direct_delta,
        "raw_bootstrap_point": float(raw["delta"]),
        "recenter_offset": offset,
        "ci_low": float(np.percentile(adjusted, 2.5)),
        "ci_high": float(np.percentile(adjusted, 97.5)),
        "bootstrap_mean_delta": fmean(adjusted),
        "bootstrap_standard_deviation": stdev(adjusted),
        "p_two_sided": bootstrap_sign_pvalue(adjusted),
        "resamples": resamples,
        "seed": BOOTSTRAP_SEED + seed_offset,
        "images": len(image_ids),
        "checkpoint": _relative(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
    }


def statistics() -> dict[str, Any]:
    point = points()
    if STATISTICS.exists():
        return _load_mapping(STATISTICS)
    if REPORT.exists():
        raise FinalTestError("final report appeared before statistics")
    rows = point.get("rows")
    if not isinstance(rows, list):
        raise FinalTestError("final-test point rows are unavailable")
    lookup = {
        (str(row["dataset"]), str(row["model"]), str(row["view"])): row
        for row in rows
        if isinstance(row, dict)
    }
    uav_ids = list(range(1, UAV_IMAGES + 1))
    hazy_document = _load_mapping(HAZY_ANNOTATION)
    hazy_ids = sorted(
        int(row["id"]) for row in hazy_document.get("images", []) if isinstance(row, dict)
    )
    uav_rows: list[dict[str, Any]] = []
    hazy_rows: list[dict[str, Any]] = []
    offset = 0
    for view in UAV_VIEWS:
        for baseline in ("source", "ALDIpp_AF_equal"):
            direct = float(lookup[("UAV_OBB_test", "CVBRA_v1", view)]["AP"]) - float(
                lookup[("UAV_OBB_test", baseline, view)]["AP"]
            )
            uav_rows.append(
                _bootstrap_contrast(
                    dataset="UAV_OBB_test",
                    view=view,
                    baseline=baseline,
                    annotation=UAV_ANNOTATION,
                    image_ids=uav_ids,
                    resamples=UAV_RESAMPLES,
                    direct_delta=direct,
                    seed_offset=offset,
                )
            )
            offset += 1
    for baseline in ("source", "ALDIpp_AF_equal"):
        direct = float(lookup[("HazyDet_test", "CVBRA_v1", "hazy")]["AP"]) - float(
            lookup[("HazyDet_test", baseline, "hazy")]["AP"]
        )
        hazy_rows.append(
            _bootstrap_contrast(
                dataset="HazyDet_test",
                view="hazy",
                baseline=baseline,
                annotation=HAZY_ANNOTATION,
                image_ids=hazy_ids,
                resamples=HAZY_RESAMPLES,
                direct_delta=direct,
                seed_offset=100 + offset,
            )
        )
        offset += 1
    for family in (uav_rows, hazy_rows):
        adjusted = holm_adjust([float(row["p_two_sided"]) for row in family])
        for row, value in zip(family, adjusted, strict=True):
            row["holm_adjusted_p"] = value
            row["holm_significant_0p05"] = value < 0.05
    payload = {
        "schema_version": 1,
        "status": "CVBRA_V1_FINAL_TEST_PAIRED_STATISTICS_COMPLETE",
        "completed_at_utc": _now(),
        "metrics_sha256": sha256_file(METRICS),
        "UAV_OBB_test": {
            "unit": "image",
            "images": UAV_IMAGES,
            "resamples": UAV_RESAMPLES,
            "multiple_testing": "Holm across six registered AP contrasts",
            "rows": uav_rows,
        },
        "HazyDet_test": {
            "unit": "image",
            "images": HAZY_IMAGES,
            "resamples": HAZY_RESAMPLES,
            "multiple_testing": "Holm across two registered AP contrasts",
            "rows": hazy_rows,
        },
        "test_driven_selection": False,
    }
    atomic_write_json(STATISTICS, payload)
    return payload


def finalize() -> dict[str, Any]:
    point = points()
    stats = statistics()
    if REPORT.exists() and COMPLETE.exists():
        report = _load_mapping(REPORT)
        marker = _load_mapping(COMPLETE)
        if marker.get("report_sha256") != sha256_file(REPORT):
            raise FinalTestError("final-test report changed")
        return report
    if REPORT.exists() or COMPLETE.exists():
        raise FinalTestError("partial final-test report requires audit")
    payload = {
        "schema_version": 1,
        "status": "COMPLETE_CVBRA_V1_FINAL_HELD_OUT_TEST_V1",
        "completed_at_utc": _now(),
        "protocol_sha256": PROTOCOL_SHA256,
        "registration_sha256": sha256_file(REGISTRATION),
        "incident_sha256": sha256_file(INCIDENT),
        "view_lock_sha256": sha256_file(VIEW_LOCK),
        "prediction_lock_sha256": sha256_file(PREDICTION_LOCK),
        "label_lock_sha256": sha256_file(LABEL_LOCK),
        "metrics": _relative(METRICS),
        "metrics_sha256": sha256_file(METRICS),
        "statistics": _relative(STATISTICS),
        "statistics_sha256": sha256_file(STATISTICS),
        "rows": point.get("rows"),
        "paired_statistics": stats,
        "evidence_boundary": {
            "UAV_OBB_test": (
                "one-shot label-blind-before-prediction 16-image directional confirmation"
            ),
            "HazyDet_test": (
                "post-freeze held-out evidence; annotation was accidentally opened during "
                "inventory before registration and was never used for selection"
            ),
            "test_driven_method_checkpoint_or_threshold_change": False,
            "all_registered_outcomes_retained": True,
            "UAV_population_level_superiority_claim": False,
        },
        "paper_body_change_authorized": True,
    }
    atomic_write_json(REPORT, payload)
    atomic_write_json(
        COMPLETE,
        {
            "status": payload["status"],
            "report_sha256": sha256_file(REPORT),
            "metrics_sha256": sha256_file(METRICS),
            "statistics_sha256": sha256_file(STATISTICS),
        },
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CVBRA-v1 final held-out tests")
    parser.add_argument(
        "--stage",
        choices=("register", "materialize", "predict", "convert", "points", "statistics", "all"),
        default="all",
    )
    args = parser.parse_args()
    if args.stage == "register":
        result = register()
    elif args.stage == "materialize":
        result = materialize_uav_views()
    elif args.stage == "predict":
        result = predict()
    elif args.stage == "convert":
        result = convert_uav_labels()
    elif args.stage == "points":
        result = points()
    elif args.stage == "statistics":
        result = statistics()
    else:
        result = finalize()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
