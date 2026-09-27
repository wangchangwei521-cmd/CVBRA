from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import os
from collections import Counter, OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any

import torch
from scripts import run_cvbra_v1_training_seed_robustness as evidence_base

from buse_uav.detectors.ultralytics_adapter import (
    UltralyticsDetector,
    configure_ultralytics_environment,
)
from buse_uav.evaluation.coco import evaluate_coco, write_coco_predictions
from buse_uav.schemas import DetectionBatch, ImageRecord
from buse_uav.utils.hashing import sha256_file
from buse_uav.utils.io import atomic_write_json, atomic_write_text

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/experiment/cvbra_v1_allocation_sensitivity_v1.yaml"
PROTOCOL_SHA256 = "9976f14d8ac2fc73c82f73f3eb0446bd9eb814c01c036af650d8bc24e1c04ee7"

SOURCE_CHECKPOINT = ROOT / "weights/hazydet/yolo11n_best.pt"
PRIMARY_CHECKPOINT = ROOT / "runs/cvbra_v1/yolo11n/cvbra_v1.pt"
PRIMARY_DATASET = ROOT / "data/processed/cvbra_v1/dataset.yaml"
PRIMARY_MANIFEST = ROOT / "data/processed/cvbra_v1/manifest.json"
PRIMARY_DATA_LOCK = ROOT / "reports/development/cvbra_v1/generated_dataset_lock.json"

NO_REPLAY_CHECKPOINT = ROOT / "runs/cvbra_v1_matched_baselines/CVBRA_noReplay/CVBRA_noReplay.pt"
NO_FREEZE_CHECKPOINT = ROOT / "runs/cvbra_v1_matched_baselines/CVBRA_noFreeze/CVBRA_noFreeze.pt"
NO_REPLAY_LOCK = (
    ROOT / "reports/development/cvbra_v1_matched_baselines/checkpoint_locks/CVBRA_noReplay.json"
)
NO_FREEZE_LOCK = (
    ROOT / "reports/development/cvbra_v1_matched_baselines/checkpoint_locks/CVBRA_noFreeze.json"
)

TARGET_MATERIALIZATION = (
    ROOT / "reports/development/cvbra_v1/uav_obb_official_validation/view_materialization_lock.json"
)
TARGET_CONVERSION = (
    ROOT
    / "reports/development/cvbra_v1/uav_obb_official_validation/annotations/conversion_lock.json"
)
TARGET_ANNOTATION = (
    ROOT
    / "reports/development/cvbra_v1/uav_obb_official_validation/annotations"
    / "official_validation_exact_car_truck_bus_hbb.coco.json"
)
HAZY_ANNOTATION = ROOT / "data/raw/HazyDet/val/val_coco.json"

PRIMARY_TARGET_ROOT = (
    ROOT / "reports/development/cvbra_v1/uav_obb_official_validation/evaluation/cells"
)
MATCHED_TARGET_ROOT = (
    ROOT
    / "reports/development/cvbra_v1_matched_baselines/uav_obb_official_validation/evaluation/cells"
)
HAZY_REFERENCE_ROOT = (
    ROOT
    / "reports/development/cvbra_v1_matched_baselines/hazydet_source_retention_v2"
    / "corrected_predictions"
)

OUTPUT = ROOT / "reports/development/cvbra_v1_allocation_sensitivity_v1"
DATA_ROOT = ROOT / "data/processed/cvbra_v1_allocation_sensitivity_v1"
RUN_ROOT = ROOT / "runs/cvbra_v1_allocation_sensitivity_v1"
REGISTRATION = OUTPUT / "REGISTRATION_LOCK.json"
REGISTERED = OUTPUT / "REGISTERED"
DATA_LOCK = OUTPUT / "qS_0p125_data_lock.json"
PREDICTION_LOCK = OUTPUT / "joint_prediction_lock.json"
PREDICTIONS_LOCKED = OUTPUT / "PREDICTIONS_LOCKED"
METRICS = OUTPUT / "allocation_sensitivity_metrics.csv"
REPORT = OUTPUT / "allocation_sensitivity_report.json"
COMPLETE = OUTPUT / "COMPLETE.json"

EPOCHS = 8
IMGSZ = 1280
BATCH = 2
SEED = 42
TRAIN_IMAGES = 3600
TARGET_VIEWS = ("original", "fog_0p6", "fog_1p0")
CLASS_NAMES = ("car", "truck", "bus")
TARGET_CATEGORY_ID_BY_CLASS = {0: 1, 1: 2, 2: 3}
HAZY_CATEGORY_ID_BY_CLASS = {0: 0, 1: 1, 2: 2}
PROBE_CONF = 0.08
PUBLISH_CONF = 0.25
NMS_IOU = 0.70
MAX_DET = 500
WARMUP_IMAGES = 16
CHUNK_SIZE = 8
METRIC_KEYS = ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large")


@dataclass(frozen=True)
class NewCell:
    dataset: Path
    first_trainable_layer: int


NEW_CELLS: dict[str, NewCell] = {
    "qS_0p125": NewCell(DATA_ROOT / "qS_0p125/dataset.yaml", 10),
    "first_trainable_5": NewCell(PRIMARY_DATASET, 5),
    "first_trainable_15": NewCell(PRIMARY_DATASET, 15),
}

REPLAY_AXIS = OrderedDict(
    (
        ("qS_0", ("reference", "CVBRA_noReplay")),
        ("qS_0p125", ("new", "qS_0p125")),
        ("qS_0p25", ("reference", "CVBRA_v1")),
    )
)
FREEZE_AXIS = OrderedDict(
    (
        ("first_trainable_0", ("reference", "CVBRA_noFreeze")),
        ("first_trainable_5", ("new", "first_trainable_5")),
        ("first_trainable_10", ("reference", "CVBRA_v1")),
        ("first_trainable_15", ("new", "first_trainable_15")),
    )
)


class SensitivityError(RuntimeError):
    pass


def _now() -> str:
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
        raise SensitivityError(f"cannot parse mapping {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SensitivityError(f"expected mapping: {path}")
    return value


def _assert_file(path: Path, *, digest: str | None = None, label: str) -> None:
    if not path.is_file() or (digest is not None and sha256_file(path) != digest):
        raise SensitivityError(f"locked {label} changed or is missing: {path}")


def _checkpoint_for_reference(model: str) -> Path:
    if model == "CVBRA_v1":
        return PRIMARY_CHECKPOINT
    if model == "CVBRA_noReplay":
        return NO_REPLAY_CHECKPOINT
    if model == "CVBRA_noFreeze":
        return NO_FREEZE_CHECKPOINT
    raise SensitivityError(f"unknown reference model: {model}")


def _checkpoint(cell: str) -> Path:
    return RUN_ROOT / cell / f"{cell}.pt"


def _training_lock(cell: str) -> Path:
    return OUTPUT / "training_locks" / f"{cell}.json"


def _checkpoint_lock(cell: str) -> Path:
    return OUTPUT / "checkpoint_locks" / f"{cell}.json"


def _prediction(cell: str, domain: str, view: str) -> Path:
    return OUTPUT / "predictions" / cell / domain / view / "predictions.coco.json"


def _reference_prediction(model: str, domain: str, view: str) -> Path:
    if domain == "UAV_OBB_validation":
        root = PRIMARY_TARGET_ROOT if model == "CVBRA_v1" else MATCHED_TARGET_ROOT
        return root / model / view / "identity/predictions.coco.json"
    if domain == "HazyDet_validation":
        return HAZY_REFERENCE_ROOT / f"{model}.coco.json"
    raise SensitivityError(f"unknown domain: {domain}")


def _validate_reference_checkpoint(lock_path: Path, checkpoint: Path, model: str) -> str:
    lock = _load_mapping(lock_path)
    digest = str(lock.get("checkpoint_sha256", ""))
    if lock.get("baseline") != model or not digest:
        raise SensitivityError(f"reference checkpoint lock changed: {model}")
    _assert_file(checkpoint, digest=digest, label=f"{model} checkpoint")
    return digest


def register() -> dict[str, Any]:
    _assert_file(PROTOCOL, digest=PROTOCOL_SHA256, label="sensitivity protocol")
    for path, label in (
        (SOURCE_CHECKPOINT, "source checkpoint"),
        (PRIMARY_CHECKPOINT, "primary checkpoint"),
        (PRIMARY_DATASET, "primary dataset YAML"),
        (PRIMARY_MANIFEST, "primary manifest"),
        (PRIMARY_DATA_LOCK, "primary data lock"),
        (TARGET_MATERIALIZATION, "target materialization"),
        (TARGET_CONVERSION, "target conversion"),
        (TARGET_ANNOTATION, "target annotation"),
        (HAZY_ANNOTATION, "HazyDet validation annotation"),
    ):
        _assert_file(path, label=label)
    primary_lock = _load_mapping(ROOT / "reports/development/cvbra_v1/checkpoint_lock.json")
    primary_digest = str(primary_lock.get("checkpoint_sha256", ""))
    if not primary_digest:
        raise SensitivityError("primary checkpoint lock is incomplete")
    _assert_file(PRIMARY_CHECKPOINT, digest=primary_digest, label="primary checkpoint")
    no_replay_digest = _validate_reference_checkpoint(
        NO_REPLAY_LOCK, NO_REPLAY_CHECKPOINT, "CVBRA_noReplay"
    )
    no_freeze_digest = _validate_reference_checkpoint(
        NO_FREEZE_LOCK, NO_FREEZE_CHECKPOINT, "CVBRA_noFreeze"
    )
    for model in ("CVBRA_v1", "CVBRA_noReplay", "CVBRA_noFreeze"):
        for view in TARGET_VIEWS:
            _assert_file(
                _reference_prediction(model, "UAV_OBB_validation", view),
                label=f"{model} target {view} prediction",
            )
        _assert_file(
            _reference_prediction(model, "HazyDet_validation", "hazy"),
            label=f"{model} HazyDet prediction",
        )
    if REGISTRATION.exists() or REGISTERED.exists():
        if not REGISTRATION.is_file() or not REGISTERED.is_file():
            raise SensitivityError("sensitivity registration is incomplete")
        lock = _load_mapping(REGISTRATION)
        marker = _load_mapping(REGISTERED)
        if (
            lock.get("protocol_sha256") != PROTOCOL_SHA256
            or lock.get("runner_sha256") != sha256_file(Path(__file__))
            or marker.get("registration_sha256") != sha256_file(REGISTRATION)
        ):
            raise SensitivityError("sensitivity registration changed")
        return lock
    later = (DATA_ROOT, RUN_ROOT, DATA_LOCK, PREDICTION_LOCK, METRICS, REPORT)
    if any(path.exists() for path in later):
        raise SensitivityError("sensitivity output appeared before registration")
    payload = {
        "schema_version": 1,
        "status": "CVBRA_V1_ALLOCATION_SENSITIVITY_REGISTERED_BEFORE_NEW_OUTPUT",
        "registered_at_utc": _now(),
        "protocol": _relative(PROTOCOL),
        "protocol_sha256": PROTOCOL_SHA256,
        "runner": _relative(Path(__file__)),
        "runner_sha256": sha256_file(Path(__file__)),
        "source_checkpoint_sha256": sha256_file(SOURCE_CHECKPOINT),
        "primary_checkpoint_sha256": primary_digest,
        "noReplay_checkpoint_sha256": no_replay_digest,
        "noFreeze_checkpoint_sha256": no_freeze_digest,
        "primary_dataset_sha256": sha256_file(PRIMARY_DATASET),
        "primary_manifest_sha256": sha256_file(PRIMARY_MANIFEST),
        "target_materialization_sha256": sha256_file(TARGET_MATERIALIZATION),
        "target_annotation_sha256": sha256_file(TARGET_ANNOTATION),
        "HazyDet_annotation_sha256": sha256_file(HAZY_ANNOTATION),
        "new_cells": {
            name: {"first_trainable_layer": spec.first_trainable_layer}
            for name, spec in NEW_CELLS.items()
        },
        "validation_labels_previously_accessed": True,
        "method_or_hyperparameter_reselection": False,
        "test_content_accessed_by_this_study": False,
        "paper_body_change_authorized": False,
    }
    atomic_write_json(REGISTRATION, payload)
    atomic_write_json(
        REGISTERED,
        {"status": payload["status"], "registration_sha256": sha256_file(REGISTRATION)},
    )
    return payload


def _hardlink(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file() or sha256_file(target) != sha256_file(source):
            raise SensitivityError(f"existing alias differs: {target}")
        return
    os.link(source, target)


def build_data() -> dict[str, Any]:
    registration = register()
    dataset = NEW_CELLS["qS_0p125"].dataset
    manifest_path = dataset.parent / "manifest.json"
    if DATA_LOCK.exists():
        lock = _load_mapping(DATA_LOCK)
        if (
            lock.get("registration_sha256") != sha256_file(REGISTRATION)
            or lock.get("dataset_sha256") != sha256_file(dataset)
            or lock.get("manifest_sha256") != sha256_file(manifest_path)
        ):
            raise SensitivityError("qS=0.125 data lock changed")
        return lock
    if RUN_ROOT.exists():
        raise SensitivityError("training output appeared before qS data lock")
    document = _load_mapping(PRIMARY_MANIFEST)
    raw_entries = document.get("entries")
    if not isinstance(raw_entries, list) or len(raw_entries) != TRAIN_IMAGES:
        raise SensitivityError("primary manifest entries changed")
    entries = [dict(row) for row in raw_entries if isinstance(row, dict)]
    if len(entries) != TRAIN_IMAGES:
        raise SensitivityError("primary manifest contains invalid entries")
    by_view: dict[str, list[dict[str, Any]]] = {view: [] for view in TARGET_VIEWS}
    replay: list[dict[str, Any]] = []
    for row in entries:
        if row.get("role") == "target" and row.get("view") in by_view:
            by_view[str(row["view"])].append(row)
        elif row.get("role") == "source_haze_replay":
            replay.append(row)
    for view in TARGET_VIEWS:
        by_view[view].sort(key=lambda row: int(row["image_id"]))
        if len(by_view[view]) != 900:
            raise SensitivityError(f"target view coverage changed: {view}")
    replay.sort(key=lambda row: int(row["image_id"]))
    if len(replay) != 900:
        raise SensitivityError("source replay coverage changed")
    plan: list[tuple[dict[str, Any], str]] = []
    for view in TARGET_VIEWS:
        plan.extend(
            (row, f"target_{view}_{index:04d}") for index, row in enumerate(by_view[view], 1)
        )
        plan.extend(
            (row, f"target_{view}_{index:04d}_extra")
            for index, row in enumerate(by_view[view][:150], 1)
        )
    plan.extend((row, f"source_{index:04d}") for index, row in enumerate(replay[:450], 1))
    if len(plan) != TRAIN_IMAGES:
        raise SensitivityError("qS=0.125 materialization plan changed")
    output_entries: list[dict[str, Any]] = []
    for ordinal, (row, alias) in enumerate(plan, 1):
        source_image = _rooted(row["image"])
        source_label = _rooted(row["label"])
        stem = f"{ordinal:04d}_{alias}"
        target_image = dataset.parent / "images/train" / f"{stem}{source_image.suffix.lower()}"
        target_label = dataset.parent / "labels/train" / f"{stem}.txt"
        _hardlink(source_image, target_image)
        _hardlink(source_label, target_label)
        output_entries.append(
            {
                "ordinal": ordinal,
                "alias": alias,
                "role": row["role"],
                "view": row.get("view", "original_hazy"),
                "image_id": int(row["image_id"]),
                "image": _relative(target_image),
                "image_sha256": sha256_file(target_image),
                "label": _relative(target_label),
                "label_sha256": sha256_file(target_label),
                "source_image": str(row["image"]),
            }
        )
    yaml_text = (
        f"path: {dataset.parent.resolve()}\n"
        "train: images/train\n"
        "val: images/train\n"
        "names:\n"
        "  0: car\n"
        "  1: truck\n"
        "  2: bus\n"
    )
    atomic_write_text(dataset, yaml_text)
    composition = dict(Counter(f"{row['role']}:{row['view']}" for row in output_entries))
    atomic_write_json(
        manifest_path,
        {
            "schema_version": 1,
            "status": "CVBRA_QS_0P125_DATASET_MATERIALIZED",
            "registration_sha256": sha256_file(REGISTRATION),
            "entries": output_entries,
            "composition": composition,
        },
    )
    payload = {
        "schema_version": 1,
        "status": "CVBRA_QS_0P125_DATA_LOCKED_BEFORE_TRAINING",
        "locked_at_utc": _now(),
        "registration_sha256": sha256_file(REGISTRATION),
        "dataset": _relative(dataset),
        "dataset_sha256": sha256_file(dataset),
        "manifest": _relative(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "images": len(output_entries),
        "composition": composition,
        "source_fraction": 0.125,
        "method_or_hyperparameter_selection": False,
        "test_content_accessed": False,
        "registration_status": registration["status"],
    }
    atomic_write_json(DATA_LOCK, payload)
    return payload


def _endpoint_state(
    source: Mapping[str, torch.Tensor],
    trained: Mapping[str, torch.Tensor],
    first_trainable_layer: int,
) -> OrderedDict[str, torch.Tensor]:
    if tuple(source) != tuple(trained):
        raise SensitivityError("source/trained state schemas differ")
    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, value in trained.items():
        source_value = source[name]
        layer = evidence_base._layer_index(name)
        selected = source_value if layer < first_trainable_layer else value
        if selected.is_floating_point() and not bool(torch.isfinite(selected).all()):
            raise SensitivityError(f"nonfinite state: {name}")
        result[name] = selected.clone()
    return result


def _validate_new_checkpoint(cell: str) -> dict[str, Any]:
    lock = _load_mapping(_checkpoint_lock(cell))
    checkpoint = _checkpoint(cell)
    if (
        lock.get("cell") != cell
        or lock.get("checkpoint_sha256") != sha256_file(checkpoint)
        or lock.get("registration_sha256") != sha256_file(REGISTRATION)
    ):
        raise SensitivityError(f"checkpoint lock changed: {cell}")
    return lock


def train_cell(cell: str) -> dict[str, Any]:
    build_data()
    if cell not in NEW_CELLS:
        raise SensitivityError(f"unregistered sensitivity cell: {cell}")
    if _checkpoint_lock(cell).exists():
        return _validate_new_checkpoint(cell)
    spec = NEW_CELLS[cell]
    fit = RUN_ROOT / cell / "raw_endpoint/fit"
    if fit.exists() or _checkpoint(cell).exists() or _training_lock(cell).exists():
        raise SensitivityError(f"partial training output requires audit: {cell}")
    configure_ultralytics_environment(ROOT)
    try:
        from ultralytics import YOLO  # type: ignore[attr-defined]
    except (ImportError, OSError, PermissionError) as exc:
        raise SensitivityError(f"cannot import Ultralytics: {exc}") from exc
    model = YOLO(str(SOURCE_CHECKPOINT))
    kwargs: dict[str, Any] = {
        "data": str(spec.dataset.resolve()),
        "epochs": EPOCHS,
        "imgsz": IMGSZ,
        "batch": BATCH,
        "device": "0",
        "workers": 4,
        "project": str(fit.parent.resolve()),
        "name": fit.name,
        "exist_ok": False,
        "pretrained": True,
        "val": False,
        "optimizer": "SGD",
        "lr0": 0.00075,
        "lrf": 0.10,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 1.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "translate": 0.1,
        "scale": 0.3,
        "close_mosaic": 0,
        "patience": EPOCHS,
        "amp": True,
        "seed": SEED,
        "deterministic": True,
        "max_det": MAX_DET,
        "cache": False,
        "plots": False,
        "verbose": False,
        "save": True,
    }
    if spec.first_trainable_layer:
        kwargs["freeze"] = spec.first_trainable_layer
    results = model.train(**kwargs)
    actual = Path(results.save_dir)
    last = actual / "weights/last.pt"
    results_csv = actual / "results.csv"
    args_yaml = actual / "args.yaml"
    for path in (last, results_csv, args_yaml):
        _assert_file(path, label=f"{cell} training output")
    raw_lock = {
        "schema_version": 1,
        "status": "CVBRA_ALLOCATION_SENSITIVITY_RAW_ENDPOINT_LOCKED",
        "locked_at_utc": _now(),
        "cell": cell,
        "registration_sha256": sha256_file(REGISTRATION),
        "last_checkpoint": _relative(last),
        "last_checkpoint_sha256": sha256_file(last),
        "results": _relative(results_csv),
        "results_sha256": sha256_file(results_csv),
        "args": _relative(args_yaml),
        "args_sha256": sha256_file(args_yaml),
        "validation_metric_used_for_training_or_selection": False,
        "test_content_accessed": False,
    }
    atomic_write_json(_training_lock(cell), raw_lock)
    source_payload, source_model = evidence_base._load_checkpoint(SOURCE_CHECKPOINT)
    _, trained_model = evidence_base._load_checkpoint(last)
    state = _endpoint_state(
        source_model.state_dict(), trained_model.state_dict(), spec.first_trainable_layer
    )
    output_model = copy.deepcopy(trained_model)
    incompatible = output_model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise SensitivityError(f"strict endpoint load failed: {cell}")
    output_model = output_model.half().eval()
    payload = copy.deepcopy(source_payload)
    payload.update(
        {
            "epoch": -1,
            "best_fitness": None,
            "model": None,
            "ema": output_model,
            "optimizer": None,
            "scaler": None,
            "updates": None,
            "date": _now(),
            "train_metrics": {},
            "train_results": {},
            "cvbra_v1_allocation_sensitivity": {
                "cell": cell,
                "first_trainable_layer": spec.first_trainable_layer,
                "endpoint": "fixed_last_epoch",
                "protocol_sha256": PROTOCOL_SHA256,
                "registration_sha256": sha256_file(REGISTRATION),
                "training_lock_sha256": sha256_file(_training_lock(cell)),
                "metric_selected": False,
            },
        }
    )
    checkpoint = _checkpoint(cell)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(checkpoint)
    _, observed = evidence_base._load_checkpoint(checkpoint)
    observed_state = observed.state_dict()
    source_state = source_model.state_dict()
    frozen_exact = all(
        torch.equal(observed_state[name].to(dtype=source_state[name].dtype), source_state[name])
        for name in observed_state
        if evidence_base._layer_index(name) < spec.first_trainable_layer
    )
    changed_trainable = sum(
        1
        for name, value in observed_state.items()
        if evidence_base._layer_index(name) >= spec.first_trainable_layer
        and value.is_floating_point()
        and not torch.equal(value.to(dtype=source_state[name].dtype), source_state[name])
    )
    if not frozen_exact or changed_trainable == 0:
        raise SensitivityError(f"endpoint verification failed: {cell}")
    lock = {
        "schema_version": 1,
        "status": "CVBRA_ALLOCATION_SENSITIVITY_CHECKPOINT_VERIFIED_AND_LOCKED",
        "locked_at_utc": _now(),
        "cell": cell,
        "first_trainable_layer": spec.first_trainable_layer,
        "checkpoint": _relative(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "registration_sha256": sha256_file(REGISTRATION),
        "training_lock_sha256": sha256_file(_training_lock(cell)),
        "frozen_state_exact_source": frozen_exact,
        "changed_trainable_floating_states": changed_trainable,
        "validation_metric_used_for_training_or_selection": False,
        "test_content_accessed": False,
    }
    atomic_write_json(_checkpoint_lock(cell), lock)
    return lock


def train_all() -> dict[str, Any]:
    return {
        "status": "ALL_NEW_ALLOCATION_SENSITIVITY_ENDPOINTS_TRAINED",
        "cells": {cell: train_cell(cell) for cell in NEW_CELLS},
    }


def _target_records() -> tuple[dict[str, tuple[ImageRecord, ...]], list[int]]:
    materialization = _load_mapping(TARGET_MATERIALIZATION)
    clean = materialization.get("clean_rows")
    fog = materialization.get("fog_rows")
    primary = materialization.get("primary_image_ids")
    if not isinstance(clean, list) or not isinstance(fog, list) or not isinstance(primary, list):
        raise SensitivityError("target materialization registry changed")
    rows_by_view: dict[str, list[dict[str, Any]]] = {
        "original": [dict(row) for row in clean if isinstance(row, dict)],
        "fog_0p6": [
            dict(row) for row in fog if isinstance(row, dict) and row.get("view") == "fog_0p6"
        ],
        "fog_1p0": [
            dict(row) for row in fog if isinstance(row, dict) and row.get("view") == "fog_1p0"
        ],
    }
    output: dict[str, tuple[ImageRecord, ...]] = {}
    for view, rows in rows_by_view.items():
        records: list[ImageRecord] = []
        for row in sorted(rows, key=lambda item: int(item["image_id"])):
            path = _rooted(row["path"])
            records.append(
                ImageRecord(
                    image_id=int(row["image_id"]),
                    path=str(path.resolve()),
                    width=int(row["width"]),
                    height=int(row["height"]),
                )
            )
        if len(records) != 218:
            raise SensitivityError(f"target evaluation coverage changed: {view}")
        output[view] = tuple(records)
    primary_ids = [int(value) for value in primary]
    if len(primary_ids) != 167 or len(set(primary_ids)) != 167:
        raise SensitivityError("primary target scope changed")
    return output, primary_ids


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


def _predict_cell(
    cell: str,
    domain: str,
    view: str,
    detector: UltralyticsDetector,
    records: Sequence[ImageRecord],
    category_map: Mapping[int, int],
) -> dict[str, Any]:
    prediction = _prediction(cell, domain, view)
    marker_path = prediction.parent / "SUCCESS.json"
    if marker_path.exists():
        marker = _load_mapping(marker_path)
        if marker.get("prediction_sha256") != sha256_file(prediction):
            raise SensitivityError(f"prediction changed: {cell}/{domain}/{view}")
        return marker
    if prediction.parent.exists():
        raise SensitivityError(f"partial prediction requires audit: {prediction.parent}")
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
        "status": "CVBRA_ALLOCATION_SENSITIVITY_PREDICTION_COMPLETE",
        "completed_at_utc": _now(),
        "cell": cell,
        "domain": domain,
        "view": view,
        "images": len(filtered),
        "checkpoint_sha256": sha256_file(_checkpoint(cell)),
        "prediction": _relative(prediction),
        "prediction_sha256": sha256_file(prediction),
        "metrics_accessed_for_prediction": False,
        "test_content_accessed": False,
    }
    atomic_write_json(marker_path, marker)
    return marker


def infer() -> dict[str, Any]:
    register()
    for cell in NEW_CELLS:
        _validate_new_checkpoint(cell)
    if PREDICTION_LOCK.exists() or PREDICTIONS_LOCKED.exists():
        if not PREDICTION_LOCK.is_file() or not PREDICTIONS_LOCKED.is_file():
            raise SensitivityError("sensitivity prediction lock is incomplete")
        lock = _load_mapping(PREDICTION_LOCK)
        marker = _load_mapping(PREDICTIONS_LOCKED)
        if marker.get("prediction_lock_sha256") != sha256_file(PREDICTION_LOCK):
            raise SensitivityError("sensitivity prediction lock changed")
        return lock
    if METRICS.exists() or REPORT.exists():
        raise SensitivityError("metrics appeared before prediction lock")
    target, _ = _target_records()
    hazy = evidence_base._hazy_records(verify_hashes=False)
    artifacts: list[dict[str, Any]] = []
    for cell in NEW_CELLS:
        detector = UltralyticsDetector(
            _checkpoint(cell),
            model_name="yolo11n",
            device="cuda:0",
            expected_class_names=CLASS_NAMES,
            project_root=ROOT,
            stream_chunk_records=CHUNK_SIZE,
            release_cuda_cache_between_chunks=False,
        )
        detector.predict(
            target["original"][:WARMUP_IMAGES],
            imgsz=IMGSZ,
            conf=PROBE_CONF,
            iou=NMS_IOU,
            max_det=MAX_DET,
            fp16=True,
        )
        for view in TARGET_VIEWS:
            artifacts.append(
                _predict_cell(
                    cell,
                    "UAV_OBB_validation",
                    view,
                    detector,
                    target[view],
                    TARGET_CATEGORY_ID_BY_CLASS,
                )
            )
        artifacts.append(
            _predict_cell(
                cell,
                "HazyDet_validation",
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
        print(json.dumps({"sensitivity_inference_complete": cell}), flush=True)
    payload = {
        "schema_version": 1,
        "status": "CVBRA_ALLOCATION_SENSITIVITY_PREDICTIONS_LOCKED_BEFORE_METRICS",
        "locked_at_utc": _now(),
        "registration_sha256": sha256_file(REGISTRATION),
        "artifacts": artifacts,
        "reference_predictions_reused": True,
        "validation_labels_previously_accessed": True,
        "new_metrics_accessed_before_lock": False,
        "test_content_accessed": False,
    }
    atomic_write_json(PREDICTION_LOCK, payload)
    atomic_write_json(
        PREDICTIONS_LOCKED,
        {"status": payload["status"], "prediction_lock_sha256": sha256_file(PREDICTION_LOCK)},
    )
    return payload


def _csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    fields = (
        "axis",
        "cell",
        "endpoint",
        "domain",
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


def _score_axis(
    axis: str,
    cells: Mapping[str, tuple[str, str]],
    primary_ids: Sequence[int],
    hazy_ids: Sequence[int | str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell, (kind, endpoint) in cells.items():
        for view in TARGET_VIEWS:
            prediction = (
                _prediction(endpoint, "UAV_OBB_validation", view)
                if kind == "new"
                else _reference_prediction(endpoint, "UAV_OBB_validation", view)
            )
            result = evaluate_coco(
                TARGET_ANNOTATION, prediction, max_det=MAX_DET, image_ids=primary_ids
            )
            rows.append(
                {
                    "axis": axis,
                    "cell": cell,
                    "endpoint": endpoint,
                    "domain": "target",
                    "view": view,
                    **{key: float(result[key]) for key in METRIC_KEYS},
                    "images_evaluated": int(result["images_evaluated"]),
                    "prediction_sha256": sha256_file(prediction),
                }
            )
        prediction = (
            _prediction(endpoint, "HazyDet_validation", "hazy")
            if kind == "new"
            else _reference_prediction(endpoint, "HazyDet_validation", "hazy")
        )
        result = evaluate_coco(HAZY_ANNOTATION, prediction, max_det=MAX_DET, image_ids=hazy_ids)
        rows.append(
            {
                "axis": axis,
                "cell": cell,
                "endpoint": endpoint,
                "domain": "source_retention",
                "view": "HazyDet",
                **{key: float(result[key]) for key in METRIC_KEYS},
                "images_evaluated": int(result["images_evaluated"]),
                "prediction_sha256": sha256_file(prediction),
            }
        )
    return rows


def score() -> dict[str, Any]:
    prediction_lock = infer()
    if REPORT.exists() and COMPLETE.exists():
        report = _load_mapping(REPORT)
        marker = _load_mapping(COMPLETE)
        if marker.get("report_sha256") != sha256_file(REPORT):
            raise SensitivityError("sensitivity report changed")
        return report
    if REPORT.exists() or COMPLETE.exists() or METRICS.exists():
        raise SensitivityError("partial sensitivity metric output requires audit")
    _, primary_ids = _target_records()
    hazy = evidence_base._hazy_records(verify_hashes=False)
    hazy_ids = [record.image_id for record in hazy]
    rows = _score_axis("replay_allocation", REPLAY_AXIS, primary_ids, hazy_ids)
    rows.extend(_score_axis("freeze_boundary", FREEZE_AXIS, primary_ids, hazy_ids))
    rows.sort(key=lambda row: (str(row["axis"]), str(row["cell"]), str(row["view"])))
    atomic_write_text(METRICS, _csv_text(rows))
    primary_by_axis = {
        "replay_allocation": "qS_0p25",
        "freeze_boundary": "first_trainable_10",
    }
    summaries: dict[str, Any] = {}
    for axis, primary_cell in primary_by_axis.items():
        selected = [row for row in rows if row["axis"] == axis]
        primary = {
            str(row["view"]): float(row["AP"]) for row in selected if row["cell"] == primary_cell
        }
        cell_profiles: dict[str, Any] = {}
        for cell in sorted({str(row["cell"]) for row in selected}):
            profile = {
                str(row["view"]): float(row["AP"]) for row in selected if row["cell"] == cell
            }
            cell_profiles[cell] = {
                "AP": profile,
                "AP_delta_vs_primary": {
                    view: value - primary[view] for view, value in profile.items()
                },
            }
        summaries[axis] = {
            "primary_cell": primary_cell,
            "cells": cell_profiles,
            "AP_range_by_coordinate": {
                view: {
                    "minimum": min(float(row["AP"]) for row in selected if row["view"] == view),
                    "maximum": max(float(row["AP"]) for row in selected if row["view"] == view),
                }
                for view in (*TARGET_VIEWS, "HazyDet")
            },
        }
    order_report = _load_mapping(
        ROOT
        / "reports/development/cvbra_v1_training_order_robustness_v4"
        / "effective_order_robustness_v4_report.json"
    )
    payload = {
        "schema_version": 1,
        "status": "COMPLETE_CVBRA_V1_ALLOCATION_AND_FREEZE_SENSITIVITY_V1",
        "completed_at_utc": _now(),
        "protocol_sha256": PROTOCOL_SHA256,
        "registration_sha256": sha256_file(REGISTRATION),
        "prediction_lock_sha256": sha256_file(PREDICTION_LOCK),
        "prediction_lock_status": prediction_lock["status"],
        "metrics": _relative(METRICS),
        "metrics_sha256": sha256_file(METRICS),
        "rows": rows,
        "summaries": summaries,
        "effective_order_variability": {
            "status": order_report.get("status"),
            "decision": order_report.get("decision"),
            "target_original_AP": order_report.get("metric_summary", {})
            .get("variability", {})
            .get("target_original", {})
            .get("AP"),
            "target_fog_1p0_AP": order_report.get("metric_summary", {})
            .get("variability", {})
            .get("target_fog_1p0", {})
            .get("AP"),
            "source_HazyDet_AP": order_report.get("metric_summary", {})
            .get("variability", {})
            .get("source_HazyDet", {})
            .get("AP"),
            "interpretation": "three independently materialized effective sample orders",
            "stochastic_seed_variance_claim": False,
        },
        "interpretation_boundary": {
            "post_freeze_descriptive": True,
            "primary_endpoint_reselected": False,
            "primary_declared_optimal": False,
            "negative_nonmonotone_or_null_results_retained": True,
            "independent_confirmation_claim": False,
            "test_content_accessed_by_this_study": False,
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
        },
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CVBRA allocation sensitivity v1")
    parser.add_argument(
        "--stage",
        choices=("register", "build-data", "train", "infer", "score", "all"),
        default="all",
    )
    parser.add_argument("--cell", choices=tuple(NEW_CELLS))
    args = parser.parse_args()
    if args.stage == "register":
        result = register()
    elif args.stage == "build-data":
        result = build_data()
    elif args.stage == "train":
        result = train_cell(args.cell) if args.cell else train_all()
    elif args.stage == "infer":
        result = infer()
    elif args.stage == "score":
        result = score()
    else:
        train_all()
        result = score()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
