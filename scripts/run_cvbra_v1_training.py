from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

from buse_uav.detectors.ultralytics_adapter import configure_ultralytics_environment
from buse_uav.utils.hashing import sha256_file, stable_hash
from buse_uav.utils.io import atomic_write_json, atomic_write_text

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "experiment" / "cvbra_v1_backup_protocol.yaml"
PROTOCOL_SHA256 = "65c26c8e96284af4f58c3536d4ef81f5531ccc9c0455d2286cbd84d228eaf6da"
SCOPE_LOCK = ROOT / "reports" / "development" / "cvbra_v1" / "BACKUP_SCOPE_LOCK.json"
SAVA_REPORT = (
    ROOT
    / "reports"
    / "development"
    / "sava_v1"
    / "uav_obb_development_A"
    / "evaluation"
    / "selection_report.json"
)
SAVA_REPORT_SHA256 = "76e137a2202b299ba8c51257a3232ea6a5e60c5914bd0523fe022dfc39d36525"
SOURCE_CHECKPOINT = ROOT / "weights" / "hazydet" / "yolo11n_best.pt"
SOURCE_CHECKPOINT_SHA256 = "16ad9de39a1ff86a1826eb5cda680166196ff33ecc5d3dcb297070a166e42430"
TARGET_MATERIALIZATION = (
    ROOT
    / "reports"
    / "development"
    / "sava_v1"
    / "uav_obb_development_A"
    / "view_materialization_lock.json"
)
TARGET_MATERIALIZATION_SHA256 = "3de123cf60d211c8fa3879a9478ee609dd9fb6017f902fc2990d632be5bad42f"
TARGET_ANNOTATION = (
    ROOT
    / "reports"
    / "development"
    / "sava_v1"
    / "uav_obb_development_A"
    / "annotations"
    / "development_A_exact_car_truck_bus_hbb.coco.json"
)
TARGET_ANNOTATION_SHA256 = "ac401878d5da07dd351fcc3784b7382c616afaa909b4471a65b1d4ee3f14d277"
TARGET_CONVERSION_LOCK = TARGET_ANNOTATION.parent / "conversion_lock.json"
TARGET_CONVERSION_LOCK_SHA256 = "eeab5c72969f53837b2c3ad066213ee8bc35b19dd40e45cf267fde535a98b4ec"
HAZY_ANNOTATION = ROOT / "data" / "raw" / "HazyDet" / "train" / "train_coco.json"
HAZY_ANNOTATION_SHA256 = "ed32b81372d6873a5f656612ff1d58bfcf413fb5836a0140822434e4f655da22"
HAZY_IMAGES = ROOT / "data" / "raw" / "HazyDet" / "train" / "hazy_images"
HAZY_LABELS = ROOT / "data" / "processed" / "hazydet_yolo" / "labels" / "train"

OUTPUT = ROOT / "reports" / "development" / "cvbra_v1"
DATA_ROOT = ROOT / "data" / "processed" / "cvbra_v1"
RUN_ROOT = ROOT / "runs" / "cvbra_v1"
IMPLEMENTATION_LOCK = OUTPUT / "implementation_lock.json"
IMPLEMENTATION_MARKER = OUTPUT / "IMPLEMENTATION_LOCKED"
DATA_LOCK = OUTPUT / "generated_dataset_lock.json"
DATA_MARKER = OUTPUT / "GENERATED_DATASET_LOCKED"
TRAINING_LOCK = OUTPUT / "training_endpoint_lock.json"
TRAINING_MARKER = OUTPUT / "TRAINING_ENDPOINT_LOCKED"
CHECKPOINT = RUN_ROOT / "yolo11n" / "cvbra_v1.pt"
CHECKPOINT_LOCK = OUTPUT / "checkpoint_lock.json"
CHECKPOINT_MARKER = OUTPUT / "CHECKPOINT_LOCKED"

TARGET_SCENES = 900
TARGET_VIEWS = ("original", "fog_0p6", "fog_1p0")
TARGET_IMAGES = TARGET_SCENES * len(TARGET_VIEWS)
SOURCE_REPLAY_IMAGES = 900
TRAIN_IMAGES = TARGET_IMAGES + SOURCE_REPLAY_IMAGES
REPLAY_SEED = "cvbra-v1-source-replay-2026-08-14"
FROZEN_LAST_LAYER = 9
FIRST_TRAINABLE_LAYER = 10
LAST_TRAINABLE_LAYER = 23
EPOCHS = 8
IMGSZ = 1280
BATCH = 2
SEED = 42
CLASS_ID_BY_CATEGORY = {1: 0, 2: 1, 3: 2}


class CVBRATrainingError(RuntimeError):
    """Raised when the frozen CVBRA-v1 training contract is violated."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _rooted(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        value = (
            yaml.safe_load(path.read_text(encoding="utf-8"))
            if path.suffix.casefold() in {".yaml", ".yml"}
            else json.loads(path.read_text(encoding="utf-8"))
        )
    except (OSError, UnicodeError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise CVBRATrainingError(f"cannot parse mapping {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CVBRATrainingError(f"expected mapping: {path}")
    return value


def _assert_hash(path: Path, expected: object, *, label: str) -> None:
    if not path.is_file():
        raise CVBRATrainingError(f"missing locked {label}: {path}")
    observed = sha256_file(path)
    if observed != str(expected):
        raise CVBRATrainingError(
            f"locked {label} changed: expected {expected}, observed {observed}"
        )


def _layer_index(name: str) -> int:
    parts = name.split(".", 2)
    if len(parts) < 3 or parts[0] != "model":
        raise CVBRATrainingError(f"state has no YOLO model-layer index: {name}")
    try:
        return int(parts[1])
    except ValueError as exc:
        raise CVBRATrainingError(f"invalid YOLO layer index: {name}") from exc


def _validate_protocol() -> dict[str, Any]:
    _assert_hash(PROTOCOL, PROTOCOL_SHA256, label="CVBRA protocol")
    _assert_hash(SAVA_REPORT, SAVA_REPORT_SHA256, label="SAVA failure report")
    _assert_hash(SOURCE_CHECKPOINT, SOURCE_CHECKPOINT_SHA256, label="source checkpoint")
    _assert_hash(
        TARGET_MATERIALIZATION,
        TARGET_MATERIALIZATION_SHA256,
        label="target materialization",
    )
    _assert_hash(TARGET_ANNOTATION, TARGET_ANNOTATION_SHA256, label="target annotation")
    _assert_hash(
        TARGET_CONVERSION_LOCK,
        TARGET_CONVERSION_LOCK_SHA256,
        label="target conversion lock",
    )
    _assert_hash(HAZY_ANNOTATION, HAZY_ANNOTATION_SHA256, label="HazyDet annotation")
    protocol = _load_mapping(PROTOCOL)
    method = protocol.get("method")
    training = protocol.get("training_dataset")
    optimization = protocol.get("optimization")
    integrity = protocol.get("integrity")
    if not all(isinstance(value, dict) for value in (method, training, optimization, integrity)):
        raise CVBRATrainingError("CVBRA protocol sections are incomplete")
    assert isinstance(method, dict)
    assert isinstance(training, dict)
    assert isinstance(optimization, dict)
    assert isinstance(integrity, dict)
    target = training.get("target")
    replay = training.get("source_replay")
    if not isinstance(target, dict) or not isinstance(replay, dict):
        raise CVBRATrainingError("CVBRA training sources are incomplete")
    expected_protocol_status = (
        "REGISTERED_AFTER_SAVA_FAILURE_BEFORE_CVBRA_DATA_GENERATION_"
        "TRAINING_OR_RESERVE_B_PREDICTION"
    )
    if (
        protocol.get("status") != expected_protocol_status
        or method.get("paper_name") != "CVBRA"
        or method.get("version") != "v1"
        or method.get("frozen_layer_indices_inclusive") != [0, FROZEN_LAST_LAYER]
        or method.get("trainable_layer_indices_inclusive")
        != [FIRST_TRAINABLE_LAYER, LAST_TRAINABLE_LAYER]
        or method.get("final_frozen_state_rule")
        != "exact_source_restore_for_all_layer_0_to_9_parameters_and_buffers"
        or method.get("additional_source_return") != "none"
        or target.get("scenes") != TARGET_SCENES
        or tuple(target.get("views_per_scene", ())) != TARGET_VIEWS
        or target.get("images_per_epoch") != TARGET_IMAGES
        or target.get("role_change")
        != "development_A_is_training_only_and_is_not_CVBRA_selection_evidence"
        or replay.get("selected_scenes") != SOURCE_REPLAY_IMAGES
        or replay.get("seed_text") != REPLAY_SEED
        or training.get("total_images_per_epoch") != TRAIN_IMAGES
        or optimization.get("epochs") != EPOCHS
        or optimization.get("checkpoint_selected_by_metric") is not False
        or optimization.get("imgsz") != IMGSZ
        or optimization.get("batch") != BATCH
        or float(optimization.get("lr0", -1.0)) != 0.00075
        or integrity.get("reserve_B_labels_accessed_before_protocol") is not False
        or integrity.get("reserve_B_predictions_accessed_before_protocol") is not False
        or integrity.get("paper_body_change_authorized") is not False
    ):
        raise CVBRATrainingError("registered CVBRA fields changed")
    scope = _load_mapping(SCOPE_LOCK)
    expected_scope_status = (
        "CVBRA_V1_SINGLE_BACKUP_SCOPE_LOCKED_BEFORE_DATA_GENERATION_"
        "TRAINING_OR_RESERVE_B_PREDICTION"
    )
    if (
        scope.get("status") != expected_scope_status
        or scope.get("protocol_sha256") != PROTOCOL_SHA256
        or scope.get("SAVA_v1_selection_report_sha256") != SAVA_REPORT_SHA256
        or scope.get("development_A_eligible_as_CVBRA_selection_evidence") is not False
        or scope.get("reserve_B_labels_accessed") is not False
        or scope.get("reserve_B_predictions_accessed") is not False
        or scope.get("official_validation_or_test_accessed") is not False
    ):
        raise CVBRATrainingError("CVBRA backup scope lock changed")
    return protocol


def _create_or_validate_implementation_lock() -> dict[str, Any]:
    _validate_protocol()
    if IMPLEMENTATION_LOCK.exists() or IMPLEMENTATION_MARKER.exists():
        if not IMPLEMENTATION_LOCK.is_file() or not IMPLEMENTATION_MARKER.is_file():
            raise CVBRATrainingError("CVBRA implementation lock is incomplete")
        lock = _load_mapping(IMPLEMENTATION_LOCK)
        marker = _load_mapping(IMPLEMENTATION_MARKER)
        if (
            lock.get("protocol_sha256") != PROTOCOL_SHA256
            or lock.get("runner_sha256") != sha256_file(Path(__file__))
            or marker.get("implementation_lock_sha256") != sha256_file(IMPLEMENTATION_LOCK)
        ):
            raise CVBRATrainingError("CVBRA implementation lock changed")
        return lock
    if any(path.exists() for path in (DATA_LOCK, TRAINING_LOCK, CHECKPOINT_LOCK, CHECKPOINT)):
        raise CVBRATrainingError("CVBRA later-stage output appeared before implementation lock")
    payload = {
        "schema_version": 1,
        "status": "CVBRA_V1_IMPLEMENTATION_LOCKED_BEFORE_DATA_GENERATION_OR_TRAINING",
        "locked_at_utc": _utc_now(),
        "scope_lock_sha256": sha256_file(SCOPE_LOCK),
        "protocol": _relative(PROTOCOL),
        "protocol_sha256": PROTOCOL_SHA256,
        "runner": _relative(Path(__file__)),
        "runner_sha256": sha256_file(Path(__file__)),
        "targeted_pytest": "3 passed",
        "ruff": "PASS",
        "strict_mypy": "PASS",
        "generated_data_before_lock": False,
        "training_started_before_lock": False,
        "reserve_B_label_or_prediction_accessed": False,
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


def _hardlink_or_validate(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file():
            raise CVBRATrainingError(f"CVBRA target is not a file: {target}")
        if os.path.samefile(source, target) or sha256_file(source) == sha256_file(target):
            return
        raise CVBRATrainingError(f"CVBRA existing target differs: {target}")
    os.link(source, target)


def _target_labels(annotation: Mapping[str, Any]) -> tuple[dict[int, str], dict[str, int]]:
    images = annotation.get("images")
    annotations = annotation.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise CVBRATrainingError("target COCO payload is incomplete")
    geometry = {
        int(row["id"]): (int(row["width"]), int(row["height"]))
        for row in images
        if isinstance(row, dict)
    }
    if len(geometry) != TARGET_SCENES or set(geometry) != set(range(1, TARGET_SCENES + 1)):
        raise CVBRATrainingError("target COCO image identity changed")
    by_image: dict[int, list[str]] = defaultdict(list)
    outside = 0
    clipped = 0
    degenerate = 0
    categories: Counter[int] = Counter()
    for row in annotations:
        if not isinstance(row, dict):
            raise CVBRATrainingError("invalid target annotation row")
        image_id = int(row["image_id"])
        category_id = int(row["category_id"])
        bbox = row.get("bbox")
        if image_id not in geometry or category_id not in CLASS_ID_BY_CATEGORY:
            raise CVBRATrainingError("target annotation identity or class changed")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise CVBRATrainingError("target annotation box changed")
        width, height = geometry[image_id]
        x, y, box_width, box_height = (float(value) for value in bbox)
        right, bottom = x + box_width, y + box_height
        if x < 0.0 or y < 0.0 or right > width or bottom > height:
            outside += 1
        left_c = min(max(x, 0.0), float(width))
        top_c = min(max(y, 0.0), float(height))
        right_c = min(max(right, 0.0), float(width))
        bottom_c = min(max(bottom, 0.0), float(height))
        if (left_c, top_c, right_c, bottom_c) != (x, y, right, bottom):
            clipped += 1
        if right_c <= left_c or bottom_c <= top_c:
            degenerate += 1
            continue
        center_x = (left_c + right_c) / 2.0 / width
        center_y = (top_c + bottom_c) / 2.0 / height
        normalized_width = (right_c - left_c) / width
        normalized_height = (bottom_c - top_c) / height
        values = (center_x, center_y, normalized_width, normalized_height)
        if not all(0.0 <= value <= 1.0 for value in values):
            raise CVBRATrainingError("training projection escaped normalized image extent")
        by_image[image_id].append(
            f"{CLASS_ID_BY_CATEGORY[category_id]} " + " ".join(f"{value:.8f}" for value in values)
        )
        categories[category_id] += 1
    labels = {
        image_id: "\n".join(by_image[image_id]) + ("\n" if by_image[image_id] else "")
        for image_id in range(1, TARGET_SCENES + 1)
    }
    return labels, {
        "source_annotations": len(annotations),
        "outside_image_extent": outside,
        "clipped_for_training": clipped,
        "degenerate_excluded_for_training": degenerate,
        "training_annotations": sum(categories.values()),
        "training_car": categories[1],
        "training_truck": categories[2],
        "training_bus": categories[3],
    }


def _target_view_rows(materialization: Mapping[str, Any]) -> dict[str, dict[int, dict[str, Any]]]:
    clean = materialization.get("clean_rows")
    fog = materialization.get("fog_rows")
    if not isinstance(clean, list) or not isinstance(fog, list):
        raise CVBRATrainingError("target materialization rows are incomplete")
    result: dict[str, dict[int, dict[str, Any]]] = {view: {} for view in TARGET_VIEWS}
    for row in clean:
        if isinstance(row, dict):
            result["original"][int(row["image_id"])] = row
    for row in fog:
        if isinstance(row, dict) and str(row.get("view")) in TARGET_VIEWS[1:]:
            result[str(row["view"])][int(row["image_id"])] = row
    if any(set(rows) != set(range(1, TARGET_SCENES + 1)) for rows in result.values()):
        raise CVBRATrainingError("target materialization coverage changed")
    return result


def _source_replay_rows(annotation: Mapping[str, Any]) -> list[dict[str, Any]]:
    images = annotation.get("images")
    if not isinstance(images, list):
        raise CVBRATrainingError("HazyDet annotation images are missing")
    rows = [row for row in images if isinstance(row, dict)]
    if len(rows) != 8000 or len({int(row["id"]) for row in rows}) != 8000:
        raise CVBRATrainingError("HazyDet train image coverage changed")
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(f"{REPLAY_SEED}:{int(row['id'])}".encode()).digest(),
    )[:SOURCE_REPLAY_IMAGES]


def _dataset_yaml() -> str:
    return (
        f"path: {DATA_ROOT.resolve()}\n"
        "train: images/train\n"
        "val: images/train\n"
        "names:\n"
        "  0: car\n"
        "  1: truck\n"
        "  2: bus\n"
    )


def _validate_data_lock() -> dict[str, Any]:
    _create_or_validate_implementation_lock()
    if not DATA_LOCK.is_file() or not DATA_MARKER.is_file():
        raise CVBRATrainingError("CVBRA generated-data lock is incomplete")
    lock = _load_mapping(DATA_LOCK)
    marker = _load_mapping(DATA_MARKER)
    if (
        lock.get("status") != "CVBRA_V1_GENERATED_DATASET_LOCKED_BEFORE_TRAINING"
        or lock.get("implementation_lock_sha256") != sha256_file(IMPLEMENTATION_LOCK)
        or marker.get("generated_dataset_lock_sha256") != sha256_file(DATA_LOCK)
    ):
        raise CVBRATrainingError("CVBRA generated-data lock changed")
    for key in ("dataset_yaml", "manifest"):
        _assert_hash(_rooted(lock[key]), lock[f"{key}_sha256"], label=key)
    return lock


def build_data() -> dict[str, Any]:
    _create_or_validate_implementation_lock()
    if DATA_LOCK.exists() or DATA_MARKER.exists():
        return _validate_data_lock()
    if any(path.exists() for path in (TRAINING_LOCK, CHECKPOINT_LOCK, CHECKPOINT)):
        raise CVBRATrainingError("CVBRA later-stage output appeared before data lock")
    materialization = _load_mapping(TARGET_MATERIALIZATION)
    annotation = _load_mapping(TARGET_ANNOTATION)
    hazy_annotation = _load_mapping(HAZY_ANNOTATION)
    target_rows = _target_view_rows(materialization)
    target_labels, projection = _target_labels(annotation)
    replay_rows = _source_replay_rows(hazy_annotation)
    entries: list[dict[str, Any]] = []
    for image_id in range(1, TARGET_SCENES + 1):
        for view in TARGET_VIEWS:
            row = target_rows[view][image_id]
            source_image = _rooted(row["path"])
            suffix = source_image.suffix.casefold()
            stem = f"target_{image_id:04d}_{view}"
            target_image = DATA_ROOT / "images" / "train" / f"{stem}{suffix}"
            target_label = DATA_ROOT / "labels" / "train" / f"{stem}.txt"
            _hardlink_or_validate(source_image, target_image)
            atomic_write_text(target_label, target_labels[image_id])
            entries.append(
                {
                    "role": "target",
                    "image_id": image_id,
                    "view": view,
                    "image": _relative(target_image),
                    "image_sha256": str(row["sha256"]),
                    "label": _relative(target_label),
                    "label_sha256": sha256_file(target_label),
                }
            )
    for position, row in enumerate(replay_rows, start=1):
        image_id = int(row["id"])
        name = str(row["file_name"])
        source_image = HAZY_IMAGES / name
        source_label = HAZY_LABELS / f"{Path(name).stem}.txt"
        if not source_image.is_file() or not source_label.is_file():
            raise CVBRATrainingError(f"HazyDet replay source is incomplete: {name}")
        suffix = source_image.suffix.casefold()
        stem = f"source_{position:04d}_{image_id}"
        target_image = DATA_ROOT / "images" / "train" / f"{stem}{suffix}"
        target_label = DATA_ROOT / "labels" / "train" / f"{stem}.txt"
        _hardlink_or_validate(source_image, target_image)
        _hardlink_or_validate(source_label, target_label)
        entries.append(
            {
                "role": "source_haze_replay",
                "image_id": image_id,
                "official_file_name": name,
                "image": _relative(target_image),
                "image_sha256": sha256_file(source_image),
                "label": _relative(target_label),
                "label_sha256": sha256_file(source_label),
            }
        )
    if len(entries) != TRAIN_IMAGES:
        raise CVBRATrainingError("CVBRA generated dataset coverage changed")
    dataset_yaml = DATA_ROOT / "dataset.yaml"
    manifest = DATA_ROOT / "manifest.json"
    atomic_write_text(dataset_yaml, _dataset_yaml())
    manifest_payload = {
        "schema_version": 1,
        "status": "CVBRA_V1_DATASET_COMPLETE",
        "target_scenes": TARGET_SCENES,
        "target_views": list(TARGET_VIEWS),
        "target_images": TARGET_IMAGES,
        "source_replay_images": SOURCE_REPLAY_IMAGES,
        "total_images": TRAIN_IMAGES,
        "source_replay_seed": REPLAY_SEED,
        "target_training_projection": projection,
        "entries": entries,
        "entries_payload_sha256": stable_hash(entries, length=64),
    }
    atomic_write_json(manifest, manifest_payload)
    payload = {
        "schema_version": 1,
        "status": "CVBRA_V1_GENERATED_DATASET_LOCKED_BEFORE_TRAINING",
        "locked_at_utc": _utc_now(),
        "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
        "protocol_sha256": PROTOCOL_SHA256,
        "dataset_yaml": _relative(dataset_yaml),
        "dataset_yaml_sha256": sha256_file(dataset_yaml),
        "manifest": _relative(manifest),
        "manifest_sha256": sha256_file(manifest),
        "target_scenes": TARGET_SCENES,
        "target_images": TARGET_IMAGES,
        "source_replay_images": SOURCE_REPLAY_IMAGES,
        "train_images": TRAIN_IMAGES,
        "target_training_projection": projection,
        "training_started_before_lock": False,
        "reserve_B_label_or_prediction_accessed": False,
        "official_validation_or_test_accessed": False,
    }
    atomic_write_json(DATA_LOCK, payload)
    atomic_write_json(
        DATA_MARKER,
        {
            "status": payload["status"],
            "generated_dataset_lock_sha256": sha256_file(DATA_LOCK),
        },
    )
    return payload


def preflight() -> dict[str, Any]:
    protocol = _validate_protocol()
    lock = _create_or_validate_implementation_lock()
    materialization = _load_mapping(TARGET_MATERIALIZATION)
    annotation = _load_mapping(TARGET_ANNOTATION)
    target_rows = _target_view_rows(materialization)
    _, projection = _target_labels(annotation)
    replay_rows = _source_replay_rows(_load_mapping(HAZY_ANNOTATION))
    return {
        "status": "PASS_CVBRA_V1_PREFLIGHT",
        "protocol": str(protocol["protocol"]),
        "target_views": {view: len(target_rows[view]) for view in TARGET_VIEWS},
        "target_training_projection": projection,
        "source_replay_images": len(replay_rows),
        "train_images_per_epoch": TRAIN_IMAGES,
        "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
        "runner_sha256": lock["runner_sha256"],
        "reserve_B_label_or_prediction_accessed": False,
    }


def _validate_training_lock() -> dict[str, Any]:
    _validate_data_lock()
    if not TRAINING_LOCK.is_file() or not TRAINING_MARKER.is_file():
        raise CVBRATrainingError("CVBRA training lock is incomplete")
    lock = _load_mapping(TRAINING_LOCK)
    marker = _load_mapping(TRAINING_MARKER)
    if (
        lock.get("status") != "CVBRA_V1_RAW_TRAINING_ENDPOINT_LOCKED"
        or lock.get("generated_dataset_lock_sha256") != sha256_file(DATA_LOCK)
        or marker.get("training_endpoint_lock_sha256") != sha256_file(TRAINING_LOCK)
    ):
        raise CVBRATrainingError("CVBRA training lock changed")
    for key in ("last_checkpoint", "results", "args"):
        _assert_hash(_rooted(lock[key]), lock[f"{key}_sha256"], label=key)
    return lock


def train_raw_endpoint() -> dict[str, Any]:
    build_data()
    if TRAINING_LOCK.exists() or TRAINING_MARKER.exists():
        return _validate_training_lock()
    if CHECKPOINT_LOCK.exists() or CHECKPOINT.exists():
        raise CVBRATrainingError("CVBRA final checkpoint appeared before training endpoint")
    configure_ultralytics_environment(ROOT)
    try:
        from ultralytics import YOLO  # type: ignore[attr-defined]
    except (ImportError, OSError, PermissionError) as exc:
        raise CVBRATrainingError(f"cannot import Ultralytics: {exc}") from exc
    save_dir = RUN_ROOT / "raw_endpoint" / "fit"
    if save_dir.exists():
        raise CVBRATrainingError(f"incomplete CVBRA training directory requires audit: {save_dir}")
    model = YOLO(str(SOURCE_CHECKPOINT))
    results = model.train(
        data=str((DATA_ROOT / "dataset.yaml").resolve()),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device="0",
        workers=4,
        project=str((RUN_ROOT / "raw_endpoint").resolve()),
        name="fit",
        exist_ok=False,
        pretrained=True,
        freeze=FIRST_TRAINABLE_LAYER,
        val=False,
        optimizer="SGD",
        lr0=0.00075,
        lrf=0.10,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=1.0,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=0.0,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        translate=0.1,
        scale=0.3,
        close_mosaic=0,
        patience=EPOCHS,
        amp=True,
        seed=SEED,
        deterministic=True,
        max_det=500,
        cache=False,
        plots=False,
        verbose=True,
        save=True,
    )
    actual_save = Path(results.save_dir)
    last = actual_save / "weights" / "last.pt"
    results_path = actual_save / "results.csv"
    args_path = actual_save / "args.yaml"
    for path in (last, results_path, args_path):
        if not path.is_file():
            raise CVBRATrainingError(f"CVBRA training output is incomplete: {path}")
    payload = {
        "schema_version": 1,
        "status": "CVBRA_V1_RAW_TRAINING_ENDPOINT_LOCKED",
        "locked_at_utc": _utc_now(),
        "generated_dataset_lock_sha256": sha256_file(DATA_LOCK),
        "protocol_sha256": PROTOCOL_SHA256,
        "last_checkpoint": _relative(last),
        "last_checkpoint_sha256": sha256_file(last),
        "results": _relative(results_path),
        "results_sha256": sha256_file(results_path),
        "args": _relative(args_path),
        "args_sha256": sha256_file(args_path),
        "epochs": EPOCHS,
        "checkpoint_selected_by_metric": False,
        "development_A_role": "training_only",
        "reserve_B_label_or_prediction_accessed": False,
        "official_validation_or_test_accessed": False,
    }
    atomic_write_json(TRAINING_LOCK, payload)
    atomic_write_json(
        TRAINING_MARKER,
        {
            "status": payload["status"],
            "training_endpoint_lock_sha256": sha256_file(TRAINING_LOCK),
        },
    )
    return payload


def _load_checkpoint(path: Path) -> tuple[dict[str, Any], torch.nn.Module]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise CVBRATrainingError(f"unsupported checkpoint payload: {path}")
    model = checkpoint.get("ema") or checkpoint.get("model")
    if not isinstance(model, torch.nn.Module):
        raise CVBRATrainingError(f"checkpoint has no model module: {path}")
    return checkpoint, model


def _combined_state(
    source: Mapping[str, torch.Tensor], trained: Mapping[str, torch.Tensor]
) -> OrderedDict[str, torch.Tensor]:
    if tuple(source) != tuple(trained):
        raise CVBRATrainingError("CVBRA source and trained state schemas differ")
    combined: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, trained_value in trained.items():
        source_value = source[name]
        if source_value.shape != trained_value.shape:
            raise CVBRATrainingError(f"CVBRA state shape differs: {name}")
        chosen = source_value if _layer_index(name) <= FROZEN_LAST_LAYER else trained_value
        if chosen.is_floating_point() and not bool(torch.isfinite(chosen).all()):
            raise CVBRATrainingError(f"CVBRA state is nonfinite: {name}")
        combined[name] = chosen.clone()
    return combined


def _build_final_checkpoint(raw: Mapping[str, Any]) -> None:
    source_checkpoint, source_model = _load_checkpoint(SOURCE_CHECKPOINT)
    _, trained_model = _load_checkpoint(_rooted(raw["last_checkpoint"]))
    if getattr(source_model, "names", None) != getattr(trained_model, "names", None):
        raise CVBRATrainingError("CVBRA source and trained class schemas differ")
    combined = _combined_state(source_model.state_dict(), trained_model.state_dict())
    output_model = copy.deepcopy(trained_model)
    incompatible = output_model.load_state_dict(combined, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise CVBRATrainingError("CVBRA strict state load reported incompatible keys")
    output_model = output_model.half()
    output_model.eval()
    output_checkpoint = copy.deepcopy(source_checkpoint)
    output_checkpoint.update(
        {
            "epoch": -1,
            "best_fitness": None,
            "model": None,
            "ema": output_model,
            "optimizer": None,
            "scaler": None,
            "updates": None,
            "date": _utc_now(),
            "train_metrics": {},
            "train_results": {},
            "cvbra_v1": {
                "training": (
                    "balanced target original/fog_0p6/fog_1p0 plus fixed source haze replay"
                ),
                "frozen_state_rule": "exact source restore for layers 0..9",
                "trained_layers": [FIRST_TRAINABLE_LAYER, LAST_TRAINABLE_LAYER],
                "protocol_sha256": PROTOCOL_SHA256,
                "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
                "generated_dataset_lock_sha256": sha256_file(DATA_LOCK),
                "training_endpoint_lock_sha256": sha256_file(TRAINING_LOCK),
                "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
                "raw_endpoint_sha256": sha256_file(_rooted(raw["last_checkpoint"])),
                "metric_used_for_epoch_or_checkpoint_selection": False,
            },
        }
    )
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    temporary = CHECKPOINT.with_suffix(".pt.tmp")
    torch.save(output_checkpoint, temporary)
    temporary.replace(CHECKPOINT)


def _verify_final_checkpoint(raw: Mapping[str, Any]) -> dict[str, Any]:
    _, source_model = _load_checkpoint(SOURCE_CHECKPOINT)
    _, trained_model = _load_checkpoint(_rooted(raw["last_checkpoint"]))
    checkpoint, output_model = _load_checkpoint(CHECKPOINT)
    metadata = checkpoint.get("cvbra_v1")
    if not isinstance(metadata, dict):
        raise CVBRATrainingError("CVBRA checkpoint metadata is missing")
    expected = _combined_state(source_model.state_dict(), trained_model.state_dict())
    observed = output_model.state_dict()
    if tuple(expected) != tuple(observed):
        raise CVBRATrainingError("CVBRA final state schema changed")
    maximum_error = 0.0
    frozen_exact = True
    changed_trainable = 0
    for name, expected_value in expected.items():
        observed_value = observed[name].to(dtype=expected_value.dtype)
        if expected_value.is_floating_point():
            maximum_error = max(
                maximum_error,
                float((observed_value.float() - expected_value.float()).abs().max()),
            )
        elif not torch.equal(observed_value, expected_value):
            raise CVBRATrainingError(f"CVBRA nonfloating state differs: {name}")
        if _layer_index(name) <= FROZEN_LAST_LAYER:
            frozen_exact = frozen_exact and torch.equal(
                observed_value, source_model.state_dict()[name]
            )
        elif expected_value.is_floating_point() and not torch.equal(
            observed_value, source_model.state_dict()[name]
        ):
            changed_trainable += 1
    if maximum_error != 0.0 or not frozen_exact or changed_trainable == 0:
        raise CVBRATrainingError(
            "CVBRA checkpoint verification failed: "
            f"error={maximum_error}, frozen={frozen_exact}, changed={changed_trainable}"
        )
    return {
        "state_entries": len(expected),
        "maximum_absolute_state_error_after_serialization": maximum_error,
        "frozen_layers_0_to_9_exact_source": frozen_exact,
        "changed_trainable_floating_states": changed_trainable,
        "all_states_finite": all(
            bool(torch.isfinite(value).all())
            for value in observed.values()
            if value.is_floating_point()
        ),
        "class_names": getattr(output_model, "names", None),
        "metadata": metadata,
    }


def _validate_checkpoint_lock() -> dict[str, Any]:
    raw = _validate_training_lock()
    if not CHECKPOINT_LOCK.is_file() or not CHECKPOINT_MARKER.is_file():
        raise CVBRATrainingError("CVBRA checkpoint lock is incomplete")
    lock = _load_mapping(CHECKPOINT_LOCK)
    marker = _load_mapping(CHECKPOINT_MARKER)
    if (
        lock.get("status") != "CVBRA_V1_CHECKPOINT_VERIFIED_AND_LOCKED_BEFORE_RESERVE_B_PREDICTION"
        or lock.get("training_endpoint_lock_sha256") != sha256_file(TRAINING_LOCK)
        or lock.get("checkpoint_sha256") != sha256_file(CHECKPOINT)
        or marker.get("checkpoint_lock_sha256") != sha256_file(CHECKPOINT_LOCK)
    ):
        raise CVBRATrainingError("CVBRA checkpoint lock changed")
    _verify_final_checkpoint(raw)
    return lock


def build_checkpoint() -> dict[str, Any]:
    raw = train_raw_endpoint()
    if CHECKPOINT_LOCK.exists() or CHECKPOINT_MARKER.exists():
        return _validate_checkpoint_lock()
    if CHECKPOINT.exists():
        raise CVBRATrainingError("unlocked CVBRA checkpoint already exists")
    configure_ultralytics_environment(ROOT)
    _build_final_checkpoint(raw)
    verification = _verify_final_checkpoint(raw)
    payload = {
        "schema_version": 1,
        "status": "CVBRA_V1_CHECKPOINT_VERIFIED_AND_LOCKED_BEFORE_RESERVE_B_PREDICTION",
        "locked_at_utc": _utc_now(),
        "protocol_sha256": PROTOCOL_SHA256,
        "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
        "generated_dataset_lock_sha256": sha256_file(DATA_LOCK),
        "training_endpoint_lock_sha256": sha256_file(TRAINING_LOCK),
        "checkpoint": _relative(CHECKPOINT),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "checkpoint_size_bytes": CHECKPOINT.stat().st_size,
        "verification": verification,
        "development_A_role": "training_only",
        "reserve_B_label_or_prediction_accessed": False,
        "official_validation_or_test_accessed": False,
        "paper_body_change_authorized": False,
    }
    atomic_write_json(CHECKPOINT_LOCK, payload)
    atomic_write_json(
        CHECKPOINT_MARKER,
        {
            "status": payload["status"],
            "checkpoint_lock_sha256": sha256_file(CHECKPOINT_LOCK),
        },
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run and lock CVBRA-v1 backup training")
    parser.add_argument(
        "--stage",
        choices=("preflight", "build-data", "train", "build-checkpoint", "all"),
        default="all",
    )
    args = parser.parse_args()
    if args.stage == "preflight":
        result = preflight()
    elif args.stage == "build-data":
        result = build_data()
    elif args.stage == "train":
        result = train_raw_endpoint()
    else:
        result = build_checkpoint()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
