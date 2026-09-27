from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import platform
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

import torch
from PIL import Image

from buse_uav.detectors.ultralytics_adapter import (
    UltralyticsDetector,
    configure_ultralytics_environment,
)
from buse_uav.evaluation.coco import evaluate_coco, write_coco_predictions
from buse_uav.schemas import DetectionBatch, ImageRecord
from buse_uav.utils.hashing import sha256_file, stable_hash
from buse_uav.utils.io import atomic_write_json, atomic_write_text

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "experiment" / "cvbra_v1_training_seed_robustness.yaml"
PROTOCOL_SHA256 = "59316cd671ade040ba48d2c24ea374a25530e3820d0c850388734d44d00a7a6f"
SOURCE_CHECKPOINT = ROOT / "weights" / "hazydet" / "yolo11n_best.pt"
SOURCE_CHECKPOINT_SHA256 = "16ad9de39a1ff86a1826eb5cda680166196ff33ecc5d3dcb297070a166e42430"
PRIMARY_CHECKPOINT = ROOT / "runs" / "cvbra_v1" / "yolo11n" / "cvbra_v1.pt"
PRIMARY_CHECKPOINT_SHA256 = "d44f0926696e93b5f2e0ec5c9201f1e6360c43d4d40fccc68646e1ced633bd42"
DATASET_YAML = ROOT / "data" / "processed" / "cvbra_v1" / "dataset.yaml"
DATASET_YAML_SHA256 = "567ab0bb75434e3c735f8bda6b2aaae2b1c81a4ed6043c215c87900ec9c039d7"
DATASET_MANIFEST = ROOT / "data" / "processed" / "cvbra_v1" / "manifest.json"
DATASET_MANIFEST_SHA256 = "4d802e360563d5d8fd85be4a62b1a6c25930df07f78dc31fc57330c10b2325f1"
DATA_LOCK = ROOT / "reports" / "development" / "cvbra_v1" / "generated_dataset_lock.json"
DATA_LOCK_SHA256 = "32ebc263674ef9b4fd123642e43b978cebcb04ec53f4eb9eef12c7e624fcbe4d"

VALIDATION_ROOT = ROOT / "reports" / "development" / "cvbra_v1" / "uav_obb_official_validation"
MATERIALIZATION = VALIDATION_ROOT / "view_materialization_lock.json"
MATERIALIZATION_SHA256 = "9f6c9db26806679b6853316156785a9b475afe3a4a24fbda1c8ef2725ebbc07d"
CONVERSION_LOCK = VALIDATION_ROOT / "annotations" / "conversion_lock.json"
CONVERSION_LOCK_SHA256 = "aee05f6635fd4d9f56cce016a8cfd79f9a91da966a6284278865ffcbcf59e179"
TARGET_POINT_REPORT = VALIDATION_ROOT / "evaluation" / "point_report.json"
TARGET_POINT_REPORT_SHA256 = "138c523f8ef88874532d1884497aa333f3e3d76200ece367f5b0b337e1bc0b5a"

HAZY_MANIFEST = ROOT / "data" / "manifests" / "hazydet_val_manifest.json"
HAZY_MANIFEST_SHA256 = "e8e5ea234dd098348a1906c8aff93d0d773d6ed1c828c09ee98d7cca6b60f916"
HAZY_ANNOTATION = ROOT / "data" / "raw" / "HazyDet" / "val" / "val_coco.json"
HAZY_ANNOTATION_SHA256 = "2b2e39f7812631dfb4f3f0fbe1e743b65ca873151ca92d8e24ddbf0e9feacb9a"
HAZY_IMAGE_ROOT = ROOT / "data" / "raw" / "HazyDet"
HAZY_REPORT = (
    ROOT
    / "reports"
    / "development"
    / "cvbra_v1_matched_baselines"
    / "hazydet_source_retention_v4"
    / "source_retention_report.json"
)
HAZY_REPORT_SHA256 = "8d4453fb07ef8cf2d027a05b2ae974f8e7bb7190d0836d2145c2ebdb79488bbd"

OUTPUT = ROOT / "reports" / "development" / "cvbra_v1_training_seed_robustness"
RUN_ROOT = ROOT / "runs" / "cvbra_v1_training_seed_robustness"
IMPLEMENTATION_LOCK = OUTPUT / "implementation_lock.json"
IMPLEMENTATION_MARKER = OUTPUT / "IMPLEMENTATION_LOCKED"
PREDICTION_LOCK = OUTPUT / "joint_prediction_lock.json"
PREDICTION_MARKER = OUTPUT / "PREDICTIONS_LOCKED"
METRICS = OUTPUT / "seed_metrics.csv"
REPORT = OUTPUT / "seed_robustness_report.json"
COMPLETE = OUTPUT / "SEED_ROBUSTNESS_COMPLETE"

PRIMARY_SEED = 42
ADDITIONAL_SEEDS = (27182, 31415)
ALL_SEEDS = (PRIMARY_SEED, *ADDITIONAL_SEEDS)
EPOCHS = 8
IMGSZ = 1280
BATCH = 2
FIRST_TRAINABLE_LAYER = 10
FROZEN_LAST_LAYER = 9
WARMUP_IMAGES = 16
CHUNK_SIZE = 8
PROBE_CONF = 0.08
PUBLISH_CONF = 0.25
NMS_IOU = 0.70
MAX_DET = 500
CLASS_NAMES = ("car", "truck", "bus")
METRIC_KEYS = ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large")
TARGET_VIEWS = ("original", "fog_1p0")
TARGET_CATEGORY_ID_BY_CLASS = {0: 1, 1: 2, 2: 3}
HAZY_CATEGORY_ID_BY_CLASS = {0: 0, 1: 1, 2: 2}


class SeedRobustnessError(RuntimeError):
    """Raised when the frozen CVBRA-v1 seed audit cannot fail closed."""


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
        raise SeedRobustnessError(f"cannot parse mapping {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SeedRobustnessError(f"expected mapping: {path}")
    return value


def _assert_hash(path: Path, expected: str, *, label: str) -> None:
    if not path.is_file() or sha256_file(path) != expected:
        raise SeedRobustnessError(f"locked {label} changed: {path}")


def _layer_index(name: str) -> int:
    parts = name.split(".", 2)
    if len(parts) < 3 or parts[0] != "model":
        raise SeedRobustnessError(f"state has no YOLO layer index: {name}")
    try:
        return int(parts[1])
    except ValueError as exc:
        raise SeedRobustnessError(f"invalid YOLO layer index: {name}") from exc


def _load_checkpoint(path: Path) -> tuple[dict[str, Any], torch.nn.Module]:
    configure_ultralytics_environment(ROOT)
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise SeedRobustnessError(f"unsupported checkpoint: {path}")
    model = value.get("ema") or value.get("model")
    if not isinstance(model, torch.nn.Module):
        raise SeedRobustnessError(f"checkpoint has no model module: {path}")
    return value, model


def combined_state(
    source: Mapping[str, torch.Tensor], trained: Mapping[str, torch.Tensor]
) -> OrderedDict[str, torch.Tensor]:
    """Restore frozen layers exactly and retain trained layers 10 onward."""
    if tuple(source) != tuple(trained):
        raise SeedRobustnessError("source and trained state schemas differ")
    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, trained_value in trained.items():
        source_value = source[name]
        if source_value.shape != trained_value.shape:
            raise SeedRobustnessError(f"state shape differs: {name}")
        chosen = source_value if _layer_index(name) <= FROZEN_LAST_LAYER else trained_value
        if chosen.is_floating_point() and not bool(torch.isfinite(chosen).all()):
            raise SeedRobustnessError(f"nonfinite state: {name}")
        result[name] = chosen.clone()
    return result


def _training_lock(seed: int) -> Path:
    return OUTPUT / "training_locks" / f"seed_{seed}.json"


def _training_marker(seed: int) -> Path:
    return OUTPUT / "training_locks" / f"seed_{seed}.LOCKED"


def _checkpoint(seed: int) -> Path:
    return RUN_ROOT / f"seed_{seed}" / f"cvbra_v1_seed_{seed}.pt"


def _checkpoint_lock(seed: int) -> Path:
    return OUTPUT / "checkpoint_locks" / f"seed_{seed}.json"


def _checkpoint_marker(seed: int) -> Path:
    return OUTPUT / "checkpoint_locks" / f"seed_{seed}.LOCKED"


def _raw_fit(seed: int) -> Path:
    return RUN_ROOT / f"seed_{seed}" / "raw_endpoint" / "fit"


def _cuda_identity() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise SeedRobustnessError("seed robustness audit requires CUDA")
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


def _validate_fixed_inputs() -> None:
    for path, digest, label in (
        (PROTOCOL, PROTOCOL_SHA256, "seed protocol"),
        (SOURCE_CHECKPOINT, SOURCE_CHECKPOINT_SHA256, "source checkpoint"),
        (PRIMARY_CHECKPOINT, PRIMARY_CHECKPOINT_SHA256, "primary checkpoint"),
        (DATASET_YAML, DATASET_YAML_SHA256, "dataset YAML"),
        (DATASET_MANIFEST, DATASET_MANIFEST_SHA256, "dataset manifest"),
        (DATA_LOCK, DATA_LOCK_SHA256, "generated-data lock"),
        (MATERIALIZATION, MATERIALIZATION_SHA256, "target materialization"),
        (CONVERSION_LOCK, CONVERSION_LOCK_SHA256, "target conversion lock"),
        (TARGET_POINT_REPORT, TARGET_POINT_REPORT_SHA256, "target point report"),
        (HAZY_MANIFEST, HAZY_MANIFEST_SHA256, "HazyDet manifest"),
        (HAZY_ANNOTATION, HAZY_ANNOTATION_SHA256, "HazyDet annotation"),
        (HAZY_REPORT, HAZY_REPORT_SHA256, "HazyDet source-retention report"),
    ):
        _assert_hash(path, digest, label=label)
    data_lock = _load_mapping(DATA_LOCK)
    conversion = _load_mapping(CONVERSION_LOCK)
    if (
        data_lock.get("train_images") != 3600
        or data_lock.get("dataset_yaml_sha256") != DATASET_YAML_SHA256
        or data_lock.get("manifest_sha256") != DATASET_MANIFEST_SHA256
        or conversion.get("status")
        != "CVBRA_UAV_OBB_OFFICIAL_VALIDATION_LABELS_CONVERTED_AND_LOCKED"
        or conversion.get("official_test_content_accessed") is not False
    ):
        raise SeedRobustnessError("fixed seed-audit input contract changed")


def _validate_existing_implementation_lock() -> dict[str, Any]:
    if not IMPLEMENTATION_LOCK.is_file() or not IMPLEMENTATION_MARKER.is_file():
        raise SeedRobustnessError("seed implementation lock is incomplete")
    lock = _load_mapping(IMPLEMENTATION_LOCK)
    marker = _load_mapping(IMPLEMENTATION_MARKER)
    if (
        lock.get("runner_sha256") != sha256_file(Path(__file__))
        or lock.get("protocol_sha256") != PROTOCOL_SHA256
        or marker.get("implementation_lock_sha256") != sha256_file(IMPLEMENTATION_LOCK)
    ):
        raise SeedRobustnessError("seed implementation lock changed")
    return lock


def _target_records(*, verify_hashes: bool) -> tuple[dict[str, tuple[ImageRecord, ...]], list[int]]:
    materialization = _load_mapping(MATERIALIZATION)
    clean = materialization.get("clean_rows")
    fog = materialization.get("fog_rows")
    primary_ids_raw = materialization.get("primary_image_ids")
    if (
        not isinstance(clean, list)
        or not isinstance(fog, list)
        or not isinstance(primary_ids_raw, list)
    ):
        raise SeedRobustnessError("target materialization rows are incomplete")
    registry: dict[str, list[dict[str, Any]]] = {view: [] for view in TARGET_VIEWS}
    registry["original"] = [row for row in clean if isinstance(row, dict)]
    registry["fog_1p0"] = [
        row for row in fog if isinstance(row, dict) and row.get("view") == "fog_1p0"
    ]
    records_by_view: dict[str, tuple[ImageRecord, ...]] = {}
    for view, rows in registry.items():
        if len(rows) != 218:
            raise SeedRobustnessError(f"target {view} coverage changed")
        records: list[ImageRecord] = []
        for row in sorted(rows, key=lambda item: int(item["image_id"])):
            path = _rooted(row["path"])
            if not path.is_file() or (
                verify_hashes and sha256_file(path) != str(row["sha256"])
            ):
                raise SeedRobustnessError(f"target image changed: {path}")
            records.append(
                ImageRecord(
                    image_id=int(row["image_id"]),
                    path=str(path.resolve()),
                    width=int(row["width"]),
                    height=int(row["height"]),
                )
            )
        records_by_view[view] = tuple(records)
    primary_ids = [int(value) for value in primary_ids_raw]
    if len(primary_ids) != 167 or len(set(primary_ids)) != 167:
        raise SeedRobustnessError("target primary image scope changed")
    return records_by_view, primary_ids


def _hazy_records(*, verify_hashes: bool) -> tuple[ImageRecord, ...]:
    manifest = _load_mapping(HAZY_MANIFEST)
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list):
        raise SeedRobustnessError("HazyDet manifest has no files")
    files = [
        row
        for row in raw_files
        if isinstance(row, dict)
        and str(row.get("path", "")).replace("\\", "/").startswith("val/hazy_images/")
    ]
    if manifest.get("dataset") != "hazydet" or len(files) != 1000:
        raise SeedRobustnessError("HazyDet validation coverage changed")
    records: list[ImageRecord] = []
    for row in files:
        path = HAZY_IMAGE_ROOT / str(row["path"])
        if not path.is_file() or (verify_hashes and sha256_file(path) != str(row["sha256"])):
            raise SeedRobustnessError(f"HazyDet validation image changed: {path}")
        try:
            with Image.open(path) as image:
                width, height = image.size
        except OSError as exc:
            raise SeedRobustnessError(f"cannot inspect HazyDet image: {path}") from exc
        records.append(
            ImageRecord(
                image_id=int(path.stem),
                path=str(path.resolve()),
                width=int(width),
                height=int(height),
            )
        )
    records.sort(key=lambda record: int(record.image_id))
    if len(records) != 1000 or len({record.image_id for record in records}) != 1000:
        raise SeedRobustnessError("HazyDet image identities changed")
    return tuple(records)


def preflight() -> dict[str, Any]:
    _validate_fixed_inputs()
    target, primary_ids = _target_records(verify_hashes=True)
    hazy = _hazy_records(verify_hashes=True)
    if IMPLEMENTATION_LOCK.exists() or IMPLEMENTATION_MARKER.exists():
        return _validate_existing_implementation_lock()
    later = [
        *(_training_lock(seed) for seed in ADDITIONAL_SEEDS),
        *(_checkpoint(seed) for seed in ADDITIONAL_SEEDS),
        PREDICTION_LOCK,
        METRICS,
        REPORT,
    ]
    if any(path.exists() for path in later):
        raise SeedRobustnessError("seed-audit output appeared before implementation lock")
    payload = {
        "schema_version": 1,
        "status": "CVBRA_V1_SEED_ROBUSTNESS_IMPLEMENTATION_LOCKED",
        "locked_at_utc": _utc_now(),
        "protocol": _relative(PROTOCOL),
        "protocol_sha256": PROTOCOL_SHA256,
        "runner": _relative(Path(__file__)),
        "runner_sha256": sha256_file(Path(__file__)),
        "python": platform.python_version(),
        "cuda_identity": _cuda_identity(),
        "primary_seed": PRIMARY_SEED,
        "additional_seeds": list(ADDITIONAL_SEEDS),
        "dataset_lock_sha256": DATA_LOCK_SHA256,
        "target_views": {view: len(records) for view, records in target.items()},
        "target_primary_ids_sha256": stable_hash(primary_ids, length=64),
        "HazyDet_images": len(hazy),
        "method_or_hyperparameter_selection": False,
        "validation_labels_previously_accessed": True,
        "test_content_accessed": False,
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


def _validate_checkpoint_lock(seed: int) -> dict[str, Any]:
    if seed not in ADDITIONAL_SEEDS:
        raise SeedRobustnessError(f"unregistered additional seed: {seed}")
    lock_path = _checkpoint_lock(seed)
    marker_path = _checkpoint_marker(seed)
    if not lock_path.is_file() or not marker_path.is_file():
        raise SeedRobustnessError(f"checkpoint lock is incomplete for seed {seed}")
    lock = _load_mapping(lock_path)
    marker = _load_mapping(marker_path)
    checkpoint = _checkpoint(seed)
    if (
        lock.get("seed") != seed
        or lock.get("checkpoint_sha256") != sha256_file(checkpoint)
        or marker.get("checkpoint_lock_sha256") != sha256_file(lock_path)
        or lock.get("implementation_lock_sha256") != sha256_file(IMPLEMENTATION_LOCK)
    ):
        raise SeedRobustnessError(f"checkpoint lock changed for seed {seed}")
    return lock


def _build_final_checkpoint(seed: int, raw_checkpoint: Path) -> dict[str, Any]:
    source_payload, source_model = _load_checkpoint(SOURCE_CHECKPOINT)
    _, trained_model = _load_checkpoint(raw_checkpoint)
    if getattr(source_model, "names", None) != getattr(trained_model, "names", None):
        raise SeedRobustnessError("source and trained class schemas differ")
    state = combined_state(source_model.state_dict(), trained_model.state_dict())
    output_model = copy.deepcopy(trained_model)
    incompatible = output_model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise SeedRobustnessError("strict final-state load reported incompatibilities")
    output_model = output_model.half().eval()
    output_payload = copy.deepcopy(source_payload)
    output_payload.update(
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
            "cvbra_v1_seed_robustness": {
                "seed": seed,
                "epochs": EPOCHS,
                "endpoint": "last_epoch",
                "frozen_state_rule": "exact source restore for layers 0..9",
                "trained_layers": [FIRST_TRAINABLE_LAYER, 23],
                "protocol_sha256": PROTOCOL_SHA256,
                "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
                "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
                "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
                "raw_endpoint_sha256": sha256_file(raw_checkpoint),
                "metric_used_for_selection": False,
            },
        }
    )
    checkpoint = _checkpoint(seed)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix(".pt.tmp")
    torch.save(output_payload, temporary)
    temporary.replace(checkpoint)
    _, observed_model = _load_checkpoint(checkpoint)
    observed = observed_model.state_dict()
    source_state = source_model.state_dict()
    frozen_exact = all(
        torch.equal(observed[name].to(dtype=source_state[name].dtype), source_state[name])
        for name in observed
        if _layer_index(name) <= FROZEN_LAST_LAYER
    )
    changed_trainable = sum(
        1
        for name, value in observed.items()
        if _layer_index(name) >= FIRST_TRAINABLE_LAYER
        and value.is_floating_point()
        and not torch.equal(value.to(dtype=source_state[name].dtype), source_state[name])
    )
    if not frozen_exact or changed_trainable == 0:
        raise SeedRobustnessError("final checkpoint state verification failed")
    return {
        "frozen_layers_0_to_9_exact_source": frozen_exact,
        "changed_trainable_floating_states": changed_trainable,
        "state_entries": len(observed),
        "parameters": sum(int(parameter.numel()) for parameter in observed_model.parameters()),
    }


def train_seed(seed: int) -> dict[str, Any]:
    preflight()
    if seed not in ADDITIONAL_SEEDS:
        raise SeedRobustnessError(f"seed is not registered: {seed}")
    if _checkpoint_lock(seed).exists() or _checkpoint_marker(seed).exists():
        return _validate_checkpoint_lock(seed)
    checkpoint = _checkpoint(seed)
    raw_lock = _training_lock(seed)
    raw_marker = _training_marker(seed)
    fit = _raw_fit(seed)
    if any(path.exists() for path in (checkpoint, raw_lock, raw_marker)) or fit.exists():
        raise SeedRobustnessError(f"partial training output requires audit for seed {seed}")
    configure_ultralytics_environment(ROOT)
    try:
        from ultralytics import YOLO  # type: ignore[attr-defined]
    except (ImportError, OSError, PermissionError) as exc:
        raise SeedRobustnessError(f"cannot import Ultralytics: {exc}") from exc
    model = YOLO(str(SOURCE_CHECKPOINT))
    results = model.train(
        data=str(DATASET_YAML.resolve()),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device="0",
        workers=4,
        project=str(fit.parent.resolve()),
        name=fit.name,
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
        seed=seed,
        deterministic=True,
        max_det=MAX_DET,
        cache=False,
        plots=False,
        verbose=False,
        save=True,
    )
    actual = Path(results.save_dir)
    last = actual / "weights" / "last.pt"
    results_csv = actual / "results.csv"
    args_yaml = actual / "args.yaml"
    for path in (last, results_csv, args_yaml):
        if not path.is_file():
            raise SeedRobustnessError(f"training output is incomplete: {path}")
    raw_payload = {
        "schema_version": 1,
        "status": "CVBRA_V1_ADDITIONAL_SEED_RAW_ENDPOINT_LOCKED",
        "locked_at_utc": _utc_now(),
        "seed": seed,
        "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
        "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "last_checkpoint": _relative(last),
        "last_checkpoint_sha256": sha256_file(last),
        "results": _relative(results_csv),
        "results_sha256": sha256_file(results_csv),
        "args": _relative(args_yaml),
        "args_sha256": sha256_file(args_yaml),
        "epochs": EPOCHS,
        "checkpoint_selected_by_metric": False,
        "validation_metric_accessed_during_training": False,
        "test_content_accessed": False,
    }
    atomic_write_json(raw_lock, raw_payload)
    atomic_write_json(
        raw_marker,
        {
            "status": raw_payload["status"],
            "training_lock_sha256": sha256_file(raw_lock),
        },
    )
    verification = _build_final_checkpoint(seed, last)
    checkpoint_payload = {
        "schema_version": 1,
        "status": "CVBRA_V1_ADDITIONAL_SEED_CHECKPOINT_VERIFIED_AND_LOCKED",
        "locked_at_utc": _utc_now(),
        "seed": seed,
        "protocol_sha256": PROTOCOL_SHA256,
        "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
        "training_lock_sha256": sha256_file(raw_lock),
        "checkpoint": _relative(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "verification": verification,
        "validation_metric_used_for_training_or_selection": False,
        "test_content_accessed": False,
        "paper_body_change_authorized": False,
    }
    lock_path = _checkpoint_lock(seed)
    atomic_write_json(lock_path, checkpoint_payload)
    atomic_write_json(
        _checkpoint_marker(seed),
        {
            "status": checkpoint_payload["status"],
            "checkpoint_lock_sha256": sha256_file(lock_path),
        },
    )
    del model
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({"seed_training_complete": seed}), flush=True)
    return checkpoint_payload


def train_all() -> dict[str, Any]:
    locks = {str(seed): train_seed(seed) for seed in ADDITIONAL_SEEDS}
    return {"status": "ALL_ADDITIONAL_SEEDS_TRAINED", "checkpoints": locks}


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


def _prediction_root(seed: int, domain: str, view: str) -> Path:
    return OUTPUT / "predictions" / f"seed_{seed}" / domain / view


def _predict_cell(
    *,
    seed: int,
    domain: str,
    view: str,
    detector: UltralyticsDetector,
    records: Sequence[ImageRecord],
    category_id_by_class: Mapping[int, int],
) -> dict[str, Any]:
    root = _prediction_root(seed, domain, view)
    prediction = root / "predictions.coco.json"
    marker_path = root / "SUCCESS.json"
    if marker_path.exists():
        marker = _load_mapping(marker_path)
        if marker.get("prediction_sha256") != sha256_file(prediction):
            raise SeedRobustnessError(f"prediction changed: {root}")
        return marker
    if root.exists():
        raise SeedRobustnessError(f"partial prediction cell requires audit: {root}")
    batches = detector.predict(
        records,
        imgsz=IMGSZ,
        conf=PROBE_CONF,
        iou=NMS_IOU,
        max_det=MAX_DET,
        fp16=True,
    )
    filtered = _filter_batches(batches)
    write_coco_predictions(
        prediction,
        filtered,
        category_id_by_class=category_id_by_class,
    )
    marker = {
        "schema_version": 1,
        "status": "CVBRA_V1_ADDITIONAL_SEED_PREDICTION_COMPLETE",
        "completed_at_utc": _utc_now(),
        "seed": seed,
        "domain": domain,
        "view": view,
        "images": len(filtered),
        "checkpoint_sha256": sha256_file(_checkpoint(seed)),
        "prediction": _relative(prediction),
        "prediction_sha256": sha256_file(prediction),
        "validation_labels_previously_accessed": True,
        "metrics_accessed_for_prediction": False,
        "test_content_accessed": False,
    }
    atomic_write_json(marker_path, marker)
    return marker


def _validate_prediction_lock() -> dict[str, Any]:
    if not PREDICTION_LOCK.is_file() or not PREDICTION_MARKER.is_file():
        raise SeedRobustnessError("seed prediction lock is incomplete")
    lock = _load_mapping(PREDICTION_LOCK)
    marker = _load_mapping(PREDICTION_MARKER)
    artifacts = lock.get("artifacts")
    if (
        lock.get("implementation_lock_sha256") != sha256_file(IMPLEMENTATION_LOCK)
        or marker.get("joint_prediction_lock_sha256") != sha256_file(PREDICTION_LOCK)
        or not isinstance(artifacts, list)
        or len(artifacts) != len(ADDITIONAL_SEEDS) * 3
    ):
        raise SeedRobustnessError("seed prediction lock changed")
    for row in artifacts:
        if not isinstance(row, dict):
            raise SeedRobustnessError("invalid seed prediction artifact")
        _assert_hash(_rooted(row["prediction"]), str(row["prediction_sha256"]), label="prediction")
    return lock


def infer() -> dict[str, Any]:
    preflight()
    for seed in ADDITIONAL_SEEDS:
        _validate_checkpoint_lock(seed)
    if PREDICTION_LOCK.exists() or PREDICTION_MARKER.exists():
        return _validate_prediction_lock()
    if any(path.exists() for path in (METRICS, REPORT, COMPLETE)):
        raise SeedRobustnessError("seed metrics appeared before prediction lock")
    target, _ = _target_records(verify_hashes=False)
    hazy = _hazy_records(verify_hashes=False)
    artifacts: list[dict[str, Any]] = []
    for seed in ADDITIONAL_SEEDS:
        detector = UltralyticsDetector(
            _checkpoint(seed),
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
                    seed=seed,
                    domain="UAV_OBB_validation",
                    view=view,
                    detector=detector,
                    records=target[view],
                    category_id_by_class=TARGET_CATEGORY_ID_BY_CLASS,
                )
            )
        artifacts.append(
            _predict_cell(
                seed=seed,
                domain="HazyDet_validation",
                view="hazy",
                detector=detector,
                records=hazy,
                category_id_by_class=HAZY_CATEGORY_ID_BY_CLASS,
            )
        )
        del detector
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        print(json.dumps({"seed_inference_complete": seed}), flush=True)
    payload = {
        "schema_version": 1,
        "status": "CVBRA_V1_ADDITIONAL_SEED_PREDICTIONS_LOCKED_BEFORE_NEW_METRICS",
        "locked_at_utc": _utc_now(),
        "protocol_sha256": PROTOCOL_SHA256,
        "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
        "checkpoint_locks": {
            str(seed): sha256_file(_checkpoint_lock(seed)) for seed in ADDITIONAL_SEEDS
        },
        "artifacts": artifacts,
        "validation_labels_previously_accessed": True,
        "new_metrics_accessed_before_lock": False,
        "test_content_accessed": False,
    }
    atomic_write_json(PREDICTION_LOCK, payload)
    atomic_write_json(
        PREDICTION_MARKER,
        {
            "status": payload["status"],
            "joint_prediction_lock_sha256": sha256_file(PREDICTION_LOCK),
        },
    )
    return payload


def _existing_metric_rows() -> tuple[list[dict[str, Any]], dict[str, float]]:
    target = _load_mapping(TARGET_POINT_REPORT)
    hazy = _load_mapping(HAZY_REPORT)
    target_rows = target.get("rows")
    hazy_rows = hazy.get("rows")
    if not isinstance(target_rows, list) or not isinstance(hazy_rows, list):
        raise SeedRobustnessError("existing metric reports are incomplete")
    rows: list[dict[str, Any]] = []
    baselines: dict[str, float] = {}
    for view in TARGET_VIEWS:
        domain = f"target_{view}"
        selected = [
            row
            for row in target_rows
            if isinstance(row, dict)
            and row.get("scope") == "decontaminated_primary"
            and row.get("view") == view
            and row.get("method") == "Identity"
            and row.get("model") in {"source", "CVBRA_v1"}
        ]
        if len(selected) != 2:
            raise SeedRobustnessError(f"existing target rows changed for {view}")
        for row in selected:
            if row["model"] == "source":
                baselines[domain] = float(row["AP"])
            else:
                rows.append(
                    {
                        "domain": domain,
                        "model": "seed_42",
                        "seed": PRIMARY_SEED,
                        **{key: float(row[key]) for key in METRIC_KEYS},
                        "images_evaluated": int(row["images_evaluated"]),
                    }
                )
    selected_hazy = [
        row
        for row in hazy_rows
        if isinstance(row, dict) and row.get("model") in {"source", "CVBRA_v1"}
    ]
    if len(selected_hazy) != 2:
        raise SeedRobustnessError("existing HazyDet seed rows changed")
    for row in selected_hazy:
        if row["model"] == "source":
            baselines["source_HazyDet"] = float(row["AP"])
        else:
            rows.append(
                {
                    "domain": "source_HazyDet",
                    "model": "seed_42",
                    "seed": PRIMARY_SEED,
                    **{key: float(row[key]) for key in METRIC_KEYS},
                    "images_evaluated": int(row["images_evaluated"]),
                }
            )
    return rows, baselines


def summarize_seed_metrics(
    rows: Sequence[Mapping[str, Any]], baselines: Mapping[str, float]
) -> dict[str, Any]:
    domains = ("target_original", "target_fog_1p0", "source_HazyDet")
    variability: dict[str, Any] = {}
    deltas: dict[str, dict[str, float]] = {}
    for domain in domains:
        selected = [row for row in rows if str(row["domain"]) == domain]
        if len(selected) != len(ALL_SEEDS) or {int(row["seed"]) for row in selected} != set(
            ALL_SEEDS
        ):
            raise SeedRobustnessError(f"incomplete seed metrics for {domain}")
        variability[domain] = {}
        for metric in METRIC_KEYS:
            values = [float(row[metric]) for row in selected]
            variability[domain][metric] = {
                "mean": fmean(values),
                "sample_standard_deviation": stdev(values),
                "minimum": min(values),
                "maximum": max(values),
            }
        deltas[domain] = {
            str(row["model"]): float(row["AP"]) - float(baselines[domain]) for row in selected
        }
    checks = {
        "every_seed_target_original_delta_at_least_0p20": min(
            deltas["target_original"].values()
        )
        >= 0.20,
        "every_seed_target_fog_1p0_delta_at_least_0p18": min(
            deltas["target_fog_1p0"].values()
        )
        >= 0.18,
        "target_original_AP_sample_sd_at_most_0p03": variability["target_original"]["AP"][
            "sample_standard_deviation"
        ]
        <= 0.03,
        "target_fog_1p0_AP_sample_sd_at_most_0p03": variability["target_fog_1p0"]["AP"][
            "sample_standard_deviation"
        ]
        <= 0.03,
        "every_seed_HazyDet_AP_at_least_0p44": variability["source_HazyDet"]["AP"][
            "minimum"
        ]
        >= 0.44,
    }
    return {
        "variability": variability,
        "AP_deltas_vs_source_checkpoint": deltas,
        "checks": checks,
        "all_registered_robustness_checks_pass": all(checks.values()),
    }


def _write_metrics(rows: Sequence[Mapping[str, Any]]) -> None:
    fields = ("domain", "model", "seed", *METRIC_KEYS, "images_evaluated")
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row[field] for field in fields})
    atomic_write_text(METRICS, buffer.getvalue())


def _validate_existing_report() -> dict[str, Any]:
    if not REPORT.is_file() or not METRICS.is_file() or not COMPLETE.is_file():
        raise SeedRobustnessError("seed robustness report is incomplete")
    report = _load_mapping(REPORT)
    marker = _load_mapping(COMPLETE)
    if (
        report.get("prediction_lock_sha256") != sha256_file(PREDICTION_LOCK)
        or report.get("metrics_sha256") != sha256_file(METRICS)
        or marker.get("seed_robustness_report_sha256") != sha256_file(REPORT)
    ):
        raise SeedRobustnessError("seed robustness report changed")
    return report


def score() -> dict[str, Any]:
    lock = infer()
    if REPORT.exists() or METRICS.exists() or COMPLETE.exists():
        return _validate_existing_report()
    conversion = _load_mapping(CONVERSION_LOCK)
    target_annotation = _rooted(conversion["annotation"])
    _assert_hash(target_annotation, str(conversion["annotation_sha256"]), label="target annotation")
    _, primary_ids = _target_records(verify_hashes=False)
    hazy_records = _hazy_records(verify_hashes=False)
    rows, baselines = _existing_metric_rows()
    for seed in ADDITIONAL_SEEDS:
        for view in TARGET_VIEWS:
            prediction = _prediction_root(
                seed, "UAV_OBB_validation", view
            ) / "predictions.coco.json"
            metrics = evaluate_coco(
                target_annotation,
                prediction,
                max_det=MAX_DET,
                image_ids=primary_ids,
            )
            rows.append(
                {
                    "domain": f"target_{view}",
                    "model": f"seed_{seed}",
                    "seed": seed,
                    **{key: float(metrics[key]) for key in METRIC_KEYS},
                    "images_evaluated": len(primary_ids),
                }
            )
        hazy_prediction = _prediction_root(
            seed, "HazyDet_validation", "hazy"
        ) / "predictions.coco.json"
        hazy_metrics = evaluate_coco(
            HAZY_ANNOTATION,
            hazy_prediction,
            max_det=MAX_DET,
            image_ids=[record.image_id for record in hazy_records],
        )
        rows.append(
            {
                "domain": "source_HazyDet",
                "model": f"seed_{seed}",
                "seed": seed,
                **{key: float(hazy_metrics[key]) for key in METRIC_KEYS},
                "images_evaluated": len(hazy_records),
            }
        )
    rows.sort(key=lambda row: (str(row["domain"]), int(row["seed"])))
    summary = summarize_seed_metrics(rows, baselines)
    _write_metrics(rows)
    payload = {
        "schema_version": 1,
        "status": "COMPLETE_CVBRA_V1_TRAINING_SEED_ROBUSTNESS_AUDIT",
        "completed_at_utc": _utc_now(),
        "protocol_sha256": PROTOCOL_SHA256,
        "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
        "prediction_lock_sha256": sha256_file(PREDICTION_LOCK),
        "checkpoint_locks": lock["checkpoint_locks"],
        "seeds": list(ALL_SEEDS),
        "baselines": baselines,
        "rows": rows,
        "summary": summary,
        "metrics": _relative(METRICS),
        "metrics_sha256": sha256_file(METRICS),
        "evidence_boundary": {
            "post_freeze_training_variability_audit": True,
            "validation_labels_previously_accessed": True,
            "method_or_checkpoint_reselection": False,
            "independent_confirmation_claim": False,
            "test_content_accessed": False,
        },
        "paper_body_change_authorized": False,
    }
    atomic_write_json(REPORT, payload)
    atomic_write_json(
        COMPLETE,
        {
            "status": payload["status"],
            "seed_robustness_report_sha256": sha256_file(REPORT),
        },
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CVBRA-v1 training-seed robustness audit")
    parser.add_argument(
        "--stage",
        choices=("preflight", "train", "infer", "score", "all"),
        default="all",
    )
    parser.add_argument("--seed", type=int, choices=ADDITIONAL_SEEDS)
    args = parser.parse_args()
    if args.stage == "preflight":
        result = preflight()
    elif args.stage == "train":
        result = train_seed(args.seed) if args.seed is not None else train_all()
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
