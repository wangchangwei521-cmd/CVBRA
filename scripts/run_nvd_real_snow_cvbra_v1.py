from __future__ import annotations

import argparse
import copy
import gc
import io
import json
import math
import time
from collections import OrderedDict, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

import torch
from scripts import prepare_nvd_real_snow_cvbra_v1 as prep

from buse_uav.detectors.ultralytics_adapter import (
    UltralyticsDetector,
    configure_ultralytics_environment,
)
from buse_uav.evaluation.bootstrap import (
    ClusterBootstrapScope,
    paired_coco_ap_cluster_bootstrap_scopes,
)
from buse_uav.evaluation.coco import evaluate_coco, write_coco_predictions
from buse_uav.schemas import ImageRecord
from buse_uav.utils.hashing import sha256_file
from buse_uav.utils.io import atomic_write_json, atomic_write_text

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/development/nvd_real_snow_cvbra_v1"
RUN_ROOT = ROOT / "runs/nvd_real_snow_cvbra_v1"
SOURCE_CHECKPOINT = RUN_ROOT / "source" / "source_yolo11n.pt"
SOURCE_LOCK = OUTPUT / "training_locks/source.json"
PREDICTION_COMPLETE = OUTPUT / "PREDICTIONS_COMPLETE.json"
REPORT = OUTPUT / "real_snow_report.json"
METRICS_CSV = OUTPUT / "metrics.csv"
COMPLETE = OUTPUT / "COMPLETE.json"

METHODS = prep.ADAPTATION_METHODS
SEEDS = (42, 27182, 31415)
SPLITS = ("source_retention", "target_validation", "target_test")
METRICS = ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large")
CLASS_NAMES = ("car",)
CATEGORY_MAP = {0: 1}
EPOCHS = 8
IMGSZ = 1280
BATCH = 2
LOW_FLOOR = 0.001
NMS_IOU = 0.70
MAX_DET = 500
FROZEN_LAST_LAYER = 9
FIRST_TRAINABLE_LAYER = 10
T95_DF2 = 4.302652729911275


class NvdExperimentError(RuntimeError):
    """Raised when the registered NVD real-snow experiment cannot fail closed."""


def _checkpoint(method: str, seed: int) -> Path:
    return RUN_ROOT / method / f"seed_{seed}" / f"{method}_seed_{seed}.pt"


def _training_lock(method: str, seed: int) -> Path:
    return OUTPUT / "training_locks" / method / f"seed_{seed}.json"


def _raw_fit(method: str, seed: int) -> Path:
    return RUN_ROOT / method / f"seed_{seed}" / "raw_endpoint" / "fit"


def _prediction(model_key: str, split: str) -> Path:
    return OUTPUT / "predictions" / model_key / f"{split}.json"


def _prediction_lock(model_key: str, split: str) -> Path:
    return OUTPUT / "prediction_locks" / model_key / f"{split}.json"


def _model_key(method: str, seed: int) -> str:
    return f"{method}_seed_{seed}"


def _protocol() -> dict[str, Any]:
    protocol = prep.load_protocol()
    prep.validate_registration()
    return protocol


def _cuda_identity() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise NvdExperimentError("the registered experiment requires CUDA")
    index = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(index)
    return {
        "device": "cuda:0",
        "index": index,
        "name": torch.cuda.get_device_name(index),
        "compute_capability": [int(major), int(minor)],
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
    }


def preflight() -> dict[str, Any]:
    lock = prep.validate_data_lock()
    protocol = _protocol()
    datasets = {
        str(row["dataset"]): int(row["images"])
        for row in lock.get("training_datasets", [])
        if isinstance(row, dict)
    }
    expected = {"source": 900, **{method: 3600 for method in METHODS}}
    if datasets != expected:
        raise NvdExperimentError(f"training dataset budgets changed: {datasets}")
    return {
        "status": "PASS_NVD_REAL_SNOW_PREFLIGHT",
        "registration_sha256": sha256_file(prep.REGISTRATION),
        "data_lock_sha256": sha256_file(prep.DATA_LOCK),
        "datasets": datasets,
        "source_epochs": int(protocol["source_training"]["epochs"]),
        "adaptation_epochs": int(protocol["adaptation"]["epochs"]),
        "seeds": list(SEEDS),
        "device": _cuda_identity(),
        "target_test_predictions_exist": (OUTPUT / "predictions").exists(),
        "uavdt_used": False,
    }


def _load_checkpoint(path: Path) -> tuple[dict[str, Any], torch.nn.Module]:
    configure_ultralytics_environment(ROOT)
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise NvdExperimentError(f"cannot load checkpoint {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise NvdExperimentError(f"unsupported checkpoint mapping: {path}")
    model = value.get("ema") or value.get("model")
    if not isinstance(model, torch.nn.Module):
        raise NvdExperimentError(f"checkpoint has no model module: {path}")
    return value, model


def _layer_index(name: str) -> int:
    parts = name.split(".", 2)
    if len(parts) < 3 or parts[0] != "model":
        raise NvdExperimentError(f"state has no YOLO layer index: {name}")
    try:
        return int(parts[1])
    except ValueError as exc:
        raise NvdExperimentError(f"invalid YOLO layer index: {name}") from exc


def _combined_state(
    source: Mapping[str, torch.Tensor], trained: Mapping[str, torch.Tensor]
) -> OrderedDict[str, torch.Tensor]:
    if tuple(source) != tuple(trained):
        raise NvdExperimentError("source and adapted state schemas differ")
    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, trained_value in trained.items():
        source_value = source[name]
        if source_value.shape != trained_value.shape:
            raise NvdExperimentError(f"state shape changed: {name}")
        selected = source_value if _layer_index(name) <= FROZEN_LAST_LAYER else trained_value
        if selected.is_floating_point() and not bool(torch.isfinite(selected).all()):
            raise NvdExperimentError(f"nonfinite state: {name}")
        result[name] = selected.clone()
    return result


def _save_endpoint(
    raw_path: Path,
    output_path: Path,
    *,
    method: str,
    seed: int,
    source_path: Path | None,
) -> dict[str, Any]:
    raw_checkpoint, raw_model = _load_checkpoint(raw_path)
    if source_path is None:
        output_model = copy.deepcopy(raw_model)
        frozen_exact = None
        changed_trainable = None
    else:
        _, source_model = _load_checkpoint(source_path)
        if getattr(source_model, "names", None) != getattr(raw_model, "names", None):
            raise NvdExperimentError("source and adapted class schemas differ")
        if method == "STF":
            output_model = copy.deepcopy(raw_model)
            frozen_exact = None
            changed_trainable = None
        else:
            combined = _combined_state(source_model.state_dict(), raw_model.state_dict())
            output_model = copy.deepcopy(raw_model)
            incompatible = output_model.load_state_dict(combined, strict=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise NvdExperimentError("strict frozen-state restoration failed")
            observed = output_model.state_dict()
            frozen_exact = all(
                torch.equal(observed[name], source_model.state_dict()[name])
                for name in observed
                if _layer_index(name) <= FROZEN_LAST_LAYER
            )
            changed_trainable = sum(
                not torch.equal(observed[name], source_model.state_dict()[name])
                for name in observed
                if _layer_index(name) >= FIRST_TRAINABLE_LAYER
                and observed[name].is_floating_point()
            )
            if not frozen_exact or changed_trainable == 0:
                raise NvdExperimentError("restored endpoint did not satisfy the L10 contract")
    output_model = output_model.half()
    output_model.eval()
    endpoint = copy.deepcopy(raw_checkpoint)
    endpoint.update(
        {
            "epoch": -1,
            "best_fitness": None,
            "model": None,
            "ema": output_model,
            "optimizer": None,
            "scaler": None,
            "updates": None,
            "date": prep.utc_now(),
            "train_metrics": {},
            "train_results": {},
            "nvd_real_snow_cvbra_v1": {
                "method": method,
                "seed": seed,
                "endpoint": "fixed_last_epoch",
                "source_checkpoint_sha256": (
                    sha256_file(source_path) if source_path is not None else None
                ),
                "raw_endpoint_sha256": sha256_file(raw_path),
                "registration_sha256": sha256_file(prep.REGISTRATION),
                "data_lock_sha256": sha256_file(prep.DATA_LOCK),
                "frozen_layers_0_to_9_exact_source": frozen_exact,
                "checkpoint_selected_by_metric": False,
            },
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".pt.tmp")
    torch.save(endpoint, temporary)
    temporary.replace(output_path)
    _, verified = _load_checkpoint(output_path)
    if tuple(getattr(verified, "names", {}).values()) != CLASS_NAMES:
        raise NvdExperimentError("saved endpoint class mapping changed")
    return {
        "checkpoint": prep.relative(output_path),
        "checkpoint_sha256": sha256_file(output_path),
        "frozen_layers_0_to_9_exact_source": frozen_exact,
        "changed_trainable_floating_states": changed_trainable,
    }


def _run_training(
    *,
    initialization: Path,
    dataset: Path,
    epochs: int,
    seed: int,
    fit: Path,
    freeze: int | None,
) -> tuple[Path, Path, Path, float]:
    configure_ultralytics_environment(ROOT)
    try:
        from ultralytics import YOLO  # type: ignore[attr-defined]
    except (ImportError, OSError, PermissionError) as exc:
        raise NvdExperimentError(f"cannot import Ultralytics: {exc}") from exc
    if fit.exists():
        raise NvdExperimentError(f"partial training output requires audit: {fit}")
    model = YOLO(str(initialization))
    started = time.perf_counter()
    results = model.train(
        data=str(dataset.resolve()),
        epochs=epochs,
        imgsz=IMGSZ,
        batch=BATCH,
        device="0",
        workers=4,
        project=str(fit.parent.resolve()),
        name=fit.name,
        exist_ok=False,
        pretrained=True,
        freeze=freeze,
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
        patience=epochs,
        amp=True,
        seed=seed,
        deterministic=True,
        max_det=MAX_DET,
        cache=False,
        plots=False,
        verbose=True,
        save=True,
    )
    elapsed = time.perf_counter() - started
    actual = Path(results.save_dir)
    last = actual / "weights/last.pt"
    results_csv = actual / "results.csv"
    args_yaml = actual / "args.yaml"
    for path in (last, results_csv, args_yaml):
        if not path.is_file():
            raise NvdExperimentError(f"training output is incomplete: {path}")
    del model, results
    gc.collect()
    torch.cuda.empty_cache()
    return last, results_csv, args_yaml, elapsed


def train_source() -> dict[str, Any]:
    preflight()
    protocol = _protocol()
    if SOURCE_LOCK.exists() and SOURCE_CHECKPOINT.exists():
        lock = prep.load_json(SOURCE_LOCK)
        if lock.get("checkpoint_sha256") != sha256_file(SOURCE_CHECKPOINT):
            raise NvdExperimentError("source checkpoint changed")
        return lock
    if SOURCE_LOCK.exists() or SOURCE_CHECKPOINT.exists():
        raise NvdExperimentError("partial source-training endpoint requires audit")
    source = protocol["source_training"]
    initialization = ROOT / str(source["initialization"])
    fit = RUN_ROOT / "source/raw_endpoint/fit"
    last, results_csv, args_yaml, elapsed = _run_training(
        initialization=initialization,
        dataset=prep.DATA_ROOT / "datasets/source/dataset.yaml",
        epochs=int(source["epochs"]),
        seed=int(source["seed"]),
        fit=fit,
        freeze=None,
    )
    endpoint = _save_endpoint(
        last,
        SOURCE_CHECKPOINT,
        method="source",
        seed=int(source["seed"]),
        source_path=None,
    )
    payload = {
        "schema_version": 1,
        "status": "NVD_SOURCE_FIXED_LAST_EPOCH_LOCKED",
        "locked_at_utc": prep.utc_now(),
        "seed": int(source["seed"]),
        "epochs": int(source["epochs"]),
        "dataset_sha256": sha256_file(prep.DATA_ROOT / "datasets/source/dataset.yaml"),
        "initialization_sha256": sha256_file(initialization),
        "raw_endpoint": prep.relative(last),
        "raw_endpoint_sha256": sha256_file(last),
        "results": prep.relative(results_csv),
        "results_sha256": sha256_file(results_csv),
        "args": prep.relative(args_yaml),
        "args_sha256": sha256_file(args_yaml),
        "elapsed_seconds": elapsed,
        **endpoint,
        "validation_during_training": False,
        "checkpoint_selected_by_metric": False,
    }
    SOURCE_LOCK.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(SOURCE_LOCK, payload)
    return payload


def train_candidate(method: str, seed: int) -> dict[str, Any]:
    if method not in METHODS or seed not in SEEDS:
        raise NvdExperimentError(f"unregistered candidate: {method}/{seed}")
    train_source()
    checkpoint = _checkpoint(method, seed)
    lock_path = _training_lock(method, seed)
    if lock_path.exists() and checkpoint.exists():
        lock = prep.load_json(lock_path)
        if lock.get("checkpoint_sha256") != sha256_file(checkpoint):
            raise NvdExperimentError(f"candidate checkpoint changed: {method}/{seed}")
        return lock
    if lock_path.exists() or checkpoint.exists():
        raise NvdExperimentError(f"partial candidate endpoint requires audit: {method}/{seed}")
    freeze = None if method == "STF" else FIRST_TRAINABLE_LAYER
    last, results_csv, args_yaml, elapsed = _run_training(
        initialization=SOURCE_CHECKPOINT,
        dataset=prep.DATA_ROOT / f"datasets/{method}/dataset.yaml",
        epochs=EPOCHS,
        seed=seed,
        fit=_raw_fit(method, seed),
        freeze=freeze,
    )
    endpoint = _save_endpoint(
        last,
        checkpoint,
        method=method,
        seed=seed,
        source_path=SOURCE_CHECKPOINT,
    )
    payload = {
        "schema_version": 1,
        "status": "NVD_ADAPTATION_FIXED_LAST_EPOCH_LOCKED",
        "locked_at_utc": prep.utc_now(),
        "method": method,
        "seed": seed,
        "epochs": EPOCHS,
        "images_per_epoch": prep.TRAIN_IMAGES,
        "dataset_sha256": sha256_file(prep.DATA_ROOT / f"datasets/{method}/dataset.yaml"),
        "source_checkpoint_sha256": sha256_file(SOURCE_CHECKPOINT),
        "raw_endpoint": prep.relative(last),
        "raw_endpoint_sha256": sha256_file(last),
        "results": prep.relative(results_csv),
        "results_sha256": sha256_file(results_csv),
        "args": prep.relative(args_yaml),
        "args_sha256": sha256_file(args_yaml),
        "elapsed_seconds": elapsed,
        **endpoint,
        "validation_during_training": False,
        "checkpoint_selected_by_metric": False,
        "target_test_access_during_training": False,
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(lock_path, payload)
    return payload


def train_all() -> list[dict[str, Any]]:
    results = [train_source()]
    results.extend(train_candidate(method, seed) for method in METHODS for seed in SEEDS)
    return results


def _split_manifest(split: str) -> dict[str, Any]:
    return prep.load_json(prep.DATA_ROOT / f"splits/{split}/manifest.json")


def _records(split: str) -> tuple[ImageRecord, ...]:
    manifest = _split_manifest(split)
    rows = manifest.get("rows")
    if not isinstance(rows, list):
        raise NvdExperimentError(f"split manifest rows are missing: {split}")
    records = tuple(
        ImageRecord(
            image_id=int(row["image_id"]),
            path=str(prep.rooted(row["image"])),
            width=1920,
            height=1080,
        )
        for row in rows
        if isinstance(row, dict)
    )
    if len(records) != int(manifest["images"]):
        raise NvdExperimentError(f"record coverage changed: {split}")
    return records


def _validate_training_complete() -> None:
    train_source()
    for method in METHODS:
        for seed in SEEDS:
            train_candidate(method, seed)


def _infer_one(
    *,
    model_key: str,
    checkpoint: Path,
    split: str,
    detector: UltralyticsDetector,
) -> dict[str, Any]:
    path = _prediction(model_key, split)
    lock_path = _prediction_lock(model_key, split)
    if path.exists() and lock_path.exists():
        lock = prep.load_json(lock_path)
        if (
            lock.get("prediction_sha256") != sha256_file(path)
            or lock.get("checkpoint_sha256") != sha256_file(checkpoint)
        ):
            raise NvdExperimentError(f"prediction lock changed: {model_key}/{split}")
        return lock
    if path.exists() or lock_path.exists():
        raise NvdExperimentError(f"partial prediction output requires audit: {model_key}/{split}")
    records = _records(split)
    started = time.perf_counter()
    batches = detector.predict(
        records,
        imgsz=IMGSZ,
        conf=LOW_FLOOR,
        iou=NMS_IOU,
        max_det=MAX_DET,
        fp16=True,
    )
    elapsed = time.perf_counter() - started
    rows = write_coco_predictions(path, batches, category_id_by_class=CATEGORY_MAP)
    payload = {
        "schema_version": 1,
        "status": "NVD_PREDICTION_ARTIFACT_LOCKED",
        "locked_at_utc": prep.utc_now(),
        "model": model_key,
        "split": split,
        "checkpoint": prep.relative(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "prediction": prep.relative(path),
        "prediction_sha256": sha256_file(path),
        "images": len(records),
        "detections": len(rows),
        "elapsed_seconds": elapsed,
        "candidate_score_floor": LOW_FLOOR,
        "nms_iou": NMS_IOU,
        "max_det": MAX_DET,
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(lock_path, payload)
    return payload


def infer_model(model_key: str) -> list[dict[str, Any]]:
    _validate_training_complete()
    if model_key == "source":
        checkpoint = SOURCE_CHECKPOINT
    else:
        matches = [
            (method, seed)
            for method in METHODS
            for seed in SEEDS
            if _model_key(method, seed) == model_key
        ]
        if len(matches) != 1:
            raise NvdExperimentError(f"unregistered model key: {model_key}")
        checkpoint = _checkpoint(*matches[0])
    detector = UltralyticsDetector(
        checkpoint,
        model_name="yolo11n",
        device="cuda:0",
        expected_class_names=CLASS_NAMES,
        project_root=ROOT,
        stream_chunk_records=8,
        release_cuda_cache_between_chunks=False,
    )
    warmup = _records("source_retention")[:16]
    detector.predict(
        warmup,
        imgsz=IMGSZ,
        conf=LOW_FLOOR,
        iou=NMS_IOU,
        max_det=MAX_DET,
        fp16=True,
    )
    artifacts = [
        _infer_one(
            model_key=model_key,
            checkpoint=checkpoint,
            split=split,
            detector=detector,
        )
        for split in SPLITS
    ]
    del detector
    gc.collect()
    torch.cuda.empty_cache()
    return artifacts


def infer_all() -> dict[str, Any]:
    if PREDICTION_COMPLETE.exists():
        payload = prep.load_json(PREDICTION_COMPLETE)
        for row in payload.get("artifacts", []):
            if not isinstance(row, dict):
                raise NvdExperimentError("malformed prediction-complete row")
            path = prep.rooted(row["prediction"])
            if row.get("prediction_sha256") != sha256_file(path):
                raise NvdExperimentError(f"completed prediction changed: {path}")
        return payload
    artifacts = infer_model("source")
    for method in METHODS:
        for seed in SEEDS:
            artifacts.extend(infer_model(_model_key(method, seed)))
    payload = {
        "schema_version": 1,
        "status": "NVD_REAL_SNOW_ALL_PREDICTIONS_LOCKED",
        "completed_at_utc": prep.utc_now(),
        "models": 1 + len(METHODS) * len(SEEDS),
        "splits": list(SPLITS),
        "artifacts": artifacts,
        "method_or_checkpoint_selection_from_predictions": False,
    }
    atomic_write_json(PREDICTION_COMPLETE, payload)
    return payload


def _annotation(split: str) -> Path:
    return prep.DATA_ROOT / f"splits/{split}/annotations.coco.json"


def _image_ids(split: str) -> list[int]:
    records = _records(split)
    return [int(record.image_id) for record in records]


def _metric_row(model_key: str, split: str) -> dict[str, Any]:
    result = evaluate_coco(
        _annotation(split),
        _prediction(model_key, split),
        max_det=MAX_DET,
        image_ids=_image_ids(split),
    )
    method = "source" if model_key == "source" else model_key.rsplit("_seed_", 1)[0]
    seed: int | None = None if model_key == "source" else int(model_key.rsplit("_seed_", 1)[1])
    return {
        "model": model_key,
        "method": method,
        "seed": seed,
        "split": split,
        **{metric: float(result[metric]) for metric in METRICS},
        "images_evaluated": int(result["images_evaluated"]),
        "prediction_sha256": sha256_file(_prediction(model_key, split)),
    }


def _summary(values: Sequence[float], *, paired: bool = False) -> dict[str, Any]:
    if len(values) != 3:
        raise NvdExperimentError("trajectory summaries require three registered seeds")
    mean = fmean(values)
    deviation = stdev(values)
    result: dict[str, Any] = {
        "values": list(values),
        "mean": mean,
        "sample_standard_deviation": deviation,
    }
    if paired:
        half = T95_DF2 * deviation / math.sqrt(3)
        result.update(
            {
                "paired_95_percent_t_interval": [mean - half, mean + half],
                "degrees_of_freedom": 2,
                "role": "descriptive_training_trajectory_interval",
            }
        )
    return result


def _trajectory_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    source = {
        (str(row["split"]), metric): float(row[metric])
        for row in rows
        if row["method"] == "source"
        for metric in METRICS
    }
    summaries: dict[str, Any] = {}
    deltas: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    for split in SPLITS:
        summaries[split] = {}
        deltas[split] = {}
        comparisons[split] = {}
        for method in METHODS:
            summaries[split][method] = {}
            deltas[split][method] = {}
            for metric in METRICS:
                values = [
                    float(
                        next(
                            row[metric]
                            for row in rows
                            if row["method"] == method
                            and row["seed"] == seed
                            and row["split"] == split
                        )
                    )
                    for seed in SEEDS
                ]
                summaries[split][method][metric] = _summary(values)
                deltas[split][method][metric] = _summary(
                    [value - source[(split, metric)] for value in values],
                    paired=True,
                )
        for baseline in ("STF", "NoVisibility_L10"):
            label = f"CVBRA_L10_minus_{baseline}"
            comparisons[split][label] = {}
            for metric in METRICS:
                values = [
                    float(
                        next(
                            row[metric]
                            for row in rows
                            if row["method"] == "CVBRA_L10"
                            and row["seed"] == seed
                            and row["split"] == split
                        )
                    )
                    - float(
                        next(
                            row[metric]
                            for row in rows
                            if row["method"] == baseline
                            and row["seed"] == seed
                            and row["split"] == split
                        )
                    )
                    for seed in SEEDS
                ]
                comparisons[split][label][metric] = _summary(values, paired=True)
    return {
        "source_fixed_checkpoint": {
            split: {metric: source[(split, metric)] for metric in METRICS} for split in SPLITS
        },
        "adapted_summary_by_split": summaries,
        "adapted_minus_source_by_split": deltas,
        "paired_method_contrasts_by_split": comparisons,
    }


def _test_clusters() -> dict[str, list[int]]:
    rows = _split_manifest("target_test").get("rows")
    if not isinstance(rows, list):
        raise NvdExperimentError("target-test manifest rows are missing")
    clusters: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        if not isinstance(row, dict) or row.get("temporal_block") is None:
            raise NvdExperimentError("target-test temporal-block identity is missing")
        block = int(row["temporal_block"])
        clusters[f"block_{block:02d}"].append(int(row["image_id"]))
    if len(clusters) != 32 or sum(map(len, clusters.values())) != 2027:
        raise NvdExperimentError("target-test temporal blocks changed")
    return dict(sorted(clusters.items()))


def _temporal_bootstrap() -> dict[str, Any]:
    seed = 42
    method_path = _prediction(_model_key("CVBRA_L10", seed), "target_test")
    baselines = {
        "CVBRA_L10_minus_source": _prediction("source", "target_test"),
        "CVBRA_L10_minus_STF": _prediction(_model_key("STF", seed), "target_test"),
        "CVBRA_L10_minus_NoVisibility_L10": _prediction(
            _model_key("NoVisibility_L10", seed), "target_test"
        ),
    }
    results = {}
    for contrast, baseline_path in baselines.items():
        checkpoint = OUTPUT / "bootstrap_checkpoints" / f"{contrast}.json"
        scopes = {
            "target_test_32_contiguous_temporal_blocks": ClusterBootstrapScope(
                clusters=_test_clusters(),
                checkpoint_path=checkpoint,
                checkpoint_identity={
                    "protocol": "nvd_real_snow_cvbra_v1",
                    "contrast": contrast,
                    "primary_training_seed": seed,
                },
            )
        }
        result = paired_coco_ap_cluster_bootstrap_scopes(
            _annotation("target_test"),
            baseline_path,
            method_path,
            scopes,
            resamples=10000,
            seed=20260825,
            max_det=MAX_DET,
            workers=4,
            chunk_resamples=100,
            accelerate_ap_only=True,
        )
        results[contrast] = result["target_test_32_contiguous_temporal_blocks"]
    return {
        "sampling_unit": "contiguous_temporal_block",
        "blocks": 32,
        "images": 2027,
        "resamples": 10000,
        "bootstrap_seed": 20260825,
        "primary_training_seed": seed,
        "contrasts": results,
        "scope_note": (
            "The blocks address serial dependence within one held-out video and are not "
            "represented as independent sequences or locations."
        ),
    }


def _metrics_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    columns = ("model", "method", "seed", "split", *METRICS, "images_evaluated")
    buffer = io.StringIO(newline="")
    buffer.write(",".join(columns) + "\n")
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column)
            if isinstance(value, float):
                values.append(f"{value:.9f}")
            elif value is None:
                values.append("")
            else:
                values.append(str(value))
        buffer.write(",".join(values) + "\n")
    return buffer.getvalue()


def score() -> dict[str, Any]:
    infer_all()
    if REPORT.exists() and COMPLETE.exists():
        report = prep.load_json(REPORT)
        if prep.load_json(COMPLETE).get("report_sha256") != sha256_file(REPORT):
            raise NvdExperimentError("completed real-snow report changed")
        return report
    model_keys = ["source"] + [
        _model_key(method, seed) for method in METHODS for seed in SEEDS
    ]
    rows = [_metric_row(model_key, split) for model_key in model_keys for split in SPLITS]
    atomic_write_text(METRICS_CSV, _metrics_csv(rows))
    payload = {
        "schema_version": 1,
        "status": "COMPLETE_NVD_REAL_SNOW_CVBRA_INDEPENDENT_CONFIRMATION",
        "completed_at_utc": prep.utc_now(),
        "registration_sha256": sha256_file(prep.REGISTRATION),
        "data_lock_sha256": sha256_file(prep.DATA_LOCK),
        "prediction_complete_sha256": sha256_file(PREDICTION_COMPLETE),
        "metrics_csv": prep.relative(METRICS_CSV),
        "metrics_csv_sha256": sha256_file(METRICS_CSV),
        "metric_rows": rows,
        "trajectory_analysis": _trajectory_report(rows),
        "target_test_temporal_block_bootstrap": _temporal_bootstrap(),
        "selection_statement": (
            "No validation or held-out test metric selected a method, hyperparameter, epoch, "
            "checkpoint, contrast, seed, or bootstrap scope. All registered results are retained."
        ),
        "uavdt_used": False,
    }
    atomic_write_json(REPORT, payload)
    atomic_write_json(
        COMPLETE,
        {
            "status": payload["status"],
            "report": prep.relative(REPORT),
            "report_sha256": sha256_file(REPORT),
        },
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the registered NVD real-snow CVBRA study")
    parser.add_argument(
        "--stage",
        choices=("preflight", "train-source", "train", "infer", "score", "all"),
        required=True,
    )
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--model")
    args = parser.parse_args()
    if args.stage == "preflight":
        result: Any = preflight()
    elif args.stage == "train-source":
        result = train_source()
    elif args.stage == "train":
        if (args.method is None) != (args.seed is None):
            raise NvdExperimentError("--method and --seed must be supplied together")
        result = (
            train_candidate(args.method, args.seed)
            if args.method is not None and args.seed is not None
            else train_all()
        )
    elif args.stage == "infer":
        result = infer_model(args.model) if args.model is not None else infer_all()
    else:
        if any(value is not None for value in (args.method, args.seed, args.model)):
            raise NvdExperimentError("method, seed, and model filters do not apply to score/all")
        result = score()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
