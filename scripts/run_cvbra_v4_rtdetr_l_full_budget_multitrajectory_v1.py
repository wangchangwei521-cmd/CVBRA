from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

import torch
from scripts import run_cvbra_v1_allocation_sensitivity_v1 as sensitivity
from scripts import run_cvbra_v1_training_seed_robustness as evidence_base

from buse_uav.detectors.ultralytics_adapter import (
    UltralyticsDetector,
    configure_ultralytics_environment,
)
from buse_uav.evaluation.coco import evaluate_coco, write_coco_predictions
from buse_uav.schemas import ImageRecord
from buse_uav.utils.hashing import sha256_file
from buse_uav.utils.io import atomic_write_json

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/experiment/cvbra_v4_rtdetr_l_full_budget_multitrajectory_v1.yaml"
SOURCE = ROOT / "weights/hazydet/rtdetr_l_best.pt"
DATASET = ROOT / "data/processed/cvbra_v1/dataset.yaml"
MANIFEST = ROOT / "data/processed/cvbra_v1/manifest.json"
DATA_LOCK = ROOT / "reports/development/cvbra_v1/generated_dataset_lock.json"
METRIC_INTEGRITY = ROOT / "reports/development/cvbra_v3_metric_integrity_v1"
METRIC_COMPLETE = METRIC_INTEGRITY / "COMPLETE.json"
METRIC_POINTS = METRIC_INTEGRITY / "low_floor_point_report.json"
HISTORICAL_CHECKPOINT = ROOT / "runs/cvbra_v1_rtdetr_l/cvbra_v1_rtdetr_l.pt"
HISTORICAL_LOCK = ROOT / "reports/development/cvbra_v1_rtdetr_l/checkpoint_lock.json"

OUTPUT = ROOT / "reports/development/cvbra_v4_rtdetr_l_full_budget_multitrajectory_v1"
RUN_ROOT = ROOT / "runs/cvbra_v4_rtdetr_l_full_budget_multitrajectory_v1"
REGISTRATION = OUTPUT / "REGISTRATION_LOCK.json"
PREDICTION_LOCK = OUTPUT / "PREDICTIONS_LOCKED.json"
REPORT = OUTPUT / "multitrajectory_report.json"
COMPLETE = OUTPUT / "COMPLETE.json"

SEEDS = (42, 27182, 31415)
TARGET_VIEWS = ("original", "fog_0p6", "fog_1p0")
COORDINATES = (*TARGET_VIEWS, "HazyDet")
CLASS_NAMES = ("car", "truck", "bus")
TARGET_CATEGORY_MAP = {0: 1, 1: 2, 2: 3}
HAZY_CATEGORY_MAP = {0: 0, 1: 1, 2: 2}
METRIC_KEYS = ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large")

SOURCE_SHA256 = "7d8fccbc1a9b66e28311ba91363f0b9f6ca0e4fdd61e88bd2f90099c6ea2a44c"
FROZEN_LAST_LAYER = 9
FIRST_TRAINABLE_LAYER = 10
LAST_TRAINABLE_LAYER = 28
MODEL_LAYERS = 29
EPOCHS = 8
TRAIN_IMGSZ = 640
INFERENCE_IMGSZ = 1280
BATCH = 1
LOW_FLOOR = 0.001
NMS_IOU = 0.70
MAX_DET = 500
CHUNK_SIZE = 4
WARMUP_IMAGES = 8
T95_DF2 = 4.302652729911275


class FullBudgetRTDETRError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FullBudgetRTDETRError(f"expected JSON object: {path}")
    return value


def _layer_index(name: str) -> int:
    parts = name.split(".", 2)
    if len(parts) < 3 or parts[0] != "model":
        raise FullBudgetRTDETRError(f"state has no model-layer index: {name}")
    try:
        return int(parts[1])
    except ValueError as exc:
        raise FullBudgetRTDETRError(f"invalid model layer index: {name}") from exc


def _load_checkpoint(path: Path) -> tuple[dict[str, Any], torch.nn.Module]:
    configure_ultralytics_environment(ROOT)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise FullBudgetRTDETRError(f"unsupported checkpoint: {path}")
    model = checkpoint.get("ema") or checkpoint.get("model")
    if not isinstance(model, torch.nn.Module):
        raise FullBudgetRTDETRError(f"checkpoint has no model module: {path}")
    return checkpoint, model


def _model_contract() -> dict[str, Any]:
    _, model = _load_checkpoint(SOURCE)
    layers = getattr(model, "model", None)
    if not isinstance(layers, torch.nn.Sequential) or len(layers) != MODEL_LAYERS:
        raise FullBudgetRTDETRError("RT-DETR-L layer graph changed")
    indices = {_layer_index(name) for name in model.state_dict()}
    if min(indices) != 0 or max(indices) != LAST_TRAINABLE_LAYER:
        raise FullBudgetRTDETRError("RT-DETR-L state coverage changed")
    return {
        "layers": len(layers),
        "state_entries": len(model.state_dict()),
        "class_names": {str(k): str(v) for k, v in dict(getattr(model, "names", {})).items()},
        "frozen_parameters": sum(
            parameter.numel()
            for index, layer in enumerate(layers)
            if index <= FROZEN_LAST_LAYER
            for parameter in layer.parameters()
        ),
        "trainable_parameters": sum(
            parameter.numel()
            for index, layer in enumerate(layers)
            if index >= FIRST_TRAINABLE_LAYER
            for parameter in layer.parameters()
        ),
    }


def _checkpoint(seed: int) -> Path:
    return RUN_ROOT / f"seed_{seed}" / f"RTDETR_L_CVBRA_full_s{seed}.pt"


def _training_lock(seed: int) -> Path:
    return OUTPUT / "training_locks" / f"seed_{seed}.json"


def register() -> dict[str, Any]:
    required = (
        PROTOCOL,
        SOURCE,
        DATASET,
        MANIFEST,
        DATA_LOCK,
        METRIC_COMPLETE,
        METRIC_POINTS,
        HISTORICAL_CHECKPOINT,
        HISTORICAL_LOCK,
        Path(__file__),
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FullBudgetRTDETRError(f"registration inputs missing: {missing}")
    if sha256_file(SOURCE) != SOURCE_SHA256:
        raise FullBudgetRTDETRError("RT-DETR-L source checkpoint changed")
    manifest = _load(MANIFEST)
    if manifest.get("total_images") != 3600:
        raise FullBudgetRTDETRError("training manifest coverage changed")
    payload = {
        "schema_version": 1,
        "status": "CVBRA_V4_RTDETR_L_FULL_BUDGET_REGISTERED_BEFORE_TRAINING",
        "registered_at_utc": _now(),
        "protocol": _relative(PROTOCOL),
        "protocol_sha256": sha256_file(PROTOCOL),
        "runner": _relative(Path(__file__)),
        "runner_sha256": sha256_file(Path(__file__)),
        "source_checkpoint_sha256": sha256_file(SOURCE),
        "dataset_sha256": sha256_file(DATASET),
        "manifest_sha256": sha256_file(MANIFEST),
        "data_lock_sha256": sha256_file(DATA_LOCK),
        "metric_integrity_complete_sha256": sha256_file(METRIC_COMPLETE),
        "historical_two_epoch_checkpoint_sha256": sha256_file(HISTORICAL_CHECKPOINT),
        "historical_two_epoch_lock_sha256": sha256_file(HISTORICAL_LOCK),
        "model_contract": _model_contract(),
        "seeds": list(SEEDS),
        "epochs": EPOCHS,
        "images_per_epoch": 3600,
        "training_imgsz": TRAIN_IMGSZ,
        "inference_imgsz": INFERENCE_IMGSZ,
        "fixed_last_epoch": True,
        "validation_during_training": False,
        "official_test_inference": False,
        "method_or_hyperparameter_selection": False,
    }
    if REGISTRATION.exists():
        existing = _load(REGISTRATION)
        stable = tuple(
            key for key in payload if key not in {"schema_version", "status", "registered_at_utc"}
        )
        if any(existing.get(key) != payload.get(key) for key in stable):
            raise FullBudgetRTDETRError("registration changed")
        return existing
    OUTPUT.mkdir(parents=True, exist_ok=True)
    atomic_write_json(REGISTRATION, payload)
    return payload


def _combined_state(
    source: Mapping[str, torch.Tensor], trained: Mapping[str, torch.Tensor]
) -> OrderedDict[str, torch.Tensor]:
    if tuple(source) != tuple(trained):
        raise FullBudgetRTDETRError("source and trained state schemas differ")
    combined: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, trained_value in trained.items():
        source_value = source[name]
        if source_value.shape != trained_value.shape:
            raise FullBudgetRTDETRError(f"state shape differs: {name}")
        chosen = source_value if _layer_index(name) <= FROZEN_LAST_LAYER else trained_value
        if chosen.is_floating_point() and not bool(torch.isfinite(chosen).all()):
            raise FullBudgetRTDETRError(f"nonfinite state: {name}")
        combined[name] = chosen.clone()
    return combined


def _build_checkpoint(seed: int, raw: Path) -> dict[str, Any]:
    source_payload, source_model = _load_checkpoint(SOURCE)
    _, trained_model = _load_checkpoint(raw)
    if getattr(source_model, "names", None) != getattr(trained_model, "names", None):
        raise FullBudgetRTDETRError("class schema changed")
    combined = _combined_state(source_model.state_dict(), trained_model.state_dict())
    output_model = copy.deepcopy(trained_model)
    incompatible = output_model.load_state_dict(combined, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise FullBudgetRTDETRError("strict endpoint state load failed")
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
            "cvbra_rtdetr_full_budget": {
                "seed": seed,
                "epochs": EPOCHS,
                "frozen_state_rule": "exact source restore for layers 0..9",
                "endpoint": "fixed_last_epoch",
                "validation_selected": False,
                "registration_sha256": sha256_file(REGISTRATION),
            },
        }
    )
    checkpoint = _checkpoint(seed)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(checkpoint)
    _, observed_model = _load_checkpoint(checkpoint)
    observed = observed_model.state_dict()
    frozen_exact = all(
        torch.equal(observed[name].to(dtype=value.dtype), value)
        for name, value in source_model.state_dict().items()
        if _layer_index(name) <= FROZEN_LAST_LAYER
    )
    changed_trainable = sum(
        1
        for name, value in source_model.state_dict().items()
        if _layer_index(name) > FROZEN_LAST_LAYER
        and value.is_floating_point()
        and not torch.equal(observed[name].to(dtype=value.dtype), value)
    )
    if not frozen_exact or changed_trainable == 0:
        raise FullBudgetRTDETRError("final checkpoint state verification failed")
    return {
        "frozen_layers_0_to_9_exact_source": frozen_exact,
        "changed_trainable_floating_states": changed_trainable,
        "all_states_finite": all(
            bool(torch.isfinite(value).all())
            for value in observed.values()
            if value.is_floating_point()
        ),
    }


def train_seed(seed: int) -> dict[str, Any]:
    if seed not in SEEDS:
        raise FullBudgetRTDETRError(f"unregistered seed: {seed}")
    register()
    checkpoint = _checkpoint(seed)
    lock_path = _training_lock(seed)
    if lock_path.exists() and checkpoint.exists():
        lock = _load(lock_path)
        if lock.get("checkpoint_sha256") != sha256_file(checkpoint):
            raise FullBudgetRTDETRError(f"checkpoint changed for seed {seed}")
        return lock
    fit = RUN_ROOT / f"seed_{seed}" / "raw_endpoint" / "fit"
    if any(path.exists() for path in (fit, lock_path, checkpoint)):
        raise FullBudgetRTDETRError(f"partial training output for seed {seed}")
    configure_ultralytics_environment(ROOT)
    from ultralytics import RTDETR  # type: ignore[attr-defined]

    model = RTDETR(str(SOURCE))
    started = time.perf_counter()
    results = model.train(
        data=str(DATASET.resolve()),
        epochs=EPOCHS,
        imgsz=TRAIN_IMGSZ,
        batch=BATCH,
        device="0",
        workers=4,
        project=str((RUN_ROOT / f"seed_{seed}" / "raw_endpoint").resolve()),
        name="fit",
        exist_ok=False,
        pretrained=True,
        freeze=FIRST_TRAINABLE_LAYER,
        val=False,
        optimizer="auto",
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
    elapsed = time.perf_counter() - started
    actual = Path(results.save_dir)
    raw = actual / "weights/last.pt"
    results_path = actual / "results.csv"
    args_path = actual / "args.yaml"
    for path in (raw, results_path, args_path):
        if not path.is_file():
            raise FullBudgetRTDETRError(f"training output missing: {path}")
    verification = _build_checkpoint(seed, raw)
    payload = {
        "schema_version": 1,
        "status": "RTDETR_L_FULL_BUDGET_TRAJECTORY_TRAINED_AND_LOCKED",
        "completed_at_utc": _now(),
        "seed": seed,
        "checkpoint": _relative(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "raw_checkpoint_sha256": sha256_file(raw),
        "results_sha256": sha256_file(results_path),
        "args_sha256": sha256_file(args_path),
        "elapsed_seconds": elapsed,
        "epochs": EPOCHS,
        "verification": verification,
        "validation_metric_used_for_training_or_selection": False,
        "official_test_inference": False,
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(lock_path, payload)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return payload


def _prediction(seed: int, view: str) -> Path:
    domain = "HazyDet_validation" if view == "HazyDet" else "UAV_OBB_validation"
    leaf = "hazy" if view == "HazyDet" else view
    return OUTPUT / "predictions" / f"seed_{seed}" / domain / leaf / "predictions.coco.json"


def _predict_cell(
    *,
    seed: int,
    view: str,
    detector: UltralyticsDetector,
    records: Sequence[ImageRecord],
    category_map: Mapping[int, int],
) -> dict[str, Any]:
    path = _prediction(seed, view)
    marker = path.parent / "SUCCESS.json"
    if marker.exists():
        payload = _load(marker)
        if payload.get("prediction_sha256") != sha256_file(path):
            raise FullBudgetRTDETRError(f"prediction changed: {path}")
        return payload
    if path.parent.exists():
        raise FullBudgetRTDETRError(f"partial prediction output: {path.parent}")
    batches = detector.predict(
        records,
        imgsz=INFERENCE_IMGSZ,
        conf=LOW_FLOOR,
        iou=NMS_IOU,
        max_det=MAX_DET,
        fp16=True,
    )
    write_coco_predictions(path, batches, category_id_by_class=category_map)
    payload = {
        "schema_version": 1,
        "status": "RTDETR_L_FULL_BUDGET_PREDICTION_COMPLETE",
        "seed": seed,
        "view": view,
        "images": len(batches),
        "checkpoint_sha256": sha256_file(_checkpoint(seed)),
        "prediction_sha256": sha256_file(path),
        "candidate_score_floor": LOW_FLOOR,
    }
    atomic_write_json(marker, payload)
    return payload


def infer_seed(seed: int) -> list[dict[str, Any]]:
    train_seed(seed)
    target_records, _ = sensitivity._target_records()
    hazy_records = evidence_base._hazy_records(verify_hashes=False)
    detector = UltralyticsDetector(
        _checkpoint(seed),
        model_name="rtdetr-l",
        device="cuda:0",
        expected_class_names=CLASS_NAMES,
        project_root=ROOT,
        stream_chunk_records=CHUNK_SIZE,
        release_cuda_cache_between_chunks=True,
    )
    detector.predict(
        target_records["original"][:WARMUP_IMAGES],
        imgsz=INFERENCE_IMGSZ,
        conf=LOW_FLOOR,
        iou=NMS_IOU,
        max_det=MAX_DET,
        fp16=True,
    )
    artifacts = [
        _predict_cell(
            seed=seed,
            view=view,
            detector=detector,
            records=target_records[view],
            category_map=TARGET_CATEGORY_MAP,
        )
        for view in TARGET_VIEWS
    ]
    artifacts.append(
        _predict_cell(
            seed=seed,
            view="HazyDet",
            detector=detector,
            records=hazy_records,
            category_map=HAZY_CATEGORY_MAP,
        )
    )
    del detector
    gc.collect()
    torch.cuda.empty_cache()
    return artifacts


def infer() -> dict[str, Any]:
    register()
    if PREDICTION_LOCK.exists():
        payload = _load(PREDICTION_LOCK)
        for row in payload.get("artifacts", []):
            path = ROOT / str(row["prediction"])
            if row.get("prediction_sha256") != sha256_file(path):
                raise FullBudgetRTDETRError(f"prediction lock changed: {path}")
        return payload
    artifacts = []
    for seed in SEEDS:
        for item in infer_seed(seed):
            view = str(item["view"])
            path = _prediction(seed, view)
            artifacts.append(
                {**item, "prediction": _relative(path), "prediction_sha256": sha256_file(path)}
            )
    payload = {
        "schema_version": 1,
        "status": "RTDETR_L_FULL_BUDGET_MULTITRAJECTORY_PREDICTIONS_LOCKED",
        "completed_at_utc": _now(),
        "candidate_score_floor": LOW_FLOOR,
        "artifacts": artifacts,
        "official_test_inference": False,
    }
    atomic_write_json(PREDICTION_LOCK, payload)
    return payload


def _metric(seed: int, view: str) -> dict[str, Any]:
    image_ids: Sequence[int | str]
    if view == "HazyDet":
        annotation = evidence_base.HAZY_ANNOTATION
        image_ids = [
            record.image_id for record in evidence_base._hazy_records(verify_hashes=False)
        ]
    else:
        conversion = _load(evidence_base.CONVERSION_LOCK)
        annotation = ROOT / str(conversion["annotation"])
        _, image_ids = sensitivity._target_records()
    result = evaluate_coco(
        annotation,
        _prediction(seed, view),
        max_det=MAX_DET,
        image_ids=image_ids,
    )
    return {
        "seed": seed,
        "view": view,
        **{key: float(result[key]) for key in METRIC_KEYS},
        "images_evaluated": int(result["images_evaluated"]),
        "prediction_sha256": sha256_file(_prediction(seed, view)),
    }


def _source_profile() -> dict[str, float]:
    points = _load(METRIC_POINTS)
    target = {
        str(row["view"]): float(row["AP"])
        for row in points["target_rows"]
        if row.get("model") == "RTDETR_source"
    }
    hazy = next(
        float(row["AP"])
        for row in points["hazydet_rows"]
        if row.get("model") == "RTDETR_source"
    )
    return {**target, "HazyDet": hazy}


def _historical_two_epoch_profile() -> dict[str, float]:
    points = _load(METRIC_POINTS)
    target = {
        str(row["view"]): float(row["AP"])
        for row in points["target_rows"]
        if row.get("model") == "RTDETR_CVBRA_L10"
    }
    hazy = next(
        float(row["AP"])
        for row in points["hazydet_rows"]
        if row.get("model") == "RTDETR_CVBRA_L10"
    )
    return {**target, "HazyDet": hazy}


def _summary(values: Sequence[float], *, paired_interval: bool = False) -> dict[str, Any]:
    if len(values) != 3:
        raise FullBudgetRTDETRError("three trajectories are required")
    mean = fmean(values)
    sd = stdev(values)
    payload: dict[str, Any] = {
        "values": list(values),
        "mean": mean,
        "sample_standard_deviation": sd,
    }
    if paired_interval:
        half = T95_DF2 * sd / math.sqrt(3)
        payload.update(
            {
                "paired_95_percent_t_interval": [mean - half, mean + half],
                "degrees_of_freedom": 2,
                "interval_role": "trajectory_level_descriptive_interval",
            }
        )
    return payload


def score() -> dict[str, Any]:
    infer()
    if REPORT.exists() and COMPLETE.exists():
        report = _load(REPORT)
        if _load(COMPLETE).get("report_sha256") != sha256_file(REPORT):
            raise FullBudgetRTDETRError("report changed")
        return report
    rows = [_metric(seed, view) for seed in SEEDS for view in COORDINATES]
    profiles = {
        str(seed): {
            view: next(r["AP"] for r in rows if r["seed"] == seed and r["view"] == view)
            for view in COORDINATES
        }
        for seed in SEEDS
    }
    source = _source_profile()
    summary = {
        view: _summary([float(profiles[str(seed)][view]) for seed in SEEDS])
        for view in COORDINATES
    }
    paired = {
        view: _summary(
            [float(profiles[str(seed)][view]) - source[view] for seed in SEEDS],
            paired_interval=True,
        )
        for view in COORDINATES
    }
    payload = {
        "schema_version": 1,
        "status": "COMPLETE_RTDETR_L_FULL_BUDGET_THREE_TRAJECTORY_CONFIRMATION",
        "completed_at_utc": _now(),
        "registration_sha256": sha256_file(REGISTRATION),
        "prediction_lock_sha256": sha256_file(PREDICTION_LOCK),
        "seeds": list(SEEDS),
        "source_profile": source,
        "historical_two_epoch_seed42_profile": _historical_two_epoch_profile(),
        "rows": rows,
        "profiles_by_seed": profiles,
        "summary_by_view": summary,
        "paired_adapted_minus_source": paired,
        "paired_interval_scope": (
            "Three predeclared stochastic trajectories; the t intervals describe "
            "training-trajectory variability and are not image-sampling intervals."
        ),
        "official_test_inference": False,
        "negative_or_mixed_results_retained": True,
    }
    atomic_write_json(REPORT, payload)
    atomic_write_json(
        COMPLETE,
        {"status": payload["status"], "report_sha256": sha256_file(REPORT)},
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full-budget RT-DETR-L CVBRA confirmation")
    parser.add_argument(
        "--stage", choices=("register", "train", "infer", "score", "all"), default="all"
    )
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.stage == "register":
        result: Any = register()
    elif args.stage == "train":
        result = train_seed(args.seed) if args.seed is not None else [train_seed(s) for s in SEEDS]
    elif args.stage == "infer" and args.seed is not None:
        result = infer_seed(args.seed)
    elif args.stage == "infer":
        result = infer()
    else:
        if args.seed is not None:
            raise FullBudgetRTDETRError("--seed is only valid for train or infer")
        result = score()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
