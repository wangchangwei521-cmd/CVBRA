from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

import torch
from scripts import run_cvbra_v1_aldi_direct_baseline as original_aldi
from scripts import run_cvbra_v1_allocation_sensitivity_v1 as sensitivity
from scripts import run_cvbra_v1_training_seed_robustness as evidence_base

from buse_uav.detectors.aldi_ultralytics import (
    ALDIRuntimeConfig,
    ALDITranslationDetectionTrainer,
    configure_aldi_runtime,
)
from buse_uav.detectors.ultralytics_adapter import (
    UltralyticsDetector,
    configure_ultralytics_environment,
)
from buse_uav.evaluation.coco import evaluate_coco, write_coco_predictions
from buse_uav.schemas import ImageRecord
from buse_uav.utils.hashing import sha256_file
from buse_uav.utils.io import atomic_write_json

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/experiment/cvbra_v4_aldi_hn_multitrajectory_v1.yaml"
SOURCE = ROOT / "weights/hazydet/yolo11n_best.pt"
DATASET = ROOT / "data/processed/cvbra_v1/dataset.yaml"
MANIFEST = ROOT / "data/processed/cvbra_v1/manifest.json"
OFFICIAL_BASE_YOLO = ROOT / "third_party/aldi_official/configs/Base-Yolo.yaml"
ALDI_MODULE = ROOT / "src/buse_uav/detectors/aldi_ultralytics.py"
SEED42_CHECKPOINT = ROOT / "runs/cvbra_v2_aldi_horizon_normalized_v1/ALDIpp_AF_HN_equal.pt"
SEED42_TRAINING_LOCK = (
    ROOT / "reports/development/cvbra_v2_aldi_horizon_normalized_v1/training_lock.json"
)
METRIC_INTEGRITY = ROOT / "reports/development/cvbra_v3_metric_integrity_v1"
METRIC_COMPLETE = METRIC_INTEGRITY / "COMPLETE.json"
METRIC_POINTS = METRIC_INTEGRITY / "low_floor_point_report.json"
FULL_GRID = (
    ROOT
    / "reports/development/cvbra_v3_full_grid_multiseed_v1/full_grid_multiseed_report.json"
)

OUTPUT = ROOT / "reports/development/cvbra_v4_aldi_hn_multitrajectory_v1"
RUN_ROOT = ROOT / "runs/cvbra_v4_aldi_hn_multitrajectory_v1"
REGISTRATION = OUTPUT / "REGISTRATION_LOCK.json"
PREDICTION_LOCK = OUTPUT / "PREDICTIONS_LOCKED.json"
REPORT = OUTPUT / "multitrajectory_report.json"
COMPLETE = OUTPUT / "COMPLETE.json"

SEEDS = (42, 27182, 31415)
NEW_SEEDS = (27182, 31415)
TARGET_VIEWS = ("original", "fog_0p6", "fog_1p0")
COORDINATES = (*TARGET_VIEWS, "HazyDet")
CLASS_NAMES = ("car", "truck", "bus")
TARGET_CATEGORY_MAP = {0: 1, 1: 2, 2: 3}
HAZY_CATEGORY_MAP = {0: 0, 1: 1, 2: 2}
METRIC_KEYS = ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large")

OFFICIAL_ALPHA = 0.9996
OFFICIAL_UPDATES = 50_000
MATCHED_UPDATES = 599
EMA_ALPHA = OFFICIAL_ALPHA ** ((OFFICIAL_UPDATES - 1) / (MATCHED_UPDATES - 1))
EPOCHS = 8
IMGSZ = 1280
BATCH = 2
LOW_FLOOR = 0.001
NMS_IOU = 0.70
MAX_DET = 500
CHUNK_SIZE = 8
WARMUP_IMAGES = 16
T95_DF2 = 4.302652729911275


class MultitrajectoryALDIError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MultitrajectoryALDIError(f"expected JSON object: {path}")
    return value


def _checkpoint(seed: int) -> Path:
    if seed == 42:
        return SEED42_CHECKPOINT
    return RUN_ROOT / f"seed_{seed}" / f"ALDIpp_AF_HN_equal_s{seed}.pt"


def _training_lock(seed: int) -> Path:
    return OUTPUT / "training_locks" / f"seed_{seed}.json"


def _diagnostics(seed: int) -> Path:
    return OUTPUT / "training_diagnostics" / f"seed_{seed}.json"


def _source_paths() -> tuple[Path, ...]:
    entries = _load(MANIFEST).get("entries")
    if not isinstance(entries, list) or len(entries) != 3600:
        raise MultitrajectoryALDIError("training manifest changed")
    paths = tuple(
        ROOT / str(row["image"])
        for row in entries
        if isinstance(row, dict) and row.get("role") == "source_haze_replay"
    )
    if len(paths) != 900 or len(set(paths)) != 900 or not all(p.is_file() for p in paths):
        raise MultitrajectoryALDIError("source replay coverage changed")
    return paths


def register() -> dict[str, Any]:
    required = (
        PROTOCOL,
        SOURCE,
        DATASET,
        MANIFEST,
        OFFICIAL_BASE_YOLO,
        ALDI_MODULE,
        SEED42_CHECKPOINT,
        SEED42_TRAINING_LOCK,
        METRIC_COMPLETE,
        METRIC_POINTS,
        FULL_GRID,
        Path(__file__),
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise MultitrajectoryALDIError(f"registration inputs missing: {missing}")
    if not math.isclose(EMA_ALPHA, 0.9671024549752264, abs_tol=1e-15):
        raise MultitrajectoryALDIError("EMA horizon changed")
    payload = {
        "schema_version": 1,
        "status": "CVBRA_V4_ALDI_HN_MULTITRAJECTORY_REGISTERED_BEFORE_NEW_TRAINING",
        "registered_at_utc": _now(),
        "protocol": _relative(PROTOCOL),
        "protocol_sha256": sha256_file(PROTOCOL),
        "runner": _relative(Path(__file__)),
        "runner_sha256": sha256_file(Path(__file__)),
        "source_checkpoint_sha256": sha256_file(SOURCE),
        "dataset_sha256": sha256_file(DATASET),
        "manifest_sha256": sha256_file(MANIFEST),
        "official_base_yolo_sha256": sha256_file(OFFICIAL_BASE_YOLO),
        "aldi_module_sha256": sha256_file(ALDI_MODULE),
        "seed42_checkpoint_sha256": sha256_file(SEED42_CHECKPOINT),
        "seed42_training_lock_sha256": sha256_file(SEED42_TRAINING_LOCK),
        "metric_integrity_complete_sha256": sha256_file(METRIC_COMPLETE),
        "full_grid_report_sha256": sha256_file(FULL_GRID),
        "complete_seed_set": list(SEEDS),
        "additional_seeds": list(NEW_SEEDS),
        "epochs": EPOCHS,
        "images_per_epoch": 3600,
        "matched_updates": MATCHED_UPDATES,
        "ema_alpha": EMA_ALPHA,
        "fixed_last_epoch_teacher": True,
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
            raise MultitrajectoryALDIError("registration changed")
        return existing
    OUTPUT.mkdir(parents=True, exist_ok=True)
    atomic_write_json(REGISTRATION, payload)
    return payload


def _reuse_seed42() -> dict[str, Any]:
    register()
    destination = _training_lock(42)
    payload = {
        "schema_version": 1,
        "status": "ALDI_HN_SEED42_LOCKED_ARTIFACT_REUSED",
        "recorded_at_utc": _now(),
        "seed": 42,
        "checkpoint": _relative(SEED42_CHECKPOINT),
        "checkpoint_sha256": sha256_file(SEED42_CHECKPOINT),
        "source_training_lock": _relative(SEED42_TRAINING_LOCK),
        "source_training_lock_sha256": sha256_file(SEED42_TRAINING_LOCK),
        "new_training_executed": False,
    }
    if destination.exists():
        existing = _load(destination)
        stable = ("seed", "checkpoint", "checkpoint_sha256", "source_training_lock_sha256")
        if any(existing.get(key) != payload.get(key) for key in stable):
            raise MultitrajectoryALDIError("seed-42 reuse record changed")
        return existing
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(destination, payload)
    return payload


def train_seed(seed: int) -> dict[str, Any]:
    if seed not in SEEDS:
        raise MultitrajectoryALDIError(f"unregistered seed: {seed}")
    if seed == 42:
        return _reuse_seed42()
    register()
    checkpoint = _checkpoint(seed)
    lock_path = _training_lock(seed)
    if lock_path.exists() and checkpoint.exists():
        lock = _load(lock_path)
        if lock.get("checkpoint_sha256") != sha256_file(checkpoint):
            raise MultitrajectoryALDIError(f"checkpoint changed for seed {seed}")
        return lock
    fit = RUN_ROOT / f"seed_{seed}" / "raw_endpoint" / "fit"
    diagnostics = _diagnostics(seed)
    if any(path.exists() for path in (fit, lock_path, checkpoint, diagnostics)):
        raise MultitrajectoryALDIError(f"partial training output for seed {seed}")
    configure_aldi_runtime(
        ALDIRuntimeConfig(
            variant="equal_supervision",
            source_image_paths=_source_paths(),
            expected_source_images=900,
            diagnostics_path=diagnostics,
            protocol_sha256=sha256_file(PROTOCOL),
            registration_sha256=sha256_file(REGISTRATION),
            implementation_lock_sha256=sha256_file(Path(__file__)),
            source_checkpoint_sha256=sha256_file(SOURCE),
            data_manifest_sha256=sha256_file(MANIFEST),
            seed=seed,
            ema_alpha=EMA_ALPHA,
            ema_initialize_from_first_student_update=True,
        )
    )
    configure_ultralytics_environment(ROOT)
    from ultralytics import YOLO  # type: ignore[attr-defined]

    model = YOLO(str(SOURCE))
    started = time.perf_counter()
    model.train(
        trainer=ALDITranslationDetectionTrainer,
        data=str(DATASET.resolve()),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device="0",
        workers=4,
        project=str((RUN_ROOT / f"seed_{seed}" / "raw_endpoint").resolve()),
        name="fit",
        exist_ok=False,
        pretrained=True,
        freeze=None,
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
    elapsed = time.perf_counter() - started
    raw = fit / "weights/last.pt"
    results = fit / "results.csv"
    args = fit / "args.yaml"
    for path in (raw, results, args, diagnostics):
        if not path.is_file():
            raise MultitrajectoryALDIError(f"training output missing: {path}")
    diagnostic_payload = _load(diagnostics)
    if (
        diagnostic_payload.get("ema_updates") != MATCHED_UPDATES
        or not math.isclose(
            float(diagnostic_payload.get("ema_alpha", -1)), EMA_ALPHA, abs_tol=1e-15
        )
        or diagnostic_payload.get("ema_initialized_from_first_student_update") is not True
        or diagnostic_payload.get("target_ground_truth_used_for_training") is not True
    ):
        raise MultitrajectoryALDIError(f"runtime contract failed for seed {seed}")
    source_payload, source_model = original_aldi._load_checkpoint(SOURCE)
    _, teacher = original_aldi._load_checkpoint(raw)
    output_model = copy.deepcopy(source_model)
    incompatible = output_model.load_state_dict(teacher.state_dict(), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise MultitrajectoryALDIError(f"teacher endpoint load failed for seed {seed}")
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
            "aldi_hn_multitrajectory": {
                "seed": seed,
                "ema_alpha": EMA_ALPHA,
                "ema_updates": MATCHED_UPDATES,
                "endpoint": "fixed_last_epoch_EMA_teacher",
                "validation_selected": False,
                "registration_sha256": sha256_file(REGISTRATION),
            },
        }
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(checkpoint)
    lock = {
        "schema_version": 1,
        "status": "ALDI_HN_ADDITIONAL_TRAJECTORY_TRAINED_AND_LOCKED",
        "completed_at_utc": _now(),
        "seed": seed,
        "checkpoint": _relative(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "raw_checkpoint_sha256": sha256_file(raw),
        "results_sha256": sha256_file(results),
        "args_sha256": sha256_file(args),
        "diagnostics_sha256": sha256_file(diagnostics),
        "elapsed_seconds": elapsed,
        "ema_alpha": EMA_ALPHA,
        "ema_updates": MATCHED_UPDATES,
        "validation_metric_used_for_training_or_selection": False,
        "official_test_inference": False,
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(lock_path, lock)
    del model, output_model, teacher, source_model
    gc.collect()
    torch.cuda.empty_cache()
    return lock


def _seed42_prediction(view: str) -> Path:
    domain = "HazyDet_validation" if view == "HazyDet" else "UAV_OBB_validation"
    leaf = "hazy" if view == "HazyDet" else view
    return (
        METRIC_INTEGRITY
        / "predictions"
        / "ALDIpp_AF_HN"
        / domain
        / leaf
        / "predictions.coco.json"
    )


def _prediction(seed: int, view: str) -> Path:
    if seed == 42:
        return _seed42_prediction(view)
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
            raise MultitrajectoryALDIError(f"prediction changed: {path}")
        return payload
    if path.parent.exists():
        raise MultitrajectoryALDIError(f"partial prediction output: {path.parent}")
    batches = detector.predict(
        records,
        imgsz=IMGSZ,
        conf=LOW_FLOOR,
        iou=NMS_IOU,
        max_det=MAX_DET,
        fp16=True,
    )
    write_coco_predictions(path, batches, category_id_by_class=category_map)
    payload = {
        "schema_version": 1,
        "status": "ALDI_HN_MULTITRAJECTORY_PREDICTION_COMPLETE",
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
    if seed == 42:
        artifacts = []
        for view in COORDINATES:
            path = _prediction(seed, view)
            if not path.is_file():
                raise MultitrajectoryALDIError(f"locked seed-42 prediction missing: {path}")
            artifacts.append(
                {
                    "seed": seed,
                    "view": view,
                    "prediction": _relative(path),
                    "prediction_sha256": sha256_file(path),
                    "reused_locked_artifact": True,
                }
            )
        return artifacts
    target_records, _ = sensitivity._target_records()
    hazy_records = evidence_base._hazy_records(verify_hashes=False)
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
        target_records["original"][:WARMUP_IMAGES],
        imgsz=IMGSZ,
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
                raise MultitrajectoryALDIError(f"prediction lock changed: {path}")
        return payload
    artifacts = []
    for seed in SEEDS:
        for item in infer_seed(seed):
            if "prediction" not in item:
                view = str(item["view"])
                path = _prediction(seed, view)
                item = {
                    **item,
                    "prediction": _relative(path),
                    "prediction_sha256": sha256_file(path),
                }
            artifacts.append(item)
    payload = {
        "schema_version": 1,
        "status": "ALDI_HN_MULTITRAJECTORY_PREDICTIONS_LOCKED",
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


def _summary(values: Sequence[float], *, paired_interval: bool = False) -> dict[str, Any]:
    if len(values) != 3:
        raise MultitrajectoryALDIError("three trajectories are required")
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
            raise MultitrajectoryALDIError("report changed")
        return report
    rows = [_metric(seed, view) for seed in SEEDS for view in COORDINATES]
    profiles = {
        str(seed): {
            view: next(r["AP"] for r in rows if r["seed"] == seed and r["view"] == view)
            for view in COORDINATES
        }
        for seed in SEEDS
    }
    summary = {
        view: _summary([float(profiles[str(seed)][view]) for seed in SEEDS])
        for view in COORDINATES
    }
    cvbra_profiles = _load(FULL_GRID)["profiles_by_seed"]
    paired = {}
    for view in COORDINATES:
        values = [
            float(cvbra_profiles[str(seed)]["qS_0p25_L10"][view])
            - float(profiles[str(seed)][view])
            for seed in SEEDS
        ]
        paired[view] = _summary(values, paired_interval=True)
    payload = {
        "schema_version": 1,
        "status": "COMPLETE_ALDI_HN_THREE_TRAJECTORY_MATCHED_COMPARISON",
        "completed_at_utc": _now(),
        "registration_sha256": sha256_file(REGISTRATION),
        "prediction_lock_sha256": sha256_file(PREDICTION_LOCK),
        "seeds": list(SEEDS),
        "rows": rows,
        "profiles_by_seed": profiles,
        "summary_by_view": summary,
        "paired_CVBRA_L10_minus_ALDI_HN": paired,
        "paired_interval_scope": (
            "Three predeclared matched stochastic trajectories; the t intervals describe "
            "training-trajectory variability and do not replace grouped image bootstrap inference."
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
    parser = argparse.ArgumentParser(description="Run three-trajectory matched ALDI++-AF control")
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
            raise MultitrajectoryALDIError("--seed is only valid for train or infer")
        result = score()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
