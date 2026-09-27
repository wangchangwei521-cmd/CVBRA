from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import math
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any

import torch
from scripts import run_cvbra_v1_aldi_direct_baseline as original_aldi
from scripts import run_cvbra_v1_allocation_sensitivity_v1 as sensitivity
from scripts import run_cvbra_v1_final_test_v1 as final_test
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
from buse_uav.schemas import DetectionBatch, ImageRecord
from buse_uav.utils.hashing import sha256_file
from buse_uav.utils.io import atomic_write_json, atomic_write_text

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/experiment/cvbra_v2_aldi_horizon_normalized_v1.yaml"
SOURCE = ROOT / "weights/hazydet/yolo11n_best.pt"
DATASET = ROOT / "data/processed/cvbra_v1/dataset.yaml"
MANIFEST = ROOT / "data/processed/cvbra_v1/manifest.json"
OFFICIAL_BASE_YOLO = ROOT / "third_party/aldi_official/configs/Base-Yolo.yaml"
ALDI_TRANSLATION_MODULE = ROOT / "src/buse_uav/detectors/aldi_ultralytics.py"
OUTPUT = ROOT / "reports/development/cvbra_v2_aldi_horizon_normalized_v1"
RUN_ROOT = ROOT / "runs/cvbra_v2_aldi_horizon_normalized_v1"
REGISTRATION = OUTPUT / "REGISTRATION_LOCK.json"
DIAGNOSTICS = OUTPUT / "train_diagnostics.json"
TRAINING_LOCK = OUTPUT / "training_lock.json"
CHECKPOINT = RUN_ROOT / "ALDIpp_AF_HN_equal.pt"
PREDICTION_LOCK = OUTPUT / "PREDICTIONS_LOCKED.json"
METRICS = OUTPUT / "metrics.csv"
REPORT = OUTPUT / "report.json"
COMPLETE = OUTPUT / "COMPLETE.json"

OFFICIAL_ALPHA = 0.9996
OFFICIAL_UPDATES = 50000
MATCHED_UPDATES = 599
EMA_ALPHA = OFFICIAL_ALPHA ** ((OFFICIAL_UPDATES - 1) / (MATCHED_UPDATES - 1))
EPOCHS = 8
IMGSZ = 1280
BATCH = 2
SEED = 42
PROBE_CONF = 0.08
PUBLISH_CONF = 0.25
NMS_IOU = 0.70
MAX_DET = 500
CHUNK_SIZE = 8
WARMUP_IMAGES = 16
CLASS_NAMES = ("car", "truck", "bus")
TARGET_VIEWS = ("original", "fog_0p6", "fog_1p0")
METRIC_KEYS = ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large")


class HorizonNormalizedALDIError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise HorizonNormalizedALDIError(f"expected JSON object: {path}")
    return value


def _source_paths() -> tuple[Path, ...]:
    manifest = _load(MANIFEST)
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != 3600:
        raise HorizonNormalizedALDIError("training manifest changed")
    paths = tuple(
        ROOT / str(row["image"])
        for row in entries
        if isinstance(row, dict) and row.get("role") == "source_haze_replay"
    )
    if len(paths) != 900 or len(set(paths)) != 900 or not all(path.is_file() for path in paths):
        raise HorizonNormalizedALDIError("source replay path coverage changed")
    return paths


def register() -> dict[str, Any]:
    for path in (
        PROTOCOL,
        SOURCE,
        DATASET,
        MANIFEST,
        OFFICIAL_BASE_YOLO,
        ALDI_TRANSLATION_MODULE,
    ):
        if not path.is_file():
            raise HorizonNormalizedALDIError(f"required input is missing: {path}")
    if not math.isclose(EMA_ALPHA, 0.9671024549752264, rel_tol=0.0, abs_tol=1e-15):
        raise HorizonNormalizedALDIError("EMA horizon derivation changed")
    payload = {
        "schema_version": 1,
        "status": "ALDI_HORIZON_NORMALIZED_REGISTERED",
        "locked_at_utc": _now(),
        "protocol_sha256": sha256_file(PROTOCOL),
        "runner_sha256": sha256_file(Path(__file__)),
        "source_checkpoint_sha256": sha256_file(SOURCE),
        "dataset_sha256": sha256_file(DATASET),
        "manifest_sha256": sha256_file(MANIFEST),
        "official_base_yolo_sha256": sha256_file(OFFICIAL_BASE_YOLO),
        "aldi_translation_module_sha256": sha256_file(ALDI_TRANSLATION_MODULE),
        "official_alpha": OFFICIAL_ALPHA,
        "official_updates": OFFICIAL_UPDATES,
        "matched_updates": MATCHED_UPDATES,
        "normalized_alpha": EMA_ALPHA,
        "first_teacher_update": "exact_student_copy",
        "post_initialization_weight_official": OFFICIAL_ALPHA ** (OFFICIAL_UPDATES - 1),
        "post_initialization_weight_normalized": EMA_ALPHA ** (MATCHED_UPDATES - 1),
        "validation_or_test_performance_used_in_derivation": False,
        "test_labels_previously_accessed": True,
    }
    if REGISTRATION.exists():
        existing = _load(REGISTRATION)
        stable = (
            "protocol_sha256",
            "runner_sha256",
            "source_checkpoint_sha256",
            "dataset_sha256",
            "manifest_sha256",
            "official_base_yolo_sha256",
            "aldi_translation_module_sha256",
            "normalized_alpha",
        )
        if any(existing.get(key) != payload.get(key) for key in stable):
            raise HorizonNormalizedALDIError("horizon-normalized registration changed")
        return existing
    OUTPUT.mkdir(parents=True, exist_ok=True)
    atomic_write_json(REGISTRATION, payload)
    return payload


def train() -> dict[str, Any]:
    register()
    if TRAINING_LOCK.exists() and CHECKPOINT.exists():
        lock = _load(TRAINING_LOCK)
        if lock.get("checkpoint_sha256") != sha256_file(CHECKPOINT):
            raise HorizonNormalizedALDIError("horizon-normalized checkpoint changed")
        return lock
    fit = RUN_ROOT / "raw_endpoint" / "fit"
    if any(path.exists() for path in (fit, TRAINING_LOCK, CHECKPOINT)):
        raise HorizonNormalizedALDIError("partial horizon-normalized training output")
    configure_aldi_runtime(
        ALDIRuntimeConfig(
            variant="equal_supervision",
            source_image_paths=_source_paths(),
            expected_source_images=900,
            diagnostics_path=DIAGNOSTICS,
            protocol_sha256=sha256_file(PROTOCOL),
            registration_sha256=sha256_file(REGISTRATION),
            implementation_lock_sha256=sha256_file(Path(__file__)),
            source_checkpoint_sha256=sha256_file(SOURCE),
            data_manifest_sha256=sha256_file(MANIFEST),
            seed=SEED,
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
        project=str((RUN_ROOT / "raw_endpoint").resolve()),
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
        seed=SEED,
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
    for path in (raw, results, args, DIAGNOSTICS):
        if not path.is_file():
            raise HorizonNormalizedALDIError(f"training output is missing: {path}")
    diagnostics = _load(DIAGNOSTICS)
    if (
        diagnostics.get("ema_updates") != MATCHED_UPDATES
        or not math.isclose(float(diagnostics.get("ema_alpha", -1)), EMA_ALPHA, abs_tol=1e-15)
        or diagnostics.get("ema_initialized_from_first_student_update") is not True
        or diagnostics.get("target_ground_truth_used_for_training") is not True
    ):
        raise HorizonNormalizedALDIError("horizon-normalized runtime contract failed")
    source_payload, source_model = original_aldi._load_checkpoint(SOURCE)
    _, teacher = original_aldi._load_checkpoint(raw)
    output_model = copy.deepcopy(source_model)
    incompatible = output_model.load_state_dict(teacher.state_dict(), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise HorizonNormalizedALDIError("teacher endpoint load failed")
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
            "aldi_horizon_normalized": {
                "variant": "equal_supervision",
                "ema_alpha": EMA_ALPHA,
                "ema_updates": MATCHED_UPDATES,
                "ema_initialized_from_first_student_update": True,
                "official_reference_updates": OFFICIAL_UPDATES,
                "endpoint": "fixed_last_epoch_EMA_teacher",
                "validation_selected": False,
            },
        }
    )
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    temporary = CHECKPOINT.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(CHECKPOINT)
    lock = {
        "schema_version": 1,
        "status": "ALDI_HORIZON_NORMALIZED_TRAINED_AND_LOCKED",
        "completed_at_utc": _now(),
        "checkpoint": _relative(CHECKPOINT),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "raw_checkpoint_sha256": sha256_file(raw),
        "results_sha256": sha256_file(results),
        "args_sha256": sha256_file(args),
        "diagnostics_sha256": sha256_file(DIAGNOSTICS),
        "elapsed_seconds": elapsed,
        "ema_alpha": EMA_ALPHA,
        "ema_updates": MATCHED_UPDATES,
        "post_initialization_weight": EMA_ALPHA ** (MATCHED_UPDATES - 1),
        "validation_metric_used_for_training_or_selection": False,
    }
    atomic_write_json(TRAINING_LOCK, lock)
    return lock


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


def _prediction(dataset: str, view: str) -> Path:
    return OUTPUT / "predictions" / dataset / view / "predictions.coco.json"


def _predict_cell(
    detector: UltralyticsDetector,
    dataset: str,
    view: str,
    records: Sequence[ImageRecord],
    category_map: Mapping[int, int],
) -> dict[str, Any]:
    path = _prediction(dataset, view)
    marker = path.parent / "SUCCESS.json"
    if marker.exists():
        value = _load(marker)
        if value.get("prediction_sha256") != sha256_file(path):
            raise HorizonNormalizedALDIError(f"prediction changed: {path}")
        return value
    if path.parent.exists():
        raise HorizonNormalizedALDIError(f"partial prediction output: {path.parent}")
    batches = detector.predict(
        records,
        imgsz=IMGSZ,
        conf=PROBE_CONF,
        iou=NMS_IOU,
        max_det=MAX_DET,
        fp16=True,
    )
    filtered = _filtered(batches)
    write_coco_predictions(path, filtered, category_id_by_class=category_map)
    payload = {
        "schema_version": 1,
        "status": "ALDI_HORIZON_NORMALIZED_PREDICTION_COMPLETE",
        "dataset": dataset,
        "view": view,
        "images": len(filtered),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "prediction_sha256": sha256_file(path),
    }
    atomic_write_json(marker, payload)
    return payload


def infer() -> dict[str, Any]:
    train()
    if PREDICTION_LOCK.exists():
        return _load(PREDICTION_LOCK)
    target_validation, _ = sensitivity._target_records()
    hazy_validation = evidence_base._hazy_records(verify_hashes=False)
    target_test = final_test._uav_records()
    hazy_test = final_test._hazy_records()
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
        target_validation["original"][:WARMUP_IMAGES],
        imgsz=IMGSZ,
        conf=PROBE_CONF,
        iou=NMS_IOU,
        max_det=MAX_DET,
        fp16=True,
    )
    artifacts = []
    for view in TARGET_VIEWS:
        artifacts.append(
            _predict_cell(
                detector,
                "UAV_OBB_validation",
                view,
                target_validation[view],
                {0: 1, 1: 2, 2: 3},
            )
        )
    artifacts.append(
        _predict_cell(
            detector,
            "HazyDet_validation",
            "hazy",
            hazy_validation,
            {0: 0, 1: 1, 2: 2},
        )
    )
    for view in TARGET_VIEWS:
        artifacts.append(
            _predict_cell(
                detector,
                "UAV_OBB_test",
                view,
                target_test[view],
                {0: 1, 1: 2, 2: 3},
            )
        )
    artifacts.append(
        _predict_cell(
            detector,
            "HazyDet_test",
            "hazy",
            hazy_test,
            {0: 0, 1: 1, 2: 2},
        )
    )
    del detector
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    payload = {
        "schema_version": 1,
        "status": "ALDI_HORIZON_NORMALIZED_PREDICTIONS_COMPLETE",
        "completed_at_utc": _now(),
        "artifacts": artifacts,
        "test_labels_previously_accessed": True,
        "new_test_predictions_are_post_hoc": True,
    }
    atomic_write_json(PREDICTION_LOCK, payload)
    return payload


def _metric_row(
    dataset: str,
    view: str,
    annotation: Path,
    image_ids: Sequence[int | str],
) -> dict[str, Any]:
    result = evaluate_coco(
        annotation,
        _prediction(dataset, view),
        max_det=MAX_DET,
        image_ids=image_ids,
    )
    return {
        "dataset": dataset,
        "view": view,
        **{key: float(result[key]) for key in METRIC_KEYS},
        "images_evaluated": int(result["images_evaluated"]),
    }


def _csv(rows: Sequence[Mapping[str, Any]]) -> str:
    fields = ("dataset", "view", *METRIC_KEYS, "images_evaluated")
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row[field] for field in fields})
    return buffer.getvalue()


def score() -> dict[str, Any]:
    infer()
    if REPORT.exists() and COMPLETE.exists():
        report = _load(REPORT)
        if _load(COMPLETE).get("report_sha256") != sha256_file(REPORT):
            raise HorizonNormalizedALDIError("horizon-normalized report changed")
        return report
    target_conversion = _load(evidence_base.CONVERSION_LOCK)
    target_validation_annotation = ROOT / str(target_conversion["annotation"])
    _, primary_ids = sensitivity._target_records()
    hazy_validation_ids = [
        record.image_id for record in evidence_base._hazy_records(verify_hashes=False)
    ]
    hazy_test_ids = [int(image["id"]) for image in _load(final_test.HAZY_ANNOTATION)["images"]]
    rows = []
    for view in TARGET_VIEWS:
        rows.append(
            _metric_row(
                "UAV_OBB_validation",
                view,
                target_validation_annotation,
                primary_ids,
            )
        )
    rows.append(
        _metric_row(
            "HazyDet_validation",
            "hazy",
            evidence_base.HAZY_ANNOTATION,
            hazy_validation_ids,
        )
    )
    for view in TARGET_VIEWS:
        rows.append(
            _metric_row(
                "UAV_OBB_test",
                view,
                final_test.UAV_ANNOTATION,
                list(range(1, final_test.UAV_IMAGES + 1)),
            )
        )
    rows.append(
        _metric_row(
            "HazyDet_test",
            "hazy",
            final_test.HAZY_ANNOTATION,
            hazy_test_ids,
        )
    )
    rows.sort(key=lambda row: (str(row["dataset"]), str(row["view"])))
    atomic_write_text(METRICS, _csv(rows))
    original_point_report = (
        ROOT
        / "reports/development/cvbra_v1_aldi_direct_baseline_v1"
        / "direct_comparison_analysis/point_report.json"
    )
    original_report = _load(original_point_report)
    payload = {
        "schema_version": 1,
        "status": "COMPLETE_ALDI_HORIZON_NORMALIZED_EVALUATION",
        "completed_at_utc": _now(),
        "registration_sha256": sha256_file(REGISTRATION),
        "training_lock_sha256": sha256_file(TRAINING_LOCK),
        "prediction_lock_sha256": sha256_file(PREDICTION_LOCK),
        "metrics": _relative(METRICS),
        "metrics_sha256": sha256_file(METRICS),
        "rows": rows,
        "ema_horizon": {
            "official_alpha": OFFICIAL_ALPHA,
            "official_updates": OFFICIAL_UPDATES,
            "matched_updates": MATCHED_UPDATES,
            "normalized_alpha": EMA_ALPHA,
            "first_teacher_update": "exact_student_copy",
            "post_initialization_weight": EMA_ALPHA ** (MATCHED_UPDATES - 1),
        },
        "original_equal_supervision_point_report_sha256": sha256_file(original_point_report),
        "original_equal_supervision_points": original_report,
        "interpretation": {
            "stronger_comparator_selected_by_formula_not_validation": True,
            "original_official_alpha_variant_retained": True,
            "test_results_are_post_hoc_fixed_model_checks": True,
            "negative_results_retained": True,
        },
    }
    atomic_write_json(REPORT, payload)
    atomic_write_json(
        COMPLETE,
        {"status": payload["status"], "report_sha256": sha256_file(REPORT)},
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run horizon-normalized ALDI++-AF control")
    parser.add_argument(
        "--stage", choices=("register", "train", "infer", "score", "all"), default="all"
    )
    args = parser.parse_args()
    if args.stage == "register":
        result = register()
    elif args.stage == "train":
        result = train()
    elif args.stage == "infer":
        result = infer()
    elif args.stage == "score":
        result = score()
    else:
        train()
        result = score()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
