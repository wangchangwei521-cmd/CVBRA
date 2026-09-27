from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch
from scripts import prepare_nvd_real_snow_cvbra_v1_1 as correction
from scripts import run_nvd_real_snow_cvbra_v1 as base

from buse_uav.detectors.ultralytics_adapter import UltralyticsDetector
from buse_uav.evaluation.bootstrap import (
    ClusterBootstrapScope,
    paired_coco_ap_cluster_bootstrap_scopes,
)
from buse_uav.evaluation.coco import evaluate_coco, write_coco_predictions
from buse_uav.utils.hashing import sha256_file
from buse_uav.utils.io import atomic_write_json, atomic_write_text

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/development/nvd_real_snow_cvbra_v1_1"
RUN_ROOT = ROOT / "runs/nvd_real_snow_cvbra_v1_1"
PREDICTION_COMPLETE = OUTPUT / "PREDICTIONS_COMPLETE.json"
METRICS_CSV = OUTPUT / "metrics.csv"
REPORT = OUTPUT / "real_snow_report.json"
COMPLETE = OUTPUT / "COMPLETE.json"

METHODS = correction.METHODS
ALL_SEEDS = base.SEEDS
REPLACEMENT_SEEDS = correction.REPLACEMENT_SEEDS
SPLITS = base.SPLITS
METRICS = base.METRICS


class NvdSeedOrderRunError(RuntimeError):
    """Raised when corrected training or inference cannot fail closed."""


def _model_key(method: str, seed: int) -> str:
    return f"{method}_seed_{seed}"


def _checkpoint(method: str, seed: int) -> Path:
    if seed == 42:
        return base._checkpoint(method, seed)
    return RUN_ROOT / method / f"seed_{seed}" / f"{method}_seed_{seed}.pt"


def _training_lock(method: str, seed: int) -> Path:
    if seed == 42:
        return base._training_lock(method, seed)
    return OUTPUT / "training_locks" / method / f"seed_{seed}.json"


def _raw_fit(method: str, seed: int) -> Path:
    return RUN_ROOT / method / f"seed_{seed}" / "raw_endpoint/fit"


def _prediction(model_key: str, split: str) -> Path:
    if model_key == "source" or model_key.endswith("_seed_42"):
        return base.OUTPUT / "predictions" / model_key / f"{split}.json"
    return OUTPUT / "predictions" / model_key / f"{split}.json"


def _prediction_lock(model_key: str, split: str) -> Path:
    if model_key == "source" or model_key.endswith("_seed_42"):
        return base.OUTPUT / "prediction_locks" / model_key / f"{split}.json"
    return OUTPUT / "prediction_locks" / model_key / f"{split}.json"


def _effective_state_sha256(path: Path) -> str:
    _, model = base._load_checkpoint(path)
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _base_lock(path: Path) -> dict[str, Any]:
    lock = correction.load_json(path)
    checkpoint = correction.rooted(str(lock["checkpoint"]))
    if lock.get("checkpoint_sha256") != sha256_file(checkpoint):
        raise NvdSeedOrderRunError(f"retained base checkpoint changed: {checkpoint}")
    return lock


def preflight() -> dict[str, Any]:
    data_lock = correction.validate_data_lock()
    incident = correction._assert_pre_metric_boundary()
    source_lock = _base_lock(base.SOURCE_LOCK)
    retained = {}
    for method in METHODS:
        retained[method] = _base_lock(base._training_lock(method, 42))
    if incident.get("scope", {}).get("uavdt_used") is not False:
        raise NvdSeedOrderRunError("withdrawn dataset boundary changed")
    return {
        "status": "PASS_NVD_REAL_SNOW_SEED_ORDER_CORRECTION_PREFLIGHT",
        "registration_sha256": sha256_file(correction.REGISTRATION),
        "data_lock_sha256": sha256_file(correction.DATA_LOCK),
        "source_checkpoint_sha256": source_lock["checkpoint_sha256"],
        "retained_seed_42_checkpoint_sha256": {
            method: retained[method]["checkpoint_sha256"] for method in METHODS
        },
        "replacement_methods": list(METHODS),
        "replacement_seeds": list(REPLACEMENT_SEEDS),
        "datasets": len(data_lock["datasets"]),
        "target_metric_values_computed_or_accessed_before_correction": False,
        "uavdt_used": False,
    }


def _annotate_endpoint(path: Path, *, method: str, seed: int, order_sha256: str) -> None:
    checkpoint, _ = base._load_checkpoint(path)
    checkpoint["nvd_real_snow_cvbra_v1_1"] = {
        "method": method,
        "seed": seed,
        "endpoint": "fixed_last_epoch",
        "correction_registration_sha256": sha256_file(correction.REGISTRATION),
        "correction_data_lock_sha256": sha256_file(correction.DATA_LOCK),
        "base_incident_sha256": sha256_file(correction.BASE_INCIDENT),
        "explicit_order_sha256": order_sha256,
        "sample_multiset_changed": False,
        "method_hyperparameter_or_checkpoint_reselected": False,
        "target_test_access_during_training": False,
    }
    temporary = path.with_suffix(".pt.correction.tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)
    base._load_checkpoint(path)


def _dataset_record(method: str, seed: int) -> dict[str, Any]:
    lock = correction.validate_data_lock()
    matches = [
        row
        for row in lock["datasets"]
        if isinstance(row, dict) and row.get("method") == method and row.get("seed") == seed
    ]
    if len(matches) != 1:
        raise NvdSeedOrderRunError(f"correction dataset is not unique: {method}/{seed}")
    return matches[0]


def _assert_method_states_distinct(method: str, *, require_all: bool) -> dict[int, str]:
    hashes = {42: _effective_state_sha256(_checkpoint(method, 42))}
    for seed in REPLACEMENT_SEEDS:
        path = _checkpoint(method, seed)
        if path.is_file():
            hashes[seed] = _effective_state_sha256(path)
        elif require_all:
            raise NvdSeedOrderRunError(f"corrected checkpoint is missing: {method}/{seed}")
    if len(set(hashes.values())) != len(hashes):
        raise NvdSeedOrderRunError(f"effective training states are not distinct: {method}/{hashes}")
    return hashes


def train_candidate(method: str, seed: int) -> dict[str, Any]:
    if method not in METHODS or seed not in REPLACEMENT_SEEDS:
        raise NvdSeedOrderRunError(f"unregistered replacement candidate: {method}/{seed}")
    preflight()
    path = _checkpoint(method, seed)
    lock_path = _training_lock(method, seed)
    if path.exists() and lock_path.exists():
        lock = correction.load_json(lock_path)
        if (
            lock.get("checkpoint_sha256") != sha256_file(path)
            or lock.get("effective_state_sha256") != _effective_state_sha256(path)
        ):
            raise NvdSeedOrderRunError(f"corrected endpoint changed: {method}/{seed}")
        return lock
    if path.exists() or lock_path.exists():
        raise NvdSeedOrderRunError(f"partial corrected endpoint requires audit: {method}/{seed}")
    dataset = _dataset_record(method, seed)
    freeze = None if method == "STF" else base.FIRST_TRAINABLE_LAYER
    started = time.perf_counter()
    last, results_csv, args_yaml, training_seconds = base._run_training(
        initialization=base.SOURCE_CHECKPOINT,
        dataset=correction.rooted(str(dataset["dataset_yaml"])),
        epochs=base.EPOCHS,
        seed=seed,
        fit=_raw_fit(method, seed),
        freeze=freeze,
    )
    endpoint = base._save_endpoint(
        last,
        path,
        method=method,
        seed=seed,
        source_path=base.SOURCE_CHECKPOINT,
    )
    _annotate_endpoint(path, method=method, seed=seed, order_sha256=str(dataset["order_sha256"]))
    state_hash = _effective_state_sha256(path)
    base_state_hash = _effective_state_sha256(_checkpoint(method, 42))
    if state_hash == base_state_hash:
        raise NvdSeedOrderRunError(
            f"order correction did not change effective state: {method}/{seed}"
        )
    for other_seed in REPLACEMENT_SEEDS:
        other = _checkpoint(method, other_seed)
        if other_seed != seed and other.is_file() and state_hash == _effective_state_sha256(other):
            raise NvdSeedOrderRunError(
                f"replacement order states collide: {method}/{seed}/{other_seed}"
            )
    payload = {
        "schema_version": 1,
        "status": "NVD_SEED_ORDER_CORRECTED_FIXED_LAST_EPOCH_LOCKED",
        "locked_at_utc": correction.utc_now(),
        "method": method,
        "seed": seed,
        "epochs": base.EPOCHS,
        "images_per_epoch": correction.TRAIN_IMAGES,
        "dataset_yaml": str(dataset["dataset_yaml"]),
        "dataset_yaml_sha256": str(dataset["dataset_yaml_sha256"]),
        "dataset_manifest_sha256": str(dataset["manifest_sha256"]),
        "dataset_multiset_sha256": str(dataset["multiset_sha256"]),
        "explicit_order_sha256": str(dataset["order_sha256"]),
        "source_checkpoint_sha256": sha256_file(base.SOURCE_CHECKPOINT),
        "base_seed_42_checkpoint_sha256": sha256_file(_checkpoint(method, 42)),
        "base_seed_42_effective_state_sha256": base_state_hash,
        "raw_endpoint": correction.relative(last),
        "raw_endpoint_sha256": sha256_file(last),
        "results": correction.relative(results_csv),
        "results_sha256": sha256_file(results_csv),
        "args": correction.relative(args_yaml),
        "args_sha256": sha256_file(args_yaml),
        "training_seconds": training_seconds,
        "total_endpoint_seconds": time.perf_counter() - started,
        "checkpoint": correction.relative(path),
        "checkpoint_sha256": sha256_file(path),
        "effective_state_sha256": state_hash,
        "effective_state_distinct_from_seed_42": True,
        "frozen_layers_0_to_9_exact_source": endpoint[
            "frozen_layers_0_to_9_exact_source"
        ],
        "changed_trainable_floating_states": endpoint["changed_trainable_floating_states"],
        "sample_multiset_changed": False,
        "validation_during_training": False,
        "checkpoint_selected_by_metric": False,
        "target_test_access_during_training": False,
        "method_hyperparameter_or_checkpoint_reselected": False,
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(lock_path, payload)
    return payload


def train_all() -> list[dict[str, Any]]:
    results = [
        train_candidate(method, seed) for method in METHODS for seed in REPLACEMENT_SEEDS
    ]
    for method in METHODS:
        _assert_method_states_distinct(method, require_all=True)
    return results


def _validate_training_complete() -> None:
    preflight()
    for method in METHODS:
        for seed in REPLACEMENT_SEEDS:
            train_candidate(method, seed)
        hashes = _assert_method_states_distinct(method, require_all=True)
        if len(hashes) != 3:
            raise NvdSeedOrderRunError(f"trajectory coverage changed: {method}")


def _checkpoint_for_model(model_key: str) -> Path:
    if model_key == "source":
        return base.SOURCE_CHECKPOINT
    method, seed_text = model_key.rsplit("_seed_", 1)
    seed = int(seed_text)
    if method not in METHODS or seed not in ALL_SEEDS:
        raise NvdSeedOrderRunError(f"unregistered model key: {model_key}")
    return _checkpoint(method, seed)


def _locked_prediction(model_key: str, split: str) -> dict[str, Any] | None:
    path = _prediction(model_key, split)
    lock_path = _prediction_lock(model_key, split)
    if path.exists() and lock_path.exists():
        lock = correction.load_json(lock_path)
        checkpoint = _checkpoint_for_model(model_key)
        if (
            lock.get("prediction_sha256") != sha256_file(path)
            or lock.get("checkpoint_sha256") != sha256_file(checkpoint)
        ):
            raise NvdSeedOrderRunError(f"prediction lock changed: {model_key}/{split}")
        return lock
    if path.exists() or lock_path.exists():
        raise NvdSeedOrderRunError(f"partial prediction requires audit: {model_key}/{split}")
    return None


def _infer_corrected_one(
    *, model_key: str, split: str, detector: UltralyticsDetector
) -> dict[str, Any]:
    existing = _locked_prediction(model_key, split)
    if existing is not None:
        return existing
    checkpoint = _checkpoint_for_model(model_key)
    records = base._records(split)
    started = time.perf_counter()
    batches = detector.predict(
        records,
        imgsz=base.IMGSZ,
        conf=base.LOW_FLOOR,
        iou=base.NMS_IOU,
        max_det=base.MAX_DET,
        fp16=True,
    )
    elapsed = time.perf_counter() - started
    path = _prediction(model_key, split)
    rows = write_coco_predictions(path, batches, category_id_by_class=base.CATEGORY_MAP)
    payload = {
        "schema_version": 1,
        "status": "NVD_CORRECTED_PREDICTION_ARTIFACT_LOCKED",
        "locked_at_utc": correction.utc_now(),
        "model": model_key,
        "split": split,
        "checkpoint": correction.relative(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "prediction": correction.relative(path),
        "prediction_sha256": sha256_file(path),
        "images": len(records),
        "detections": len(rows),
        "elapsed_seconds": elapsed,
        "candidate_score_floor": base.LOW_FLOOR,
        "nms_iou": base.NMS_IOU,
        "max_det": base.MAX_DET,
        "seed_order_correction": True,
    }
    lock_path = _prediction_lock(model_key, split)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(lock_path, payload)
    return payload


def infer_model(model_key: str) -> list[dict[str, Any]]:
    _validate_training_complete()
    if model_key == "source" or model_key.endswith("_seed_42"):
        artifacts = []
        for split in SPLITS:
            lock = _locked_prediction(model_key, split)
            if lock is None:
                raise NvdSeedOrderRunError(
                    f"retained seed-42 prediction is missing: {model_key}/{split}"
                )
            artifacts.append(lock)
        return artifacts
    checkpoint = _checkpoint_for_model(model_key)
    detector = UltralyticsDetector(
        checkpoint,
        model_name="yolo11n",
        device="cuda:0",
        expected_class_names=base.CLASS_NAMES,
        project_root=ROOT,
        stream_chunk_records=8,
        release_cuda_cache_between_chunks=False,
    )
    detector.predict(
        base._records("source_retention")[:16],
        imgsz=base.IMGSZ,
        conf=base.LOW_FLOOR,
        iou=base.NMS_IOU,
        max_det=base.MAX_DET,
        fp16=True,
    )
    artifacts = [
        _infer_corrected_one(model_key=model_key, split=split, detector=detector)
        for split in SPLITS
    ]
    del detector
    gc.collect()
    torch.cuda.empty_cache()
    return artifacts


def infer_all() -> dict[str, Any]:
    if PREDICTION_COMPLETE.exists():
        payload = correction.load_json(PREDICTION_COMPLETE)
        for row in payload.get("artifacts", []):
            if not isinstance(row, dict):
                raise NvdSeedOrderRunError("malformed corrected prediction-complete row")
            path = correction.rooted(str(row["prediction"]))
            if row.get("prediction_sha256") != sha256_file(path):
                raise NvdSeedOrderRunError(f"completed corrected prediction changed: {path}")
        return payload
    artifacts = infer_model("source")
    for method in METHODS:
        for seed in ALL_SEEDS:
            artifacts.extend(infer_model(_model_key(method, seed)))
    payload = {
        "schema_version": 1,
        "status": "NVD_REAL_SNOW_CORRECTED_ALL_PREDICTIONS_LOCKED",
        "completed_at_utc": correction.utc_now(),
        "models": 1 + len(METHODS) * len(ALL_SEEDS),
        "splits": list(SPLITS),
        "artifacts": artifacts,
        "retained_seed_42_models": 1 + len(METHODS),
        "corrected_replacement_models": len(METHODS) * len(REPLACEMENT_SEEDS),
        "all_method_states_distinct_across_three_trajectories": True,
        "method_or_checkpoint_selection_from_predictions": False,
    }
    atomic_write_json(PREDICTION_COMPLETE, payload)
    return payload


def _metric_row(model_key: str, split: str) -> dict[str, Any]:
    result = evaluate_coco(
        base._annotation(split),
        _prediction(model_key, split),
        max_det=base.MAX_DET,
        image_ids=base._image_ids(split),
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
                clusters=base._test_clusters(),
                checkpoint_path=checkpoint,
                checkpoint_identity={
                    "protocol": "nvd_real_snow_cvbra_v1_1",
                    "contrast": contrast,
                    "primary_training_seed": seed,
                    "seed_order_correction": True,
                },
            )
        }
        result = paired_coco_ap_cluster_bootstrap_scopes(
            base._annotation("target_test"),
            baseline_path,
            method_path,
            scopes,
            resamples=10000,
            seed=20260825,
            max_det=base.MAX_DET,
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


def score() -> dict[str, Any]:
    infer_all()
    if REPORT.exists() and COMPLETE.exists():
        report = correction.load_json(REPORT)
        if correction.load_json(COMPLETE).get("report_sha256") != sha256_file(REPORT):
            raise NvdSeedOrderRunError("completed corrected real-snow report changed")
        return report
    model_keys = ["source"] + [
        _model_key(method, seed) for method in METHODS for seed in ALL_SEEDS
    ]
    rows = [_metric_row(model_key, split) for model_key in model_keys for split in SPLITS]
    atomic_write_text(METRICS_CSV, base._metrics_csv(rows))
    state_hashes = {
        method: _assert_method_states_distinct(method, require_all=True) for method in METHODS
    }
    payload = {
        "schema_version": 2,
        "status": "COMPLETE_NVD_REAL_SNOW_CVBRA_INDEPENDENT_CONFIRMATION",
        "completed_at_utc": correction.utc_now(),
        "protocol": "nvd_real_snow_cvbra_v1_1",
        "base_incident_sha256": sha256_file(correction.BASE_INCIDENT),
        "correction_registration_sha256": sha256_file(correction.REGISTRATION),
        "correction_data_lock_sha256": sha256_file(correction.DATA_LOCK),
        "prediction_complete_sha256": sha256_file(PREDICTION_COMPLETE),
        "metrics_csv": correction.relative(METRICS_CSV),
        "metrics_csv_sha256": sha256_file(METRICS_CSV),
        "metric_rows": rows,
        "trajectory_analysis": base._trajectory_report(rows),
        "target_test_temporal_block_bootstrap": _temporal_bootstrap(),
        "effective_model_state_sha256": state_hashes,
        "all_method_states_distinct_across_three_trajectories": True,
        "selection_statement": (
            "The seed-order correction was triggered only by exact endpoint-state identity "
            "before any metric computation. No validation or held-out test metric selected a "
            "method, hyperparameter, epoch, checkpoint, contrast, seed, or bootstrap scope. "
            "All registered results are retained."
        ),
        "uavdt_used": False,
    }
    atomic_write_json(REPORT, payload)
    atomic_write_json(
        COMPLETE,
        {
            "status": payload["status"],
            "report": correction.relative(REPORT),
            "report_sha256": sha256_file(REPORT),
        },
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the corrected NVD real-snow study")
    parser.add_argument(
        "--stage",
        choices=("preflight", "train", "infer", "score", "all"),
        required=True,
    )
    args = parser.parse_args()
    if args.stage == "preflight":
        result: Any = preflight()
    elif args.stage == "train":
        result = train_all()
    elif args.stage == "infer":
        result = infer_all()
    else:
        train_all()
        result = score()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
