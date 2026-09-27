from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import statistics
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any

import torch
from scripts import run_cvbra_v1_training_seed_robustness as base

from buse_uav.detectors.ultralytics_adapter import (
    UltralyticsDetector,
    configure_ultralytics_environment,
)
from buse_uav.evaluation.coco import evaluate_coco, write_coco_predictions
from buse_uav.schemas import DetectionBatch, ImageRecord
from buse_uav.utils.hashing import sha256_file
from buse_uav.utils.io import atomic_write_json, atomic_write_text

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/experiment/cvbra_v1_training_order_robustness_v2.yaml"
PROTOCOL_SHA256 = "6516e0eb1479d23e02500efe1441ced514e61a660eeeac49e04fb6d4ec335679"
ORDER_MANIFEST = ROOT / "data/processed/cvbra_v1_order_robustness_v2/manifest.json"
ORDER_MANIFEST_SHA256 = "12b972972d56aae7bd27c8d46e5d2477c9c2552979acd40876aaa0f25df074de"
BASE_RUNNER = ROOT / "scripts/run_cvbra_v1_training_seed_robustness.py"
BASE_RUNNER_SHA256 = "4f3656d4e47c2ec34a2e573f4986058462d2cf22e6be3bfe1f17765b714f08ac"
SEED_AMENDMENT = (
    ROOT
    / "reports/development/cvbra_v1_training_seed_robustness"
    / "INEFFECTIVE_SEED_PERTURBATION_AMENDMENT_1.json"
)
SEED_AMENDMENT_SHA256 = "8561c63d404d5337e96b65d9a6f1c5605dbed7c345ba943b0dc035281beeb2da"
PRIMARY_CHECKPOINT = ROOT / "runs/cvbra_v1/yolo11n/cvbra_v1.pt"
PRIMARY_CHECKPOINT_SHA256 = "d44f0926696e93b5f2e0ec5c9201f1e6360c43d4d40fccc68646e1ced633bd42"

ORDER_CONFIGS = {
    "hash_a": {
        "yaml": ROOT / "data/processed/cvbra_v1_order_robustness_v2/dataset_hash_a.yaml",
        "yaml_sha256": "07604451ecaf4fd58f8153449cec591fca66645f5e27339cf84f43967043328c",
        "list": ROOT / "data/processed/cvbra_v1_order_robustness_v2/train_hash_a.txt",
        "list_sha256": "d91942aa3411277b5d9f157ac937d935ff7a1121f08e12d4f85808907d0dd96d",
    },
    "hash_b": {
        "yaml": ROOT / "data/processed/cvbra_v1_order_robustness_v2/dataset_hash_b.yaml",
        "yaml_sha256": "eb098688e8aa88e70b67dad5ba905f62ba46b49d78aefec0f42150bc8f76cbee",
        "list": ROOT / "data/processed/cvbra_v1_order_robustness_v2/train_hash_b.txt",
        "list_sha256": "dd07b36b5b50ccb18ef3aa2b95813eb180d3cd834ee842972647d1e561c1bb8e",
    },
}
ORDERS = tuple(ORDER_CONFIGS)
ENDPOINTS = ("primary_locked", *ORDERS)

OUTPUT = ROOT / "reports/development/cvbra_v1_training_order_robustness_v2"
RUN_ROOT = ROOT / "runs/cvbra_v1_training_order_robustness_v2"
IMPLEMENTATION_LOCK = OUTPUT / "implementation_lock.json"
PREDICTION_LOCK = OUTPUT / "joint_prediction_lock.json"
PREDICTION_MARKER = OUTPUT / "PREDICTIONS_LOCKED"
METRICS = OUTPUT / "order_metrics.csv"
REPORT = OUTPUT / "order_robustness_report.json"
COMPLETE = OUTPUT / "COMPLETE.json"

EPOCHS = 8
IMGSZ = 1280
BATCH = 2
TRAIN_SEED = 42
FIRST_TRAINABLE_LAYER = 10
MAX_DET = base.MAX_DET
METRIC_KEYS = base.METRIC_KEYS


class OrderRobustnessError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OrderRobustnessError(f"cannot parse mapping {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise OrderRobustnessError(f"expected mapping: {path}")
    return value


def _assert_hash(path: Path, expected: str, *, label: str) -> None:
    if not path.is_file() or sha256_file(path) != expected:
        raise OrderRobustnessError(f"locked {label} changed: {path}")


def _checkpoint(order: str) -> Path:
    return RUN_ROOT / order / f"cvbra_v1_order_{order}.pt"


def _raw_fit(order: str) -> Path:
    return RUN_ROOT / order / "raw_endpoint" / "fit"


def _training_lock(order: str) -> Path:
    return OUTPUT / "training_locks" / f"{order}.json"


def _checkpoint_lock(order: str) -> Path:
    return OUTPUT / "checkpoint_locks" / f"{order}.json"


def _prediction_root(order: str, domain: str, view: str) -> Path:
    return OUTPUT / "predictions" / order / domain / view


def _runner_path() -> Path:
    return Path(__file__).resolve()


def _validate_implementation_lock() -> dict[str, Any]:
    lock = _load_mapping(IMPLEMENTATION_LOCK)
    if (
        lock.get("protocol_sha256") != PROTOCOL_SHA256
        or lock.get("runner_sha256") != sha256_file(_runner_path())
        or lock.get("order_manifest_sha256") != ORDER_MANIFEST_SHA256
        or lock.get("base_runner_sha256") != BASE_RUNNER_SHA256
    ):
        raise OrderRobustnessError("order-robustness implementation lock changed")
    return lock


def preflight() -> dict[str, Any]:
    base.preflight()
    for path, expected, label in (
        (PROTOCOL, PROTOCOL_SHA256, "protocol"),
        (ORDER_MANIFEST, ORDER_MANIFEST_SHA256, "order manifest"),
        (BASE_RUNNER, BASE_RUNNER_SHA256, "base runner"),
        (SEED_AMENDMENT, SEED_AMENDMENT_SHA256, "seed amendment"),
        (PRIMARY_CHECKPOINT, PRIMARY_CHECKPOINT_SHA256, "primary checkpoint"),
    ):
        _assert_hash(path, expected, label=label)
    manifest = _load_mapping(ORDER_MANIFEST)
    if (
        manifest.get("images") != 3600
        or manifest.get("same_training_multiset") is not True
        or manifest.get("image_or_label_bytes_modified") is not False
    ):
        raise OrderRobustnessError("order manifest does not preserve the training multiset")
    for order, config in ORDER_CONFIGS.items():
        yaml_path = config["yaml"]
        list_path = config["list"]
        assert isinstance(yaml_path, Path)
        assert isinstance(list_path, Path)
        _assert_hash(yaml_path, str(config["yaml_sha256"]), label=f"{order} dataset YAML")
        _assert_hash(list_path, str(config["list_sha256"]), label=f"{order} train list")
        if len(list_path.read_text(encoding="utf-8").splitlines()) != 3600:
            raise OrderRobustnessError(f"{order} train list length changed")
    target, primary_ids = base._target_records(verify_hashes=True)
    hazy = base._hazy_records(verify_hashes=True)
    if len(target["original"]) != 218 or len(primary_ids) != 167 or len(hazy) != 1000:
        raise OrderRobustnessError("locked evaluation records changed")
    if IMPLEMENTATION_LOCK.exists():
        return _validate_implementation_lock()
    if any(path.exists() for path in (RUN_ROOT, PREDICTION_LOCK, METRICS, REPORT, COMPLETE)):
        raise OrderRobustnessError("outputs appeared before implementation lock")
    payload = {
        "schema_version": 1,
        "status": "CVBRA_V1_ORDER_ROBUSTNESS_V2_IMPLEMENTATION_LOCKED",
        "locked_at_utc": _utc_now(),
        "protocol": _relative(PROTOCOL),
        "protocol_sha256": PROTOCOL_SHA256,
        "runner": _relative(_runner_path()),
        "runner_sha256": sha256_file(_runner_path()),
        "base_runner": _relative(BASE_RUNNER),
        "base_runner_sha256": BASE_RUNNER_SHA256,
        "seed_amendment_sha256": SEED_AMENDMENT_SHA256,
        "order_manifest_sha256": ORDER_MANIFEST_SHA256,
        "orders": list(ORDERS),
        "test_content_accessed": False,
        "paper_body_change_authorized": False,
    }
    atomic_write_json(IMPLEMENTATION_LOCK, payload)
    return payload


def _state_difference(left: torch.nn.Module, right: torch.nn.Module) -> dict[str, Any]:
    left_state = left.state_dict()
    right_state = right.state_dict()
    if tuple(left_state) != tuple(right_state):
        raise OrderRobustnessError("checkpoint state schemas differ")
    differing = 0
    max_absolute = 0.0
    for name, left_value in left_state.items():
        if base._layer_index(name) < FIRST_TRAINABLE_LAYER:
            continue
        right_value = right_state[name]
        if not torch.equal(left_value, right_value):
            differing += 1
            if left_value.is_floating_point():
                difference = float(
                    (left_value.float() - right_value.float()).abs().max().item()
                )
                max_absolute = max(max_absolute, difference)
    return {
        "differing_trainable_state_entries": differing,
        "maximum_absolute_trainable_state_difference": max_absolute,
    }


def _build_final_checkpoint(order: str, raw_checkpoint: Path) -> dict[str, Any]:
    source_payload, source_model = base._load_checkpoint(base.SOURCE_CHECKPOINT)
    _, trained_model = base._load_checkpoint(raw_checkpoint)
    if getattr(source_model, "names", None) != getattr(trained_model, "names", None):
        raise OrderRobustnessError("source and trained class schemas differ")
    state = base.combined_state(source_model.state_dict(), trained_model.state_dict())
    output_model = copy.deepcopy(trained_model)
    incompatible = output_model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise OrderRobustnessError("strict final-state load reported incompatibilities")
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
            "cvbra_v1_order_robustness_v2": {
                "order": order,
                "epochs": EPOCHS,
                "endpoint": "last_epoch",
                "fixed_exposed_seed": TRAIN_SEED,
                "frozen_state_rule": "exact source restore for layers 0..9",
                "trained_layers": [FIRST_TRAINABLE_LAYER, 23],
                "protocol_sha256": PROTOCOL_SHA256,
                "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
                "order_manifest_sha256": ORDER_MANIFEST_SHA256,
                "raw_endpoint_sha256": sha256_file(raw_checkpoint),
                "metric_used_for_selection": False,
            },
        }
    )
    checkpoint = _checkpoint(order)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix(".pt.tmp")
    torch.save(output_payload, temporary)
    temporary.replace(checkpoint)
    _, observed_model = base._load_checkpoint(checkpoint)
    source_state = source_model.state_dict()
    observed_state = observed_model.state_dict()
    frozen_exact = all(
        torch.equal(observed_state[name].to(dtype=source_state[name].dtype), source_state[name])
        for name in observed_state
        if base._layer_index(name) <= 9
    )
    if not frozen_exact:
        raise OrderRobustnessError("final checkpoint did not restore frozen source state")
    _, primary_model = base._load_checkpoint(PRIMARY_CHECKPOINT)
    return {
        "frozen_layers_0_to_9_exact_source": True,
        "parameters": sum(int(parameter.numel()) for parameter in observed_model.parameters()),
        "state_entries": len(observed_state),
        "difference_from_primary": _state_difference(observed_model, primary_model),
    }


def _validate_checkpoint_lock(order: str) -> dict[str, Any]:
    lock_path = _checkpoint_lock(order)
    checkpoint = _checkpoint(order)
    if not lock_path.is_file() or not checkpoint.is_file():
        raise OrderRobustnessError(f"checkpoint lock is incomplete for {order}")
    lock = _load_mapping(lock_path)
    if (
        lock.get("order") != order
        or lock.get("checkpoint_sha256") != sha256_file(checkpoint)
        or lock.get("implementation_lock_sha256") != sha256_file(IMPLEMENTATION_LOCK)
    ):
        raise OrderRobustnessError(f"checkpoint lock changed for {order}")
    return lock


def train_order(order: str) -> dict[str, Any]:
    preflight()
    if order not in ORDERS:
        raise OrderRobustnessError(f"order is not registered: {order}")
    if _checkpoint_lock(order).exists():
        return _validate_checkpoint_lock(order)
    checkpoint = _checkpoint(order)
    fit = _raw_fit(order)
    if checkpoint.exists() or fit.exists() or _training_lock(order).exists():
        raise OrderRobustnessError(f"partial training output requires audit for {order}")
    configure_ultralytics_environment(ROOT)
    try:
        from ultralytics import YOLO  # type: ignore[attr-defined]
    except (ImportError, OSError, PermissionError) as exc:
        raise OrderRobustnessError(f"cannot import Ultralytics: {exc}") from exc
    dataset_yaml = ORDER_CONFIGS[order]["yaml"]
    assert isinstance(dataset_yaml, Path)
    model = YOLO(str(base.SOURCE_CHECKPOINT))
    results = model.train(
        data=str(dataset_yaml.resolve()),
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
        seed=TRAIN_SEED,
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
            raise OrderRobustnessError(f"training output is incomplete: {path}")
    raw_payload = {
        "schema_version": 1,
        "status": "CVBRA_V1_ORDER_ROBUSTNESS_V2_RAW_ENDPOINT_LOCKED",
        "locked_at_utc": _utc_now(),
        "order": order,
        "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
        "last_checkpoint": _relative(last),
        "last_checkpoint_sha256": sha256_file(last),
        "results": _relative(results_csv),
        "results_sha256": sha256_file(results_csv),
        "args": _relative(args_yaml),
        "args_sha256": sha256_file(args_yaml),
        "epochs": EPOCHS,
        "validation_metric_used_for_training_or_selection": False,
        "test_content_accessed": False,
    }
    atomic_write_json(_training_lock(order), raw_payload)
    verification = _build_final_checkpoint(order, last)
    lock = {
        "schema_version": 1,
        "status": "CVBRA_V1_ORDER_ROBUSTNESS_V2_CHECKPOINT_VERIFIED_AND_LOCKED",
        "locked_at_utc": _utc_now(),
        "order": order,
        "checkpoint": _relative(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "training_lock_sha256": sha256_file(_training_lock(order)),
        "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
        "protocol_sha256": PROTOCOL_SHA256,
        "verification": verification,
        "validation_metric_used_for_training_or_selection": False,
        "test_content_accessed": False,
        "paper_body_change_authorized": False,
    }
    atomic_write_json(_checkpoint_lock(order), lock)
    return lock


def train() -> dict[str, Any]:
    locks = {order: train_order(order) for order in ORDERS}
    return {"status": "ALL_ORDER_ENDPOINTS_TRAINED", "checkpoints": locks}


def _filter_batches(batches: Sequence[DetectionBatch]) -> tuple[DetectionBatch, ...]:
    return tuple(
        DetectionBatch(
            image_id=batch.image_id,
            boxes=tuple(box for box in batch.boxes if box.score >= base.PUBLISH_CONF),
            latency_ms=batch.latency_ms,
            meta={**batch.meta, "publish_conf": base.PUBLISH_CONF},
        )
        for batch in batches
    )


def _predict_cell(
    *,
    order: str,
    domain: str,
    view: str,
    detector: UltralyticsDetector,
    records: Sequence[ImageRecord],
    category_id_by_class: Mapping[int, int],
) -> dict[str, Any]:
    root = _prediction_root(order, domain, view)
    prediction = root / "predictions.coco.json"
    marker_path = root / "SUCCESS.json"
    if marker_path.exists():
        marker = _load_mapping(marker_path)
        if marker.get("prediction_sha256") != sha256_file(prediction):
            raise OrderRobustnessError(f"prediction changed: {root}")
        return marker
    if root.exists():
        raise OrderRobustnessError(f"partial prediction cell requires audit: {root}")
    batches = detector.predict(
        records,
        imgsz=IMGSZ,
        conf=base.PROBE_CONF,
        iou=base.NMS_IOU,
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
        "status": "CVBRA_V1_ORDER_ROBUSTNESS_V2_PREDICTION_COMPLETE",
        "completed_at_utc": _utc_now(),
        "order": order,
        "domain": domain,
        "view": view,
        "images": len(filtered),
        "checkpoint_sha256": sha256_file(_checkpoint(order)),
        "prediction": _relative(prediction),
        "prediction_sha256": sha256_file(prediction),
        "validation_labels_previously_accessed": True,
        "metrics_accessed_for_prediction": False,
        "test_content_accessed": False,
    }
    atomic_write_json(marker_path, marker)
    return marker


def infer() -> dict[str, Any]:
    preflight()
    for order in ORDERS:
        _validate_checkpoint_lock(order)
    if PREDICTION_LOCK.exists() and PREDICTION_MARKER.exists():
        lock = _load_mapping(PREDICTION_LOCK)
        marker = _load_mapping(PREDICTION_MARKER)
        if marker.get("joint_prediction_lock_sha256") != sha256_file(PREDICTION_LOCK):
            raise OrderRobustnessError("prediction lock changed")
        return lock
    if any(path.exists() for path in (PREDICTION_LOCK, PREDICTION_MARKER, METRICS, REPORT)):
        raise OrderRobustnessError("partial prediction or metric output requires audit")
    target, _ = base._target_records(verify_hashes=False)
    hazy = base._hazy_records(verify_hashes=False)
    artifacts: list[dict[str, Any]] = []
    for order in ORDERS:
        detector = UltralyticsDetector(
            _checkpoint(order),
            model_name="yolo11n",
            device="cuda:0",
            expected_class_names=base.CLASS_NAMES,
            project_root=ROOT,
            stream_chunk_records=base.CHUNK_SIZE,
            release_cuda_cache_between_chunks=False,
        )
        detector.predict(
            target["original"][: base.WARMUP_IMAGES],
            imgsz=IMGSZ,
            conf=base.PROBE_CONF,
            iou=base.NMS_IOU,
            max_det=MAX_DET,
            fp16=True,
        )
        for view in base.TARGET_VIEWS:
            artifacts.append(
                _predict_cell(
                    order=order,
                    domain="UAV_OBB_validation",
                    view=view,
                    detector=detector,
                    records=target[view],
                    category_id_by_class=base.TARGET_CATEGORY_ID_BY_CLASS,
                )
            )
        artifacts.append(
            _predict_cell(
                order=order,
                domain="HazyDet_validation",
                view="hazy",
                detector=detector,
                records=hazy,
                category_id_by_class=base.HAZY_CATEGORY_ID_BY_CLASS,
            )
        )
        del detector
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        print(json.dumps({"order_inference_complete": order}), flush=True)
    payload = {
        "schema_version": 1,
        "status": "CVBRA_V1_ORDER_ROBUSTNESS_V2_PREDICTIONS_LOCKED_BEFORE_METRICS",
        "locked_at_utc": _utc_now(),
        "protocol_sha256": PROTOCOL_SHA256,
        "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
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


def summarize_order_metrics(
    rows: Sequence[Mapping[str, Any]], baselines: Mapping[str, float]
) -> dict[str, Any]:
    variability: dict[str, dict[str, dict[str, float]]] = {}
    deltas: dict[str, dict[str, float]] = {}
    for domain in ("target_original", "target_fog_1p0", "source_HazyDet"):
        selected = [row for row in rows if row.get("domain") == domain]
        endpoints = {str(row.get("endpoint")) for row in selected}
        if len(selected) != len(ENDPOINTS) or endpoints != set(ENDPOINTS):
            raise OrderRobustnessError(f"incomplete endpoint rows for {domain}")
        variability[domain] = {}
        for metric in METRIC_KEYS:
            values = [float(row[metric]) for row in selected]
            variability[domain][metric] = {
                "mean": statistics.mean(values),
                "sample_standard_deviation": statistics.stdev(values),
                "minimum": min(values),
                "maximum": max(values),
            }
        deltas[domain] = {
            str(row["endpoint"]): float(row["AP"]) - float(baselines[domain])
            for row in selected
        }
    checks = {
        "every_endpoint_target_original_delta_at_least_0p20": min(
            deltas["target_original"].values()
        )
        >= 0.20,
        "every_endpoint_target_fog_1p0_delta_at_least_0p18": min(
            deltas["target_fog_1p0"].values()
        )
        >= 0.18,
        "target_original_AP_sample_sd_at_most_0p03": variability["target_original"][
            "AP"
        ]["sample_standard_deviation"]
        <= 0.03,
        "target_fog_1p0_AP_sample_sd_at_most_0p03": variability["target_fog_1p0"][
            "AP"
        ]["sample_standard_deviation"]
        <= 0.03,
        "every_endpoint_HazyDet_AP_at_least_0p44": variability["source_HazyDet"]["AP"][
            "minimum"
        ]
        >= 0.44,
    }
    return {
        "variability": variability,
        "AP_deltas_vs_source_checkpoint": deltas,
        "registered_metric_checks": checks,
        "all_registered_metric_checks_pass": all(checks.values()),
    }


def _write_metrics(rows: Sequence[Mapping[str, Any]]) -> None:
    fields = ("domain", "endpoint", *METRIC_KEYS, "images_evaluated")
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row[field] for field in fields})
    atomic_write_text(METRICS, buffer.getvalue())


def _primary_rows() -> tuple[list[dict[str, Any]], dict[str, float]]:
    existing, baselines = base._existing_metric_rows()
    rows = [
        {
            **row,
            "endpoint": "primary_locked",
        }
        for row in existing
    ]
    for row in rows:
        row.pop("model", None)
        row.pop("seed", None)
    return rows, baselines


def _state_diversity() -> dict[str, Any]:
    _, primary = base._load_checkpoint(PRIMARY_CHECKPOINT)
    _, first = base._load_checkpoint(_checkpoint("hash_a"))
    _, second = base._load_checkpoint(_checkpoint("hash_b"))
    comparisons = {
        "hash_a_minus_primary": _state_difference(first, primary),
        "hash_b_minus_primary": _state_difference(second, primary),
        "hash_a_minus_hash_b": _state_difference(first, second),
    }
    checks = {
        "hash_a_trainable_state_differs_from_primary": comparisons[
            "hash_a_minus_primary"
        ]["differing_trainable_state_entries"]
        > 0,
        "hash_b_trainable_state_differs_from_primary": comparisons[
            "hash_b_minus_primary"
        ]["differing_trainable_state_entries"]
        > 0,
        "hash_a_and_hash_b_trainable_states_differ": comparisons["hash_a_minus_hash_b"][
            "differing_trainable_state_entries"
        ]
        > 0,
    }
    return {
        "comparisons": comparisons,
        "registered_effective_perturbation_checks": checks,
        "all_registered_effective_perturbation_checks_pass": all(checks.values()),
    }


def score() -> dict[str, Any]:
    prediction_lock = infer()
    if REPORT.exists() and METRICS.exists() and COMPLETE.exists():
        report = _load_mapping(REPORT)
        marker = _load_mapping(COMPLETE)
        if marker.get("order_robustness_report_sha256") != sha256_file(REPORT):
            raise OrderRobustnessError("order robustness report changed")
        return report
    if any(path.exists() for path in (REPORT, METRICS, COMPLETE)):
        raise OrderRobustnessError("partial metric output requires audit")
    target_conversion = _load_mapping(base.CONVERSION_LOCK)
    target_annotation = base._rooted(target_conversion["annotation"])
    base._assert_hash(
        target_annotation,
        str(target_conversion["annotation_sha256"]),
        label="target annotation",
    )
    _, primary_ids = base._target_records(verify_hashes=False)
    hazy_records = base._hazy_records(verify_hashes=False)
    rows, baselines = _primary_rows()
    for order in ORDERS:
        for view in base.TARGET_VIEWS:
            prediction = _prediction_root(
                order, "UAV_OBB_validation", view
            ) / "predictions.coco.json"
            result = evaluate_coco(
                target_annotation,
                prediction,
                max_det=MAX_DET,
                image_ids=primary_ids,
            )
            rows.append(
                {
                    "domain": f"target_{view}",
                    "endpoint": order,
                    **{key: float(result[key]) for key in METRIC_KEYS},
                    "images_evaluated": len(primary_ids),
                }
            )
        prediction = _prediction_root(
            order, "HazyDet_validation", "hazy"
        ) / "predictions.coco.json"
        result = evaluate_coco(
            base.HAZY_ANNOTATION,
            prediction,
            max_det=MAX_DET,
            image_ids=[record.image_id for record in hazy_records],
        )
        rows.append(
            {
                "domain": "source_HazyDet",
                "endpoint": order,
                **{key: float(result[key]) for key in METRIC_KEYS},
                "images_evaluated": len(hazy_records),
            }
        )
    rows.sort(key=lambda row: (str(row["domain"]), str(row["endpoint"])))
    metric_summary = summarize_order_metrics(rows, baselines)
    state_diversity = _state_diversity()
    _write_metrics(rows)
    all_checks = bool(metric_summary["all_registered_metric_checks_pass"]) and bool(
        state_diversity["all_registered_effective_perturbation_checks_pass"]
    )
    payload = {
        "schema_version": 1,
        "status": "COMPLETE_CVBRA_V1_TRAINING_ORDER_ROBUSTNESS_V2_AUDIT",
        "completed_at_utc": _utc_now(),
        "protocol_sha256": PROTOCOL_SHA256,
        "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
        "prediction_lock_sha256": sha256_file(PREDICTION_LOCK),
        "prediction_lock_status": prediction_lock.get("status"),
        "endpoints": list(ENDPOINTS),
        "baselines": baselines,
        "rows": rows,
        "metric_summary": metric_summary,
        "state_diversity": state_diversity,
        "all_registered_checks_pass": all_checks,
        "metrics": _relative(METRICS),
        "metrics_sha256": sha256_file(METRICS),
        "evidence_boundary": {
            "post_freeze_order_robustness_audit": True,
            "validation_labels_previously_accessed": True,
            "method_or_checkpoint_reselection": False,
            "independent_confirmation_claim": False,
            "test_content_accessed": False,
            "same_training_multiset": True,
        },
        "paper_body_change_authorized": False,
    }
    atomic_write_json(REPORT, payload)
    atomic_write_json(
        COMPLETE,
        {
            "status": payload["status"],
            "order_robustness_report_sha256": sha256_file(REPORT),
            "metrics_sha256": sha256_file(METRICS),
        },
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CVBRA-v1 effective-order robustness v2")
    parser.add_argument(
        "--stage",
        choices=("preflight", "train", "infer", "score", "all"),
        default="all",
    )
    parser.add_argument("--order", choices=ORDERS)
    args = parser.parse_args()
    if args.stage == "preflight":
        result = preflight()
    elif args.stage == "train":
        result = train_order(args.order) if args.order else train()
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
